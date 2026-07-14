using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using OpenUtau.Core.Format;
using OpenUtau.Core.HifiNeural;
using OpenUtau.Core.Render;
using OpenUtau.Core.SignalChain;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.Neutrino {
    public class NeutrinoLegacyV2Renderer : IRenderer {
        public const int headTicks = 480;
        public const int tailTicks = 480;

        const int outputSampleRate = 44100;
        const int frameRate = 200; // NEUTRINO v2 NSF frames are 5 ms.
        const int pitchInterval = 5;
        const int numMelBins = 100;
        const int cacheVersion = 17;
        const int postEffectCacheVersion = 1;
        const int pitchCacheMagic = 0x4E563246; // NV2F
        const float f0Min = 40f;
        const float f0Max = 2000f;
        const float melspecMin = -6f;
        const float melspecMax = 1f;
        const float wavScale = 0.9885531068f;
        const float wavClamp = 0.9988493919f;
        const int legacyPau = 25;
        const int legacySil = 31;
        const int legacyAcousticPhonemeOffset = 3;

        static readonly Dictionary<string, int> legacyPhonemeToId =
            new Dictionary<string, int>(StringComparer.Ordinal) {
                {"a", 0},
                {"b", 1},
                {"br", 2},
                {"by", 3},
                {"ch", 4},
                {"cl", 5},
                {"d", 6},
                {"dy", 7},
                {"e", 8},
                {"f", 9},
                {"g", 10},
                {"gy", 11},
                {"h", 12},
                {"hy", 13},
                {"i", 14},
                {"j", 15},
                {"k", 16},
                {"ky", 17},
                {"m", 18},
                {"my", 19},
                {"n", 20},
                {"N", 21},
                {"ny", 22},
                {"o", 23},
                {"p", 24},
                {"pau", legacyPau},
                {"py", 26},
                {"r", 27},
                {"ry", 28},
                {"s", 29},
                {"sh", 30},
                {"sil", legacySil},
                {"t", 32},
                {"ts", 33},
                {"ty", 34},
                {"u", 35},
                {"v", 36},
                {"w", 37},
                {"y", 38},
                {"z", 39},
                {"AP", legacyPau},
                {"ap", legacyPau},
            };

        static readonly HashSet<string> supportedExp = new HashSet<string>() {
            Format.Ustx.DYN,
            Format.Ustx.PITD,
            Format.Ustx.GENC,
            Format.Ustx.BREC,
            Format.Ustx.SHFC,
            Format.Ustx.TENC,
            Format.Ustx.VOIC,
        };

        static readonly object lockObj = new object();

        sealed class LegacyRawRender {
            public float[] Samples { get; }
            public LegacyPitchTrack PitchTrack { get; }

            public LegacyRawRender(float[] samples, LegacyPitchTrack pitchTrack) {
                Samples = samples;
                PitchTrack = pitchTrack;
            }
        }

        sealed class LegacyTimingContext {
            public long[] PhonemeIds { get; }
            public long[] MidiNotes { get; }
            public float[] Durations { get; }
            public long[] SlurFlags { get; }
            public long[] FramePhonemeMap { get; }
            public float[] ScorePitchCents { get; }
            public int TotalFrames { get; }
            public int SampleRate { get; }
            public int HopSize { get; }
            public double OriginMs { get; }

            public LegacyTimingContext(
                long[] phonemeIds,
                long[] midiNotes,
                float[] durations,
                long[] slurFlags,
                long[] framePhonemeMap,
                float[] scorePitchCents,
                int totalFrames,
                int sampleRate,
                double originMs) {

                PhonemeIds = phonemeIds;
                MidiNotes = midiNotes;
                Durations = durations;
                SlurFlags = slurFlags;
                FramePhonemeMap = framePhonemeMap;
                ScorePitchCents = scorePitchCents;
                TotalFrames = totalFrames;
                SampleRate = sampleRate;
                HopSize = Math.Max(1, sampleRate / frameRate);
                OriginMs = originMs;
            }
        }

        sealed class LegacyPitchTrack {
            readonly int sampleRate;
            readonly int hopSize;
            readonly int headSamples;
            readonly int vocoderSamples;
            readonly float[] f0;

            public int SampleRate => sampleRate;
            public int HopSize => hopSize;
            public int HeadSamples => headSamples;
            public int VocoderSamples => vocoderSamples;
            public float[] Frames => f0;

            public LegacyPitchTrack(int sampleRate, int hopSize, int headSamples, int vocoderSamples, float[] f0) {
                this.sampleRate = sampleRate;
                this.hopSize = Math.Max(1, hopSize);
                this.headSamples = Math.Max(0, headSamples);
                this.vocoderSamples = Math.Max(0, vocoderSamples);
                this.f0 = f0 ?? Array.Empty<float>();
            }

            public double GetF0AtOutputSample(double outputSample, int _) {
                if (f0.Length == 0) {
                    return 0;
                }
                double sourceSample = outputSample * sampleRate / outputSampleRate;
                double frameIndex = (sourceSample - headSamples) / hopSize;
                if (frameIndex < 0 || sourceSample >= headSamples + vocoderSamples) {
                    return 0;
                }
                frameIndex = Math.Clamp(frameIndex, 0, f0.Length - 1);
                int left = (int)Math.Floor(frameIndex);
                int right = Math.Min(f0.Length - 1, left + 1);
                double alpha = frameIndex - left;
                double value = f0[left] + (f0[right] - f0[left]) * alpha;
                return double.IsFinite(value) && value >= f0Min && value <= f0Max ? value : 0;
            }
        }

        struct PhoneSegment {
            public string phoneme;
            public int noteIndex;
            public int positionInNote;
            public int phonesInNote;
            public double startSec;
            public double endSec;
            public double scoreStartSec;
            public double scoreEndSec;
        }

        public USingerType SingerType => USingerType.Neutrino;
        public bool SupportsRenderPitch => true;

        public bool SupportsExpression(UExpressionDescriptor descriptor) {
            return supportedExp.Contains(descriptor.abbr);
        }

        public RenderResult Layout(RenderPhrase phrase) {
            var headMs = phrase.positionMs - phrase.timeAxis.TickPosToMsPos(phrase.position - headTicks);
            var tailMs = phrase.timeAxis.TickPosToMsPos(phrase.end + tailTicks) - phrase.endMs;
            return new RenderResult() {
                leadingMs = headMs,
                positionMs = phrase.positionMs,
                estimatedLengthMs = headMs + phrase.durationMs + tailMs,
            };
        }

        public Task<RenderResult> Render(RenderPhrase phrase, Progress progress,
            int trackNo, CancellationTokenSource cancellation, bool isPreRender) {

            return Task.Run(() => {
                lock (lockObj) {
                    if (cancellation.IsCancellationRequested) {
                        return new RenderResult();
                    }

                    string progressInfo = $"Track {trackNo + 1}: {this} " +
                        $"\"{string.Join(" ", phrase.phones.Select(p => p.phoneme))}\"";
                    progress.Complete(0, progressInfo);

                    var result = Layout(phrase);
                    ulong rawHash = phrase.GetHashExcludingPostEffects(
                        Format.Ustx.DYN,
                        Format.Ustx.GENC,
                        Format.Ustx.BREC,
                        Format.Ustx.SHFC,
                        Format.Ustx.TENC,
                        Format.Ustx.VOIC);
                    ulong processedHash = phrase.GetHashExcludingPostEffects(Format.Ustx.DYN, Format.Ustx.SHFC);
                    bool hasHnsepControls = HasHnsepParameterControls(phrase);
                    string hnsepKey = hasHnsepControls
                        ? HifiHnsepOnnx.CacheKeyOrDisabled()
                        : "neutral";
                    string qualityKey = NeutrinoSinger.LegacyV2RenderQualityCacheKey();
                    string separationCacheKey =
                        $"neutrino-v2-v{cacheVersion}-q{qualityKey}-raw-{rawHash:x16}-hnsep-{hnsepKey}";
                    var wavPath = Path.Join(PathManager.Inst.CachePath,
                        $"neutrino-v2-v{cacheVersion}-q{qualityKey}-post{postEffectCacheVersion}-{processedHash:x16}-hnsep{hnsepKey}.wav");
                    var rawWavPath = Path.Join(PathManager.Inst.CachePath,
                        $"neutrino-v2-v{cacheVersion}-q{qualityKey}-raw-{rawHash:x16}.wav");
                    var rawPitchPath = Path.ChangeExtension(rawWavPath, ".f0");
                    bool needsPitchTrack = HasNonDefaultValue(phrase.gender, 0);
                    phrase.AddCacheFile(wavPath);

                    if (TryLoadWaveCache(wavPath, "processed", out var cachedSamples)) {
                        result.samples = cachedSamples;
                    }

                    if (result.samples == null) {
                        bool hasRawWaveform = TryLoadWaveCache(rawWavPath, "acoustic", out var rawSamples);
                        bool hasPitchTrack = TryLoadRawPitchCache(rawPitchPath, out var pitchTrack);
                        if (!hasRawWaveform || (needsPitchTrack && !hasPitchTrack)) {
                            var rawRender = InvokeNeutrinoV2(phrase, cancellation);
                            rawSamples = rawRender?.Samples;
                            pitchTrack = rawRender?.PitchTrack;
                            if (rawSamples != null) {
                                SaveRawWaveCache(rawWavPath, rawSamples);
                            }
                            if (pitchTrack != null) {
                                SaveRawPitchCache(rawPitchPath, pitchTrack);
                            }
                        }
                        if (rawSamples != null) {
                            result.samples = ApplyHnsepParameters(
                                phrase,
                                result,
                                rawSamples,
                                pitchTrack,
                                separationCacheKey);
                            Wave.CorrectSampleScale(result.samples);
                            SaveProcessedWaveCache(wavPath, result.samples);
                        }
                    }

                    if (result.samples != null) {
                        Renderers.ApplyDynamics(phrase, result);
                    }
                    progress.Complete(phrase.phones.Length, progressInfo);
                    return result;
                }
            });
        }

        LegacyRawRender InvokeNeutrinoV2(RenderPhrase phrase, CancellationTokenSource cancellation) {
            if (cancellation.IsCancellationRequested) return null;
            var singer = phrase.singer as NeutrinoSinger;
            if (singer == null) {
                throw new InvalidOperationException("NEUTRINO v2 renderer requires a NEUTRINO singer.");
            }
            singer.EnsureLegacyV2Sessions();
            var layout = Layout(phrase);
            var timing = BuildTimingContext(phrase, singer, singer.LegacyV2SampleRate, layout);
            if (timing.PhonemeIds.Length == 0 || timing.TotalFrames <= 0) {
                return new LegacyRawRender(
                    Array.Empty<float>(),
                    new LegacyPitchTrack(singer.LegacyV2SampleRate, singer.LegacyV2SamplesPerFrame, 0, 0, Array.Empty<float>()));
            }

            if (cancellation.IsCancellationRequested) return null;

            int numPhones = timing.PhonemeIds.Length;
            int totalFrames = timing.TotalFrames;
            var embeddingInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("electron",
                    new DenseTensor<long>(timing.PhonemeIds, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("muon",
                    new DenseTensor<long>(timing.MidiNotes, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("tau",
                    new DenseTensor<float>(timing.Durations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("selectron",
                    new DenseTensor<long>(timing.SlurFlags, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("smuon",
                    new DenseTensor<long>(timing.FramePhonemeMap, new[] { 1, totalFrames })),
            };

            var elementary = NeutrinoInferenceUtil.RequireLength(
                singer.RunLegacyEmbedding(embeddingInputs),
                totalFrames * 256,
                "NEUTRINO v2 e.bin embedding output");
            if (cancellation.IsCancellationRequested) return null;

            var acousticInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("elementary_particle",
                    new DenseTensor<float>(elementary, new[] { 1, totalFrames, 256 })),
            };
            var melSpectrogram = NeutrinoInferenceUtil.RequireLength(
                singer.RunLegacyAcoustic(acousticInputs),
                totalFrames * numMelBins,
                "NEUTRINO v2 d*.bin mel output");
            ClampMelspec(melSpectrogram);
            ApplyMelspecSilenceMask(timing, melSpectrogram);
            if (cancellation.IsCancellationRequested) return null;

            var naturalF0 = RunWorldF0(singer, melSpectrogram, totalFrames);
            var vocoderF0 = ApplyPitchEditsToNaturalF0(phrase, timing, naturalF0);
            ApplyF0SilenceMask(timing, vocoderF0);
            if (cancellation.IsCancellationRequested) return null;

            var vocoderInput = new float[totalFrames * (numMelBins + 1)];
            for (int frame = 0; frame < totalFrames; frame++) {
                for (int bin = 0; bin < numMelBins; bin++) {
                    vocoderInput[frame * (numMelBins + 1) + bin] =
                        melSpectrogram[frame * numMelBins + bin];
                }
                vocoderInput[frame * (numMelBins + 1) + numMelBins] = vocoderF0[frame];
            }

            var vocoderInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("input",
                    new DenseTensor<float>(vocoderInput, new[] { 1, totalFrames, numMelBins + 1 })),
            };

            var waveform = singer.RunLegacyVocoder(vocoderInputs);
            PostProcessWaveform(waveform, timing.SampleRate);

            int tailSamples = Math.Max(0,
                (int)(layout.estimatedLengthMs / 1000.0 * timing.SampleRate) - waveform.Length);
            int totalSamples = waveform.Length + tailSamples;
            var rendered = new float[totalSamples];
            Array.Copy(waveform, 0, rendered, 0, waveform.Length);

            if (timing.SampleRate != outputSampleRate) {
                var signal = new NWaves.Signals.DiscreteSignal(timing.SampleRate, rendered);
                signal = NWaves.Operations.Operation.Resample(signal, outputSampleRate);
                rendered = signal.Samples;
            }

            return new LegacyRawRender(
                rendered,
                new LegacyPitchTrack(timing.SampleRate, timing.HopSize, 0, waveform.Length, vocoderF0));
        }

        float[] RunWorldF0(NeutrinoSinger singer, float[] melSpectrogram, int totalFrames) {
            var worldF0Inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("input",
                    new DenseTensor<float>(melSpectrogram, new[] { 1, totalFrames, numMelBins })),
            };
            var f0 = NeutrinoInferenceUtil.RequireLength(
                singer.RunLegacyWorldF0(worldF0Inputs),
                totalFrames,
                "NEUTRINO v2 world_f0.bin F0 output");
            ClampF0(f0);
            return f0;
        }

        float[] ApplyPitchEditsToNaturalF0(RenderPhrase phrase, LegacyTimingContext timing, float[] naturalF0) {
            var f0 = (float[])naturalF0.Clone();
            for (int frame = 0; frame < f0.Length; frame++) {
                if (f0[frame] <= 0) {
                    continue;
                }
                int phoneIndex = GetFramePhoneIndex(timing, frame);
                if (phoneIndex < 0 || timing.ScorePitchCents[phoneIndex] <= 0) {
                    f0[frame] = 0;
                    continue;
                }
                int pitchIndex = GetFramePitchIndex(phrase, timing, frame);
                float editorPitchCents = phrase.pitches == null || phrase.pitches.Length == 0
                    ? timing.ScorePitchCents[phoneIndex]
                    : phrase.pitches[pitchIndex];
                float deltaCents = editorPitchCents - timing.ScorePitchCents[phoneIndex];
                deltaCents = Math.Clamp(deltaCents, -2400f, 2400f);
                if (Math.Abs(deltaCents) > 0.01f) {
                    f0[frame] *= (float)Math.Pow(2.0, deltaCents / 1200.0);
                }
            }
            ClampF0(f0);
            return f0;
        }

        LegacyTimingContext BuildTimingContext(RenderPhrase phrase, NeutrinoSinger singer, int sampleRate, RenderResult layout) {
            double originMs = layout.positionMs - layout.leadingMs;
            double estimatedEndMs = originMs + layout.estimatedLengthMs;
            var segments = BuildPhoneSegments(phrase, originMs, estimatedEndMs);
            int numPhones = segments.Count;
            if (numPhones == 0) {
                return new LegacyTimingContext(
                    Array.Empty<long>(),
                    Array.Empty<long>(),
                    Array.Empty<float>(),
                    Array.Empty<long>(),
                    Array.Empty<long>(),
                    Array.Empty<float>(),
                    0,
                    sampleRate,
                    originMs);
            }

            ApplyLegacyTimingModel(phrase, singer, segments);
            var phonemeIds = new long[numPhones];
            var midiNotes = new long[numPhones];
            var durations = new float[numPhones];
            var slurFlags = new long[numPhones];
            var scorePitchCents = new float[numPhones];
            for (int i = 0; i < segments.Count; i++) {
                var segment = segments[i];
                var note = phrase.notes[Math.Clamp(segment.noteIndex, 0, phrase.notes.Length - 1)];
                int timingId = GetLegacyPhonemeId(segment.phoneme);
                bool rest = IsLegacyPauseTimingId(timingId);
                phonemeIds[i] = GetLegacyAcousticPhonemeId(segment.phoneme);
                midiNotes[i] = rest ? 0 : Math.Clamp((int)Math.Round(note.adjustedTone), 0, 255);
                durations[i] = Math.Max(0.005f, (float)(segment.scoreEndSec - segment.scoreStartSec));
                slurFlags[i] = GetLegacySelectron(segments, i);
                scorePitchCents[i] = rest ? 0 : note.adjustedTone * 100;
            }

            double frameSec = 1.0 / frameRate;
            double totalSec = Math.Max(frameSec, segments.Max(segment => segment.endSec));
            int totalFrames = Math.Max(1, (int)Math.Ceiling(totalSec / frameSec));
            var framePhonemeMap = BuildFramePhonemeMap(segments, totalFrames, frameSec);

            return new LegacyTimingContext(
                phonemeIds,
                midiNotes,
                durations,
                slurFlags,
                framePhonemeMap,
                scorePitchCents,
                totalFrames,
                sampleRate,
                originMs);
        }

        List<PhoneSegment> BuildPhoneSegments(RenderPhrase phrase, double originMs, double estimatedEndMs) {
            var segments = new List<PhoneSegment>();
            foreach (var phone in phrase.phones) {
                var phoneStrs = NeutrinoPhoneme.RenderPhoneToPhonemes(phone.phoneme);
                if (phoneStrs.Length == 0) {
                    continue;
                }
                int noteIndex = Math.Clamp(phone.noteIndex, 0, phrase.notes.Length - 1);
                double phoneStartSec = Math.Max(0, (phone.positionMs - originMs) / 1000.0);
                double phoneEndSec = Math.Max(phoneStartSec + 0.005, (phone.endMs - originMs) / 1000.0);
                var note = phrase.notes[noteIndex];
                double scoreStartSec = Math.Max(0, (note.positionMs - originMs) / 1000.0);
                double scoreEndSec = Math.Max(scoreStartSec + 0.005, (note.endMs - originMs) / 1000.0);
                var boundaries = SplitPhoneBoundaries(phoneStartSec, phoneEndSec, phoneStrs);
                for (int i = 0; i < phoneStrs.Length; i++) {
                    segments.Add(new PhoneSegment {
                        phoneme = phoneStrs[i],
                        noteIndex = noteIndex,
                        startSec = boundaries[i],
                        endSec = boundaries[i + 1],
                        scoreStartSec = scoreStartSec,
                        scoreEndSec = scoreEndSec,
                    });
                }
            }
            segments.Sort((a, b) => a.startSec.CompareTo(b.startSec));
            AddBoundaryPauses(segments, phrase, originMs, estimatedEndMs);
            var notePositions = new Dictionary<int, int>();
            var noteCounts = segments
                .GroupBy(segment => segment.noteIndex)
                .ToDictionary(group => group.Key, group => group.Count());
            for (int i = 0; i < segments.Count; i++) {
                var segment = segments[i];
                notePositions.TryGetValue(segment.noteIndex, out int positionInNote);
                segment.positionInNote = positionInNote;
                segment.phonesInNote = noteCounts.TryGetValue(segment.noteIndex, out int count) ? count : 1;
                notePositions[segment.noteIndex] = positionInNote + 1;
                segments[i] = segment;
            }
            for (int i = 1; i < segments.Count; i++) {
                if (segments[i].startSec < segments[i - 1].endSec) {
                    var segment = segments[i];
                    segment.startSec = segments[i - 1].endSec;
                    segment.endSec = Math.Max(segment.startSec + 0.005, segment.endSec);
                    segments[i] = segment;
                }
            }
            return segments;
        }

        void AddBoundaryPauses(List<PhoneSegment> segments, RenderPhrase phrase, double originMs, double estimatedEndMs) {
            const double frameSec = 1.0 / frameRate;
            if (segments.Count == 0) {
                double totalSec = Math.Max(frameSec, (estimatedEndMs - originMs) / 1000.0);
                segments.Add(new PhoneSegment {
                    phoneme = "pau",
                    noteIndex = -1,
                    startSec = 0,
                    endSec = totalSec,
                    scoreStartSec = 0,
                    scoreEndSec = totalSec,
                });
                return;
            }

            double firstStart = segments[0].startSec;
            if (firstStart > frameSec) {
                segments.Insert(0, new PhoneSegment {
                    phoneme = "pau",
                    noteIndex = -1,
                    startSec = 0,
                    endSec = firstStart,
                    scoreStartSec = 0,
                    scoreEndSec = firstStart,
                });
            }

            double totalEndSec = Math.Max(
                segments[^1].endSec + frameSec,
                (estimatedEndMs - originMs) / 1000.0);
            double lastEnd = segments[^1].endSec;
            if (totalEndSec - lastEnd > frameSec) {
                segments.Add(new PhoneSegment {
                    phoneme = "pau",
                    noteIndex = phrase.notes.Length,
                    startSec = lastEnd,
                    endSec = totalEndSec,
                    scoreStartSec = lastEnd,
                    scoreEndSec = totalEndSec,
                });
            }
        }

        void ApplyLegacyTimingModel(RenderPhrase phrase, NeutrinoSinger singer, List<PhoneSegment> segments) {
            try {
                TryApplyLegacyTimingChunk(phrase, singer, segments, 0, segments.Count);
            } catch (Exception e) {
                Log.Warning(e, "Failed to run native NEUTRINO v2 timing model; using score phoneme timing");
            }
        }

        bool TryApplyLegacyTimingChunk(
            RenderPhrase phrase,
            NeutrinoSinger singer,
            List<PhoneSegment> segments,
            int start,
            int length) {

            if (length <= 0) {
                return false;
            }
            var rawFeatures = BuildLegacyTimingFeatures(phrase, segments, start, length);
            var deltasMs = singer.RunLegacyTiming(rawFeatures, length);
            if (deltasMs.Length != length) {
                Log.Warning("NEUTRINO v2 timing output length mismatch: actual {Actual}, expected {Expected}", deltasMs.Length, length);
                return false;
            }
            if (!AreLegacyTimingDeltasUsable(deltasMs, length, out double maxAbsDelta)) {
                Log.Warning(
                    "Rejected implausible NEUTRINO v2 timing deltas maxAbs={MaxAbs:F1}ms phones={Phones}; using score phoneme timing for this phrase",
                    maxAbsDelta,
                    length);
                return false;
            }

            double frameMs = 1000.0 / frameRate;
            var startsMs = new double[length];
            var endsMs = new double[length];
            for (int i = 0; i < length; i++) {
                var segment = segments[start + i];
                startsMs[i] = segment.scoreStartSec * 1000.0;
                endsMs[i] = segment.scoreEndSec * 1000.0;
            }

            int firstPredicted = start == 0 ? 1 : 0;
            for (int i = firstPredicted; i < length; i++) {
                var segment = segments[start + i];
                double deltaMs = Math.Clamp(deltasMs[i], -250f, 450f);
                double candidate = segment.scoreStartSec * 1000.0 + deltaMs;
                double minStart = i == 0
                    ? (start > 0 ? segments[start - 1].startSec * 1000.0 + 2 * frameMs : 0)
                    : startsMs[i - 1] + 2 * frameMs;
                double maxStart = segment.scoreEndSec * 1000.0 - frameMs;
                candidate = Math.Clamp(candidate, minStart, Math.Max(minStart, maxStart));
                startsMs[i] = Math.Floor(candidate / frameMs) * frameMs;
                if (i > 0) {
                    endsMs[i - 1] = startsMs[i];
                }
            }

            for (int i = 0; i < length; i++) {
                var segment = segments[start + i];
                segment.startSec = startsMs[i] / 1000.0;
                segment.endSec = Math.Max(segment.startSec + frameMs / 1000.0, endsMs[i] / 1000.0);
                segments[start + i] = segment;
            }
            if (start > 0) {
                var previous = segments[start - 1];
                previous.endSec = Math.Max(previous.startSec + frameMs / 1000.0, segments[start].startSec);
                segments[start - 1] = previous;
            }
            return true;
        }

        static bool AreLegacyTimingDeltasUsable(float[] deltasMs, int length, out double maxAbsDelta) {
            maxAbsDelta = 0;
            for (int i = 0; i < length; i++) {
                float delta = deltasMs[i];
                if (!float.IsFinite(delta)) {
                    maxAbsDelta = double.PositiveInfinity;
                    return false;
                }
                maxAbsDelta = Math.Max(maxAbsDelta, Math.Abs(delta));
            }
            return maxAbsDelta <= 500.0;
        }

        float[] BuildLegacyTimingFeatures(RenderPhrase phrase, List<PhoneSegment> segments, int start, int length) {
            const int rawFeatureSize = 243;
            var features = new float[length * rawFeatureSize];
            int end = start + length;
            double phraseStart = segments[start].scoreStartSec;
            double phraseEnd = segments[start].scoreEndSec;
            bool hasNonPause = false;
            for (int index = start; index < end; index++) {
                var segment = segments[index];
                if (IsLegacyPauseTimingId(GetLegacyPhonemeId(segment.phoneme))) {
                    continue;
                }
                if (!hasNonPause) {
                    phraseStart = segment.scoreStartSec;
                    phraseEnd = segment.scoreEndSec;
                    hasNonPause = true;
                } else {
                    phraseStart = Math.Min(phraseStart, segment.scoreStartSec);
                    phraseEnd = Math.Max(phraseEnd, segment.scoreEndSec);
                }
            }
            for (int i = 0; i < length; i++) {
                int segmentIndex = start + i;
                int offset = i * rawFeatureSize;
                SetPhoneOneHot(features, offset, 0, GetSegmentPhoneIdForTimingContext(segments, segmentIndex - 2));
                SetPhoneOneHot(features, offset, 1, GetSegmentPhoneIdForTimingContext(segments, segmentIndex - 1));
                SetPhoneOneHot(features, offset, 2, GetSegmentPhoneIdForTimingContext(segments, segmentIndex));
                SetPhoneOneHot(features, offset, 3, GetSegmentPhoneIdForTimingContext(segments, segmentIndex + 1));
                SetPhoneOneHot(features, offset, 4, GetSegmentPhoneIdForTimingContext(segments, segmentIndex + 2));
                SetPhoneClassFeatures(features, offset, 0, GetSegmentPhoneIdForTimingContext(segments, segmentIndex - 2));
                SetPhoneClassFeatures(features, offset, 1, GetSegmentPhoneIdForTimingContext(segments, segmentIndex - 1));
                SetPhoneClassFeatures(features, offset, 2, GetSegmentPhoneIdForTimingContext(segments, segmentIndex));
                SetPhoneClassFeatures(features, offset, 3, GetSegmentPhoneIdForTimingContext(segments, segmentIndex + 1));
                SetPhoneClassFeatures(features, offset, 4, GetSegmentPhoneIdForTimingContext(segments, segmentIndex + 2));

                var segment = segments[segmentIndex];
                int prevNoteIndex = segment.noteIndex - 1;
                int currentNoteIndex = segment.noteIndex;
                int nextNoteIndex = segment.noteIndex + 1;
                bool hasCurrentNote = HasLegacyTimingNote(phrase, currentNoteIndex);
                bool currentRest = !hasCurrentNote || IsRestLyric(phrase.notes[currentNoteIndex].lyric);
                int numeric = offset + 215;
                bool hasNextNote = HasLegacyTimingNote(phrase, nextNoteIndex);
                features[numeric + 0] = segment.positionInNote + 1;
                features[numeric + 1] = Math.Max(1, segment.phonesInNote - segment.positionInNote);
                features[numeric + 2] = DistanceFromPreviousVowel(segments, segmentIndex);
                features[numeric + 3] = DistanceToNextVowel(segments, segmentIndex);
                features[numeric + 4] = GetLegacyTimingPhonesInNote(phrase, segments, prevNoteIndex);
                features[numeric + 5] = LegacyTimingLanguage(phrase, prevNoteIndex);
                features[numeric + 6] = GetLegacyTimingPhonesInCurrentSegmentNote(phrase, segment);
                features[numeric + 7] = currentRest ? -1 : 0;
                features[numeric + 8] = GetLegacyTimingPhonesInNote(phrase, segments, nextNoteIndex);
                features[numeric + 9] = LegacyTimingLanguage(phrase, nextNoteIndex);
                features[numeric + 10] = LegacyTimingAbsScale(phrase, prevNoteIndex);
                features[numeric + 11] = LegacyTimingRelScale(phrase, prevNoteIndex);
                features[numeric + 12] = LegacyTimingNoteSyllables(phrase, prevNoteIndex);
                features[numeric + 13] = LegacyTimingNoteLengthCentiseconds(phrase, prevNoteIndex, null);
                features[numeric + 14] = LegacyTimingAbsScale(phrase, currentNoteIndex);
                features[numeric + 15] = LegacyTimingRelScale(phrase, currentNoteIndex);
                features[numeric + 16] = 1;
                features[numeric + 17] = LegacyTimingNoteLengthCentiseconds(phrase, currentNoteIndex, segment);
                features[numeric + 18] = currentRest ? 0 : Math.Clamp((float)((segment.scoreStartSec - phraseStart) * 10.0), 0f, 75f);
                features[numeric + 19] = currentRest ? 0 : Math.Clamp((float)((phraseEnd - segment.scoreStartSec) * 10.0), 0f, 75f);
                features[numeric + 20] = hasCurrentNote
                    && NeutrinoInferenceUtil.IsExtensionLyric(phrase.notes[currentNoteIndex].lyric) ? 1 : 0;
                features[numeric + 21] = hasNextNote
                    && NeutrinoInferenceUtil.IsExtensionLyric(phrase.notes[nextNoteIndex].lyric) ? 1 : 0;
                features[numeric + 22] = LegacyTimingDeltaScale(phrase, prevNoteIndex, currentNoteIndex);
                features[numeric + 23] = LegacyTimingDeltaScale(phrase, nextNoteIndex, currentNoteIndex);
                features[numeric + 24] = LegacyTimingAbsScale(phrase, nextNoteIndex);
                features[numeric + 25] = LegacyTimingRelScale(phrase, nextNoteIndex);
                features[numeric + 26] = LegacyTimingNoteSyllables(phrase, nextNoteIndex);
                features[numeric + 27] = LegacyTimingNoteLengthCentiseconds(phrase, nextNoteIndex, null);
            }
            return features;
        }

        static bool HasLegacyTimingNote(RenderPhrase phrase, int noteIndex) {
            return noteIndex >= 0 && noteIndex < phrase.notes.Length;
        }

        static bool IsLegacyTimingPitchedNote(RenderPhrase phrase, int noteIndex) {
            return HasLegacyTimingNote(phrase, noteIndex)
                && phrase.notes[noteIndex].adjustedTone > 0
                && !IsRestLyric(phrase.notes[noteIndex].lyric);
        }

        static float LegacyTimingLanguage(RenderPhrase phrase, int noteIndex) {
            return IsLegacyTimingPitchedNote(phrase, noteIndex) ? 0 : -1;
        }

        static float LegacyTimingAbsScale(RenderPhrase phrase, int noteIndex) {
            if (!IsLegacyTimingPitchedNote(phrase, noteIndex)) {
                return 0;
            }
            return LegacyTimingAbsScale(phrase.notes[noteIndex]);
        }

        static float LegacyTimingAbsScale(RenderNote note) {
            if (note.adjustedTone <= 0 || IsRestLyric(note.lyric)) {
                return 0;
            }
            return (float)Math.Log(NeutrinoConfig.MidiToFreq(note.adjustedTone));
        }

        static float LegacyTimingRelScale(RenderPhrase phrase, int noteIndex) {
            if (!IsLegacyTimingPitchedNote(phrase, noteIndex)) {
                return 0;
            }
            var note = phrase.notes[noteIndex];
            int semitone = (int)Math.Round(note.adjustedTone);
            return (semitone % 12 + 12) % 12;
        }

        static float LegacyTimingDeltaScale(RenderPhrase phrase, int otherNoteIndex, int currentNoteIndex) {
            if (!IsLegacyTimingPitchedNote(phrase, otherNoteIndex)
                || !IsLegacyTimingPitchedNote(phrase, currentNoteIndex)) {
                return 0;
            }
            var otherNote = phrase.notes[otherNoteIndex];
            var currentNote = phrase.notes[currentNoteIndex];
            int otherTone = (int)Math.Round(otherNote.adjustedTone);
            int currentTone = (int)Math.Round(currentNote.adjustedTone);
            return otherTone - currentTone;
        }

        static float LegacyTimingNoteSyllables(RenderPhrase phrase, int noteIndex) {
            return HasLegacyTimingNote(phrase, noteIndex) ? 1 : 0;
        }

        static float LegacyTimingNoteLengthCentiseconds(
            RenderPhrase phrase,
            int noteIndex,
            PhoneSegment? syntheticRestSegment) {

            if (HasLegacyTimingNote(phrase, noteIndex)) {
                return (float)(phrase.notes[noteIndex].durationMs / 10.0);
            }
            if (syntheticRestSegment.HasValue) {
                var segment = syntheticRestSegment.Value;
                return Math.Max(1f, (float)((segment.scoreEndSec - segment.scoreStartSec) * 100.0));
            }
            return 0;
        }

        static bool IsRestLyric(string lyric) {
            return string.IsNullOrWhiteSpace(lyric)
                || lyric == "R"
                || lyric.Equals("SP", StringComparison.OrdinalIgnoreCase)
                || lyric.Equals("rest", StringComparison.OrdinalIgnoreCase);
        }

        static void SetPhoneOneHot(float[] features, int offset, int group, int phoneId) {
            const int phonemeClassCount = 40;
            phoneId = Math.Clamp(phoneId, 0, phonemeClassCount - 1);
            features[offset + group * phonemeClassCount + phoneId] = 1f;
        }

        static void SetPhoneClassFeatures(float[] features, int offset, int group, int phoneId) {
            const int phoneClassOffset = 200;
            const int phoneClassSize = 3;
            int classOffset = offset + phoneClassOffset + group * phoneClassSize;
            if (IsLegacyNoSoundTimingId(phoneId)) {
                features[classOffset + 2] = 1f;
            } else if (IsLegacyVoicedTimingId(phoneId)) {
                features[classOffset] = 1f;
            } else {
                features[classOffset + 1] = 1f;
            }
        }

        int GetSegmentPhoneIdForTimingContext(List<PhoneSegment> segments, int index) {
            if (index < 0 || index >= segments.Count) {
                return legacySil;
            }
            return GetLegacyPhonemeId(segments[index].phoneme);
        }

        int DistanceFromPreviousVowel(List<PhoneSegment> segments, int index) {
            if (NeutrinoPhoneme.IsVowelPhoneme(segments[index].phoneme)) {
                return 0;
            }
            for (int i = index - 1; i >= 0 && segments[i].noteIndex == segments[index].noteIndex; i--) {
                if (NeutrinoPhoneme.IsVowelPhoneme(segments[i].phoneme)) {
                    return index - i;
                }
            }
            return 0;
        }

        int DistanceToNextVowel(List<PhoneSegment> segments, int index) {
            if (NeutrinoPhoneme.IsVowelPhoneme(segments[index].phoneme)) {
                return 0;
            }
            for (int i = index + 1; i < segments.Count && segments[i].noteIndex == segments[index].noteIndex; i++) {
                if (NeutrinoPhoneme.IsVowelPhoneme(segments[i].phoneme)) {
                    return i - index;
                }
            }
            return 0;
        }

        static int GetPhonesInNote(List<PhoneSegment> segments, int noteIndex) {
            return noteIndex < 0 ? 0 : segments.FirstOrDefault(segment => segment.noteIndex == noteIndex).phonesInNote;
        }

        static int GetLegacyTimingPhonesInNote(RenderPhrase phrase, List<PhoneSegment> segments, int noteIndex) {
            if (!HasLegacyTimingNote(phrase, noteIndex)) {
                return 0;
            }
            int phones = GetPhonesInNote(segments, noteIndex);
            return phones > 0 ? phones : 1;
        }

        static int GetLegacyTimingPhonesInCurrentSegmentNote(RenderPhrase phrase, PhoneSegment segment) {
            if (!HasLegacyTimingNote(phrase, segment.noteIndex)) {
                return Math.Max(1, segment.phonesInNote);
            }
            return Math.Max(1, segment.phonesInNote);
        }

        long GetLegacySelectron(List<PhoneSegment> segments, int index) {
            if (index <= 0 || segments[index].noteIndex != segments[index - 1].noteIndex) {
                return 0;
            }
            return Math.Abs(segments[index].scoreStartSec - segments[index - 1].scoreStartSec) < 0.0000001
                ? GetLegacySelectron(segments, index - 1) + 1
                : 0;
        }

        double[] SplitPhoneBoundaries(double startSec, double endSec, string[] phonemes) {
            var boundaries = new double[phonemes.Length + 1];
            boundaries[0] = startSec;
            boundaries[^1] = endSec;
            if (phonemes.Length == 1) {
                return boundaries;
            }

            double duration = Math.Max(0.005, endSec - startSec);
            int firstVowel = Array.FindIndex(phonemes, NeutrinoPhoneme.IsVowelPhoneme);
            if (firstVowel > 0) {
                double consonantSec = Math.Min(0.060, duration * 0.45);
                for (int i = 1; i <= firstVowel; i++) {
                    boundaries[i] = startSec + consonantSec * i / firstVowel;
                }
                int vowelParts = phonemes.Length - firstVowel;
                for (int i = firstVowel + 1; i < phonemes.Length; i++) {
                    boundaries[i] = startSec + consonantSec
                        + (duration - consonantSec) * (i - firstVowel) / vowelParts;
                }
            } else {
                for (int i = 1; i < phonemes.Length; i++) {
                    boundaries[i] = startSec + duration * i / phonemes.Length;
                }
            }
            for (int i = 1; i < boundaries.Length; i++) {
                boundaries[i] = Math.Max(boundaries[i], boundaries[i - 1] + 0.005);
            }
            boundaries[^1] = Math.Max(boundaries[^2] + 0.005, endSec);
            return boundaries;
        }

        long[] BuildFramePhonemeMap(List<PhoneSegment> segments, int totalFrames, double frameSec) {
            var framePhonemeMap = new long[totalFrames];
            for (int phone = 0; phone < segments.Count; phone++) {
                int startFrame = Math.Clamp((int)Math.Floor(segments[phone].startSec / frameSec), 0, totalFrames);
                int endFrame = Math.Clamp((int)Math.Ceiling(segments[phone].endSec / frameSec), startFrame + 1, totalFrames);
                for (int frame = startFrame; frame < endFrame; frame++) {
                    framePhonemeMap[frame] = phone + 1;
                }
            }

            long last = 1;
            for (int frame = 0; frame < framePhonemeMap.Length; frame++) {
                if (framePhonemeMap[frame] == 0) {
                    framePhonemeMap[frame] = last;
                } else {
                    last = framePhonemeMap[frame];
                }
            }
            return framePhonemeMap;
        }

        int GetFramePhoneIndex(LegacyTimingContext timing, int frame) {
            if (timing.FramePhonemeMap.Length == 0 || timing.PhonemeIds.Length == 0) {
                return -1;
            }
            int mapIndex = Math.Clamp(frame, 0, timing.FramePhonemeMap.Length - 1);
            return Math.Clamp((int)timing.FramePhonemeMap[mapIndex] - 1, 0, timing.PhonemeIds.Length - 1);
        }

        int GetFramePitchIndex(RenderPhrase phrase, LegacyTimingContext timing, int frame) {
            if (phrase.pitches == null || phrase.pitches.Length == 0) {
                return 0;
            }
            int ticks = GetFramePitchTick(phrase, timing, frame);
            return Math.Clamp((int)(ticks / (double)pitchInterval), 0, phrase.pitches.Length - 1);
        }

        int GetFramePitchTick(RenderPhrase phrase, LegacyTimingContext timing, int frame) {
            double frameMs = 1000.0 / frameRate;
            double posMs = timing.OriginMs + frame * frameMs;
            return phrase.timeAxis.MsPosToTickPos(posMs) - (phrase.position - phrase.leading);
        }

        int GetFrameResultTick(RenderPhrase phrase, LegacyTimingContext timing, int frame) {
            double frameMs = 1000.0 / frameRate;
            double posMs = timing.OriginMs + frame * frameMs;
            return phrase.timeAxis.MsPosToTickPos(posMs) - phrase.position;
        }

        void ClampF0(float[] f0) {
            for (int i = 0; i < f0.Length; i++) {
                if (!float.IsFinite(f0[i]) || f0[i] < f0Min) {
                    f0[i] = 0;
                } else if (f0[i] > f0Max) {
                    f0[i] = f0Max;
                }
            }
        }

        void ClampMelspec(float[] melSpectrogram) {
            for (int i = 0; i < melSpectrogram.Length; i++) {
                if (!float.IsFinite(melSpectrogram[i])) {
                    melSpectrogram[i] = melspecMin;
                } else if (melSpectrogram[i] < melspecMin) {
                    melSpectrogram[i] = melspecMin;
                } else if (melSpectrogram[i] > melspecMax) {
                    melSpectrogram[i] = melspecMax;
                }
            }
        }

        void ApplyMelspecSilenceMask(LegacyTimingContext timing, float[] melSpectrogram) {
            int frames = Math.Min(timing.TotalFrames, melSpectrogram.Length / numMelBins);
            for (int frame = 0; frame < frames; frame++) {
                int phoneIndex = GetFramePhoneIndex(timing, frame);
                if (phoneIndex >= 0 && IsLegacyPauseAcousticId((int)timing.PhonemeIds[phoneIndex])) {
                    int offset = frame * numMelBins;
                    for (int bin = 0; bin < numMelBins; bin++) {
                        melSpectrogram[offset + bin] = melspecMin;
                    }
                }
            }
        }

        void ApplyF0SilenceMask(LegacyTimingContext timing, float[] f0) {
            int frames = Math.Min(timing.TotalFrames, f0.Length);
            for (int frame = 0; frame < frames; frame++) {
                int phoneIndex = GetFramePhoneIndex(timing, frame);
                if (phoneIndex >= 0 && IsLegacyPauseAcousticId((int)timing.PhonemeIds[phoneIndex])) {
                    f0[frame] = 0;
                }
            }
        }

        void PostProcessWaveform(float[] waveform, int sampleRate) {
            int edgeSamples = Math.Max(1, sampleRate / frameRate);
            int fadeSamples = edgeSamples;
            int edge = Math.Min(edgeSamples, waveform.Length / 2);
            for (int i = 0; i < edge; i++) {
                waveform[i] = 0;
                waveform[waveform.Length - 1 - i] = 0;
            }

            int fadeIn = Math.Min(fadeSamples, Math.Max(0, waveform.Length - edge));
            for (int i = 0; i < fadeIn; i++) {
                int index = edge + i;
                if (index >= waveform.Length) break;
                float gain = (float)Math.Pow((double)i / fadeSamples, 2.0);
                waveform[index] *= gain;
            }

            int fadeOut = Math.Min(fadeSamples, Math.Max(0, waveform.Length - edge));
            for (int i = 0; i < fadeOut; i++) {
                int index = waveform.Length - edge - 1 - i;
                if (index < 0) break;
                float gain = (float)Math.Pow((double)i / fadeSamples, 2.0);
                waveform[index] *= gain;
            }

            for (int i = 0; i < waveform.Length; i++) {
                float value = waveform[i] * wavScale;
                if (value > wavClamp) value = wavClamp;
                if (value < -wavClamp) value = -wavClamp;
                waveform[i] = value;
            }
        }

        static bool HasHnsepParameterControls(RenderPhrase phrase) {
            return HasNonDefaultValue(phrase.gender, 0)
                || HasNonDefaultValue(phrase.breathiness, 0)
                || HasNonDefaultValue(phrase.tension, 0)
                || HasNonDefaultValue(phrase.voicing, 100);
        }

        static bool HasNonDefaultValue(float[] values, float defaultValue) {
            if (values == null) {
                return false;
            }
            foreach (float value in values) {
                if (Math.Abs(value - defaultValue) > 0.5f) {
                    return true;
                }
            }
            return false;
        }

        float[] ApplyHnsepParameters(
            RenderPhrase phrase,
            RenderResult layout,
            float[] waveform,
            LegacyPitchTrack pitchTrack,
            string separationCacheKey) {
            if (waveform.Length == 0) {
                return waveform;
            }

            int frameCount = Math.Max(1,
                (int)Math.Ceiling(waveform.Length / (double)HifiOnnxVocoder.HopSize));
            double phraseStartMs = layout.positionMs - layout.leadingMs;
            var parameterTrack = HifiParameterCurves.TrackForFrames(
                phrase,
                phraseStartMs,
                startFrame: 0,
                frameCount);
            if (!parameterTrack.NeedsHnsep && !parameterTrack.HasGender) {
                return waveform;
            }

            Func<double, int, double>? pitchAtSourceSample = pitchTrack == null
                ? null
                : pitchTrack.GetF0AtOutputSample;
            var processed = HifiHnsepSourceProcessor.ApplyGeneratedWaveform(
                waveform,
                parameterTrack,
                separationCacheKey,
                pitchAtSourceSample,
                out var report);
            if (report.Applied) {
                return processed;
            } else {
                Log.Warning(
                    "NEUTRINO v2 HNSEP parameters skipped genc={Genc:F2} brec={Brec:F2} voic={Voic:F2} tenc={Tenc:F2} reason={Reason}",
                    parameterTrack.Average.Gender,
                    parameterTrack.Average.Breathiness,
                    parameterTrack.Average.Voicing,
                    parameterTrack.Average.Tension,
                    report.Reason);
                return waveform;
            }
        }

        static bool TryLoadWaveCache(string path, string cacheKind, out float[] samples) {
            samples = null;
            if (!File.Exists(path)) {
                return false;
            }
            try {
                using var waveStream = Wave.OpenFile(path);
                samples = Wave.GetSamples(waveStream.ToSampleProvider().ToMono(1, 0));
                return true;
            } catch (Exception e) {
                Log.Error(e, "Failed to read NEUTRINO v2 {CacheKind} cache, re-rendering", cacheKind);
                return false;
            }
        }

        static void SaveRawWaveCache(string path, float[] samples) {
            using var writer = new WaveFileWriter(
                path,
                WaveFormat.CreateIeeeFloatWaveFormat(outputSampleRate, 1));
            writer.WriteSamples(samples, 0, samples.Length);
        }

        static bool TryLoadRawPitchCache(string path, out LegacyPitchTrack pitchTrack) {
            pitchTrack = null;
            if (!File.Exists(path)) {
                return false;
            }
            try {
                using var stream = File.OpenRead(path);
                using var reader = new BinaryReader(stream);
                if (reader.ReadInt32() != pitchCacheMagic) {
                    return false;
                }
                int sampleRate = reader.ReadInt32();
                int hopSize = reader.ReadInt32();
                int headSamples = reader.ReadInt32();
                int vocoderSamples = reader.ReadInt32();
                int count = reader.ReadInt32();
                if (sampleRate <= 0 || hopSize <= 0 || headSamples < 0 || vocoderSamples < 0
                    || count < 0 || count > 1_000_000
                    || stream.Length - stream.Position < count * sizeof(float)) {
                    return false;
                }
                var f0 = new float[count];
                for (int i = 0; i < f0.Length; i++) {
                    float value = reader.ReadSingle();
                    f0[i] = float.IsFinite(value) ? value : 0;
                }
                pitchTrack = new LegacyPitchTrack(sampleRate, hopSize, headSamples, vocoderSamples, f0);
                return true;
            } catch (Exception e) {
                Log.Warning(e, "Failed to read NEUTRINO v2 F0 cache, regenerating acoustic output");
                return false;
            }
        }

        static void SaveRawPitchCache(string path, LegacyPitchTrack pitchTrack) {
            string tempPath = path + ".tmp";
            try {
                using (var stream = File.Create(tempPath))
                using (var writer = new BinaryWriter(stream)) {
                    writer.Write(pitchCacheMagic);
                    writer.Write(pitchTrack.SampleRate);
                    writer.Write(pitchTrack.HopSize);
                    writer.Write(pitchTrack.HeadSamples);
                    writer.Write(pitchTrack.VocoderSamples);
                    writer.Write(pitchTrack.Frames.Length);
                    foreach (float value in pitchTrack.Frames) {
                        writer.Write(float.IsFinite(value) ? value : 0);
                    }
                }
                File.Move(tempPath, path, overwrite: true);
            } catch (Exception e) {
                Log.Warning(e, "Failed to write NEUTRINO v2 F0 cache path={Path}", path);
                try {
                    if (File.Exists(tempPath)) {
                        File.Delete(tempPath);
                    }
                } catch (Exception cleanupException) {
                    Log.Debug(cleanupException, "Failed to remove incomplete NEUTRINO v2 F0 cache path={Path}", tempPath);
                }
            }
        }

        static void SaveProcessedWaveCache(string path, float[] samples) {
            var source = new WaveSource(0, 0, 0, 1);
            source.SetSamples(samples);
            WaveFileWriter.CreateWaveFile16(path, new ExportAdapter(source).ToMono(1, 0));
        }

        static int GetLegacyPhonemeId(string phoneme) {
            phoneme = phoneme?.Trim();
            if (string.IsNullOrEmpty(phoneme)
                || phoneme == "R"
                || phoneme.Equals("SP", StringComparison.OrdinalIgnoreCase)
                || phoneme.Equals("rest", StringComparison.OrdinalIgnoreCase)) {
                return legacyPau;
            }
            if (legacyPhonemeToId.TryGetValue(phoneme, out int id)) {
                return id;
            }
            if (legacyPhonemeToId.TryGetValue(phoneme.ToLowerInvariant(), out id)) {
                return id;
            }
            Log.Warning("Unknown NEUTRINO v2 phoneme: {Phoneme}", phoneme);
            return legacyPau;
        }

        static int GetLegacyAcousticPhonemeId(string phoneme) {
            return GetLegacyPhonemeId(phoneme) + legacyAcousticPhonemeOffset;
        }

        static bool IsLegacyPauseTimingId(int id) {
            return id == legacyPau || id == legacySil;
        }

        static bool IsLegacyPhraseBreakPhoneme(string phoneme) {
            return string.Equals(phoneme, "br", StringComparison.OrdinalIgnoreCase);
        }

        static bool IsLegacyNoSoundTimingId(int id) {
            return id == legacyPau || id == legacySil;
        }

        static bool IsLegacyVoicedTimingId(int id) {
            return id == 0   // a
                || id == 1   // b
                || id == 3   // by
                || id == 6   // d
                || id == 7   // dy
                || id == 8   // e
                || id == 10  // g
                || id == 11  // gy
                || id == 14  // i
                || id == 15  // j
                || id == 18  // m
                || id == 19  // my
                || id == 20  // n
                || id == 21  // N
                || id == 22  // ny
                || id == 23  // o
                || id == 27  // r
                || id == 28  // ry
                || id == 35  // u
                || id == 36  // v
                || id == 37  // w
                || id == 38  // y
                || id == 39; // z
        }

        static bool IsLegacyPauseAcousticId(int id) {
            return id == legacyPau + legacyAcousticPhonemeOffset
                || id == legacySil + legacyAcousticPhonemeOffset;
        }

        public RenderPitchResult LoadRenderedPitch(RenderPhrase phrase) {
            var singer = phrase.singer as NeutrinoSinger;
            if (singer == null) {
                return null;
            }
            singer.EnsureLegacyV2Sessions();
            var layout = Layout(phrase);
            var timing = BuildTimingContext(phrase, singer, singer.LegacyV2SampleRate, layout);
            if (timing.TotalFrames <= 0) {
                return null;
            }

            int numPhones = timing.PhonemeIds.Length;
            int totalFrames = timing.TotalFrames;
            var embeddingInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("electron",
                    new DenseTensor<long>(timing.PhonemeIds, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("muon",
                    new DenseTensor<long>(timing.MidiNotes, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("tau",
                    new DenseTensor<float>(timing.Durations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("selectron",
                    new DenseTensor<long>(timing.SlurFlags, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("smuon",
                    new DenseTensor<long>(timing.FramePhonemeMap, new[] { 1, totalFrames })),
            };
            var elementary = NeutrinoInferenceUtil.RequireLength(
                singer.RunLegacyEmbedding(embeddingInputs),
                totalFrames * 256,
                "NEUTRINO v2 e.bin embedding output");
            var acousticInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("elementary_particle",
                    new DenseTensor<float>(elementary, new[] { 1, totalFrames, 256 })),
            };
            var melSpectrogram = NeutrinoInferenceUtil.RequireLength(
                singer.RunLegacyAcoustic(acousticInputs),
                totalFrames * numMelBins,
                "NEUTRINO v2 d*.bin mel output");
            ClampMelspec(melSpectrogram);
            ApplyMelspecSilenceMask(timing, melSpectrogram);
            var f0 = RunWorldF0(singer, melSpectrogram, totalFrames);
            ApplyF0SilenceMask(timing, f0);
            var result = new RenderPitchResult {
                ticks = new float[f0.Length],
                tones = new float[f0.Length],
            };

            for (int frame = 0; frame < f0.Length; frame++) {
                result.ticks[frame] = GetFrameResultTick(phrase, timing, frame);
                int phoneIndex = GetFramePhoneIndex(timing, frame);
                bool voiced = phoneIndex >= 0
                    && !IsLegacyPauseAcousticId((int)timing.PhonemeIds[phoneIndex])
                    && timing.ScorePitchCents[phoneIndex] > 0
                    && f0[frame] > 0;
                result.tones[frame] = voiced
                    ? (float)MusicMath.FreqToTone(f0[frame])
                    : -1f;
            }
            return result;
        }

        public UExpressionDescriptor[] GetSuggestedExpressions(
            USinger singer, URenderSettings renderSettings) {
            return Array.Empty<UExpressionDescriptor>();
        }

        public override string ToString() => Renderers.NEUTRINO_V2;
    }
}
