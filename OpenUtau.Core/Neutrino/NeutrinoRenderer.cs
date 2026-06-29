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
using Serilog;

namespace OpenUtau.Core.Neutrino {
    public class NeutrinoRenderer : IRenderer {
        public const int headTicks = 480;
        public const int tailTicks = 480;

        const int sampleRate = 48000;
        const int outputSampleRate = 44100;
        const int hopSize = 480;
        const int pitchInterval = 5;
        const int numMelBins = 100;
        const int cacheVersion = 12;
        const int pitchCacheMagic = 0x4E465032; // NFP2
        const int edgeSilenceSamples = 240;
        const int fadeInSamples = 240;
        const int fadeOutSamples = 240;
        const float f0Min = 40f;
        const float f0Max = 2000f;
        const float melspecMin = -7f;
        const float melspecMax = 1f;
        const float wavScale = 0.9885531068f;
        const float wavClamp = 0.9988493919f;

        static readonly HashSet<string> supportedExp = new HashSet<string>() {
            Format.Ustx.DYN,
            Format.Ustx.PITD,
            Format.Ustx.GENC,
            Format.Ustx.BREC,
            Format.Ustx.TENC,
            Format.Ustx.VOIC,
        };

        static readonly object lockObj = new object();

        sealed class NeutrinoRawRender {
            public float[] Samples { get; }
            public NeutrinoPitchTrack PitchTrack { get; }

            public NeutrinoRawRender(float[] samples, NeutrinoPitchTrack pitchTrack) {
                Samples = samples;
                PitchTrack = pitchTrack;
            }
        }

        sealed class NeutrinoTimingContext {
            public long[] PhonemeIds { get; }
            public float[] ScorePitchesHz { get; }
            public float[] ScoreDurations { get; }
            public long[] PhonePositions { get; }
            public float[] TimingDurations { get; }
            public long[] FramePhonemeMap { get; }
            public int TotalFrames { get; }

            public NeutrinoTimingContext(
                long[] phonemeIds,
                float[] scorePitchesHz,
                float[] scoreDurations,
                long[] phonePositions,
                float[] timingDurations,
                long[] framePhonemeMap,
                int totalFrames) {

                PhonemeIds = phonemeIds;
                ScorePitchesHz = scorePitchesHz;
                ScoreDurations = scoreDurations;
                PhonePositions = phonePositions;
                TimingDurations = timingDurations;
                FramePhonemeMap = framePhonemeMap;
                TotalFrames = totalFrames;
            }
        }

        sealed class NeutrinoPitchTrack {
            readonly int headSamplesAt48k;
            readonly int vocoderSamplesAt48k;
            readonly float[] f0;

            public int HeadSamplesAt48k => headSamplesAt48k;
            public int VocoderSamplesAt48k => vocoderSamplesAt48k;
            public float[] Frames => f0;

            public NeutrinoPitchTrack(int headSamplesAt48k, int vocoderSamplesAt48k, float[] f0) {
                this.headSamplesAt48k = Math.Max(0, headSamplesAt48k);
                this.vocoderSamplesAt48k = Math.Max(0, vocoderSamplesAt48k);
                this.f0 = f0 ?? Array.Empty<float>();
            }

            public double GetF0AtOutputSample(double outputSample, int _) {
                if (f0.Length == 0) {
                    return 0;
                }
                double sourceSample = outputSample * sampleRate / outputSampleRate;
                double frameIndex = (sourceSample - headSamplesAt48k) / hopSize;
                if (frameIndex < 0 || sourceSample >= headSamplesAt48k + vocoderSamplesAt48k) {
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
                        Format.Ustx.TENC,
                        Format.Ustx.VOIC);
                    ulong processedHash = phrase.GetHashExcludingPostEffects(Format.Ustx.DYN);
                    bool hasHnsepControls = HasHnsepParameterControls(phrase);
                    string hnsepKey = hasHnsepControls
                        ? HifiHnsepOnnx.CacheKeyOrDisabled()
                        : "neutral";
                    string separationCacheKey =
                        $"neutrino-v{cacheVersion}-raw-{rawHash:x16}-hnsep-{hnsepKey}";
                    var wavPath = Path.Join(PathManager.Inst.CachePath,
                        $"neutrino-v{cacheVersion}-{processedHash:x16}-hnsep{hnsepKey}.wav");
                    var rawWavPath = Path.Join(PathManager.Inst.CachePath,
                        $"neutrino-v{cacheVersion}-raw-{rawHash:x16}.wav");
                    var rawPitchPath = Path.ChangeExtension(rawWavPath, ".f0");
                    bool needsPitchTrack = HasNonDefaultValue(phrase.gender, 0);
                    phrase.AddCacheFile(wavPath);
                    // Keep raw and separated-waveform caches across post-effect edits. Their
                    // hashes contain the acoustic input and HNSEP model identity, so stale
                    // entries cannot be reused for changed notes, timing, or models.

                    if (TryLoadWaveCache(wavPath, "processed", out var cachedSamples)) {
                        result.samples = cachedSamples;
                    }

                    if (result.samples == null) {
                        bool hasRawWaveform = TryLoadWaveCache(rawWavPath, "acoustic", out var rawSamples);
                        bool hasPitchTrack = TryLoadRawPitchCache(rawPitchPath, out var pitchTrack);
                        if (!hasRawWaveform || (needsPitchTrack && !hasPitchTrack)) {
                            var rawRender = InvokeNeutrino(phrase, cancellation);
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

        NeutrinoRawRender InvokeNeutrino(RenderPhrase phrase, CancellationTokenSource cancellation) {
            if (cancellation.IsCancellationRequested) return null;
            var timing = BuildTimingContext(phrase);
            if (timing.PhonemeIds.Length == 0) {
                return new NeutrinoRawRender(
                    Array.Empty<float>(),
                    new NeutrinoPitchTrack(0, 0, Array.Empty<float>()));
            }

            if (cancellation.IsCancellationRequested) return null;

            float[] f0 = BuildEditorF0(phrase, timing);
            ClampF0(f0);

            if (cancellation.IsCancellationRequested) return null;

            int numPhones = timing.PhonemeIds.Length;
            int totalFrames = timing.TotalFrames;
            var singer = phrase.singer as NeutrinoSinger;
            var melspecInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("electron",
                    new DenseTensor<long>(timing.PhonemeIds, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("muon",
                    new DenseTensor<float>(timing.TimingDurations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("tau",
                    new DenseTensor<float>(timing.ScorePitchesHz, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("selectron",
                    new DenseTensor<float>(timing.ScoreDurations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("smuon",
                    new DenseTensor<long>(timing.PhonePositions, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("stau",
                    new DenseTensor<long>(timing.FramePhonemeMap, new[] { 1, totalFrames })),
                NamedOnnxValue.CreateFromTensor("photon",
                    new DenseTensor<float>(f0, new[] { 1, totalFrames })),
            };

            float[] melSpectrogram;
            melSpectrogram = singer.RunMelspec(melspecInputs);
            melSpectrogram = FitLength(melSpectrogram, totalFrames * numMelBins);
            ClampMelspec(melSpectrogram);

            if (cancellation.IsCancellationRequested) return null;

            int vocoderFrames = totalFrames;
            var vocoderInput = new float[vocoderFrames * (numMelBins + 1)];
            for (int frame = 0; frame < vocoderFrames; frame++) {
                for (int bin = 0; bin < numMelBins; bin++) {
                    vocoderInput[frame * (numMelBins + 1) + bin] =
                        melSpectrogram[frame * numMelBins + bin];
                }
                vocoderInput[frame * (numMelBins + 1) + numMelBins] = f0[frame];
            }

            var vocoderInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("input",
                    new DenseTensor<float>(vocoderInput, new[] { 1, vocoderFrames, numMelBins + 1 })),
            };

            float[] waveform;
            waveform = singer.RunVocoder(vocoderInputs);
            PostProcessWaveform(waveform);

            var layout = Layout(phrase);
            int headSamples = (int)(layout.leadingMs / 1000.0 * sampleRate);
            int tailSamples = Math.Max(0,
                (int)(layout.estimatedLengthMs / 1000.0 * sampleRate) - headSamples - waveform.Length);

            int totalSamples = headSamples + waveform.Length + tailSamples;
            var result = new float[totalSamples];
            Array.Copy(waveform, 0, result, headSamples, waveform.Length);

            if (sampleRate != outputSampleRate) {
                var signal = new NWaves.Signals.DiscreteSignal(sampleRate, result);
                signal = NWaves.Operations.Operation.Resample(signal, outputSampleRate);
                result = signal.Samples;
            }

            return new NeutrinoRawRender(
                result,
                new NeutrinoPitchTrack(headSamples, waveform.Length, f0));
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
            NeutrinoPitchTrack pitchTrack,
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
                    "NEUTRINO HNSEP parameters skipped genc={Genc:F2} brec={Brec:F2} voic={Voic:F2} tenc={Tenc:F2} reason={Reason}",
                    parameterTrack.Average.Gender,
                    parameterTrack.Average.Breathiness,
                    parameterTrack.Average.Voicing,
                    parameterTrack.Average.Tension,
                    report.Reason);
                return waveform;
            }
        }

        NeutrinoTimingContext BuildTimingContext(RenderPhrase phrase) {
            var singer = phrase.singer as NeutrinoSinger;
            var (phonemeIds, scorePitchesHz, scoreDurations, phonePositions, manualBoundaries) =
                BuildPhonemeSequence(phrase);

            int numPhones = phonemeIds.Length;
            if (numPhones == 0) {
                return new NeutrinoTimingContext(
                    phonemeIds,
                    scorePitchesHz,
                    scoreDurations,
                    phonePositions,
                    Array.Empty<float>(),
                    Array.Empty<long>(),
                    0);
            }

            var timingInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("electron",
                    new DenseTensor<long>(phonemeIds, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("muon",
                    new DenseTensor<float>(scorePitchesHz, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("tau",
                    new DenseTensor<float>(scoreDurations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("selectron",
                    new DenseTensor<long>(phonePositions, new[] { 1, numPhones })),
            };

            float[] boundaryShifts = singer.RunTiming(timingInputs);
            var baseBoundaries = BuildBaseBoundaryTimes(scoreDurations, phonePositions);
            var boundaries = ApplyTimingBoundaryShifts(baseBoundaries, boundaryShifts);
            ApplyManualBoundaryOverrides(boundaries, manualBoundaries);
            var timingDurations = BuildTimingDurations(boundaries);
            int totalFrames = Math.Max(1, (int)Math.Round(boundaries[^1] * sampleRate / hopSize));
            var framePhonemeMap = BuildFramePhonemeMap(timingDurations, totalFrames);

            return new NeutrinoTimingContext(
                phonemeIds,
                scorePitchesHz,
                scoreDurations,
                phonePositions,
                timingDurations,
                framePhonemeMap,
                totalFrames);
        }

        float[] RunPredictedF0(RenderPhrase phrase, NeutrinoTimingContext timing) {
            if (timing.TotalFrames <= 0 || timing.PhonemeIds.Length == 0) {
                return Array.Empty<float>();
            }

            int numPhones = timing.PhonemeIds.Length;
            var pitchInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("electron",
                    new DenseTensor<long>(timing.PhonemeIds, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("muon",
                    new DenseTensor<float>(timing.TimingDurations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("tau",
                    new DenseTensor<float>(timing.ScorePitchesHz, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("selectron",
                    new DenseTensor<float>(timing.ScoreDurations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("smuon",
                    new DenseTensor<long>(timing.PhonePositions, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("stau",
                    new DenseTensor<long>(timing.FramePhonemeMap, new[] { 1, timing.TotalFrames })),
            };

            var singer = phrase.singer as NeutrinoSinger;
            var f0 = FitLength(singer.RunPitch(pitchInputs), timing.TotalFrames);
            ClampF0(f0);
            return f0;
        }

        float[] BuildEditorF0(RenderPhrase phrase, NeutrinoTimingContext timing) {
            var f0 = new float[timing.TotalFrames];
            for (int frame = 0; frame < f0.Length; frame++) {
                int phoneIndex = GetFramePhoneIndex(timing, frame);
                if (phoneIndex < 0
                    || timing.PhonemeIds[phoneIndex] == NeutrinoPhoneme.PAU
                    || timing.ScorePitchesHz[phoneIndex] <= 0) {
                    continue;
                }

                if (phrase.pitches == null || phrase.pitches.Length == 0) {
                    f0[frame] = timing.ScorePitchesHz[phoneIndex];
                    continue;
                }

                int pitchIndex = GetFramePitchIndex(phrase, frame);
                f0[frame] = (float)MusicMath.ToneToFreq(phrase.pitches[pitchIndex] * 0.01);
            }
            return f0;
        }

        int GetFramePhoneIndex(NeutrinoTimingContext timing, int frame) {
            if (timing.FramePhonemeMap.Length == 0 || timing.PhonemeIds.Length == 0) {
                return -1;
            }
            int mapIndex = Math.Clamp(frame, 0, timing.FramePhonemeMap.Length - 1);
            return Math.Clamp((int)timing.FramePhonemeMap[mapIndex] - 1, 0, timing.PhonemeIds.Length - 1);
        }

        int GetFramePitchIndex(RenderPhrase phrase, int frame) {
            int ticks = GetFramePitchTick(phrase, frame);
            return Math.Clamp((int)(ticks / (double)pitchInterval), 0, phrase.pitches.Length - 1);
        }

        int GetFramePitchTick(RenderPhrase phrase, int frame) {
            double frameMs = 1000.0 * hopSize / sampleRate;
            double posMs = phrase.positionMs - phrase.leadingMs + frame * frameMs;
            return phrase.timeAxis.MsPosToTickPos(posMs) - (phrase.position - phrase.leading);
        }

        int GetFrameResultTick(RenderPhrase phrase, int frame) {
            double frameMs = 1000.0 * hopSize / sampleRate;
            double posMs = phrase.positionMs - phrase.leadingMs + frame * frameMs;
            return phrase.timeAxis.MsPosToTickPos(posMs) - phrase.position;
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
                Log.Error(e, "Failed to read NEUTRINO {CacheKind} cache, re-rendering", cacheKind);
                return false;
            }
        }

        static void SaveRawWaveCache(string path, float[] samples) {
            using var writer = new WaveFileWriter(
                path,
                WaveFormat.CreateIeeeFloatWaveFormat(outputSampleRate, 1));
            writer.WriteSamples(samples, 0, samples.Length);
        }

        static bool TryLoadRawPitchCache(string path, out NeutrinoPitchTrack pitchTrack) {
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
                int headSamplesAt48k = reader.ReadInt32();
                int vocoderSamplesAt48k = reader.ReadInt32();
                int count = reader.ReadInt32();
                if (headSamplesAt48k < 0 || vocoderSamplesAt48k < 0 || count < 0 || count > 1_000_000
                    || stream.Length - stream.Position < count * sizeof(float)) {
                    return false;
                }
                var f0 = new float[count];
                for (int i = 0; i < f0.Length; i++) {
                    float value = reader.ReadSingle();
                    f0[i] = float.IsFinite(value) ? value : 0;
                }
                pitchTrack = new NeutrinoPitchTrack(headSamplesAt48k, vocoderSamplesAt48k, f0);
                return true;
            } catch (Exception e) {
                Log.Warning(e, "Failed to read NEUTRINO F0 cache, regenerating acoustic output");
                return false;
            }
        }

        static void SaveRawPitchCache(string path, NeutrinoPitchTrack pitchTrack) {
            string tempPath = path + ".tmp";
            try {
                using (var stream = File.Create(tempPath))
                using (var writer = new BinaryWriter(stream)) {
                    writer.Write(pitchCacheMagic);
                    writer.Write(pitchTrack.HeadSamplesAt48k);
                    writer.Write(pitchTrack.VocoderSamplesAt48k);
                    writer.Write(pitchTrack.Frames.Length);
                    foreach (float value in pitchTrack.Frames) {
                        writer.Write(float.IsFinite(value) ? value : 0);
                    }
                }
                File.Move(tempPath, path, overwrite: true);
            } catch (Exception e) {
                Log.Warning(e, "Failed to write NEUTRINO F0 cache path={Path}", path);
                try {
                    if (File.Exists(tempPath)) {
                        File.Delete(tempPath);
                    }
                } catch (Exception cleanupException) {
                    Log.Debug(cleanupException, "Failed to remove incomplete NEUTRINO F0 cache path={Path}", tempPath);
                }
            }
        }

        static void SaveProcessedWaveCache(string path, float[] samples) {
            var source = new WaveSource(0, 0, 0, 1);
            source.SetSamples(samples);
            WaveFileWriter.CreateWaveFile16(path, new ExportAdapter(source).ToMono(1, 0));
        }

        (long[] phonemeIds, float[] scorePitchesHz, float[] scoreDurations, long[] phonePositions, double?[] manualBoundaries)
            BuildPhonemeSequence(RenderPhrase phrase) {

            var phonemeIds = new List<long>();
            var scorePitchesHz = new List<float>();
            var scoreDurations = new List<float>();
            var phonePositions = new List<long>();
            var manualBoundaries = new List<double?>();
            int lastNoteIndex = -1;
            int positionInNote = 0;

            foreach (var phone in phrase.phones) {
                var phoneStrs = NeutrinoPhoneme.KanaToPhonemes(phone.phoneme);
                int noteIndex = Math.Clamp(phone.noteIndex, 0, phrase.notes.Length - 1);
                if (noteIndex != lastNoteIndex) {
                    positionInNote = 0;
                    lastNoteIndex = noteIndex;
                }

                var note = phrase.notes[noteIndex];
                float notePitchHz = phoneStrs.All(p => NeutrinoPhoneme.GetPhonemeId(p) == NeutrinoPhoneme.PAU)
                    ? 0
                    : (float)NeutrinoConfig.MidiToFreq(note.tone + note.tuning * 0.01f);
                float noteDurationSec = Math.Max(0.001f, (float)(GetExtendedNoteDurationMs(phrase.notes, noteIndex) / 1000.0));

                for (int i = 0; i < phoneStrs.Length; i++) {
                    int id = NeutrinoPhoneme.GetPhonemeId(phoneStrs[i]);
                    phonemeIds.Add(id);
                    scorePitchesHz.Add(id == NeutrinoPhoneme.PAU ? 0 : notePitchHz);
                    scoreDurations.Add(noteDurationSec);
                    phonePositions.Add(positionInNote++);
                    manualBoundaries.Add(phone.positionOverridden && i == 0
                        ? Math.Max(0, (phone.positionMs - phrase.positionMs) / 1000.0)
                        : null);
                }
            }

            if (phonemeIds.Count == 0) {
                return (
                    Array.Empty<long>(),
                    Array.Empty<float>(),
                    Array.Empty<float>(),
                    Array.Empty<long>(),
                    new double?[] { null }
                );
            }
            manualBoundaries.Add(null);
            return (
                phonemeIds.ToArray(),
                scorePitchesHz.ToArray(),
                scoreDurations.ToArray(),
                phonePositions.ToArray(),
                manualBoundaries.ToArray()
            );
        }

        double GetExtendedNoteDurationMs(RenderNote[] notes, int noteIndex) {
            double endMs = notes[noteIndex].endMs;
            for (int i = noteIndex + 1; i < notes.Length && IsExtensionLyric(notes[i].lyric); i++) {
                endMs = notes[i].endMs;
            }
            return Math.Max(1, endMs - notes[noteIndex].positionMs);
        }

        bool IsExtensionLyric(string lyric) {
            return lyric == "+" || lyric == "-";
        }

        double[] BuildBaseBoundaryTimes(float[] scoreDurations, long[] phonePositions) {
            int numPhones = scoreDurations.Length;
            var boundaries = new double[numPhones + 1];
            double time = 0;
            for (int i = 0; i < numPhones; i++) {
                boundaries[i] = time;
                long nextPosition = i + 1 < numPhones ? phonePositions[i + 1] : -1;
                if (i == numPhones - 1 || nextPosition <= phonePositions[i]) {
                    time += scoreDurations[i];
                }
            }
            boundaries[numPhones] = time;
            return boundaries;
        }

        double[] ApplyTimingBoundaryShifts(double[] baseBoundaries, float[] boundaryShifts) {
            var boundaries = (double[])baseBoundaries.Clone();
            double frameSec = (double)hopSize / sampleRate;
            for (int i = 1; i < boundaries.Length - 1; i++) {
                double shift = i < boundaryShifts.Length ? boundaryShifts[i] : 0;
                double shifted = baseBoundaries[i] + shift;
                boundaries[i] = Math.Round(Math.Max(shifted, boundaries[i - 1] + frameSec) * 1000.0) / 1000.0;
            }
            for (int i = 1; i < boundaries.Length; i++) {
                if (boundaries[i] <= boundaries[i - 1]) {
                    boundaries[i] = Math.Round((boundaries[i - 1] + frameSec) * 1000.0) / 1000.0;
                }
            }
            return boundaries;
        }

        void ApplyManualBoundaryOverrides(double[] boundaries, double?[] manualBoundaries) {
            if (manualBoundaries == null || manualBoundaries.Length == 0) {
                return;
            }

            double frameSec = (double)hopSize / sampleRate;
            int count = Math.Min(boundaries.Length - 1, manualBoundaries.Length - 1);
            for (int i = 1; i < count; i++) {
                if (!manualBoundaries[i].HasValue) {
                    continue;
                }

                double min = boundaries[i - 1] + frameSec;
                double max = boundaries[i + 1] - frameSec;
                if (max < min) {
                    max = min;
                }
                boundaries[i] = Math.Round(Math.Clamp(manualBoundaries[i].Value, min, max) * 1000.0) / 1000.0;
            }

            for (int i = 1; i < boundaries.Length; i++) {
                if (boundaries[i] <= boundaries[i - 1]) {
                    boundaries[i] = Math.Round((boundaries[i - 1] + frameSec) * 1000.0) / 1000.0;
                }
            }
        }

        float[] BuildTimingDurations(double[] boundaries) {
            var durations = new float[boundaries.Length - 1];
            for (int i = 0; i < durations.Length; i++) {
                durations[i] = Math.Max(0.001f, (float)(boundaries[i + 1] - boundaries[i]));
            }
            return durations;
        }

        long[] BuildFramePhonemeMap(float[] timingDurations, int totalFrames) {
            var stau = new long[totalFrames];
            double frameSec = (double)hopSize / sampleRate;
            double time = 0;
            for (int phone = 0; phone < timingDurations.Length; phone++) {
                int startFrame = (int)Math.Round(time / frameSec);
                time += timingDurations[phone];
                int endFrame = Math.Min(totalFrames, (int)Math.Round(time / frameSec));
                for (int frame = startFrame; frame < endFrame; frame++) {
                    stau[frame] = phone + 1;
                }
            }
            long lastPhone = timingDurations.Length;
            for (int frame = 0; frame < totalFrames; frame++) {
                if (stau[frame] == 0) {
                    stau[frame] = lastPhone;
                }
            }
            return stau;
        }

        float[] FitLength(float[] values, int length) {
            if (values.Length == length) {
                return values;
            }
            var fitted = new float[length];
            Array.Copy(values, fitted, Math.Min(values.Length, length));
            return fitted;
        }

        void ClampF0(float[] f0) {
            for (int i = 0; i < f0.Length; i++) {
                if (f0[i] < f0Min) {
                    f0[i] = 0;
                } else if (f0[i] > f0Max) {
                    f0[i] = f0Max;
                }
            }
        }

        void ClampMelspec(float[] melSpectrogram) {
            for (int i = 0; i < melSpectrogram.Length; i++) {
                if (melSpectrogram[i] < melspecMin) {
                    melSpectrogram[i] = melspecMin;
                } else if (melSpectrogram[i] > melspecMax) {
                    melSpectrogram[i] = melspecMax;
                }
            }
        }

        void PostProcessWaveform(float[] waveform) {
            int edge = Math.Min(edgeSilenceSamples, waveform.Length / 2);
            for (int i = 0; i < edge; i++) {
                waveform[i] = 0;
                waveform[waveform.Length - 1 - i] = 0;
            }

            int fadeIn = Math.Min(fadeInSamples, Math.Max(0, waveform.Length - edge));
            for (int i = 0; i < fadeIn; i++) {
                int index = edge + i;
                if (index >= waveform.Length) break;
                float gain = (float)Math.Pow((double)i / fadeInSamples, 2.0);
                waveform[index] *= gain;
            }

            int fadeOut = Math.Min(fadeOutSamples, Math.Max(0, waveform.Length - edge));
            for (int i = 0; i < fadeOut; i++) {
                int index = waveform.Length - edge - 1 - i;
                if (index < 0) break;
                float gain = (float)Math.Pow((double)i / fadeOutSamples, 2.0);
                waveform[index] *= gain;
            }

            for (int i = 0; i < waveform.Length; i++) {
                float value = waveform[i] * wavScale;
                if (value > wavClamp) value = wavClamp;
                if (value < -wavClamp) value = -wavClamp;
                waveform[i] = value;
            }
        }

        public RenderPitchResult LoadRenderedPitch(RenderPhrase phrase) {
            var timing = BuildTimingContext(phrase);
            if (timing.TotalFrames <= 0) {
                return null;
            }

            var f0 = RunPredictedF0(phrase, timing);
            var result = new RenderPitchResult {
                ticks = new float[f0.Length],
                tones = new float[f0.Length],
            };

            for (int frame = 0; frame < f0.Length; frame++) {
                result.ticks[frame] = GetFrameResultTick(phrase, frame);
                int phoneIndex = GetFramePhoneIndex(timing, frame);
                bool voiced = phoneIndex >= 0
                    && timing.PhonemeIds[phoneIndex] != NeutrinoPhoneme.PAU
                    && timing.ScorePitchesHz[phoneIndex] > 0
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

        public override string ToString() => Renderers.NEUTRINO;
    }
}
