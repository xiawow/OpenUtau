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
using OpenUtau.Core.Render;
using OpenUtau.Core.SignalChain;
using OpenUtau.Core.Ustx;
using Serilog;

namespace OpenUtau.Core.Neutrino {
    public class NeutrinoRenderer : IRenderer {
        public const int headTicks = 480;
        public const int tailTicks = 480;

        const int sampleRate = 48000;
        const int hopSize = 480;
        const int numMelBins = 100;
        const int cacheVersion = 6;
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
        };

        static readonly object lockObj = new object();

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
                    ulong hash = phrase.hash;
                    var wavPath = Path.Join(PathManager.Inst.CachePath,
                        $"neutrino-v{cacheVersion}-{hash:x16}.wav");
                    phrase.AddCacheFile(wavPath);

                    if (File.Exists(wavPath)) {
                        try {
                            using (var waveStream = Wave.OpenFile(wavPath)) {
                                result.samples = Wave.GetSamples(
                                    waveStream.ToSampleProvider().ToMono(1, 0));
                            }
                        } catch (Exception e) {
                            Log.Error(e, "Failed to read NEUTRINO cache, re-rendering");
                        }
                    }

                    if (result.samples == null) {
                        result.samples = InvokeNeutrino(phrase, cancellation);
                        if (result.samples != null) {
                            var source = new WaveSource(0, 0, 0, 1);
                            source.SetSamples(result.samples);
                            WaveFileWriter.CreateWaveFile16(wavPath,
                                new ExportAdapter(source).ToMono(1, 0));
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

        float[] InvokeNeutrino(RenderPhrase phrase, CancellationTokenSource cancellation) {
            var singer = phrase.singer as NeutrinoSinger;
            singer.EnsureSessions();

            var (phonemeIds, scorePitchesHz, scoreDurations, phonePositions, manualBoundaries) =
                BuildPhonemeSequence(phrase);

            if (cancellation.IsCancellationRequested) return null;
            int numPhones = phonemeIds.Length;
            if (numPhones == 0) {
                return Array.Empty<float>();
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

            float[] boundaryShifts;
            boundaryShifts = singer.RunTiming(timingInputs);

            var baseBoundaries = BuildBaseBoundaryTimes(scoreDurations, phonePositions);
            var boundaries = ApplyTimingBoundaryShifts(baseBoundaries, boundaryShifts);
            ApplyManualBoundaryOverrides(boundaries, manualBoundaries);
            var timingDurations = BuildTimingDurations(boundaries);
            int totalFrames = Math.Max(1, (int)Math.Round(boundaries[^1] * sampleRate / hopSize));
            var stauData = BuildFramePhonemeMap(timingDurations, totalFrames);

            if (cancellation.IsCancellationRequested) return null;

            var pitchInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("electron",
                    new DenseTensor<long>(phonemeIds, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("muon",
                    new DenseTensor<float>(timingDurations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("tau",
                    new DenseTensor<float>(scorePitchesHz, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("selectron",
                    new DenseTensor<float>(scoreDurations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("smuon",
                    new DenseTensor<long>(phonePositions, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("stau",
                    new DenseTensor<long>(stauData, new[] { 1, totalFrames })),
            };

            float[] f0;
            f0 = singer.RunPitch(pitchInputs);
            ClampF0(f0);
            ApplyPitchCurve(phrase, f0);
            ClampF0(f0);
            f0 = FitLength(f0, totalFrames);

            if (cancellation.IsCancellationRequested) return null;

            var melspecInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("electron",
                    new DenseTensor<long>(phonemeIds, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("muon",
                    new DenseTensor<float>(timingDurations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("tau",
                    new DenseTensor<float>(scorePitchesHz, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("selectron",
                    new DenseTensor<float>(scoreDurations, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("smuon",
                    new DenseTensor<long>(phonePositions, new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("stau",
                    new DenseTensor<long>(stauData, new[] { 1, totalFrames })),
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

            if (sampleRate != 44100) {
                var signal = new NWaves.Signals.DiscreteSignal(sampleRate, result);
                signal = NWaves.Operations.Operation.Resample(signal, 44100);
                result = signal.Samples;
            }

            Wave.CorrectSampleScale(result);
            return result;
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

        void ApplyPitchCurve(RenderPhrase phrase, float[] f0) {
            if (phrase.pitches == null || phrase.pitches.Length == 0) {
                return;
            }
            var layout = Layout(phrase);
            double startMs = layout.positionMs - layout.leadingMs;
            double frameDurationMs = 1000.0 * hopSize / sampleRate;
            for (int frame = 0; frame < f0.Length; frame++) {
                if (f0[frame] <= 0) {
                    continue;
                }
                double posMs = startMs + frame * frameDurationMs;
                int ticks = phrase.timeAxis.MsPosToTickPos(posMs) - (phrase.position - phrase.leading);
                int index = Math.Max(0, Math.Min((int)((double)ticks / 5), phrase.pitches.Length - 1));
                double pitchDev = 0;
                if (index < phrase.pitchesBeforeDeviation.Length) {
                    pitchDev = (phrase.pitches[index] - phrase.pitchesBeforeDeviation[index]) * 0.01;
                }
                if (Math.Abs(pitchDev) > 0.01) {
                    f0[frame] = (float)(f0[frame] * Math.Pow(2, pitchDev / 12.0));
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
            return null;
        }

        public UExpressionDescriptor[] GetSuggestedExpressions(
            USinger singer, URenderSettings renderSettings) {
            return Array.Empty<UExpressionDescriptor>();
        }

        public override string ToString() => Renderers.NEUTRINO;
    }
}
