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
        const int cacheVersion = 4;
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

            var (phonemeIds, scorePitchesHz, scoreDurations, phonePositions) =
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
            lock (singer.timingSession) {
                using var outputs = singer.timingSession.Run(timingInputs);
                boundaryShifts = outputs.First().AsTensor<float>().ToArray();
            }

            var baseBoundaries = BuildBaseBoundaryTimes(scoreDurations, phonePositions);
            var boundaries = ApplyTimingBoundaryShifts(baseBoundaries, boundaryShifts);
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
            lock (singer.pitchSession) {
                using var outputs = singer.pitchSession.Run(pitchInputs);
                f0 = outputs.First().AsTensor<float>().ToArray();
            }
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
            lock (singer.melspecSession) {
                using var outputs = singer.melspecSession.Run(melspecInputs);
                melSpectrogram = outputs.First().AsTensor<float>().ToArray();
            }
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
            lock (singer.vocoderSession) {
                using var outputs = singer.vocoderSession.Run(vocoderInputs);
                waveform = outputs.First().AsTensor<float>().ToArray();
            }
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

        (long[] phonemeIds, float[] scorePitchesHz, float[] scoreDurations, long[] phonePositions)
            BuildPhonemeSequence(RenderPhrase phrase) {

            var phonemeIds = new List<long>();
            var scorePitchesHz = new List<float>();
            var scoreDurations = new List<float>();
            var phonePositions = new List<long>();
            var lastNotePhoneIndices = new List<int>();

            foreach (var note in phrase.notes) {
                var lyric = note.lyric;
                if (lyric == "+" || lyric == "-") {
                    if (lastNotePhoneIndices.Count > 0) {
                        float addDur = (float)(note.durationMs / 1000.0);
                        foreach (int idx in lastNotePhoneIndices) {
                            scoreDurations[idx] += addDur;
                        }
                    }
                    continue;
                }

                var phoneStrs = NeutrinoPhoneme.KanaToPhonemes(lyric);
                var ids = phoneStrs.Select(p => NeutrinoPhoneme.GetPhonemeId(p)).ToArray();
                if (ids.Length == 1 && ids[0] == NeutrinoPhoneme.PAU
                    && lyric != "R" && lyric != "r" && lyric != "rest") {
                    var parts = lyric.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length > 1) {
                        ids = parts.Select(p => NeutrinoPhoneme.GetPhonemeId(p)).ToArray();
                    } else {
                        ids = new[] { NeutrinoPhoneme.GetPhonemeId(lyric) };
                    }
                }

                lastNotePhoneIndices.Clear();
                float pitchHz = ids.All(id => id == NeutrinoPhoneme.PAU)
                    ? 0
                    : (float)NeutrinoConfig.MidiToFreq(note.tone + note.tuning * 0.01f);
                float durationSec = Math.Max(0.001f, (float)(note.durationMs / 1000.0));

                for (int i = 0; i < ids.Length; i++) {
                    int index = phonemeIds.Count;
                    phonemeIds.Add(ids[i]);
                    scorePitchesHz.Add(ids[i] == NeutrinoPhoneme.PAU ? 0 : pitchHz);
                    scoreDurations.Add(durationSec);
                    phonePositions.Add(i);
                    lastNotePhoneIndices.Add(index);
                }
            }

            return (
                phonemeIds.ToArray(),
                scorePitchesHz.ToArray(),
                scoreDurations.ToArray(),
                phonePositions.ToArray()
            );
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
