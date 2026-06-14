using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using K4os.Hash.xxHash;
using NAudio.Wave;
using OpenUtau.Core.Format;
using OpenUtau.Core.Render;
using OpenUtau.Core.SignalChain;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiNeuralPhraseRenderer : IRenderer {
        public const string RendererId = "HIFI-NEURA";

        // Limit concurrent renders to avoid saturating CPU/memory while still allowing
        // multi-phrase parallelism. The previous global lock serialized all renders.
        static readonly SemaphoreSlim renderGate = new SemaphoreSlim(
            Math.Max(1, Environment.ProcessorCount - 1));

        public USingerType SingerType => USingerType.Classic;
        public bool SupportsRenderPitch => false;

        public bool SupportsExpression(UExpressionDescriptor descriptor) {
            return descriptor.abbr == Format.Ustx.DYN
                || descriptor.abbr == Format.Ustx.PITD
                || descriptor.abbr == Format.Ustx.VEL
                || descriptor.abbr == Format.Ustx.SHFC
                || descriptor.abbr == Format.Ustx.GENC
                || descriptor.abbr == Format.Ustx.BREC
                || descriptor.abbr == Format.Ustx.TENC
                || descriptor.abbr == Format.Ustx.VOIC
                || descriptor.abbr == Format.Ustx.HE
                || string.Equals(descriptor.abbr, HifiGrowlProcessor.CurveAbbr, StringComparison.OrdinalIgnoreCase);
        }

        public RenderResult Layout(RenderPhrase phrase) {
            double phraseStartMs = phrase.positionMs - phrase.leadingMs;
            double phonesEndMs = phrase.phones.Length > 0
                ? phrase.phones.Max(p => p.positionMs + p.durationMs)
                : phrase.positionMs + phrase.durationMs;
            double estimatedLengthMs = Math.Max(
                phrase.durationMs + phrase.leadingMs,
                phonesEndMs - phraseStartMs + HifiF0Builder.FrameMs);
            return new RenderResult {
                leadingMs = phrase.leadingMs,
                positionMs = phrase.positionMs,
                estimatedLengthMs = estimatedLengthMs,
            };
        }

        public Task<RenderResult> Render(RenderPhrase phrase, Progress progress, int trackNo, CancellationTokenSource cancellation, bool isPreRender = false) {
            return Task.Run(() => {
                renderGate.Wait(cancellation.Token);
                try {
                    var result = Layout(phrase);
                    string progressInfo = $"Track {trackNo + 1}: {this} notes={phrase.notes.Length} phones={phrase.phones.Length} duration={result.estimatedLengthMs:F1}ms sr={HifiMelExtractor.SampleRate}";
                    progress.Complete(0, progressInfo);
                    if (cancellation.IsCancellationRequested) {
                        result.samples = Array.Empty<float>();
                        return result;
                    }

                    bool hasModel = HifiOnnxVocoder.TryResolveModelPath(out var modelPath, out var modelDiagnostic);
                    string? wavPath = hasModel
                        ? GetCachePath(phrase.hash, HifiRenderConfig.CacheKey(), modelPath)
                        : null;
                    if (wavPath != null) {
                        phrase.AddCacheFile(wavPath);
                    }
                    if (wavPath != null && File.Exists(wavPath)) {
                        using var waveStream = Wave.OpenFile(wavPath);
                        result.samples = Wave.GetSamples(waveStream.ToSampleProvider().ToMono(1, 0));
                    }
                    if (result.samples == null) {
                        result.samples = RenderInternal(phrase, result, cancellation, wavPath, modelPath, modelDiagnostic);
                    }
                    if (result.samples != null) {
                        Renderers.ApplyDynamics(phrase, result);
                    }
                    progress.Complete(Math.Max(1, phrase.phones.Length), progressInfo);
                    return result;
                } finally {
                    renderGate.Release();
                }
            });
        }

        float[] RenderInternal(RenderPhrase phrase, RenderResult layout, CancellationTokenSource cancellation, string? wavPath, string modelPath, string modelDiagnostic) {
            Log.Information("HifiNeuralPhraseRenderer phrase notes={Notes} phones={Phones} durationMs={Duration:F1} sampleRate={SampleRate}",
                phrase.notes.Length, phrase.phones.Length, layout.estimatedLengthMs, HifiMelExtractor.SampleRate);
            if (phrase.phones.Length == 0) {
                return Array.Empty<float>();
            }
            Log.Information("HifiNeuralPhraseRenderer mel_domain_concat mode=overlap_only phones={Phones}", phrase.phones.Length);
            var featureBuilder = new HifiPhraseFeatureBuilder(HifiRenderConfig.CreateMelEnhancer());
            var features = featureBuilder.Build(phrase, layout);
            string debugKey = $"{phrase.hash:x16}";
            if (HifiRenderConfig.DebugExportEnabled) {
                HifiDebugExporter.Export(debugKey, features);
            }

            if (cancellation.IsCancellationRequested) {
                return Array.Empty<float>();
            }

            if (string.IsNullOrWhiteSpace(modelPath)) {
                var missing = new FileNotFoundException(modelDiagnostic);
                Log.Error(missing, "Hifi ONNX vocoder model is missing. Phrase features were exported, but no wav cache was written.");
                throw new MessageCustomizableException(
                    "HifiNeuralPhraseRenderer requires a PC-NSF-HiFiGAN ONNX model. Phrase features were exported for debugging, but rendering cannot continue.",
                    "HifiNeuralPhraseRenderer requires a PC-NSF-HiFiGAN ONNX model.",
                    missing);
            }

            using var vocoder = new HifiOnnxVocoder(modelPath);
            float[] samples = vocoder.Infer(features);
            // Post-processing chain order:
            // 1. Leveler: per-frame RMS leveling to even out dynamics within the phrase.
            //    Runs first because it operates on the raw vocoder output's local loudness profile.
            // 2. Growl: pitch modulation on highpass band. Runs after the leveler so it operates
            //    on a dynamically balanced signal; its internal RMS matching compensates for any
            //    level change it introduces.
            // 3. Normalizer: global RMS targeting to -17 dBFS with soft-knee limiting. Runs last
            //    among the DSP processors because it sets the final loudness for the OpenUtau
            //    dynamics/mixing stage that follows.
            // 4. Edge guard: cosine fade-in/fade-out at phrase boundaries to suppress clicks.
            //    Runs after all DSP to ensure the fades see the final signal.
            HifiPostVocoderLeveler.LevelInPlace(samples, features, HifiMelExtractor.SampleRate);
            HifiGrowlProcessor.ApplyInPlace(samples, phrase, layout.positionMs - layout.leadingMs, HifiMelExtractor.SampleRate);
            HifiLoudnessNormalizer.NormalizeInPlace(samples, HifiMelExtractor.SampleRate);
            ApplyPhraseEdgeGuard(samples, HifiMelExtractor.SampleRate);
            if (HifiRenderConfig.DebugExportEnabled) {
                HifiClickDiagnostic.Export(debugKey, features, samples);
            }
            if (wavPath != null) {
                SaveCache(wavPath, samples);
            }
            return samples;
        }

        internal static void ApplyPhraseEdgeGuard(float[] samples, int sampleRate) {
            const double edgeProbeMs = 2.0;
            const double fadeInMs = 8.0;
            const double fadeOutMs = 10.0;
            const double absoluteThreshold = 0.0015;
            const double peakRatioThreshold = 0.018;
            if (samples.Length == 0) {
                return;
            }

            double peak = PeakAbs(samples);
            if (peak <= 1e-6) {
                return;
            }
            double threshold = Math.Max(absoluteThreshold, peak * peakRatioThreshold);
            int probeSamples = Math.Clamp((int)Math.Round(sampleRate * edgeProbeMs / 1000.0), 1, samples.Length);
            int maxFade = Math.Max(1, samples.Length / 2);
            int fadeInSamples = Math.Min(maxFade, Math.Max(1, (int)Math.Round(sampleRate * fadeInMs / 1000.0)));
            int fadeOutSamples = Math.Min(maxFade, Math.Max(1, (int)Math.Round(sampleRate * fadeOutMs / 1000.0)));

            if (EdgePeak(samples, 0, probeSamples) > threshold) {
                ApplyFadeIn(samples, fadeInSamples);
            }
            if (EdgePeak(samples, samples.Length - probeSamples, probeSamples) > threshold) {
                ApplyFadeOut(samples, fadeOutSamples);
            }
        }

        static void ApplyFadeIn(float[] samples, int fadeSamples) {
            fadeSamples = Math.Clamp(fadeSamples, 0, samples.Length);
            for (int i = 0; i < fadeSamples; i++) {
                float t = fadeSamples <= 1 ? 1f : i / (float)(fadeSamples - 1);
                float gain = (float)(0.5 - 0.5 * Math.Cos(Math.PI * t));
                samples[i] *= gain;
            }
        }

        static void ApplyFadeOut(float[] samples, int fadeSamples) {
            fadeSamples = Math.Clamp(fadeSamples, 0, samples.Length);
            for (int i = 0; i < fadeSamples; i++) {
                int index = samples.Length - 1 - i;
                float t = fadeSamples <= 1 ? 1f : i / (float)(fadeSamples - 1);
                float gain = (float)(0.5 - 0.5 * Math.Cos(Math.PI * t));
                samples[index] *= gain;
            }
        }

        static double PeakAbs(float[] samples) {
            double peak = 0;
            foreach (float sample in samples) {
                if (float.IsNaN(sample) || float.IsInfinity(sample)) {
                    continue;
                }
                peak = Math.Max(peak, Math.Abs(sample));
            }
            return peak;
        }

        static double EdgePeak(float[] samples, int start, int count) {
            if (count <= 0 || samples.Length == 0) {
                return 0;
            }
            start = Math.Clamp(start, 0, samples.Length);
            int end = Math.Clamp(start + count, start, samples.Length);
            double peak = 0;
            for (int i = start; i < end; i++) {
                float sample = samples[i];
                if (float.IsNaN(sample) || float.IsInfinity(sample)) {
                    continue;
                }
                peak = Math.Max(peak, Math.Abs(sample));
            }
            return peak;
        }

        static void SaveCache(string wavPath, float[] samples) {
            var source = new WaveSource(0, 0, 0, 1);
            source.SetSamples(samples);
            WaveFileWriter.CreateWaveFile16(wavPath, new ExportAdapter(source).ToMono(1, 0));
        }

        static string GetCachePath(ulong phraseHash, string configKey, string modelPath) {
            string configHash = $"{XXH64.DigestOf(Encoding.UTF8.GetBytes(configKey)):x16}";
            string modelHash = HifiOnnxVocoder.ModelCacheKey(modelPath);
            return Path.Join(PathManager.Inst.CachePath, $"hifi-{phraseHash:x16}-cfg{configHash}-{modelHash}.wav");
        }

        public RenderPitchResult LoadRenderedPitch(RenderPhrase phrase) {
            return null!;
        }

        public UExpressionDescriptor[] GetSuggestedExpressions(USinger singer, URenderSettings renderSettings) {
            return new[] {
                new UExpressionDescriptor {
                    name = HifiGrowlProcessor.CurveName,
                    abbr = HifiGrowlProcessor.CurveAbbr,
                    type = UExpressionType.Curve,
                    min = 0,
                    max = 100,
                    defaultValue = 0,
                    isFlag = false,
                },
                new UExpressionDescriptor(
                    "Hifi sustain mode",
                    Format.Ustx.HE,
                    false,
                    (string[])HifiSustainModes.Options.Clone()) {
                    defaultValue = HifiSustainModes.Auto,
                },
            };
        }

        public override string ToString() => RendererId;
    }
}
