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
        public const string RendererId = "HIFI-NEURAL-PHRASE";

        static readonly object lockObj = new object();
        readonly HifiPhraseFeatureBuilder featureBuilder = new HifiPhraseFeatureBuilder();
        readonly IHifiTransitionRefiner refiner = new IdentityHifiTransitionRefiner();

        public USingerType SingerType => USingerType.Classic;
        public bool SupportsRenderPitch => false;

        public bool SupportsExpression(UExpressionDescriptor descriptor) {
            return descriptor.abbr == Format.Ustx.DYN || descriptor.abbr == Format.Ustx.PITD || descriptor.abbr == Format.Ustx.SHFC;
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
                lock (lockObj) {
                    var result = Layout(phrase);
                    string progressInfo = $"Track {trackNo + 1}: {this} notes={phrase.notes.Length} phones={phrase.phones.Length} duration={result.estimatedLengthMs:F1}ms sr={HifiMelExtractor.SampleRate}";
                    progress.Complete(0, progressInfo);
                    if (cancellation.IsCancellationRequested) {
                        result.samples = Array.Empty<float>();
                        return result;
                    }

                    bool hasModel = HifiOnnxVocoder.TryResolveModelPath(out var modelPath, out var modelDiagnostic);
                    string? wavPath = hasModel
                        ? GetCachePath(phrase.hash, HifiNeuralConfig.CacheKey(), modelPath)
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
                }
            });
        }

        float[] RenderInternal(RenderPhrase phrase, RenderResult layout, CancellationTokenSource cancellation, string? wavPath, string modelPath, string modelDiagnostic) {
            Log.Information("HifiNeuralPhraseRenderer phrase notes={Notes} phones={Phones} durationMs={Duration:F1} sampleRate={SampleRate}",
                phrase.notes.Length, phrase.phones.Length, layout.estimatedLengthMs, HifiMelExtractor.SampleRate);
            var features = featureBuilder.Build(phrase, layout);
            var refined = refiner.Refine(features);
            string debugKey = $"{phrase.hash:x16}";
            HifiDebugExporter.Export(debugKey, refined);
            HifiTransitionDatasetExporter.Export(debugKey, refined);

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
            float[] samples = vocoder.Infer(refined);
            if (HifiNeuralConfig.EnableAudioEnvelopeNormalization) {
                HifiAudioEnvelopeNormalizer.Apply(
                    samples,
                    refined,
                    HifiNeuralConfig.AudioEnvelopeMaxCutDb,
                    HifiNeuralConfig.AudioEnvelopeHeadroomDb,
                    HifiNeuralConfig.AudioEnvelopeStrength);
            }
            if (HifiNeuralConfig.EnableInternalClickSuppression) {
                HifiInternalClickSuppressor.Apply(
                    samples,
                    refined.Metadata,
                    HifiMelExtractor.SampleRate,
                    HifiNeuralConfig.InternalClickSuppressorWindowMs,
                    HifiNeuralConfig.InternalClickSuppressorThresholdRatio);
            }
            ApplyClickGuard(samples, HifiMelExtractor.SampleRate, 5);
            HifiClickDiagnostic.Export(debugKey, refined, samples);
            if (wavPath != null) {
                SaveCache(wavPath, samples);
            }
            return samples;
        }

        static void ApplyClickGuard(float[] samples, int sampleRate, double fadeMs) {
            int fadeSamples = Math.Min(samples.Length / 2, Math.Max(1, (int)Math.Round(sampleRate * fadeMs / 1000.0)));
            for (int i = 0; i < fadeSamples; i++) {
                float gain = (float)(0.5 - 0.5 * Math.Cos(Math.PI * (i + 1) / (fadeSamples + 1)));
                samples[i] *= gain;
                int tail = samples.Length - 1 - i;
                if (tail > i) {
                    samples[tail] *= gain;
                }
            }
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
            return Array.Empty<UExpressionDescriptor>();
        }

        public override string ToString() => RendererId;
    }
}
