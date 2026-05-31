using System;
using System.IO;
using System.Linq;
using System.Reflection;
using OpenUtau.Core.HifiNeural;
using OpenUtau.Core.Render;
using OpenUtau.Core.Util;
using Xunit;

namespace OpenUtau.Core.Test.HifiNeural {
    public class HifiNeuralTest {
        [Fact]
        public void RendererIsRegistered() {
            Assert.Contains(Renderers.HIFI_NEURAL_PHRASE, Renderers.GetSupportedRenderers(Ustx.USingerType.Classic));
            Assert.Contains(Renderers.HIFI_NEURAL_PHRASE, Renderers.getRendererOptions());
            Assert.IsType<HifiNeuralPhraseRenderer>(Renderers.CreateRenderer(Renderers.HIFI_NEURAL_PHRASE));
        }

        [Fact]
        public void LegacyHifiRendererNameMapsToNewRenderer() {
            string original = Preferences.Default.DefaultRenderer;
            try {
                Preferences.Default.DefaultRenderer = Renderers.HIFI_NEURAL_PHRASE_LEGACY;
                Assert.Equal(Renderers.HIFI_NEURAL_PHRASE, Renderers.GetDefaultRenderer(Ustx.USingerType.Classic));
                Assert.IsType<HifiNeuralPhraseRenderer>(Renderers.CreateRenderer(Renderers.HIFI_NEURAL_PHRASE_LEGACY));
            } finally {
                Preferences.Default.DefaultRenderer = original;
            }
        }

        [Fact]
        public void MelExtractorReturnsValidShapeAndValues() {
            var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            var file = Path.Join(dir, "Files", "sine.wav");
            var mel = new HifiMelExtractor().ExtractFromFile(file);

            Assert.Equal(HifiMelExtractor.NMels, mel.GetLength(0));
            Assert.True(mel.GetLength(1) > 0);
            foreach (var value in mel) {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            }
        }

        [Fact]
        public void NormalizeMelEnhanceModeIsStrict() {
            Assert.Equal(HifiRenderConfig.MelEnhanceNone, HifiRenderConfig.NormalizeMelEnhanceMode(null));
            Assert.Equal(HifiRenderConfig.MelEnhanceNone, HifiRenderConfig.NormalizeMelEnhanceMode("unknown"));
            Assert.Equal(HifiRenderConfig.MelEnhanceLight, HifiRenderConfig.NormalizeMelEnhanceMode("LIGHT"));
        }

        [Fact]
        public void CreateMelEnhancerRespectsPreference() {
            string originalMode = Preferences.Default.HifiNeuralMelEnhanceMode;
            try {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE", null);

                Preferences.Default.HifiNeuralMelEnhanceMode = HifiRenderConfig.MelEnhanceNone;
                Assert.IsType<NoOpMelEnhancer>(HifiRenderConfig.CreateMelEnhancer());

                Preferences.Default.HifiNeuralMelEnhanceMode = HifiRenderConfig.MelEnhanceLight;
                Assert.IsType<LightSmoothMelEnhancer>(HifiRenderConfig.CreateMelEnhancer());
            } finally {
                Preferences.Default.HifiNeuralMelEnhanceMode = originalMode;
            }
        }

        [Fact]
        public void LightSmoothEnhancerKeepsShapeAndNoNan() {
            var enhancer = new LightSmoothMelEnhancer();
            var mel = new float[HifiMelExtractor.NMels, 6];
            for (int m = 0; m < mel.GetLength(0); m++) {
                for (int t = 0; t < mel.GetLength(1); t++) {
                    mel[m, t] = (float)(-5.0 + m * 0.001 + t * 0.01);
                }
            }
            var result = enhancer.Enhance(mel, new float[6]);

            Assert.Equal(mel.GetLength(0), result.GetLength(0));
            Assert.Equal(mel.GetLength(1), result.GetLength(1));
            foreach (var value in result) {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            }
        }

        [Fact]
        public void AlignMelFramesProducesExpectedLength() {
            var method = typeof(HifiRoughFeatureBuilder)
                .GetMethod("AlignMelFrames", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);

            var source = new float[2, 4] {
                { 0f, 1f, 2f, 3f },
                { 10f, 11f, 12f, 13f },
            };
            var aligned = (float[,])method!.Invoke(null, new object[] { source, 7 })!;

            Assert.Equal(2, aligned.GetLength(0));
            Assert.Equal(7, aligned.GetLength(1));
            Assert.Equal(source[0, 0], aligned[0, 0], 5);
            Assert.Equal(source[0, 3], aligned[0, 6], 5);
            Assert.Equal(source[1, 0], aligned[1, 0], 5);
            Assert.Equal(source[1, 3], aligned[1, 6], 5);
        }

        [Fact]
        public void TrimInactiveTailRemovesRoughSilenceGap() {
            var method = typeof(HifiRoughFeatureBuilder)
                .GetMethod("TrimInactiveTailFrames", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);

            var mel = new float[2, 20];
            for (int t = 0; t < 20; t++) {
                float value = t < 10 ? -3f : -11f;
                mel[0, t] = value;
                mel[1, t] = value;
            }

            int frames = (int)method!.Invoke(null, new object[] { mel, 0, 20, 2, "a" })!;
            Assert.Equal(16, frames);
        }

        [Fact]
        public void SustainTemplateHandlesTwoFrameOutput() {
            var method = typeof(HifiRoughFeatureBuilder)
                .GetMethod(
                    "WriteSustainTemplateExtension",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] { typeof(float[,]), typeof(int), typeof(int), typeof(float[,]), typeof(int), typeof(int), typeof(bool) },
                    modifiers: null);
            Assert.NotNull(method);

            var source = new float[HifiMelExtractor.NMels, 3];
            for (int m = 0; m < source.GetLength(0); m++) {
                for (int t = 0; t < source.GetLength(1); t++) {
                    source[m, t] = -5f + m * 0.001f + t * 0.02f;
                }
            }
            var output = new float[HifiMelExtractor.NMels, 2];

            bool applied = (bool)method!.Invoke(null, new object[] { source, 0, 3, output, 0, 2, false })!;

            Assert.True(applied);
            foreach (float value in output) {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            }
        }

        [Fact]
        public void VariablePositionMelExtractorReturnsRequestedFrames() {
            var samples = new float[HifiMelExtractor.SampleRate / 8];
            for (int i = 0; i < samples.Length; i++) {
                samples[i] = (float)(0.15 * Math.Sin(2.0 * Math.PI * 220.0 * i / HifiMelExtractor.SampleRate));
            }
            double[] positions = {
                0,
                64.5,
                512,
                1536.25,
                samples.Length - 1,
                samples.Length + 400,
            };

            var mel = new HifiMelExtractor().ExtractAtPositions(samples, positions);

            Assert.Equal(HifiMelExtractor.NMels, mel.GetLength(0));
            Assert.Equal(positions.Length, mel.GetLength(1));
            foreach (var value in mel) {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            }
        }

        [Fact]
        public void LoudnessNormalizerBoostsQuietActiveAudioWithoutClipping() {
            var samples = new float[HifiMelExtractor.SampleRate / 2];
            for (int i = 0; i < samples.Length; i++) {
                samples[i] = (float)(0.02 * Math.Sin(2.0 * Math.PI * 220.0 * i / HifiMelExtractor.SampleRate));
            }

            double gain = HifiLoudnessNormalizer.NormalizeInPlace(samples, HifiMelExtractor.SampleRate);

            Assert.True(gain > 1.0);
            Assert.True(samples.Max(Math.Abs) <= 0.9701f);
        }

        [Fact]
        public void PostVocoderLevelerReducesLocalLoudnessJump() {
            int sampleRate = HifiMelExtractor.SampleRate;
            var samples = new float[sampleRate * 2];
            for (int i = 0; i < samples.Length; i++) {
                double amp = i < sampleRate ? 0.08 : 0.32;
                samples[i] = (float)(amp * Math.Sin(2.0 * Math.PI * 220.0 * i / sampleRate));
            }
            double beforeDb = SegmentRmsDb(samples, sampleRate, sampleRate) - SegmentRmsDb(samples, 0, sampleRate);
            var features = CreateLevelerFeatures(samples.Length, 220f);

            var report = HifiPostVocoderLeveler.LevelInPlace(samples, features, sampleRate);

            double afterDb = SegmentRmsDb(samples, sampleRate, sampleRate) - SegmentRmsDb(samples, 0, sampleRate);
            Assert.True(report.CutFrames > 0);
            Assert.True(afterDb < beforeDb - 2.0, $"expected local jump reduction, before={beforeDb:F2}dB after={afterDb:F2}dB");
            Assert.True(samples.Max(Math.Abs) < 0.33f);
        }

        [Fact]
        public void PostVocoderLevelerAppliesConservativeHighF0Attenuation() {
            int sampleRate = HifiMelExtractor.SampleRate;
            var samples = new float[sampleRate * 2];
            for (int i = 0; i < samples.Length; i++) {
                samples[i] = (float)(0.16 * Math.Sin(2.0 * Math.PI * 220.0 * i / sampleRate));
            }
            int frames = Math.Max(1, (samples.Length + HifiOnnxVocoder.HopSize - 1) / HifiOnnxVocoder.HopSize);
            var f0 = new float[frames];
            for (int i = 0; i < f0.Length; i++) {
                f0[i] = i < f0.Length / 2 ? 220f : 880f;
            }
            var features = CreateLevelerFeatures(samples.Length, f0);

            var report = HifiPostVocoderLeveler.LevelInPlace(samples, features, sampleRate);

            double lowF0Db = SegmentRmsDb(samples, sampleRate / 4, sampleRate / 2);
            double highF0Db = SegmentRmsDb(samples, sampleRate + sampleRate / 4, sampleRate / 2);
            Assert.True(report.MinGainDb < -0.5);
            Assert.True(highF0Db < lowF0Db - 0.4, $"expected high-f0 attenuation, low={lowF0Db:F2}dB high={highF0Db:F2}dB");
        }

        [Fact]
        public void CacheKeyReflectsMinimalConfig() {
            string oldMode = Environment.GetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE");
            string oldDebug = Environment.GetEnvironmentVariable("HIFI_NEURAL_DEBUG_EXPORT");
            try {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE", "none");
                Environment.SetEnvironmentVariable("HIFI_NEURAL_DEBUG_EXPORT", "false");
                string key = HifiRenderConfig.CacheKey();
                Assert.Contains("v28-meldomainconcat-overlapcrossfade-stablepool-consonantfix-f0continuous-postleveler-loud17-vowelalign-cbound", key);
                Assert.Contains("enhnone", key);
                Assert.Contains("dbgFalse", key);
            } finally {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE", oldMode);
                Environment.SetEnvironmentVariable("HIFI_NEURAL_DEBUG_EXPORT", oldDebug);
            }
        }

        static float CrossfadeLogMel(float logOld, float logNew, double u) {
            var method = typeof(HifiMelPhraseAssembler)
                .GetMethod("CrossfadeLogMel", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);
            return (float)method!.Invoke(null, new object[] { logOld, logNew, u })!;
        }

        static double CrossfadeProgress(int overlapOffset, int overlapFrames) {
            var method = typeof(HifiMelPhraseAssembler)
                .GetMethod("CrossfadeProgress", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);
            return (double)method!.Invoke(null, new object[] { overlapOffset, overlapFrames })!;
        }

        [Fact]
        public void OverlapCrossfadeProgressReachesBothEndpoints() {
            Assert.Equal(0.0, CrossfadeProgress(0, 4), 6);
            Assert.Equal(1.0 / 3.0, CrossfadeProgress(1, 4), 6);
            Assert.Equal(2.0 / 3.0, CrossfadeProgress(2, 4), 6);
            Assert.Equal(1.0, CrossfadeProgress(3, 4), 6);
            Assert.Equal(0.5, CrossfadeProgress(0, 1), 6);
        }

        static HifiPhraseFeatures CreateLevelerFeatures(int sampleCount, float f0) {
            int frames = Math.Max(1, (sampleCount + HifiOnnxVocoder.HopSize - 1) / HifiOnnxVocoder.HopSize);
            return CreateLevelerFeatures(sampleCount, Enumerable.Repeat(f0, frames).ToArray());
        }

        static HifiPhraseFeatures CreateLevelerFeatures(int sampleCount, float[] f0) {
            int frames = f0.Length;
            return new HifiPhraseFeatures {
                Mel = new float[HifiMelExtractor.NMels, frames],
                F0 = f0,
                Metadata = new HifiPhraseMetadata {
                    SampleRate = HifiMelExtractor.SampleRate,
                    HopSize = HifiOnnxVocoder.HopSize,
                    FrameMs = 1000.0 * HifiOnnxVocoder.HopSize / HifiMelExtractor.SampleRate,
                    EstimatedLengthMs = 1000.0 * sampleCount / HifiMelExtractor.SampleRate,
                },
            };
        }

        static double SegmentRmsDb(float[] samples, int start, int count) {
            start = Math.Clamp(start, 0, samples.Length);
            int end = Math.Clamp(start + count, start, samples.Length);
            double sum = 0;
            for (int i = start; i < end; i++) {
                sum += samples[i] * samples[i];
            }
            double rms = Math.Sqrt(sum / Math.Max(1, end - start));
            return 20.0 * Math.Log10(Math.Max(1e-7, rms));
        }

        [Fact]
        public void OverlapCrossfadeEndpointsArePure() {
            // u=0 -> fully old, u=1 -> fully new. Endpoints must not be contaminated by the blend.
            float old = -2.0f;
            float @new = -7.0f;
            Assert.Equal(old, CrossfadeLogMel(old, @new, 0.0), 4);
            Assert.Equal(@new, CrossfadeLogMel(old, @new, 1.0), 4);
        }

        [Fact]
        public void OverlapCrossfadePreservesEnergyForEqualVowels() {
            // The core of the VCV/CVVC fix: when two voiced segments of equal energy overlap, the
            // equal-power cross-fade must keep energy flat across the whole overlap instead of
            // dipping (which sounded like the boundary "breaking" under stretch).
            float level = -1.5f;
            double expectedPower = Math.Exp(level);
            for (double u = 0.0; u <= 1.0001; u += 0.1) {
                float mixed = CrossfadeLogMel(level, level, u);
                double power = Math.Exp(mixed);
                Assert.False(float.IsNaN(mixed));
                Assert.False(float.IsInfinity(mixed));
                // cos^2 + sin^2 = 1, so equal-level equal-power blend is exactly the input level.
                Assert.Equal(expectedPower, power, 4);
            }
        }

        [Fact]
        public void OverlapCrossfadeIsMonotonicBetweenLevels() {
            // Blending from a high level down to a low level should move monotonically with u and
            // never overshoot below the lower endpoint or above the higher one (no jump artifacts).
            float high = -1.0f;
            float low = -6.0f;
            float prev = CrossfadeLogMel(high, low, 0.0);
            for (double u = 0.1; u <= 1.0001; u += 0.1) {
                float cur = CrossfadeLogMel(high, low, u);
                Assert.True(cur <= prev + 1e-4, $"non-monotonic at u={u}: {prev} -> {cur}");
                Assert.True(cur >= low - 1e-3 && cur <= high + 1e-3, $"out of range at u={u}: {cur}");
                prev = cur;
            }
        }
    }
}
