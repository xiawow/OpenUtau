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
        public void CacheKeyReflectsMinimalConfig() {
            string oldMode = Environment.GetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE");
            string oldDebug = Environment.GetEnvironmentVariable("HIFI_NEURAL_DEBUG_EXPORT");
            try {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE", "none");
                Environment.SetEnvironmentVariable("HIFI_NEURAL_DEBUG_EXPORT", "false");
                string key = HifiRenderConfig.CacheKey();
                Assert.Contains("v12-overlaponly-edgeguard", key);
                Assert.Contains("enhnone", key);
                Assert.Contains("dbgFalse", key);
            } finally {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE", oldMode);
                Environment.SetEnvironmentVariable("HIFI_NEURAL_DEBUG_EXPORT", oldDebug);
            }
        }
    }
}
