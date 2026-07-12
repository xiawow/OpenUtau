using System;
using System.IO;
using System.Text;
using OpenUtau.Classic;
using OpenUtau.Core.Format;
using OpenUtau.Core.HifiNeural;
using OpenUtau.Core.Neutrino;
using OpenUtau.Core.Render;
using OpenUtau.Core.Ustx;
using Xunit;
using FormatUstx = OpenUtau.Core.Format.Ustx;

namespace OpenUtau.Core.Test.Neutrino {
    public class NeutrinoRendererTest {
        [Theory]
        [InlineData(FormatUstx.TENC)]
        [InlineData(FormatUstx.BREC)]
        [InlineData(FormatUstx.VOIC)]
        [InlineData(FormatUstx.GENC)]
        public void SupportsHnsepExpressions(string abbr) {
            var descriptor = new UExpressionDescriptor(abbr, abbr, -100, 100, 0) {
                type = UExpressionType.Curve,
            };

            Assert.True(new NeutrinoRenderer().SupportsExpression(descriptor));
            Assert.True(new NeutrinoLegacyV2Renderer().SupportsExpression(descriptor));
        }

        [Fact]
        public void RegistersLegacyV2RendererSeparately() {
            Assert.Contains(Renderers.NEUTRINO, Renderers.GetSupportedRenderers(USingerType.Neutrino));
            Assert.Contains(Renderers.NEUTRINO_V2, Renderers.GetSupportedRenderers(USingerType.Neutrino));
            Assert.IsType<NeutrinoRenderer>(Renderers.CreateRenderer(Renderers.NEUTRINO));
            Assert.IsType<NeutrinoLegacyV2Renderer>(Renderers.CreateRenderer(Renderers.NEUTRINO_V2));
        }

        [Fact]
        public void LegacyV2SingerDefaultsToLegacyV2Renderer() {
            string dir = Path.Combine(Path.GetTempPath(), $"neutrino-v2-renderer-{Guid.NewGuid():N}");
            try {
                string modelDir = Path.Combine(dir, "model", "MERROW");
                Directory.CreateDirectory(modelDir);
                string characterPath = Path.Combine(dir, "character.txt");
                File.WriteAllText(characterPath, "name=Legacy NEUTRINO\n", Encoding.UTF8);
                foreach (string model in new[] { "t.bin", "e.bin", "ds.bin", "vs.bin" }) {
                    File.WriteAllBytes(Path.Combine(modelDir, model), new byte[] { 0 });
                }
                var singer = new NeutrinoSinger(new Voicebank {
                    BasePath = Path.GetDirectoryName(dir),
                    File = characterPath,
                    Name = "Legacy NEUTRINO",
                    Id = "Legacy NEUTRINO",
                    SingerType = USingerType.Neutrino,
                });

                Assert.Equal(Renderers.NEUTRINO_V2, Renderers.GetDefaultRenderer(singer));
            } finally {
                Directory.Delete(dir, recursive: true);
            }
        }

        [Fact]
        public void RenderPhoneKeepsPanelConsonantN() {
            Assert.Equal(new[] { "N" }, NeutrinoPhoneme.KanaToPhonemes("n"));
            Assert.Equal(new[] { "n" }, NeutrinoPhoneme.RenderPhoneToPhonemes("n"));
            Assert.Equal(new[] { "n", "o" }, NeutrinoPhoneme.RenderPhoneToPhonemes("no"));
        }

        [Fact]
        public void FrameMapAssignsUncoveredFramesToFinalPhone() {
            Assert.Equal(
                new long[] { 2 },
                NeutrinoRenderer.BuildFramePhonemeMap(new[] { 0.001f, 0.001f }, 1));
            Assert.Equal(
                new long[] { 1, 2 },
                NeutrinoRenderer.BuildFramePhonemeMap(new[] { 0.011f, 0.001f }, 2));
        }

        [Theory]
        [InlineData("+")]
        [InlineData("+~")]
        [InlineData("+*")]
        [InlineData("+anything")]
        public void PlusPrefixedLyricsMatchOpenUtauExtensionSemantics(string lyric) {
            Assert.True(NeutrinoInferenceUtil.IsExtensionLyric(lyric));
        }

        [Fact]
        public void LegacyMinusExtensionRemainsSupported() {
            Assert.True(NeutrinoInferenceUtil.IsExtensionLyric("-"));
        }

        [Theory]
        [InlineData("")]
        [InlineData("a")]
        [InlineData("~+")]
        public void NonExtensionLyricsRemainIndependent(string lyric) {
            Assert.False(NeutrinoInferenceUtil.IsExtensionLyric(lyric));
        }

        [Fact]
        public void FixedShapeModelOutputsRejectLengthMismatch() {
            var output = new[] { 0.1f, 0.2f };
            Assert.Same(output, NeutrinoInferenceUtil.RequireLength(output, 2, "test output"));

            var error = Assert.Throws<InvalidDataException>(
                () => NeutrinoInferenceUtil.RequireLength(output, 3, "test output"));
            Assert.Equal("test output length mismatch: actual 2, expected 3.", error.Message);
        }

        [Fact]
        public void TimingModelReturnsOneMoreBoundaryThanPhonemes() {
            var boundaries = new[] { 0f, 0.1f, 0.2f };
            Assert.Same(
                boundaries,
                NeutrinoInferenceUtil.RequireTimingBoundaryLength(boundaries, 2, "timing output"));

            var error = Assert.Throws<InvalidDataException>(
                () => NeutrinoInferenceUtil.RequireTimingBoundaryLength(
                    new[] { 0f, 0.1f }, 2, "timing output"));
            Assert.Equal("timing output length mismatch: actual 2, expected 3.", error.Message);
        }

        [Fact]
        public void NeutralHnsepParametersLeaveWaveformUntouched() {
            var waveform = new[] { 0.1f, -0.2f, 0.3f };
            var parameters = HifiFrameParameterTrack.Constant(
                new HifiFrameParameterAverages(0, 0, 0, 100));

            var result = HifiHnsepSourceProcessor.ApplyGeneratedWaveform(
                waveform,
                parameters,
                separationCacheKey: null,
                out var report);

            Assert.Same(waveform, result);
            Assert.False(report.Requested);
            Assert.False(report.Applied);
        }

        [Fact]
        public void GeneratedWaveformGencUsesPitchGuidedHarmonicPreparation() {
            const int sampleRate = 44100;
            var harmonic = new float[sampleRate / 2];
            for (int i = 0; i < harmonic.Length; i++) {
                double time = i / (double)sampleRate;
                harmonic[i] = (float)(
                    0.45 * System.Math.Sin(2 * System.Math.PI * 220 * time)
                    + 0.25 * System.Math.Sin(2 * System.Math.PI * 2640 * time));
            }
            var parameters = HifiFrameParameterTrack.Constant(
                new HifiFrameParameterAverages(100, 0, 0, 100), frameCount: 40);

            var shifted = HifiHnsepSourceProcessor.PrepareGeneratedHarmonicForRemix(
                harmonic,
                parameters,
                (sample, sampleCount) => 220.0);

            Assert.Equal(harmonic.Length, shifted.Length);
            Assert.All(shifted, sample => Assert.True(float.IsFinite(sample)));
            Assert.NotEqual(harmonic, shifted);
        }
    }
}
