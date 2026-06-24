using OpenUtau.Core.Format;
using OpenUtau.Core.HifiNeural;
using OpenUtau.Core.Neutrino;
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
