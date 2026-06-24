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
    }
}
