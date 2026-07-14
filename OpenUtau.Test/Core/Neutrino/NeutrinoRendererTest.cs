using System;
using System.IO;
using System.Linq;
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
        public void SpectralDesignerSupportsBothNeutrinoRenderers() {
            Assert.True(Renderers.SupportsHnSpectralDesigner(
                Renderers.NEUTRINO,
                new NeutrinoRenderer()));
            Assert.True(Renderers.SupportsHnSpectralDesigner(
                Renderers.NEUTRINO_V2,
                new NeutrinoLegacyV2Renderer()));
            Assert.False(Renderers.SupportsHnSpectralDesigner(
                Renderers.CLASSIC,
                Renderers.CreateRenderer(Renderers.CLASSIC)));
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

        [Fact]
        public void InferenceChunksSplitAfterBreathAndAroundPauses() {
            var chunks = NeutrinoInferenceUtil.BuildPhoneChunks(new long[] {
                NeutrinoPhoneme.PAU,
                NeutrinoPhoneme.PAU,
                1,
                NeutrinoPhoneme.BR,
                2,
                NeutrinoPhoneme.PAU,
                4,
            });

            Assert.Equal(5, chunks.Length);
            AssertChunk(chunks[0], 0, 2, false);
            AssertChunk(chunks[1], 2, 2, true);
            AssertChunk(chunks[2], 4, 1, true);
            AssertChunk(chunks[3], 5, 1, false);
            AssertChunk(chunks[4], 6, 1, true);
        }

        [Fact]
        public void ConsecutiveBreathsStayInTheSameActiveChunk() {
            var chunks = NeutrinoInferenceUtil.BuildPhoneChunks(new long[] {
                1,
                NeutrinoPhoneme.BR,
                NeutrinoPhoneme.BR,
                2,
            });

            Assert.Equal(2, chunks.Length);
            AssertChunk(chunks[0], 0, 3, true);
            AssertChunk(chunks[1], 3, 1, true);
        }

        [Fact]
        public void BreathAfterPauseRemainsInTheInactiveChunk() {
            var chunks = NeutrinoInferenceUtil.BuildPhoneChunks(new long[] {
                NeutrinoPhoneme.PAU,
                NeutrinoPhoneme.BR,
                1,
            });

            Assert.Equal(2, chunks.Length);
            AssertChunk(chunks[0], 0, 2, false);
            AssertChunk(chunks[1], 2, 1, true);
        }

        [Fact]
        public void FrameChunksUseGlobalRoundedBoundariesWithoutGaps() {
            var phoneChunks = NeutrinoInferenceUtil.BuildPhoneChunks(new long[] {
                1,
                NeutrinoPhoneme.BR,
                NeutrinoPhoneme.PAU,
                2,
            });
            var frameChunks = NeutrinoInferenceUtil.BuildFrameChunks(
                phoneChunks,
                new[] { 0.0, 0.011, 0.024, 0.032, 0.051 },
                totalFrames: 5,
                frameSeconds: 0.01);

            Assert.Equal(3, frameChunks.Length);
            AssertFrameChunk(frameChunks[0], 0, 2, 0, 2, true);
            AssertFrameChunk(frameChunks[1], 2, 1, 2, 1, false);
            AssertFrameChunk(frameChunks[2], 3, 1, 3, 2, true);
            Assert.Equal(5, frameChunks.Sum(chunk => chunk.FrameCount));
        }

        [Fact]
        public void ChunkedTimingKeepsNextActiveChunkInitialShift() {
            var chunks = NeutrinoInferenceUtil.BuildPhoneChunks(new long[] {
                1,
                NeutrinoPhoneme.PAU,
                2,
            });

            var boundaries = NeutrinoInferenceUtil.BuildTimingBoundaries(
                new[] { 0.3f, 0.2f, 0.4f },
                new long[] { 0, 0, 0 },
                chunks,
                frameSeconds: 0.01,
                chunk => chunk.PhoneStart switch {
                    0 => new[] { 0f, 123f },
                    2 => new[] { -0.05f, 123f },
                    _ => throw new InvalidOperationException(),
                });

            Assert.Equal(0.0, boundaries[0], 3);
            Assert.Equal(0.3, boundaries[1], 3);
            Assert.Equal(0.45, boundaries[2], 3);
            Assert.Equal(0.9, boundaries[3], 3);
        }

        [Fact]
        public void LeadingContextKeepsFirstActiveChunkInitialShift() {
            var chunks = NeutrinoInferenceUtil.BuildPhoneChunks(new long[] { 2, 24 });

            var boundaries = NeutrinoInferenceUtil.BuildTimingBoundaries(
                new[] { 0.5f, 0.5f },
                new long[] { 0, 1 },
                chunks,
                frameSeconds: 0.01,
                chunk => new[] { -0.07f, 0.01f, 123f },
                leadingContextSeconds: 0.5);

            Assert.Equal(-0.07, boundaries[0], 3);
            Assert.Equal(0.01, boundaries[1], 3);
            Assert.Equal(0.5, boundaries[2], 3);
            Assert.Equal(0.08, boundaries[1] - boundaries[0], 3);

            double start = NeutrinoInferenceUtil.NormalizeBoundaryStart(boundaries);
            Assert.Equal(-0.07, start, 3);
            Assert.Equal(0.0, boundaries[0], 3);
            Assert.Equal(0.08, boundaries[1], 3);
            Assert.Equal(0.57, boundaries[2], 3);
        }

        [Fact]
        public void LeadingContextClampsFirstPhoneInsideVirtualPause() {
            var chunks = NeutrinoInferenceUtil.BuildPhoneChunks(new long[] { 2, 24 });

            var boundaries = NeutrinoInferenceUtil.BuildTimingBoundaries(
                new[] { 0.5f, 0.5f },
                new long[] { 0, 1 },
                chunks,
                frameSeconds: 0.01,
                chunk => new[] { -1f, 0.01f, 123f },
                leadingContextSeconds: 0.5);

            Assert.Equal(-0.49, boundaries[0], 3);
            Assert.Equal(0.01, boundaries[1], 3);
        }

        [Fact]
        public void ShortLeadingContextCannotOverlapPreviousPhrase() {
            var chunks = NeutrinoInferenceUtil.BuildPhoneChunks(new long[] { 2, 24 });

            var boundaries = NeutrinoInferenceUtil.BuildTimingBoundaries(
                new[] { 0.5f, 0.5f },
                new long[] { 0, 1 },
                chunks,
                frameSeconds: 0.01,
                chunk => new[] { -0.07f, 0.01f, 123f },
                leadingContextSeconds: 0.03);

            Assert.Equal(-0.02, boundaries[0], 3);
            Assert.Equal(0.01, boundaries[1], 3);
        }

        [Fact]
        public void ZeroLeadingContextKeepsFirstPhoneAtScoreStart() {
            var chunks = NeutrinoInferenceUtil.BuildPhoneChunks(new long[] { 2, 24 });

            var boundaries = NeutrinoInferenceUtil.BuildTimingBoundaries(
                new[] { 0.5f, 0.5f },
                new long[] { 0, 1 },
                chunks,
                frameSeconds: 0.01,
                chunk => new[] { -0.07f, 0.01f, 123f },
                leadingContextSeconds: 0);

            Assert.Equal(0, boundaries[0], 3);
            Assert.Equal(0.01, boundaries[1], 3);
        }

        [Fact]
        public void ManualFirstBoundaryCanExtendConsonantIntoLeadingContext() {
            var boundaries = new[] { -0.057, 0.003, 0.5 };

            NeutrinoRenderer.ApplyManualBoundaryOverrides(
                boundaries,
                new double?[] { -0.1, null, null },
                leadingContextSeconds: 0.5);

            Assert.Equal(-0.1, boundaries[0], 3);
            Assert.Equal(0.003, boundaries[1], 3);
            Assert.Equal(0.103, boundaries[1] - boundaries[0], 3);
        }

        [Fact]
        public void ManualFirstBoundaryCannotExceedLeadingContext() {
            var boundaries = new[] { -0.02, 0.01, 0.5 };

            NeutrinoRenderer.ApplyManualBoundaryOverrides(
                boundaries,
                new double?[] { -0.1, null, null },
                leadingContextSeconds: 0.03);

            Assert.Equal(-0.02, boundaries[0], 3);
            Assert.Equal(0.01, boundaries[1], 3);
        }

        [Fact]
        public void ChunkedTimingDoesNotRepeatOneNoteDuration() {
            var chunks = NeutrinoInferenceUtil.BuildPhoneChunks(new long[] {
                1,
                NeutrinoPhoneme.PAU,
                2,
            });

            var boundaries = NeutrinoInferenceUtil.BuildTimingBoundaries(
                new[] { 0.5f, 0.5f, 0.5f },
                new long[] { 0, 1, 2 },
                chunks,
                frameSeconds: 0.01,
                chunk => new float[chunk.PhoneCount + 1]);

            Assert.Equal(0.5, boundaries[^1], 3);
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
        public void FrameAwareSpectralProfileOnlyShapesAssignedFrames() {
            const int frameCount = 32;
            const int hop = HifiOnnxVocoder.HopSize;
            var profile = new HifiHnSpectralProfile {
                BalanceDb = Enumerable.Repeat(HifiHnSpectralProfile.MaxBalanceDb, 5).ToArray(),
            };
            var profiles = new HifiHnSpectralProfile?[frameCount];
            for (int frame = 12; frame < 24; frame++) {
                profiles[frame] = profile;
            }
            var profileTrack = new HifiHnSpectralProfileTrack(profiles);
            var harmonic = new float[frameCount * hop];
            for (int i = 0; i < harmonic.Length; i++) {
                harmonic[i] = (float)(0.2 * Math.Sin(2 * Math.PI * 220 * i / 44100.0));
            }
            var noise = new float[harmonic.Length];

            var shaped = HifiHnSpectralProcessor.Process(
                harmonic,
                harmonic,
                noise,
                profileTrack);

            double neutralBefore = SegmentRms(shaped, 3 * hop, 8 * hop);
            double active = SegmentRms(shaped, 15 * hop, 21 * hop);
            double neutralAfter = SegmentRms(shaped, 27 * hop, 31 * hop);
            double source = SegmentRms(harmonic, 3 * hop, 8 * hop);
            Assert.InRange(neutralBefore / source, 0.98, 1.02);
            Assert.InRange(neutralAfter / source, 0.98, 1.02);
            Assert.True(active > neutralBefore * 2.0);
            Assert.All(shaped, sample => Assert.True(float.IsFinite(sample)));
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

        static void AssertChunk(
            NeutrinoPhoneChunk chunk,
            int phoneStart,
            int phoneCount,
            bool isActive) {

            Assert.Equal(phoneStart, chunk.PhoneStart);
            Assert.Equal(phoneCount, chunk.PhoneCount);
            Assert.Equal(isActive, chunk.IsActive);
        }

        static void AssertFrameChunk(
            NeutrinoFrameChunk chunk,
            int phoneStart,
            int phoneCount,
            int frameStart,
            int frameCount,
            bool isActive) {

            Assert.Equal(phoneStart, chunk.PhoneStart);
            Assert.Equal(phoneCount, chunk.PhoneCount);
            Assert.Equal(frameStart, chunk.FrameStart);
            Assert.Equal(frameCount, chunk.FrameCount);
            Assert.Equal(isActive, chunk.IsActive);
        }

        static double SegmentRms(float[] samples, int start, int end) {
            double sum = 0;
            int count = Math.Max(1, end - start);
            for (int i = start; i < end; i++) {
                sum += samples[i] * samples[i];
            }
            return Math.Sqrt(sum / count);
        }
    }
}
