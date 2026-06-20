using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.Serialization;
using OpenUtau.Core.HifiNeural;
using OpenUtau.Core.Render;
using OpenUtau.Core.Util;
using OpenUtau.Core.Ustx;
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
        public void OtoCacheHashChangesForTimingEdits() {
            var oto = new UOto {
                Offset = 10,
                Consonant = 50,
                Cutoff = -80,
                Preutter = 70,
                Overlap = 25,
            };

            ulong baseline = RenderPhone.OtoCacheHash(oto);

            oto.Offset = 11;
            Assert.NotEqual(baseline, RenderPhone.OtoCacheHash(oto));

            oto.Offset = 10;
            oto.Consonant = 51;
            Assert.NotEqual(baseline, RenderPhone.OtoCacheHash(oto));

            oto.Consonant = 50;
            oto.Cutoff = -81;
            Assert.NotEqual(baseline, RenderPhone.OtoCacheHash(oto));

            oto.Cutoff = -80;
            oto.Preutter = 71;
            Assert.NotEqual(baseline, RenderPhone.OtoCacheHash(oto));

            oto.Preutter = 70;
            oto.Overlap = 26;
            Assert.NotEqual(baseline, RenderPhone.OtoCacheHash(oto));
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
        public void MelExtractorKeyShiftKeepsShapeAndChangesSpectrum() {
            var samples = new float[HifiMelExtractor.SampleRate / 5];
            for (int i = 0; i < samples.Length; i++) {
                double t = i / (double)HifiMelExtractor.SampleRate;
                samples[i] = (float)(
                    0.18 * Math.Sin(2.0 * Math.PI * 220.0 * t)
                    + 0.08 * Math.Sin(2.0 * Math.PI * 880.0 * t)
                    + 0.03 * Math.Sin(2.0 * Math.PI * 2400.0 * t));
            }

            var neutral = new HifiMelExtractor().Extract(samples, 0);
            var shifted = new HifiMelExtractor().Extract(samples, 6);
            var shiftedDown = new HifiMelExtractor().Extract(samples, -6);

            Assert.Equal(neutral.GetLength(0), shifted.GetLength(0));
            Assert.Equal(neutral.GetLength(1), shifted.GetLength(1));
            Assert.True(Rms(Difference(Flatten(neutral), Flatten(shifted))) > 1e-3);
            Assert.True(MelCentroid(shifted) > MelCentroid(neutral), "positive key shift should move spectral envelope upward");
            Assert.True(MelCentroid(shiftedDown) < MelCentroid(neutral), "negative key shift should move spectral envelope downward");
            foreach (var value in shifted) {
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
            var method = typeof(HifiPhraseFeatureBuilder)
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
            var method = typeof(HifiPhraseFeatureBuilder)
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
            var method = typeof(HifiPhraseFeatureBuilder)
                .GetMethod(
                    "WriteSustainTemplateExtension",
                    BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);

            var source = new float[HifiMelExtractor.NMels, 3];
            for (int m = 0; m < source.GetLength(0); m++) {
                for (int t = 0; t < source.GetLength(1); t++) {
                    source[m, t] = -5f + m * 0.001f + t * 0.02f;
                }
            }
            var output = new float[HifiMelExtractor.NMels, 2];

            bool applied = (bool)method!.Invoke(null, new object[] { source, 0, 3, output, 0, 2, false, null, null, 0, 0, 0d })!;

            Assert.True(applied);
            foreach (float value in output) {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            }
        }

        [Fact]
        public void SustainTextureTrajectoryAnchorsEdges() {
            var method = typeof(HifiPhraseFeatureBuilder)
                .GetMethod(
                    "WriteSustainTemplateExtension",
                    BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);

            var source = new float[HifiMelExtractor.NMels, 10];
            for (int m = 0; m < source.GetLength(0); m++) {
                for (int t = 0; t < source.GetLength(1); t++) {
                    source[m, t] = -6f + m * 0.001f + t * 0.04f;
                }
            }
            var output = new float[HifiMelExtractor.NMels, 40];

            bool applied = (bool)method!.Invoke(null, new object[] { source, 0, 10, output, 0, 40, false, null, null, 0, 0, 0d })!;

            Assert.True(applied);
            Assert.Equal(source[0, 0], output[0, 0], 5);
            Assert.Equal(source[0, 9], output[0, 39], 5);
            foreach (float value in output) {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            }
        }

        [Fact]
        public void WaveformSustainTextureWritesValidMelAndAnchorsEdges() {
            var method = typeof(HifiPhraseFeatureBuilder)
                .GetMethod(
                    "TryWriteWaveformSustainTexture",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] {
                        typeof(float[,]),
                        typeof(float[]),
                        typeof(int),
                        typeof(int),
                        typeof(int),
                        typeof(int),
                        typeof(float[,]),
                        typeof(int),
                        typeof(int),
                        typeof(int),
                        typeof(int),
                        typeof(double),
                        typeof(bool),
                        typeof(float[]),
                        typeof(int),
                        typeof(int),
                        typeof(double),
                    },
                    modifiers: null);
            Assert.NotNull(method);

            var sourceSamples = new float[HifiMelExtractor.SampleRate / 4];
            for (int i = 0; i < sourceSamples.Length; i++) {
                double carrier = Math.Sin(2.0 * Math.PI * 220.0 * i / HifiMelExtractor.SampleRate);
                double slow = 0.8 + 0.15 * Math.Sin(2.0 * Math.PI * 5.0 * i / HifiMelExtractor.SampleRate);
                sourceSamples[i] = (float)(0.12 * slow * carrier);
            }
            var sourceMel = new HifiMelExtractor().Extract(sourceSamples);
            int sourceFrames = sourceMel.GetLength(1);
            int stableStart = Math.Max(1, sourceFrames / 5);
            int stableFrames = Math.Max(4, sourceFrames - stableStart * 2);
            int outputFrames = sourceFrames * 3;
            var output = new float[HifiMelExtractor.NMels, outputFrames];
            var targetF0 = Enumerable.Repeat(880f, outputFrames).ToArray();

            bool applied = (bool)method!.Invoke(null, new object[] {
                sourceMel,
                sourceSamples,
                0,
                sourceFrames,
                stableStart,
                stableFrames,
                output,
                0,
                outputFrames,
                3,
                6,
                3.0,
                false,
                targetF0,
                0,
                60,
                0d,
            })!;

            Assert.True(applied);
            Assert.Equal(sourceMel[0, 0], output[0, 0], 5);
            Assert.Equal(sourceMel[0, sourceFrames - 1], output[0, outputFrames - 1], 5);
            foreach (float value in output) {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            }
        }

        [Fact]
        public void WaveformSustainEnergyDeltaDoesNotFlattenNormalVariation() {
            var method = typeof(HifiPhraseFeatureBuilder)
                .GetMethod("ResolveWaveformSustainEnergyDelta", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);

            double delta = (double)method!.Invoke(null, new object[] {
                -4.0, // targetEnergy
                -4.22, // currentEnergy, normal local variation
                0.03, // global offset
            })!;

            Assert.Equal(0.03, delta, 6);
        }

        [Fact]
        public void SustainTextureStepAllowsNaturalHopRate() {
            var method = typeof(HifiPhraseFeatureBuilder)
                .GetMethod("LimitTextureIndexStep", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);

            double limited = (double)method!.Invoke(null, new object[] {
                10.0,
                14.0,
                3.0,
            })!;

            Assert.True(limited > 13.5, $"expected near-natural texture step, got {limited}");
        }

        [Fact]
        public void LongSustainParameterMapFollowsDirectContour() {
            var method = typeof(HifiPhraseFeatureBuilder)
                .GetMethod("WriteSustainFrameMap", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);

            var map = new double[24];
            method!.Invoke(null, new object[] {
                map,
                0,
                map.Length,
                5,
                20,
            });

            for (int t = 0; t < map.Length; t++) {
                double expected = 5 + t * 19.0 / (map.Length - 1);
                Assert.Equal(expected, map[t], 6);
                if (t > 0) {
                    Assert.True(map[t] >= map[t - 1], "long sustain parameter map should not follow residual texture wander");
                }
            }
        }

        [Fact]
        public void SustainTextureBodyRemovesFrameMeanEnergy() {
            var meanMethod = typeof(HifiPhraseFeatureBuilder)
                .GetMethod("TextureResidualMean", BindingFlags.NonPublic | BindingFlags.Static);
            var composeMethod = typeof(HifiPhraseFeatureBuilder)
                .GetMethod("ComposeSustainTextureBodyValue", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(meanMethod);
            Assert.NotNull(composeMethod);

            float[] texture = { -3.9f, -3.8f, -3.7f, -3.6f };
            float[] envelope = { -4.0f, -3.9f, -3.8f, -3.7f };
            double residualMean = (double)meanMethod!.Invoke(null, new object[] { texture, envelope })!;
            var values = new double[texture.Length];
            for (int i = 0; i < texture.Length; i++) {
                values[i] = (float)composeMethod!.Invoke(null, new object[] {
                    -4.0f,
                    -4.0f,
                    texture[i],
                    envelope[i],
                    residualMean,
                    i,
                    texture.Length,
                    1.0,
                    0.0,
                })!;
            }

            double meanDelta = values.Average() - -4.0;
            Assert.True(Math.Abs(meanDelta) < 0.01, $"texture body should not change frame mean, delta={meanDelta:F6}");
        }

        [Fact]
        public void VcvPreferredOnsetUsesConsonantBoundary() {
            var method = typeof(HifiPhraseFeatureBuilder)
                .GetMethod(
                    "ResolveVowelSections",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] {
                        typeof(int),
                        typeof(int),
                        typeof(int).MakeByRefType(),
                        typeof(int).MakeByRefType(),
                        typeof(int).MakeByRefType(),
                    },
                    modifiers: null);
            Assert.NotNull(method);

            object[] args = { 24, 10, 0, 0, 0 };
            method!.Invoke(null, args);

            Assert.Equal(10, (int)args[2]);
            Assert.True((int)args[4] > 0);
            Assert.Equal(24, (int)args[2] + (int)args[3] + (int)args[4]);
        }

        [Fact]
        public void VcvShortTargetPreservesOnsetBeforeCompressingSustain() {
            var method = typeof(HifiPhraseFeatureBuilder)
                .GetMethod(
                    "AllocateVowelTargetSections",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] { typeof(int), typeof(int), typeof(int), typeof(int) },
                    modifiers: null);
            Assert.NotNull(method);

            var result = (ValueTuple<int, int, int>)method!.Invoke(null, new object[] { 24, 24, 4, 10 })!;

            Assert.Equal(6, result.Item1);
            Assert.Equal(3, result.Item2);
            Assert.Equal(1, result.Item3);
            Assert.Equal(10, result.Item1 + result.Item2 + result.Item3);
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
        public void F0FallbackChoosesNearestNoteInsteadOfFirstNote() {
            var method = typeof(HifiF0Builder)
                .GetMethod("NearestNoteAt", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);
            var notes = new[] {
                CreateRenderNoteForFallback(60, 0, 500),
                CreateRenderNoteForFallback(67, 1000, 1500),
            };

            var note = (RenderNote?)method!.Invoke(null, new object[] { notes, 1200.0 });

            Assert.NotNull(note);
            Assert.Equal(67, note!.tone);
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
        public void PostVocoderLevelerDoesNotCutEqualLoudnessSolelyByF0() {
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
            Assert.True(Math.Abs(report.MinGainDb) < 0.05);
            Assert.True(Math.Abs(highF0Db - lowF0Db) < 0.15, $"expected no f0-only attenuation, low={lowF0Db:F2}dB high={highF0Db:F2}dB");
        }

        [Fact]
        public void CacheKeyReflectsMinimalConfig() {
            string oldMode = Environment.GetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE");
            string oldDebug = Environment.GetEnvironmentVariable("HIFI_NEURAL_DEBUG_EXPORT");
            try {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE", "none");
                Environment.SetEnvironmentVariable("HIFI_NEURAL_DEBUG_EXPORT", "false");
                string key = HifiRenderConfig.CacheKey();
                Assert.Contains("v62-soft-consonant-boost-phoneplan-softleadskip-vel-support-finaledge-audibletargetfixed-sourceotopreutter-sustaintexturebody-conditionaledge-restgapend-finaltail-vowelbudget-anchorlead-targetfixed-envelopeend-directparammap-meldomainconcat-waveformsustain-naturalrate-f0fallback-postleveler-loud17-grocv1-genc-hnsepslice-rms-sourceparams-tencremixfix-nonlinearparammap-", key);
                Assert.Contains("enhnone", key);
                Assert.Contains("dbgFalse", key);
            } finally {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE", oldMode);
                Environment.SetEnvironmentVariable("HIFI_NEURAL_DEBUG_EXPORT", oldDebug);
            }
        }

        [Fact]
        public void HnsepDiskCacheRoundTripsHarmonicSourceFeatures() {
            string dir = Path.Combine(Path.GetTempPath(), "hifi-hnsep-test-" + Guid.NewGuid().ToString("N"));
            string path = Path.Combine(dir, "cache.f32");
            try {
                var result = new HifiHnsepResult {
                    Harmonic = new[] { 0.1f, -0.2f, 0.3f, float.NaN },
                };

                HifiHnsepDiskCache.TrySave(path, result);

                Assert.True(HifiHnsepDiskCache.TryLoad(path, 4, out var loaded));
                Assert.NotNull(loaded);
                Assert.Equal(4, loaded!.Harmonic.Length);
                Assert.Equal(0.1f, loaded.Harmonic[0], 6);
                Assert.Equal(-0.2f, loaded.Harmonic[1], 6);
                Assert.Equal(0.3f, loaded.Harmonic[2], 6);
                Assert.Equal(0f, loaded.Harmonic[3]);
                Assert.False(HifiHnsepDiskCache.TryLoad(path, 3, out _));
            } finally {
                if (Directory.Exists(dir)) {
                    Directory.Delete(dir, recursive: true);
                }
            }
        }

        [Fact]
        public void HifiDebugDumpPreservesPhoneParameterMetadata() {
            string dir = Path.Combine(Path.GetTempPath(), "hifi-debug-test-" + Guid.NewGuid().ToString("N"));
            try {
                var features = new HifiPhraseFeatures {
                    Mel = new float[HifiMelExtractor.NMels, 2],
                    F0 = new[] { 220f, 221f },
                    Metadata = new HifiPhraseMetadata {
                        SampleRate = HifiMelExtractor.SampleRate,
                        HopSize = HifiOnnxVocoder.HopSize,
                        FrameMs = HifiF0Builder.FrameMs,
                        Phones = {
                            new HifiPhoneMetadata {
                                Index = 0,
                                Phoneme = "a",
                                FrameCount = 2,
                                Parameters = new HifiPhoneParameterMetadata {
                                    Gender = 50,
                                    Breathiness = 25,
                                    Tension = -30,
                                    Voicing = 60,
                                    GenderKeyShiftSemitones = 6,
                                    BreathNoiseGain = 1.5,
                                    VoicingGain = 0.6,
                                    HnsepRequested = true,
                                    HnsepApplied = false,
                                    HnsepReason = "no_model_or_separation_failed",
                                },
                            },
                        },
                    },
                };

                HifiDebugExporter.ExportToDirectory(dir, features);
                var loaded = HifiDebugExporter.Load(dir);

                var parameters = loaded.Metadata.Phones.Single().Parameters;
                Assert.Equal(50, parameters.Gender, 6);
                Assert.Equal(25, parameters.Breathiness, 6);
                Assert.Equal(-30, parameters.Tension, 6);
                Assert.Equal(60, parameters.Voicing, 6);
                Assert.Equal(6, parameters.GenderKeyShiftSemitones, 6);
                Assert.Equal(1.5, parameters.BreathNoiseGain, 6);
                Assert.Equal(0.6, parameters.VoicingGain, 6);
                Assert.True(parameters.HnsepRequested);
                Assert.False(parameters.HnsepApplied);
                Assert.Equal("no_model_or_separation_failed", parameters.HnsepReason);
            } finally {
                if (Directory.Exists(dir)) {
                    Directory.Delete(dir, recursive: true);
                }
            }
        }

        [Fact]
        public void HifiFrameParameterMappingsMatchHifisamplerStyleDefaults() {
            var neutral = new HifiFrameParameterAverages(0, 0, 0, 100);
            Assert.Equal(0, neutral.GenderKeyShiftSemitones, 6);
            Assert.Equal(1, neutral.BreathNoiseGain, 6);
            Assert.Equal(1, neutral.VoicingGain, 6);
            Assert.False(neutral.NeedsHnsep);

            var expressive = new HifiFrameParameterAverages(50, 25, -30, 60);
            Assert.Equal(6.0, expressive.GenderKeyShiftSemitones, 6);
            Assert.Equal(1.5, expressive.BreathNoiseGain, 6);
            Assert.Equal(0.6, expressive.VoicingGain, 6);
            Assert.True(expressive.NeedsHnsep);
        }

        [Fact]
        public void HifiFrameParameterTrackKeepsCurveShapeForCacheAndSourceSampling() {
            var rising = new HifiFrameParameterTrack(
                new[] { 0.0, 0.0 },
                new[] { 0.0, 0.0 },
                new[] { -100.0, 100.0 },
                new[] { 100.0, 100.0 });
            var falling = new HifiFrameParameterTrack(
                new[] { 0.0, 0.0 },
                new[] { 0.0, 0.0 },
                new[] { 100.0, -100.0 },
                new[] { 100.0, 100.0 });

            Assert.Equal(0, rising.Average.Tension, 6);
            Assert.Equal(0, falling.Average.Tension, 6);
            Assert.NotEqual(rising.CacheKey, falling.CacheKey);
            Assert.Equal(-100, rising.SampleAtSourceSample(0, 100).Tension, 6);
            Assert.Equal(100, rising.SampleAtSourceSample(99, 100).Tension, 6);
            Assert.True(rising.HasTension);
            Assert.True(rising.NeedsHnsep);
        }

        [Fact]
        public void HifiFrameParameterTrackProjectsTargetCurveThroughNonlinearSourceMap() {
            var target = new HifiFrameParameterTrack(
                new[] { 0.0, 0.0, 0.0, 0.0 },
                new[] { 0.0, 0.0, 0.0, 0.0 },
                new[] { 100.0, 100.0, -100.0, -100.0 },
                new[] { 100.0, 100.0, 100.0, 100.0 });
            var targetToSourceFrameMap = new[] { 0.0, 0.0, 3.0, 3.0 };

            var projected = target.ProjectToSourceFrames(targetToSourceFrameMap, 4);

            Assert.Equal(4, projected.FrameCount);
            Assert.NotEqual(target.CacheKey, projected.CacheKey);
            Assert.True(projected.SampleAtSourceSample(0, 400).Tension > 80);
            Assert.True(projected.SampleAtSourceSample(399, 400).Tension < -80);
            Assert.Equal(100, projected.SampleAtSourceSample(0, 400).Voicing, 6);
        }

        [Fact]
        public void HnsepTensionPreparationDoesNotMutateCachedHarmonic() {
            var method = typeof(HifiHnsepSourceProcessor)
                .GetMethod(
                    "PrepareHarmonicForRemix",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] { typeof(float[]), typeof(double) },
                    modifiers: null);
            Assert.NotNull(method);
            var cachedHarmonic = new float[HifiMelExtractor.SampleRate / 5];
            for (int i = 0; i < cachedHarmonic.Length; i++) {
                double t = i / (double)HifiMelExtractor.SampleRate;
                cachedHarmonic[i] = (float)(
                    0.12 * Math.Sin(2.0 * Math.PI * 220.0 * t)
                    + 0.04 * Math.Sin(2.0 * Math.PI * 2800.0 * t));
            }
            var before = cachedHarmonic.ToArray();

            var prepared = (float[])method!.Invoke(null, new object[] { cachedHarmonic, -50.0 })!;

            Assert.Equal(before, cachedHarmonic);
            Assert.NotSame(cachedHarmonic, prepared);
            Assert.True(Rms(Difference(prepared, before)) > 1e-5);
        }

        [Fact]
        public void HnsepRemixPreservesSourceRmsWhileChangingTexture() {
            var method = typeof(HifiHnsepSourceProcessor)
                .GetMethod(
                    "RemixHarmonicNoiseWithSourceEnergy",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] { typeof(float[]), typeof(float[]), typeof(double), typeof(double) },
                    modifiers: null);
            Assert.NotNull(method);
            var source = new float[HifiMelExtractor.SampleRate / 5];
            var harmonic = new float[source.Length];
            for (int i = 0; i < source.Length; i++) {
                double t = i / (double)HifiMelExtractor.SampleRate;
                harmonic[i] = (float)(0.12 * Math.Sin(2.0 * Math.PI * 220.0 * t));
                source[i] = harmonic[i] + (float)(0.02 * Math.Sin(2.0 * Math.PI * 3600.0 * t));
            }
            double sourceRms = Rms(source);

            var remixed = (float[])method!.Invoke(null, new object[] { source, harmonic, 3.0, 0.7 })!;

            Assert.Equal(source.Length, remixed.Length);
            double rmsDeltaDb = 20.0 * Math.Log10(Rms(remixed) / sourceRms);
            Assert.True(Math.Abs(rmsDeltaDb) < 0.1, $"expected RMS preserved, delta={rmsDeltaDb:F3}dB");
            Assert.True(Rms(Difference(remixed, source)) > 1e-4);
            foreach (float sample in remixed) {
                Assert.False(float.IsNaN(sample));
                Assert.False(float.IsInfinity(sample));
            }
        }

        [Fact]
        public void HnsepTensionSurvivesNeutralRemix() {
            var prepare = typeof(HifiHnsepSourceProcessor)
                .GetMethod(
                    "PrepareHarmonicForRemix",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] { typeof(float[]), typeof(double) },
                    modifiers: null);
            var remix = typeof(HifiHnsepSourceProcessor)
                .GetMethod(
                    "RemixHarmonicNoiseWithSourceEnergy",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] { typeof(float[]), typeof(float[]), typeof(float[]), typeof(HifiFrameParameterTrack) },
                    modifiers: null);
            Assert.NotNull(prepare);
            Assert.NotNull(remix);

            var source = new float[HifiMelExtractor.SampleRate / 3];
            var harmonic = new float[source.Length];
            for (int i = 0; i < source.Length; i++) {
                double t = i / (double)HifiMelExtractor.SampleRate;
                harmonic[i] = (float)(
                    0.12 * Math.Sin(2.0 * Math.PI * 220.0 * t)
                    + 0.035 * Math.Sin(2.0 * Math.PI * 3200.0 * t));
                source[i] = harmonic[i] + (float)(0.015 * Math.Sin(2.0 * Math.PI * 6200.0 * t));
            }

            var processedHarmonic = (float[])prepare!.Invoke(null, new object[] { harmonic, -50.0 })!;
            var neutralTrack = HifiFrameParameterTrack.Constant(new HifiFrameParameterAverages(0, 0, -50, 100));
            var remixed = (float[])remix!.Invoke(null, new object[] { source, harmonic, processedHarmonic, neutralTrack })!;

            Assert.Equal(source.Length, remixed.Length);
            Assert.True(Rms(Difference(remixed, source)) > 1e-4, "TENC must not cancel out when BREC=0 and VOIC=100.");
            Assert.True(BandRmsAt(remixed, 3200) / BandRmsAt(remixed, 220)
                > BandRmsAt(source, 3200) / BandRmsAt(source, 220));
        }

        [Fact]
        public void SourceFrameAwareTensionChangesOnlyRequestedSourceRegion() {
            var method = typeof(HifiHnsepSourceProcessor)
                .GetMethod(
                    "ApplyHifisamplerStyleSpectralTension",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] { typeof(float[]), typeof(HifiFrameParameterTrack) },
                    modifiers: null);
            Assert.NotNull(method);
            int sampleRate = HifiMelExtractor.SampleRate;
            var samples = new float[sampleRate];
            for (int i = 0; i < samples.Length; i++) {
                double t = i / (double)sampleRate;
                samples[i] = (float)(
                    0.12 * Math.Sin(2.0 * Math.PI * 220.0 * t)
                    + 0.12 * Math.Sin(2.0 * Math.PI * 3500.0 * t));
            }
            int frames = 64;
            var tension = new double[frames];
            for (int i = 0; i < frames; i++) {
                tension[i] = i < frames / 2 ? 100.0 : -100.0;
            }
            var track = new HifiFrameParameterTrack(
                new double[frames],
                new double[frames],
                tension,
                Enumerable.Repeat(100.0, frames).ToArray());

            var processed = (float[])method!.Invoke(null, new object[] { samples, track })!;

            double firstRatio = SegmentBandRmsAt(processed, sampleRate / 8, sampleRate / 4, 3500)
                / Math.Max(1e-9, SegmentBandRmsAt(processed, sampleRate / 8, sampleRate / 4, 220));
            double secondRatio = SegmentBandRmsAt(processed, sampleRate * 5 / 8, sampleRate / 4, 3500)
                / Math.Max(1e-9, SegmentBandRmsAt(processed, sampleRate * 5 / 8, sampleRate / 4, 220));
            Assert.True(firstRatio > secondRatio * 1.2, $"expected frame-aware spectral contrast, first={firstRatio:F4} second={secondRatio:F4}");
        }

        [Fact]
        public void SourceFrameAwareBreathinessChangesOnlyRequestedSourceRegion() {
            var method = typeof(HifiHnsepSourceProcessor)
                .GetMethod(
                    "RemixHarmonicNoiseWithSourceEnergy",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] { typeof(float[]), typeof(float[]), typeof(HifiFrameParameterTrack) },
                    modifiers: null);
            Assert.NotNull(method);
            int sampleRate = HifiMelExtractor.SampleRate;
            var source = new float[sampleRate];
            var harmonic = new float[sampleRate];
            for (int i = 0; i < source.Length; i++) {
                double t = i / (double)sampleRate;
                harmonic[i] = (float)(0.12 * Math.Sin(2.0 * Math.PI * 220.0 * t));
                source[i] = harmonic[i] + (float)(0.02 * Math.Sin(2.0 * Math.PI * 3600.0 * t));
            }
            int frames = 64;
            var breathiness = new double[frames];
            for (int i = 0; i < frames; i++) {
                breathiness[i] = i < frames / 2 ? 100.0 : 0.0;
            }
            var track = new HifiFrameParameterTrack(
                new double[frames],
                breathiness,
                new double[frames],
                Enumerable.Repeat(100.0, frames).ToArray());

            var remixed = (float[])method!.Invoke(null, new object[] { source, harmonic, track })!;
            var diff = Difference(remixed, source);

            double firstDiff = SegmentRms(diff, sampleRate / 8, sampleRate / 4);
            double secondDiff = SegmentRms(diff, sampleRate * 5 / 8, sampleRate / 4);
            Assert.True(firstDiff > secondDiff * 2.0, $"expected frame-aware breathiness, first={firstDiff:F6} second={secondDiff:F6}");
        }

        [Fact]
        public void TensionSpectralTiltMovesEnergyLikeHifisampler() {
            var method = typeof(HifiHnsepSourceProcessor)
                .GetMethod(
                    "ApplyHifisamplerStyleSpectralTension",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] { typeof(float[]), typeof(double) },
                    modifiers: null);
            Assert.NotNull(method);
            var samples = new float[HifiMelExtractor.SampleRate / 2];
            for (int i = 0; i < samples.Length; i++) {
                double t = i / (double)HifiMelExtractor.SampleRate;
                samples[i] = (float)(
                    0.12 * Math.Sin(2.0 * Math.PI * 220.0 * t)
                    + 0.12 * Math.Sin(2.0 * Math.PI * 3500.0 * t));
            }

            var tense = (float[])method!.Invoke(null, new object[] { samples, -1.0 })!;
            var loose = (float[])method!.Invoke(null, new object[] { samples, 1.0 })!;

            Assert.Equal(samples.Length, tense.Length);
            Assert.True(BandRmsAt(tense, 3500) / BandRmsAt(tense, 220)
                > BandRmsAt(loose, 3500) / BandRmsAt(loose, 220));
            foreach (float sample in tense.Concat(loose)) {
                Assert.False(float.IsNaN(sample));
                Assert.False(float.IsInfinity(sample));
            }
        }

        [Fact]
        public void RendererSuggestsAndSupportsGrowlCurve() {
            var renderer = new HifiNeuralPhraseRenderer();
            var descriptor = new UExpressionDescriptor {
                name = HifiGrowlProcessor.CurveName,
                abbr = HifiGrowlProcessor.CurveAbbr,
                type = UExpressionType.Curve,
                min = 0,
                max = 100,
                defaultValue = 0,
                isFlag = false,
            };

            var suggestions = renderer.GetSuggestedExpressions(null!, null!);

            Assert.True(renderer.SupportsExpression(descriptor));
            descriptor.abbr = "GROC";
            Assert.True(renderer.SupportsExpression(descriptor));
            Assert.Contains(suggestions, d => d.abbr == HifiGrowlProcessor.CurveAbbr && d.type == UExpressionType.Curve);
        }

        [Fact]
        public void DefaultExpressionsRegisterGrowlCurve() {
            var project = new UProject();
            OpenUtau.Core.Format.Ustx.AddDefaultExpressions(project);

            Assert.True(project.expressions.TryGetValue(OpenUtau.Core.Format.Ustx.GROC, out var descriptor));
            Assert.Equal(UExpressionType.Curve, descriptor.type);
            Assert.True(new HifiNeuralPhraseRenderer().SupportsExpression(descriptor));
        }

        [Fact]
        public void RendererSupportsHifiLinearParameterCurves() {
            var renderer = new HifiNeuralPhraseRenderer();
            Assert.True(renderer.SupportsExpression(new UExpressionDescriptor {
                name = OpenUtau.Core.Format.Ustx.VEL,
                abbr = OpenUtau.Core.Format.Ustx.VEL,
                type = UExpressionType.Numerical,
                min = 0,
                max = 200,
                defaultValue = 100,
            }));
            foreach (string abbr in new[] {
                OpenUtau.Core.Format.Ustx.GENC,
                OpenUtau.Core.Format.Ustx.BREC,
                OpenUtau.Core.Format.Ustx.TENC,
                OpenUtau.Core.Format.Ustx.VOIC,
            }) {
                Assert.True(renderer.SupportsExpression(new UExpressionDescriptor {
                    name = abbr,
                    abbr = abbr,
                    type = UExpressionType.Curve,
                    min = -100,
                    max = 100,
                    defaultValue = 0,
                }), $"expected renderer to support {abbr}");
            }
        }

        [Fact]
        public void HnsepConfigRejectsNonPowerOfTwoNfft() {
            string dir = Path.Combine(Path.GetTempPath(), "hifi-hnsep-config-test-" + Guid.NewGuid().ToString("N"));
            try {
                Directory.CreateDirectory(dir);
                string modelPath = Path.Combine(dir, "model.onnx");
                File.WriteAllText(Path.Combine(dir, "config.yaml"), "n_fft: 1500\nhop_length: 512\n");
                var method = typeof(HifiHnsepOnnx)
                    .GetMethod("ResolveModelConfig", BindingFlags.NonPublic | BindingFlags.Static);
                Assert.NotNull(method);

                var result = (ValueTuple<int, int>)method!.Invoke(null, new object[] { modelPath })!;

                Assert.Equal(2048, result.Item1);
                Assert.Equal(512, result.Item2);
            } finally {
                if (Directory.Exists(dir)) {
                    Directory.Delete(dir, recursive: true);
                }
            }
        }

        [Fact]
        public void HnsepCpuThreadCountFollowsOpenUtauRenderThreads() {
            var method = typeof(HifiHnsepOnnx)
                .GetMethod("ResolveCpuThreadCount", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);
            string oldEnv = Environment.GetEnvironmentVariable("HIFI_NEURAL_HNSEP_THREADS");
            int oldRenderThreads = Preferences.Default.NumRenderThreads;
            try {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_HNSEP_THREADS", null);
                Preferences.Default.NumRenderThreads = 999;

                int threads = (int)method!.Invoke(null, Array.Empty<object>())!;

                int expected = Math.Clamp(999, 1, Math.Max(1, Environment.ProcessorCount / 2));
                Assert.Equal(expected, threads);

                Environment.SetEnvironmentVariable("HIFI_NEURAL_HNSEP_THREADS", "1");
                threads = (int)method.Invoke(null, Array.Empty<object>())!;
                Assert.Equal(1, threads);
            } finally {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_HNSEP_THREADS", oldEnv);
                Preferences.Default.NumRenderThreads = oldRenderThreads;
            }
        }

        [Fact]
        public void HnsepConcurrencyLimitsCpuOversubscription() {
            var method = typeof(HifiHnsepOnnx)
                .GetMethod("ResolveMaxConcurrentSeparations", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);
            string oldEnv = Environment.GetEnvironmentVariable("HIFI_NEURAL_HNSEP_CONCURRENCY");
            int oldRenderThreads = Preferences.Default.NumRenderThreads;
            try {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_HNSEP_CONCURRENCY", null);
                Preferences.Default.NumRenderThreads = Math.Max(2, Environment.ProcessorCount);
                int workerThreads = Math.Max(1, Environment.ProcessorCount);

                int concurrency = (int)method!.Invoke(null, new object[] { workerThreads })!;

                Assert.Equal(1, concurrency);

                Environment.SetEnvironmentVariable("HIFI_NEURAL_HNSEP_CONCURRENCY", "2");
                concurrency = (int)method.Invoke(null, new object[] { workerThreads })!;
                Assert.Equal(Math.Min(2, Preferences.Default.NumRenderThreads), concurrency);
            } finally {
                Environment.SetEnvironmentVariable("HIFI_NEURAL_HNSEP_CONCURRENCY", oldEnv);
                Preferences.Default.NumRenderThreads = oldRenderThreads;
            }
        }

        [Fact]
        public void HnsepIstftConstrainsDcAndNyquistToRealValues() {
            var method = typeof(HifiHnsepOnnx)
                .GetMethod("ConstrainRealSignalSpectrum", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);
            var fft = new[] {
                new Complex(1, 2),
                new Complex(3, 4),
                new Complex(5, 6),
            };

            method!.Invoke(null, new object[] { fft, 3 });

            Assert.Equal(0, fft[0].Imaginary, 6);
            Assert.Equal(4, fft[1].Imaginary, 6);
            Assert.Equal(0, fft[2].Imaginary, 6);
        }

        [Fact]
        public void GrowlZeroStrengthDoesNotChangeSamples() {
            var samples = GrowlTestWave();
            var before = samples.ToArray();
            var strength = new float[samples.Length];

            HifiGrowlProcessor.ApplyInPlace(samples, HifiMelExtractor.SampleRate, strength);

            Assert.Equal(before, samples);
        }

        [Fact]
        public void GrowlModifiesHighBandWithoutLengthOrPeakChange() {
            var samples = GrowlTestWave();
            var before = samples.ToArray();
            var strength = Enumerable.Repeat(0.75f, samples.Length).ToArray();
            double beforeRms = Rms(samples);

            HifiGrowlProcessor.ApplyInPlace(samples, HifiMelExtractor.SampleRate, strength);

            Assert.Equal(before.Length, samples.Length);
            Assert.True(samples.Max(Math.Abs) <= 0.9801f);
            Assert.True(Rms(Difference(samples, before)) > 1e-4);
            Assert.True(Math.Abs(20.0 * Math.Log10(Rms(samples) / beforeRms)) < 0.75);
            foreach (float sample in samples) {
                Assert.False(float.IsNaN(sample));
                Assert.False(float.IsInfinity(sample));
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

        [Fact]
        public void PhraseEdgeGuardKeepsNaturallyQuietTail() {
            var samples = new float[44100 / 5];
            for (int i = 0; i < samples.Length; i++) {
                samples[i] = 0.2f;
            }
            for (int i = 0; i < 512; i++) {
                int index = samples.Length - 1 - i;
                samples[index] = 0.0002f * (i / 512f);
            }
            float before = samples[^128];

            HifiNeuralPhraseRenderer.ApplyPhraseEdgeGuard(samples, 44100);

            Assert.Equal(before, samples[^128], 6);
        }

        [Fact]
        public void PhraseEdgeGuardFadesNonZeroEdges() {
            var samples = Enumerable.Repeat(0.2f, 44100 / 5).ToArray();

            HifiNeuralPhraseRenderer.ApplyPhraseEdgeGuard(samples, 44100);

            Assert.Equal(0f, samples[0], 6);
            Assert.Equal(0f, samples[^1], 6);
            Assert.True(samples[400] > samples[0]);
            Assert.True(samples[^401] > samples[^1]);
        }

        [Fact]
        public void SegmentEndDoesNotStretchAcrossRestGap() {
            int end = ResolveSegmentEndFrame(
                startFrame: 10,
                nextAnchorFrame: 80,
                overlapTailFrames: 0,
                targetFrames: 100,
                hasNextPhone: true,
                hasRestGap: true,
                phoneReleaseEndFrame: 10 + FramesForMs(260),
                correctedEnvelopeEndFrame: 10 + FramesForMs(340));

            Assert.Equal(10 + FramesForMs(260), end);
            Assert.True(end < 80, "phone mel should end near its release instead of filling a rest gap");
        }

        [Fact]
        public void SegmentEndKeepsOtoOverlapTailForConnectedPhones() {
            int startFrame = 5;
            int nextAnchorFrame = 85;
            int overlapTailFrames = 10;

            int end = ResolveSegmentEndFrame(
                startFrame,
                nextAnchorFrame,
                overlapTailFrames,
                targetFrames: 120,
                hasNextPhone: true,
                hasRestGap: false,
                phoneReleaseEndFrame: 40,
                correctedEnvelopeEndFrame: 35);

            Assert.Equal(nextAnchorFrame + overlapTailFrames, end);
        }

        [Fact]
        public void SegmentEndUsesCorrectedEnvelopeOnlyForRestGap() {
            int end = ResolveSegmentEndFrame(
                startFrame: 10,
                nextAnchorFrame: 90,
                overlapTailFrames: 8,
                targetFrames: 120,
                hasNextPhone: true,
                hasRestGap: true,
                phoneReleaseEndFrame: 70,
                correctedEnvelopeEndFrame: 55);

            Assert.Equal(55, end);
        }

        [Fact]
        public void SegmentEndDoesNotCutPhraseFinalTail() {
            int end = ResolveSegmentEndFrame(
                startFrame: 10,
                nextAnchorFrame: 100,
                overlapTailFrames: 0,
                targetFrames: 100,
                hasNextPhone: false,
                hasRestGap: true,
                phoneReleaseEndFrame: 40,
                correctedEnvelopeEndFrame: 30);

            Assert.Equal(100, end);
        }

        [Fact]
        public void TargetFixedLeadCapsLargeVcvPreutter() {
            double fixedMs = ResolveTargetFixedMs(
                preutterMs: 180,
                overlapMs: 80,
                durationMs: 120,
                hasReliableConsonant: true);

            Assert.True(fixedMs is >= 60 and <= 75, $"large preutter should be shortened but still audible, got {fixedMs:F3}ms");
            Assert.True(fixedMs >= HifiF0Builder.FrameMs);
        }

        [Fact]
        public void TargetFixedLeadKeepsShortConsonantsAudible() {
            double fixedMs = ResolveTargetFixedMs(
                preutterMs: 180,
                overlapMs: 80,
                durationMs: 70,
                hasReliableConsonant: true);

            Assert.True(fixedMs >= 40, $"short-note consonant lead should not collapse, got {fixedMs:F3}ms");
            Assert.True(fixedMs < 55, $"short-note consonant lead should not dominate the note, got {fixedMs:F3}ms");
        }

        [Fact]
        public void TargetFixedLeadUsesSourceOtoTimingInsteadOfPhonemeSpelling() {
            var romaji = CreateRenderPhoneForTiming(
                preutterMs: 180,
                overlapMs: 80,
                durationMs: 70,
                consonantMs: 230,
                phoneme: "fang");
            var kana = CreateRenderPhoneForTiming(
                preutterMs: 180,
                overlapMs: 80,
                durationMs: 70,
                consonantMs: 230,
                phoneme: "か");

            double romajiFixedMs = HifiPhraseFeatureBuilder.ResolveTargetFixedMs(romaji);
            double kanaFixedMs = HifiPhraseFeatureBuilder.ResolveTargetFixedMs(kana);

            Assert.Equal(romajiFixedMs, kanaFixedMs, 3);
        }

        [Fact]
        public void TargetFixedLeadAdaptsToSourcePreutterBudget() {
            double shortSourceFixedMs = HifiPhraseFeatureBuilder.ResolveTargetFixedMs(
                preutterMs: 180,
                overlapMs: 80,
                durationMs: 70,
                hasReliableConsonant: true,
                sourcePreutterMs: 60,
                sourceOverlapMs: 40,
                sourceStableStartMs: 90);
            double longSourceFixedMs = HifiPhraseFeatureBuilder.ResolveTargetFixedMs(
                preutterMs: 180,
                overlapMs: 80,
                durationMs: 70,
                hasReliableConsonant: true,
                sourcePreutterMs: 180,
                sourceOverlapMs: 80,
                sourceStableStartMs: 230);

            Assert.True(longSourceFixedMs > shortSourceFixedMs, $"longer source preutter budget should keep slightly more fixed lead, got short={shortSourceFixedMs:F3}ms long={longSourceFixedMs:F3}ms");
        }

        [Fact]
        public void TargetFixedLeadDoesNotExceedUpperBudget() {
            double fixedMs = HifiPhraseFeatureBuilder.ResolveTargetFixedMs(
                preutterMs: 180,
                overlapMs: 40,
                durationMs: 25,
                hasReliableConsonant: true,
                sourcePreutterMs: 220,
                sourceOverlapMs: 20,
                sourceStableStartMs: 260);

            double durationCapMs = Math.Max(HifiF0Builder.FrameMs * 2.0, 25 * 0.60);

            Assert.True(fixedMs <= durationCapMs + 0.001, $"fixed lead must not exceed duration cap, got fixed={fixedMs:F3}ms cap={durationCapMs:F3}ms");
        }

        [Fact]
        public void TargetFixedLeadUsesSourceOtoOverlap() {
            var smallOverlap = CreateRenderPhoneForTiming(
                preutterMs: 180,
                overlapMs: 80,
                durationMs: 140,
                consonantMs: 230,
                sourcePreutterMs: 180,
                sourceOverlapMs: 10);
            var largeOverlap = CreateRenderPhoneForTiming(
                preutterMs: 180,
                overlapMs: 80,
                durationMs: 140,
                consonantMs: 230,
                sourcePreutterMs: 180,
                sourceOverlapMs: 120);

            double smallOverlapFixedMs = HifiPhraseFeatureBuilder.ResolveTargetFixedMs(smallOverlap);
            double largeOverlapFixedMs = HifiPhraseFeatureBuilder.ResolveTargetFixedMs(largeOverlap);

            Assert.True(smallOverlapFixedMs > largeOverlapFixedMs, $"larger source overlap should reduce fixed lead budget, got small={smallOverlapFixedMs:F3}ms large={largeOverlapFixedMs:F3}ms");
        }

        [Fact]
        public void TargetFixedLeadPreservesSmallPreutter() {
            double fixedMs = ResolveTargetFixedMs(
                preutterMs: 24,
                overlapMs: 4,
                durationMs: 240,
                hasReliableConsonant: true);

            Assert.Equal(24, fixedMs, 3);
        }

        [Fact]
        public void TargetFixedLeadKeepsRawPreutterAnchorInSourceMap() {
            double preutterMs = 180;
            int sourceFrames = 120;
            int targetFrames = 60;
            var phone = CreateRenderPhoneForTiming(
                preutterMs,
                overlapMs: 80,
                durationMs: 120,
                consonantMs: 230);

            double[] map = HifiPhraseFeatureBuilder.BuildPhoneTargetToSourceFrameMap(sourceFrames, targetFrames, phone);
            int targetLeadFrames = Math.Min(targetFrames - 1, FramesForMs(preutterMs));
            int sourceLeadFrames = Math.Min(sourceFrames - 4, SourceFramesForMs(preutterMs));

            Assert.True(ResolveTargetFixedMs(180, 80, 120, true) < preutterMs);
            Assert.Equal(sourceLeadFrames, map[targetLeadFrames], 1.0);
        }

        [Fact]
        public void SourceMapSoftlyCatchesUpWhenTargetPreutterIsCapped() {
            var phone = CreateRenderPhoneForTiming(
                preutterMs: 80,
                overlapMs: 40,
                durationMs: 120,
                consonantMs: 250,
                sourcePreutterMs: 180);

            double[] map = HifiPhraseFeatureBuilder.BuildPhoneTargetToSourceFrameMap(
                sourceFrames: 160,
                outputFrames: 80,
                phone);
            int targetLeadFrames = ResolveTargetLeadFrames(phone, outputFrames: 80);
            int targetPreutterSourceFrames = SourceFramesForMs(80);
            int rawOtoPreutterSourceFrames = SourceFramesForMs(180);
            int midLead = Math.Max(2, targetLeadFrames / 2);
            double linearMid = midLead * (rawOtoPreutterSourceFrames - 1.0) / Math.Max(1, targetLeadFrames - 1);
            double linearStep = (rawOtoPreutterSourceFrames - 1.0) / Math.Max(1, targetLeadFrames - 1);
            double maxStep = 0;
            for (int i = 1; i < targetLeadFrames; i++) {
                maxStep = Math.Max(maxStep, map[i] - map[i - 1]);
            }

            Assert.True(rawOtoPreutterSourceFrames > targetPreutterSourceFrames);
            Assert.Equal(0, map[0], 6);
            Assert.True(map[1] < SourceFramesForMs(25), $"lead should not hard-skip at onset, got frame {map[1]:F3}");
            Assert.True(map[midLead] > linearMid + 2.0, $"lead should smoothly catch up inside preutter, got {map[midLead]:F3} vs linear {linearMid:F3}");
            Assert.True(maxStep <= linearStep * 1.30, $"lead catch-up step is too abrupt, max={maxStep:F3} linear={linearStep:F3}");
            Assert.Equal(rawOtoPreutterSourceFrames, map[targetLeadFrames], 1.0);
        }

        [Fact]
        public void TargetLeadLeavesVowelCoreOnShortPhones() {
            var phone = CreateRenderPhoneForTiming(
                preutterMs: 180,
                overlapMs: 80,
                durationMs: 70,
                consonantMs: 230);

            int leadFrames = ResolveTargetLeadFrames(phone, outputFrames: 6);

            Assert.Equal(4, leadFrames);
        }

        static double ResolveTargetFixedMs(
            double preutterMs,
            double overlapMs,
            double durationMs,
            bool hasReliableConsonant) {
            var method = typeof(HifiPhraseFeatureBuilder)
                .GetMethod(
                    "ResolveTargetFixedMs",
                    BindingFlags.NonPublic | BindingFlags.Static,
                    binder: null,
                    types: new[] {
                        typeof(double),
                        typeof(double),
                        typeof(double),
                        typeof(bool),
                    },
                    modifiers: null);
            Assert.NotNull(method);
            return (double)method!.Invoke(null, new object[] {
                preutterMs,
                overlapMs,
                durationMs,
                hasReliableConsonant,
            })!;
        }

        static int ResolveTargetLeadFrames(RenderPhone phone, int outputFrames) {
            var method = typeof(HifiPhraseFeatureBuilder)
                .GetMethod("ResolveTargetLeadFrames", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);
            return (int)method!.Invoke(null, new object[] {
                phone,
                (double?)phone.oto.Consonant,
                outputFrames,
            })!;
        }

        static int ResolveSegmentEndFrame(
            int startFrame,
            int nextAnchorFrame,
            int overlapTailFrames,
            int targetFrames,
            bool hasNextPhone,
            bool hasRestGap,
            int phoneReleaseEndFrame,
            int correctedEnvelopeEndFrame) {
            var method = typeof(HifiMelPhraseAssembler)
                .GetMethod("ResolveSegmentEndFrame", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(method);
            return (int)method!.Invoke(null, new object[] {
                startFrame,
                nextAnchorFrame,
                overlapTailFrames,
                targetFrames,
                hasNextPhone,
                hasRestGap,
                phoneReleaseEndFrame,
                correctedEnvelopeEndFrame,
            })!;
        }

        static int FramesForMs(double ms) => (int)Math.Round(ms / HifiF0Builder.FrameMs);

        static int SourceFramesForMs(double ms) => (int)Math.Round(ms / (1000.0 * HifiMelExtractor.OriginHopSize / HifiMelExtractor.SampleRate));

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
            double rms = SegmentRms(samples, start, count);
            return 20.0 * Math.Log10(Math.Max(1e-7, rms));
        }

        static double SegmentRms(float[] samples, int start, int count) {
            start = Math.Clamp(start, 0, samples.Length);
            int end = Math.Clamp(start + count, start, samples.Length);
            double sum = 0;
            for (int i = start; i < end; i++) {
                sum += samples[i] * samples[i];
            }
            return Math.Sqrt(sum / Math.Max(1, end - start));
        }

        static float[] GrowlTestWave() {
            int sampleRate = HifiMelExtractor.SampleRate;
            var samples = new float[sampleRate / 2];
            for (int i = 0; i < samples.Length; i++) {
                double t = i / (double)sampleRate;
                samples[i] = (float)(
                    0.16 * Math.Sin(2.0 * Math.PI * 220.0 * t)
                    + 0.04 * Math.Sin(2.0 * Math.PI * 1800.0 * t)
                    + 0.02 * Math.Sin(2.0 * Math.PI * 3600.0 * t));
            }
            return samples;
        }

        static float[] Difference(float[] left, float[] right) {
            int length = Math.Min(left.Length, right.Length);
            var diff = new float[length];
            for (int i = 0; i < length; i++) {
                diff[i] = left[i] - right[i];
            }
            return diff;
        }

        static float[] Flatten(float[,] values) {
            var flattened = new float[values.Length];
            int index = 0;
            for (int m = 0; m < values.GetLength(0); m++) {
                for (int t = 0; t < values.GetLength(1); t++) {
                    flattened[index++] = values[m, t];
                }
            }
            return flattened;
        }

        static double MelCentroid(float[,] mel) {
            double weighted = 0;
            double total = 0;
            for (int m = 0; m < mel.GetLength(0); m++) {
                for (int t = 0; t < mel.GetLength(1); t++) {
                    double power = Math.Exp(mel[m, t]);
                    weighted += m * power;
                    total += power;
                }
            }
            return weighted / Math.Max(1e-9, total);
        }

        static double BandRmsAt(float[] samples, double frequency) {
            double re = 0;
            double im = 0;
            for (int i = 0; i < samples.Length; i++) {
                double phase = 2.0 * Math.PI * frequency * i / HifiMelExtractor.SampleRate;
                re += samples[i] * Math.Cos(phase);
                im -= samples[i] * Math.Sin(phase);
            }
            return Math.Sqrt(re * re + im * im) / Math.Max(1, samples.Length);
        }

        static double SegmentBandRmsAt(float[] samples, int start, int count, double frequency) {
            start = Math.Clamp(start, 0, samples.Length);
            int end = Math.Clamp(start + count, start, samples.Length);
            double re = 0;
            double im = 0;
            for (int i = start; i < end; i++) {
                double phase = 2.0 * Math.PI * frequency * i / HifiMelExtractor.SampleRate;
                re += samples[i] * Math.Cos(phase);
                im -= samples[i] * Math.Sin(phase);
            }
            return Math.Sqrt(re * re + im * im) / Math.Max(1, end - start);
        }

        static double Rms(float[] samples) {
            if (samples.Length == 0) {
                return 0;
            }
            double sum = 0;
            for (int i = 0; i < samples.Length; i++) {
                sum += samples[i] * samples[i];
            }
            return Math.Sqrt(sum / samples.Length);
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

        static RenderNote CreateRenderNoteForFallback(int tone, double positionMs, double endMs) {
            var note = (RenderNote)FormatterServices.GetUninitializedObject(typeof(RenderNote));
            SetReadonlyField(note, "tone", tone);
            SetReadonlyField(note, "adjustedTone", 0f);
            SetReadonlyField(note, "positionMs", positionMs);
            SetReadonlyField(note, "endMs", endMs);
            return note;
        }

        static RenderPhone CreateRenderPhoneForTiming(
            double preutterMs,
            double overlapMs,
            double durationMs,
            double consonantMs,
            double? sourcePreutterMs = null,
            double? sourceOverlapMs = null,
            string phoneme = "a") {
            var phone = (RenderPhone)FormatterServices.GetUninitializedObject(typeof(RenderPhone));
            var oto = new UOto {
                Consonant = consonantMs,
                Preutter = sourcePreutterMs ?? preutterMs,
                Overlap = sourceOverlapMs ?? overlapMs,
            };
            SetReadonlyField(phone, "phoneme", phoneme);
            SetReadonlyField(phone, "preutterMs", preutterMs);
            SetReadonlyField(phone, "overlapMs", overlapMs);
            SetReadonlyField(phone, "durationMs", durationMs);
            SetReadonlyField(phone, "oto", oto);
            return phone;
        }

        static void SetReadonlyField(object target, string fieldName, object value) {
            var field = target.GetType().GetField(fieldName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.NotNull(field);
            field!.SetValue(target, value);
        }
    }
}
