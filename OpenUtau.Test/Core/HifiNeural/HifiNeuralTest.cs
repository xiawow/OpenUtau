using System;
using System.Collections.Generic;
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
        public void HifiRendererCanBeClassicDefaultRenderer() {
            string original = Preferences.Default.DefaultRenderer;
            try {
                Preferences.Default.DefaultRenderer = Renderers.HIFI_NEURAL_PHRASE;
                Assert.Equal(Renderers.HIFI_NEURAL_PHRASE, Renderers.GetDefaultRenderer(Ustx.USingerType.Classic));
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
        public void SimpleStretchKeepsRequestedFrameCount() {
            var mel = CreateTestMel(frames: 16);
            var result = HifiPhoneMelStretcher.Stretch(mel, 9, "a", consonantMs: null, HifiStretchMode.Simple);

            Assert.Equal(9, result.Mel.GetLength(1));
            Assert.Equal("mode=simple", result.Debug.FallbackReason);
            Assert.All(result.Mel.Cast<float>(), value => {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            });
        }

        [Fact]
        public void ConsonantVowelSplitContinuouslyStretchesVowelForLongTargets() {
            var mel = CreateTestMel(frames: 32);
            var result = HifiPhoneMelStretcher.Stretch(mel, 40, "ka", consonantMs: 20, HifiStretchMode.ConsonantVowelSplit);

            Assert.Equal(40, result.Mel.GetLength(1));
            Assert.Equal(string.Empty, result.Debug.FallbackReason);
            Assert.True(result.Debug.ConsonantFrames > 0);
            Assert.True(result.Debug.VowelSourceFrames >= 6);
            Assert.True(result.Debug.VowelTargetFrames > result.Debug.VowelSourceFrames);
            Assert.Equal(0, result.Debug.LoopCount);
            Assert.Equal(0, result.Debug.CrossfadeFrames);
            Assert.Equal("continuous_sustain_stretch", result.Debug.VowelStretchStrategy);
            Assert.All(result.Mel.Cast<float>(), value => {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            });
        }

        [Fact]
        public void ConsonantVowelSplitUsesContinuousSustainRegionAndKeepsTargetFrames() {
            var mel = CreateTestMel(frames: 48);
            for (int m = 0; m < mel.GetLength(0); m++) {
                for (int t = 30; t < 40; t++) {
                    mel[m, t] = 1.0f + m * 0.0001f;
                }
            }
            var result = HifiPhoneMelStretcher.Stretch(mel, 72, "la", consonantMs: 20, HifiStretchMode.ConsonantVowelSplit);

            Assert.Equal(72, result.Mel.GetLength(1));
            Assert.Equal(string.Empty, result.Debug.FallbackReason);
            Assert.InRange(result.Debug.LoopStartFrame, result.Debug.SourceConsonantFrames, 47);
            Assert.True(result.Debug.LoopEndFrame > result.Debug.LoopStartFrame);
            Assert.Equal(0, result.Debug.LoopCount);
        }

        [Fact]
        public void ConsonantVowelSplitUsesContinuousStretchOnStableF0Region() {
            var mel = CreateTestMel(frames: 48);
            var targetF0 = Enumerable.Repeat(220f, 72).ToArray();
            var result = HifiPhoneMelStretcher.Stretch(
                mel,
                72,
                "la",
                consonantMs: 20,
                HifiStretchMode.ConsonantVowelSplit,
                targetF0: targetF0);

            Assert.Equal(72, result.Mel.GetLength(1));
            Assert.Equal("continuous_sustain_stretch", result.Debug.VowelStretchStrategy);
            Assert.Equal(0, result.Debug.LoopCount);
        }

        [Fact]
        public void ConsonantVowelSplitLabelsFastPitchMotionWithoutLooping() {
            var mel = CreateTestMel(frames: 48);
            var targetF0 = Enumerable.Range(0, 72)
                .Select(i => (float)(220.0 * Math.Pow(2.0, i * 100.0 / 1200.0)))
                .ToArray();
            var result = HifiPhoneMelStretcher.Stretch(
                mel,
                72,
                "la",
                consonantMs: 20,
                HifiStretchMode.ConsonantVowelSplit,
                targetF0: targetF0);

            Assert.Equal(72, result.Mel.GetLength(1));
            Assert.Equal("continuous_sustain_stretch_f0_motion", result.Debug.VowelStretchStrategy);
            Assert.Equal(0, result.Debug.LoopCount);
        }

        [Fact]
        public void ConsonantVowelSplitFallsBackWithoutConsonantInfo() {
            var mel = CreateTestMel(frames: 32);
            var result = HifiPhoneMelStretcher.Stretch(mel, 24, "a", consonantMs: null, HifiStretchMode.ConsonantVowelSplit);

            Assert.Equal(24, result.Mel.GetLength(1));
            Assert.Equal("missing_consonant_ms", result.Debug.FallbackReason);
        }

        [Fact]
        public void StretchEnergyCompensationIsDisabledByDefault() {
            var mel = CreateLowHighLowMel(frames: 48, low: -5f, high: -2f);
            var result = HifiPhoneMelStretcher.Stretch(mel, 120, "ka", consonantMs: 20, HifiStretchMode.ConsonantVowelSplit);

            Assert.Equal(120, result.Mel.GetLength(1));
            Assert.Equal(0, result.Debug.StretchEnergyCutDb, precision: 3);
            Assert.Equal(mel[0, 0], result.Mel[0, 0]);
            Assert.All(result.Mel.Cast<float>(), value => {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            });
        }

        [Fact]
        public void SimpleStretchDoesNotApplyStretchEnergyByDefault() {
            var mel = CreateLowHighLowMel(frames: 32, low: -5f, high: -2f);
            var result = HifiPhoneMelStretcher.Stretch(mel, 96, "a", consonantMs: null, HifiStretchMode.Simple);

            Assert.Equal("mode=simple", result.Debug.FallbackReason);
            Assert.Equal(0, result.Debug.StretchEnergyCutDb, precision: 3);
        }

        [Fact]
        public void HifiStretchModeConfigParsesModes() {
            Assert.Equal(HifiStretchMode.Simple, HifiNeuralConfig.ParseStretchMode("simple"));
            Assert.Equal(HifiStretchMode.ConsonantVowelSplit, HifiNeuralConfig.ParseStretchMode("consonant_vowel_split"));
            Assert.Equal(HifiStretchMode.Simple, HifiNeuralConfig.ParseStretchMode("unknown"));
        }

        [Fact]
        public void DurationRedistributionDisabledPreservesDurations() {
            var plans = CreateDurationPlans();
            var result = HifiDurationRedistributor.Redistribute(
                plans,
                enabled: false,
                phraseFrames: 40,
                minConsonantMs: 40,
                minVowelMs: 30,
                maxBorrowMs: 30,
                maxBorrowRatio: 0.5,
                lockMode: HifiConsonantLockMode.Readable);

            Assert.Equal(plans[0].OriginalStartFrame, result.Plans[0].AdjustedStartFrame);
            Assert.Equal(plans[0].OriginalDurationFrames, result.Plans[0].AdjustedDurationFrames);
            Assert.Equal(plans[1].OriginalStartFrame, result.Plans[1].AdjustedStartFrame);
            Assert.Equal(plans[1].OriginalDurationFrames, result.Plans[1].AdjustedDurationFrames);
            Assert.Equal(0, result.Summary.TotalBorrowedFrames);
            Assert.Equal(result.Summary.PhraseFramesBefore, result.Summary.PhraseFramesAfter);
        }

        [Fact]
        public void DurationRedistributionBorrowsFromPreviousVowelFirst() {
            var plans = CreateDurationPlans();
            var result = HifiDurationRedistributor.Redistribute(
                plans,
                enabled: true,
                phraseFrames: 40,
                minConsonantMs: 40,
                minVowelMs: 30,
                maxBorrowMs: 30,
                maxBorrowRatio: 0.5,
                lockMode: HifiConsonantLockMode.Readable);

            Assert.True(result.Plans[1].BorrowedFromPreviousFrames > 0);
            Assert.Equal(0, result.Plans[1].BorrowedFromNextFrames);
            Assert.True(result.Plans[0].AdjustedDurationFrames < plans[0].OriginalDurationFrames);
            Assert.True(result.Plans[1].AdjustedDurationFrames > plans[1].OriginalDurationFrames);
            Assert.Equal(result.Plans[0].DonatedFrames, result.Summary.TotalBorrowedFrames);
            Assert.Equal("consonant", result.Plans[1].BorrowedFramesAppliedTo);
            Assert.Equal(result.Summary.PhraseFramesBefore, result.Summary.PhraseFramesAfter);
        }

        [Fact]
        public void DurationRedistributionPreserveModeAppliesBorrowToVowel() {
            var plans = CreateDurationPlans();
            var result = HifiDurationRedistributor.Redistribute(
                plans,
                enabled: true,
                phraseFrames: 40,
                minConsonantMs: 40,
                minVowelMs: 30,
                maxBorrowMs: 30,
                maxBorrowRatio: 0.5,
                lockMode: HifiConsonantLockMode.Preserve);

            Assert.True(result.Plans[1].BorrowedFromPreviousFrames > 0);
            Assert.Equal("vowel", result.Plans[1].BorrowedFramesAppliedTo);
            Assert.True(result.Plans[1].TargetConsonantFrames <= plans[1].ConsonantFrames);
        }

        [Fact]
        public void PreserveLockKeepsConsonantStretchNearOne() {
            var mel = CreateTestMel(frames: 32);
            var preserve = HifiPhoneMelStretcher.Stretch(
                mel, 48, "ka", consonantMs: 20, HifiStretchMode.ConsonantVowelSplit,
                targetConsonantFrames: 20, borrowedFrames: 8, borrowedFramesAppliedTo: "vowel",
                lockModeOverride: HifiConsonantLockMode.Preserve);
            var readable = HifiPhoneMelStretcher.Stretch(
                mel, 48, "ka", consonantMs: 20, HifiStretchMode.ConsonantVowelSplit,
                targetConsonantFrames: 20, borrowedFrames: 8, borrowedFramesAppliedTo: "consonant",
                lockModeOverride: HifiConsonantLockMode.Readable);

            Assert.Equal("preserve", preserve.Debug.ConsonantLockMode);
            Assert.True(preserve.Debug.StretchRatioConsonant <= 1.15);
            Assert.True(readable.Debug.StretchRatioConsonant >= preserve.Debug.StretchRatioConsonant);
        }

        [Fact]
        public void PreserveLockRespectsShortPhoneVowelFloor() {
            var mel = CreateTestMel(frames: 32);
            var result = HifiPhoneMelStretcher.Stretch(
                mel,
                targetFrames: 4,
                phoneme: "ka",
                consonantMs: 60,
                mode: HifiStretchMode.ConsonantVowelSplit,
                targetConsonantFrames: 2,
                lockModeOverride: HifiConsonantLockMode.Preserve);

            Assert.Equal(4, result.Mel.GetLength(1));
            Assert.Equal(2, result.Debug.TargetConsonantFrames);
            Assert.Equal(2, result.Debug.TargetVowelFrames);
        }

        [Fact]
        public void DurationRedistributionDisabledKeepsAudibleVowelInShortSyllable() {
            var plans = new List<HifiPhoneDurationPlan> {
                new HifiPhoneDurationPlan {
                    Index = 0,
                    Phoneme = "ka",
                    OriginalStartFrame = 0,
                    OriginalDurationFrames = 4,
                    AdjustedStartFrame = 0,
                    AdjustedDurationFrames = 4,
                    ConsonantFrames = 4,
                    TargetConsonantFrames = 4,
                    CanDonateVowel = true,
                },
            };

            var result = HifiDurationRedistributor.Redistribute(
                plans,
                enabled: false,
                phraseFrames: 4,
                minConsonantMs: 40,
                minVowelMs: 30,
                maxBorrowMs: 30,
                maxBorrowRatio: 0.5,
                lockMode: HifiConsonantLockMode.Preserve);

            Assert.Equal(4, result.Plans[0].AdjustedDurationFrames);
            Assert.Equal(1, result.Plans[0].TargetConsonantFrames);
            Assert.Equal(3, result.Plans[0].AdjustedDurationFrames - result.Plans[0].TargetConsonantFrames);
        }

        [Fact]
        public void DurationRedistributionFallsBackWhenNoVowelAvailable() {
            var plans = new List<HifiPhoneDurationPlan> {
                new HifiPhoneDurationPlan {
                    Index = 0,
                    Phoneme = "k",
                    OriginalStartFrame = 0,
                    OriginalDurationFrames = 3,
                    AdjustedStartFrame = 0,
                    AdjustedDurationFrames = 3,
                    ConsonantFrames = 2,
                    TargetConsonantFrames = 2,
                    CanDonateVowel = false,
                },
            };
            var result = HifiDurationRedistributor.Redistribute(
                plans,
                enabled: true,
                phraseFrames: 3,
                minConsonantMs: 40,
                minVowelMs: 30,
                maxBorrowMs: 30,
                maxBorrowRatio: 0.5,
                lockMode: HifiConsonantLockMode.Readable);

            Assert.Equal("no_adjacent_vowel_available", result.Plans[0].FallbackReason);
            Assert.Equal(0, result.Summary.TotalBorrowedFrames);
            Assert.Equal(3, result.Plans[0].AdjustedDurationFrames);
        }

        [Fact]
        public void MelPostProcessorNormalizesAndSmoothsWithoutChangingShape() {
            var mel = new float[2, 5];
            var weights = new[] { 1f, 2f, 0f, 1f, 1f };
            mel[0, 0] = 2;
            mel[0, 1] = 4;
            mel[0, 3] = 6;
            mel[0, 4] = 8;
            mel[1, 0] = 1;
            mel[1, 1] = 2;
            mel[1, 3] = 3;
            mel[1, 4] = 4;

            HifiMelPostProcessor.NormalizeWeightedMel(mel, weights);
            HifiMelPostProcessor.SmoothBoundaries(
                mel,
                new[] { 2 },
                smoothFrames: 1,
                reason: "test",
                protectedRanges: Array.Empty<(int start, int end)>());

            Assert.Equal(2, mel.GetLength(0));
            Assert.Equal(5, mel.GetLength(1));
            Assert.All(mel.Cast<float>(), value => {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            });
        }

        [Fact]
        public void MelPhraseComposerCreatesRealVowelOverlap() {
            var phraseMel = new float[HifiMelExtractor.NMels, 24];
            var weights = new float[24];
            var phones = new[] {
                CreatePreparedPhone(0, "a", start: 0, nominalStart: 0, frames: 12, value: -2f, consonantFrames: 0, crossfadeFrames: 6),
                CreatePreparedPhone(1, "i", start: 12, nominalStart: 12, frames: 12, value: -6f, consonantFrames: 0, crossfadeFrames: 6),
            };

            var report = HifiMelPhraseComposer.Compose(phraseMel, weights, phones, phraseFrames: 24);
            HifiMelPostProcessor.NormalizeWeightedMel(phraseMel, weights);

            Assert.Equal("vowel_to_vowel", report.Boundaries[0].Kind);
            Assert.True(report.UnderlayFrames > 0);
            Assert.True(weights[12] > 1f);
            Assert.InRange(phraseMel[0, 12], -5.9f, -4.0f);
            Assert.NotEqual(-6f, phraseMel[0, 12]);
        }

        [Fact]
        public void MelPhraseComposerUsesContinuousF0ForVowelBoundary() {
            var phraseMel = new float[HifiMelExtractor.NMels, 24];
            var weights = new float[24];
            var f0 = Enumerable.Repeat(220f, 24).ToArray();
            var phones = new[] {
                CreatePreparedPhone(0, "a", start: 0, nominalStart: 0, frames: 12, value: -2f, consonantFrames: 0, crossfadeFrames: 6),
                CreatePreparedPhone(1, "i", start: 12, nominalStart: 12, frames: 12, value: -6f, consonantFrames: 0, crossfadeFrames: 6),
            };

            var report = HifiMelPhraseComposer.Compose(phraseMel, weights, phones, phraseFrames: 24, phraseF0: f0);

            Assert.Equal("vowel_to_vowel_f0_continuous", report.Boundaries[0].Kind);
            Assert.True(report.Boundaries[0].OverlapFrames > 10);
            Assert.Equal("continuous", report.Boundaries[0].F0AwareAdjustment);
        }

        [Fact]
        public void MelPhraseComposerReducesLargeF0JumpOverlap() {
            var phraseMel = new float[HifiMelExtractor.NMels, 24];
            var weights = new float[24];
            var f0 = Enumerable.Range(0, 24)
                .Select(i => i < 12 ? 220f : 440f)
                .ToArray();
            var phones = new[] {
                CreatePreparedPhone(0, "a", start: 0, nominalStart: 0, frames: 12, value: -2f, consonantFrames: 0, crossfadeFrames: 6),
                CreatePreparedPhone(1, "i", start: 12, nominalStart: 12, frames: 12, value: -6f, consonantFrames: 0, crossfadeFrames: 6),
            };

            var report = HifiMelPhraseComposer.Compose(phraseMel, weights, phones, phraseFrames: 24, phraseF0: f0);

            Assert.Equal("vowel_to_vowel_f0_jump", report.Boundaries[0].Kind);
            Assert.True(report.Boundaries[0].OverlapFrames < 10);
            Assert.Equal("jump", report.Boundaries[0].F0AwareAdjustment);
        }

        [Fact]
        public void MelPhraseComposerKeepsConsonantOnsetDominant() {
            var phraseMel = new float[HifiMelExtractor.NMels, 24];
            var weights = new float[24];
            var phones = new[] {
                CreatePreparedPhone(0, "a", start: 0, nominalStart: 0, frames: 12, value: -2f, consonantFrames: 0, crossfadeFrames: 6),
                CreatePreparedPhone(1, "te", start: 12, nominalStart: 12, frames: 12, value: -6f, consonantFrames: 4, crossfadeFrames: 6),
            };

            var report = HifiMelPhraseComposer.Compose(phraseMel, weights, phones, phraseFrames: 24);
            HifiMelPostProcessor.NormalizeWeightedMel(phraseMel, weights);

            Assert.Equal("vowel_to_consonant", report.Boundaries[0].Kind);
            Assert.True(weights[12] > 1f);
            Assert.True(phraseMel[0, 12] < -4.5f);
            Assert.True(phraseMel[0, 12] > -6f);
        }

        [Fact]
        public void MelPhraseComposerFillsDelayedPhoneStartWithPreviousTail() {
            var phraseMel = new float[HifiMelExtractor.NMels, 28];
            var weights = new float[28];
            var phones = new[] {
                CreatePreparedPhone(0, "a", start: 0, nominalStart: 0, frames: 12, value: -2f, consonantFrames: 0, crossfadeFrames: 6),
                CreatePreparedPhone(1, "i", start: 15, nominalStart: 12, frames: 12, value: -6f, consonantFrames: 0, crossfadeFrames: 6),
            };

            var report = HifiMelPhraseComposer.Compose(phraseMel, weights, phones, phraseFrames: 28);
            HifiMelPostProcessor.NormalizeWeightedMel(phraseMel, weights);

            Assert.Equal(3, report.Boundaries[0].GapFrames);
            Assert.True(weights[12] > 0);
            Assert.True(weights[13] > 0);
            Assert.True(weights[14] > 0);
            Assert.Equal(-2f, phraseMel[0, 12], precision: 5);
            Assert.Equal(-2f, phraseMel[0, 14], precision: 5);
            Assert.InRange(phraseMel[0, 15], -5.9f, -4.0f);
        }

        [Fact]
        public void MelPhraseComposerProtectsRapidShortVowelOnset() {
            var phraseMel = new float[HifiMelExtractor.NMels, 20];
            var weights = new float[20];
            var phones = new[] {
                CreatePreparedPhone(0, "a", start: 0, nominalStart: 0, frames: 12, value: -2f, consonantFrames: 0, crossfadeFrames: 6),
                CreatePreparedPhone(1, "i", start: 12, nominalStart: 12, frames: 5, value: -6f, consonantFrames: 0, crossfadeFrames: 6),
            };

            var report = HifiMelPhraseComposer.Compose(phraseMel, weights, phones, phraseFrames: 20);
            HifiMelPostProcessor.NormalizeWeightedMel(phraseMel, weights);

            Assert.Equal("rapid_vowel_to_vowel", report.Boundaries[0].Kind);
            Assert.True(weights[12] > 1f);
            Assert.True(phraseMel[0, 12] < -5.7f);
        }

        [Fact]
        public void MelPhraseComposerUsesLightUnderlayForRapidConsonantOnset() {
            var phraseMel = new float[HifiMelExtractor.NMels, 20];
            var weights = new float[20];
            var phones = new[] {
                CreatePreparedPhone(0, "a", start: 0, nominalStart: 0, frames: 12, value: -2f, consonantFrames: 0, crossfadeFrames: 6),
                CreatePreparedPhone(1, "te", start: 12, nominalStart: 12, frames: 5, value: -6f, consonantFrames: 3, crossfadeFrames: 6),
            };

            var report = HifiMelPhraseComposer.Compose(phraseMel, weights, phones, phraseFrames: 20);
            HifiMelPostProcessor.NormalizeWeightedMel(phraseMel, weights);

            Assert.Equal("rapid_to_consonant", report.Boundaries[0].Kind);
            Assert.InRange(report.Boundaries[0].OverlapFrames, 1, 3);
            Assert.True(weights[12] > 1f);
            Assert.InRange(phraseMel[0, 12], -5.95f, -5.55f);
        }

        [Fact]
        public void VoicingMaskAppliesSoftOnsetAfterConsonantMask() {
            var f0 = Enumerable.Repeat(220f, 8).ToArray();
            var plans = new[] {
                new HifiPhoneDurationPlan {
                    Index = 0,
                    Phoneme = "te",
                    OriginalStartFrame = 0,
                    OriginalDurationFrames = 8,
                    AdjustedStartFrame = 0,
                    AdjustedDurationFrames = 8,
                    ConsonantFrames = 3,
                    TargetConsonantFrames = 3,
                    CanDonateVowel = true,
                },
            };

            HifiMelPostProcessor.ApplyVoicingMask(f0, plans);

            Assert.Equal(0f, f0[0]);
            Assert.Equal(0f, f0[1]);
            Assert.Equal(0f, f0[2]);
            Assert.True(f0[3] > 0f && f0[3] < 220f);
            Assert.True(f0[4] > f0[3]);
            Assert.True(f0[5] > f0[4]);
        }

        [Fact]
        public void VoicedIslandSmootherMorphsStableVowelBoundary() {
            var mel = CreateTwoLevelMel(frames: 24, split: 12, left: -2f, right: -6f);
            var f0 = Enumerable.Repeat(220f, 24).ToArray();
            var plans = new[] {
                new HifiPhoneDurationPlan { Index = 0, Phoneme = "a", OriginalStartFrame = 0, OriginalDurationFrames = 12, AdjustedStartFrame = 0, AdjustedDurationFrames = 12, ConsonantFrames = 0, TargetConsonantFrames = 0, CanDonateVowel = true },
                new HifiPhoneDurationPlan { Index = 1, Phoneme = "i", OriginalStartFrame = 12, OriginalDurationFrames = 12, AdjustedStartFrame = 12, AdjustedDurationFrames = 12, ConsonantFrames = 0, TargetConsonantFrames = 0, CanDonateVowel = true },
            };

            HifiVoicedIslandSmoother.Apply(mel, f0, plans, morphFrames: 4, smoothRadiusFrames: 0, morphStrength: 1f, smoothStrength: 0f, maxF0BridgeFrames: 0, maxF0BridgeCents: 500);

            Assert.InRange(mel[0, 10], -5.9f, -2.1f);
            Assert.InRange(mel[0, 13], -5.9f, -2.1f);
        }

        [Fact]
        public void VoicedIslandSmootherDoesNotSmearConsonantFrames() {
            var mel = CreateTwoLevelMel(frames: 24, split: 12, left: -2f, right: -6f);
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = 12; t < 16; t++) {
                    mel[m, t] = -8f;
                }
            }
            var f0 = Enumerable.Repeat(220f, 24).ToArray();
            var plans = new[] {
                new HifiPhoneDurationPlan { Index = 0, Phoneme = "a", OriginalStartFrame = 0, OriginalDurationFrames = 12, AdjustedStartFrame = 0, AdjustedDurationFrames = 12, ConsonantFrames = 0, TargetConsonantFrames = 0, CanDonateVowel = true },
                new HifiPhoneDurationPlan { Index = 1, Phoneme = "te", OriginalStartFrame = 12, OriginalDurationFrames = 12, AdjustedStartFrame = 12, AdjustedDurationFrames = 12, ConsonantFrames = 4, TargetConsonantFrames = 4, CanDonateVowel = true },
            };

            HifiVoicedIslandSmoother.Apply(mel, f0, plans, morphFrames: 5, smoothRadiusFrames: 0, morphStrength: 1f, smoothStrength: 0f, maxF0BridgeFrames: 0, maxF0BridgeCents: 500);

            Assert.Equal(-8f, mel[0, 12]);
            Assert.Equal(-8f, mel[0, 15]);
            Assert.True(mel[0, 17] > -6f);
        }

        [Fact]
        public void VoicedIslandSmootherBridgesShortInternalF0Gap() {
            var mel = CreateTwoLevelMel(frames: 6, split: 3, left: -2f, right: -2f);
            var f0 = new[] { 220f, 220f, 0f, 0f, 330f, 330f };
            var plans = new[] {
                new HifiPhoneDurationPlan { Index = 0, Phoneme = "a", OriginalStartFrame = 0, OriginalDurationFrames = 6, AdjustedStartFrame = 0, AdjustedDurationFrames = 6, ConsonantFrames = 0, TargetConsonantFrames = 0, CanDonateVowel = true },
            };

            HifiVoicedIslandSmoother.Apply(mel, f0, plans, morphFrames: 0, smoothRadiusFrames: 0, morphStrength: 0f, smoothStrength: 0f, maxF0BridgeFrames: 2, maxF0BridgeCents: 900);

            Assert.True(f0[2] > 0);
            Assert.True(f0[3] > 0);
        }

        [Fact]
        public void VoicedIslandSmootherDoesNotBridgeAcrossRest() {
            var mel = CreateTwoLevelMel(frames: 6, split: 3, left: -2f, right: -2f);
            var f0 = new[] { 220f, 220f, 0f, 0f, 330f, 330f };
            var plans = new[] {
                new HifiPhoneDurationPlan { Index = 0, Phoneme = "a", OriginalStartFrame = 0, OriginalDurationFrames = 2, AdjustedStartFrame = 0, AdjustedDurationFrames = 2, ConsonantFrames = 0, TargetConsonantFrames = 0, CanDonateVowel = true },
                new HifiPhoneDurationPlan { Index = 1, Phoneme = "r", OriginalStartFrame = 2, OriginalDurationFrames = 2, AdjustedStartFrame = 2, AdjustedDurationFrames = 2, ConsonantFrames = 0, TargetConsonantFrames = 0, CanDonateVowel = false },
                new HifiPhoneDurationPlan { Index = 2, Phoneme = "a", OriginalStartFrame = 4, OriginalDurationFrames = 2, AdjustedStartFrame = 4, AdjustedDurationFrames = 2, ConsonantFrames = 0, TargetConsonantFrames = 0, CanDonateVowel = true },
            };

            HifiVoicedIslandSmoother.Apply(mel, f0, plans, morphFrames: 0, smoothRadiusFrames: 0, morphStrength: 0f, smoothStrength: 0f, maxF0BridgeFrames: 2, maxF0BridgeCents: 900);

            Assert.Equal(0f, f0[2]);
            Assert.Equal(0f, f0[3]);
        }

        [Fact]
        public void VoicedIslandSmootherDoesNotBridgeConsonantF0Gap() {
            var mel = CreateTwoLevelMel(frames: 4, split: 2, left: -2f, right: -2f);
            var f0 = new[] { 220f, 0f, 330f, 330f };
            var plans = new[] {
                new HifiPhoneDurationPlan { Index = 0, Phoneme = "ka", OriginalStartFrame = 0, OriginalDurationFrames = 4, AdjustedStartFrame = 0, AdjustedDurationFrames = 4, ConsonantFrames = 2, TargetConsonantFrames = 2, CanDonateVowel = true },
            };

            HifiVoicedIslandSmoother.Apply(mel, f0, plans, morphFrames: 0, smoothRadiusFrames: 0, morphStrength: 0f, smoothStrength: 0f, maxF0BridgeFrames: 2, maxF0BridgeCents: 900, shortPhoneFrames: 0, minStableFrames: 1);

            Assert.Equal(0f, f0[1]);
        }

        [Fact]
        public void BoundaryEnergyMatcherConservativeIsLocal() {
            var mel = new float[HifiMelExtractor.NMels, 64];
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = 0; t < 64; t++) {
                    mel[m, t] = t < 32 ? -2f : -4f;
                }
            }
            var original = (float[,])mel.Clone();
            var boundaries = new[] {
                new HifiBoundaryMetadata { Index = 0, LeftPhoneIndex = 0, RightPhoneIndex = 1, LeftPhone = "a", RightPhone = "i", Frame = 32 },
            };
            var plans = new[] {
                new HifiPhoneDurationPlan { Index = 0, Phoneme = "a", OriginalStartFrame = 0, OriginalDurationFrames = 32, AdjustedStartFrame = 0, AdjustedDurationFrames = 32, ConsonantFrames = 0, TargetConsonantFrames = 0, CanDonateVowel = true },
                new HifiPhoneDurationPlan { Index = 1, Phoneme = "i", OriginalStartFrame = 32, OriginalDurationFrames = 32, AdjustedStartFrame = 32, AdjustedDurationFrames = 32, ConsonantFrames = 0, TargetConsonantFrames = 0, CanDonateVowel = true },
            };
            HifiBoundaryEnergyMatcher.ApplyConservative(mel, boundaries, plans, windowFrames: 12, maxGainDb: 0.8);

            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                Assert.Equal(original[m, 5], mel[m, 5]); // outside boundary window should remain
                Assert.True(mel[m, 38] > original[m, 38]); // right side is locally lifted
            }
            Assert.All(mel.Cast<float>(), value => {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            });
        }

        [Fact]
        public void BoundaryEnergyMatcherSkipsProtectedConsonant() {
            var mel = new float[HifiMelExtractor.NMels, 64];
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = 0; t < 64; t++) {
                    mel[m, t] = t < 32 ? -2f : -4f;
                }
            }
            var original = (float[,])mel.Clone();
            var boundaries = new[] {
                new HifiBoundaryMetadata { Index = 0, LeftPhoneIndex = 0, RightPhoneIndex = 1, LeftPhone = "a", RightPhone = "te", Frame = 32 },
            };
            var plans = new[] {
                new HifiPhoneDurationPlan { Index = 0, Phoneme = "a", OriginalStartFrame = 0, OriginalDurationFrames = 32, AdjustedStartFrame = 0, AdjustedDurationFrames = 32, ConsonantFrames = 0, TargetConsonantFrames = 0, CanDonateVowel = true },
                new HifiPhoneDurationPlan { Index = 1, Phoneme = "te", OriginalStartFrame = 32, OriginalDurationFrames = 32, AdjustedStartFrame = 32, AdjustedDurationFrames = 32, ConsonantFrames = 4, TargetConsonantFrames = 4, CanDonateVowel = true },
            };

            HifiBoundaryEnergyMatcher.ApplyConservative(mel, boundaries, plans, windowFrames: 12, maxGainDb: 0.8);

            Assert.Equal(original[0, 33], mel[0, 33]);
            Assert.Equal(original[0, 38], mel[0, 38]);
        }

        [Fact]
        public void HeadAttackDoesNotFadeProtectedConsonant() {
            var mel = new float[HifiMelExtractor.NMels, 16];
            var f0 = Enumerable.Repeat(220f, 16).ToArray();
            var plans = new[] {
                new HifiPhoneDurationPlan { Index = 0, Phoneme = "te", OriginalStartFrame = 0, OriginalDurationFrames = 16, AdjustedStartFrame = 0, AdjustedDurationFrames = 16, ConsonantFrames = 4, TargetConsonantFrames = 4, CanDonateVowel = true },
            };

            HifiMelPostProcessor.ApplyHeadAttack(mel, f0, plans, attackFrames: 4, attackGainDb: -9, f0FadeFrames: 2);

            Assert.Equal(0f, mel[0, 0]);
            Assert.Equal(0f, mel[0, 3]);
            Assert.True(mel[0, 4] < 0f);
            Assert.Equal(220f, f0[0]);
            Assert.Equal(220f, f0[3]);
            Assert.True(f0[4] < 220f);
        }

        [Fact]
        public void AudioEnvelopeNormalizerBalancesPhraseAndAppliesMakeup() {
            int frames = 16;
            var samples = new float[frames * HifiF0Builder.HopSize];
            for (int t = 0; t < frames; t++) {
                float amp = t >= 6 && t <= 9 ? 0.6f : 0.1f;
                for (int i = t * HifiF0Builder.HopSize; i < (t + 1) * HifiF0Builder.HopSize; i++) {
                    samples[i] = amp;
                }
            }
            var original = samples.ToArray();
            var features = CreateFlatFeatures(frames, voiced: true);

            HifiAudioEnvelopeNormalizer.Apply(samples, features, maxAdjustDb: 3, deadbandDb: 1.0, strength: 1);

            Assert.True(FrameRms(samples, 8) < FrameRms(original, 8));
            Assert.True(FrameRms(samples, 0) > FrameRms(original, 0));
            Assert.True(samples.Max(sample => Math.Abs(sample)) <= HifiNeuralConfig.OutputPeakLimit + 1e-3);
        }

        [Fact]
        public void AudioEnvelopeNormalizerAddsMildExciterColor() {
            int frames = 12;
            var samples = new float[frames * HifiF0Builder.HopSize];
            for (int t = 0; t < frames; t++) {
                float amp = 0.22f;
                for (int i = t * HifiF0Builder.HopSize; i < (t + 1) * HifiF0Builder.HopSize; i++) {
                    samples[i] = amp;
                }
            }
            var original = samples.ToArray();
            var features = CreateFlatFeatures(frames, voiced: true);

            HifiAudioEnvelopeNormalizer.Apply(samples, features, maxAdjustDb: 2.5, deadbandDb: 1.0, strength: 0.8);

            double diff = 0;
            for (int i = 0; i < samples.Length; i++) {
                diff += Math.Abs(samples[i] - original[i]);
            }
            Assert.True(diff > 1e-3);
            Assert.True(samples.Max(sample => Math.Abs(sample)) <= HifiNeuralConfig.OutputPeakLimit + 1e-3);
        }

        [Fact]
        public void PitchMelCompensatorOnlyAdjustsVoicedVowelRegion() {
            var mel = new float[HifiMelExtractor.NMels, 16];
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = 0; t < 16; t++) {
                    mel[m, t] = -4f;
                }
            }
            var original = (float[,])mel.Clone();
            var f0 = Enumerable.Repeat(440f, 16).ToArray();

            var report = HifiPitchMelCompensator.Apply(mel, f0, "a", protectedFrames: 2, sourceF0: 220);

            Assert.Equal(string.Empty, report.SkippedReason);
            Assert.True(report.GainCutDb > 0);
            Assert.Equal(original[0, 0], mel[0, 0]);
            Assert.Equal(original[0, 2], mel[0, 2]);
            Assert.True(MeanEnergy(mel, report.StartFrame, report.EndFrame) < MeanEnergy(original, report.StartFrame, report.EndFrame));
            Assert.All(mel.Cast<float>(), value => {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            });
        }

        [Fact]
        public void PitchMelCompensatorNeverAmplifiesMelBins() {
            var mel = new float[HifiMelExtractor.NMels, 16];
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = 0; t < 16; t++) {
                    mel[m, t] = -4f;
                }
            }
            var original = (float[,])mel.Clone();
            var f0 = Enumerable.Repeat(110f, 16).ToArray();

            var report = HifiPitchMelCompensator.Apply(mel, f0, "a", protectedFrames: 0, sourceF0: 220);

            Assert.Equal(string.Empty, report.SkippedReason);
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = report.StartFrame; t < report.EndFrame; t++) {
                    Assert.True(mel[m, t] <= original[m, t] + 1e-6f);
                }
            }
        }

        [Fact]
        public void SourceLeadingSanitizerSkipsMismatchedVcvAtPhraseStart() {
            bool original = Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer;
            try {
                Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer = true;
                var report = HifiSourceLeadingSanitizer.Analyze(
                    "a ka",
                    previousPhoneme: null,
                    isFirstOrAfterSilence: true,
                    sourcePreutterMs: 100,
                    maxExtraStartOffsetFrames: 20);
                var samples = Enumerable.Repeat(1f, 10000).ToArray();
                var sanitized = HifiSourceLeadingSanitizer.Apply(samples, report);

                Assert.True(report.Applied);
                Assert.Equal("leading_context_after_silence", report.Reason);
                Assert.Equal("a", report.LeadingToken);
                Assert.Equal(0, report.ExtraStartOffsetFrames);
                Assert.True(sanitized.Length < samples.Length);
                Assert.True(sanitized[0] < 1f);
            } finally {
                Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer = original;
            }
        }

        [Fact]
        public void SourceLeadingSanitizerTrimsMatchedConnectedVcv() {
            bool original = Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer;
            try {
                Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer = true;
                var report = HifiSourceLeadingSanitizer.Analyze(
                    "a ka",
                    previousPhoneme: "a",
                    isFirstOrAfterSilence: false,
                    sourcePreutterMs: 100,
                    maxExtraStartOffsetFrames: 20);

                Assert.True(report.Applied);
                Assert.Equal("connected_vcv_trim_match", report.Reason);
                Assert.True(report.ExtraSkipMs > 0);
            } finally {
                Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer = original;
            }
        }

        [Fact]
        public void SourceLeadingSanitizerTrimsConnectedMismatchConservatively() {
            bool original = Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer;
            try {
                Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer = true;
                var report = HifiSourceLeadingSanitizer.Analyze(
                    "i ka",
                    previousPhoneme: "a",
                    isFirstOrAfterSilence: false,
                    sourcePreutterMs: 100,
                    maxExtraStartOffsetFrames: 20);

                Assert.True(report.Applied);
                Assert.Equal("connected_vcv_trim", report.Reason);
                Assert.True(report.ExtraSkipMs > 0);
            } finally {
                Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer = original;
            }
        }

        [Fact]
        public void SourceLeadingSanitizerTrimsConnectedJapaneseVcv() {
            bool original = Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer;
            try {
                Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer = true;
                var report = HifiSourceLeadingSanitizer.Analyze(
                    "a し",
                    previousPhoneme: "か",
                    isFirstOrAfterSilence: false,
                    sourcePreutterMs: 100,
                    maxExtraStartOffsetFrames: 20);

                Assert.True(report.Applied);
                Assert.Equal("connected_vcv_trim_match", report.Reason);
                Assert.Equal("a", report.PreviousStableToken);
            } finally {
                Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer = original;
            }
        }

        [Fact]
        public void SourceLeadingSanitizerSkipsWhenNoTargetRoom() {
            bool original = Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer;
            try {
                Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer = true;
                var report = HifiSourceLeadingSanitizer.Analyze(
                    "a ka",
                    previousPhoneme: null,
                    isFirstOrAfterSilence: true,
                    sourcePreutterMs: 100,
                    maxExtraStartOffsetFrames: 0);

                Assert.False(report.Applied);
                Assert.Equal("no_target_room_for_sanitize", report.Reason);
            } finally {
                Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer = original;
            }
        }

        [Fact]
        public void AudioEnvelopeNormalizerSkipsUnvoicedFrames() {
            int frames = 8;
            var samples = Enumerable.Repeat(0.5f, frames * HifiF0Builder.HopSize).ToArray();
            var original = samples.ToArray();
            var features = CreateFlatFeatures(frames, voiced: false);

            HifiAudioEnvelopeNormalizer.Apply(samples, features, maxAdjustDb: 3, deadbandDb: 1.0, strength: 1);

            Assert.Equal(original, samples);
        }

        [Fact]
        public void F0BoundarySmoothingSoftensVoicedNoteJump() {
            var f0 = new[] { 220f, 220f, 220f, 220f, 220f, 440f, 440f, 440f, 440f, 440f };
            var metadata = CreateBoundaryMetadata(boundaryFrame: 5);

            HifiMelPostProcessor.SmoothInternalF0Boundaries(f0, metadata, radiusFrames: 2, maxJumpCents: 35);

            Assert.True(f0[4] > 220f);
            Assert.True(f0[5] < 440f);
            Assert.True(f0.All(value => value > 0 && !float.IsNaN(value) && !float.IsInfinity(value)));
        }

        [Fact]
        public void InternalClickSuppressorReducesNoteBoundaryJump() {
            int sampleRate = HifiMelExtractor.SampleRate;
            var samples = new float[sampleRate / 10];
            int center = sampleRate / 20;
            for (int i = 0; i < samples.Length; i++) {
                samples[i] = i < center ? -0.4f : 0.4f;
            }
            var metadata = new HifiPhraseMetadata {
                SampleRate = sampleRate,
                HopSize = HifiF0Builder.HopSize,
                FrameMs = HifiF0Builder.FrameMs,
                PhraseStartMs = 0,
                EstimatedLengthMs = samples.Length * 1000.0 / sampleRate,
            };
            metadata.Notes.Add(new HifiNoteMetadata { Index = 0, Lyric = "a", PositionMs = 0, DurationMs = 50 });
            metadata.Notes.Add(new HifiNoteMetadata { Index = 1, Lyric = "a", PositionMs = center * 1000.0 / sampleRate, DurationMs = 50 });
            double before = Math.Abs(samples[center] - samples[center - 1]);

            HifiInternalClickSuppressor.Apply(samples, metadata, sampleRate, windowMs: 1.5, thresholdRatio: 0.35);
            double after = Math.Abs(samples[center] - samples[center - 1]);

            Assert.True(after < before);
        }

        [Fact]
        public void VoicedOnsetDipRepairLiftsQuietStretchedPhoneDip() {
            var mel = CreateTwoLevelMel(frames: 10, split: 10, left: -5f, right: -5f);
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                mel[m, 4] = -8f;
                mel[m, 5] = -8f;
            }
            var f0 = Enumerable.Repeat(220f, 10).ToArray();
            var plans = new List<HifiPhoneDurationPlan> {
                new HifiPhoneDurationPlan {
                    Index = 0,
                    Phoneme = "a",
                    OriginalStartFrame = 4,
                    OriginalDurationFrames = 5,
                    AdjustedStartFrame = 4,
                    AdjustedDurationFrames = 5,
                    ConsonantFrames = 0,
                    TargetConsonantFrames = 0,
                    CanDonateVowel = true,
                },
            };
            var diagnostics = new List<HifiPhoneFeatureDiagnostic> {
                new HifiPhoneFeatureDiagnostic {
                    Index = 0,
                    Phoneme = "a",
                    StartFrame = 4,
                    FrameCount = 5,
                    SourceOnsetPeak = 0.002,
                    SourceOnsetRms = 0.001,
                    SourceOnsetMaxJump = 0.0005,
                    SourceMelOnsetDelta = 0.1,
                    StretchedMelOnsetMean = -8,
                    StretchedMelAfterMean = -5,
                    StretchedMelOnsetDelta = 3,
                    TargetF0AtStart = 220,
                    TargetF0AfterStart = 220,
                },
            };
            float before = mel[0, 4];

            HifiMelPostProcessor.RepairVoicedOnsetDips(
                mel,
                f0,
                plans,
                diagnostics,
                repairFrames: 3,
                minDip: 0.75,
                maxLift: 0.9);

            Assert.True(mel[0, 4] > before);
            Assert.Equal(-5f, mel[0, 8]);
        }

        [Fact]
        public void VoicedOnsetDipRepairSkipsRealConsonantTransient() {
            var mel = CreateTwoLevelMel(frames: 10, split: 10, left: -5f, right: -5f);
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                mel[m, 4] = -8f;
                mel[m, 5] = -8f;
            }
            var f0 = Enumerable.Repeat(220f, 10).ToArray();
            var plans = new List<HifiPhoneDurationPlan> {
                new HifiPhoneDurationPlan {
                    Index = 0,
                    Phoneme = "ta",
                    OriginalStartFrame = 4,
                    OriginalDurationFrames = 5,
                    AdjustedStartFrame = 4,
                    AdjustedDurationFrames = 5,
                    ConsonantFrames = 3,
                    TargetConsonantFrames = 3,
                    CanDonateVowel = false,
                },
            };
            var diagnostics = new List<HifiPhoneFeatureDiagnostic> {
                new HifiPhoneFeatureDiagnostic {
                    Index = 0,
                    Phoneme = "ta",
                    StartFrame = 4,
                    FrameCount = 5,
                    SourceOnsetPeak = 0.12,
                    SourceOnsetRms = 0.05,
                    SourceOnsetMaxJump = 0.03,
                    SourceMelOnsetDelta = 0.2,
                    StretchedMelOnsetMean = -8,
                    StretchedMelAfterMean = -5,
                    StretchedMelOnsetDelta = 3,
                    TargetF0AtStart = 220,
                    TargetF0AfterStart = 220,
                },
            };

            HifiMelPostProcessor.RepairVoicedOnsetDips(
                mel,
                f0,
                plans,
                diagnostics,
                repairFrames: 3,
                minDip: 0.75,
                maxLift: 0.9);

            Assert.Equal(-8f, mel[0, 4]);
        }

        [Fact]
        public void ClickDiagnosticFlagsPhoneBoundaryArtifacts() {
            var features = CreateFlatFeatures(frames: 4, voiced: false);
            features.Metadata.Phones.Add(new HifiPhoneMetadata {
                Index = 0,
                Phoneme = "a",
                StartFrame = 1,
                FrameCount = 2,
            });
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                features.Mel[m, 0] = -8f;
                features.Mel[m, 1] = -2f;
            }
            features.F0[0] = 0;
            features.F0[1] = 220;
            var samples = new float[features.Frames * HifiF0Builder.HopSize];
            int center = HifiF0Builder.HopSize;
            samples[center - 1] = -0.5f;
            samples[center] = 0.5f;

            var report = HifiClickDiagnostic.Analyze(features, samples);

            Assert.Equal(1, report.SuspectPhones);
            Assert.Contains("wav_jump", report.Phones[0].SuspectReason);
            Assert.Contains("phrase_mel_boundary_delta", report.Phones[0].SuspectReason);
            Assert.Contains("f0_voiced_onset", report.Phones[0].SuspectReason);
        }

        [Fact]
        public void ClickDiagnosticCapturesSourceOnsetStats() {
            var source = Enumerable.Repeat(0f, 4096).ToArray();
            source[1] = 0.7f;
            source[2] = -0.7f;
            var sourceMel = CreateTwoLevelMel(frames: 8, split: 2, left: -8f, right: -3f);
            var stretchedMel = CreateTwoLevelMel(frames: 8, split: 2, left: -7f, right: -3f);
            var f0 = Enumerable.Repeat(220f, 8).ToArray();

            var diagnostic = HifiClickDiagnostic.BuildPhoneFeatureDiagnostic(
                index: 2,
                phoneme: "a",
                startFrame: 4,
                frameCount: 8,
                source: source,
                sourceMel: sourceMel,
                stretchedMel: stretchedMel,
                targetF0: f0,
                leadingSanitizerReason: "disabled");

            Assert.Equal(2, diagnostic.Index);
            Assert.True(diagnostic.SourceOnsetMaxJump > 1.0);
            Assert.True(diagnostic.SourceMelOnsetDelta > 1.0);
            Assert.Equal("disabled", diagnostic.LeadingSanitizerReason);
        }

        [Fact]
        public void VocoderReportsMissingExplicitModel() {
            string missing = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "missing.onnx");
            var ex = Assert.Throws<FileNotFoundException>(() => new HifiOnnxVocoder(missing));
            Assert.Contains("Hifi ONNX vocoder model not found", ex.Message);
        }

        [Fact]
        public void VocoderRunsDummyInferenceWhenModelExists() {
            if (!HifiOnnxVocoder.TryResolveModelPath(out _, out _)) {
                return;
            }
            using var vocoder = new HifiOnnxVocoder();
            var samples = vocoder.InferDummy(frames: 8);
            Assert.NotEmpty(samples);
            foreach (var sample in samples) {
                Assert.False(float.IsNaN(sample));
                Assert.False(float.IsInfinity(sample));
            }
        }

        [Fact]
        public void DebugDumpRoundTripsFeatures() {
            string dir = Path.Combine(Path.GetTempPath(), "hifi-debug-test-" + Guid.NewGuid().ToString("N"));
            try {
                var metadata = new HifiPhraseMetadata {
                    SampleRate = HifiMelExtractor.SampleRate,
                    HopSize = HifiF0Builder.HopSize,
                    FrameMs = HifiF0Builder.FrameMs,
                    EstimatedLengthMs = HifiF0Builder.FrameMs * 3,
                };
                metadata.Notes.Add(new HifiNoteMetadata {
                    Index = 0,
                    Lyric = "a",
                    Tone = 60,
                    AdjustedTone = 60,
                    DurationMs = HifiF0Builder.FrameMs * 3,
                });
                metadata.Phones.Add(new HifiPhoneMetadata {
                    Index = 0,
                    Phoneme = "a",
                    Tone = 60,
                    FrameCount = 3,
                });
                metadata.Boundaries.Add(new HifiBoundaryMetadata {
                    Index = 0,
                    LeftPhoneIndex = 0,
                    RightPhoneIndex = 1,
                    LeftPhone = "a",
                    RightPhone = "i",
                    Frame = 1,
                });

                var mel = new float[HifiMelExtractor.NMels, 3];
                for (int m = 0; m < mel.GetLength(0); m++) {
                    for (int t = 0; t < mel.GetLength(1); t++) {
                        mel[m, t] = m * 0.01f + t;
                    }
                }
                var f0 = new[] { 220f, 221f, 222f };
                var features = new HifiPhraseFeatures {
                    Mel = mel,
                    F0 = f0,
                    Metadata = metadata,
                };

                HifiDebugExporter.ExportToDirectory(dir, features);
                var loaded = HifiDebugExporter.Load(dir);

                Assert.Equal(HifiMelExtractor.NMels, loaded.MelBins);
                Assert.Equal(3, loaded.Frames);
                Assert.Equal(f0, loaded.F0);
                Assert.Equal(metadata.Notes.Count, loaded.Metadata.Notes.Count);
                Assert.Equal(metadata.Phones.Count, loaded.Metadata.Phones.Count);
                Assert.Equal(metadata.Boundaries.Count, loaded.Metadata.Boundaries.Count);
                for (int m = 0; m < mel.GetLength(0); m++) {
                    for (int t = 0; t < mel.GetLength(1); t++) {
                        Assert.Equal(mel[m, t], loaded.Mel[m, t]);
                    }
                }
            } finally {
                if (Directory.Exists(dir)) {
                    Directory.Delete(dir, recursive: true);
                }
            }
        }

        [Fact]
        public void TransitionDatasetExporterWritesExpectedFiles() {
            string dir = Path.Combine(Path.GetTempPath(), "hifi-transition-test-" + Guid.NewGuid().ToString("N"));
            try {
                var metadata = new HifiPhraseMetadata {
                    SampleRate = HifiMelExtractor.SampleRate,
                    HopSize = HifiF0Builder.HopSize,
                    FrameMs = HifiF0Builder.FrameMs,
                    EstimatedLengthMs = HifiF0Builder.FrameMs * 5,
                };
                metadata.Boundaries.Add(new HifiBoundaryMetadata {
                    Index = 2,
                    LeftPhoneIndex = 0,
                    RightPhoneIndex = 1,
                    LeftPhone = "a",
                    RightPhone = "i",
                    Frame = 2,
                    PositionMs = HifiF0Builder.FrameMs * 2,
                });
                var mel = new float[HifiMelExtractor.NMels, 5];
                var f0 = new[] { 100f, 110f, 120f, 130f, 140f };
                var features = new HifiPhraseFeatures {
                    Mel = mel,
                    F0 = f0,
                    Metadata = metadata,
                };

                HifiTransitionDatasetExporter.ExportToDirectory(dir, features, radiusFrames: 1);

                string prefix = Path.Combine(dir, "transition-0002");
                Assert.True(File.Exists(prefix + ".mel.f32"));
                Assert.True(File.Exists(prefix + ".f0.f32"));
                Assert.True(File.Exists(prefix + ".mask.f32"));
                Assert.True(File.Exists(prefix + ".json"));
                Assert.Equal(HifiMelExtractor.NMels * 3 * sizeof(float), new FileInfo(prefix + ".mel.f32").Length);
                Assert.Equal(3 * sizeof(float), new FileInfo(prefix + ".f0.f32").Length);
                Assert.Equal(3 * sizeof(float), new FileInfo(prefix + ".mask.f32").Length);
                string json = File.ReadAllText(prefix + ".json");
                Assert.Contains("\"LeftPhone\": \"a\"", json);
                Assert.Contains("\"RightPhone\": \"i\"", json);
                Assert.Contains("\"WindowFrames\": 3", json);
            } finally {
                if (Directory.Exists(dir)) {
                    Directory.Delete(dir, recursive: true);
                }
            }
        }

        static float[,] CreateTestMel(int frames) {
            var mel = new float[HifiMelExtractor.NMels, frames];
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = 0; t < frames; t++) {
                    mel[m, t] = 0.1f + m * 0.001f + t * 0.01f;
                }
            }
            return mel;
        }

        static float[,] CreateLowHighLowMel(int frames, float low, float high) {
            var mel = new float[HifiMelExtractor.NMels, frames];
            int highStart = frames / 4;
            int highEnd = frames - frames / 5;
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = 0; t < frames; t++) {
                    mel[m, t] = t >= highStart && t < highEnd ? high : low;
                }
            }
            return mel;
        }

        static HifiPreparedPhoneMel CreatePreparedPhone(
            int index,
            string phoneme,
            int start,
            int nominalStart,
            int frames,
            float value,
            int consonantFrames,
            int crossfadeFrames) {
            var mel = new float[HifiMelExtractor.NMels, frames];
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = 0; t < frames; t++) {
                    mel[m, t] = value;
                }
            }
            return new HifiPreparedPhoneMel {
                Index = index,
                Phoneme = phoneme,
                Mel = mel,
                StartFrame = start,
                NominalStartFrame = nominalStart,
                TargetConsonantFrames = consonantFrames,
                CrossfadeFrames = crossfadeFrames,
            };
        }

        static float[,] CreateTwoLevelMel(int frames, int split, float left, float right) {
            var mel = new float[HifiMelExtractor.NMels, frames];
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = 0; t < frames; t++) {
                    mel[m, t] = t < split ? left : right;
                }
            }
            return mel;
        }

        static List<HifiPhoneDurationPlan> CreateDurationPlans() {
            return new List<HifiPhoneDurationPlan> {
                new HifiPhoneDurationPlan {
                    Index = 0,
                    Phoneme = "a",
                    OriginalStartFrame = 0,
                    OriginalDurationFrames = 20,
                    AdjustedStartFrame = 0,
                    AdjustedDurationFrames = 20,
                    ConsonantFrames = 1,
                    TargetConsonantFrames = 1,
                    CanDonateVowel = true,
                },
                new HifiPhoneDurationPlan {
                    Index = 1,
                    Phoneme = "ka",
                    OriginalStartFrame = 20,
                    OriginalDurationFrames = 3,
                    AdjustedStartFrame = 20,
                    AdjustedDurationFrames = 3,
                    ConsonantFrames = 2,
                    TargetConsonantFrames = 2,
                    CanDonateVowel = false,
                },
                new HifiPhoneDurationPlan {
                    Index = 2,
                    Phoneme = "a",
                    OriginalStartFrame = 23,
                    OriginalDurationFrames = 17,
                    AdjustedStartFrame = 23,
                    AdjustedDurationFrames = 17,
                    ConsonantFrames = 1,
                    TargetConsonantFrames = 1,
                    CanDonateVowel = true,
                },
            };
        }

        static HifiPhraseFeatures CreateFlatFeatures(int frames, bool voiced) {
            var mel = new float[HifiMelExtractor.NMels, frames];
            for (int m = 0; m < HifiMelExtractor.NMels; m++) {
                for (int t = 0; t < frames; t++) {
                    mel[m, t] = -4f;
                }
            }
            var f0 = Enumerable.Repeat(voiced ? 220f : 0f, frames).ToArray();
            return new HifiPhraseFeatures {
                Mel = mel,
                F0 = f0,
                Metadata = new HifiPhraseMetadata {
                    SampleRate = HifiMelExtractor.SampleRate,
                    HopSize = HifiF0Builder.HopSize,
                    FrameMs = HifiF0Builder.FrameMs,
                    EstimatedLengthMs = frames * HifiF0Builder.FrameMs,
                },
            };
        }

        static HifiPhraseMetadata CreateBoundaryMetadata(int boundaryFrame) {
            var metadata = new HifiPhraseMetadata {
                SampleRate = HifiMelExtractor.SampleRate,
                HopSize = HifiF0Builder.HopSize,
                FrameMs = HifiF0Builder.FrameMs,
                PhraseStartMs = 0,
                EstimatedLengthMs = HifiF0Builder.FrameMs * 10,
            };
            metadata.Notes.Add(new HifiNoteMetadata {
                Index = 0,
                Lyric = "a",
                PositionMs = 0,
                DurationMs = boundaryFrame * HifiF0Builder.FrameMs,
            });
            metadata.Notes.Add(new HifiNoteMetadata {
                Index = 1,
                Lyric = "i",
                PositionMs = boundaryFrame * HifiF0Builder.FrameMs,
                DurationMs = (10 - boundaryFrame) * HifiF0Builder.FrameMs,
            });
            return metadata;
        }

        static double FrameRms(float[] samples, int frame) {
            int start = frame * HifiF0Builder.HopSize;
            int end = Math.Min(samples.Length, start + HifiF0Builder.HopSize);
            double sum = 0;
            for (int i = start; i < end; i++) {
                sum += samples[i] * samples[i];
            }
            return Math.Sqrt(sum / Math.Max(1, end - start));
        }

        static float MeanEnergy(float[,] mel, int start, int end) {
            double sum = 0;
            int count = 0;
            int bins = mel.GetLength(0);
            for (int t = start; t < end; t++) {
                for (int m = 0; m < bins; m++) {
                    sum += mel[m, t];
                    count++;
                }
            }
            return count > 0 ? (float)(sum / count) : 0f;
        }
    }
}
