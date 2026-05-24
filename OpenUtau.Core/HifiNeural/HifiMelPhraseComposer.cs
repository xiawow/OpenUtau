using System;
using System.Collections.Generic;
using System.Linq;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiPreparedPhoneMel {
        public int Index { get; init; }
        public string Phoneme { get; init; } = string.Empty;
        public required float[,] Mel { get; init; }
        public int NominalStartFrame { get; init; }
        public int StartFrame { get; init; }
        public int TargetConsonantFrames { get; init; }
        public int CrossfadeFrames { get; init; }

        public int FrameCount => Mel.GetLength(1);
        public int EndFrame => StartFrame + FrameCount;
    }

    public sealed class HifiMelBoundaryCompositionReport {
        public int BoundaryIndex { get; init; }
        public string Kind { get; init; } = string.Empty;
        public string LeftPhoneme { get; init; } = string.Empty;
        public string RightPhoneme { get; init; } = string.Empty;
        public int NominalBoundaryFrame { get; init; }
        public int LeftEndFrame { get; init; }
        public int RightStartFrame { get; init; }
        public int GapFrames { get; init; }
        public int OverlapFrames { get; init; }
        public double MelDeltaBefore { get; init; }
        public double EnergyDeltaBefore { get; init; }
        public float LeftF0 { get; init; }
        public float RightF0 { get; init; }
        public double F0JumpCents { get; init; }
        public double F0MaxSlopeCents { get; init; }
        public string F0AwareAdjustment { get; init; } = string.Empty;
        public float RightMinWeight { get; init; }
        public float UnderlayMaxWeight { get; init; }
        public string SkippedReason { get; init; } = string.Empty;
    }

    public sealed class HifiMelCompositionReport {
        public int PhoneCount { get; init; }
        public int PhraseFrames { get; init; }
        public int ClippedFrames { get; init; }
        public int GapFramesBeforeUnderlay { get; init; }
        public int UnderlayFrames { get; init; }
        public IReadOnlyList<HifiMelBoundaryCompositionReport> Boundaries { get; init; } =
            Array.Empty<HifiMelBoundaryCompositionReport>();
    }

    public static class HifiMelPhraseComposer {
        const int MaxBoundaryOverlapFrames = 14;

        readonly record struct BoundaryRule(
            string Kind,
            int OverlapFrames,
            float RightMinWeight,
            float PreStartUnderlayWeight,
            float PostStartUnderlayWeight,
            bool FadeRight);

        readonly record struct BoundaryF0Context(
            bool IsValid,
            float LeftF0,
            float RightF0,
            double JumpCents,
            double MaxSlopeCents);

        public static HifiMelCompositionReport Compose(
            float[,] phraseMel,
            float[] weights,
            IReadOnlyList<HifiPreparedPhoneMel> inputPhones,
            int phraseFrames,
            float[]? phraseF0 = null) {
            Array.Clear(phraseMel, 0, phraseMel.Length);
            Array.Clear(weights, 0, weights.Length);
            var phones = inputPhones
                .Where(phone => phone.FrameCount > 0)
                .OrderBy(phone => phone.NominalStartFrame)
                .ThenBy(phone => phone.Index)
                .ToArray();
            if (phones.Length == 0) {
                return new HifiMelCompositionReport {
                    PhoneCount = 0,
                    PhraseFrames = phraseFrames,
                };
            }

            var phoneWeights = phones
                .Select(phone => Enumerable.Repeat(1f, phone.FrameCount).ToArray())
                .ToArray();
            var underlays = new List<MelUnderlay>();
            var boundaryReports = new List<HifiMelBoundaryCompositionReport>();
            int gapFramesBeforeUnderlay = 0;

            for (int i = 1; i < phones.Length; i++) {
                var left = phones[i - 1];
                var right = phones[i];
                int nominalBoundary = right.NominalStartFrame;
                int rightDelayFrames = Math.Max(0, right.StartFrame - nominalBoundary);
                int physicalGapFrames = Math.Max(0, right.StartFrame - left.EndFrame);
                int gapFrames = Math.Max(rightDelayFrames, physicalGapFrames);
                gapFramesBeforeUnderlay += gapFrames;
                var f0Context = BoundaryF0At(phraseF0, left, right, nominalBoundary);
                var rule = ResolveBoundaryRule(left, right, f0Context);
                double melDeltaBefore = BoundaryMelDelta(left, right);
                double energyDeltaBefore = BoundaryEnergyDelta(left, right);

                if (rule.OverlapFrames <= 0) {
                    boundaryReports.Add(new HifiMelBoundaryCompositionReport {
                        BoundaryIndex = i - 1,
                        Kind = rule.Kind,
                        LeftPhoneme = left.Phoneme,
                        RightPhoneme = right.Phoneme,
                        NominalBoundaryFrame = nominalBoundary,
                        LeftEndFrame = left.EndFrame,
                        RightStartFrame = right.StartFrame,
                        GapFrames = gapFrames,
                        OverlapFrames = 0,
                        MelDeltaBefore = melDeltaBefore,
                        EnergyDeltaBefore = energyDeltaBefore,
                        LeftF0 = f0Context.LeftF0,
                        RightF0 = f0Context.RightF0,
                        F0JumpCents = f0Context.JumpCents,
                        F0MaxSlopeCents = f0Context.MaxSlopeCents,
                        F0AwareAdjustment = F0AwareAdjustmentName(f0Context, rule.Kind),
                        SkippedReason = "boundary_not_blendable",
                    });
                    continue;
                }

                int rightOverlap = Math.Min(rule.OverlapFrames, right.FrameCount);
                if (rule.FadeRight) {
                    ApplyRightBoundaryFade(phoneWeights[i], rightOverlap, rule.RightMinWeight);
                }
                int underlayFrames = Math.Min(
                    Math.Max(gapFrames, 0) + rule.OverlapFrames,
                    Math.Min(phraseFrames - nominalBoundary, Math.Max(0, left.FrameCount)));
                if (underlayFrames > 0 && nominalBoundary < phraseFrames) {
                    underlays.Add(new MelUnderlay(
                        Source: left,
                        DestinationStartFrame: Math.Max(0, nominalBoundary),
                        FrameCount: underlayFrames,
                        RightStartFrame: right.StartFrame,
                        OverlapFrames: rightOverlap,
                        PreStartWeight: rule.PreStartUnderlayWeight,
                        PostStartWeight: rule.PostStartUnderlayWeight));
                }

                boundaryReports.Add(new HifiMelBoundaryCompositionReport {
                    BoundaryIndex = i - 1,
                    Kind = rule.Kind,
                    LeftPhoneme = left.Phoneme,
                    RightPhoneme = right.Phoneme,
                    NominalBoundaryFrame = nominalBoundary,
                    LeftEndFrame = left.EndFrame,
                    RightStartFrame = right.StartFrame,
                    GapFrames = gapFrames,
                    OverlapFrames = rightOverlap,
                    MelDeltaBefore = melDeltaBefore,
                    EnergyDeltaBefore = energyDeltaBefore,
                    LeftF0 = f0Context.LeftF0,
                    RightF0 = f0Context.RightF0,
                    F0JumpCents = f0Context.JumpCents,
                    F0MaxSlopeCents = f0Context.MaxSlopeCents,
                    F0AwareAdjustment = F0AwareAdjustmentName(f0Context, rule.Kind),
                    RightMinWeight = rule.FadeRight ? rule.RightMinWeight : 1f,
                    UnderlayMaxWeight = Math.Max(rule.PreStartUnderlayWeight, rule.PostStartUnderlayWeight),
                });
            }

            int clippedFrames = 0;
            for (int i = 0; i < phones.Length; i++) {
                clippedFrames += AddPhone(phraseMel, weights, phones[i], phoneWeights[i]);
            }

            int underlayFrameCount = 0;
            foreach (var underlay in underlays) {
                underlayFrameCount += AddUnderlay(phraseMel, weights, underlay);
            }

            LogReport(phones.Length, phraseFrames, clippedFrames, gapFramesBeforeUnderlay, underlayFrameCount, boundaryReports);
            return new HifiMelCompositionReport {
                PhoneCount = phones.Length,
                PhraseFrames = phraseFrames,
                ClippedFrames = clippedFrames,
                GapFramesBeforeUnderlay = gapFramesBeforeUnderlay,
                UnderlayFrames = underlayFrameCount,
                Boundaries = boundaryReports,
            };
        }

        static int AddPhone(float[,] phraseMel, float[] weights, HifiPreparedPhoneMel phone, float[] phoneWeights) {
            int clipped = 0;
            int bins = phraseMel.GetLength(0);
            for (int t = 0; t < phone.FrameCount; t++) {
                int dst = phone.StartFrame + t;
                if (dst < 0 || dst >= weights.Length) {
                    clipped++;
                    continue;
                }
                float weight = Math.Clamp(phoneWeights[t], 0.001f, 1f);
                for (int m = 0; m < bins; m++) {
                    phraseMel[m, dst] += phone.Mel[m, t] * weight;
                }
                weights[dst] += weight;
            }
            if (clipped > 0) {
                Log.Warning(
                    "HifiMelPhraseComposer phone_clipped index={Index} phoneme={Phoneme} start_frame={StartFrame} phone_frames={PhoneFrames} clipped_frames={ClippedFrames}",
                    phone.Index,
                    phone.Phoneme,
                    phone.StartFrame,
                    phone.FrameCount,
                    clipped);
            }
            return clipped;
        }

        static int AddUnderlay(float[,] phraseMel, float[] weights, MelUnderlay underlay) {
            int bins = phraseMel.GetLength(0);
            int added = 0;
            for (int t = 0; t < underlay.FrameCount; t++) {
                int dst = underlay.DestinationStartFrame + t;
                if (dst < 0 || dst >= weights.Length) {
                    continue;
                }
                float weight = UnderlayWeight(underlay, dst);
                if (weight <= 0.001f) {
                    continue;
                }
                double sourceIndex = underlay.FrameCount == 1 || underlay.Source.FrameCount == 1
                    ? underlay.Source.FrameCount - 1
                    : underlay.Source.FrameCount - underlay.FrameCount
                        + (double)t * (underlay.FrameCount - 1) / Math.Max(1, underlay.FrameCount - 1);
                sourceIndex = Math.Clamp(sourceIndex, 0, underlay.Source.FrameCount - 1);
                int left = (int)Math.Floor(sourceIndex);
                int right = Math.Min(underlay.Source.FrameCount - 1, left + 1);
                float alpha = (float)(sourceIndex - left);
                for (int m = 0; m < bins; m++) {
                    float v0 = underlay.Source.Mel[m, left];
                    float v1 = underlay.Source.Mel[m, right];
                    phraseMel[m, dst] += (v0 + (v1 - v0) * alpha) * weight;
                }
                weights[dst] += weight;
                added++;
            }
            return added;
        }

        static BoundaryRule ResolveBoundaryRule(HifiPreparedPhoneMel left, HifiPreparedPhoneMel right, BoundaryF0Context f0) {
            if (IsSilence(left.Phoneme) || IsSilence(right.Phoneme)) {
                return new BoundaryRule("silence", 0, 1f, 0f, 0f, false);
            }
            bool leftStable = IsStableVowelOrNasal(left.Phoneme);
            bool rightStable = IsStableVowelOrNasal(right.Phoneme) && right.TargetConsonantFrames <= 0;
            bool rightConsonant = right.TargetConsonantFrames > 0 || StartsWithUnvoicedConsonant(right.Phoneme);
            bool vcvLinked = IsVcvLinkedBoundary(left.Phoneme, right.Phoneme);
            bool rapidShort = IsRapidShortBoundary(left, right);
            int baseOverlap = Math.Clamp(
                Math.Max(Math.Max(left.CrossfadeFrames, right.CrossfadeFrames), 4),
                2,
                MaxBoundaryOverlapFrames);

            if (rapidShort) {
                if (rightConsonant) {
                    int overlap = Math.Clamp(
                        Math.Min(baseOverlap, Math.Max(2, right.TargetConsonantFrames + 1)),
                        2,
                        3);
                    float preStart = right.TargetConsonantFrames >= 3 ? 0.24f : 0.30f;
                    float postStart = right.TargetConsonantFrames >= 3 ? 0.03f : 0.05f;
                    if (vcvLinked) {
                        overlap = Math.Clamp(overlap + 1, 2, 4);
                        preStart = Math.Clamp(preStart + 0.08f, 0f, 1f);
                        postStart = Math.Clamp(postStart + 0.03f, 0f, 1f);
                    }
                    if (IsLargeF0Jump(f0)) {
                        overlap = Math.Max(1, overlap - 1);
                        preStart *= 0.75f;
                        postStart *= 0.5f;
                    }
                    return new BoundaryRule("rapid_to_consonant", overlap, 1f, preStart, postStart, false);
                }
                if (leftStable && rightStable) {
                    int overlap = Math.Clamp(baseOverlap, 2, 3);
                    if (IsContinuousF0(f0)) {
                        return new BoundaryRule("rapid_vowel_to_vowel_f0_continuous", Math.Clamp(overlap + 1, 2, 4), 1f, 0.55f, 0.06f, false);
                    }
                    if (IsLargeF0Jump(f0)) {
                        return new BoundaryRule("rapid_vowel_to_vowel_f0_jump", Math.Clamp(overlap - 1, 1, 2), 1f, 0.28f, 0f, false);
                    }
                    return new BoundaryRule("rapid_vowel_to_vowel", overlap, 1f, 0.45f, 0.04f, false);
                }
                if (leftStable || rightStable) {
                    int overlap = Math.Clamp(baseOverlap, 1, 2);
                    if (IsContinuousF0(f0)) {
                        return new BoundaryRule("rapid_mixed_voiced_f0_continuous", Math.Clamp(overlap + 1, 1, 3), 1f, 0.36f, 0.04f, false);
                    }
                    return new BoundaryRule("rapid_mixed_voiced", overlap, 1f, 0.28f, 0f, false);
                }
                return new BoundaryRule("rapid_transient", 0, 1f, 0f, 0f, false);
            }

            if (leftStable && rightStable) {
                int overlap = Math.Clamp(baseOverlap + 4, 6, 12);
                if (IsContinuousF0(f0)) {
                    return new BoundaryRule("vowel_to_vowel_f0_continuous", Math.Clamp(overlap + 2, 6, MaxBoundaryOverlapFrames), 1f, 0.92f, 0.34f, false);
                }
                if (IsLargeF0Jump(f0)) {
                    return new BoundaryRule("vowel_to_vowel_f0_jump", Math.Clamp(overlap / 2, 3, 7), 1f, 0.45f, 0.08f, false);
                }
                if (IsFastF0Motion(f0)) {
                    return new BoundaryRule("vowel_to_vowel_f0_motion", Math.Clamp(overlap, 5, 10), 1f, 0.70f, 0.18f, false);
                }
                return new BoundaryRule("vowel_to_vowel", overlap, 1f, 0.85f, 0.28f, false);
            }
            if (leftStable && rightConsonant) {
                int overlap = Math.Clamp(Math.Min(baseOverlap, Math.Max(2, right.TargetConsonantFrames + 2)), 2, 6);
                if (vcvLinked) {
                    return new BoundaryRule("vowel_to_consonant_vcv_linked", Math.Clamp(overlap + 1, 3, 7), 1f, 0.70f, 0.26f, false);
                }
                return new BoundaryRule("vowel_to_consonant", overlap, 1f, 0.55f, 0.18f, false);
            }
            if (!leftStable && rightStable) {
                int overlap = Math.Clamp(baseOverlap, 3, 8);
                if (IsContinuousF0(f0)) {
                    return new BoundaryRule("consonant_to_vowel_f0_continuous", Math.Clamp(overlap + 1, 3, 9), 1f, 0.52f, 0.24f, false);
                }
                return new BoundaryRule("consonant_to_vowel", overlap, 1f, 0.45f, 0.22f, false);
            }
            if (leftStable || rightStable) {
                int overlap = Math.Clamp(baseOverlap, 3, 6);
                if (IsContinuousF0(f0)) {
                    return new BoundaryRule("mixed_voiced_f0_continuous", Math.Clamp(overlap + 1, 3, 7), 1f, 0.52f, 0.20f, false);
                }
                if (IsLargeF0Jump(f0)) {
                    return new BoundaryRule("mixed_voiced_f0_jump", Math.Clamp(overlap - 1, 2, 4), 1f, 0.32f, 0.08f, false);
                }
                return new BoundaryRule("mixed_voiced", overlap, 1f, 0.45f, 0.18f, false);
            }
            return new BoundaryRule("transient", 1, 1f, 0.12f, 0.06f, false);
        }

        static BoundaryF0Context BoundaryF0At(float[]? f0, HifiPreparedPhoneMel left, HifiPreparedPhoneMel right, int nominalBoundary) {
            if (!HifiNeuralConfig.EnableF0AwareBoundaryCompose || f0 == null || f0.Length < 2) {
                return new BoundaryF0Context(false, 0, 0, 0, 0);
            }
            int leftFrame = Math.Clamp(Math.Min(left.EndFrame - 1, nominalBoundary - 1), 0, f0.Length - 1);
            int rightFrame = Math.Clamp(Math.Max(right.StartFrame, nominalBoundary), 0, f0.Length - 1);
            float leftF0 = f0[leftFrame];
            float rightF0 = f0[rightFrame];
            if (leftF0 <= 0 || rightF0 <= 0 || float.IsNaN(leftF0) || float.IsNaN(rightF0) || float.IsInfinity(leftF0) || float.IsInfinity(rightF0)) {
                return new BoundaryF0Context(false, leftF0, rightF0, 0, 0);
            }
            int start = Math.Clamp(nominalBoundary - 2, 0, f0.Length - 1);
            int end = Math.Clamp(nominalBoundary + 2, start + 1, f0.Length);
            double maxSlope = 0;
            for (int i = start + 1; i < end; i++) {
                double slope = F0Cents(f0[i - 1], f0[i]);
                if (IsFinite(slope)) {
                    maxSlope = Math.Max(maxSlope, slope);
                }
            }
            return new BoundaryF0Context(true, leftF0, rightF0, F0Cents(leftF0, rightF0), maxSlope);
        }

        static bool IsContinuousF0(BoundaryF0Context f0) {
            return f0.IsValid
                && f0.JumpCents <= HifiNeuralConfig.BoundaryF0ContinuousJumpCents
                && f0.MaxSlopeCents <= HifiNeuralConfig.BoundaryF0ContinuousMaxSlopeCents;
        }

        static bool IsLargeF0Jump(BoundaryF0Context f0) {
            return f0.IsValid && f0.JumpCents >= HifiNeuralConfig.BoundaryF0LargeJumpCents;
        }

        static bool IsFastF0Motion(BoundaryF0Context f0) {
            return f0.IsValid && f0.MaxSlopeCents >= HifiNeuralConfig.BoundaryF0FastMotionCents;
        }

        static string F0AwareAdjustmentName(BoundaryF0Context f0, string kind) {
            if (!f0.IsValid) {
                return string.Empty;
            }
            if (kind.Contains("f0_continuous", StringComparison.Ordinal)) {
                return "continuous";
            }
            if (kind.Contains("f0_jump", StringComparison.Ordinal)) {
                return "jump";
            }
            if (kind.Contains("f0_motion", StringComparison.Ordinal)) {
                return "motion";
            }
            return "neutral";
        }

        static bool IsRapidShortBoundary(HifiPreparedPhoneMel left, HifiPreparedPhoneMel right) {
            int threshold = Math.Max(1, HifiNeuralConfig.RapidShortPhoneFrames);
            return left.FrameCount <= threshold
                || right.FrameCount <= threshold
                || left.FrameCount + right.FrameCount <= threshold * 2;
        }

        static bool IsVcvLinkedBoundary(string leftPhoneme, string rightPhoneme) {
            var rightTokens = SplitTokens(Normalize(rightPhoneme));
            if (rightTokens.Length < 2) {
                return false;
            }
            string leading = ExtractVowelClass(rightTokens[0]);
            if (string.IsNullOrEmpty(leading)) {
                return false;
            }
            string leftStable = ExtractStableVowelClass(leftPhoneme);
            return !string.IsNullOrEmpty(leftStable) && leftStable == leading;
        }

        static string ExtractStableVowelClass(string phoneme) {
            var tokens = SplitTokens(Normalize(phoneme));
            for (int i = tokens.Length - 1; i >= 0; i--) {
                string v = ExtractVowelClass(tokens[i]);
                if (!string.IsNullOrEmpty(v)) {
                    return v;
                }
            }
            return string.Empty;
        }

        static string ExtractVowelClass(string token) {
            if (string.IsNullOrWhiteSpace(token)) {
                return string.Empty;
            }
            token = token.Trim().ToLowerInvariant();
            for (int i = token.Length - 1; i >= 0; i--) {
                char c = token[i];
                if (c is 'a' or 'i' or 'u' or 'e' or 'o') {
                    return c.ToString();
                }
                if (c == 'n') {
                    return "n";
                }
                string kanaVowel = JapaneseKanaVowel(c);
                if (!string.IsNullOrEmpty(kanaVowel)) {
                    return kanaVowel;
                }
                if (JapaneseA.Contains(c)) return "a";
                if (JapaneseI.Contains(c)) return "i";
                if (JapaneseU.Contains(c)) return "u";
                if (JapaneseE.Contains(c)) return "e";
                if (JapaneseO.Contains(c)) return "o";
                if (c == 'ん' || c == 'ン') return "n";
            }
            return string.Empty;
        }

        static string JapaneseKanaVowel(char c) {
            return c switch {
                '\u3042' or '\u304b' or '\u304c' or '\u3055' or '\u3056' or '\u305f' or '\u3060' or '\u306a' or '\u306f' or '\u3070' or '\u3071' or '\u307e' or '\u3084' or '\u3089' or '\u308f'
                    or '\u30a2' or '\u30ab' or '\u30ac' or '\u30b5' or '\u30b6' or '\u30bf' or '\u30c0' or '\u30ca' or '\u30cf' or '\u30d0' or '\u30d1' or '\u30de' or '\u30e4' or '\u30e9' or '\u30ef' => "a",
                '\u3044' or '\u304d' or '\u304e' or '\u3057' or '\u3058' or '\u3061' or '\u3062' or '\u306b' or '\u3072' or '\u3073' or '\u3074' or '\u307f' or '\u308a'
                    or '\u30a4' or '\u30ad' or '\u30ae' or '\u30b7' or '\u30b8' or '\u30c1' or '\u30c2' or '\u30cb' or '\u30d2' or '\u30d3' or '\u30d4' or '\u30df' or '\u30ea' => "i",
                '\u3046' or '\u304f' or '\u3050' or '\u3059' or '\u305a' or '\u3064' or '\u3065' or '\u306c' or '\u3075' or '\u3076' or '\u3077' or '\u3080' or '\u3086' or '\u308b'
                    or '\u30a6' or '\u30af' or '\u30b0' or '\u30b9' or '\u30ba' or '\u30c4' or '\u30c5' or '\u30cc' or '\u30d5' or '\u30d6' or '\u30d7' or '\u30e0' or '\u30e6' or '\u30eb' => "u",
                '\u3048' or '\u3051' or '\u3052' or '\u305b' or '\u305c' or '\u3066' or '\u3067' or '\u306d' or '\u3078' or '\u3079' or '\u307a' or '\u3081' or '\u308c'
                    or '\u30a8' or '\u30b1' or '\u30b2' or '\u30bb' or '\u30bc' or '\u30c6' or '\u30c7' or '\u30cd' or '\u30d8' or '\u30d9' or '\u30da' or '\u30e1' or '\u30ec' => "e",
                '\u304a' or '\u3053' or '\u3054' or '\u305d' or '\u305e' or '\u3068' or '\u3069' or '\u306e' or '\u307b' or '\u307c' or '\u307d' or '\u3082' or '\u3088' or '\u308d' or '\u3092'
                    or '\u30aa' or '\u30b3' or '\u30b4' or '\u30bd' or '\u30be' or '\u30c8' or '\u30c9' or '\u30ce' or '\u30db' or '\u30dc' or '\u30dd' or '\u30e2' or '\u30e8' or '\u30ed' or '\u30f2' => "o",
                '\u3093' or '\u30f3' => "n",
                _ => string.Empty,
            };
        }

        static void ApplyRightBoundaryFade(float[] weights, int frames, float minWeight) {
            frames = Math.Clamp(frames, 0, weights.Length);
            minWeight = Math.Clamp(minWeight, 0.001f, 1f);
            for (int t = 0; t < frames; t++) {
                double x = frames == 1 ? 1.0 : (double)t / (frames - 1);
                float smooth = (float)(0.5 - 0.5 * Math.Cos(Math.PI * Math.Clamp(x, 0, 1)));
                weights[t] = Math.Min(weights[t], minWeight + (1f - minWeight) * smooth);
            }
        }

        static float UnderlayWeight(MelUnderlay underlay, int dstFrame) {
            if (dstFrame < underlay.RightStartFrame) {
                return underlay.PreStartWeight;
            }
            if (underlay.OverlapFrames <= 1) {
                return 0f;
            }
            double x = (dstFrame - underlay.RightStartFrame) / (double)(underlay.OverlapFrames - 1);
            float fade = (float)(0.5 + 0.5 * Math.Cos(Math.PI * Math.Clamp(x, 0, 1)));
            return Math.Clamp(underlay.PostStartWeight * fade, 0f, underlay.PostStartWeight);
        }

        static double BoundaryMelDelta(HifiPreparedPhoneMel left, HifiPreparedPhoneMel right) {
            if (left.FrameCount <= 0 || right.FrameCount <= 0) {
                return 0;
            }
            int bins = left.Mel.GetLength(0);
            double sum = 0;
            for (int m = 0; m < bins; m++) {
                sum += Math.Abs(left.Mel[m, left.FrameCount - 1] - right.Mel[m, 0]);
            }
            return sum / Math.Max(1, bins);
        }

        static double BoundaryEnergyDelta(HifiPreparedPhoneMel left, HifiPreparedPhoneMel right) {
            if (left.FrameCount <= 0 || right.FrameCount <= 0) {
                return 0;
            }
            return MeanFrame(left.Mel, left.FrameCount - 1) - MeanFrame(right.Mel, 0);
        }

        static double F0Cents(float left, float right) {
            if (left <= 0 || right <= 0 || float.IsNaN(left) || float.IsNaN(right) || float.IsInfinity(left) || float.IsInfinity(right)) {
                return double.PositiveInfinity;
            }
            return Math.Abs(1200.0 * Math.Log(right / left, 2.0));
        }

        static double MeanFrame(float[,] mel, int frame) {
            int bins = mel.GetLength(0);
            double sum = 0;
            for (int m = 0; m < bins; m++) {
                sum += mel[m, frame];
            }
            return sum / Math.Max(1, bins);
        }

        static bool IsStableVowelOrNasal(string phoneme) {
            if (string.IsNullOrWhiteSpace(phoneme)) {
                return false;
            }
            string p = Normalize(phoneme);
            if (IsSilence(p) || p.Contains("cl")) {
                return false;
            }
            if (p is "p" or "t" or "k" or "q" or "s" or "sh" or "ch" or "ts" or "f" or "h" or "hh" or "th") {
                return false;
            }
            return p.Any(c => "aeiou".Contains(c)) || SplitTokens(p).Any(token => token is "m" or "n" or "ng");
        }

        static bool StartsWithUnvoicedConsonant(string phoneme) {
            string p = Normalize(phoneme);
            if (string.IsNullOrWhiteSpace(p)) {
                return false;
            }
            var tokens = SplitTokens(p);
            if (tokens.Length > 0 && UnvoicedTokens.Contains(tokens[0])) {
                return true;
            }
            return UnvoicedPrefixes.Any(prefix => p.StartsWith(prefix, StringComparison.Ordinal));
        }

        static string[] SplitTokens(string phoneme) {
            return phoneme
                .Split(new[] { ' ', '-', '_', '+', '/' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(token => token.Trim())
                .Where(token => token.Length > 0)
                .ToArray();
        }

        static bool IsSilence(string phoneme) {
            string p = Normalize(phoneme);
            return p == "r" || p == "rest" || p == "sil" || p == "pau" || p == "-" || p == "br";
        }

        static string Normalize(string phoneme) => (phoneme ?? string.Empty).Trim().ToLowerInvariant();

        static void LogReport(
            int phoneCount,
            int phraseFrames,
            int clippedFrames,
            int gapFramesBeforeUnderlay,
            int underlayFrames,
            IReadOnlyList<HifiMelBoundaryCompositionReport> boundaries) {
            Log.Information(
                "HifiMelPhraseComposer summary phones={Phones} boundaries={Boundaries} phrase_frames={PhraseFrames} clipped_frames={ClippedFrames} gap_frames_before_underlay={GapFramesBeforeUnderlay} underlay_frames={UnderlayFrames}",
                phoneCount,
                boundaries.Count,
                phraseFrames,
                clippedFrames,
                gapFramesBeforeUnderlay,
                underlayFrames);
            foreach (var boundary in boundaries) {
                Log.Information(
                    "HifiMelPhraseComposer boundary index={Index} kind={Kind} left={LeftPhoneme} right={RightPhoneme} nominal_boundary_frame={NominalBoundaryFrame} left_end_frame={LeftEndFrame} right_start_frame={RightStartFrame} gap_frames={GapFrames} overlap_frames={OverlapFrames} mel_delta_before={MelDeltaBefore:F4} energy_delta_before={EnergyDeltaBefore:F4} left_f0={LeftF0:F2} right_f0={RightF0:F2} f0_jump_cents={F0JumpCents:F2} f0_max_slope_cents={F0MaxSlopeCents:F2} f0_adjustment={F0Adjustment} right_min_weight={RightMinWeight:F3} underlay_max_weight={UnderlayMaxWeight:F3} skipped_reason={SkippedReason}",
                    boundary.BoundaryIndex,
                    boundary.Kind,
                    boundary.LeftPhoneme,
                    boundary.RightPhoneme,
                    boundary.NominalBoundaryFrame,
                    boundary.LeftEndFrame,
                    boundary.RightStartFrame,
                    boundary.GapFrames,
                    boundary.OverlapFrames,
                    boundary.MelDeltaBefore,
                    boundary.EnergyDeltaBefore,
                    boundary.LeftF0,
                    boundary.RightF0,
                    boundary.F0JumpCents,
                    boundary.F0MaxSlopeCents,
                    boundary.F0AwareAdjustment,
                    boundary.RightMinWeight,
                    boundary.UnderlayMaxWeight,
                    boundary.SkippedReason);
            }
        }

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

        static readonly HashSet<char> JapaneseA = new(new[] {
            'あ','ぁ','か','が','さ','ざ','た','だ','な','は','ば','ぱ','ま','ゃ','や','ら','ゎ','わ',
            'ア','ァ','カ','ガ','サ','ザ','タ','ダ','ナ','ハ','バ','パ','マ','ャ','ヤ','ラ','ヮ','ワ',
        });
        static readonly HashSet<char> JapaneseI = new(new[] {
            'い','ぃ','き','ぎ','し','じ','ち','ぢ','に','ひ','び','ぴ','み','り',
            'イ','ィ','キ','ギ','シ','ジ','チ','ヂ','ニ','ヒ','ビ','ピ','ミ','リ',
        });
        static readonly HashSet<char> JapaneseU = new(new[] {
            'う','ぅ','く','ぐ','す','ず','つ','づ','ぬ','ふ','ぶ','ぷ','む','ゅ','ゆ','る','ゔ',
            'ウ','ゥ','ク','グ','ス','ズ','ツ','ヅ','ヌ','フ','ブ','プ','ム','ュ','ユ','ル','ヴ',
        });
        static readonly HashSet<char> JapaneseE = new(new[] {
            'え','ぇ','け','げ','せ','ぜ','て','で','ね','へ','べ','ぺ','め','れ',
            'エ','ェ','ケ','ゲ','セ','ゼ','テ','デ','ネ','ヘ','ベ','ペ','メ','レ',
        });
        static readonly HashSet<char> JapaneseO = new(new[] {
            'お','ぉ','こ','ご','そ','ぞ','と','ど','の','ほ','ぼ','ぽ','も','ょ','よ','ろ','を',
            'オ','ォ','コ','ゴ','ソ','ゾ','ト','ド','ノ','ホ','ボ','ポ','モ','ョ','ヨ','ロ','ヲ',
        });

        static readonly HashSet<string> UnvoicedTokens = new(StringComparer.Ordinal) {
            "p", "t", "k", "q",
            "s", "sh", "ch", "ts",
            "f", "h", "hh",
            "th",
        };

        static readonly string[] UnvoicedPrefixes = {
            "sh", "ch", "ts", "hh", "th", "p", "t", "k", "q", "s", "f", "h",
        };

        readonly record struct MelUnderlay(
            HifiPreparedPhoneMel Source,
            int DestinationStartFrame,
            int FrameCount,
            int RightStartFrame,
            int OverlapFrames,
            float PreStartWeight,
            float PostStartWeight);
    }
}
