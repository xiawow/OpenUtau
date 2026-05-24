using System;
using System.Collections.Generic;
using System.Linq;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiPhoneDurationPlan {
        public int Index { get; init; }
        public string Phoneme { get; init; } = string.Empty;
        public int OriginalStartFrame { get; init; }
        public int OriginalDurationFrames { get; init; }
        public int OriginalEndFrame => OriginalStartFrame + OriginalDurationFrames;
        public int AdjustedStartFrame { get; set; }
        public int AdjustedDurationFrames { get; set; }
        public int AdjustedEndFrame => AdjustedStartFrame + AdjustedDurationFrames;
        public int ConsonantFrames { get; init; }
        public int TargetConsonantFrames { get; set; }
        public bool CanDonateVowel { get; init; }
        public string FallbackReason { get; set; } = string.Empty;
        public int BorrowNeededFrames { get; set; }
        public int BorrowedFromPreviousFrames { get; set; }
        public int BorrowedFromNextFrames { get; set; }
        public string BorrowedFramesAppliedTo { get; set; } = "none";
        public string DonorPhonePhoneme { get; set; } = string.Empty;
        public int DonorOriginalVowelFrames { get; set; }
        public int DonorAdjustedVowelFrames { get; set; }
        public int DonatedFrames { get; set; }
    }

    public sealed class HifiDurationRedistributionSummary {
        public int PhraseFramesBefore { get; init; }
        public int PhraseFramesAfter { get; init; }
        public int TotalBorrowedFrames { get; init; }
        public IReadOnlyList<int> BorrowTriggeredPhones { get; init; } = Array.Empty<int>();
        public IReadOnlyList<int> DonorPhones { get; init; } = Array.Empty<int>();
    }

    public sealed class HifiDurationRedistributionResult {
        public required List<HifiPhoneDurationPlan> Plans { get; init; }
        public required HifiDurationRedistributionSummary Summary { get; init; }
    }

    public static class HifiDurationRedistributor {
        public static HifiDurationRedistributionResult Redistribute(
            IReadOnlyList<HifiPhoneDurationPlan> inputPlans,
            bool enabled,
            int phraseFrames,
            double minConsonantMs,
            double minVowelMs,
            double maxBorrowMs,
            double maxBorrowRatio,
            HifiConsonantLockMode lockMode,
            bool enableCodaProtection = false,
            double minCodaMs = 40) {
            var plans = inputPlans.Select(Clone).ToList();
            int minConsonantFrames = MsToFrames(minConsonantMs);
            int minVowelFrames = Math.Max(1, MsToFrames(minVowelMs));
            int maxBorrowFrames = Math.Max(1, MsToFrames(maxBorrowMs));
            int minCodaFrames = Math.Max(1, MsToFrames(minCodaMs));
            if (!enabled) {
                ClampAllConsonantsForAudibleVowels(plans, minVowelFrames, "disabled");
                return Finish(plans, phraseFrames);
            }
            if (plans.Count == 0) {
                return Finish(plans, phraseFrames);
            }

            for (int i = 0; i < plans.Count; i++) {
                var plan = plans[i];
                plan.TargetConsonantFrames = Math.Min(plan.AdjustedDurationFrames, Math.Max(plan.TargetConsonantFrames, plan.ConsonantFrames));
                ClampConsonantForAudibleVowel(plan, minVowelFrames, "initial");

                bool shortConsonant = plan.ConsonantFrames > 0 && plan.ConsonantFrames < minConsonantFrames;
                bool shortPhoneRisk = plan.AdjustedDurationFrames < minConsonantFrames + minVowelFrames;
                bool shortCoda = enableCodaProtection && IsCodaPhoneme(plan.Phoneme) && plan.AdjustedDurationFrames < minCodaFrames;
                if (!shortConsonant && !shortPhoneRisk && !shortCoda) {
                    continue;
                }

                int borrowNeeded = Math.Max(0, minConsonantFrames - plan.TargetConsonantFrames);
                if (shortPhoneRisk) {
                    borrowNeeded = Math.Max(borrowNeeded, minConsonantFrames + minVowelFrames - plan.AdjustedDurationFrames);
                }
                if (shortCoda) {
                    borrowNeeded = Math.Max(borrowNeeded, minCodaFrames - plan.AdjustedDurationFrames);
                }
                int maxForCurrent = Math.Max(0, (int)Math.Floor(plan.OriginalDurationFrames * maxBorrowRatio));
                borrowNeeded = Math.Min(borrowNeeded, maxForCurrent);
                plan.BorrowNeededFrames = borrowNeeded;
                if (borrowNeeded <= 0) {
                    plan.FallbackReason = "borrow_needed_frames<=0";
                    continue;
                }

                int remaining = borrowNeeded;
                if (i > 0) {
                    int borrowed = BorrowFromPrevious(plans[i - 1], plan, remaining, minVowelFrames, maxBorrowFrames);
                    remaining -= borrowed;
                }
                if (remaining > 0 && i + 1 < plans.Count) {
                    int borrowed = BorrowFromNext(plans[i + 1], plan, remaining, minVowelFrames, maxBorrowFrames);
                    remaining -= borrowed;
                }

                int borrowedTotal = plan.BorrowedFromPreviousFrames + plan.BorrowedFromNextFrames;
                if (borrowedTotal <= 0) {
                    plan.FallbackReason = "no_adjacent_vowel_available";
                    continue;
                }
                if (lockMode == HifiConsonantLockMode.Readable) {
                    plan.TargetConsonantFrames = Math.Min(plan.AdjustedDurationFrames, plan.TargetConsonantFrames + borrowedTotal);
                    plan.BorrowedFramesAppliedTo = "consonant";
                } else if (lockMode == HifiConsonantLockMode.Preserve) {
                    int maxConsonant = Math.Max(0, plan.AdjustedDurationFrames - minVowelFrames);
                    plan.TargetConsonantFrames = Math.Min(maxConsonant, Math.Max(0, plan.ConsonantFrames));
                    if (plan.AdjustedDurationFrames - plan.TargetConsonantFrames < minVowelFrames) {
                        plan.FallbackReason = "insufficient_vowel_after_borrow";
                    }
                    plan.BorrowedFramesAppliedTo = shortCoda ? "coda" : "vowel";
                } else {
                    plan.BorrowedFramesAppliedTo = "none";
                }
                ClampConsonantForAudibleVowel(plan, minVowelFrames, "after_borrow");
                string? validationError = ValidatePlans(plans, phraseFrames);
                if (!string.IsNullOrEmpty(validationError)) {
                    plan.FallbackReason = validationError;
                    LogRedistribution(plans, Finish(plans, phraseFrames).Summary, minConsonantFrames);
                    return Redistribute(inputPlans, enabled: false, phraseFrames, minConsonantMs, minVowelMs, maxBorrowMs, maxBorrowRatio, lockMode, enableCodaProtection, minCodaMs);
                }
            }

            ClampAllConsonantsForAudibleVowels(plans, minVowelFrames, "final");
            var result = Finish(plans, phraseFrames);
            LogRedistribution(result.Plans, result.Summary, minConsonantFrames);
            return result;
        }

        static int BorrowFromPrevious(HifiPhoneDurationPlan donor, HifiPhoneDurationPlan receiver, int requested, int minVowelFrames, int maxBorrowFrames) {
            int donorVowel = VowelFrames(donor);
            int available = Math.Min(requested, Math.Min(maxBorrowFrames, donorVowel - minVowelFrames));
            if (!donor.CanDonateVowel || available <= 0) {
                return 0;
            }
            donor.AdjustedDurationFrames -= available;
            donor.DonatedFrames += available;
            receiver.AdjustedStartFrame -= available;
            receiver.AdjustedDurationFrames += available;
            receiver.BorrowedFromPreviousFrames += available;
            receiver.DonorPhonePhoneme = donor.Phoneme;
            receiver.DonorOriginalVowelFrames = donorVowel;
            receiver.DonorAdjustedVowelFrames = VowelFrames(donor);
            return available;
        }

        static int BorrowFromNext(HifiPhoneDurationPlan donor, HifiPhoneDurationPlan receiver, int requested, int minVowelFrames, int maxBorrowFrames) {
            int donorVowel = VowelFrames(donor);
            int available = Math.Min(requested, Math.Min(maxBorrowFrames, donorVowel - minVowelFrames));
            if (!donor.CanDonateVowel || available <= 0) {
                return 0;
            }
            receiver.AdjustedDurationFrames += available;
            donor.AdjustedStartFrame += available;
            donor.AdjustedDurationFrames -= available;
            donor.DonatedFrames += available;
            receiver.BorrowedFromNextFrames += available;
            receiver.DonorPhonePhoneme = donor.Phoneme;
            receiver.DonorOriginalVowelFrames = donorVowel;
            receiver.DonorAdjustedVowelFrames = VowelFrames(donor);
            return available;
        }

        static HifiDurationRedistributionResult Finish(List<HifiPhoneDurationPlan> plans, int phraseFrames) {
            int phraseFramesAfter = plans.Count > 0 ? plans.Max(p => p.AdjustedEndFrame) : phraseFrames;
            var summary = new HifiDurationRedistributionSummary {
                PhraseFramesBefore = phraseFrames,
                PhraseFramesAfter = phraseFramesAfter,
                TotalBorrowedFrames = plans.Sum(p => p.BorrowedFromPreviousFrames + p.BorrowedFromNextFrames),
                BorrowTriggeredPhones = plans.Where(p => p.BorrowedFromPreviousFrames + p.BorrowedFromNextFrames > 0).Select(p => p.Index).ToArray(),
                DonorPhones = plans.Where(p => p.DonatedFrames > 0).Select(p => p.Index).Distinct().ToArray(),
            };
            return new HifiDurationRedistributionResult {
                Plans = plans,
                Summary = summary,
            };
        }

        static int VowelFrames(HifiPhoneDurationPlan plan) {
            return Math.Max(0, plan.AdjustedDurationFrames - plan.TargetConsonantFrames);
        }

        static void ClampAllConsonantsForAudibleVowels(List<HifiPhoneDurationPlan> plans, int minVowelFrames, string stage) {
            foreach (var plan in plans) {
                ClampConsonantForAudibleVowel(plan, minVowelFrames, stage);
            }
        }

        static void ClampConsonantForAudibleVowel(HifiPhoneDurationPlan plan, int minVowelFrames, string stage) {
            if (!HasStableVoice(plan.Phoneme) || plan.AdjustedDurationFrames <= 1) {
                return;
            }
            int requiredVowelFrames = RequiredAudibleVowelFrames(plan.AdjustedDurationFrames, minVowelFrames);
            int maxConsonantFrames = Math.Max(0, plan.AdjustedDurationFrames - requiredVowelFrames);
            if (plan.TargetConsonantFrames <= maxConsonantFrames) {
                return;
            }
            Log.Information(
                "HifiDurationRedistributor audible_vowel_clamp stage={Stage} phoneme={Phoneme} duration_frames={DurationFrames} target_consonant_before={Before} target_consonant_after={After} required_vowel_frames={RequiredVowelFrames}",
                stage,
                plan.Phoneme,
                plan.AdjustedDurationFrames,
                plan.TargetConsonantFrames,
                maxConsonantFrames,
                requiredVowelFrames);
            plan.TargetConsonantFrames = maxConsonantFrames;
        }

        static int RequiredAudibleVowelFrames(int durationFrames, int minVowelFrames) {
            if (durationFrames <= 0) {
                return 0;
            }
            int requested = durationFrames <= 8
                ? Math.Max(1, (int)Math.Ceiling(durationFrames * 0.6))
                : Math.Max(1, (durationFrames + 1) / 2);
            return Math.Clamp(requested, 1, Math.Min(minVowelFrames, durationFrames));
        }

        static string? ValidatePlans(List<HifiPhoneDurationPlan> plans, int phraseFrames) {
            foreach (var plan in plans) {
                if (plan.AdjustedStartFrame < 0 || plan.AdjustedDurationFrames <= 0 || plan.TargetConsonantFrames < 0) {
                    return "negative_duration_or_frame";
                }
                if (plan.TargetConsonantFrames > plan.AdjustedDurationFrames) {
                    return "consonant_exceeds_duration";
                }
                if (plan.AdjustedEndFrame > phraseFrames) {
                    return "phone_exceeds_phrase_frames";
                }
            }
            for (int i = 1; i < plans.Count; i++) {
                if (plans[i].AdjustedStartFrame < plans[i - 1].AdjustedStartFrame) {
                    return "phone_order_inverted";
                }
                if (plans[i].AdjustedStartFrame != plans[i - 1].AdjustedEndFrame) {
                    return "phone_gap_or_overlap";
                }
            }
            return null;
        }

        static HifiPhoneDurationPlan Clone(HifiPhoneDurationPlan plan) {
            return new HifiPhoneDurationPlan {
                Index = plan.Index,
                Phoneme = plan.Phoneme,
                OriginalStartFrame = plan.OriginalStartFrame,
                OriginalDurationFrames = plan.OriginalDurationFrames,
                AdjustedStartFrame = plan.AdjustedStartFrame,
                AdjustedDurationFrames = plan.AdjustedDurationFrames,
                ConsonantFrames = plan.ConsonantFrames,
                TargetConsonantFrames = plan.TargetConsonantFrames,
                CanDonateVowel = plan.CanDonateVowel,
            };
        }

        public static int MsToFrames(double ms) {
            return Math.Max(0, (int)Math.Round(ms / HifiF0Builder.FrameMs));
        }

        static bool IsCodaPhoneme(string phoneme) {
            if (string.IsNullOrWhiteSpace(phoneme)) {
                return false;
            }
            string p = phoneme.Trim().ToLowerInvariant();
            return p == "ng" || p == "n" || p == "m" || p == "k" || p == "t" || p == "p";
        }

        static bool HasStableVoice(string phoneme) {
            if (string.IsNullOrWhiteSpace(phoneme)) {
                return false;
            }
            string p = phoneme.Trim().ToLowerInvariant();
            if (p == "r" || p == "rest" || p == "sil" || p == "pau" || p == "-" || p == "br" || p.Contains("cl")) {
                return false;
            }
            return p.Any(c => "aeiou".Contains(c))
                || p.Split(new[] { ' ', '-', '_', '+', '/' }, StringSplitOptions.RemoveEmptyEntries)
                    .Any(token => token is "m" or "n" or "ng");
        }

        static void LogRedistribution(IReadOnlyList<HifiPhoneDurationPlan> plans, HifiDurationRedistributionSummary summary, int minConsonantFrames) {
            Log.Information(
                "HifiDurationRedistributor summary total_borrowed_frames={TotalBorrowedFrames} triggered=[{Triggered}] donors=[{Donors}] phrase_frames_before={Before} phrase_frames_after={After}",
                summary.TotalBorrowedFrames,
                string.Join(",", summary.BorrowTriggeredPhones),
                string.Join(",", summary.DonorPhones),
                summary.PhraseFramesBefore,
                summary.PhraseFramesAfter);
            foreach (var plan in plans) {
                if (plan.BorrowNeededFrames <= 0 && plan.DonatedFrames <= 0) {
                    continue;
                }
                Log.Information(
                    "HifiDurationRedistributor phone index={Index} phoneme={Phoneme} original_start_frame={OriginalStartFrame} original_end_frame={OriginalEndFrame} adjusted_start_frame={AdjustedStartFrame} adjusted_end_frame={AdjustedEndFrame} original_duration_frames={OriginalDurationFrames} adjusted_duration_frames={AdjustedDurationFrames} consonant_frames={ConsonantFrames} min_consonant_frames={MinConsonantFrames} borrow_needed_frames={BorrowNeededFrames} borrowed_from_previous_frames={BorrowedFromPreviousFrames} borrowed_from_next_frames={BorrowedFromNextFrames} borrowed_frames_applied_to={BorrowedFramesAppliedTo} donor_phone={DonorPhone} donor_original_vowel_frames={DonorOriginalVowelFrames} donor_adjusted_vowel_frames={DonorAdjustedVowelFrames} fallback_reason={FallbackReason}",
                    plan.Index,
                    plan.Phoneme,
                    plan.OriginalStartFrame,
                    plan.OriginalEndFrame,
                    plan.AdjustedStartFrame,
                    plan.AdjustedEndFrame,
                    plan.OriginalDurationFrames,
                    plan.AdjustedDurationFrames,
                    plan.ConsonantFrames,
                    minConsonantFrames,
                    plan.BorrowNeededFrames,
                    plan.BorrowedFromPreviousFrames,
                    plan.BorrowedFromNextFrames,
                    plan.BorrowedFramesAppliedTo,
                    plan.DonorPhonePhoneme,
                    plan.DonorOriginalVowelFrames,
                    plan.DonorAdjustedVowelFrames,
                    plan.FallbackReason);
            }
        }
    }
}
