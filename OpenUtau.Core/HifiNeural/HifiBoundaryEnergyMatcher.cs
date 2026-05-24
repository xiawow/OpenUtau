using System;
using System.Collections.Generic;
using System.Linq;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiBoundaryEnergyMatcher {
        public static void ApplyConservative(
            float[,] mel,
            IReadOnlyList<HifiBoundaryMetadata> boundaries,
            IReadOnlyList<HifiPhoneDurationPlan> plans,
            int windowFrames,
            double maxGainDb) {
            int frames = mel.GetLength(1);
            int bins = mel.GetLength(0);
            int win = Math.Clamp(windowFrames, 8, 16);
            double maxDb = Math.Clamp(maxGainDb, 0.5, 1.0);

            for (int i = 0; i < boundaries.Count; i++) {
                var boundary = boundaries[i];
                int frame = boundary.Frame;
                string skipped = string.Empty;
                double leftEnergy = 0;
                double rightEnergy = 0;
                double rawGainDb = 0;
                double clippedGainDb = 0;
                int appliedFrames = 0;

                if (frame - win < 0 || frame + win >= frames) {
                    skipped = "boundary_out_of_range";
                } else if (!IsVowelStable(boundary.LeftPhone) || !IsVowelStable(boundary.RightPhone)) {
                    skipped = "non_vowel_boundary";
                } else if (TouchesProtectedConsonant(boundary, plans, frame, win)) {
                    skipped = "protected_consonant_boundary";
                } else {
                    leftEnergy = MeanEnergy(mel, frame - win, frame, bins);
                    rightEnergy = MeanEnergy(mel, frame, frame + win, bins);
                    if (!IsFinite(leftEnergy) || !IsFinite(rightEnergy)) {
                        skipped = "nan_or_inf_energy";
                    } else if (leftEnergy < -11.0 || rightEnergy < -11.0) {
                        skipped = "low_energy_or_silence";
                    } else {
                        rawGainDb = Math.Clamp((leftEnergy - rightEnergy) * 3.0, -6.0, 6.0);
                        clippedGainDb = Math.Clamp(rawGainDb, -maxDb, maxDb);
                        if (Math.Abs(clippedGainDb) < 0.05) {
                            skipped = "tiny_gain";
                        } else {
                            ApplyLocalGain(mel, frame, win, clippedGainDb / 20.0 * Math.Log(10), bins);
                            appliedFrames = win;
                        }
                    }
                }

                Log.Information(
                    "HifiBoundaryEnergyMatcher boundary_index={BoundaryIndex} left_energy={LeftEnergy:F4} right_energy={RightEnergy:F4} raw_gain_db={RawGainDb:F4} clipped_gain_db={ClippedGainDb:F4} applied_frames={AppliedFrames} skipped_reason={SkippedReason}",
                    i,
                    leftEnergy,
                    rightEnergy,
                    rawGainDb,
                    clippedGainDb,
                    appliedFrames,
                    skipped);
            }
        }

        static void ApplyLocalGain(float[,] mel, int boundaryFrame, int win, double gainLn, int bins) {
            int start = boundaryFrame;
            int end = boundaryFrame + win;
            for (int t = start; t < end; t++) {
                double x = (double)(t - start) / Math.Max(1, win - 1);
                double env = 0.5 - 0.5 * Math.Cos(2 * Math.PI * x);
                float offset = (float)(gainLn * env);
                for (int m = 0; m < bins; m++) {
                    mel[m, t] += offset;
                }
            }
        }

        static double MeanEnergy(float[,] mel, int start, int end, int bins) {
            double sum = 0;
            int count = 0;
            for (int t = start; t < end; t++) {
                for (int m = 0; m < bins; m++) {
                    float v = mel[m, t];
                    if (float.IsNaN(v) || float.IsInfinity(v)) {
                        return double.NaN;
                    }
                    sum += v;
                    count++;
                }
            }
            return count > 0 ? sum / count : double.NaN;
        }

        static bool TouchesProtectedConsonant(
            HifiBoundaryMetadata boundary,
            IReadOnlyList<HifiPhoneDurationPlan> plans,
            int frame,
            int win) {
            if (plans.Count == 0) {
                return false;
            }
            int left = Math.Clamp(boundary.LeftPhoneIndex, 0, plans.Count - 1);
            int right = Math.Clamp(boundary.RightPhoneIndex, 0, plans.Count - 1);
            return OverlapsConsonant(plans[left], frame - win, frame)
                || OverlapsConsonant(plans[right], frame, frame + win);
        }

        static bool OverlapsConsonant(HifiPhoneDurationPlan plan, int start, int end) {
            int consonantStart = plan.AdjustedStartFrame;
            int consonantEnd = plan.AdjustedStartFrame + Math.Max(0, plan.TargetConsonantFrames);
            return Math.Max(start, consonantStart) < Math.Min(end, consonantEnd);
        }

        static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

        static bool IsVowelStable(string phoneme) {
            if (string.IsNullOrWhiteSpace(phoneme)) {
                return false;
            }
            string p = phoneme.Trim().ToLowerInvariant();
            if (p == "sil" || p == "pau" || p == "cl" || p == "br" || p == "r") {
                return false;
            }
            return p.Any(c => "aeiouAEIOU".Contains(c));
        }
    }
}
