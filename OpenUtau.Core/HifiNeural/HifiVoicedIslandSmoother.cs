using System;
using System.Collections.Generic;
using System.Linq;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiVoicedIslandSmoother {
        sealed class Island {
            public int StartPlan { get; init; }
            public int EndPlan { get; init; }
            public int StartFrame { get; init; }
            public int EndFrame { get; init; }
        }

        readonly record struct MelAnchor(double Energy, float[] Shape);

        public static void Apply(
            float[,] mel,
            float[] f0,
            IReadOnlyList<HifiPhoneDurationPlan> plans,
            int morphFrames,
            int smoothRadiusFrames,
            float morphStrength,
            float smoothStrength,
            int maxF0BridgeFrames,
            double maxF0BridgeCents,
            int shortPhoneFrames = 0,
            int minStableFrames = 3) {
            if (mel.Length == 0 || plans.Count == 0) {
                return;
            }
            morphFrames = Math.Clamp(morphFrames, 0, 12);
            smoothRadiusFrames = Math.Clamp(smoothRadiusFrames, 0, 8);
            morphStrength = Math.Clamp(morphStrength, 0f, 1f);
            smoothStrength = Math.Clamp(smoothStrength, 0f, 0.5f);
            maxF0BridgeFrames = Math.Clamp(maxF0BridgeFrames, 0, 4);
            maxF0BridgeCents = Math.Clamp(maxF0BridgeCents, 1, 1200);
            shortPhoneFrames = Math.Clamp(shortPhoneFrames, 0, 32);
            minStableFrames = Math.Clamp(minStableFrames, 1, 16);

            var islands = BuildIslands(plans, mel.GetLength(1));
            var stableMask = BuildStableMask(plans, mel.GetLength(1), shortPhoneFrames, minStableFrames);
            int morphedBoundaries = morphFrames > 0 && morphStrength > 0
                ? ApplyBoundaryMorphs(mel, plans, islands, morphFrames, morphStrength, shortPhoneFrames, minStableFrames)
                : 0;
            int smoothedFrames = smoothRadiusFrames > 0 && smoothStrength > 0
                ? SmoothStableTrajectories(mel, stableMask, islands, smoothRadiusFrames, smoothStrength)
                : 0;
            int bridgedF0Frames = maxF0BridgeFrames > 0
                ? BridgeShortF0Gaps(f0, BuildBridgeableMask(plans, f0.Length, shortPhoneFrames, minStableFrames), maxF0BridgeFrames, maxF0BridgeCents)
                : 0;

            Log.Information(
                "HifiVoicedIslandSmoother summary islands={Islands} morph_frames={MorphFrames} morphed_boundaries={MorphedBoundaries} smooth_radius={SmoothRadius} smoothed_frames={SmoothedFrames} f0_bridge_frames={F0BridgeFrames}",
                islands.Count,
                morphFrames,
                morphedBoundaries,
                smoothRadiusFrames,
                smoothedFrames,
                bridgedF0Frames);
        }

        static IReadOnlyList<Island> BuildIslands(IReadOnlyList<HifiPhoneDurationPlan> plans, int frames) {
            var islands = new List<Island>();
            int startPlan = -1;
            int startFrame = 0;
            for (int i = 0; i < plans.Count; i++) {
                bool barrier = IsIslandBarrier(plans[i].Phoneme);
                if (barrier) {
                    if (startPlan >= 0) {
                        islands.Add(new Island {
                            StartPlan = startPlan,
                            EndPlan = i - 1,
                            StartFrame = startFrame,
                            EndFrame = Math.Clamp(plans[i - 1].AdjustedEndFrame, startFrame, frames),
                        });
                        startPlan = -1;
                    }
                    continue;
                }
                if (startPlan < 0) {
                    startPlan = i;
                    startFrame = Math.Clamp(plans[i].AdjustedStartFrame, 0, frames);
                }
            }
            if (startPlan >= 0) {
                islands.Add(new Island {
                    StartPlan = startPlan,
                    EndPlan = plans.Count - 1,
                    StartFrame = startFrame,
                    EndFrame = Math.Clamp(plans[^1].AdjustedEndFrame, startFrame, frames),
                });
            }
            return islands;
        }

        static int ApplyBoundaryMorphs(
            float[,] mel,
            IReadOnlyList<HifiPhoneDurationPlan> plans,
            IReadOnlyList<Island> islands,
            int morphFrames,
            float strength,
            int shortPhoneFrames,
            int minStableFrames) {
            int applied = 0;
            foreach (var island in islands) {
                for (int i = island.StartPlan + 1; i <= island.EndPlan; i++) {
                    var left = plans[i - 1];
                    var right = plans[i];
                    if (IsShortPlan(left, shortPhoneFrames) || IsShortPlan(right, shortPhoneFrames)) {
                        continue;
                    }
                    if (!HasStableVoice(left.Phoneme) || !HasStableVoice(right.Phoneme)) {
                        continue;
                    }
                    int leftStableStart = Math.Clamp(left.AdjustedStartFrame + Math.Max(0, left.TargetConsonantFrames), 0, mel.GetLength(1));
                    int leftStableEnd = Math.Clamp(left.AdjustedEndFrame, leftStableStart, mel.GetLength(1));
                    int rightStableStart = Math.Clamp(right.AdjustedStartFrame + Math.Max(0, right.TargetConsonantFrames), 0, mel.GetLength(1));
                    int rightStableEnd = Math.Clamp(right.AdjustedEndFrame, rightStableStart, mel.GetLength(1));
                    if (leftStableEnd - leftStableStart < minStableFrames || rightStableEnd - rightStableStart < minStableFrames) {
                        continue;
                    }
                    int leftStart = Math.Max(leftStableStart, leftStableEnd - morphFrames);
                    int rightEnd = Math.Min(rightStableEnd, rightStableStart + morphFrames);
                    if (leftStart >= leftStableEnd || rightStableStart >= rightEnd) {
                        continue;
                    }

                    var leftAnchor = AverageAnchor(mel, Math.Max(leftStableStart, leftStableEnd - 3), leftStableEnd);
                    var rightAnchor = AverageAnchor(mel, rightStableStart, Math.Min(rightStableEnd, rightStableStart + 3));
                    int totalSpan = (leftStableEnd - leftStart) + Math.Max(0, rightStableStart - leftStableEnd) + (rightEnd - rightStableStart);
                    if (totalSpan <= 1) {
                        continue;
                    }
                    ApplyMorphRegion(mel, leftStart, leftStableEnd, leftStart, totalSpan, leftAnchor, rightAnchor, strength);
                    ApplyMorphRegion(mel, rightStableStart, rightEnd, leftStart, totalSpan, leftAnchor, rightAnchor, strength);
                    applied++;
                    Log.Information(
                        "HifiVoicedIslandSmoother boundary_morph left={Left} right={Right} left_start={LeftStart} left_end={LeftEnd} right_start={RightStart} right_end={RightEnd} gap_frames={GapFrames} strength={Strength:F3}",
                        left.Phoneme,
                        right.Phoneme,
                        leftStart,
                        leftStableEnd,
                        rightStableStart,
                        rightEnd,
                        Math.Max(0, rightStableStart - leftStableEnd),
                        strength);
                }
            }
            return applied;
        }

        static void ApplyMorphRegion(
            float[,] mel,
            int start,
            int end,
            int globalStart,
            int totalSpan,
            MelAnchor leftAnchor,
            MelAnchor rightAnchor,
            float strength) {
            int bins = mel.GetLength(0);
            for (int t = start; t < end; t++) {
                double x = (t - globalStart) / (double)Math.Max(1, totalSpan - 1);
                x = 0.5 - 0.5 * Math.Cos(Math.PI * Math.Clamp(x, 0, 1));
                double targetEnergy = leftAnchor.Energy + (rightAnchor.Energy - leftAnchor.Energy) * x;
                for (int m = 0; m < bins; m++) {
                    double targetShape = leftAnchor.Shape[m] + (rightAnchor.Shape[m] - leftAnchor.Shape[m]) * x;
                    float target = (float)(targetEnergy + targetShape);
                    mel[m, t] = mel[m, t] * (1f - strength) + target * strength;
                }
            }
        }

        static int SmoothStableTrajectories(
            float[,] mel,
            bool[] stableMask,
            IReadOnlyList<Island> islands,
            int radius,
            float strength) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var original = (float[,])mel.Clone();
            var energy = MeasureEnergy(original);
            int appliedFrames = 0;
            foreach (var island in islands) {
                int start = Math.Clamp(island.StartFrame, 0, frames);
                int end = Math.Clamp(island.EndFrame, start, frames);
                for (int t = start; t < end; t++) {
                    if (!stableMask[t]) {
                        continue;
                    }
                    double weightSum = 0;
                    double energySum = 0;
                    var shapeSum = new double[bins];
                    for (int i = Math.Max(start, t - radius); i <= Math.Min(end - 1, t + radius); i++) {
                        if (!stableMask[i]) {
                            continue;
                        }
                        double distance = Math.Abs(i - t) / (double)(radius + 1);
                        double weight = 0.5 + 0.5 * Math.Cos(Math.PI * Math.Clamp(distance, 0, 1));
                        weightSum += weight;
                        energySum += energy[i] * weight;
                        for (int m = 0; m < bins; m++) {
                            shapeSum[m] += (original[m, i] - energy[i]) * weight;
                        }
                    }
                    if (weightSum <= 0) {
                        continue;
                    }
                    double smoothedEnergy = energySum / weightSum;
                    for (int m = 0; m < bins; m++) {
                        float target = (float)(smoothedEnergy + shapeSum[m] / weightSum);
                        mel[m, t] = original[m, t] * (1f - strength) + target * strength;
                    }
                    appliedFrames++;
                }
            }
            return appliedFrames;
        }

        static int BridgeShortF0Gaps(float[] f0, bool[] bridgeable, int maxGapFrames, double maxJumpCents) {
            if (f0.Length == 0) {
                return 0;
            }
            int bridged = 0;
            int i = 0;
            while (i < f0.Length) {
                if (f0[i] > 0 || !bridgeable[i]) {
                    i++;
                    continue;
                }
                int start = i;
                while (i < f0.Length && f0[i] <= 0 && bridgeable[i]) {
                    i++;
                }
                int end = i;
                int length = end - start;
                if (length <= 0 || length > maxGapFrames || start <= 0 || end >= f0.Length) {
                    continue;
                }
                float left = f0[start - 1];
                float right = f0[end];
                if (left <= 0 || right <= 0) {
                    continue;
                }
                double jumpCents = Math.Abs(1200.0 * Math.Log(right / left, 2));
                if (jumpCents > maxJumpCents) {
                    continue;
                }
                for (int t = start; t < end; t++) {
                    double x = (t - start + 1) / (double)(length + 1);
                    double logF0 = Math.Log(left) + (Math.Log(right) - Math.Log(left)) * x;
                    f0[t] = (float)Math.Exp(logF0);
                    bridged++;
                }
                Log.Information(
                    "HifiVoicedIslandSmoother f0_bridge start_frame={StartFrame} end_frame={EndFrame} frames={Frames} left_f0={LeftF0:F2} right_f0={RightF0:F2} jump_cents={JumpCents:F2}",
                    start,
                    end - 1,
                    length,
                    left,
                    right,
                    jumpCents);
            }
            return bridged;
        }

        static bool[] BuildStableMask(IReadOnlyList<HifiPhoneDurationPlan> plans, int frames, int shortPhoneFrames, int minStableFrames) {
            var mask = new bool[frames];
            foreach (var plan in plans) {
                if (IsShortPlan(plan, shortPhoneFrames)) {
                    continue;
                }
                if (!HasStableVoice(plan.Phoneme)) {
                    continue;
                }
                int start = Math.Clamp(plan.AdjustedStartFrame + Math.Max(0, plan.TargetConsonantFrames), 0, frames);
                int end = Math.Clamp(plan.AdjustedEndFrame, start, frames);
                if (end - start < minStableFrames) {
                    continue;
                }
                for (int t = start; t < end; t++) {
                    mask[t] = true;
                }
            }
            return mask;
        }

        static bool[] BuildBridgeableMask(IReadOnlyList<HifiPhoneDurationPlan> plans, int frames, int shortPhoneFrames, int minStableFrames) {
            var mask = new bool[frames];
            foreach (var plan in plans) {
                if (IsIslandBarrier(plan.Phoneme)) {
                    continue;
                }
                if (IsShortPlan(plan, shortPhoneFrames) || !HasStableVoice(plan.Phoneme)) {
                    continue;
                }
                int start = Math.Clamp(plan.AdjustedStartFrame + Math.Max(0, plan.TargetConsonantFrames), 0, frames);
                int end = Math.Clamp(plan.AdjustedEndFrame, start, frames);
                if (end - start < minStableFrames) {
                    continue;
                }
                for (int t = start; t < end; t++) {
                    mask[t] = true;
                }
            }
            return mask;
        }

        static bool IsShortPlan(HifiPhoneDurationPlan plan, int shortPhoneFrames) {
            return shortPhoneFrames > 0 && plan.AdjustedDurationFrames <= shortPhoneFrames;
        }

        static MelAnchor AverageAnchor(float[,] mel, int start, int end) {
            int bins = mel.GetLength(0);
            start = Math.Clamp(start, 0, mel.GetLength(1));
            end = Math.Clamp(end, start, mel.GetLength(1));
            double energy = 0;
            var values = new float[bins];
            int frames = Math.Max(1, end - start);
            for (int t = start; t < end; t++) {
                double frameEnergy = 0;
                for (int m = 0; m < bins; m++) {
                    frameEnergy += mel[m, t];
                }
                energy += frameEnergy / bins;
                for (int m = 0; m < bins; m++) {
                    values[m] += mel[m, t];
                }
            }
            energy /= frames;
            for (int m = 0; m < bins; m++) {
                values[m] = (float)(values[m] / frames - energy);
            }
            return new MelAnchor(energy, values);
        }

        static double[] MeasureEnergy(float[,] mel) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var energy = new double[frames];
            for (int t = 0; t < frames; t++) {
                double sum = 0;
                for (int m = 0; m < bins; m++) {
                    sum += mel[m, t];
                }
                energy[t] = sum / bins;
            }
            return energy;
        }

        static bool IsIslandBarrier(string phoneme) {
            string p = Normalize(phoneme);
            return IsRestOrSilence(p) || IsPureUnvoiced(p);
        }

        static bool HasStableVoice(string phoneme) {
            string p = Normalize(phoneme);
            return !IsRestOrSilence(p) && HasVowelOrNasal(p);
        }

        static bool IsRestOrSilence(string phoneme) {
            return phoneme == "r"
                || phoneme == "rest"
                || phoneme == "sil"
                || phoneme == "pau"
                || phoneme == "-"
                || phoneme == "br"
                || phoneme.Contains("cl");
        }

        static bool IsPureUnvoiced(string phoneme) {
            if (string.IsNullOrWhiteSpace(phoneme) || HasVowelOrNasal(phoneme)) {
                return false;
            }
            return SplitTokens(phoneme).All(token => UnvoicedTokens.Contains(token));
        }

        static bool HasVowelOrNasal(string phoneme) {
            return phoneme.Any(c => "aeiou".Contains(c))
                || SplitTokens(phoneme).Any(token => token is "m" or "n" or "ng");
        }

        static string[] SplitTokens(string phoneme) {
            return phoneme
                .Split(new[] { ' ', '-', '_', '+', '/' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(token => token.Trim())
                .Where(token => token.Length > 0)
                .ToArray();
        }

        static string Normalize(string phoneme) => (phoneme ?? string.Empty).Trim().ToLowerInvariant();

        static readonly HashSet<string> UnvoicedTokens = new(StringComparer.Ordinal) {
            "p", "t", "k", "q",
            "s", "sh", "ch", "ts",
            "f", "h", "hh",
            "th",
        };
    }
}
