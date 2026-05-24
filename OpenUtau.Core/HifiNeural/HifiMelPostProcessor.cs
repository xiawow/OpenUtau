using System;
using System.Collections.Generic;
using System.Linq;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiMelPostProcessor {
        const int ConsonantOnsetRampFrames = 3;
        const float ConsonantOnsetRampMinGain = 0.35f;

        public static void NormalizeWeightedMel(float[,] mel, float[] weights) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            for (int t = 0; t < frames; t++) {
                if (weights[t] <= 0) {
                    continue;
                }
                float inv = 1f / weights[t];
                for (int m = 0; m < bins; m++) {
                    mel[m, t] *= inv;
                }
            }
        }

        public static void SmoothBoundaries(
            float[,] mel,
            IReadOnlyList<int> boundaryFrames,
            int smoothFrames,
            string reason,
            IReadOnlyList<(int start, int end)> protectedRanges) {
            if (smoothFrames <= 0 || boundaryFrames.Count == 0) {
                return;
            }
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var original = (float[,])mel.Clone();
            int radius = Math.Clamp(smoothFrames, 1, 16);
            int applied = 0;
            foreach (int boundary in boundaryFrames.Distinct()) {
                int start = Math.Max(1, boundary - radius);
                int end = Math.Min(frames - 2, boundary + radius);
                if (start > end) {
                    continue;
                }
                if (InsideProtectedRange(boundary, protectedRanges)) {
                    start = Math.Max(boundary - 1, start);
                    end = Math.Min(boundary + 1, end);
                    if (InsideProtectedRange(start, protectedRanges) && InsideProtectedRange(end, protectedRanges)) {
                        continue;
                    }
                }
                applied++;
                for (int t = start; t <= end; t++) {
                    if (InsideProtectedRange(t, protectedRanges)) {
                        continue;
                    }
                    double distance = Math.Abs(t - boundary) / (double)(radius + 1);
                    float strength = (float)(0.5 * (1.0 + Math.Cos(Math.PI * Math.Clamp(distance, 0, 1))));
                    strength *= 0.35f;
                    for (int m = 0; m < bins; m++) {
                        float avg = (original[m, t - 1] + original[m, t] + original[m, t + 1]) / 3f;
                        mel[m, t] = original[m, t] * (1f - strength) + avg * strength;
                    }
                }
            }
            Log.Information(
                "HifiMelPostProcessor boundary_smoothing reason={Reason} boundaries={BoundaryCount} applied={Applied} smooth_frames={SmoothFrames}",
                reason,
                boundaryFrames.Count,
                applied,
                smoothFrames);
        }

        public static void RepairVoicedOnsetDips(
            float[,] mel,
            float[] f0,
            IReadOnlyList<HifiPhoneDurationPlan> plans,
            IReadOnlyList<HifiPhoneFeatureDiagnostic> diagnostics,
            int repairFrames,
            double minDip,
            double maxLift) {
            int frames = mel.GetLength(1);
            if (frames <= 4 || repairFrames <= 0 || minDip <= 0 || maxLift <= 0 || diagnostics.Count == 0) {
                return;
            }

            int bins = mel.GetLength(0);
            var planMap = plans.ToDictionary(plan => plan.Index);
            repairFrames = Math.Clamp(repairFrames, 1, 4);
            float maxLiftValue = (float)Math.Clamp(maxLift, 0.05, 1.5);
            int applied = 0;
            foreach (var diagnostic in diagnostics) {
                if (!planMap.TryGetValue(diagnostic.Index, out var plan)) {
                    continue;
                }
                int start = diagnostic.StartFrame;
                if (start <= 0 || start + 3 >= frames || plan.AdjustedDurationFrames <= 1) {
                    continue;
                }
                if (!IsVoicedNear(f0, start)) {
                    continue;
                }

                int consonantEnd = plan.AdjustedStartFrame + Math.Max(0, plan.TargetConsonantFrames);
                bool insideConsonant = start < consonantEnd;
                bool quietSource = diagnostic.SourceOnsetPeak < 0.012
                    && diagnostic.SourceOnsetRms < 0.006
                    && diagnostic.SourceOnsetMaxJump < 0.004;
                bool sourceStable = diagnostic.SourceMelOnsetDelta < 0.45;
                if (insideConsonant && !quietSource) {
                    continue;
                }
                if (!sourceStable) {
                    continue;
                }

                double stretchDip = diagnostic.StretchedMelAfterMean - diagnostic.StretchedMelOnsetMean;
                double phraseDip = MeanMelRange(mel, start + 2, Math.Min(frames, start + 6))
                    - MeanMelRange(mel, start, Math.Min(frames, start + 2));
                if (Math.Max(stretchDip, phraseDip) < minDip) {
                    continue;
                }

                int count = Math.Min(repairFrames, Math.Min(plan.AdjustedEndFrame, frames) - start);
                count = Math.Min(count, frames - start - 2);
                if (count <= 0) {
                    continue;
                }

                int futureStart = Math.Min(frames - 1, start + 2);
                int futureEnd = Math.Min(frames, start + 6);
                if (futureEnd <= futureStart) {
                    continue;
                }

                for (int t = 0; t < count; t++) {
                    int frame = start + t;
                    float alpha = (float)(t + 1) / (count + 1);
                    float strength = 0.85f - 0.2f * t / Math.Max(1, count - 1);
                    for (int m = 0; m < bins; m++) {
                        float previous = mel[m, start - 1];
                        float future = MeanBin(mel, m, futureStart, futureEnd);
                        float bridge = previous + (future - previous) * alpha;
                        float lift = Math.Clamp(bridge - mel[m, frame], 0f, maxLiftValue);
                        mel[m, frame] += lift * strength;
                    }
                }
                applied++;
                Log.Information(
                    "HifiMelPostProcessor onset_dip_repair phone_index={Index} phoneme={Phoneme} start_frame={StartFrame} repair_frames={RepairFrames} source_peak={SourcePeak:F6} source_mel_delta={SourceMelDelta:F4} stretch_dip={StretchDip:F4} phrase_dip={PhraseDip:F4} inside_consonant={InsideConsonant}",
                    diagnostic.Index,
                    diagnostic.Phoneme,
                    start,
                    count,
                    diagnostic.SourceOnsetPeak,
                    diagnostic.SourceMelOnsetDelta,
                    stretchDip,
                    phraseDip,
                    insideConsonant);
            }
            if (applied > 0) {
                Log.Information(
                    "HifiMelPostProcessor onset_dip_repair summary applied={Applied} repair_frames={RepairFrames} min_dip={MinDip:F3} max_lift={MaxLift:F3}",
                    applied,
                    repairFrames,
                    minDip,
                    maxLift);
            }
        }

        public static void ApplyVoicingMask(float[] f0, IReadOnlyList<HifiPhoneDurationPlan> plans) {
            if (f0.Length == 0 || plans.Count == 0) {
                return;
            }
            int fullyMasked = 0;
            int consonantMasked = 0;
            foreach (var plan in plans) {
                int start = Math.Clamp(plan.AdjustedStartFrame, 0, f0.Length);
                int end = Math.Clamp(plan.AdjustedEndFrame, start, f0.Length);
                if (start >= end) {
                    continue;
                }

                string phoneme = NormalizePhoneme(plan.Phoneme);
                if (IsRestOrSilence(phoneme) || IsPureUnvoiced(phoneme)) {
                    ZeroF0(f0, start, end);
                    fullyMasked++;
                    Log.Information(
                        "HifiMelPostProcessor voicing_mask phoneme={Phoneme} mode=full start_frame={StartFrame} end_frame={EndFrame}",
                        plan.Phoneme,
                        start,
                        end - 1);
                    continue;
                }

                if (StartsWithUnvoicedConsonant(phoneme) && plan.TargetConsonantFrames > 0) {
                    int consonantEnd = Math.Clamp(start + plan.TargetConsonantFrames, start, end);
                    if (consonantEnd > start) {
                        ZeroF0(f0, start, consonantEnd);
                        ApplyConsonantOnsetRamp(f0, consonantEnd, end, plan.Phoneme);
                        consonantMasked++;
                        Log.Information(
                            "HifiMelPostProcessor voicing_mask phoneme={Phoneme} mode=consonant start_frame={StartFrame} end_frame={EndFrame} consonant_frames={ConsonantFrames}",
                            plan.Phoneme,
                            start,
                            consonantEnd - 1,
                            consonantEnd - start);
                    }
                }
            }
            Log.Information(
                "HifiMelPostProcessor voicing_mask summary fully_masked={FullyMasked} consonant_masked={ConsonantMasked}",
                fullyMasked,
                consonantMasked);
        }

        static void ApplyConsonantOnsetRamp(float[] f0, int start, int end, string phoneme) {
            if (start >= end || start >= f0.Length) {
                return;
            }
            int voicedStart = -1;
            for (int i = Math.Clamp(start, 0, f0.Length - 1); i < end && i < f0.Length; i++) {
                if (f0[i] > 0 && !float.IsNaN(f0[i]) && !float.IsInfinity(f0[i])) {
                    voicedStart = i;
                    break;
                }
            }
            if (voicedStart < 0) {
                return;
            }
            int rampEnd = Math.Min(end, Math.Min(f0.Length, voicedStart + ConsonantOnsetRampFrames));
            int rampFrames = rampEnd - voicedStart;
            if (rampFrames <= 0) {
                return;
            }
            for (int i = voicedStart; i < rampEnd; i++) {
                float alpha = (float)(i - voicedStart + 1) / (rampFrames + 1);
                float gain = ConsonantOnsetRampMinGain
                    + (1f - ConsonantOnsetRampMinGain) * (float)Math.Sin(alpha * Math.PI * 0.5);
                f0[i] *= Math.Clamp(gain, 0f, 1f);
            }
            Log.Information(
                "HifiMelPostProcessor voicing_mask_onset_ramp phoneme={Phoneme} start_frame={StartFrame} ramp_frames={RampFrames} min_gain={MinGain:F2}",
                phoneme,
                voicedStart,
                rampFrames,
                ConsonantOnsetRampMinGain);
        }

        public static void SmoothInternalF0Boundaries(
            float[] f0,
            HifiPhraseMetadata metadata,
            int radiusFrames,
            double maxJumpCents) {
            if (f0.Length < 3 || metadata.Notes.Count <= 1 || radiusFrames <= 0) {
                return;
            }
            radiusFrames = Math.Clamp(radiusFrames, 1, 4);
            maxJumpCents = Math.Max(1, maxJumpCents);
            int applied = 0;
            foreach (var note in metadata.Notes.Skip(1)) {
                int boundary = (int)Math.Round((note.PositionMs - metadata.PhraseStartMs) / metadata.FrameMs);
                if (boundary - radiusFrames < 0 || boundary + radiusFrames >= f0.Length) {
                    continue;
                }
                if (HasUnvoicedInWindow(f0, boundary - radiusFrames, boundary + radiusFrames + 1)) {
                    continue;
                }
                double left = f0[boundary - 1];
                double right = f0[boundary];
                if (left <= 0 || right <= 0) {
                    continue;
                }
                double jumpCents = Math.Abs(1200.0 * Math.Log(right / left, 2));
                if (jumpCents < maxJumpCents) {
                    continue;
                }

                float start = f0[boundary - radiusFrames];
                float end = f0[boundary + radiusFrames];
                if (start <= 0 || end <= 0) {
                    continue;
                }
                for (int t = boundary - radiusFrames; t <= boundary + radiusFrames; t++) {
                    double x = (t - (boundary - radiusFrames)) / (double)(radiusFrames * 2);
                    double smooth = 0.5 - 0.5 * Math.Cos(Math.PI * Math.Clamp(x, 0, 1));
                    double logF0 = Math.Log(start) + (Math.Log(end) - Math.Log(start)) * smooth;
                    f0[t] = (float)Math.Exp(logF0);
                }
                applied++;
                Log.Information(
                    "HifiMelPostProcessor f0_boundary_smooth note_index={NoteIndex} lyric={Lyric} boundary_frame={BoundaryFrame} radius_frames={RadiusFrames} jump_cents={JumpCents:F2}",
                    note.Index,
                    note.Lyric,
                    boundary,
                    radiusFrames,
                    jumpCents);
            }
            if (applied > 0) {
                Log.Information(
                    "HifiMelPostProcessor f0_boundary_smooth summary applied={Applied} radius_frames={RadiusFrames} max_jump_cents={MaxJumpCents:F2}",
                    applied,
                    radiusFrames,
                    maxJumpCents);
            }
        }

        public static void ApplyTailRelease(
            float[,] mel,
            float[] f0,
            IReadOnlyList<HifiPhoneDurationPlan> plans,
            int releaseFrames,
            double releaseGainDb,
            int f0FadeFrames) {
            int frames = mel.GetLength(1);
            if (frames <= 1 || releaseFrames <= 0 || plans.Count == 0) {
                return;
            }
            int bins = mel.GetLength(0);
            float minGain = (float)Math.Pow(10, releaseGainDb / 20.0);
            minGain = Math.Clamp(minGain, 0.1f, 1f);
            int applied = 0;
            for (int i = 0; i < plans.Count; i++) {
                var plan = plans[i];
                if (!IsVoicedReleaseCandidate(plan.Phoneme)) {
                    continue;
                }
                if (!ShouldApplyTailRelease(plans, i)) {
                    continue;
                }
                int end = Math.Clamp(plan.AdjustedEndFrame, 1, frames);
                int stableStart = StableStartFrame(plan, frames);
                int start = Math.Max(stableStart, end - releaseFrames);
                if (start >= end) {
                    continue;
                }
                ApplyMelRelease(mel, bins, start, end, minGain);
                ApplyF0Release(f0, start, end, Math.Clamp(f0FadeFrames, 0, releaseFrames));
                applied++;
                Log.Information(
                    "HifiMelPostProcessor tail_release phone_index={Index} phoneme={Phoneme} start_frame={StartFrame} end_frame={EndFrame} release_frames={ReleaseFrames} release_gain_db={ReleaseGainDb:F1} f0_fade_frames={F0FadeFrames}",
                    plan.Index,
                    plan.Phoneme,
                    start,
                    end - 1,
                    end - start,
                    releaseGainDb,
                    f0FadeFrames);
            }
            Log.Information(
                "HifiMelPostProcessor tail_release summary candidates={CandidateCount} applied={Applied}",
                plans.Count,
                applied);
        }

        public static void ApplyHeadAttack(
            float[,] mel,
            float[] f0,
            IReadOnlyList<HifiPhoneDurationPlan> plans,
            int attackFrames,
            double attackGainDb,
            int f0FadeFrames) {
            int frames = mel.GetLength(1);
            if (frames <= 1 || attackFrames <= 0 || plans.Count == 0) {
                return;
            }
            int bins = mel.GetLength(0);
            float minGain = (float)Math.Pow(10, attackGainDb / 20.0);
            minGain = Math.Clamp(minGain, 0.1f, 1f);
            int applied = 0;
            for (int i = 0; i < plans.Count; i++) {
                var plan = plans[i];
                if (!IsVoicedReleaseCandidate(plan.Phoneme)) {
                    continue;
                }
                if (!ShouldApplyHeadAttack(plans, i)) {
                    continue;
                }
                int start = StableStartFrame(plan, frames);
                int end = Math.Min(frames, start + attackFrames);
                if (start >= end) {
                    continue;
                }
                ApplyMelAttack(mel, bins, start, end, minGain);
                ApplyF0Attack(f0, start, end, Math.Clamp(f0FadeFrames, 0, attackFrames));
                applied++;
                Log.Information(
                    "HifiMelPostProcessor head_attack phone_index={Index} phoneme={Phoneme} start_frame={StartFrame} end_frame={EndFrame} attack_frames={AttackFrames} attack_gain_db={AttackGainDb:F1} f0_fade_frames={F0FadeFrames}",
                    plan.Index,
                    plan.Phoneme,
                    start,
                    end - 1,
                    end - start,
                    attackGainDb,
                    f0FadeFrames);
            }
            Log.Information(
                "HifiMelPostProcessor head_attack summary candidates={CandidateCount} applied={Applied}",
                plans.Count,
                applied);
        }

        public static void ApplyVoicedContinuity(
            float[,] mel,
            IReadOnlyList<HifiPhoneDurationPlan> plans,
            IReadOnlyList<(int start, int end)> protectedRanges,
            int radius,
            float strength) {
            int frames = mel.GetLength(1);
            if (frames <= 2 || plans.Count == 0 || radius <= 0 || strength <= 0) {
                return;
            }
            int bins = mel.GetLength(0);
            radius = Math.Clamp(radius, 1, 3);
            strength = Math.Clamp(strength, 0f, 0.35f);
            var voiced = BuildVoicedMask(frames, plans, protectedRanges);
            var original = (float[,])mel.Clone();
            int appliedFrames = 0;
            for (int t = radius; t < frames - radius; t++) {
                if (!voiced[t]) {
                    continue;
                }
                bool windowVoiced = true;
                for (int dt = -radius; dt <= radius; dt++) {
                    windowVoiced &= voiced[t + dt];
                }
                if (!windowVoiced) {
                    continue;
                }
                appliedFrames++;
                for (int m = 0; m < bins; m++) {
                    float sum = 0;
                    for (int dt = -radius; dt <= radius; dt++) {
                        sum += original[m, t + dt];
                    }
                    float avg = sum / (radius * 2 + 1);
                    mel[m, t] = original[m, t] * (1f - strength) + avg * strength;
                }
            }
            SmoothEnergyEnvelope(mel, voiced, protectedRanges, radius: 2, strength: strength * 0.5f);
            Log.Information(
                "HifiMelPostProcessor voiced_continuity applied_frames={AppliedFrames} radius={Radius} strength={Strength:F3}",
                appliedFrames,
                radius,
                strength);
        }

        static void ApplyMelRelease(float[,] mel, int bins, int start, int end, float minGain) {
            for (int t = start; t < end; t++) {
                float alpha = (float)(t - start + 1) / (end - start + 1);
                float curve = (float)(0.5 - 0.5 * Math.Cos(alpha * Math.PI));
                float gain = 1f + (minGain - 1f) * curve;
                for (int m = 0; m < bins; m++) {
                    mel[m, t] += (float)Math.Log(Math.Max(1e-5f, gain));
                }
            }
        }

        static void ZeroF0(float[] f0, int start, int end) {
            for (int i = start; i < end; i++) {
                f0[i] = 0;
            }
        }

        static bool HasUnvoicedInWindow(float[] f0, int start, int end) {
            start = Math.Clamp(start, 0, f0.Length);
            end = Math.Clamp(end, start, f0.Length);
            for (int i = start; i < end; i++) {
                if (f0[i] <= 0 || float.IsNaN(f0[i]) || float.IsInfinity(f0[i])) {
                    return true;
                }
            }
            return false;
        }

        static bool IsVoicedNear(float[] f0, int frame) {
            if (f0.Length == 0) {
                return false;
            }
            int start = Math.Clamp(frame - 1, 0, f0.Length);
            int end = Math.Clamp(frame + 3, start, f0.Length);
            for (int i = start; i < end; i++) {
                if (f0[i] > 0 && !float.IsNaN(f0[i]) && !float.IsInfinity(f0[i])) {
                    return true;
                }
            }
            return false;
        }

        static string NormalizePhoneme(string phoneme) {
            return (phoneme ?? string.Empty).Trim().ToLowerInvariant();
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
            if (string.IsNullOrWhiteSpace(phoneme) || HasVoicedVowelOrNasal(phoneme)) {
                return false;
            }
            return SplitPhonemeTokens(phoneme).All(IsUnvoicedToken);
        }

        static bool StartsWithUnvoicedConsonant(string phoneme) {
            if (string.IsNullOrWhiteSpace(phoneme)) {
                return false;
            }
            var tokens = SplitPhonemeTokens(phoneme);
            if (tokens.Length > 0 && IsUnvoicedToken(tokens[0])) {
                return true;
            }
            return UnvoicedPrefixes.Any(prefix => phoneme.StartsWith(prefix, StringComparison.Ordinal));
        }

        static string[] SplitPhonemeTokens(string phoneme) {
            return phoneme
                .Split(new[] { ' ', '-', '_', '+', '/' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(token => token.Trim())
                .Where(token => token.Length > 0)
                .ToArray();
        }

        static bool HasVoicedVowelOrNasal(string phoneme) {
            return phoneme.Any(c => "aeiou".Contains(c))
                || SplitPhonemeTokens(phoneme).Any(token => token is "m" or "n" or "ng");
        }

        static bool IsUnvoicedToken(string token) {
            return UnvoicedTokens.Contains(token);
        }

        static readonly HashSet<string> UnvoicedTokens = new(StringComparer.Ordinal) {
            "p", "t", "k", "q",
            "s", "sh", "ch", "ts",
            "f", "h", "hh",
            "th",
        };

        static readonly string[] UnvoicedPrefixes = {
            "sh", "ch", "ts", "hh", "th", "p", "t", "k", "q", "s", "f", "h",
        };

        static void ApplyMelAttack(float[,] mel, int bins, int start, int end, float minGain) {
            for (int t = start; t < end; t++) {
                float alpha = (float)(t - start + 1) / (end - start + 1);
                float curve = (float)(0.5 - 0.5 * Math.Cos(alpha * Math.PI));
                float gain = minGain + (1f - minGain) * curve;
                for (int m = 0; m < bins; m++) {
                    mel[m, t] += (float)Math.Log(Math.Max(1e-5f, gain));
                }
            }
        }

        static void ApplyF0Release(float[] f0, int start, int end, int f0FadeFrames) {
            if (f0FadeFrames <= 0) {
                return;
            }
            int fadeStart = Math.Max(start, end - f0FadeFrames);
            for (int t = fadeStart; t < end && t < f0.Length; t++) {
                if (t < 0) {
                    continue;
                }
                float alpha = (float)(t - fadeStart + 1) / (end - fadeStart + 1);
                float gain = (float)Math.Cos(alpha * Math.PI * 0.5);
                f0[t] *= Math.Clamp(gain, 0f, 1f);
            }
        }

        static void ApplyF0Attack(float[] f0, int start, int end, int f0FadeFrames) {
            if (f0FadeFrames <= 0) {
                return;
            }
            int fadeEnd = Math.Min(end, start + f0FadeFrames);
            for (int t = start; t < fadeEnd && t < f0.Length; t++) {
                if (t < 0) {
                    continue;
                }
                float alpha = (float)(t - start + 1) / (fadeEnd - start + 1);
                float gain = (float)Math.Sin(alpha * Math.PI * 0.5);
                f0[t] *= Math.Clamp(gain, 0f, 1f);
            }
        }

        static bool ShouldApplyTailRelease(IReadOnlyList<HifiPhoneDurationPlan> plans, int index) {
            if (index + 1 >= plans.Count) {
                return true;
            }
            var current = plans[index];
            var next = plans[index + 1];
            int gap = next.AdjustedStartFrame - current.AdjustedEndFrame;
            return gap > 2 || IsPhraseBreak(next.Phoneme);
        }

        static bool ShouldApplyHeadAttack(IReadOnlyList<HifiPhoneDurationPlan> plans, int index) {
            if (index <= 0) {
                return true;
            }
            var previous = plans[index - 1];
            var current = plans[index];
            int gap = current.AdjustedStartFrame - previous.AdjustedEndFrame;
            return gap > 2 || IsPhraseBreak(previous.Phoneme);
        }

        static bool IsPhraseBreak(string phoneme) {
            return IsRestOrSilence(NormalizePhoneme(phoneme));
        }

        static int StableStartFrame(HifiPhoneDurationPlan plan, int frames) {
            return Math.Clamp(plan.AdjustedStartFrame + Math.Max(0, plan.TargetConsonantFrames), 0, frames);
        }

        static bool IsVoicedReleaseCandidate(string phoneme) {
            if (string.IsNullOrWhiteSpace(phoneme)) {
                return false;
            }
            string p = phoneme.Trim().ToLowerInvariant();
            if (p == "r" || p == "rest" || p == "sil" || p == "pau" || p == "-" || p.Contains("cl")) {
                return false;
            }
            if (p is "k" or "t" or "p" or "s" or "sh" or "ch" or "ts" or "f" or "h") {
                return false;
            }
            return true;
        }

        static bool[] BuildVoicedMask(int frames, IReadOnlyList<HifiPhoneDurationPlan> plans, IReadOnlyList<(int start, int end)> protectedRanges) {
            var voiced = new bool[frames];
            foreach (var plan in plans) {
                if (!IsVoicedReleaseCandidate(plan.Phoneme)) {
                    continue;
                }
                int start = Math.Clamp(plan.AdjustedStartFrame + Math.Max(0, plan.TargetConsonantFrames), 0, frames);
                int end = Math.Clamp(plan.AdjustedEndFrame, start, frames);
                for (int t = start; t < end; t++) {
                    voiced[t] = !InsideProtectedRange(t, protectedRanges);
                }
            }
            return voiced;
        }

        static void SmoothEnergyEnvelope(float[,] mel, bool[] voiced, IReadOnlyList<(int start, int end)> protectedRanges, int radius, float strength) {
            if (strength <= 0) {
                return;
            }
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var energy = new float[frames];
            for (int t = 0; t < frames; t++) {
                double sum = 0;
                for (int m = 0; m < bins; m++) {
                    sum += mel[m, t];
                }
                energy[t] = (float)(sum / bins);
            }
            var smoothed = new float[frames];
            Array.Copy(energy, smoothed, frames);
            for (int t = radius; t < frames - radius; t++) {
                if (!voiced[t] || InsideProtectedRange(t, protectedRanges)) {
                    continue;
                }
                bool windowVoiced = true;
                for (int dt = -radius; dt <= radius; dt++) {
                    windowVoiced &= voiced[t + dt];
                }
                if (!windowVoiced) {
                    continue;
                }
                float sum = 0;
                for (int dt = -radius; dt <= radius; dt++) {
                    sum += energy[t + dt];
                }
                smoothed[t] = energy[t] * (1f - strength) + (sum / (radius * 2 + 1)) * strength;
            }
            for (int t = 0; t < frames; t++) {
                if (!voiced[t] || InsideProtectedRange(t, protectedRanges)) {
                    continue;
                }
                float offset = Math.Clamp(smoothed[t] - energy[t], -0.08f, 0.08f);
                for (int m = 0; m < bins; m++) {
                    mel[m, t] += offset;
                }
            }
        }

        static double MeanMelRange(float[,] mel, int start, int end) {
            int frames = mel.GetLength(1);
            int bins = mel.GetLength(0);
            start = Math.Clamp(start, 0, frames);
            end = Math.Clamp(end, start, frames);
            if (end <= start) {
                return double.NegativeInfinity;
            }
            double sum = 0;
            int count = 0;
            for (int t = start; t < end; t++) {
                for (int m = 0; m < bins; m++) {
                    sum += mel[m, t];
                    count++;
                }
            }
            return count > 0 ? sum / count : double.NegativeInfinity;
        }

        static float MeanBin(float[,] mel, int bin, int start, int end) {
            int frames = mel.GetLength(1);
            start = Math.Clamp(start, 0, frames);
            end = Math.Clamp(end, start, frames);
            if (end <= start) {
                return mel[bin, Math.Clamp(start, 0, Math.Max(0, frames - 1))];
            }
            double sum = 0;
            for (int t = start; t < end; t++) {
                sum += mel[bin, t];
            }
            return (float)(sum / (end - start));
        }

        static bool InsideProtectedRange(int frame, IReadOnlyList<(int start, int end)> ranges) {
            for (int i = 0; i < ranges.Count; i++) {
                if (frame >= ranges[i].start && frame < ranges[i].end) {
                    return true;
                }
            }
            return false;
        }
    }
}
