using System;
using System.Collections.Generic;
using System.Linq;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public enum HifiStretchMode {
        Simple,
        ConsonantVowelSplit,
    }
    public enum HifiConsonantLockMode {
        Preserve,
        Readable,
        Off,
    }

    public sealed class HifiPhoneStretchDebug {
        public string Phoneme { get; init; } = string.Empty;
        public int SourceFrames { get; init; }
        public int TargetFrames { get; init; }
        public double ConsonantMs { get; init; }
        public int ConsonantFrames { get; init; }
        public int VowelSourceFrames { get; init; }
        public int VowelTargetFrames { get; init; }
        public double StretchRatio { get; init; }
        public int LoopStartFrame { get; init; }
        public int LoopEndFrame { get; init; }
        public int LoopCount { get; init; }
        public int CrossfadeFrames { get; init; }
        public int SourceConsonantFrames { get; init; }
        public int TargetConsonantFrames { get; init; }
        public double StretchRatioConsonant { get; init; }
        public int SourceVowelFrames { get; init; }
        public int TargetVowelFrames { get; init; }
        public double StretchRatioVowel { get; init; }
        public string ConsonantLockMode { get; init; } = string.Empty;
        public int BorrowedFrames { get; init; }
        public string BorrowedFramesAppliedTo { get; init; } = "none";
        public double StretchEnergyCutDb { get; init; }
        public double StretchEnergySourceMean { get; init; }
        public double StretchEnergyTargetMean { get; init; }
        public string VowelStretchStrategy { get; init; } = string.Empty;
        public int F0StableStartFrame { get; init; }
        public int F0StableEndFrame { get; init; }
        public double F0MaxSlopeCents { get; init; }
        public double F0MeanSlopeCents { get; init; }
        public string FallbackReason { get; init; } = string.Empty;
    }

    public sealed class HifiPhoneStretchResult {
        public required float[,] Mel { get; init; }
        public required HifiPhoneStretchDebug Debug { get; init; }
    }

    public static class HifiPhoneMelStretcher {
        const int MinSourceFrames = 12;
        const int MinVowelSourceFrames = 6;
        const int MinVowelTargetFrames = 1;
        const double MaxStretchRatio = 8.0;
        const double MinStretchRatio = 0.25;

        readonly record struct StretchEnergyReport(
            double CutDb,
            double SourceMean,
            double TargetMean,
            string Reason);

        readonly record struct VowelMapReport(
            int LoopStartFrame,
            int LoopEndFrame,
            int LoopCount,
            int CrossfadeFrames,
            string Strategy,
            int F0StableStartFrame,
            int F0StableEndFrame,
            double F0MaxSlopeCents,
            double F0MeanSlopeCents);

        public static HifiPhoneStretchResult Stretch(
            float[,] sourceMel,
            int targetFrames,
            string phoneme,
            double? consonantMs,
            HifiStretchMode mode,
            int? targetConsonantFrames = null,
            int borrowedFrames = 0,
            string borrowedFramesAppliedTo = "none",
            HifiConsonantLockMode? lockModeOverride = null,
            float[]? targetF0 = null) {
            var lockMode = lockModeOverride ?? HifiNeuralConfig.ConsonantLockMode;
            return mode == HifiStretchMode.ConsonantVowelSplit
                ? StretchConsonantVowelSplit(sourceMel, targetFrames, phoneme, consonantMs, targetConsonantFrames, borrowedFrames, borrowedFramesAppliedTo, lockMode, targetF0)
                : Simple(sourceMel, targetFrames, phoneme, consonantMs, "mode=simple");
        }

        static HifiPhoneStretchResult StretchConsonantVowelSplit(
            float[,] sourceMel,
            int targetFrames,
            string phoneme,
            double? consonantMs,
            int? targetConsonantFrames,
            int borrowedFrames,
            string borrowedFramesAppliedTo,
            HifiConsonantLockMode lockMode,
            float[]? targetF0) {
            int sourceFrames = sourceMel.GetLength(1);
            double ratio = SourceToTargetDurationRatio(sourceFrames, targetFrames);
            int rawSourceConsonantFrames = consonantMs.HasValue
                ? (int)Math.Round(consonantMs.Value * HifiMelExtractor.SampleRate / 1000.0 / HifiMelExtractor.OriginHopSize)
                : 0;
            int sourceConsonantFrames = NormalizeSourceConsonantFrames(rawSourceConsonantFrames, sourceFrames, phoneme);
            int vowelSourceFrames = sourceFrames - sourceConsonantFrames;
            int naturalTargetConsonantFrames = consonantMs.HasValue
                ? HifiDurationRedistributor.MsToFrames(consonantMs.Value)
                : SourceFramesToTargetFrames(sourceConsonantFrames);

            string fallback = GetSplitFallbackReason(sourceMel, targetFrames, consonantMs, sourceConsonantFrames, vowelSourceFrames, ratio);
            if (!string.IsNullOrEmpty(fallback)) {
                return Simple(sourceMel, targetFrames, phoneme, consonantMs, fallback);
            }

            int consonantTargetFrames = ResolveConsonantTargetFrames(
                sourceConsonantFrames,
                naturalTargetConsonantFrames,
                targetFrames,
                targetConsonantFrames,
                lockMode);
            int vowelTargetFrames = targetFrames - consonantTargetFrames;
            if (vowelTargetFrames < MinVowelTargetFrames) {
                vowelTargetFrames = MinVowelTargetFrames;
                consonantTargetFrames = Math.Max(0, targetFrames - vowelTargetFrames);
            }
            double consonantDurationRatio = SourceToTargetDurationRatio(sourceConsonantFrames, consonantTargetFrames);
            if (sourceConsonantFrames > 0 && consonantDurationRatio > 1.15) {
                Log.Warning(
                    "HifiPhoneMelStretcher consonant stretched phoneme={Phoneme} source_consonant_frames={SourceConsonantFrames} target_consonant_frames={TargetConsonantFrames} stretch_ratio_consonant={StretchRatioConsonant:F4} mode={Mode}",
                    phoneme,
                    sourceConsonantFrames,
                    consonantTargetFrames,
                    consonantDurationRatio,
                    lockMode.ToString().ToLowerInvariant());
            }

            var output = new float[sourceMel.GetLength(0), targetFrames];
            CopyResampled(sourceMel, 0, sourceConsonantFrames, output, 0, consonantTargetFrames);

            var vowelMap = WriteVowelSourceToTargetMap(
                sourceMel,
                sourceConsonantFrames,
                vowelSourceFrames,
                output,
                consonantTargetFrames,
                vowelTargetFrames,
                targetF0,
                phoneme);
            ApplyConsonantVowelMicroBridge(output, consonantTargetFrames);
            double vowelDurationRatio = SourceToTargetDurationRatio(vowelSourceFrames, vowelTargetFrames);
            var energyReport = ApplyStretchEnergyCompensation(
                sourceMel,
                sourceConsonantFrames,
                vowelSourceFrames,
                output,
                consonantTargetFrames,
                vowelTargetFrames,
                phoneme,
                vowelDurationRatio,
                "vowel",
                targetF0,
                consonantTargetFrames,
                vowelMap.LoopCount);

            string postFallback = ValidateOutput(output);
            if (!string.IsNullOrEmpty(postFallback)) {
                return Simple(sourceMel, targetFrames, phoneme, consonantMs, postFallback);
            }

            var debug = new HifiPhoneStretchDebug {
                Phoneme = phoneme,
                SourceFrames = sourceFrames,
                TargetFrames = targetFrames,
                ConsonantMs = consonantMs.Value,
                ConsonantFrames = sourceConsonantFrames,
                VowelSourceFrames = vowelSourceFrames,
                VowelTargetFrames = vowelTargetFrames,
                StretchRatio = ratio,
                LoopStartFrame = vowelMap.LoopStartFrame,
                LoopEndFrame = vowelMap.LoopEndFrame,
                LoopCount = vowelMap.LoopCount,
                CrossfadeFrames = vowelMap.CrossfadeFrames,
                SourceConsonantFrames = sourceConsonantFrames,
                TargetConsonantFrames = consonantTargetFrames,
                StretchRatioConsonant = consonantDurationRatio,
                SourceVowelFrames = vowelSourceFrames,
                TargetVowelFrames = vowelTargetFrames,
                StretchRatioVowel = vowelDurationRatio,
                ConsonantLockMode = lockMode.ToString().ToLowerInvariant(),
                BorrowedFrames = borrowedFrames,
                BorrowedFramesAppliedTo = borrowedFramesAppliedTo,
                StretchEnergyCutDb = energyReport.CutDb,
                StretchEnergySourceMean = energyReport.SourceMean,
                StretchEnergyTargetMean = energyReport.TargetMean,
                VowelStretchStrategy = vowelMap.Strategy,
                F0StableStartFrame = vowelMap.F0StableStartFrame,
                F0StableEndFrame = vowelMap.F0StableEndFrame,
                F0MaxSlopeCents = vowelMap.F0MaxSlopeCents,
                F0MeanSlopeCents = vowelMap.F0MeanSlopeCents,
            };
            LogDebug(debug);
            return new HifiPhoneStretchResult {
                Mel = output,
                Debug = debug,
            };
        }

        static HifiPhoneStretchResult Simple(float[,] sourceMel, int targetFrames, string phoneme, double? consonantMs, string fallbackReason) {
            int outputFrames = Math.Max(1, targetFrames);
            var output = new float[sourceMel.GetLength(0), outputFrames];
            CopyResampled(sourceMel, 0, sourceMel.GetLength(1), output, 0, outputFrames);
            double durationRatio = SourceToTargetDurationRatio(sourceMel.GetLength(1), outputFrames);
            var energyReport = ApplyStretchEnergyCompensation(
                sourceMel,
                0,
                sourceMel.GetLength(1),
                output,
                0,
                outputFrames,
                phoneme,
                durationRatio,
                "simple",
                targetF0: null,
                targetF0Start: 0,
                loopCount: 0);
            var debug = new HifiPhoneStretchDebug {
                Phoneme = phoneme,
                SourceFrames = sourceMel.GetLength(1),
                TargetFrames = targetFrames,
                ConsonantMs = consonantMs ?? 0,
                ConsonantFrames = consonantMs.HasValue
                    ? (int)Math.Round(consonantMs.Value * HifiMelExtractor.SampleRate / 1000.0 / HifiMelExtractor.OriginHopSize)
                    : 0,
                VowelSourceFrames = 0,
                VowelTargetFrames = 0,
                StretchRatio = SourceToTargetDurationRatio(sourceMel.GetLength(1), targetFrames),
                LoopStartFrame = 0,
                LoopEndFrame = 0,
                LoopCount = 0,
                CrossfadeFrames = 0,
                SourceConsonantFrames = consonantMs.HasValue
                    ? (int)Math.Round(consonantMs.Value * HifiMelExtractor.SampleRate / 1000.0 / HifiMelExtractor.OriginHopSize)
                    : 0,
                TargetConsonantFrames = 0,
                StretchRatioConsonant = 0,
                SourceVowelFrames = 0,
                TargetVowelFrames = 0,
                StretchRatioVowel = 0,
                ConsonantLockMode = HifiNeuralConfig.ConsonantLockMode.ToString().ToLowerInvariant(),
                BorrowedFrames = 0,
                BorrowedFramesAppliedTo = "none",
                StretchEnergyCutDb = energyReport.CutDb,
                StretchEnergySourceMean = energyReport.SourceMean,
                StretchEnergyTargetMean = energyReport.TargetMean,
                VowelStretchStrategy = "simple",
                FallbackReason = fallbackReason,
            };
            Log.Warning(
                "HifiPhoneMelStretcher fallback phoneme={Phoneme} reason={Reason} consonant_protection_effective=false stretch_ratio_consonant={StretchRatioConsonant:F4}",
                phoneme,
                fallbackReason,
                debug.StretchRatioConsonant);
            LogDebug(debug);
            return new HifiPhoneStretchResult {
                Mel = output,
                Debug = debug,
            };
        }

        static string GetSplitFallbackReason(
            float[,] sourceMel,
            int targetFrames,
            double? consonantMs,
            int consonantFrames,
            int vowelSourceFrames,
            double ratio) {
            int sourceFrames = sourceMel.GetLength(1);
            if (targetFrames <= 0) return "target_frames<=0";
            if (sourceFrames < MinSourceFrames) return "source_frames<12";
            if (!consonantMs.HasValue || consonantMs.Value <= 0) return "missing_consonant_ms";
            if (consonantFrames <= 0) return "consonant_frames<=0";
            if (vowelSourceFrames < MinVowelSourceFrames) return "vowel_source_frames<6";
            if (ratio > MaxStretchRatio) return "stretch_ratio>8";
            if (ratio < MinStretchRatio) return "stretch_ratio<0.25";
            return ValidateOutput(sourceMel);
        }

        static int NormalizeSourceConsonantFrames(int sourceConsonantFrames, int sourceFrames, string phoneme) {
            if (sourceFrames <= 0) {
                return 0;
            }
            sourceConsonantFrames = Math.Clamp(sourceConsonantFrames, 0, sourceFrames - 1);
            int maxConsonant = Math.Max(1, sourceFrames - MinVowelSourceFrames);
            if (sourceConsonantFrames > maxConsonant) {
                Log.Information(
                    "HifiPhoneMelStretcher source_consonant_clamped phoneme={Phoneme} source_frames={SourceFrames} original_source_consonant_frames={OriginalSourceConsonantFrames} clamped_source_consonant_frames={ClampedSourceConsonantFrames} min_vowel_source_frames={MinVowelSourceFrames}",
                    phoneme,
                    sourceFrames,
                    sourceConsonantFrames,
                    maxConsonant,
                    MinVowelSourceFrames);
                sourceConsonantFrames = maxConsonant;
            }
            return sourceConsonantFrames;
        }

        static int ResolveConsonantTargetFrames(
            int sourceConsonantFrames,
            int naturalTargetConsonantFrames,
            int targetFrames,
            int? targetConsonantFramesFromPlan,
            HifiConsonantLockMode mode) {
            int max = Math.Max(0, targetFrames - MinVowelTargetFrames);
            if (mode == HifiConsonantLockMode.Preserve) {
                int natural = naturalTargetConsonantFrames > 0
                    ? naturalTargetConsonantFrames
                    : SourceFramesToTargetFrames(sourceConsonantFrames);
                int plannedLimit = targetConsonantFramesFromPlan ?? natural;
                return Math.Clamp(Math.Min(natural, plannedLimit), 0, max);
            }
            if (mode == HifiConsonantLockMode.Off) {
                int raw = targetConsonantFramesFromPlan ?? naturalTargetConsonantFrames;
                return Math.Clamp(raw, 0, max);
            }
            int readable = targetConsonantFramesFromPlan ?? naturalTargetConsonantFrames;
            return Math.Clamp(readable, 0, max);
        }

        static int SourceFramesToTargetFrames(int sourceFrames) {
            if (sourceFrames <= 0) {
                return 0;
            }
            double ms = sourceFrames * HifiMelExtractor.OriginHopSize * 1000.0 / HifiMelExtractor.SampleRate;
            return Math.Max(1, (int)Math.Round(ms / HifiF0Builder.FrameMs));
        }

        static double SourceToTargetDurationRatio(int sourceFrames, int targetFrames) {
            if (sourceFrames <= 0) {
                return targetFrames > 0 ? double.PositiveInfinity : 0;
            }
            double sourceMs = sourceFrames * HifiMelExtractor.OriginHopSize * 1000.0 / HifiMelExtractor.SampleRate;
            double targetMs = targetFrames * HifiF0Builder.FrameMs;
            return sourceMs > 0 ? targetMs / sourceMs : 0;
        }

        static VowelMapReport WriteVowelSourceToTargetMap(
            float[,] sourceMel,
            int vowelStart,
            int vowelSourceFrames,
            float[,] output,
            int dstStart,
            int dstFrames,
            float[]? targetF0,
            string phoneme) {
            if (dstFrames <= 0) {
                return EmptyVowelMapReport("empty_vowel_target");
            }
            ResolveVowelSections(vowelSourceFrames, out int onsetFrames, out int releaseFrames, out int sourceSustainFrames);
            var (targetOnsetFrames, targetSustainFrames, targetReleaseFrames) = AllocateVowelTargetSections(
                onsetFrames,
                sourceSustainFrames,
                releaseFrames,
                dstFrames);

            if (targetOnsetFrames > 0) {
                WriteMappedRegion(
                    sourceMel,
                    vowelStart,
                    onsetFrames,
                    output,
                    dstStart,
                    targetOnsetFrames);
            }
            if (targetSustainFrames > 0) {
                WriteMappedRegion(
                    sourceMel,
                    vowelStart + onsetFrames,
                    sourceSustainFrames,
                    output,
                    dstStart + targetOnsetFrames,
                    targetSustainFrames);
            }
            if (targetReleaseFrames > 0) {
                WriteMappedRegion(
                    sourceMel,
                    vowelStart + vowelSourceFrames - releaseFrames,
                    releaseFrames,
                    output,
                    dstStart + targetOnsetFrames + targetSustainFrames,
                    targetReleaseFrames);
            }

            string strategy = dstFrames == vowelSourceFrames
                ? "continuous_equal"
                : dstFrames > vowelSourceFrames
                    ? "continuous_sustain_stretch"
                    : "area_compress";
            if (targetF0 != null && targetF0.Length > 1 && HasFastF0Motion(targetF0, dstStart, dstFrames)) {
                strategy += "_f0_motion";
            }
            return new VowelMapReport(
                vowelStart + onsetFrames,
                vowelStart + onsetFrames + sourceSustainFrames,
                0,
                0,
                strategy,
                0,
                0,
                0,
                0);
        }

        static void ResolveVowelSections(int vowelSourceFrames, out int onsetFrames, out int releaseFrames, out int sustainFrames) {
            if (vowelSourceFrames <= 3) {
                onsetFrames = Math.Max(1, vowelSourceFrames - 1);
                releaseFrames = 0;
                sustainFrames = Math.Max(1, vowelSourceFrames - onsetFrames);
                return;
            }
            onsetFrames = Math.Clamp((int)Math.Round(vowelSourceFrames * 0.10), 1, 5);
            releaseFrames = Math.Clamp((int)Math.Round(vowelSourceFrames * 0.08), 1, 5);
            while (onsetFrames + releaseFrames > vowelSourceFrames - 2 && (onsetFrames > 1 || releaseFrames > 0)) {
                if (onsetFrames >= releaseFrames && onsetFrames > 1) {
                    onsetFrames--;
                } else if (releaseFrames > 0) {
                    releaseFrames--;
                } else {
                    break;
                }
            }
            sustainFrames = Math.Max(1, vowelSourceFrames - onsetFrames - releaseFrames);
        }

        static (int Onset, int Sustain, int Release) AllocateVowelTargetSections(
            int sourceOnsetFrames,
            int sourceSustainFrames,
            int sourceReleaseFrames,
            int targetTotalFrames) {
            if (targetTotalFrames <= 0) {
                return (0, 0, 0);
            }
            if (targetTotalFrames == 1) {
                return sourceOnsetFrames > 0 ? (1, 0, 0) : (0, 1, 0);
            }
            if (targetTotalFrames <= 8) {
                int quickOnsetFrames = sourceOnsetFrames > 0 ? 1 : 0;
                int quickReleaseFrames = sourceReleaseFrames > 0 && targetTotalFrames >= 6 ? 1 : 0;
                int quickSustainFrames = Math.Max(1, targetTotalFrames - quickOnsetFrames - quickReleaseFrames);
                int quickSum = quickOnsetFrames + quickSustainFrames + quickReleaseFrames;
                if (quickSum != targetTotalFrames) {
                    quickSustainFrames += targetTotalFrames - quickSum;
                }
                return (Math.Max(0, quickOnsetFrames), Math.Max(0, quickSustainFrames), Math.Max(0, quickReleaseFrames));
            }

            int sourceTotal = Math.Max(1, sourceOnsetFrames + sourceSustainFrames + sourceReleaseFrames);
            double ratio = Math.Min(1.0, targetTotalFrames / (double)sourceTotal);
            int onsetFrames = Math.Clamp(
                (int)Math.Round(sourceOnsetFrames * ratio),
                sourceOnsetFrames > 0 ? 1 : 0,
                Math.Min(sourceOnsetFrames, targetTotalFrames));
            int releaseFrames = Math.Clamp(
                (int)Math.Round(sourceReleaseFrames * ratio),
                0,
                Math.Min(sourceReleaseFrames, Math.Max(0, targetTotalFrames - onsetFrames)));

            while (targetTotalFrames >= 3 && onsetFrames + releaseFrames > targetTotalFrames - 1) {
                if (onsetFrames >= releaseFrames && onsetFrames > 1) {
                    onsetFrames--;
                } else if (releaseFrames > 0) {
                    releaseFrames--;
                } else {
                    break;
                }
            }
            int sustainFrames = Math.Max(0, targetTotalFrames - onsetFrames - releaseFrames);
            if (targetTotalFrames >= 3 && sourceSustainFrames > 0 && sustainFrames <= 0) {
                if (releaseFrames > 0) {
                    releaseFrames--;
                } else if (onsetFrames > 1) {
                    onsetFrames--;
                }
                sustainFrames = Math.Max(0, targetTotalFrames - onsetFrames - releaseFrames);
            }
            int sum = onsetFrames + sustainFrames + releaseFrames;
            if (sum != targetTotalFrames) {
                sustainFrames += targetTotalFrames - sum;
            }
            return (Math.Max(0, onsetFrames), Math.Max(0, sustainFrames), Math.Max(0, releaseFrames));
        }

        static bool HasFastF0Motion(float[] f0, int start, int frames) {
            if (!HifiNeuralConfig.EnableF0AwareStretch || f0.Length < 2 || frames <= 1) {
                return false;
            }
            int s = Math.Clamp(start, 0, f0.Length - 1);
            int e = Math.Clamp(start + frames, s + 1, f0.Length);
            for (int i = s + 1; i < e; i++) {
                if (F0SlopeCents(f0[i - 1], f0[i]) > HifiNeuralConfig.F0AwareMaxSlopeCentsPerFrame) {
                    return true;
                }
            }
            return false;
        }

        static void ApplyConsonantVowelMicroBridge(float[,] mel, int consonantBoundaryFrame) {
            int frames = mel.GetLength(1);
            int bins = mel.GetLength(0);
            if (consonantBoundaryFrame <= 0 || consonantBoundaryFrame >= frames) {
                return;
            }
            int rightFrames = Math.Min(3, frames - consonantBoundaryFrame);
            if (rightFrames <= 0) {
                return;
            }
            int consonantRef = consonantBoundaryFrame - 1;
            for (int r = 0; r < rightFrames; r++) {
                int frame = consonantBoundaryFrame + r;
                float weight = r switch {
                    0 => 0.24f,
                    1 => 0.12f,
                    _ => 0.06f,
                };
                for (int m = 0; m < bins; m++) {
                    float left = mel[m, consonantRef];
                    float current = mel[m, frame];
                    mel[m, frame] = current * (1f - weight) + left * weight;
                }
            }
            if (consonantBoundaryFrame - 2 >= 0) {
                int frame = consonantBoundaryFrame - 1;
                int next = consonantBoundaryFrame;
                for (int m = 0; m < bins; m++) {
                    float current = mel[m, frame];
                    float right = mel[m, next];
                    mel[m, frame] = current * 0.92f + right * 0.08f;
                }
            }
        }

        static VowelMapReport EmptyVowelMapReport(string strategy) {
            return new VowelMapReport(0, 0, 0, 0, strategy, 0, 0, 0, 0);
        }

        static StretchEnergyReport ApplyStretchEnergyCompensation(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int targetStart,
            int targetFrames,
            string phoneme,
            double stretchRatio,
            string region,
            float[]? targetF0,
            int targetF0Start,
            int loopCount) {
            if (!HifiNeuralConfig.EnableStretchEnergyCompensation) {
                return new StretchEnergyReport(0, 0, 0, "disabled");
            }
            if (sourceFrames < 4 || targetFrames < 4) {
                return new StretchEnergyReport(0, 0, 0, "too_few_frames");
            }
            bool stretched = stretchRatio > HifiNeuralConfig.StretchEnergyStartRatio
                && !double.IsNaN(stretchRatio)
                && !double.IsInfinity(stretchRatio);
            if (!stretched && loopCount <= 0) {
                return new StretchEnergyReport(0, 0, 0, "stretch_ratio_below_threshold");
            }

            int edgeFrames = Math.Clamp(targetFrames / 8, 2, 12);
            int stableStart = targetStart + edgeFrames;
            int stableEnd = targetStart + targetFrames - edgeFrames;
            if (stableEnd <= stableStart) {
                stableStart = targetStart;
                stableEnd = targetStart + targetFrames;
            }
            stableStart = Math.Clamp(stableStart, 0, output.GetLength(1));
            stableEnd = Math.Clamp(stableEnd, stableStart, output.GetLength(1));
            if (stableEnd <= stableStart) {
                return new StretchEnergyReport(0, 0, 0, "empty_target_stable_region");
            }

            double sourceMean = MeanMel(sourceMel, sourceStart, sourceStart + sourceFrames);
            double targetMean = MeanMel(output, stableStart, stableEnd);
            if (!IsFinite(sourceMean) || !IsFinite(targetMean)) {
                return new StretchEnergyReport(0, sourceMean, targetMean, "nan_or_inf_energy");
            }

            double meanExcessDb = Math.Max(0, LogAmplitudeToDb(targetMean - sourceMean) - HifiNeuralConfig.StretchEnergyHeadroomDb);
            double ratioCutDb = stretched
                ? Math.Max(0, Math.Log(stretchRatio, 2.0) * HifiNeuralConfig.StretchEnergyRatioCutDbPerOctave)
                : 0;
            double cutDb = Math.Max(meanExcessDb, ratioCutDb) * HifiNeuralConfig.StretchEnergyStrength;
            cutDb = Math.Clamp(cutDb, 0, HifiNeuralConfig.StretchEnergyMaxCutDb);
            double loopCutDb = loopCount > 0
                ? Math.Clamp(
                    HifiNeuralConfig.StretchEnergyLoopExtraCutDb
                    + Math.Log(Math.Max(1, loopCount), 2.0) * HifiNeuralConfig.StretchEnergyPerLoopCutDb,
                    0,
                    HifiNeuralConfig.StretchEnergyLoopMaxCutDb)
                : 0;
            double highPitchCutDb = HighPitchStretchCutDb(targetF0, targetF0Start, targetFrames);
            cutDb = Math.Clamp(
                cutDb + loopCutDb + highPitchCutDb,
                0,
                HifiNeuralConfig.StretchEnergyMaxTotalCutDb);
            if (cutDb <= 0.05) {
                Log.Information(
                    "HifiPhoneMelStretcher stretch_energy skipped phoneme={Phoneme} region={Region} reason=no_cut_needed stretch_ratio={StretchRatio:F4} source_mean={SourceMean:F4} target_mean={TargetMean:F4} mean_excess_db={MeanExcessDb:F3} ratio_cut_db={RatioCutDb:F3} loop_cut_db={LoopCutDb:F3} high_pitch_cut_db={HighPitchCutDb:F3}",
                    phoneme,
                    region,
                    stretchRatio,
                    sourceMean,
                    targetMean,
                    meanExcessDb,
                    ratioCutDb,
                    loopCutDb,
                    highPitchCutDb);
                return new StretchEnergyReport(0, sourceMean, targetMean, "no_cut_needed");
            }

            int fadeFrames = Math.Clamp((stableEnd - stableStart) / 8, 2, 8);
            ApplySmoothLogGain(output, stableStart, stableEnd, -DbToLogAmplitude(cutDb), fadeFrames);
            Log.Information(
                "HifiPhoneMelStretcher stretch_energy applied phoneme={Phoneme} region={Region} stretch_ratio={StretchRatio:F4} source_mean={SourceMean:F4} target_mean={TargetMean:F4} mean_excess_db={MeanExcessDb:F3} ratio_cut_db={RatioCutDb:F3} loop_cut_db={LoopCutDb:F3} high_pitch_cut_db={HighPitchCutDb:F3} cut_db={CutDb:F3} stable_start={StableStart} stable_end={StableEnd} fade_frames={FadeFrames}",
                phoneme,
                region,
                stretchRatio,
                sourceMean,
                targetMean,
                meanExcessDb,
                ratioCutDb,
                loopCutDb,
                highPitchCutDb,
                cutDb,
                stableStart,
                stableEnd,
                fadeFrames);
            return new StretchEnergyReport(cutDb, sourceMean, targetMean, string.Empty);
        }

        static double HighPitchStretchCutDb(float[]? targetF0, int start, int frames) {
            if (!HifiNeuralConfig.EnableHighPitchLoudnessControl
                || targetF0 == null
                || targetF0.Length == 0
                || frames <= 0) {
                return 0;
            }
            int s = Math.Clamp(start, 0, targetF0.Length);
            int e = Math.Clamp(start + frames, s, targetF0.Length);
            if (e <= s) {
                return 0;
            }
            var voiced = targetF0
                .Skip(s)
                .Take(e - s)
                .Where(value => value > 0 && !float.IsNaN(value) && !float.IsInfinity(value))
                .Select(value => (double)value)
                .OrderBy(value => value)
                .ToArray();
            if (voiced.Length == 0) {
                return 0;
            }
            double median = voiced[voiced.Length / 2];
            if (median <= HifiNeuralConfig.StretchEnergyHighF0StartHz) {
                return 0;
            }
            double span = Math.Max(0.25, HifiNeuralConfig.StretchEnergyHighF0RangeOctaves);
            double ratio = Math.Log(median / HifiNeuralConfig.StretchEnergyHighF0StartHz, 2.0);
            ratio = Math.Clamp(ratio / span, 0, 1);
            ratio = Math.Pow(ratio, HifiNeuralConfig.StretchEnergyHighF0Curve);
            return ratio * HifiNeuralConfig.StretchEnergyHighF0MaxCutDb;
        }

        static void ApplySmoothLogGain(float[,] mel, int start, int end, double logGain, int fadeFrames) {
            int bins = mel.GetLength(0);
            int frames = Math.Max(0, end - start);
            fadeFrames = Math.Clamp(fadeFrames, 0, Math.Max(0, frames / 2));
            for (int t = start; t < end; t++) {
                double weight = 1.0;
                if (fadeFrames > 0) {
                    int fromStart = t - start;
                    int fromEnd = end - 1 - t;
                    double edge = Math.Min(1.0, Math.Min(
                        (fromStart + 1) / (double)(fadeFrames + 1),
                        (fromEnd + 1) / (double)(fadeFrames + 1)));
                    weight = 0.5 - 0.5 * Math.Cos(Math.PI * Math.Clamp(edge, 0, 1));
                }
                float offset = (float)(logGain * weight);
                for (int m = 0; m < bins; m++) {
                    mel[m, t] += offset;
                }
            }
        }

        static double MeanMel(float[,] mel, int start, int end) {
            int frames = mel.GetLength(1);
            int bins = mel.GetLength(0);
            start = Math.Clamp(start, 0, frames);
            end = Math.Clamp(end, start, frames);
            if (end <= start) {
                return double.NaN;
            }
            double sum = 0;
            int count = 0;
            for (int t = start; t < end; t++) {
                for (int m = 0; m < bins; m++) {
                    float value = mel[m, t];
                    if (!float.IsNaN(value) && !float.IsInfinity(value)) {
                        sum += value;
                        count++;
                    }
                }
            }
            return count > 0 ? sum / count : double.NaN;
        }

        static void WriteMappedRegion(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames) {
            if (outputFrames <= 0) {
                return;
            }
            int bins = sourceMel.GetLength(0);
            int totalSourceFrames = sourceMel.GetLength(1);
            sourceStart = Math.Clamp(sourceStart, 0, Math.Max(0, totalSourceFrames - 1));
            sourceFrames = Math.Max(1, Math.Min(sourceFrames, totalSourceFrames - sourceStart));
            outputFrames = Math.Min(outputFrames, output.GetLength(1) - outputStart);
            if (outputFrames <= 0) {
                return;
            }
            if (outputFrames < sourceFrames) {
                WriteAreaResampledRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return;
            }
            for (int t = 0; t < outputFrames; t++) {
                double sourceIndex = outputFrames == 1 || sourceFrames == 1
                    ? 0
                    : (double)t * (sourceFrames - 1) / (outputFrames - 1);
                int left = sourceStart + (int)Math.Floor(sourceIndex);
                int right = Math.Min(sourceStart + sourceFrames - 1, left + 1);
                float alpha = (float)(sourceIndex - Math.Floor(sourceIndex));
                for (int m = 0; m < bins; m++) {
                    float v0 = sourceMel[m, left];
                    float v1 = sourceMel[m, right];
                    output[m, outputStart + t] = v0 + (v1 - v0) * alpha;
                }
            }
        }

        static void WriteAreaResampledRegion(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames) {
            int bins = sourceMel.GetLength(0);
            for (int t = 0; t < outputFrames; t++) {
                double rangeStart = t * sourceFrames / (double)outputFrames;
                double rangeEnd = (t + 1) * sourceFrames / (double)outputFrames;
                int first = Math.Clamp((int)Math.Floor(rangeStart), 0, sourceFrames - 1);
                int last = Math.Clamp((int)Math.Ceiling(rangeEnd) - 1, first, sourceFrames - 1);
                for (int m = 0; m < bins; m++) {
                    double sum = 0;
                    double weightSum = 0;
                    for (int s = first; s <= last; s++) {
                        double overlap = Math.Min(rangeEnd, s + 1.0) - Math.Max(rangeStart, s);
                        if (overlap <= 0) {
                            continue;
                        }
                        sum += sourceMel[m, sourceStart + s] * overlap;
                        weightSum += overlap;
                    }
                    output[m, outputStart + t] = weightSum > 0
                        ? (float)(sum / weightSum)
                        : sourceMel[m, sourceStart + first];
                }
            }
        }

        static double F0SlopeCents(float left, float right) {
            if (left <= 0 || right <= 0 || float.IsNaN(left) || float.IsNaN(right) || float.IsInfinity(left) || float.IsInfinity(right)) {
                return double.PositiveInfinity;
            }
            return Math.Abs(1200.0 * Math.Log(right / left, 2.0));
        }

        static void CopyResampled(float[,] sourceMel, int sourceStart, int sourceFrames, float[,] output, int outputStart, int outputFrames) {
            WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
        }

        static string ValidateOutput(float[,] mel) {
            if (mel.Length == 0) {
                return "empty_mel";
            }
            double sumAbs = 0;
            foreach (float value in mel) {
                if (float.IsNaN(value) || float.IsInfinity(value)) {
                    return "mel_nan_or_inf";
                }
                sumAbs += Math.Abs(value);
            }
            return sumAbs / mel.Length < 1e-8 ? "mel_almost_all_zero" : string.Empty;
        }

        static void LogDebug(HifiPhoneStretchDebug debug) {
            Log.Information(
                "HifiPhoneMelStretcher phoneme={Phoneme} source_frames={SourceFrames} target_frames={TargetFrames} consonant_ms={ConsonantMs:F3} consonant_frames={ConsonantFrames} vowel_source_frames={VowelSourceFrames} vowel_target_frames={VowelTargetFrames} stretch_ratio={StretchRatio:F4} loop_start_frame={LoopStartFrame} loop_end_frame={LoopEndFrame} loop_count={LoopCount} crossfade_frames={CrossfadeFrames} fallback_reason={FallbackReason}",
                debug.Phoneme,
                debug.SourceFrames,
                debug.TargetFrames,
                debug.ConsonantMs,
                debug.ConsonantFrames,
                debug.VowelSourceFrames,
                debug.VowelTargetFrames,
                debug.StretchRatio,
                debug.LoopStartFrame,
                debug.LoopEndFrame,
                debug.LoopCount,
                debug.CrossfadeFrames,
                debug.FallbackReason);
            Log.Information(
                "HifiPhoneMelStretcher lock phoneme={Phoneme} source_consonant_frames={SourceConsonantFrames} target_consonant_frames={TargetConsonantFrames} stretch_ratio_consonant={StretchRatioConsonant:F4} source_vowel_frames={SourceVowelFrames} target_vowel_frames={TargetVowelFrames} stretch_ratio_vowel={StretchRatioVowel:F4} consonant_lock_mode={ConsonantLockMode} borrowed_frames={BorrowedFrames} borrowed_frames_applied_to={BorrowedFramesAppliedTo}",
                debug.Phoneme,
                debug.SourceConsonantFrames,
                debug.TargetConsonantFrames,
                debug.StretchRatioConsonant,
                debug.SourceVowelFrames,
                debug.TargetVowelFrames,
                debug.StretchRatioVowel,
                debug.ConsonantLockMode,
                debug.BorrowedFrames,
                debug.BorrowedFramesAppliedTo);
            Log.Information(
                "HifiPhoneMelStretcher stretch_energy_debug phoneme={Phoneme} cut_db={CutDb:F3} source_mean={SourceMean:F4} target_mean={TargetMean:F4}",
                debug.Phoneme,
                debug.StretchEnergyCutDb,
                debug.StretchEnergySourceMean,
                debug.StretchEnergyTargetMean);
            Log.Information(
                "HifiPhoneMelStretcher f0_aware_debug phoneme={Phoneme} strategy={Strategy} f0_stable_start_frame={StableStart} f0_stable_end_frame={StableEnd} f0_max_slope_cents={MaxSlope:F2} f0_mean_slope_cents={MeanSlope:F2}",
                debug.Phoneme,
                debug.VowelStretchStrategy,
                debug.F0StableStartFrame,
                debug.F0StableEndFrame,
                debug.F0MaxSlopeCents,
                debug.F0MeanSlopeCents);
        }

        static double DbToLogAmplitude(double db) => db * Math.Log(10.0) / 20.0;

        static double LogAmplitudeToDb(double logAmplitude) => logAmplitude * 20.0 / Math.Log(10.0);

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }

    public static class HifiNeuralConfig {
        public const string Simple = "simple";
        public const string ConsonantVowelSplit = "consonant_vowel_split";
        public const string EnergyOff = "energy_off";
        public const string EnergyConservative = "energy_conservative";

        public static HifiStretchMode StretchMode {
            get {
                string mode = Environment.GetEnvironmentVariable("HIFI_NEURAL_STRETCH_MODE");
                if (string.IsNullOrWhiteSpace(mode)) {
                    mode = OpenUtau.Core.Util.Preferences.Default.HifiNeuralStretchMode;
                }
                return ParseStretchMode(mode);
            }
        }

        public static HifiStretchMode ParseStretchMode(string? mode) {
            return string.Equals(mode, ConsonantVowelSplit, StringComparison.OrdinalIgnoreCase)
                ? HifiStretchMode.ConsonantVowelSplit
                : HifiStretchMode.Simple;
        }

        public static string StretchModeName(HifiStretchMode mode) {
            return mode == HifiStretchMode.ConsonantVowelSplit ? ConsonantVowelSplit : Simple;
        }

        public static string CacheKey() {
            return $"v5-sync1-{StretchModeName(StretchMode)}-lock{ConsonantLockModeName(ConsonantLockMode)}-borrow{EnableDurationBorrow}-pmc{EnablePitchMelCompensation}-se{EnableStretchEnergyCompensation}-aenv{EnableAudioEnvelopeNormalization}-{AudioEnvelopeMaxCutDb:0.0}-{AudioEnvelopeHeadroomDb:0.0}-{AudioEnvelopeStrength:0.00}-out{OutputTargetVoicedRmsDb:0.0}-{OutputMaxMakeupDb:0.0}-{OutputPeakLimit:0.00}-exc{EnableOutputExciter}-{OutputExciterDrive:0.00}-{OutputExciterMix:0.00}";
        }
        public const string LockPreserve = "preserve";
        public const string LockReadable = "readable";
        public const string LockOff = "off";

        public static HifiConsonantLockMode ConsonantLockMode => ParseConsonantLockMode(
            Environment.GetEnvironmentVariable("HIFI_NEURAL_CONSONANT_LOCK_MODE")
            ?? OpenUtau.Core.Util.Preferences.Default.HifiNeuralConsonantLockMode);

        public static HifiConsonantLockMode ParseConsonantLockMode(string? mode) {
            if (string.Equals(mode, LockReadable, StringComparison.OrdinalIgnoreCase)) {
                return HifiConsonantLockMode.Readable;
            }
            if (string.Equals(mode, LockOff, StringComparison.OrdinalIgnoreCase)) {
                return HifiConsonantLockMode.Off;
            }
            return HifiConsonantLockMode.Preserve;
        }

        public static string ConsonantLockModeName(HifiConsonantLockMode mode) {
            return mode switch {
                HifiConsonantLockMode.Readable => LockReadable,
                HifiConsonantLockMode.Off => LockOff,
                _ => LockPreserve,
            };
        }

        public static bool EnableDurationBorrow {
            get {
                string value = Environment.GetEnvironmentVariable("HIFI_NEURAL_ENABLE_DURATION_BORROW");
                if (bool.TryParse(value, out bool envValue)) {
                    return envValue;
                }
                return OpenUtau.Core.Util.Preferences.Default.HifiNeuralEnableDurationBorrow;
            }
        }

        public static double MinConsonantMs => PositiveOrDefault(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralMinConsonantMs,
            40);

        public static double MinVowelMs => PositiveOrDefault(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralMinVowelMs,
            30);

        public static double MaxBorrowMs => PositiveOrDefault(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralMaxBorrowMs,
            30);

        public static double MaxBorrowRatio => Math.Clamp(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralMaxBorrowRatio,
            0,
            1);

        public static bool EnableCodaProtection => OpenUtau.Core.Util.Preferences.Default.HifiNeuralEnableCodaProtection;

        public static double MinCodaMs => PositiveOrDefault(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralMinCodaMs,
            40);

        public static int BoundaryCrossfadeFrames => PositiveIntOrDefault(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralBoundaryCrossfadeFrames,
            2);

        public static int BoundarySmoothFrames => NonNegativeIntOrDefault(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralBoundarySmoothFrames,
            2);

        public static int DurationBorrowSmoothFrames => NonNegativeIntOrDefault(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralDurationBorrowSmoothFrames,
            3);

        public static int VoicedContinuityRadius => 1;

        public static float VoicedContinuityStrength => 0.18f;

        public static bool EnableVoicedIslandContinuity {
            get {
                string value = Environment.GetEnvironmentVariable("HIFI_NEURAL_ENABLE_VOICED_ISLAND_CONTINUITY");
                return bool.TryParse(value, out bool enabled) && enabled;
            }
        }

        public static int VoicedIslandMorphFrames => 5;

        public static int VoicedIslandSmoothRadiusFrames => 3;

        public static float VoicedIslandMorphStrength => 0.35f;

        public static float VoicedIslandSmoothStrength => 0.18f;

        public static int VoicedIslandMaxF0BridgeFrames => 2;

        public static double VoicedIslandMaxF0BridgeCents => 500;

        public static int RapidShortPhoneFrames => 8;

        public static int VoicedIslandMinStableFrames => 4;

        public static int ShortPhoneMinRenderedFrames => 4;

        public static int SourceStartOffsetMaxFrames => 4;

        public static bool EnableVoicedOnsetDipRepair {
            get {
                string value = Environment.GetEnvironmentVariable("HIFI_NEURAL_ENABLE_VOICED_ONSET_DIP_REPAIR");
                return bool.TryParse(value, out bool enabled) && enabled;
            }
        }

        public static int VoicedOnsetDipRepairFrames => 3;

        public static double VoicedOnsetDipMinDelta => 0.75;

        public static double VoicedOnsetDipMaxLift => 0.9;

        public static bool EnableSourceLeadingSanitizer {
            get {
                string value = Environment.GetEnvironmentVariable("HIFI_NEURAL_ENABLE_SOURCE_LEADING_SANITIZER");
                if (bool.TryParse(value, out bool enabled)) {
                    return enabled;
                }
                return OpenUtau.Core.Util.Preferences.Default.HifiNeuralEnableSourceLeadingSanitizer;
            }
        }

        public static double SourceLeadingSanitizeGuardMs => 12;

        public static double SourceLeadingSanitizeMaxSkipMs => 80;

        public static double SourceLeadingSanitizeDeclickMs => 0.8;

        public static bool EnableF0AwareStretch {
            get {
                string value = Environment.GetEnvironmentVariable("HIFI_NEURAL_ENABLE_F0_AWARE_STRETCH");
                if (bool.TryParse(value, out bool enabled)) {
                    return enabled;
                }
                return OpenUtau.Core.Util.Preferences.Default.HifiNeuralEnableF0AwareStretch;
            }
        }

        public static double F0AwareMaxSlopeCentsPerFrame => 45;

        public static int F0AwareMinStableFrames => 5;

        public static bool EnableF0AwareBoundaryCompose {
            get {
                string value = Environment.GetEnvironmentVariable("HIFI_NEURAL_ENABLE_F0_AWARE_BOUNDARY_COMPOSE");
                if (bool.TryParse(value, out bool enabled)) {
                    return enabled;
                }
                return OpenUtau.Core.Util.Preferences.Default.HifiNeuralEnableF0AwareBoundaryCompose;
            }
        }

        public static double BoundaryF0ContinuousJumpCents => 80;

        public static double BoundaryF0ContinuousMaxSlopeCents => 140;

        public static double BoundaryF0FastMotionCents => 180;

        public static double BoundaryF0LargeJumpCents => 260;

        public static bool EnablePitchMelCompensation {
            get {
                string value = Environment.GetEnvironmentVariable("HIFI_NEURAL_ENABLE_PITCH_MEL_COMPENSATION");
                if (bool.TryParse(value, out bool enabled)) {
                    return enabled;
                }
                return OpenUtau.Core.Util.Preferences.Default.HifiNeuralEnablePitchMelCompensation;
            }
        }

        public static double PitchMelCompensationStartCents => 90;

        public static double PitchMelCompensationMaxCutDb => 0.7;

        public static double PitchMelCompensationMaxTiltDb => 0.4;

        public static bool EnableHighPitchLoudnessControl {
            get {
                string value = Environment.GetEnvironmentVariable("HIFI_NEURAL_ENABLE_HIGH_PITCH_LOUDNESS_CONTROL");
                return !bool.TryParse(value, out bool enabled) || enabled;
            }
        }

        public static bool PitchMelCompensationOnlyCut => true;

        public static double PitchMelCompensationHighF0StartHz => 330;

        public static double PitchMelCompensationHighF0RangeOctaves => 1.8;

        public static double PitchMelCompensationHighF0Curve => 0.8;

        public static double PitchMelCompensationHighF0MaxCutDb => 0.4;

        public static double PitchMelCompensationTotalMaxCutDb => 1.1;

        public static double ClickDiagnosticMelDeltaThreshold => 1.2;

        public static bool EnableAudioEnvelopeNormalization => true;

        public static double AudioEnvelopeMaxCutDb => 0.8;

        public static double AudioEnvelopeHeadroomDb => 3.0;

        public static double AudioEnvelopeStrength => 0.25;

        public static double AudioEnvelopeHighF0StartHz => 300;

        public static double AudioEnvelopeHighF0RangeOctaves => 2.0;

        public static double AudioEnvelopeHighF0Curve => 0.8;

        public static double AudioEnvelopeHighF0ExtraScale => 0.10;

        public static double AudioEnvelopeHighF0ExtraMaxCutDb => 0.25;

        public static bool EnableStretchEnergyCompensation {
            get {
                string value = Environment.GetEnvironmentVariable("HIFI_NEURAL_ENABLE_STRETCH_ENERGY_COMPENSATION");
                return bool.TryParse(value, out bool enabled) && enabled;
            }
        }

        public static double StretchEnergyStartRatio => 1.15;

        public static double StretchEnergyMaxCutDb => 1.2;

        public static double StretchEnergyHeadroomDb => 0.25;

        public static double StretchEnergyStrength => 0.35;

        public static double StretchEnergyRatioCutDbPerOctave => 0.3;

        public static double StretchEnergyHighF0StartHz => 320;

        public static double StretchEnergyHighF0RangeOctaves => 2.0;

        public static double StretchEnergyHighF0Curve => 0.8;

        public static double StretchEnergyHighF0MaxCutDb => 0.35;

        public static double StretchEnergyLoopExtraCutDb => 0.0;

        public static double StretchEnergyPerLoopCutDb => 0.0;

        public static double StretchEnergyLoopMaxCutDb => 0.0;

        public static double StretchEnergyMaxTotalCutDb => 1.6;

        public static int F0BoundarySmoothFrames => 2;

        public static double F0BoundarySmoothMinJumpCents => 35;

        public static bool EnableInternalClickSuppression => true;

        public static double InternalClickSuppressorWindowMs => 1.5;

        public static double InternalClickSuppressorThresholdRatio => 0.35;

        public static bool EnableBoundaryEnergyMatching => OpenUtau.Core.Util.Preferences.Default.HifiNeuralEnableBoundaryEnergyMatching;

        public static string BoundaryEnergyPreset {
            get {
                string preset = OpenUtau.Core.Util.Preferences.Default.HifiNeuralBoundaryEnergyPreset;
                return preset == EnergyOff ? EnergyOff : EnergyConservative;
            }
        }

        public static double BoundaryEnergyMaxGainDb => Math.Clamp(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralBoundaryEnergyMaxGainDb,
            0.5,
            1.0);

        public static int BoundaryEnergyWindowFrames => Math.Clamp(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralBoundaryEnergyWindowFrames,
            8,
            16);

        public static int TailReleaseFrames => 0;

        public static int HeadAttackFrames => 0;

        public static double HeadAttackGainDb => -4;

        public static int HeadAttackF0Frames => 0;

        public static double TailReleaseGainDb => Math.Clamp(
            OpenUtau.Core.Util.Preferences.Default.HifiNeuralTailReleaseGainDb,
            -18,
            0);

        public static int TailReleaseF0Frames => 0;

        public static double OutputTargetVoicedRmsDb => -15.5;

        public static double OutputMaxMakeupDb => 6.0;

        public static double OutputPeakLimit => 0.97;

        public static bool EnableOutputExciter => true;

        public static double OutputExciterDrive => 1.35;

        public static double OutputExciterMix => 0.08;

        static double PositiveOrDefault(double value, double fallback) {
            return value > 0 ? value : fallback;
        }

        static int PositiveIntOrDefault(int value, int fallback) {
            return value > 0 ? value : fallback;
        }

        static int NonNegativeIntOrDefault(int value, int fallback) {
            return value >= 0 ? value : fallback;
        }
    }
}
