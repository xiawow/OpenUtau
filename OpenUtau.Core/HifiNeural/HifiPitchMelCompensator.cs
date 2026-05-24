using System;
using System.Collections.Generic;
using System.Linq;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiPitchMelCompensationReport {
        public string Phoneme { get; init; } = string.Empty;
        public double SourceF0 { get; init; }
        public double TargetF0 { get; init; }
        public double MismatchCents { get; init; }
        public double GainCutDb { get; init; }
        public double TiltDb { get; init; }
        public double HighPitchPenaltyDb { get; init; }
        public int StartFrame { get; init; }
        public int EndFrame { get; init; }
        public string SkippedReason { get; init; } = string.Empty;
    }

    public static class HifiPitchMelCompensator {
        const int AnalysisWindow = 4096;
        const int AnalysisHop = 1024;
        const double MinF0 = 65;
        const double MaxF0 = 900;
        const double MinCorrelation = 0.32;

        public static double EstimateSourceF0(float[] samples, double? consonantMs) {
            if (samples.Length < AnalysisWindow) {
                return 0;
            }
            int start = consonantMs.HasValue
                ? (int)Math.Round(Math.Max(0, consonantMs.Value) * HifiMelExtractor.SampleRate / 1000.0)
                : 0;
            start = Math.Clamp(start, 0, Math.Max(0, samples.Length - AnalysisWindow));
            int end = Math.Max(start + AnalysisWindow, samples.Length - AnalysisWindow / 2);
            end = Math.Clamp(end, start + AnalysisWindow, samples.Length);
            var estimates = new List<double>();
            for (int frameStart = start; frameStart + AnalysisWindow <= end; frameStart += AnalysisHop) {
                var estimate = EstimateFrameF0(samples, frameStart, AnalysisWindow);
                if (estimate > 0) {
                    estimates.Add(estimate);
                }
                if (estimates.Count >= 24) {
                    break;
                }
            }
            if (estimates.Count == 0) {
                return 0;
            }
            estimates.Sort();
            return estimates[estimates.Count / 2];
        }

        public static HifiPitchMelCompensationReport Apply(
            float[,] mel,
            float[] targetF0,
            string phoneme,
            int protectedFrames,
            double sourceF0) {
            if (!HifiNeuralConfig.EnablePitchMelCompensation) {
                return LogSkip(phoneme, sourceF0, 0, "disabled");
            }
            int frames = mel.GetLength(1);
            if (frames <= 0 || targetF0.Length == 0) {
                return LogSkip(phoneme, sourceF0, 0, "empty_input");
            }
            if (IsSilenceOrTransient(phoneme)) {
                return LogSkip(phoneme, sourceF0, 0, "transient_or_silence");
            }
            if (sourceF0 <= 0 || double.IsNaN(sourceF0) || double.IsInfinity(sourceF0)) {
                return LogSkip(phoneme, sourceF0, 0, "missing_source_f0");
            }

            int start = Math.Clamp(protectedFrames, 0, frames);
            int end = frames;
            if (end - start > 8) {
                start += 2;
                end -= 2;
            }
            if (end - start < 3) {
                return LogSkip(phoneme, sourceF0, 0, "region_too_short");
            }

            double target = MedianTargetF0(targetF0, start, end);
            if (target <= 0) {
                return LogSkip(phoneme, sourceF0, 0, "missing_target_f0");
            }
            double mismatchCents = 1200.0 * Math.Log(target / sourceF0, 2.0);
            double absCents = Math.Abs(mismatchCents);
            if (absCents < HifiNeuralConfig.PitchMelCompensationStartCents) {
                return LogSkip(phoneme, sourceF0, target, "mismatch_below_threshold");
            }

            double ratio = Math.Clamp(
                (absCents - HifiNeuralConfig.PitchMelCompensationStartCents) / 1200.0,
                0,
                1);
            double gainCutDb = Math.Clamp(
                ratio * HifiNeuralConfig.PitchMelCompensationMaxCutDb,
                0,
                HifiNeuralConfig.PitchMelCompensationMaxCutDb);
            double tiltDb = Math.Clamp(
                ratio * HifiNeuralConfig.PitchMelCompensationMaxTiltDb,
                0,
                HifiNeuralConfig.PitchMelCompensationMaxTiltDb);
            double highPitchPenaltyDb = HighPitchPenaltyDb(target);
            gainCutDb = Math.Clamp(
                gainCutDb + highPitchPenaltyDb,
                0,
                HifiNeuralConfig.PitchMelCompensationTotalMaxCutDb);
            tiltDb = Math.Min(tiltDb, gainCutDb);
            ApplyCompensation(
                mel,
                start,
                end,
                Math.Sign(mismatchCents),
                gainCutDb,
                tiltDb,
                HifiNeuralConfig.PitchMelCompensationOnlyCut);

            var report = new HifiPitchMelCompensationReport {
                Phoneme = phoneme,
                SourceF0 = sourceF0,
                TargetF0 = target,
                MismatchCents = mismatchCents,
                GainCutDb = gainCutDb,
                TiltDb = tiltDb,
                HighPitchPenaltyDb = highPitchPenaltyDb,
                StartFrame = start,
                EndFrame = end,
            };
            Log.Information(
                "HifiPitchMelCompensator applied phoneme={Phoneme} source_f0={SourceF0:F2} target_f0={TargetF0:F2} mismatch_cents={MismatchCents:F2} gain_cut_db={GainCutDb:F3} tilt_db={TiltDb:F3} high_pitch_penalty_db={HighPitchPenaltyDb:F3} start_frame={StartFrame} end_frame={EndFrame}",
                report.Phoneme,
                report.SourceF0,
                report.TargetF0,
                report.MismatchCents,
                report.GainCutDb,
                report.TiltDb,
                report.HighPitchPenaltyDb,
                report.StartFrame,
                report.EndFrame);
            return report;
        }

        static double EstimateFrameF0(float[] samples, int start, int window) {
            int minLag = Math.Max(1, (int)Math.Floor(HifiMelExtractor.SampleRate / MaxF0));
            int maxLag = Math.Min(window / 2, (int)Math.Ceiling(HifiMelExtractor.SampleRate / MinF0));
            double energy = 0;
            for (int i = 0; i < window; i++) {
                float w = Hann(i, window);
                double value = samples[start + i] * w;
                energy += value * value;
            }
            if (energy < 1e-7) {
                return 0;
            }

            int bestLag = 0;
            double best = 0;
            for (int lag = minLag; lag <= maxLag; lag++) {
                double sum = 0;
                double leftEnergy = 0;
                double rightEnergy = 0;
                int count = window - lag;
                for (int i = 0; i < count; i++) {
                    float w = Hann(i, count);
                    double left = samples[start + i] * w;
                    double right = samples[start + i + lag] * w;
                    sum += left * right;
                    leftEnergy += left * left;
                    rightEnergy += right * right;
                }
                double corr = sum / Math.Sqrt(Math.Max(1e-12, leftEnergy * rightEnergy));
                if (corr > best) {
                    best = corr;
                    bestLag = lag;
                }
            }
            return best >= MinCorrelation && bestLag > 0
                ? HifiMelExtractor.SampleRate / (double)bestLag
                : 0;
        }

        static void ApplyCompensation(
            float[,] mel,
            int start,
            int end,
            double direction,
            double gainCutDb,
            double tiltDb,
            bool onlyCut) {
            int bins = mel.GetLength(0);
            int frames = end - start;
            double gainLog = -DbToLogAmplitude(gainCutDb);
            for (int t = start; t < end; t++) {
                double edge = EdgeWeight(t - start, frames);
                for (int m = 0; m < bins; m++) {
                    double bin = bins <= 1 ? 0.5 : m / (double)(bins - 1);
                    double centered = bin * 2.0 - 1.0;
                    double tiltLog = DbToLogAmplitude(tiltDb * direction * centered);
                    double offset = (gainLog + tiltLog) * edge;
                    if (onlyCut && offset > 0) {
                        offset = 0;
                    }
                    mel[m, t] += (float)offset;
                }
            }
        }

        static double HighPitchPenaltyDb(double targetF0) {
            if (!HifiNeuralConfig.EnableHighPitchLoudnessControl
                || targetF0 <= 0
                || double.IsNaN(targetF0)
                || double.IsInfinity(targetF0)
                || targetF0 <= HifiNeuralConfig.PitchMelCompensationHighF0StartHz) {
                return 0;
            }
            double spanOctaves = Math.Max(0.25, HifiNeuralConfig.PitchMelCompensationHighF0RangeOctaves);
            double ratio = Math.Log(targetF0 / HifiNeuralConfig.PitchMelCompensationHighF0StartHz, 2.0);
            ratio = Math.Clamp(ratio / spanOctaves, 0, 1);
            ratio = Math.Pow(ratio, HifiNeuralConfig.PitchMelCompensationHighF0Curve);
            return ratio * HifiNeuralConfig.PitchMelCompensationHighF0MaxCutDb;
        }

        static double MedianTargetF0(float[] f0, int start, int end) {
            start = Math.Clamp(start, 0, f0.Length);
            end = Math.Clamp(end, start, f0.Length);
            var voiced = f0
                .Skip(start)
                .Take(end - start)
                .Where(value => value > 0 && !float.IsNaN(value) && !float.IsInfinity(value))
                .Select(value => (double)value)
                .OrderBy(value => value)
                .ToArray();
            return voiced.Length > 0 ? voiced[voiced.Length / 2] : 0;
        }

        static HifiPitchMelCompensationReport LogSkip(string phoneme, double sourceF0, double targetF0, string reason) {
            Log.Information(
                "HifiPitchMelCompensator skipped phoneme={Phoneme} source_f0={SourceF0:F2} target_f0={TargetF0:F2} reason={Reason}",
                phoneme,
                sourceF0,
                targetF0,
                reason);
            return new HifiPitchMelCompensationReport {
                Phoneme = phoneme,
                SourceF0 = sourceF0,
                TargetF0 = targetF0,
                SkippedReason = reason,
            };
        }

        static bool IsSilenceOrTransient(string phoneme) {
            string p = (phoneme ?? string.Empty).Trim().ToLowerInvariant();
            if (p == "r" || p == "rest" || p == "sil" || p == "pau" || p == "-" || p == "br" || p.Contains("cl")) {
                return true;
            }
            return p is "p" or "t" or "k" or "q" or "s" or "sh" or "ch" or "ts" or "f" or "h" or "hh" or "th";
        }

        static float Hann(int index, int count) {
            if (count <= 1) {
                return 1f;
            }
            return (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * index / (count - 1)));
        }

        static double EdgeWeight(int frame, int totalFrames) {
            int fadeFrames = Math.Clamp(totalFrames / 6, 2, 8);
            if (totalFrames <= fadeFrames * 2) {
                fadeFrames = Math.Max(1, totalFrames / 2);
            }
            double fromStart = fadeFrames > 0 && frame < fadeFrames
                ? (frame + 1) / (double)(fadeFrames + 1)
                : 1.0;
            int fromEndFrame = totalFrames - 1 - frame;
            double fromEnd = fadeFrames > 0 && fromEndFrame < fadeFrames
                ? (fromEndFrame + 1) / (double)(fadeFrames + 1)
                : 1.0;
            double x = Math.Clamp(Math.Min(fromStart, fromEnd), 0, 1);
            return 0.5 - 0.5 * Math.Cos(Math.PI * x);
        }

        static double DbToLogAmplitude(double db) => db * Math.Log(10.0) / 20.0;
    }
}
