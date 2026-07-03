using System;
using System.Collections.Generic;

namespace OpenUtau.Core.HifiNeural {
    /// <summary>
    /// Cheap waveform-domain analysis of oto source slices, shared by the mel assembler and the
    /// feature builder so both sides derive timing/pitch decisions from the same numbers.
    /// </summary>
    public static class HifiSourceAnalysis {
        const int F0WindowSize = 2048;
        const int F0MaxWindows = 8;
        const double F0MinHz = 55.0;
        const double F0MaxHz = 1000.0;
        const double F0MinNormalizedPeak = 0.45;
        const double F0OctavePeakTolerance = 0.87;
        const double F0WindowEnergyFloor = 1e-4;
        // Matches HifiPhraseFeatureBuilder.InactiveTailLogDrop: both trims must agree on where the
        // audible tail of a slice ends so the HNSEP parameter projection stays aligned with the
        // actual mel mapping.
        const double ActiveTailLogDrop = 2.8;

        /// <summary>
        /// Median F0 (Hz) of the voiced portion of a source slice via normalized autocorrelation
        /// over a few windows spread across the central region. Returns 0 when no reliable voiced
        /// estimate is found.
        /// </summary>
        public static double EstimateF0Hz(float[] samples, int sampleRate = HifiMelExtractor.SampleRate) {
            if (samples.Length < F0WindowSize) {
                return 0;
            }
            int minLag = Math.Max(2, (int)Math.Floor(sampleRate / F0MaxHz));
            int maxLag = Math.Min(F0WindowSize / 2, (int)Math.Ceiling(sampleRate / F0MinHz));
            if (maxLag <= minLag + 2) {
                return 0;
            }

            // Skip the outer 20% of the slice: leading consonants and release tails are the least
            // periodic parts and would only add octave errors.
            int usableStart = samples.Length / 5;
            int usableEnd = samples.Length - samples.Length / 5;
            if (usableEnd - usableStart < F0WindowSize) {
                usableStart = 0;
                usableEnd = samples.Length;
            }
            int span = usableEnd - usableStart - F0WindowSize;
            int windows = Math.Clamp(span / F0WindowSize + 1, 1, F0MaxWindows);

            var estimates = new List<double>(windows);
            for (int w = 0; w < windows; w++) {
                int start = windows == 1
                    ? usableStart
                    : usableStart + (int)Math.Round(span * (double)w / (windows - 1));
                double hz = EstimateWindowF0Hz(samples, start, sampleRate, minLag, maxLag);
                if (hz > 0) {
                    estimates.Add(hz);
                }
            }
            if (estimates.Count == 0) {
                return 0;
            }
            estimates.Sort();
            return estimates[estimates.Count / 2];
        }

        static double EstimateWindowF0Hz(float[] samples, int start, int sampleRate, int minLag, int maxLag) {
            double mean = 0;
            for (int i = 0; i < F0WindowSize; i++) {
                mean += samples[start + i];
            }
            mean /= F0WindowSize;
            double energy = 0;
            for (int i = 0; i < F0WindowSize; i++) {
                double v = samples[start + i] - mean;
                energy += v * v;
            }
            if (energy < F0WindowEnergyFloor) {
                return 0;
            }

            int compareLength = F0WindowSize - maxLag;
            double energy0 = 0;
            for (int i = 0; i < compareLength; i++) {
                double v = samples[start + i] - mean;
                energy0 += v * v;
            }
            if (energy0 <= 1e-9) {
                return 0;
            }

            var nac = new double[maxLag + 1];
            for (int lag = minLag; lag <= maxLag; lag++) {
                double cross = 0;
                double energyLag = 0;
                for (int i = 0; i < compareLength; i++) {
                    double a = samples[start + i] - mean;
                    double b = samples[start + i + lag] - mean;
                    cross += a * b;
                    energyLag += b * b;
                }
                nac[lag] = energyLag > 1e-9 ? cross / Math.Sqrt(energy0 * energyLag) : 0;
            }

            double bestPeak = 0;
            for (int lag = minLag; lag <= maxLag; lag++) {
                bestPeak = Math.Max(bestPeak, nac[lag]);
            }
            if (bestPeak < F0MinNormalizedPeak) {
                return 0;
            }
            // Prefer the smallest lag whose local-maximum peak comes close to the global best:
            // picking the global maximum alone often lands on 2x the period (one octave low).
            for (int lag = minLag + 1; lag < maxLag; lag++) {
                if (nac[lag] >= bestPeak * F0OctavePeakTolerance
                        && nac[lag] >= nac[lag - 1]
                        && nac[lag] >= nac[lag + 1]) {
                    double refined = RefinePeakLag(nac, lag);
                    return sampleRate / refined;
                }
            }
            return 0;
        }

        static double RefinePeakLag(double[] nac, int lag) {
            if (lag <= 0 || lag >= nac.Length - 1) {
                return lag;
            }
            double left = nac[lag - 1];
            double center = nac[lag];
            double right = nac[lag + 1];
            double denom = left - 2 * center + right;
            if (Math.Abs(denom) < 1e-12) {
                return lag;
            }
            double delta = 0.5 * (left - right) / denom;
            return lag + Math.Clamp(delta, -0.5, 0.5);
        }

        /// <summary>
        /// Number of frames (hop = <see cref="HifiMelExtractor.OriginHopSize"/>) from the start of
        /// the slice up to and including the last frame whose log RMS stays within
        /// <see cref="ActiveTailLogDrop"/> of the loudest frame. Returns 0 for silent slices.
        /// </summary>
        public static int EstimateActiveFrameCount(float[] samples) {
            int totalFrames = HifiMelExtractor.EstimateFrameCount(samples.Length);
            if (totalFrames <= 0) {
                return 0;
            }
            var logRms = new double[totalFrames];
            double maxLog = double.NegativeInfinity;
            for (int frame = 0; frame < totalFrames; frame++) {
                int start = frame * HifiMelExtractor.OriginHopSize;
                int end = Math.Min(samples.Length, start + HifiMelExtractor.OriginHopSize);
                double sum = 0;
                int count = 0;
                for (int i = start; i < end; i++) {
                    float v = samples[i];
                    if (!float.IsFinite(v)) {
                        continue;
                    }
                    sum += v * v;
                    count++;
                }
                double rms = count > 0 ? Math.Sqrt(sum / count) : 0;
                logRms[frame] = Math.Log(Math.Max(rms, 1e-6));
                maxLog = Math.Max(maxLog, logRms[frame]);
            }
            if (double.IsNegativeInfinity(maxLog) || maxLog <= Math.Log(1e-5)) {
                return 0;
            }
            double threshold = maxLog - ActiveTailLogDrop;
            for (int frame = totalFrames - 1; frame >= 0; frame--) {
                if (logRms[frame] >= threshold) {
                    return frame + 1;
                }
            }
            return 0;
        }
    }
}
