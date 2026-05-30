using System;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiLoudnessNormalizer {
        // Target active loudness. The mel-domain concat path inherits whatever level the source
        // slices were recorded at, which is usually quiet; we lift it toward a commercial vocal
        // mix level (~-14 dB active RMS) instead of the previous conservative -18 dB.
        const double TargetActiveRmsDb = -14.0;
        const double MaxBoostDb = 15.0;
        const double MaxCutDb = 3.0;
        const double MaxPeak = 0.97;
        // Below this absolute level samples pass through linearly; above it a tanh soft knee bends
        // them toward MaxPeak. This lets the full target gain be applied so the RMS actually
        // reaches the target, while peaks are smoothly tamed rather than hard-clipped or rolled
        // back (which used to leave dynamic material quiet).
        const double SoftKneeThreshold = 0.80;
        const double SilenceThreshold = 1e-5;

        public static double NormalizeInPlace(float[] samples, int sampleRate) {
            if (samples.Length == 0) {
                return 1.0;
            }
            double peak = Peak(samples);
            if (peak <= SilenceThreshold) {
                return 1.0;
            }
            double activeRms = ActiveRms(samples, sampleRate, peak);
            if (activeRms <= SilenceThreshold) {
                return 1.0;
            }

            double currentDb = LinearToDb(activeRms);
            double desiredDb = TargetActiveRmsDb - currentDb;
            desiredDb = Math.Clamp(desiredDb, -MaxCutDb, MaxBoostDb);
            double gain = Math.Pow(10.0, desiredDb / 20.0);
            if (!IsFinite(gain) || gain <= 0) {
                return 1.0;
            }

            double peakAfterGain = peak * gain;
            bool softLimited = peakAfterGain > MaxPeak;
            if (Math.Abs(gain - 1.0) < 0.01 && !softLimited) {
                return 1.0;
            }

            for (int i = 0; i < samples.Length; i++) {
                double v = samples[i] * gain;
                samples[i] = (float)SoftLimit(v);
            }
            Log.Information(
                "Hifi loudness normalized active_rms_db={ActiveRmsDb:F2} target_db={TargetDb:F2} gain_db={GainDb:F2} peak_before={PeakBefore:F4} peak_after_gain={PeakAfterGain:F4} soft_limited={SoftLimited}",
                currentDb,
                TargetActiveRmsDb,
                LinearToDb(gain),
                peak,
                peakAfterGain,
                softLimited);
            return gain;
        }

        /// <summary>
        /// Soft-knee limiter: linear below the knee threshold, tanh-bent toward ±MaxPeak above it.
        /// Continuous and monotonic, so it adds no hard-clip discontinuity, and the output is always
        /// within [-MaxPeak, MaxPeak].
        /// </summary>
        static double SoftLimit(double x) {
            double sign = x < 0 ? -1.0 : 1.0;
            double abs = Math.Abs(x);
            if (abs <= SoftKneeThreshold) {
                return Math.Clamp(x, -MaxPeak, MaxPeak);
            }
            double range = MaxPeak - SoftKneeThreshold;
            if (range <= 0) {
                return sign * MaxPeak;
            }
            double over = (abs - SoftKneeThreshold) / range;
            double shaped = SoftKneeThreshold + range * Math.Tanh(over);
            return sign * Math.Min(shaped, MaxPeak);
        }

        static double ActiveRms(float[] samples, int sampleRate, double peak) {
            int window = Math.Clamp(sampleRate / 20, 512, 4096);
            double threshold = Math.Max(0.004, peak * 0.035);
            double sum = 0;
            int count = 0;
            for (int start = 0; start < samples.Length; start += window) {
                int end = Math.Min(samples.Length, start + window);
                double rms = Rms(samples, start, end - start);
                if (rms < threshold) {
                    continue;
                }
                for (int i = start; i < end; i++) {
                    sum += samples[i] * samples[i];
                    count++;
                }
            }
            if (count == 0) {
                return Rms(samples, 0, samples.Length);
            }
            return Math.Sqrt(sum / count);
        }

        static double Rms(float[] samples, int start, int count) {
            if (count <= 0) {
                return 0;
            }
            double sum = 0;
            int end = Math.Min(samples.Length, start + count);
            for (int i = start; i < end; i++) {
                sum += samples[i] * samples[i];
            }
            return Math.Sqrt(sum / Math.Max(1, end - start));
        }

        static double Peak(float[] samples) {
            double peak = 0;
            foreach (float sample in samples) {
                if (float.IsNaN(sample) || float.IsInfinity(sample)) {
                    continue;
                }
                peak = Math.Max(peak, Math.Abs(sample));
            }
            return peak;
        }

        static double LinearToDb(double value) {
            return 20.0 * Math.Log10(Math.Max(SilenceThreshold, value));
        }

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
