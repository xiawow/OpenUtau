using System;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiLoudnessNormalizer {
        const double TargetActiveRmsDb = -18.0;
        const double MaxBoostDb = 9.0;
        const double MaxCutDb = 3.0;
        const double MaxPeak = 0.97;
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
            if (peak * gain > MaxPeak) {
                gain = MaxPeak / peak;
            }
            if (!IsFinite(gain) || gain <= 0 || Math.Abs(gain - 1.0) < 0.01) {
                return 1.0;
            }

            for (int i = 0; i < samples.Length; i++) {
                samples[i] = (float)Math.Clamp(samples[i] * gain, -MaxPeak, MaxPeak);
            }
            Log.Information(
                "Hifi loudness normalized active_rms_db={ActiveRmsDb:F2} target_db={TargetDb:F2} gain_db={GainDb:F2} peak_before={PeakBefore:F4}",
                currentDb,
                TargetActiveRmsDb,
                LinearToDb(gain),
                peak);
            return gain;
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
