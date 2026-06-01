using System;
using System.Collections.Generic;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public readonly record struct HifiPostVocoderLevelerReport(
        int FrameCount,
        int ActiveFrames,
        double ReferenceRmsDb,
        double ReferenceF0,
        double MinGainDb,
        double MaxGainDb,
        int CutFrames,
        int BoostFrames);

    public static class HifiPostVocoderLeveler {
        const int HopSize = HifiOnnxVocoder.HopSize;
        const double WindowMs = 72.0;
        const double ActiveFloor = 0.0025;
        const double ActivePeakRatio = 0.015;
        const double ReferencePercentile = 0.40;
        const double F0ReferencePercentile = 0.40;
        const double LoudToleranceDb = 1.5;
        const double QuietToleranceDb = 4.0;
        const double CutStrength = 0.78;
        const double BoostStrength = 0.22;
        const double MaxCutDb = 5.5;
        const double MaxBoostDb = 1.2;
        const int PhoneStartNoBoostFrames = 4;
        const int PhraseEdgeNoBoostFrames = 5;
        const int SmoothHalfFrames = 4;
        const double MaxCutStepDb = 1.15;
        const double MaxReleaseStepDb = 0.40;
        const double SilenceThreshold = 1e-7;

        public static HifiPostVocoderLevelerReport LevelInPlace(float[] samples, HifiPhraseFeatures features, int sampleRate) {
            if (samples.Length == 0) {
                return default;
            }
            int frameCount = Math.Max(1, (samples.Length + HopSize - 1) / HopSize);
            var rms = ComputeFrameRms(samples, sampleRate, frameCount);
            double peak = Peak(samples);
            double activeThreshold = Math.Max(ActiveFloor, peak * ActivePeakRatio);
            var activeDbs = new List<double>(frameCount);
            var voicedF0s = new List<double>(frameCount);
            var active = new bool[frameCount];
            var frameF0 = new double[frameCount];

            for (int i = 0; i < frameCount; i++) {
                double f0 = features.F0.Length == 0 ? 0 : features.F0[Math.Min(features.F0.Length - 1, i)];
                frameF0[i] = f0;
                if (rms[i] >= activeThreshold && IsFinite(rms[i])) {
                    active[i] = true;
                    activeDbs.Add(LinearToDb(rms[i]));
                    if (f0 >= 55 && f0 <= 1400) {
                        voicedF0s.Add(f0);
                    }
                }
            }

            if (activeDbs.Count < 3) {
                return new HifiPostVocoderLevelerReport(frameCount, activeDbs.Count, 0, 0, 0, 0, 0, 0);
            }

            double referenceDb = Percentile(activeDbs, ReferencePercentile);
            double referenceF0 = voicedF0s.Count >= 3 ? Percentile(voicedF0s, F0ReferencePercentile) : 0;
            bool[] noBoost = BuildNoBoostMask(frameCount, features.Metadata);
            var desiredGainDb = new double[frameCount];

            for (int i = 0; i < frameCount; i++) {
                if (!active[i]) {
                    desiredGainDb[i] = 0;
                    continue;
                }

                double localDb = LinearToDb(rms[i]);
                double gainDb = 0;
                double loudExcess = localDb - (referenceDb + LoudToleranceDb);
                if (loudExcess > 0) {
                    gainDb -= Math.Min(MaxCutDb, loudExcess * CutStrength);
                }

                bool voiced = frameF0[i] >= 55 && frameF0[i] <= 1400;
                if (!noBoost[i] && voiced) {
                    double quietDeficit = (referenceDb - QuietToleranceDb) - localDb;
                    if (quietDeficit > 0) {
                        gainDb += Math.Min(MaxBoostDb, quietDeficit * BoostStrength);
                    }
                }

                desiredGainDb[i] = Math.Clamp(gainDb, -MaxCutDb, MaxBoostDb);
            }

            double[] gainDbEnvelope = SmoothGainEnvelope(desiredGainDb);
            ApplyGainEnvelope(samples, gainDbEnvelope);

            double minGainDb = 0;
            double maxGainDb = 0;
            int cutFrames = 0;
            int boostFrames = 0;
            foreach (double gainDb in gainDbEnvelope) {
                minGainDb = Math.Min(minGainDb, gainDb);
                maxGainDb = Math.Max(maxGainDb, gainDb);
                if (gainDb < -0.05) {
                    cutFrames++;
                } else if (gainDb > 0.05) {
                    boostFrames++;
                }
            }

            Log.Information(
                "Hifi post-vocoder local loudness leveler frames={Frames} active_frames={ActiveFrames} reference_rms_db={ReferenceDb:F2} reference_f0={ReferenceF0:F2} min_gain_db={MinGainDb:F2} max_gain_db={MaxGainDb:F2} cut_frames={CutFrames} boost_frames={BoostFrames}",
                frameCount,
                activeDbs.Count,
                referenceDb,
                referenceF0,
                minGainDb,
                maxGainDb,
                cutFrames,
                boostFrames);

            return new HifiPostVocoderLevelerReport(
                frameCount,
                activeDbs.Count,
                referenceDb,
                referenceF0,
                minGainDb,
                maxGainDb,
                cutFrames,
                boostFrames);
        }

        static double[] ComputeFrameRms(float[] samples, int sampleRate, int frameCount) {
            int window = Math.Clamp((int)Math.Round(sampleRate * WindowMs / 1000.0), 1024, 4096);
            int halfWindow = window / 2;
            var rms = new double[frameCount];
            for (int frame = 0; frame < frameCount; frame++) {
                int center = frame * HopSize + HopSize / 2;
                int start = Math.Clamp(center - halfWindow, 0, samples.Length);
                int end = Math.Clamp(center + halfWindow, start, samples.Length);
                double sum = 0;
                for (int i = start; i < end; i++) {
                    double sample = samples[i];
                    sum += sample * sample;
                }
                rms[frame] = end > start ? Math.Sqrt(sum / (end - start)) : 0;
            }
            return rms;
        }

        static bool[] BuildNoBoostMask(int frameCount, HifiPhraseMetadata metadata) {
            var noBoost = new bool[frameCount];
            for (int i = 0; i < Math.Min(PhraseEdgeNoBoostFrames, frameCount); i++) {
                noBoost[i] = true;
                noBoost[frameCount - 1 - i] = true;
            }
            foreach (var phone in metadata.Phones) {
                int start = Math.Clamp(phone.StartFrame, 0, frameCount);
                int end = Math.Clamp(start + Math.Min(PhoneStartNoBoostFrames, Math.Max(0, phone.FrameCount)), start, frameCount);
                for (int frame = start; frame < end; frame++) {
                    noBoost[frame] = true;
                }
            }
            return noBoost;
        }

        static double[] SmoothGainEnvelope(double[] desiredGainDb) {
            if (desiredGainDb.Length <= 1) {
                return desiredGainDb;
            }
            double[] smoothed = WeightedSmooth(desiredGainDb, SmoothHalfFrames);
            double[] forward = new double[smoothed.Length];
            double[] backward = new double[smoothed.Length];

            forward[0] = smoothed[0];
            for (int i = 1; i < smoothed.Length; i++) {
                forward[i] = LimitStep(forward[i - 1], smoothed[i]);
            }
            backward[^1] = smoothed[^1];
            for (int i = smoothed.Length - 2; i >= 0; i--) {
                backward[i] = LimitStep(backward[i + 1], smoothed[i]);
            }

            var result = new double[smoothed.Length];
            for (int i = 0; i < result.Length; i++) {
                // Prefer the stronger cut from either pass so a short loud region is controlled
                // before and after its center, but keep boosts conservative.
                result[i] = Math.Min(forward[i], backward[i]);
                if (result[i] > 0) {
                    result[i] = Math.Min(result[i], smoothed[i]);
                }
            }
            return result;
        }

        static double[] WeightedSmooth(double[] values, int halfWindow) {
            var output = new double[values.Length];
            for (int i = 0; i < values.Length; i++) {
                double sum = 0;
                double weightSum = 0;
                int start = Math.Max(0, i - halfWindow);
                int end = Math.Min(values.Length - 1, i + halfWindow);
                for (int j = start; j <= end; j++) {
                    double distance = Math.Abs(i - j) / (double)Math.Max(1, halfWindow + 1);
                    double weight = 0.5 + 0.5 * Math.Cos(Math.PI * Math.Clamp(distance, 0, 1));
                    sum += values[j] * weight;
                    weightSum += weight;
                }
                output[i] = weightSum > 0 ? sum / weightSum : values[i];
            }
            return output;
        }

        static double LimitStep(double previous, double target) {
            double delta = target - previous;
            if (delta < -MaxCutStepDb) {
                return previous - MaxCutStepDb;
            }
            if (delta > MaxReleaseStepDb) {
                return previous + MaxReleaseStepDb;
            }
            return target;
        }

        static void ApplyGainEnvelope(float[] samples, double[] gainDbEnvelope) {
            if (gainDbEnvelope.Length == 0) {
                return;
            }
            for (int i = 0; i < samples.Length; i++) {
                double position = i / (double)HopSize;
                int left = (int)Math.Floor(position);
                int right = Math.Min(gainDbEnvelope.Length - 1, left + 1);
                left = Math.Clamp(left, 0, gainDbEnvelope.Length - 1);
                double alpha = Math.Clamp(position - Math.Floor(position), 0, 1);
                double gainDb = gainDbEnvelope[left] + (gainDbEnvelope[right] - gainDbEnvelope[left]) * alpha;
                double gain = Math.Pow(10.0, gainDb / 20.0);
                if (IsFinite(gain)) {
                    samples[i] = (float)(samples[i] * gain);
                }
            }
        }

        static double Percentile(List<double> values, double percentile) {
            values.Sort();
            if (values.Count == 0) {
                return 0;
            }
            double index = Math.Clamp(percentile, 0, 1) * (values.Count - 1);
            int left = (int)Math.Floor(index);
            int right = Math.Min(values.Count - 1, left + 1);
            double alpha = index - left;
            return values[left] + (values[right] - values[left]) * alpha;
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
