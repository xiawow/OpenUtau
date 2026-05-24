using System;
using System.Collections.Generic;
using System.Linq;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiAudioEnvelopeNormalizer {
        public static void Apply(
            float[] samples,
            HifiPhraseFeatures features,
            double maxAdjustDb,
            double deadbandDb,
            double strength) {
            if (samples.Length == 0 || features.Frames <= 1 || strength <= 0) {
                return;
            }

            int frames = Math.Min(features.Frames, (samples.Length + HifiF0Builder.HopSize - 1) / HifiF0Builder.HopSize);
            if (frames <= 1) {
                return;
            }

            var frameDb = MeasureFrameDb(samples, frames);
            var voiced = BuildVoicedMask(features.F0, frames);
            var voicedDb = Enumerable.Range(0, frames)
                .Where(t => voiced[t] && IsFinite(frameDb[t]) && frameDb[t] > -80)
                .Select(t => frameDb[t])
                .OrderBy(value => value)
                .ToArray();
            if (voicedDb.Length < 4) {
                return;
            }

            maxAdjustDb = Math.Clamp(maxAdjustDb, 0.5, 6.0);
            deadbandDb = Math.Clamp(deadbandDb, 0.5, 4.0);
            strength = Math.Clamp(strength, 0.05, 1.0);
            double referenceDb = Percentile(voicedDb, 0.60);

            var gainDb = new double[frames];
            int adjustedFrames = 0;
            for (int t = 0; t < frames; t++) {
                if (!voiced[t] || !IsFinite(frameDb[t])) {
                    continue;
                }
                double deltaDb = referenceDb - frameDb[t];
                if (Math.Abs(deltaDb) <= deadbandDb) {
                    continue;
                }

                double mappedDb;
                if (deltaDb > 0) {
                    mappedDb = (deltaDb - deadbandDb) * 0.45 * strength;
                    mappedDb = Math.Clamp(mappedDb, 0, maxAdjustDb * 0.75);
                } else {
                    mappedDb = (deltaDb + deadbandDb) * 0.75 * strength;
                    mappedDb = Math.Clamp(mappedDb, -maxAdjustDb, 0);
                }
                gainDb[t] = mappedDb;
                adjustedFrames++;
            }

            if (adjustedFrames > 0) {
                gainDb = SmoothGain(gainDb, radius: 4);
                ApplyInterpolatedFrameGain(samples, gainDb);
            }

            double voicedRmsBeforeDb = VoicedRmsDb(frameDb, voiced);
            double voicedRmsAfterDb = VoicedRmsDb(MeasureFrameDb(samples, frames), voiced);
            double makeupDb = 0;
            if (IsFinite(voicedRmsAfterDb)) {
                makeupDb = Math.Clamp(
                    HifiNeuralConfig.OutputTargetVoicedRmsDb - voicedRmsAfterDb,
                    -2.0,
                    HifiNeuralConfig.OutputMaxMakeupDb);
                if (Math.Abs(makeupDb) >= 0.05) {
                    ApplyScalarGain(samples, makeupDb);
                }
            }

            if (HifiNeuralConfig.EnableOutputExciter) {
                ApplySoftExciter(samples, HifiNeuralConfig.OutputExciterDrive, HifiNeuralConfig.OutputExciterMix);
            }
            ApplyPeakLimiter(samples, HifiNeuralConfig.OutputPeakLimit);

            Log.Information(
                "HifiAudioEnvelopeNormalizer minimal adjusted_frames={AdjustedFrames} total_frames={Frames} reference_db={ReferenceDb:F2} voiced_rms_before_db={VoicedRmsBeforeDb:F2} voiced_rms_after_db={VoicedRmsAfterDb:F2} makeup_db={MakeupDb:F2} exciter={Exciter} exciter_drive={ExciterDrive:F2} exciter_mix={ExciterMix:F2} peak_limit={PeakLimit:F2}",
                adjustedFrames,
                frames,
                referenceDb,
                voicedRmsBeforeDb,
                voicedRmsAfterDb,
                makeupDb,
                HifiNeuralConfig.EnableOutputExciter,
                HifiNeuralConfig.OutputExciterDrive,
                HifiNeuralConfig.OutputExciterMix,
                HifiNeuralConfig.OutputPeakLimit);
        }

        static double[] MeasureFrameDb(float[] samples, int frames) {
            var result = new double[frames];
            for (int t = 0; t < frames; t++) {
                int start = t * HifiF0Builder.HopSize;
                int end = Math.Min(samples.Length, start + HifiF0Builder.HopSize);
                if (start >= end) {
                    result[t] = double.NegativeInfinity;
                    continue;
                }
                double sum = 0;
                for (int i = start; i < end; i++) {
                    float s = samples[i];
                    if (!float.IsFinite(s)) {
                        sum = double.NaN;
                        break;
                    }
                    sum += s * s;
                }
                if (double.IsNaN(sum)) {
                    result[t] = double.NegativeInfinity;
                    continue;
                }
                double rms = Math.Sqrt(sum / Math.Max(1, end - start));
                result[t] = 20.0 * Math.Log10(Math.Max(1e-9, rms));
            }
            return result;
        }

        static bool[] BuildVoicedMask(float[] f0, int frames) {
            var voiced = new bool[frames];
            int f0Frames = Math.Min(frames, f0.Length);
            for (int t = 0; t < f0Frames; t++) {
                float value = f0[t];
                voiced[t] = value > 0 && float.IsFinite(value);
            }
            return voiced;
        }

        static double VoicedRmsDb(IReadOnlyList<double> frameDb, IReadOnlyList<bool> voiced) {
            double sum = 0;
            int count = 0;
            for (int i = 0; i < frameDb.Count; i++) {
                if (!voiced[i] || !IsFinite(frameDb[i])) {
                    continue;
                }
                sum += frameDb[i];
                count++;
            }
            return count > 0 ? sum / count : double.NaN;
        }

        static double[] SmoothGain(IReadOnlyList<double> gainDb, int radius) {
            var result = new double[gainDb.Count];
            for (int t = 0; t < gainDb.Count; t++) {
                double sum = 0;
                double weightSum = 0;
                int start = Math.Max(0, t - radius);
                int end = Math.Min(gainDb.Count - 1, t + radius);
                for (int i = start; i <= end; i++) {
                    double x = Math.Abs(i - t) / (double)(radius + 1);
                    double w = 0.5 + 0.5 * Math.Cos(Math.PI * Math.Clamp(x, 0, 1));
                    sum += gainDb[i] * w;
                    weightSum += w;
                }
                result[t] = weightSum > 0 ? sum / weightSum : gainDb[t];
            }
            return result;
        }

        static void ApplyInterpolatedFrameGain(float[] samples, IReadOnlyList<double> frameGainDb) {
            for (int i = 0; i < samples.Length; i++) {
                double frame = i / (double)HifiF0Builder.HopSize;
                int left = Math.Clamp((int)Math.Floor(frame), 0, frameGainDb.Count - 1);
                int right = Math.Min(frameGainDb.Count - 1, left + 1);
                double alpha = frame - Math.Floor(frame);
                double gainDb = frameGainDb[left] + (frameGainDb[right] - frameGainDb[left]) * alpha;
                samples[i] *= (float)Math.Pow(10, gainDb / 20.0);
            }
        }

        static void ApplyScalarGain(float[] samples, double gainDb) {
            if (Math.Abs(gainDb) < 1e-6) {
                return;
            }
            float gain = (float)Math.Pow(10, gainDb / 20.0);
            for (int i = 0; i < samples.Length; i++) {
                samples[i] *= gain;
            }
        }

        static void ApplySoftExciter(float[] samples, double drive, double mix) {
            drive = Math.Clamp(drive, 1.0, 4.0);
            mix = Math.Clamp(mix, 0.0, 0.35);
            if (mix <= 1e-6) {
                return;
            }
            float norm = (float)(1.0 / Math.Tanh(drive));
            float wetMix = (float)mix;
            float dryMix = 1f - wetMix;
            for (int i = 0; i < samples.Length; i++) {
                float dry = samples[i];
                float wet = (float)Math.Tanh(dry * drive) * norm;
                samples[i] = dry * dryMix + wet * wetMix;
            }
        }

        static void ApplyPeakLimiter(float[] samples, double peakLimit) {
            peakLimit = Math.Clamp(peakLimit, 0.75, 0.999);
            float peak = 0;
            for (int i = 0; i < samples.Length; i++) {
                peak = Math.Max(peak, Math.Abs(samples[i]));
            }
            if (peak <= peakLimit || peak <= 1e-9f) {
                return;
            }
            float scale = (float)(peakLimit / peak);
            for (int i = 0; i < samples.Length; i++) {
                samples[i] *= scale;
            }
        }

        static double Percentile(IReadOnlyList<double> sorted, double q) {
            if (sorted.Count == 0) {
                return double.NaN;
            }
            if (sorted.Count == 1) {
                return sorted[0];
            }
            q = Math.Clamp(q, 0, 1);
            double p = q * (sorted.Count - 1);
            int i = (int)Math.Floor(p);
            int j = Math.Min(sorted.Count - 1, i + 1);
            double a = p - i;
            return sorted[i] + (sorted[j] - sorted[i]) * a;
        }

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
