using System;
using System.Linq;
using OpenUtau.Core.Render;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiGrowlProcessor {
        public const string CurveAbbr = "groc";
        public const string CurveName = "growl/roughness (curve)";

        const double MaxVibratoCents = 100.0;
        const double LfoHz = 80.0;
        const double HighpassHz = 400.0;
        const double MaxPeak = 0.98;

        public static bool HasActiveCurve(RenderPhrase phrase) {
            var curve = phrase.curves.FirstOrDefault(c => string.Equals(c.Item1, CurveAbbr, StringComparison.OrdinalIgnoreCase))?.Item2;
            return curve != null && curve.Any(v => Math.Abs(v) > 0.5f);
        }

        public static void ApplyInPlace(float[] samples, RenderPhrase phrase, double phraseStartMs, int sampleRate) {
            if (samples.Length == 0) {
                return;
            }
            var curve = phrase.curves.FirstOrDefault(c => string.Equals(c.Item1, CurveAbbr, StringComparison.OrdinalIgnoreCase))?.Item2;
            if (curve == null || curve.Length == 0 || curve.All(v => Math.Abs(v) <= 0.5f)) {
                return;
            }
            var strength = BuildStrengthEnvelope(phrase, curve, samples.Length, sampleRate, phraseStartMs);
            ApplyInPlace(samples, sampleRate, strength);
        }

        public static void ApplyInPlace(float[] samples, int sampleRate, float[] strength) {
            if (samples.Length == 0 || strength.Length == 0) {
                return;
            }
            int length = Math.Min(samples.Length, strength.Length);
            double maxStrength = 0;
            for (int i = 0; i < length; i++) {
                maxStrength = Math.Max(maxStrength, Math.Abs(strength[i]));
            }
            if (maxStrength <= 1e-4) {
                return;
            }

            var band = samples.Take(length).ToArray();
            ApplyHighpassInPlace(band, sampleRate, HighpassHz);
            var complement = new float[length];
            for (int i = 0; i < length; i++) {
                complement[i] = samples[i] - band[i];
            }

            var modulated = ApplyPitchModulation(band, sampleRate, strength);
            double bandRms = Rms(band);
            double modRms = Rms(modulated);
            if (bandRms > 1e-8 && modRms > 1e-8) {
                double gain = bandRms / modRms;
                for (int i = 0; i < length; i++) {
                    modulated[i] = (float)(modulated[i] * gain);
                }
            }

            double originalRms = Rms(samples, length);
            for (int i = 0; i < length; i++) {
                samples[i] = complement[i] + modulated[i];
            }
            MatchRms(samples, length, originalRms);
            LimitPeak(samples);
        }

        static float[] BuildStrengthEnvelope(RenderPhrase phrase, float[] curve, int sampleCount, int sampleRate, double phraseStartMs) {
            const int interval = 5;
            var strength = new float[sampleCount];
            if (sampleCount == 0) {
                return strength;
            }
            for (int i = 0; i < sampleCount; i++) {
                double posMs = phraseStartMs + i * 1000.0 / sampleRate;
                int tick = phrase.timeAxis.MsPosToTickPos(posMs);
                double curveIndex = (tick - (phrase.position - phrase.leading)) / (double)interval;
                double value;
                if (curveIndex <= 0) {
                    value = curve[0];
                } else if (curveIndex >= curve.Length - 1) {
                    value = curve[^1];
                } else {
                    int left = (int)Math.Floor(curveIndex);
                    int right = left + 1;
                    double alpha = curveIndex - left;
                    value = curve[left] + (curve[right] - curve[left]) * alpha;
                }
                strength[i] = (float)Math.Clamp(value / 100.0, 0.0, 1.0);
            }
            SmoothEnvelopeInPlace(strength, Math.Max(1, (int)Math.Round(sampleRate * 0.012)));
            return strength;
        }

        static float[] ApplyPitchModulation(float[] band, int sampleRate, float[] strength) {
            int length = Math.Min(band.Length, strength.Length);
            var output = new float[length];
            if (length == 0) {
                return output;
            }
            var drift = new double[length];
            double phase = 0;
            double cumulative = 0;
            for (int i = 0; i < length; i++) {
                phase += LfoHz / sampleRate;
                phase -= Math.Floor(phase);
                double lfo = phase < 0.5 ? 1.0 : -1.0;
                double cents = lfo * strength[i] * MaxVibratoCents;
                double ratio = Math.Pow(2.0, cents / 1200.0);
                cumulative += ratio - 1.0;
                drift[i] = cumulative;
            }
            RemoveSlowDriftInPlace(drift, sampleRate);
            for (int i = 0; i < length; i++) {
                double sourceIndex = Math.Clamp(i + drift[i], 0, length - 1);
                output[i] = SampleLinear(band, sourceIndex);
            }
            return output;
        }

        static void ApplyHighpassInPlace(float[] samples, int sampleRate, double cutoffHz) {
            // Four-pole RBJ high-pass, close in intent to hifisampler's 4th-order Butterworth.
            ApplyHighpassBiquadInPlace(samples, sampleRate, cutoffHz, 0.5411961);
            ApplyHighpassBiquadInPlace(samples, sampleRate, cutoffHz, 1.3065630);
        }

        static void ApplyHighpassBiquadInPlace(float[] samples, int sampleRate, double cutoffHz, double q) {
            double w0 = 2.0 * Math.PI * Math.Clamp(cutoffHz, 1.0, sampleRate * 0.45) / sampleRate;
            double cos = Math.Cos(w0);
            double sin = Math.Sin(w0);
            double alpha = sin / (2.0 * q);
            double b0 = (1.0 + cos) * 0.5;
            double b1 = -(1.0 + cos);
            double b2 = (1.0 + cos) * 0.5;
            double a0 = 1.0 + alpha;
            double a1 = -2.0 * cos;
            double a2 = 1.0 - alpha;
            b0 /= a0;
            b1 /= a0;
            b2 /= a0;
            a1 /= a0;
            a2 /= a0;

            double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
            for (int i = 0; i < samples.Length; i++) {
                double x0 = samples[i];
                double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                samples[i] = (float)y0;
                x2 = x1;
                x1 = x0;
                y2 = y1;
                y1 = y0;
            }
        }

        static void RemoveSlowDriftInPlace(double[] drift, int sampleRate) {
            int window = Math.Max(1, (int)Math.Round(sampleRate / 20.0));
            double sum = 0;
            var slow = new double[drift.Length];
            for (int i = 0; i < drift.Length; i++) {
                sum += drift[i];
                if (i >= window) {
                    sum -= drift[i - window];
                }
                slow[i] = sum / Math.Min(i + 1, window);
            }
            for (int i = 0; i < drift.Length; i++) {
                drift[i] -= slow[i];
            }
        }

        static void SmoothEnvelopeInPlace(float[] values, int radius) {
            if (values.Length == 0 || radius <= 0) {
                return;
            }
            var copy = values.ToArray();
            for (int i = 0; i < values.Length; i++) {
                double sum = 0;
                double weightSum = 0;
                int start = Math.Max(0, i - radius);
                int end = Math.Min(values.Length - 1, i + radius);
                for (int j = start; j <= end; j++) {
                    double distance = Math.Abs(i - j) / (double)(radius + 1);
                    double weight = 0.5 + 0.5 * Math.Cos(Math.PI * Math.Clamp(distance, 0, 1));
                    sum += copy[j] * weight;
                    weightSum += weight;
                }
                values[i] = (float)(sum / Math.Max(1e-9, weightSum));
            }
        }

        static float SampleLinear(float[] samples, double index) {
            int left = (int)Math.Floor(index);
            int right = Math.Min(samples.Length - 1, left + 1);
            double alpha = index - left;
            return (float)(samples[left] + (samples[right] - samples[left]) * alpha);
        }

        static void MatchRms(float[] samples, int length, double targetRms) {
            double current = Rms(samples, length);
            if (current <= 1e-8 || targetRms <= 1e-8) {
                return;
            }
            double gain = Math.Clamp(targetRms / current, 0.25, 4.0);
            for (int i = 0; i < length; i++) {
                samples[i] = (float)(samples[i] * gain);
            }
        }

        static void LimitPeak(float[] samples) {
            double peak = 0;
            for (int i = 0; i < samples.Length; i++) {
                peak = Math.Max(peak, Math.Abs(samples[i]));
            }
            if (peak <= MaxPeak || peak <= 1e-9) {
                return;
            }
            double gain = MaxPeak / peak;
            for (int i = 0; i < samples.Length; i++) {
                samples[i] = (float)(samples[i] * gain);
            }
        }

        static double Rms(float[] samples) {
            return Rms(samples, samples.Length);
        }

        static double Rms(float[] samples, int length) {
            length = Math.Min(samples.Length, Math.Max(0, length));
            if (length == 0) {
                return 0;
            }
            double sum = 0;
            for (int i = 0; i < length; i++) {
                sum += samples[i] * samples[i];
            }
            return Math.Sqrt(sum / length);
        }
    }
}
