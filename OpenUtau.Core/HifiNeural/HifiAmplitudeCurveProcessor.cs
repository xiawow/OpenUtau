using System;
using System.Linq;
using OpenUtau.Core.Format;
using OpenUtau.Core.Render;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiAmplitudeCurveProcessor {
        public const string CurveName = "amplitude by pitch (curve)";

        const int CurveTickInterval = 5;
        const double MinVoicedHz = 55.0;
        const double MaxVoicedHz = 1400.0;
        const double SlopeClampSemitonesPerSecond = 18.0;
        const double MaxGainDb = 1.1;
        const double SmoothMs = 60.0;
        const double ActiveCurveThreshold = 0.5;

        public static bool HasActiveCurve(RenderPhrase phrase) {
            var curve = FindCurve(phrase);
            return curve != null && curve.Any(v => Math.Abs(v) > ActiveCurveThreshold);
        }

        public static void ApplyInPlace(float[] samples, RenderPhrase phrase, HifiPhraseFeatures features, double phraseStartMs, int sampleRate) {
            if (samples.Length == 0 || features.F0.Length < 2) {
                return;
            }
            var curve = FindCurve(phrase);
            if (curve == null || curve.Length == 0 || curve.All(v => Math.Abs(v) <= ActiveCurveThreshold)) {
                return;
            }

            var gainDb = BuildFrameGainDb(phrase, curve, features.F0, phraseStartMs);
            if (!gainDb.Any(v => Math.Abs(v) > 1e-4)) {
                return;
            }
            SmoothInPlace(gainDb, Math.Max(1, (int)Math.Round(SmoothMs / HifiF0Builder.FrameMs)));
            ApplyGainEnvelope(samples, gainDb);

            double minGain = 0;
            double maxGain = 0;
            int activeFrames = 0;
            foreach (double gain in gainDb) {
                minGain = Math.Min(minGain, gain);
                maxGain = Math.Max(maxGain, gain);
                if (Math.Abs(gain) > 0.05) {
                    activeFrames++;
                }
            }
            Log.Information(
                "Hifi AC amplitude-by-pitch applied frames={Frames} active_frames={ActiveFrames} min_gain_db={MinGainDb:F2} max_gain_db={MaxGainDb:F2}",
                gainDb.Length,
                activeFrames,
                minGain,
                maxGain);
        }

        static float[]? FindCurve(RenderPhrase phrase) {
            return phrase.curves
                .FirstOrDefault(c => string.Equals(c.Item1, Format.Ustx.AC, StringComparison.OrdinalIgnoreCase))
                ?.Item2;
        }

        static double[] BuildFrameGainDb(RenderPhrase phrase, float[] curve, float[] f0, double phraseStartMs) {
            var gainDb = new double[f0.Length];
            for (int frame = 0; frame < f0.Length; frame++) {
                double hz = f0[frame];
                if (!IsVoiced(hz)) {
                    continue;
                }
                int leftFrame = Math.Max(0, frame - 1);
                int rightFrame = Math.Min(f0.Length - 1, frame + 1);
                if (!IsVoiced(f0[leftFrame]) || !IsVoiced(f0[rightFrame]) || leftFrame == rightFrame) {
                    continue;
                }

                double seconds = (rightFrame - leftFrame) * HifiF0Builder.FrameMs / 1000.0;
                if (seconds <= 1e-6) {
                    continue;
                }
                double semitoneDelta = 12.0 * Math.Log(f0[rightFrame] / f0[leftFrame], 2.0);
                double semitonesPerSecond = Math.Clamp(semitoneDelta / seconds, -SlopeClampSemitonesPerSecond, SlopeClampSemitonesPerSecond);
                double ac = SampleCurveAtFrame(phrase, curve, frame, phraseStartMs);
                if (Math.Abs(ac) <= ActiveCurveThreshold) {
                    continue;
                }

                // Match hifisampler's intent: gain follows pitch slope, not absolute pitch.
                double gain = Math.Pow(5.0, 1e-4 * Math.Clamp(ac, -100.0, 100.0) * semitonesPerSecond);
                if (!IsFinite(gain) || gain <= 0) {
                    continue;
                }
                gainDb[frame] = Math.Clamp(20.0 * Math.Log10(gain), -MaxGainDb, MaxGainDb);
            }
            return gainDb;
        }

        static double SampleCurveAtFrame(RenderPhrase phrase, float[] curve, int frame, double phraseStartMs) {
            if (curve.Length == 0) {
                return 0;
            }
            double posMs = phraseStartMs + (frame + 0.5) * HifiF0Builder.FrameMs;
            int tick = phrase.timeAxis.MsPosToTickPos(posMs);
            double index = (tick - (phrase.position - phrase.leading)) / (double)CurveTickInterval;
            if (index <= 0) {
                return curve[0];
            }
            if (index >= curve.Length - 1) {
                return curve[^1];
            }
            int left = (int)Math.Floor(index);
            int right = left + 1;
            double alpha = index - left;
            return curve[left] + (curve[right] - curve[left]) * alpha;
        }

        static void SmoothInPlace(double[] values, int halfWindow) {
            if (values.Length == 0 || halfWindow <= 0) {
                return;
            }
            var copy = (double[])values.Clone();
            for (int i = 0; i < values.Length; i++) {
                double sum = 0;
                double weightSum = 0;
                int start = Math.Max(0, i - halfWindow);
                int end = Math.Min(values.Length - 1, i + halfWindow);
                for (int j = start; j <= end; j++) {
                    double distance = Math.Abs(i - j) / (double)(halfWindow + 1);
                    double weight = 0.5 + 0.5 * Math.Cos(Math.PI * Math.Clamp(distance, 0, 1));
                    sum += copy[j] * weight;
                    weightSum += weight;
                }
                values[i] = weightSum > 0 ? Math.Clamp(sum / weightSum, -MaxGainDb, MaxGainDb) : copy[i];
            }
        }

        static void ApplyGainEnvelope(float[] samples, double[] gainDb) {
            if (gainDb.Length == 0) {
                return;
            }
            for (int i = 0; i < samples.Length; i++) {
                double position = Math.Max(0, (i - HifiF0Builder.HopSize * 0.5) / HifiF0Builder.HopSize);
                int left = Math.Clamp((int)Math.Floor(position), 0, gainDb.Length - 1);
                int right = Math.Min(gainDb.Length - 1, left + 1);
                double alpha = Math.Clamp(position - Math.Floor(position), 0, 1);
                double db = gainDb[left] + (gainDb[right] - gainDb[left]) * alpha;
                double gain = Math.Pow(10.0, db / 20.0);
                if (IsFinite(gain)) {
                    samples[i] = (float)(samples[i] * gain);
                }
            }
        }

        static bool IsVoiced(double hz) => hz >= MinVoicedHz && hz <= MaxVoicedHz && IsFinite(hz);

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
