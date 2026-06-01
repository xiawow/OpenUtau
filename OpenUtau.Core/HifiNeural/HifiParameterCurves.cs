using System;
using OpenUtau.Core.Render;

namespace OpenUtau.Core.HifiNeural {
    public readonly record struct HifiFrameParameterAverages(
        double Gender,
        double Breathiness,
        double Tension,
        double Voicing) {
        public double GenderKeyShiftSemitones => Math.Clamp(Gender, -100.0, 100.0) * 0.12;
        public double BreathNoiseGain => Math.Clamp(1.0 + Breathiness * 0.02, 0.0, 3.0);
        public double VoicingGain => Math.Clamp(Voicing / 100.0, 0.0, 1.5);
        public bool HasGender => Math.Abs(Gender) > 0.5;
        public bool NeedsHnsep => Math.Abs(Breathiness) > 0.5 || Math.Abs(Tension) > 0.5 || Math.Abs(Voicing - 100.0) > 0.5;
    }

    public static class HifiParameterCurves {
        const int CurveTickInterval = 5;

        public static HifiFrameParameterAverages AverageForFrames(
            RenderPhrase phrase,
            double phraseStartMs,
            int startFrame,
            int frameCount) {
            frameCount = Math.Max(1, frameCount);
            double gender = 0;
            double breathiness = 0;
            double tension = 0;
            double voicing = 0;
            for (int i = 0; i < frameCount; i++) {
                double frameCenterMs = phraseStartMs + (startFrame + i + 0.5) * HifiF0Builder.FrameMs;
                gender += Sample(phrase, phrase.gender, frameCenterMs, 0);
                breathiness += Sample(phrase, phrase.breathiness, frameCenterMs, 0);
                tension += Sample(phrase, phrase.tension, frameCenterMs, 0);
                voicing += Sample(phrase, phrase.voicing, frameCenterMs, 100);
            }
            double scale = 1.0 / frameCount;
            return new HifiFrameParameterAverages(
                gender * scale,
                breathiness * scale,
                tension * scale,
                voicing * scale);
        }

        public static double Sample(RenderPhrase phrase, float[]? curve, double absoluteMs, double defaultValue) {
            if (curve == null || curve.Length == 0) {
                return defaultValue;
            }
            int tick = phrase.timeAxis.MsPosToTickPos(absoluteMs);
            double curveIndex = (tick - (phrase.position - phrase.leading)) / (double)CurveTickInterval;
            if (curveIndex <= 0) {
                return curve[0];
            }
            if (curveIndex >= curve.Length - 1) {
                return curve[^1];
            }
            int left = (int)Math.Floor(curveIndex);
            int right = left + 1;
            double alpha = curveIndex - left;
            return curve[left] + (curve[right] - curve[left]) * alpha;
        }
    }
}
