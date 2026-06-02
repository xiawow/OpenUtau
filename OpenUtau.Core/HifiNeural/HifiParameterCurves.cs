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

    public sealed class HifiFrameParameterTrack {
        const double CacheQuantizeScale = 10.0;
        const ulong FnvOffset = 14695981039346656037UL;
        const ulong FnvPrime = 1099511628211UL;

        public double[] Gender { get; }
        public double[] Breathiness { get; }
        public double[] Tension { get; }
        public double[] Voicing { get; }
        public int FrameCount => Gender.Length;
        public HifiFrameParameterAverages Average { get; }
        public bool NeedsHnsep { get; }
        public bool HasTension { get; }
        public string CacheKey { get; }

        public HifiFrameParameterTrack(double[] gender, double[] breathiness, double[] tension, double[] voicing) {
            if (gender.Length == 0 && breathiness.Length == 0 && tension.Length == 0 && voicing.Length == 0) {
                gender = new[] { 0.0 };
                breathiness = new[] { 0.0 };
                tension = new[] { 0.0 };
                voicing = new[] { 100.0 };
            }
            int count = gender.Length;
            if (breathiness.Length != count || tension.Length != count || voicing.Length != count) {
                throw new ArgumentException("Hifi parameter arrays must have the same length.");
            }
            Gender = (double[])gender.Clone();
            Breathiness = (double[])breathiness.Clone();
            Tension = (double[])tension.Clone();
            Voicing = (double[])voicing.Clone();

            double genderSum = 0;
            double breathSum = 0;
            double tensionSum = 0;
            double voicingSum = 0;
            bool needsHnsep = false;
            bool hasTension = false;
            for (int i = 0; i < count; i++) {
                genderSum += Gender[i];
                breathSum += Breathiness[i];
                tensionSum += Tension[i];
                voicingSum += Voicing[i];
                needsHnsep |= Math.Abs(Breathiness[i]) > 0.5
                    || Math.Abs(Tension[i]) > 0.5
                    || Math.Abs(Voicing[i] - 100.0) > 0.5;
                hasTension |= Math.Abs(Tension[i]) > 0.5;
            }
            double scale = 1.0 / count;
            Average = new HifiFrameParameterAverages(
                genderSum * scale,
                breathSum * scale,
                tensionSum * scale,
                voicingSum * scale);
            NeedsHnsep = needsHnsep;
            HasTension = hasTension;
            CacheKey = BuildCacheKey();
        }

        public static HifiFrameParameterTrack Constant(HifiFrameParameterAverages value, int frameCount = 1) {
            frameCount = Math.Max(1, frameCount);
            var gender = new double[frameCount];
            var breathiness = new double[frameCount];
            var tension = new double[frameCount];
            var voicing = new double[frameCount];
            for (int i = 0; i < frameCount; i++) {
                gender[i] = value.Gender;
                breathiness[i] = value.Breathiness;
                tension[i] = value.Tension;
                voicing[i] = value.Voicing;
            }
            return new HifiFrameParameterTrack(gender, breathiness, tension, voicing);
        }

        public HifiFrameParameterAverages SampleAtSourceSample(double sourceSample, int sourceSamples) {
            double index = SourceSampleToTargetFrameIndex(sourceSample, sourceSamples);
            return SampleAtFrameIndex(index);
        }

        public double TensionAtSourceSample(double sourceSample, int sourceSamples) {
            return Sample(Tension, SourceSampleToTargetFrameIndex(sourceSample, sourceSamples));
        }

        public double BreathNoiseGainAtSourceSample(double sourceSample, int sourceSamples) {
            double breathiness = Sample(Breathiness, SourceSampleToTargetFrameIndex(sourceSample, sourceSamples));
            return Math.Clamp(1.0 + breathiness * 0.02, 0.0, 3.0);
        }

        public double VoicingGainAtSourceSample(double sourceSample, int sourceSamples) {
            double voicing = Sample(Voicing, SourceSampleToTargetFrameIndex(sourceSample, sourceSamples));
            return Math.Clamp(voicing / 100.0, 0.0, 1.5);
        }

        HifiFrameParameterAverages SampleAtFrameIndex(double index) {
            return new HifiFrameParameterAverages(
                Sample(Gender, index),
                Sample(Breathiness, index),
                Sample(Tension, index),
                Sample(Voicing, index));
        }

        double SourceSampleToTargetFrameIndex(double sourceSample, int sourceSamples) {
            if (FrameCount <= 1 || sourceSamples <= 1) {
                return 0;
            }
            double u = Math.Clamp(sourceSample / Math.Max(1.0, sourceSamples - 1.0), 0.0, 1.0);
            return u * (FrameCount - 1);
        }

        static double Sample(double[] values, double index) {
            if (values.Length == 0) {
                return 0;
            }
            if (values.Length == 1) {
                return values[0];
            }
            index = Math.Clamp(index, 0, values.Length - 1);
            int left = (int)Math.Floor(index);
            int right = Math.Min(values.Length - 1, left + 1);
            double alpha = index - left;
            return values[left] + (values[right] - values[left]) * alpha;
        }

        string BuildCacheKey() {
            unchecked {
                ulong hash = FnvOffset;
                hash = HashInt(hash, FrameCount);
                hash = HashArray(hash, Gender);
                hash = HashArray(hash, Breathiness);
                hash = HashArray(hash, Tension);
                hash = HashArray(hash, Voicing);
                return $"{FrameCount:x}-{hash:x16}";
            }
        }

        static ulong HashArray(ulong hash, double[] values) {
            for (int i = 0; i < values.Length; i++) {
                hash = HashInt(hash, (int)Math.Round(values[i] * CacheQuantizeScale));
            }
            return hash;
        }

        static ulong HashInt(ulong hash, int value) {
            unchecked {
                hash ^= (uint)value;
                hash *= FnvPrime;
                return hash;
            }
        }
    }

    public static class HifiParameterCurves {
        const int CurveTickInterval = 5;

        public static HifiFrameParameterAverages AverageForFrames(
            RenderPhrase phrase,
            double phraseStartMs,
            int startFrame,
            int frameCount) {
            return TrackForFrames(phrase, phraseStartMs, startFrame, frameCount).Average;
        }

        public static HifiFrameParameterTrack TrackForFrames(
            RenderPhrase phrase,
            double phraseStartMs,
            int startFrame,
            int frameCount) {
            frameCount = Math.Max(1, frameCount);
            var gender = new double[frameCount];
            var breathiness = new double[frameCount];
            var tension = new double[frameCount];
            var voicing = new double[frameCount];
            for (int i = 0; i < frameCount; i++) {
                double frameCenterMs = phraseStartMs + (startFrame + i + 0.5) * HifiF0Builder.FrameMs;
                gender[i] = Sample(phrase, phrase.gender, frameCenterMs, 0);
                breathiness[i] = Sample(phrase, phrase.breathiness, frameCenterMs, 0);
                tension[i] = Sample(phrase, phrase.tension, frameCenterMs, 0);
                voicing[i] = Sample(phrase, phrase.voicing, frameCenterMs, 100);
            }
            return new HifiFrameParameterTrack(gender, breathiness, tension, voicing);
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
