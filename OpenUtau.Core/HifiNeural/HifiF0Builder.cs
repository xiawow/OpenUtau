using System;
using System.Linq;
using OpenUtau.Core.Render;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiF0Builder {
        public const int SampleRate = 44100;
        public const int HopSize = 512;
        public const double FrameMs = 1000.0 * HopSize / SampleRate;

        public float[] Build(RenderPhrase phrase, int frames, double startMs) {
            var f0 = new float[frames];
            for (int i = 0; i < frames; i++) {
                double posMs = startMs + i * FrameMs;
                int tick = phrase.timeAxis.MsPosToTickPos(posMs);
                double tone = ToneAt(phrase, tick, posMs);
                f0[i] = tone > 0 ? (float)MusicMath.ToneToFreq(tone) : 0f;
            }
            return f0;
        }

        double ToneAt(RenderPhrase phrase, int absoluteTick, double posMs) {
            const int pitchInterval = 5;
            int relativeTick = absoluteTick - (phrase.position - phrase.leading);
            if (phrase.pitches != null && phrase.pitches.Length > 0) {
                double index = Math.Clamp(relativeTick / (double)pitchInterval, 0, phrase.pitches.Length - 1);
                int left = (int)Math.Floor(index);
                int right = Math.Min(left + 1, phrase.pitches.Length - 1);
                double alpha = index - left;
                return (phrase.pitches[left] + (phrase.pitches[right] - phrase.pitches[left]) * alpha) * 0.01;
            }
            var note = phrase.notes.LastOrDefault(n => posMs >= n.positionMs && posMs < n.endMs)
                ?? NearestNoteAt(phrase.notes, posMs);
            return note?.adjustedTone ?? note?.tone ?? 0;
        }

        static RenderNote? NearestNoteAt(RenderNote[] notes, double posMs) {
            if (notes.Length == 0) {
                return null;
            }
            RenderNote? best = null;
            double bestDistance = double.PositiveInfinity;
            foreach (var note in notes) {
                double distance = posMs < note.positionMs
                    ? note.positionMs - posMs
                    : posMs > note.endMs
                        ? posMs - note.endMs
                        : 0;
                if (distance < bestDistance) {
                    best = note;
                    bestDistance = distance;
                }
            }
            return best;
        }
    }
}
