using System;
using System.Collections.Generic;
using System.Linq;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiInternalClickSuppressor {
        public static void Apply(float[] samples, HifiPhraseMetadata metadata, int sampleRate, double windowMs, double thresholdRatio) {
            if (samples.Length < 8 || metadata.Notes.Count <= 1 || windowMs <= 0) {
                return;
            }

            int halfWindow = Math.Clamp((int)Math.Round(sampleRate * windowMs / 1000.0), 16, 160);
            thresholdRatio = Math.Clamp(thresholdRatio, 0.1, 1.0);
            int applied = 0;
            foreach (var note in metadata.Notes.Skip(1)) {
                int center = (int)Math.Round((note.PositionMs - metadata.PhraseStartMs) * sampleRate / 1000.0);
                if (center - halfWindow < 1 || center + halfWindow >= samples.Length - 1) {
                    continue;
                }

                double jump = Math.Abs(samples[center] - samples[center - 1]);
                double localRms = LocalRms(samples, center - halfWindow, center + halfWindow);
                if (localRms < 1e-5 || jump < Math.Max(0.015, localRms * thresholdRatio)) {
                    continue;
                }

                SmoothWindow(samples, center, halfWindow);
                applied++;
                Log.Information(
                    "HifiInternalClickSuppressor note_index={NoteIndex} lyric={Lyric} center_sample={CenterSample} half_window={HalfWindow} jump={Jump:F6} local_rms={LocalRms:F6}",
                    note.Index,
                    note.Lyric,
                    center,
                    halfWindow,
                    jump,
                    localRms);
            }

            if (applied > 0) {
                Log.Information(
                    "HifiInternalClickSuppressor summary applied={Applied} note_boundaries={NoteBoundaries} window_ms={WindowMs:F3} threshold_ratio={ThresholdRatio:F3}",
                    applied,
                    Math.Max(0, metadata.Notes.Count - 1),
                    windowMs,
                    thresholdRatio);
            }
        }

        static double LocalRms(float[] samples, int start, int end) {
            start = Math.Clamp(start, 0, samples.Length);
            end = Math.Clamp(end, start, samples.Length);
            double sum = 0;
            int count = 0;
            for (int i = start; i < end; i++) {
                float sample = samples[i];
                if (float.IsNaN(sample) || float.IsInfinity(sample)) {
                    continue;
                }
                sum += sample * sample;
                count++;
            }
            return count > 0 ? Math.Sqrt(sum / count) : 0;
        }

        static void SmoothWindow(float[] samples, int center, int halfWindow) {
            var original = new float[halfWindow * 2 + 1];
            int start = center - halfWindow;
            for (int i = 0; i < original.Length; i++) {
                original[i] = samples[start + i];
            }

            for (int i = 1; i < original.Length - 1; i++) {
                int sampleIndex = start + i;
                double distance = Math.Abs(sampleIndex - center) / (double)(halfWindow + 1);
                float strength = (float)(0.5 + 0.5 * Math.Cos(Math.PI * Math.Clamp(distance, 0, 1)));
                strength *= 0.5f;
                float smoothed = (original[i - 1] + original[i] * 2f + original[i + 1]) * 0.25f;
                samples[sampleIndex] = original[i] * (1f - strength) + smoothed * strength;
            }
        }
    }
}
