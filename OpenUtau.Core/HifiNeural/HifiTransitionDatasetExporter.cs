using System;
using System.IO;
using Newtonsoft.Json;
using OpenUtau.Core.Util;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiTransitionDatasetExporter {
        public static string Export(string key, HifiPhraseFeatures features, int radiusFrames = 16) {
            var dir = Path.Combine(PathManager.Inst.CachePath, "hifi-transition-data", key);
            ExportToDirectory(dir, features, radiusFrames);
            return dir;
        }

        public static void ExportToDirectory(string dir, HifiPhraseFeatures features, int radiusFrames = 16) {
            Directory.CreateDirectory(dir);
            foreach (var boundary in features.Metadata.Boundaries) {
                int start = Math.Max(0, boundary.Frame - radiusFrames);
                int end = Math.Min(features.Frames, boundary.Frame + radiusFrames + 1);
                int frames = end - start;
                string prefix = $"transition-{boundary.Index:D4}";
                WriteMelWindow(Path.Combine(dir, prefix + ".mel.f32"), features.Mel, start, frames);
                WriteF0Window(Path.Combine(dir, prefix + ".f0.f32"), features.F0, start, frames);
                WriteMask(Path.Combine(dir, prefix + ".mask.f32"), start, frames, boundary.Frame);
                File.WriteAllText(Path.Combine(dir, prefix + ".json"),
                    JsonConvert.SerializeObject(new {
                        boundary.Index,
                        boundary.LeftPhoneIndex,
                        boundary.RightPhoneIndex,
                        boundary.LeftPhone,
                        boundary.RightPhone,
                        boundary.PositionMs,
                        boundary.Frame,
                        boundary.TransitionType,
                        WindowStartFrame = start,
                        WindowFrames = frames,
                        MelShape = new[] { HifiMelExtractor.NMels, frames },
                        F0Shape = new[] { frames },
                    }, Formatting.Indented));
            }
        }

        static void WriteMelWindow(string path, float[,] mel, int start, int frames) {
            using var writer = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write, FileShare.Read));
            for (int t = start; t < start + frames; t++) {
                for (int m = 0; m < mel.GetLength(0); m++) {
                    writer.Write(mel[m, t]);
                }
            }
        }

        static void WriteF0Window(string path, float[] f0, int start, int frames) {
            using var writer = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write, FileShare.Read));
            for (int t = start; t < start + frames; t++) {
                writer.Write(f0[t]);
            }
        }

        static void WriteMask(string path, int start, int frames, int boundaryFrame) {
            using var writer = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write, FileShare.Read));
            for (int t = start; t < start + frames; t++) {
                writer.Write(t == boundaryFrame ? 1f : 0f);
            }
        }
    }
}
