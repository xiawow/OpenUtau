using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using OpenUtau.Core.Util;

namespace OpenUtau.Core.HifiNeural {
    public static class HifiDebugExporter {
        public static string Export(string key, HifiPhraseFeatures features) {
            var dir = Path.Combine(PathManager.Inst.CachePath, "hifi-debug", key);
            ExportToDirectory(dir, features);
            return dir;
        }

        public static void ExportToDirectory(string dir, HifiPhraseFeatures features) {
            Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir, "metadata.json"),
                JsonConvert.SerializeObject(features.Metadata, Formatting.Indented));
            WriteRaw(Path.Combine(dir, "phrase_mel.f32"), features.Mel);
            WriteRaw(Path.Combine(dir, "phrase_f0.f32"), features.F0);
            File.WriteAllText(Path.Combine(dir, "shape.txt"),
                $"phrase_mel=float32[{features.MelBins},{features.Frames}]{Environment.NewLine}phrase_f0=float32[{features.F0.Length}]{Environment.NewLine}");
        }

        public static HifiPhraseFeatures Load(string dir) {
            string metadataPath = Path.Combine(dir, "metadata.json");
            string shapePath = Path.Combine(dir, "shape.txt");
            string melPath = Path.Combine(dir, "phrase_mel.f32");
            string f0Path = Path.Combine(dir, "phrase_f0.f32");
            if (!File.Exists(metadataPath) || !File.Exists(shapePath) || !File.Exists(melPath) || !File.Exists(f0Path)) {
                throw new FileNotFoundException($"Invalid Hifi debug dump. Expected metadata.json, shape.txt, phrase_mel.f32, and phrase_f0.f32 in {dir}");
            }

            var metadata = JsonConvert.DeserializeObject<HifiPhraseMetadata>(File.ReadAllText(metadataPath))
                ?? throw new InvalidDataException($"Failed to parse {metadataPath}");
            var shapeText = File.ReadAllText(shapePath);
            var (melBins, frames) = ParseMelShape(shapeText);
            int f0Frames = ParseF0Shape(shapeText);
            if (melBins != HifiMelExtractor.NMels) {
                throw new InvalidDataException($"Expected {HifiMelExtractor.NMels} mel bins, got {melBins}.");
            }
            if (f0Frames != frames) {
                throw new InvalidDataException($"Mel frames {frames} do not match f0 frames {f0Frames}.");
            }

            var mel = ReadMelRaw(melPath, melBins, frames);
            var f0 = ReadVectorRaw(f0Path, f0Frames);
            Validate(mel, f0);
            return new HifiPhraseFeatures {
                Mel = mel,
                F0 = f0,
                Metadata = metadata,
            };
        }

        public static float[] RevocodeDump(string dir, string? modelPath = null) {
            using var vocoder = new HifiOnnxVocoder(modelPath);
            return vocoder.Infer(Load(dir));
        }

        static void WriteRaw(string path, float[,] values) {
            using var writer = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write, FileShare.Read));
            for (int t = 0; t < values.GetLength(1); t++) {
                for (int m = 0; m < values.GetLength(0); m++) {
                    writer.Write(values[m, t]);
                }
            }
        }

        static void WriteRaw(string path, float[] values) {
            using var writer = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write, FileShare.Read));
            foreach (float value in values) {
                writer.Write(value);
            }
        }

        static (int melBins, int frames) ParseMelShape(string shapeText) {
            var match = Regex.Match(shapeText, @"phrase_mel\s*=\s*float32\[(\d+),(\d+)\]");
            if (!match.Success) {
                throw new InvalidDataException("shape.txt is missing phrase_mel=float32[mel_bins,frames].");
            }
            return (
                int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture),
                int.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture));
        }

        static int ParseF0Shape(string shapeText) {
            var match = Regex.Match(shapeText, @"phrase_f0\s*=\s*float32\[(\d+)\]");
            if (!match.Success) {
                throw new InvalidDataException("shape.txt is missing phrase_f0=float32[frames].");
            }
            return int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
        }

        static float[,] ReadMelRaw(string path, int melBins, int frames) {
            var values = ReadVectorRaw(path, melBins * frames);
            var mel = new float[melBins, frames];
            int index = 0;
            for (int t = 0; t < frames; t++) {
                for (int m = 0; m < melBins; m++) {
                    mel[m, t] = values[index++];
                }
            }
            return mel;
        }

        static float[] ReadVectorRaw(string path, int count) {
            var bytes = File.ReadAllBytes(path);
            int expectedBytes = count * sizeof(float);
            if (bytes.Length != expectedBytes) {
                throw new InvalidDataException($"{Path.GetFileName(path)} has {bytes.Length} bytes, expected {expectedBytes}.");
            }
            var values = new float[count];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        static void Validate(float[,] mel, float[] f0) {
            if (mel.Cast<float>().Any(value => float.IsNaN(value) || float.IsInfinity(value))) {
                throw new InvalidDataException("Loaded phrase_mel contains NaN or Inf.");
            }
            if (f0.Any(value => float.IsNaN(value) || float.IsInfinity(value))) {
                throw new InvalidDataException("Loaded phrase_f0 contains NaN or Inf.");
            }
        }
    }
}
