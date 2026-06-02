using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using K4os.Hash.xxHash;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiHnsepResult {
        public required float[] Harmonic { get; init; }
    }

    public sealed class HifiHnsepOnnx {
        const int DefaultNfft = 2048;
        const int DefaultHop = 512;
        static readonly ConcurrentDictionary<string, Lazy<InferenceSession>> sessionCache = new();

        readonly string modelPath;
        readonly InferenceSession session;
        readonly int nfft;
        readonly int hop;
        readonly float[] window;

        HifiHnsepOnnx(string modelPath, int nfft, int hop) {
            this.modelPath = modelPath;
            this.nfft = nfft;
            this.hop = hop;
            window = BuildHannWindow(nfft);
            session = GetCachedSession(modelPath);
        }

        public static bool TryCreate(out HifiHnsepOnnx? model, out string diagnostic) {
            if (!TryResolveModelPath(out var path, out diagnostic)) {
                model = null;
                return false;
            }
            var (nfft, hop) = ResolveModelConfig(path);
            try {
                model = new HifiHnsepOnnx(path, nfft, hop);
                diagnostic = string.Empty;
                return true;
            } catch (Exception e) {
                model = null;
                diagnostic = e.Message;
                Log.Warning(e, "Failed to initialize Hifi HNSEP ONNX model path={Path}", path);
                return false;
            }
        }

        public static string CacheKeyOrDisabled() {
            return TryResolveModelPath(out var path, out _)
                ? ModelCacheKey(path)
                : "hnsep-disabled";
        }

        public static string ModelCacheKey(string path) {
            var info = new FileInfo(path);
            string key = $"{info.FullName}|{info.Length}|{info.LastWriteTimeUtc.Ticks}";
            return $"{XXH64.DigestOf(Encoding.UTF8.GetBytes(key)):x16}";
        }

        public HifiHnsepResult Separate(float[] samples) {
            if (samples.Length == 0) {
                return new HifiHnsepResult { Harmonic = Array.Empty<float>() };
            }

            int originalLength = samples.Length;
            int t1 = originalLength + hop;
            int segmentLength = 32 * hop;
            int padTotal = segmentLength * ((t1 - 1) / segmentLength + 1) - t1;
            int leftPad = (padTotal / 2 / hop) * hop;
            int rightPad = padTotal - leftPad;
            var padded = new float[leftPad + originalLength + rightPad];
            Array.Copy(samples, 0, padded, leftPad, originalLength);

            var spec = Stft(padded);
            int bins = spec.GetLength(0);
            int frames = spec.GetLength(1);
            var inputData = new float[2 * bins * frames];
            for (int f = 0; f < bins; f++) {
                for (int t = 0; t < frames; t++) {
                    int index = f * frames + t;
                    inputData[index] = (float)spec[f, t].Real;
                    inputData[bins * frames + index] = (float)spec[f, t].Imaginary;
                }
            }

            string inputName = session.InputMetadata.Keys.First();
            using var outputs = session.Run(new[] {
                NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<float>(inputData, new[] { 1, 2, bins, frames })),
            });
            var output = outputs.First().AsTensor<float>().ToArray();

            var predicted = new Complex[bins, frames];
            DecodeMaskOutput(output, spec, predicted, bins, frames);

            var separatedPadded = Istft(predicted, padded.Length);
            var harmonic = new float[originalLength];
            Array.Copy(separatedPadded, leftPad, harmonic, 0, originalLength);
            return new HifiHnsepResult { Harmonic = harmonic };
        }

        static void DecodeMaskOutput(float[] output, Complex[,] spec, Complex[,] predicted, int bins, int frames) {
            int complexSize = 2 * bins * frames;
            int realSize = bins * frames;
            if (output.Length == complexSize) {
                for (int f = 0; f < bins; f++) {
                    for (int t = 0; t < frames; t++) {
                        int index = f * frames + t;
                        var mask = new Complex(output[index], output[realSize + index]);
                        predicted[f, t] = spec[f, t] * mask;
                    }
                }
                return;
            }
            if (output.Length == realSize) {
                for (int f = 0; f < bins; f++) {
                    for (int t = 0; t < frames; t++) {
                        int index = f * frames + t;
                        predicted[f, t] = spec[f, t] * output[index];
                    }
                }
                return;
            }
            throw new InvalidDataException($"HNSEP output size {output.Length} does not match complex mask size {complexSize} or real mask size {realSize}.");
        }

        static bool TryResolveModelPath(out string path, out string diagnostic) {
            var candidates = new List<string>();
            AddExplicitCandidate(candidates, Environment.GetEnvironmentVariable("HIFI_NEURAL_HNSEP_ONNX"));
            foreach (var root in CandidateRoots(AppContext.BaseDirectory).Concat(CandidateRoots(Directory.GetCurrentDirectory())).Distinct()) {
                AddCandidates(candidates, root);
            }
            path = candidates.FirstOrDefault(File.Exists) ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(path)) {
                diagnostic = string.Empty;
                return true;
            }
            diagnostic = "Hifi HNSEP ONNX model was not found. Set HIFI_NEURAL_HNSEP_ONNX or place hnsep.onnx next to the HIFI-NEURA model.";
            return false;
        }

        static void AddExplicitCandidate(List<string> candidates, string? path) {
            if (!string.IsNullOrWhiteSpace(path)) {
                candidates.Add(path);
            }
        }

        static IEnumerable<string> CandidateRoots(string start) {
            string? current = Path.GetFullPath(start);
            while (!string.IsNullOrWhiteSpace(current)) {
                yield return current;
                current = Directory.GetParent(current)?.FullName;
            }
        }

        static void AddCandidates(List<string> candidates, string dir) {
            try {
                if (!Directory.Exists(dir)) {
                    return;
                }
                candidates.Add(Path.Combine(dir, "hnsep.onnx"));
                candidates.Add(Path.Combine(dir, "hnsep_model.onnx"));
                foreach (var sub in Directory.EnumerateDirectories(dir, "*hnsep*")) {
                    candidates.Add(Path.Combine(sub, "model.onnx"));
                    candidates.Add(Path.Combine(sub, "hnsep.onnx"));
                    candidates.AddRange(Directory.EnumerateFiles(sub, "*.onnx"));
                    foreach (var child in Directory.EnumerateDirectories(sub)) {
                        candidates.Add(Path.Combine(child, "model.onnx"));
                        candidates.Add(Path.Combine(child, "hnsep.onnx"));
                        candidates.AddRange(Directory.EnumerateFiles(child, "*.onnx"));
                    }
                }
            } catch (Exception e) {
                Log.Warning(e, "Failed to search Hifi HNSEP candidates in {Dir}", dir);
            }
        }

        static (int Nfft, int Hop) ResolveModelConfig(string path) {
            string configPath = Path.Combine(Path.GetDirectoryName(path) ?? string.Empty, "config.yaml");
            if (!File.Exists(configPath)) {
                return (DefaultNfft, DefaultHop);
            }
            int nfft = DefaultNfft;
            int hop = DefaultHop;
            foreach (var line in File.ReadLines(configPath)) {
                var parts = line.Split(':', 2);
                if (parts.Length != 2) {
                    continue;
                }
                string key = parts[0].Trim();
                string value = parts[1].Split('#', 2)[0].Trim();
                if (key == "n_fft" && int.TryParse(value, out var parsedNfft)) {
                    nfft = parsedNfft;
                } else if ((key == "hop_length" || key == "hop_size") && int.TryParse(value, out var parsedHop)) {
                    hop = parsedHop;
                }
            }
            if (nfft <= 0 || !IsPowerOfTwo(nfft)) {
                Log.Warning(
                    "HNSEP config.yaml n_fft={Nfft} is invalid; falling back to default n_fft={DefaultNfft}.",
                    nfft,
                    DefaultNfft);
                nfft = DefaultNfft;
            }
            if (hop <= 0) {
                Log.Warning(
                    "HNSEP config.yaml hop_length={Hop} is invalid; falling back to default hop_length={DefaultHop}.",
                    hop,
                    DefaultHop);
                hop = DefaultHop;
            }
            return (Math.Max(256, nfft), Math.Max(64, hop));
        }

        Complex[,] Stft(float[] samples) {
            int centerPad = nfft / 2;
            var padded = new float[samples.Length + centerPad * 2];
            Array.Copy(samples, 0, padded, centerPad, samples.Length);
            int frames = Math.Max(1, 1 + Math.Max(0, padded.Length - nfft) / hop);
            int bins = nfft / 2 + 1;
            var spec = new Complex[bins, frames];
            var fft = new Complex[nfft];
            for (int frame = 0; frame < frames; frame++) {
                Array.Clear(fft, 0, fft.Length);
                int start = frame * hop;
                for (int i = 0; i < nfft; i++) {
                    fft[i] = new Complex(padded[start + i] * window[i], 0);
                }
                ForwardFft(fft, inverse: false);
                for (int f = 0; f < bins; f++) {
                    spec[f, frame] = fft[f];
                }
            }
            return spec;
        }

        float[] Istft(Complex[,] spec, int outputLength) {
            int bins = spec.GetLength(0);
            int frames = spec.GetLength(1);
            int centerPad = nfft / 2;
            var padded = new double[outputLength + centerPad * 2];
            var windowSum = new double[padded.Length];
            var fft = new Complex[nfft];
            for (int frame = 0; frame < frames; frame++) {
                Array.Clear(fft, 0, fft.Length);
                for (int f = 0; f < bins; f++) {
                    fft[f] = spec[f, frame];
                }
                ConstrainRealSignalSpectrum(fft, bins);
                for (int f = 1; f < bins - 1; f++) {
                    fft[nfft - f] = Complex.Conjugate(fft[f]);
                }
                ForwardFft(fft, inverse: true);
                int start = frame * hop;
                for (int i = 0; i < nfft && start + i < padded.Length; i++) {
                    double w = window[i];
                    padded[start + i] += fft[i].Real * w;
                    windowSum[start + i] += w * w;
                }
            }

            var output = new float[outputLength];
            for (int i = 0; i < output.Length; i++) {
                int src = i + centerPad;
                double value = padded[src];
                if (windowSum[src] > 1e-9) {
                    value /= windowSum[src];
                }
                output[i] = (float)Math.Clamp(value, -1.0, 1.0);
            }
            return output;
        }

        static InferenceSession GetCachedSession(string modelPath) {
            string fullPath = Path.GetFullPath(modelPath);
            string key = $"{fullPath}|runner=CPU";
            var lazy = sessionCache.GetOrAdd(key, _ => new Lazy<InferenceSession>(() => {
                // DirectML currently creates this HNSEP graph but fails at runtime on Resize nodes.
                // Keep HNSEP on CPU; the vocoder still uses the user's ONNX runner choice.
                var session = Onnx.getInferenceSession(fullPath, OnnxRunnerChoice.CPU);
                Log.Information("Loaded Hifi HNSEP ONNX model path={Path} runner=CPU", fullPath);
                return session;
            }));
            return lazy.Value;
        }

        static float[] BuildHannWindow(int size) {
            var result = new float[size];
            for (int i = 0; i < size; i++) {
                result[i] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / size));
            }
            return result;
        }

        static void ForwardFft(Complex[] buffer, bool inverse) {
            int n = buffer.Length;
            if (!IsPowerOfTwo(n)) {
                throw new NotSupportedException($"HNSEP STFT n_fft must be a power of two, got {n}.");
            }
            for (int i = 1, j = 0; i < n; i++) {
                int bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1) {
                    j ^= bit;
                }
                j ^= bit;
                if (i < j) {
                    (buffer[i], buffer[j]) = (buffer[j], buffer[i]);
                }
            }
            for (int len = 2; len <= n; len <<= 1) {
                double angle = 2.0 * Math.PI / len * (inverse ? 1 : -1);
                var wlen = new Complex(Math.Cos(angle), Math.Sin(angle));
                for (int i = 0; i < n; i += len) {
                    var w = Complex.One;
                    for (int j = 0; j < len / 2; j++) {
                        var u = buffer[i + j];
                        var v = buffer[i + j + len / 2] * w;
                        buffer[i + j] = u + v;
                        buffer[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
            }
            if (inverse) {
                for (int i = 0; i < n; i++) {
                    buffer[i] /= n;
                }
            }
        }

        static bool IsPowerOfTwo(int value) {
            return value > 0 && (value & (value - 1)) == 0;
        }

        static void ConstrainRealSignalSpectrum(Complex[] fft, int bins) {
            if (bins <= 0) {
                return;
            }
            fft[0] = new Complex(fft[0].Real, 0);
            if (bins > 1) {
                int nyquist = bins - 1;
                fft[nyquist] = new Complex(fft[nyquist].Real, 0);
            }
        }
    }
}
