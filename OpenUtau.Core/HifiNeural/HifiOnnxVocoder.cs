using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using K4os.Hash.xxHash;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json.Linq;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiOnnxVocoder : IDisposable {
        public const int SampleRate = 44100;
        public const int HopSize = 512;
        public const int MelBins = 128;

        static readonly ConcurrentDictionary<string, Lazy<InferenceSession>> sessionCache = new();
        // Cache immutable session input metadata (dimensions) keyed by session cache key.
        // Avoids re-querying InputMetadata and re-allocating dimension arrays on every inference.
        static readonly ConcurrentDictionary<string, int[][]> sessionDimsCache = new();

        readonly string modelPath;
        readonly InferenceSession session;
        readonly int[] melDims;
        readonly int[] f0Dims;

        public string ModelPath => modelPath;

        public HifiOnnxVocoder(string? modelPath = null) {
            this.modelPath = ResolveModelPath(modelPath);
            session = GetCachedSession(this.modelPath);
            (melDims, f0Dims) = GetCachedDims(session, this.modelPath);
        }

        public float[] Infer(HifiPhraseFeatures features) {
            return Infer(features.Mel, features.F0);
        }

        public float[] Infer(float[,] mel, float[] f0) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            if (bins != MelBins) {
                throw new ArgumentException($"Expected {MelBins} mel bins, got {bins}.");
            }
            if (f0.Length != frames) {
                throw new ArgumentException($"F0 length {f0.Length} does not match mel frames {frames}.");
            }

            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("mel", CreateMelTensor(mel)),
                NamedOnnxValue.CreateFromTensor("f0", CreateF0Tensor(f0)),
            };
            Onnx.VerifyInputNames(session, inputs);
            using var outputs = session.Run(inputs);
            var tensor = outputs.First().AsTensor<float>();
            return tensor.ToArray();
        }

        public float[] InferDummy(int frames = 8) {
            var mel = new float[MelBins, frames];
            var f0 = Enumerable.Repeat(220f, frames).ToArray();
            for (int m = 0; m < MelBins; m++) {
                for (int t = 0; t < frames; t++) {
                    mel[m, t] = -5f;
                }
            }
            return Infer(mel, f0);
        }

        DenseTensor<float> CreateMelTensor(float[,] mel) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            var dims = melDims;

            bool channelLast = true;
            if (dims.Length == 3) {
                if (DimensionMatches(dims[1], bins) && !DimensionMatches(dims[2], bins)) {
                    channelLast = false;
                } else if (DimensionMatches(dims[2], bins)) {
                    channelLast = true;
                }
            }

            if (channelLast) {
                var data = new float[frames * bins];
                for (int t = 0; t < frames; t++) {
                    for (int m = 0; m < bins; m++) {
                        data[t * bins + m] = mel[m, t];
                    }
                }
                return new DenseTensor<float>(data, new[] { 1, frames, bins });
            } else {
                var data = new float[bins * frames];
                for (int m = 0; m < bins; m++) {
                    for (int t = 0; t < frames; t++) {
                        data[m * frames + t] = mel[m, t];
                    }
                }
                return new DenseTensor<float>(data, new[] { 1, bins, frames });
            }
        }

        DenseTensor<float> CreateF0Tensor(float[] f0) {
            var dims = f0Dims;
            if (dims.Length == 3) {
                if (DimensionMatches(dims[1], 1)) {
                    return new DenseTensor<float>(f0, new[] { 1, 1, f0.Length });
                }
                if (DimensionMatches(dims[2], 1)) {
                    return new DenseTensor<float>(f0, new[] { 1, f0.Length, 1 });
                }
            }
            return new DenseTensor<float>(f0, new[] { 1, f0.Length });
        }

        static bool DimensionMatches(int dim, int value) {
            return dim < 0 || dim == value;
        }

        public static bool TryResolveModelPath(out string modelPath, out string diagnostic) {
            try {
                modelPath = ResolveModelPath(null);
                diagnostic = string.Empty;
                return true;
            } catch (FileNotFoundException e) {
                modelPath = string.Empty;
                diagnostic = e.Message;
                return false;
            }
        }

        public static string ModelCacheKey(string modelPath) {
            var info = new FileInfo(modelPath);
            string key = $"{info.FullName}|{info.Length}|{info.LastWriteTimeUtc.Ticks}";
            return $"{XXH64.DigestOf(Encoding.UTF8.GetBytes(key)):x16}";
        }

        public static string ResolveModelPath(string? explicitPath) {
            if (!string.IsNullOrWhiteSpace(explicitPath)) {
                if (!File.Exists(explicitPath)) {
                    throw new FileNotFoundException($"Hifi ONNX vocoder model not found: {explicitPath}", explicitPath);
                }
                return explicitPath;
            }

            var candidates = new List<string>();
            string baseDir = AppContext.BaseDirectory;
            AddExplicitCandidate(candidates, Environment.GetEnvironmentVariable("HIFI_NEURAL_VOCODER_ONNX"));
            foreach (var root in CandidateRoots(baseDir).Concat(CandidateRoots(Directory.GetCurrentDirectory())).Distinct()) {
                AddCandidates(candidates, root);
                AddCandidates(candidates, Path.Combine(root, "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"));
            }

            var found = candidates.FirstOrDefault(File.Exists);
            if (found != null) {
                return found;
            }
            string searched = string.Join(Environment.NewLine, candidates.Distinct());
            throw new FileNotFoundException(
                "Hifi ONNX vocoder model was not found. Provide a PC-NSF-HiFiGAN ONNX export named model.onnx or *.onnx in the package directory. Searched:" +
                Environment.NewLine + searched);
        }

        static IEnumerable<string> CandidateRoots(string start) {
            string? current = Path.GetFullPath(start);
            while (!string.IsNullOrWhiteSpace(current)) {
                yield return current;
                current = Directory.GetParent(current)?.FullName;
            }
        }

        static void AddExplicitCandidate(List<string> candidates, string? path) {
            if (!string.IsNullOrWhiteSpace(path)) {
                candidates.Add(path);
            }
        }

        static void AddCandidates(List<string> candidates, string dir) {
            try {
                if (!Directory.Exists(dir)) {
                    return;
                }
                candidates.Add(Path.Combine(dir, "model.onnx"));
                candidates.AddRange(Directory.EnumerateFiles(dir, "*.onnx"));
                foreach (var sub in Directory.EnumerateDirectories(dir, "pc_nsf_hifigan*")) {
                    candidates.Add(Path.Combine(sub, "model.onnx"));
                    candidates.AddRange(Directory.EnumerateFiles(sub, "*.onnx"));
                    var config = Path.Combine(sub, "config.json");
                    if (File.Exists(config)) {
                        var json = JObject.Parse(File.ReadAllText(config));
                        var configured = (string?)json["model"] ?? (string?)json["onnx_model"];
                        if (!string.IsNullOrWhiteSpace(configured)) {
                            candidates.Add(Path.Combine(sub, configured));
                        }
                    }
                }
            } catch (Exception e) {
                Log.Warning(e, "Failed to search Hifi ONNX vocoder candidates in {Dir}", dir);
            }
        }

        public void Dispose() {
            // No-op: sessions are shared across phrases and live for the process lifetime.
        }

        static InferenceSession GetCachedSession(string modelPath) {
            string fullPath = Path.GetFullPath(modelPath);
            var sessionContext = ResolveSessionContext();
            string cacheKey = BuildSessionCacheKey(fullPath, sessionContext.Runner, sessionContext.Gpu);
            var lazy = sessionCache.GetOrAdd(cacheKey, _ => new Lazy<InferenceSession>(
                () => {
                    var created = Onnx.getInferenceSession(fullPath, OnnxRunnerChoice.Default);
                    Log.Information(
                        "Loaded Hifi ONNX vocoder session from {ModelPath} runner={Runner} gpu={Gpu} cache_key={CacheKey}",
                        fullPath,
                        sessionContext.Runner,
                        sessionContext.Gpu,
                        cacheKey);
                    return created;
                },
                LazyThreadSafetyMode.ExecutionAndPublication));
            return lazy.Value;
        }

        static (int[] MelDims, int[] F0Dims) GetCachedDims(InferenceSession session, string modelPath) {
            string fullPath = Path.GetFullPath(modelPath);
            var sessionContext = ResolveSessionContext();
            string cacheKey = BuildSessionCacheKey(fullPath, sessionContext.Runner, sessionContext.Gpu);
            var dims = sessionDimsCache.GetOrAdd(cacheKey, _ => {
                var melMeta = session.InputMetadata.TryGetValue("mel", out var melVal) ? melVal : null;
                var f0Meta = session.InputMetadata.TryGetValue("f0", out var f0Val) ? f0Val : null;
                return new[] {
                    melMeta?.Dimensions?.ToArray() ?? Array.Empty<int>(),
                    f0Meta?.Dimensions?.ToArray() ?? Array.Empty<int>(),
                };
            });
            return (dims[0], dims[1]);
        }

        static string BuildSessionCacheKey(string fullPath, string runner, int gpu) {
            return $"{fullPath}|runner={runner}|gpu={gpu}";
        }

        static (string Runner, int Gpu) ResolveSessionContext() {
            string runner = Preferences.Default.OnnxRunner;
            if (string.IsNullOrWhiteSpace(runner)) {
                var options = Onnx.getRunnerOptions();
                runner = options.Count > 0 ? options[0] : "CPU";
            }
            if (!Onnx.getRunnerOptions().Contains(runner)) {
                runner = "CPU";
            }
            int gpu = Math.Max(0, Preferences.Default.OnnxGpu);
            return (runner, gpu);
        }
    }
}
