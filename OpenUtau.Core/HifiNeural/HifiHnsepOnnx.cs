using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
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
        public const string RunnerCpu = "CPU";
        public const string RunnerDirectML = "DirectML";

        const int DefaultNfft = 2048;
        const int DefaultHop = 512;
        const int ParallelFrameThreshold = 16;
        static readonly ConcurrentDictionary<string, Lazy<InferenceSession>> sessionCache = new();
        static readonly ConcurrentDictionary<int, SemaphoreSlim> separationGates = new();

        readonly string modelPath;
        readonly InferenceSession session;
        readonly int nfft;
        readonly int hop;
        readonly int workerThreads;
        readonly int maxConcurrentSeparations;
        readonly float[] window;

        HifiHnsepOnnx(string modelPath, int nfft, int hop) {
            this.modelPath = modelPath;
            this.nfft = nfft;
            this.hop = hop;
            workerThreads = ResolveCpuThreadCount();
            maxConcurrentSeparations = ResolveMaxConcurrentSeparations(workerThreads);
            window = BuildHannWindow(nfft);
            session = GetCachedSession(modelPath, workerThreads, maxConcurrentSeparations);
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
            string key = $"{info.FullName}|{info.Length}|{info.LastWriteTimeUtc.Ticks}|runner={ResolveHnsepRunner(Path.GetFullPath(path))}";
            return $"{XXH64.DigestOf(Encoding.UTF8.GetBytes(key)):x16}";
        }

        public static List<string> RunnerOptions() {
            var options = new List<string> { RunnerCpu };
            if (Onnx.getRunnerOptions().Contains(RunnerDirectML)) {
                options.Add(RunnerDirectML);
            }
            return options;
        }

        public static string NormalizeRunner(string? runner) {
            if (string.Equals(runner, RunnerDirectML, StringComparison.OrdinalIgnoreCase)
                    && Onnx.getRunnerOptions().Contains(RunnerDirectML)) {
                return RunnerDirectML;
            }
            return RunnerCpu;
        }

        public HifiHnsepResult Separate(float[] samples) {
            if (samples.Length == 0) {
                return new HifiHnsepResult { Harmonic = Array.Empty<float>() };
            }

            var gate = separationGates.GetOrAdd(maxConcurrentSeparations, count => new SemaphoreSlim(count, count));
            gate.Wait();
            try {
                return SeparateCore(samples);
            } finally {
                gate.Release();
            }
        }

        HifiHnsepResult SeparateCore(float[] samples) {
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
            WriteInputTensorData(spec, inputData, bins, frames);

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

        void WriteInputTensorData(Complex[,] spec, float[] inputData, int bins, int frames) {
            if (bins < ParallelFrameThreshold || workerThreads <= 1) {
                for (int f = 0; f < bins; f++) {
                    WriteInputTensorBand(spec, inputData, bins, frames, f);
                }
                return;
            }
            Parallel.For(0, bins, HnsepParallelOptions(), f => {
                WriteInputTensorBand(spec, inputData, bins, frames, f);
            });
        }

        static void WriteInputTensorBand(Complex[,] spec, float[] inputData, int bins, int frames, int f) {
            for (int t = 0; t < frames; t++) {
                int index = f * frames + t;
                inputData[index] = (float)spec[f, t].Real;
                inputData[bins * frames + index] = (float)spec[f, t].Imaginary;
            }
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
            bool preferDmlSafe = IsAcceleratedRunnerRequested();
            foreach (var root in CandidateRoots(AppContext.BaseDirectory).Concat(CandidateRoots(Directory.GetCurrentDirectory())).Distinct()) {
                AddCandidates(candidates, root, preferDmlSafe);
            }
            path = candidates.FirstOrDefault(File.Exists) ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(path)) {
                diagnostic = string.Empty;
                return true;
            }
            diagnostic = "Hifi HNSEP ONNX model was not found. Set HIFI_NEURAL_HNSEP_ONNX or place model.onnx/model_dml.onnx/hnsep.onnx next to the HIFI-NEURA model.";
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
                // Never search a volume root. Model discovery is best-effort and must not
                // enumerate unrelated or protected system directories.
                string? parent = Directory.GetParent(current)?.FullName;
                if (string.IsNullOrWhiteSpace(parent)) {
                    yield break;
                }
                yield return current;
                current = parent;
            }
        }

        static void AddCandidates(List<string> candidates, string dir, bool preferDmlSafe) {
            try {
                if (!Directory.Exists(dir)) {
                    return;
                }
                AddModelCandidates(candidates, dir, preferDmlSafe);
                AddModelCandidates(candidates, Path.Combine(dir, "hnsep"), preferDmlSafe);
                AddModelCandidates(candidates, Path.Combine(dir, "hnsep", "vr"), preferDmlSafe);
                AddModelCandidates(candidates, Path.Combine(dir, "models", "hnsep"), preferDmlSafe);
                AddModelCandidates(candidates, Path.Combine(dir, "models", "hnsep", "vr"), preferDmlSafe);
            } catch (Exception e) {
                Log.Warning(e, "Failed to search Hifi HNSEP candidates in {Dir}", dir);
            }
        }

        static void AddModelCandidates(List<string> candidates, string dir, bool preferDmlSafe) {
            if (preferDmlSafe) {
                candidates.Add(Path.Combine(dir, "model_dml.onnx"));
            }
            candidates.Add(Path.Combine(dir, "model.onnx"));
            candidates.Add(Path.Combine(dir, "hnsep.onnx"));
            candidates.Add(Path.Combine(dir, "hnsep_model.onnx"));
            if (!preferDmlSafe) {
                candidates.Add(Path.Combine(dir, "model_dml.onnx"));
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
            if (frames < ParallelFrameThreshold || workerThreads <= 1) {
                var fft = new Complex[nfft];
                for (int frame = 0; frame < frames; frame++) {
                    WriteStftFrame(padded, spec, fft, frame, bins);
                }
                return spec;
            }

            Parallel.For(
                0,
                frames,
                HnsepParallelOptions(),
                () => new Complex[nfft],
                (frame, _, fft) => {
                    WriteStftFrame(padded, spec, fft, frame, bins);
                    return fft;
                },
                _ => { });
            return spec;
        }

        void WriteStftFrame(float[] padded, Complex[,] spec, Complex[] fft, int frame, int bins) {
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

        ParallelOptions HnsepParallelOptions() {
            return new ParallelOptions {
                MaxDegreeOfParallelism = Math.Max(1, workerThreads),
            };
        }

        static InferenceSession GetCachedSession(string modelPath, int workerThreads, int maxConcurrentSeparations) {
            string fullPath = Path.GetFullPath(modelPath);
            string runner = ResolveHnsepRunner(fullPath);
            string key = $"{fullPath}|runner={runner}|gpu={Math.Max(0, Preferences.Default.OnnxGpu)}|threads={workerThreads}";
            var lazy = sessionCache.GetOrAdd(key, _ => new Lazy<InferenceSession>(() => {
                if (runner != "CPU") {
                    try {
                        var accelerated = CreateAcceleratedSession(fullPath, runner);
                        Log.Information(
                            "Loaded Hifi HNSEP ONNX model path={Path} runner={Runner} gpu={Gpu} max_concurrent={MaxConcurrent}",
                            fullPath,
                            runner,
                            Math.Max(0, Preferences.Default.OnnxGpu),
                            maxConcurrentSeparations);
                        return accelerated;
                    } catch (Exception e) {
                        Log.Warning(e, "Failed to load Hifi HNSEP ONNX model path={Path} runner={Runner}; falling back to CPU", fullPath, runner);
                    }
                }

                // The original HNSEP ONNX graph contains Resize nodes that fail on DirectML.
                // Keep that model on CPU. DML-safe exports are named model_dml.onnx.
                var options = new SessionOptions {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    IntraOpNumThreads = workerThreads,
                    InterOpNumThreads = 1,
                };
                var session = new InferenceSession(fullPath, options);
                Log.Information(
                    "Loaded Hifi HNSEP ONNX model path={Path} runner=CPU intra_threads={Threads} max_concurrent={MaxConcurrent}",
                    fullPath,
                    workerThreads,
                    maxConcurrentSeparations);
                return session;
            }));
            return lazy.Value;
        }

        static string ResolveHnsepRunner(string fullPath) {
            if (!IsDmlSafeModelPath(fullPath)) {
                return RunnerCpu;
            }
            string runner = Environment.GetEnvironmentVariable("HIFI_NEURAL_HNSEP_RUNNER");
            if (string.IsNullOrWhiteSpace(runner)) {
                runner = Preferences.Default.HifiNeuralHnsepRunner;
            }
            return NormalizeRunner(runner);
        }

        static bool IsAcceleratedRunnerRequested() {
            return NormalizeRunner(Environment.GetEnvironmentVariable("HIFI_NEURAL_HNSEP_RUNNER")
                ?? Preferences.Default.HifiNeuralHnsepRunner) != RunnerCpu;
        }

        static InferenceSession CreateAcceleratedSession(string fullPath, string runner) {
            if (!string.Equals(runner, RunnerDirectML, StringComparison.OrdinalIgnoreCase)) {
                throw new NotSupportedException($"Unsupported HNSEP runner: {runner}");
            }
            var options = new SessionOptions {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            };
            options.AppendExecutionProvider_DML(Math.Max(0, Preferences.Default.OnnxGpu));
            return new InferenceSession(fullPath, options);
        }

        static bool IsDmlSafeModelPath(string fullPath) {
            string fileName = Path.GetFileNameWithoutExtension(fullPath);
            return fileName.Contains("dml", StringComparison.OrdinalIgnoreCase);
        }

        internal static int ResolveCpuThreadCount() {
            if (TryReadPositiveEnvironmentInt("HIFI_NEURAL_HNSEP_THREADS", out int explicitThreads)) {
                return Math.Clamp(explicitThreads, 1, Math.Max(1, Environment.ProcessorCount));
            }
            int renderThreads = Math.Max(1, Preferences.Default.NumRenderThreads);
            int cpuThreadCap = Math.Max(1, Environment.ProcessorCount / 2);
            return Math.Clamp(renderThreads, 1, cpuThreadCap);
        }

        internal static int ResolveMaxConcurrentSeparations(int workerThreads) {
            if (TryReadPositiveEnvironmentInt("HIFI_NEURAL_HNSEP_CONCURRENCY", out int explicitConcurrency)) {
                return Math.Clamp(explicitConcurrency, 1, Math.Max(1, Preferences.Default.NumRenderThreads));
            }
            workerThreads = Math.Max(1, workerThreads);
            int renderThreads = Math.Max(1, Preferences.Default.NumRenderThreads);
            int cpuBound = Math.Max(1, Environment.ProcessorCount / workerThreads);
            return Math.Clamp(Math.Min(renderThreads, cpuBound), 1, renderThreads);
        }

        static bool TryReadPositiveEnvironmentInt(string name, out int value) {
            value = 0;
            string? raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out value) && value > 0;
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
