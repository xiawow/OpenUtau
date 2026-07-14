using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Serilog;

namespace OpenUtau.Core.Util {
    public static class OnnxRuntimeLibraryLoader {
        public const string RunnerCuda = "CUDA";

        static bool resolverConfigured;

        public static bool IsCudaRuntimeConfigured => resolverConfigured;

        public static void ConfigureCudaRuntimeIfRequested() {
            if (!OS.IsWindows() || !IsCudaRunnerRequested()) {
                return;
            }
            if (!TryResolveCudaRuntimeDirectory(out var runtimeDir)) {
                return;
            }
            ConfigureCudaRuntime(runtimeDir);
        }

        public static bool IsCudaRuntimeAvailable() {
            return OS.IsWindows() && TryResolveCudaRuntimeDirectory(out _);
        }

        static bool IsCudaRunnerRequested() {
            try {
                var prefsPath = PathManager.Inst.PrefsFilePath;
                if (!File.Exists(prefsPath)) {
                    return false;
                }
                using var document = JsonDocument.Parse(File.ReadAllText(prefsPath));
                if (IsJsonStringEqual(document.RootElement, "OnnxRunner", RunnerCuda)
                    || IsJsonStringEqual(document.RootElement, "HifiNeuralHnsepRunner", RunnerCuda)) {
                    return true;
                }
            } catch {
                // Logging is not initialized yet when this runs from Program.Main.
            }
            return false;
        }

        static string GetRawPreferenceString(string propertyName) {
            try {
                var prefsPath = PathManager.Inst.PrefsFilePath;
                if (!File.Exists(prefsPath)) {
                    return string.Empty;
                }
                using var document = JsonDocument.Parse(File.ReadAllText(prefsPath));
                return document.RootElement.TryGetProperty(propertyName, out var property)
                    && property.ValueKind == JsonValueKind.String
                    ? property.GetString() ?? string.Empty
                    : string.Empty;
            } catch {
                return string.Empty;
            }
        }

        static bool IsJsonStringEqual(JsonElement root, string propertyName, string value) {
            return root.TryGetProperty(propertyName, out var property)
                && property.ValueKind == JsonValueKind.String
                && string.Equals(property.GetString(), value, StringComparison.OrdinalIgnoreCase);
        }

        static void ConfigureCudaRuntime(string runtimeDir) {
            if (resolverConfigured) {
                return;
            }
            string onnxRuntimePath = Path.Combine(runtimeDir, "onnxruntime.dll");
            try {
                ConfigureCudaDependencyPaths(runtimeDir);
                NativeLibrary.SetDllImportResolver(typeof(OrtEnv).Assembly, (libraryName, assembly, searchPath) => {
                    if (string.Equals(libraryName, "onnxruntime", StringComparison.OrdinalIgnoreCase)
                        || string.Equals(libraryName, "onnxruntime.dll", StringComparison.OrdinalIgnoreCase)) {
                        return NativeLibrary.Load(onnxRuntimePath, assembly, searchPath);
                    }
                    return IntPtr.Zero;
                });
                resolverConfigured = true;
                Log.Information("Using CUDA ONNX Runtime from {RuntimeDir}", runtimeDir);
            } catch (InvalidOperationException) {
                resolverConfigured = true;
                ConfigureCudaDependencyPaths(runtimeDir);
            } catch (Exception e) {
                Log.Warning(e, "Failed to configure CUDA ONNX Runtime path {RuntimeDir}", runtimeDir);
            }
        }

        static void ConfigureCudaDependencyPaths(string runtimeDir) {
            var paths = new List<string> { runtimeDir };
            paths.AddRange(ResolveExistingDependencyPaths(GetRawPreferenceString("CudaPath"), CudaRuntimeProbeFile));
            paths.AddRange(ResolveExistingDependencyPaths(GetRawPreferenceString("CudnnPath"), CudnnRuntimeProbeFile));
            paths.AddRange(ResolveConfiguredDependencyPaths("CUDA_PATH", "bin"));
            paths.AddRange(ResolveConfiguredDependencyPaths("CUDNN_PATH", "bin"));
            paths.AddRange(ResolveInstalledCudaBinDirectories());
            paths.AddRange(ResolveInstalledCudnnBinDirectories());
            foreach (var path in paths.Where(Directory.Exists).Reverse()) {
                PrependProcessPath(path);
            }
        }

        public static string GetCudaDependencyPathDisplay() {
            return GetDependencyPathDisplay(
                Preferences.Default.CudaPath,
                ResolveInstalledCudaBinDirectories,
                CudaRuntimeProbeFile);
        }

        public static string GetCudnnDependencyPathDisplay() {
            return GetDependencyPathDisplay(
                Preferences.Default.CudnnPath,
                ResolveInstalledCudnnBinDirectories,
                CudnnRuntimeProbeFile);
        }

        static string GetDependencyPathDisplay(string configuredPath, Func<IEnumerable<string>> autoDetect, string probeFile) {
            var configured = ResolveDependencyPathCandidates(configuredPath, probeFile)
                .FirstOrDefault(path => HasProbeFile(path, probeFile));
            if (!string.IsNullOrWhiteSpace(configured)) {
                return configured;
            }
            return autoDetect().FirstOrDefault() ?? "None";
        }

        const string CudaRuntimeProbeFile = "cudart64_12.dll";
        const string CudnnRuntimeProbeFile = "cudnn64_9.dll";

        static IEnumerable<string> ResolveConfiguredDependencyPaths(string variableName, string subdir) {
            var value = Environment.GetEnvironmentVariable(variableName);
            if (string.IsNullOrWhiteSpace(value)) {
                yield break;
            }
            foreach (var path in ResolveDependencyPathCandidates(value, string.Empty)) {
                yield return path;
            }
            yield return Path.Combine(value, subdir);
        }

        static IEnumerable<string> ResolveDependencyPathCandidates(string path, string probeFile) {
            if (string.IsNullOrWhiteSpace(path)) {
                yield break;
            }
            yield return path;
            yield return Path.Combine(path, "bin");
            var bin = Path.Combine(path, "bin");
            foreach (var dir in ResolveVersionedDirectories(bin, "12.*")) {
                yield return Path.Combine(dir, "x64");
            }
            foreach (var versionDir in ResolveVersionedDirectories(path, "v*")) {
                var versionBin = Path.Combine(versionDir, "bin");
                yield return versionBin;
                foreach (var dir in ResolveVersionedDirectories(versionBin, "12.*")) {
                    yield return Path.Combine(dir, "x64");
                }
            }
        }

        static IEnumerable<string> ResolveExistingDependencyPaths(string path, string probeFile) {
            return ResolveDependencyPathCandidates(path, probeFile)
                .Where(candidate => HasProbeFile(candidate, probeFile));
        }

        static bool HasProbeFile(string path, string probeFile) {
            return Directory.Exists(path)
                && (string.IsNullOrWhiteSpace(probeFile) || File.Exists(Path.Combine(path, probeFile)));
        }

        static IEnumerable<string> ResolveInstalledCudaBinDirectories() {
            string root = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "NVIDIA GPU Computing Toolkit",
                "CUDA");
            return ResolveVersionedDirectories(root, "v*")
                .Select(dir => Path.Combine(dir, "bin"))
                .Where(dir => File.Exists(Path.Combine(dir, CudaRuntimeProbeFile)));
        }

        static IEnumerable<string> ResolveInstalledCudnnBinDirectories() {
            string root = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "NVIDIA",
                "CUDNN");
            return ResolveVersionedDirectories(root, "v*")
                .SelectMany(versionDir => ResolveVersionedDirectories(Path.Combine(versionDir, "bin"), "12.*"))
                .Select(dir => Path.Combine(dir, "x64"))
                .Where(dir => File.Exists(Path.Combine(dir, CudnnRuntimeProbeFile)));
        }

        static IEnumerable<string> ResolveVersionedDirectories(string root, string pattern) {
            if (!Directory.Exists(root)) {
                return Enumerable.Empty<string>();
            }
            return Directory.GetDirectories(root, pattern)
                .OrderByDescending(ParseLastVersionSegment);
        }

        static Version ParseLastVersionSegment(string path) {
            var name = Path.GetFileName(path).TrimStart('v', 'V');
            return Version.TryParse(name, out var version) ? version : new Version(0, 0);
        }

        static void PrependProcessPath(string runtimeDir) {
            string path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            if (!path.Split(Path.PathSeparator).Contains(runtimeDir, StringComparer.OrdinalIgnoreCase)) {
                Environment.SetEnvironmentVariable("PATH", runtimeDir + Path.PathSeparator + path);
            }
        }

        public static bool TryResolveCudaRuntimeDirectory(out string runtimeDir) {
            var dir = GetCudaRuntimeDirectory();
            if (HasCudaOrtRuntime(dir)) {
                runtimeDir = dir;
                return true;
            }
            runtimeDir = string.Empty;
            return false;
        }

        static string GetCudaRuntimeDirectory() {
            return Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64-cuda", "native");
        }

        static bool HasCudaOrtRuntime(string dir) {
            return Directory.Exists(dir)
                && File.Exists(Path.Combine(dir, "onnxruntime.dll"))
                && File.Exists(Path.Combine(dir, "onnxruntime_providers_shared.dll"))
                && File.Exists(Path.Combine(dir, "onnxruntime_providers_cuda.dll"));
        }
    }
}
