using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Classic;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.Neutrino {
    public class NeutrinoSinger : USinger {
        public override string Id => voicebank.Id;
        public override string Name => voicebank.Name;
        public override Dictionary<string, string> LocalizedNames => voicebank.LocalizedNames;
        public override USingerType SingerType => USingerType.Neutrino;
        public override string BasePath => voicebank.BasePath;
        public override string Author => voicebank.Author;
        public override string Voice => voicebank.Voice;
        public override string Location => Path.GetDirectoryName(voicebank.File);
        public override string Web => voicebank.Web;
        public override string Version => voicebank.Version;
        public override string OtherInfo => voicebank.OtherInfo;
        public override IList<string> Errors => errors;
        public override string Avatar => voicebank.Image == null ? null : Path.Combine(Location, voicebank.Image);
        public override byte[] AvatarData => avatarData;
        public override string Portrait => voicebank.Portrait == null ? null : Path.Combine(Location, voicebank.Portrait);
        public override float PortraitOpacity => voicebank.PortraitOpacity;
        public override int PortraitHeight => voicebank.PortraitHeight;
        public override string Sample => voicebank.Sample == null ? null : Path.Combine(Location, voicebank.Sample);
        public override string DefaultPhonemizer =>
            voicebank.DefaultPhonemizer ?? "OpenUtau.Core.Neutrino.NeutrinoPhonemizer";
        public override Encoding TextFileEncoding => voicebank.TextFileEncoding;
        public override IList<USubbank> Subbanks => subbanks;
        public override IList<UOto> Otos => otos;

        Voicebank voicebank;
        List<string> errors = new List<string>();
        List<USubbank> subbanks = new List<USubbank>();
        List<UOto> otos = new List<UOto>();
        Dictionary<string, UOto> otoMap = new Dictionary<string, UOto>();
        public byte[] avatarData;

        public NeutrinoConfig config;
        public InferenceSession timingSession;
        public InferenceSession pitchSession;
        public InferenceSession melspecSession;
        public InferenceSession vocoderSession;
        public InferenceSession legacyEmbeddingSession;
        public InferenceSession legacyAcousticSession;
        public InferenceSession legacyWorldF0Session;
        public InferenceSession legacyVocoderSession;
        NeutrinoLegacyTimingModel legacyTimingModel;
        string timingModelPath = string.Empty;
        string pitchModelPath = string.Empty;
        string melspecModelPath = string.Empty;
        string vocoderModelPath = string.Empty;
        string legacyTimingModelPath = string.Empty;
        string legacyTimingStatsPath = string.Empty;
        string legacyEmbeddingModelPath = string.Empty;
        string legacyAcousticModelPath = string.Empty;
        string legacyWorldF0ModelPath = string.Empty;
        string legacyVocoderModelPath = string.Empty;
        int legacyV2RenderQuality = -1;

        public bool IsLegacyV2 => config?.isLegacyV2 == true || HasLegacyV2ModelFiles(ResolveModelDir());
        public int LegacyV2SampleRate { get; private set; } = 48000;
        public int LegacyV2SamplesPerFrame => Math.Max(1, LegacyV2SampleRate / 200);

        static readonly object sessionLock = new object();

        public NeutrinoSinger(Voicebank voicebank) {
            this.voicebank = voicebank;
            found = true;
        }

        public override void EnsureLoaded() {
            if (Loaded) return;
            Reload();
        }

        public override void Reload() {
            if (!Found) return;
            try {
                voicebank.Reload();
                Load();
                loaded = true;
            } catch (Exception e) {
                Log.Error(e, $"Failed to load NEUTRINO singer {voicebank.File}");
            }
        }

        void Load() {
            config = NeutrinoConfig.Load(Location);
            config.isLegacyV2 |= HasLegacyV2ModelFiles(ResolveModelDir());

            var dictPath = ResolveDictionaryPath();
            if (!string.IsNullOrEmpty(dictPath)) {
                NeutrinoPhoneme.LoadDictionary(dictPath);
            } else {
                Log.Warning($"NEUTRINO dictionary not found near {Location}");
            }

            subbanks.Clear();
            otos.Clear();
            otoMap.Clear();
            subbanks.Add(new USubbank(new Subbank() {
                Prefix = string.Empty,
                Suffix = string.Empty,
                ToneRanges = new[] { "C1-B7" },
            }));
            foreach (var phone in NeutrinoPhoneme.AllPhonemes) {
                var uOto = UOto.OfDummy(phone);
                if (!otoMap.ContainsKey(uOto.Alias)) {
                    otos.Add(uOto);
                    otoMap.Add(uOto.Alias, uOto);
                }
            }

            if (Avatar != null && File.Exists(Avatar)) {
                try {
                    using (var stream = new FileStream(Avatar, FileMode.Open, FileAccess.Read))
                    using (var memoryStream = new MemoryStream()) {
                        stream.CopyTo(memoryStream);
                        avatarData = memoryStream.ToArray();
                    }
                } catch (Exception e) {
                    avatarData = null;
                    Log.Error(e, "Failed to load NEUTRINO avatar");
                }
            }
        }

        string ResolveDictionaryPath() {
            var candidates = new List<string>();
            void AddCandidate(string path) {
                if (!string.IsNullOrEmpty(path) && !candidates.Contains(path)) {
                    candidates.Add(path);
                }
            }

            AddCandidate(Path.Combine(Location, "settings", "dic", "japanese.utf_8.table"));

            var locationDir = new DirectoryInfo(Location);
            for (int i = 0; i < 5 && locationDir != null; i++, locationDir = locationDir.Parent) {
                AddCandidate(Path.Combine(locationDir.FullName, "settings", "dic", "japanese.utf_8.table"));
            }

            var modelDir = ResolveModelDir();
            if (Directory.Exists(modelDir)) {
                var dir = new DirectoryInfo(modelDir);
                for (int i = 0; i < 5 && dir != null; i++, dir = dir.Parent) {
                    AddCandidate(Path.Combine(dir.FullName, "settings", "dic", "japanese.utf_8.table"));
                }
            }

            AddCandidate(Path.Combine(PathManager.Inst.DataPath, "settings", "dic", "japanese.utf_8.table"));
            return candidates.FirstOrDefault(File.Exists);
        }

        /// <summary>
        /// Get or create ONNX inference sessions. Lazy-loaded with DML support.
        /// </summary>
        public void EnsureSessions() {
            if (timingSession != null
                && pitchSession != null
                && melspecSession != null
                && vocoderSession != null) {
                return;
            }
            lock (sessionLock) {
                EnsureModelPaths();
                timingSession ??= LoadSession(timingModelPath, OnnxRunnerChoice.Default);
                pitchSession ??= LoadSession(pitchModelPath, OnnxRunnerChoice.Default);
                melspecSession ??= LoadSession(melspecModelPath, OnnxRunnerChoice.Default);
                vocoderSession ??= LoadSession(vocoderModelPath, OnnxRunnerChoice.Default);
                Log.Information($"Loaded NEUTRINO ONNX sessions for {Name}");
            }
        }

        public void EnsureTimingSession() {
            if (timingSession != null) return;
            lock (sessionLock) {
                EnsureModelPaths();
                timingSession ??= LoadSession(timingModelPath, OnnxRunnerChoice.Default);
            }
        }

        public void EnsurePitchSession() {
            if (pitchSession != null) return;
            lock (sessionLock) {
                EnsureModelPaths();
                pitchSession ??= LoadSession(pitchModelPath, OnnxRunnerChoice.Default);
            }
        }

        public void EnsureMelspecSession() {
            if (melspecSession != null) return;
            lock (sessionLock) {
                EnsureModelPaths();
                melspecSession ??= LoadSession(melspecModelPath, OnnxRunnerChoice.Default);
            }
        }

        public void EnsureVocoderSession() {
            if (vocoderSession != null) return;
            lock (sessionLock) {
                EnsureModelPaths();
                vocoderSession ??= LoadSession(vocoderModelPath, OnnxRunnerChoice.Default);
            }
        }

        public void EnsureLegacyV2Sessions() {
            lock (sessionLock) {
                EnsureLegacyV2ModelPaths();
                if (legacyEmbeddingSession != null
                    && legacyAcousticSession != null
                    && legacyWorldF0Session != null
                    && legacyVocoderSession != null) {
                    return;
                }
                legacyEmbeddingSession ??= LoadSession(legacyEmbeddingModelPath, OnnxRunnerChoice.Default);
                legacyAcousticSession ??= LoadSession(legacyAcousticModelPath, OnnxRunnerChoice.Default);
                legacyWorldF0Session ??= LoadSession(legacyWorldF0ModelPath, OnnxRunnerChoice.Default);
                legacyVocoderSession ??= LoadSession(legacyVocoderModelPath, OnnxRunnerChoice.Default);
                Log.Information($"Loaded NEUTRINO v2 ONNX sessions for {Name}");
            }
        }

        public void EnsureLegacyV2TimingModel() {
            if (legacyTimingModel != null) return;
            lock (sessionLock) {
                EnsureLegacyV2ModelPaths();
                legacyTimingModel ??= NeutrinoLegacyTimingModel.Load(legacyTimingModelPath, legacyTimingStatsPath);
                Log.Information($"Loaded NEUTRINO v2 timing model for {Name}");
            }
        }

        void EnsureModelPaths() {
            if (!string.IsNullOrEmpty(timingModelPath)
                && !string.IsNullOrEmpty(pitchModelPath)
                && !string.IsNullOrEmpty(melspecModelPath)
                && !string.IsNullOrEmpty(vocoderModelPath)) {
                return;
            }
            var modelDir = ResolveModelDir();
            timingModelPath = Path.Combine(modelDir, "t.bin");
            pitchModelPath = Path.Combine(modelDir, "p.bin");
            melspecModelPath = Path.Combine(modelDir, "s.bin");
            vocoderModelPath = Path.Combine(modelDir, "v.bin");
        }

        void EnsureLegacyV2ModelPaths() {
            int requestedQuality = NormalizeLegacyV2RenderQuality(Preferences.Default.NeutrinoLegacyV2RenderQuality);
            if (!string.IsNullOrEmpty(legacyEmbeddingModelPath)
                && !string.IsNullOrEmpty(legacyTimingModelPath)
                && !string.IsNullOrEmpty(legacyTimingStatsPath)
                && !string.IsNullOrEmpty(legacyAcousticModelPath)
                && !string.IsNullOrEmpty(legacyWorldF0ModelPath)
                && !string.IsNullOrEmpty(legacyVocoderModelPath)
                && legacyV2RenderQuality == requestedQuality) {
                return;
            }

            if (legacyV2RenderQuality >= 0 && legacyV2RenderQuality != requestedQuality) {
                legacyAcousticSession?.Dispose();
                legacyAcousticSession = null;
                legacyVocoderSession?.Dispose();
                legacyVocoderSession = null;
                legacyAcousticModelPath = string.Empty;
                legacyVocoderModelPath = string.Empty;
            }
            legacyV2RenderQuality = requestedQuality;

            var modelDir = ResolveModelDir();
            legacyTimingModelPath = RequireExisting(Path.Combine(modelDir, "t.bin"), "NEUTRINO v2 timing model");
            legacyTimingStatsPath = RequireExisting(Path.Combine(modelDir, "ts.bin"), "NEUTRINO v2 timing stats");
            legacyEmbeddingModelPath = RequireExisting(Path.Combine(modelDir, "e.bin"), "NEUTRINO v2 embedding model");
            var suffixes = ResolveLegacyV2QualitySuffixFallbacks(requestedQuality);
            foreach (var suffix in suffixes) {
                var acoustic = Path.Combine(modelDir, $"d{suffix}.bin");
                var vocoder = Path.Combine(modelDir, $"v{suffix}.bin");
                if (File.Exists(acoustic) && File.Exists(vocoder)) {
                    legacyAcousticModelPath = acoustic;
                    legacyVocoderModelPath = vocoder;
                    break;
                }
            }
            if (string.IsNullOrEmpty(legacyAcousticModelPath)) {
                legacyAcousticModelPath = FirstExisting(suffixes
                    .Select(suffix => Path.Combine(modelDir, $"d{suffix}.bin"))
                    .ToArray());
            }
            if (string.IsNullOrEmpty(legacyVocoderModelPath)) {
                legacyVocoderModelPath = FirstExisting(suffixes
                    .Select(suffix => Path.Combine(modelDir, $"v{suffix}.bin"))
                    .ToArray());
            }
            legacyWorldF0ModelPath = ResolveLegacyWorldF0ModelPath(modelDir);

            if (string.IsNullOrEmpty(legacyAcousticModelPath)) {
                throw new FileNotFoundException($"NEUTRINO v2 acoustic model was not found in {modelDir}");
            }
            if (string.IsNullOrEmpty(legacyVocoderModelPath)) {
                throw new FileNotFoundException($"NEUTRINO v2 vocoder model was not found in {modelDir}");
            }
            if (string.IsNullOrEmpty(legacyWorldF0ModelPath)) {
                throw new FileNotFoundException(
                    $"NEUTRINO v2 world_f0.bin was not found near {modelDir}. " +
                    "Place it in the original NEUTRINO bin directory or OpenUtau data Neutrino/v2 directory.");
            }

            LegacyV2SampleRate = Path.GetFileName(legacyVocoderModelPath).Equals("ve.bin", StringComparison.OrdinalIgnoreCase)
                ? 24000
                : 48000;
        }

        string ResolveModelDir() {
            var nested = Path.Combine(Location, "model");
            if (HasV3ModelFiles(nested)) {
                return nested;
            }
            if (HasV3ModelFiles(Location)) {
                return Location;
            }
            if (HasLegacyV2ModelFiles(Location)) {
                return Location;
            }
            if (HasLegacyV2ModelFiles(nested)) {
                return nested;
            }
            if (Directory.Exists(nested)) {
                var legacyModelDir = Directory.EnumerateDirectories(nested)
                    .FirstOrDefault(HasLegacyV2ModelFiles);
                if (!string.IsNullOrEmpty(legacyModelDir)) {
                    return legacyModelDir;
                }
            }
            return nested;
        }

        static bool HasV3ModelFiles(string directory) {
            return File.Exists(Path.Combine(directory, "t.bin"))
                && File.Exists(Path.Combine(directory, "p.bin"))
                && File.Exists(Path.Combine(directory, "s.bin"))
                && File.Exists(Path.Combine(directory, "v.bin"));
        }

        static bool HasLegacyV2ModelFiles(string directory) {
            return File.Exists(Path.Combine(directory, "e.bin"))
                && File.Exists(Path.Combine(directory, "t.bin"))
                && (File.Exists(Path.Combine(directory, "ds.bin"))
                    || File.Exists(Path.Combine(directory, "da.bin"))
                    || File.Exists(Path.Combine(directory, "de.bin")))
                && (File.Exists(Path.Combine(directory, "vs.bin"))
                    || File.Exists(Path.Combine(directory, "va.bin"))
                    || File.Exists(Path.Combine(directory, "ve.bin")));
        }

        public static int NormalizeLegacyV2RenderQuality(int quality) {
            return Math.Clamp(quality, 0, 2);
        }

        public static string LegacyV2RenderQualityCacheKey() {
            return ResolveLegacyV2QualitySuffix(NormalizeLegacyV2RenderQuality(
                Preferences.Default.NeutrinoLegacyV2RenderQuality));
        }

        static string ResolveLegacyV2QualitySuffix(int quality) {
            return quality switch {
                0 => "e",
                2 => "a",
                _ => "s",
            };
        }

        static string[] ResolveLegacyV2QualitySuffixFallbacks(int quality) {
            return quality switch {
                0 => new[] { "e", "s", "a" },
                2 => new[] { "a", "s", "e" },
                _ => new[] { "s", "a", "e" },
            };
        }

        static string FirstExisting(params string[] paths) {
            return paths.FirstOrDefault(File.Exists) ?? string.Empty;
        }

        static string RequireExisting(string path, string description) {
            if (!File.Exists(path)) {
                throw new FileNotFoundException($"{description} was not found: {path}", path);
            }
            return path;
        }

        string ResolveLegacyWorldF0ModelPath(string modelDir) {
            var candidates = new List<string>();
            void AddCandidate(string path) {
                if (!string.IsNullOrEmpty(path) && !candidates.Contains(path)) {
                    candidates.Add(path);
                }
            }

            var dir = new DirectoryInfo(modelDir);
            for (int i = 0; i < 8 && dir != null; i++, dir = dir.Parent) {
                AddCandidate(Path.Combine(dir.FullName, "world_f0.bin"));
                AddCandidate(Path.Combine(dir.FullName, "bin", "world_f0.bin"));
            }
            AddCandidate(Path.Combine(PathManager.Inst.DataPath, "Neutrino", "v2", "world_f0.bin"));
            AddCandidate(Path.Combine(AppContext.BaseDirectory, "Neutrino", "v2", "world_f0.bin"));
            return candidates.FirstOrDefault(File.Exists) ?? string.Empty;
        }

        static InferenceSession LoadSession(string path, OnnxRunnerChoice runnerChoice) {
            var bytes = File.ReadAllBytes(path);
            return Onnx.getInferenceSession(bytes, runnerChoice);
        }

        public float[] RunTiming(IReadOnlyCollection<NamedOnnxValue> inputs) {
            EnsureTimingSession();
            return RunWithCpuFallback(ref timingSession, timingModelPath, inputs, "timing");
        }

        public float[] RunPitch(IReadOnlyCollection<NamedOnnxValue> inputs) {
            EnsurePitchSession();
            return RunWithCpuFallback(ref pitchSession, pitchModelPath, inputs, "pitch");
        }

        public float[] RunMelspec(IReadOnlyCollection<NamedOnnxValue> inputs) {
            EnsureMelspecSession();
            return RunWithCpuFallback(ref melspecSession, melspecModelPath, inputs, "melspec");
        }

        public float[] RunVocoder(IReadOnlyCollection<NamedOnnxValue> inputs) {
            EnsureVocoderSession();
            return RunWithCpuFallback(ref vocoderSession, vocoderModelPath, inputs, "vocoder");
        }

        public float[] RunLegacyEmbedding(IReadOnlyCollection<NamedOnnxValue> inputs) {
            EnsureLegacyV2Sessions();
            return RunWithCpuFallback(ref legacyEmbeddingSession, legacyEmbeddingModelPath, inputs, "v2 embedding");
        }

        public float[] RunLegacyTiming(float[] rawFeatures, int phones) {
            EnsureLegacyV2TimingModel();
            return legacyTimingModel.PredictDeltasMs(rawFeatures, phones);
        }

        public float[] RunLegacyAcoustic(IReadOnlyCollection<NamedOnnxValue> inputs) {
            EnsureLegacyV2Sessions();
            return RunWithCpuFallback(ref legacyAcousticSession, legacyAcousticModelPath, inputs, "v2 acoustic");
        }

        public float[] RunLegacyWorldF0(IReadOnlyCollection<NamedOnnxValue> inputs) {
            EnsureLegacyV2Sessions();
            return RunWithCpuFallback(ref legacyWorldF0Session, legacyWorldF0ModelPath, inputs, "v2 world_f0");
        }

        public float[] RunLegacyVocoder(IReadOnlyCollection<NamedOnnxValue> inputs) {
            EnsureLegacyV2Sessions();
            return RunWithCpuFallback(ref legacyVocoderSession, legacyVocoderModelPath, inputs, "v2 vocoder");
        }

        float[] RunWithCpuFallback(
            ref InferenceSession session,
            string path,
            IReadOnlyCollection<NamedOnnxValue> inputs,
            string modelName) {

            try {
                return RunFirstOutput(session, inputs);
            } catch (OnnxRuntimeException e) when (Preferences.Default.OnnxRunner == "DirectML") {
                Log.Warning(e, $"NEUTRINO {modelName} model failed on DirectML, retrying on CPU");
                lock (sessionLock) {
                    session?.Dispose();
                    session = LoadSession(path, OnnxRunnerChoice.CPU);
                }
                return RunFirstOutput(session, inputs);
            }
        }

        static float[] RunFirstOutput(
            InferenceSession session,
            IReadOnlyCollection<NamedOnnxValue> inputs) {

            lock (session) {
                using var outputs = session.Run(inputs);
                return outputs.First().AsTensor<float>().ToArray();
            }
        }

        /// <summary>
        /// Free all ONNX sessions to release GPU/memory resources.
        /// </summary>
        public void FreeSessions() {
            lock (sessionLock) {
                timingSession?.Dispose();
                timingSession = null;
                pitchSession?.Dispose();
                pitchSession = null;
                melspecSession?.Dispose();
                melspecSession = null;
                vocoderSession?.Dispose();
                vocoderSession = null;
                legacyEmbeddingSession?.Dispose();
                legacyEmbeddingSession = null;
                legacyAcousticSession?.Dispose();
                legacyAcousticSession = null;
                legacyWorldF0Session?.Dispose();
                legacyWorldF0Session = null;
                legacyVocoderSession?.Dispose();
                legacyVocoderSession = null;
                legacyTimingModel = null;
            }
        }

        public override bool TryGetOto(string phoneme, out UOto oto) {
            oto = UOto.OfDummy(phoneme);
            return true;
        }

        public override IEnumerable<UOto> GetSuggestions(string text) {
            if (text != null) text = text.Replace(" ", "");
            bool all = string.IsNullOrEmpty(text);
            return otos.Where(o => all || o.Alias.Contains(text, StringComparison.OrdinalIgnoreCase));
        }

        public override byte[] LoadPortrait() {
            return string.IsNullOrEmpty(Portrait) ? null : File.ReadAllBytes(Portrait);
        }

        public override byte[] LoadSample() {
            return string.IsNullOrEmpty(Sample) ? null : File.ReadAllBytes(Sample);
        }
    }
}
