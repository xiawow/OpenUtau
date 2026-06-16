using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Classic;
using OpenUtau.Core.Ustx;
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
            if (timingSession != null) return;
            lock (sessionLock) {
                if (timingSession != null) return;
                var modelDir = ResolveModelDir();
                timingSession = LoadSession(Path.Combine(modelDir, "t.bin"));
                pitchSession = LoadSession(Path.Combine(modelDir, "p.bin"));
                melspecSession = LoadSession(Path.Combine(modelDir, "s.bin"));
                vocoderSession = LoadSession(Path.Combine(modelDir, "v.bin"));
                Log.Information($"Loaded NEUTRINO ONNX sessions for {Name}");
            }
        }

        string ResolveModelDir() {
            var nested = Path.Combine(Location, "model");
            if (File.Exists(Path.Combine(nested, "t.bin"))) {
                return nested;
            }
            if (File.Exists(Path.Combine(Location, "t.bin"))) {
                return Location;
            }
            return nested;
        }

        static InferenceSession LoadSession(string path) {
            var bytes = File.ReadAllBytes(path);
            return Onnx.getInferenceSession(bytes, OnnxRunnerChoice.Default);
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
