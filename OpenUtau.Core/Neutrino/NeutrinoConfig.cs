using System;
using System.IO;
using Serilog;

namespace OpenUtau.Core.Neutrino {
    /// <summary>
    /// NEUTRINO singer configuration. Reads info.toml from the model directory.
    /// </summary>
    public class NeutrinoConfig {
        public string singerName = "";
        public string gender = "female";
        public string language = "Japanese";
        public int topKey = 86;     // MIDI note upper limit
        public int bottomKey = 41;  // MIDI note lower limit
        public string modelVersion = "";
        public bool support = true;

        // Model file paths (relative to singer location)
        public string timingModel = "model/t.bin";
        public string pitchModel = "model/p.bin";
        public string melspecModel = "model/s.bin";
        public string vocoderModel = "model/v.bin";

        // Synthesis parameters
        public int sampleRate = 48000;
        public int hopSize = 480;    // 100 frames/sec at 48kHz
        public int numMelBins = 100; // mel spectrogram bins

        /// <summary>
        /// Parse info.toml file. Simple parser for the NEUTRINO TOML format.
        /// </summary>
        public static NeutrinoConfig Load(string singerPath) {
            var config = new NeutrinoConfig();
            var infoPath = Path.Combine(singerPath, "model", "info.toml");
            if (!File.Exists(infoPath)) {
                // Try alternative path
                infoPath = Path.Combine(singerPath, "info.toml");
            }
            if (!File.Exists(infoPath)) {
                Log.Warning($"NEUTRINO info.toml not found in {singerPath}");
                return config;
            }

            string section = "";
            foreach (var line in File.ReadAllLines(infoPath)) {
                var trimmed = line.Trim();
                if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#")) continue;

                // Section header
                if (trimmed.StartsWith("[") && trimmed.EndsWith("]")) {
                    section = trimmed.Substring(1, trimmed.Length - 2).Trim();
                    continue;
                }

                // Key = value
                var eqIndex = trimmed.IndexOf('=');
                if (eqIndex < 0) continue;
                var key = trimmed.Substring(0, eqIndex).Trim();
                var value = trimmed.Substring(eqIndex + 1).Trim().Trim('"');

                if (section == "" || section == "speaker") {
                    switch (key) {
                        case "name": config.singerName = value; break;
                        case "gender": config.gender = value; break;
                        case "language": config.language = value; break;
                        case "top_key": int.TryParse(value, out config.topKey); break;
                        case "bottom_key": int.TryParse(value, out config.bottomKey); break;
                        case "version": config.modelVersion = value; break;
                        case "support": config.support = value.ToLower() == "true"; break;
                    }
                }
            }
            return config;
        }

        /// <summary>
        /// Convert MIDI note number to frequency in Hz.
        /// </summary>
        public static double MidiToFreq(double midi) {
            return 440.0 * Math.Pow(2.0, (midi - 69.0) / 12.0);
        }

        /// <summary>
        /// Convert frequency in Hz to MIDI note number.
        /// </summary>
        public static double FreqToMidi(double freq) {
            if (freq <= 0) return 0;
            return 69.0 + 12.0 * Math.Log2(freq / 440.0);
        }

        /// <summary>
        /// Calculate number of frames for a given duration in milliseconds.
        /// </summary>
        public int MsToFrames(double ms) {
            return (int)Math.Round(ms / 1000.0 * sampleRate / hopSize);
        }

        /// <summary>
        /// Calculate frame duration in milliseconds.
        /// </summary>
        public double FrameMs => (double)hopSize / sampleRate * 1000.0;
    }
}
