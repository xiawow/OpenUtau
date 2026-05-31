using System;

namespace OpenUtau.Core.HifiNeural {
    public interface IHifiMelEnhancer {
        float[,] Enhance(float[,] mel, float[] f0);
    }

    public sealed class NoOpMelEnhancer : IHifiMelEnhancer {
        public float[,] Enhance(float[,] mel, float[] f0) {
            return mel;
        }
    }

    public sealed class LightSmoothMelEnhancer : IHifiMelEnhancer {
        // Ultra-light temporal smoothing to reduce rough frame jitter while keeping transients.
        const float CenterWeight = 0.84f;
        const float NeighborWeight = 0.08f;

        public float[,] Enhance(float[,] mel, float[] f0) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            if (frames <= 2) {
                return mel;
            }
            var output = new float[bins, frames];
            for (int m = 0; m < bins; m++) {
                output[m, 0] = mel[m, 0];
                output[m, frames - 1] = mel[m, frames - 1];
                for (int t = 1; t < frames - 1; t++) {
                    float left = mel[m, t - 1];
                    float center = mel[m, t];
                    float right = mel[m, t + 1];
                    output[m, t] = left * NeighborWeight + center * CenterWeight + right * NeighborWeight;
                }
            }
            return output;
        }
    }

    public static class HifiRenderConfig {
        public const string MelEnhanceNone = "none";
        public const string MelEnhanceLight = "light";

        public static string MelEnhanceMode {
            get {
                string? env = Environment.GetEnvironmentVariable("HIFI_NEURAL_MEL_ENHANCE_MODE");
                if (!string.IsNullOrWhiteSpace(env)) {
                    return NormalizeMelEnhanceMode(env);
                }
                return NormalizeMelEnhanceMode(OpenUtau.Core.Util.Preferences.Default.HifiNeuralMelEnhanceMode);
            }
        }

        public static bool DebugExportEnabled {
            get {
                string? env = Environment.GetEnvironmentVariable("HIFI_NEURAL_DEBUG_EXPORT");
                if (bool.TryParse(env, out bool enabled)) {
                    return enabled;
                }
                return OpenUtau.Core.Util.Preferences.Default.HifiNeuralDebugExportEnabled;
            }
        }

        public static string CacheKey() {
            return $"v24-meldomainconcat-overlapcrossfade-sustainfreeze-consonantfix-f0continuous-loudnessloud-microvar-enh{MelEnhanceMode}-dbg{DebugExportEnabled}";
        }

        public static IHifiMelEnhancer CreateMelEnhancer() {
            return MelEnhanceMode == MelEnhanceLight
                ? new LightSmoothMelEnhancer()
                : new NoOpMelEnhancer();
        }

        public static string NormalizeMelEnhanceMode(string? value) {
            return string.Equals(value, MelEnhanceLight, StringComparison.OrdinalIgnoreCase)
                ? MelEnhanceLight
                : MelEnhanceNone;
        }
    }
}
