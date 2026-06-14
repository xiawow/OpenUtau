namespace OpenUtau.Core.HifiNeural {
    internal static class HifiSustainModes {
        public const int Loop = 0;
        public const int Texture = 1;
        public const int Natural = 2;

        public static readonly string[] Options = { "loop", "texture", "natural" };

        public static int Normalize(double value) {
            int mode = (int)System.Math.Round(value);
            if (mode < Loop || mode > Natural) {
                return Loop;
            }
            return mode;
        }

        public static string StrategyName(int mode) {
            return Normalize(mode) switch {
                Texture => "he_texture",
                Natural => "he_natural",
                _ => "he_loop",
            };
        }
    }
}
