namespace OpenUtau.Core.HifiNeural {
    internal static class HifiSustainModes {
        public const int Loop = 0;
        public const int Texture = 1;
        public const int Natural = 2;
        public const int Auto = 3;

        public static readonly string[] Options = { "loop", "texture", "natural", "auto" };

        public static int Normalize(double value) {
            int mode = (int)System.Math.Round(value);
            if (mode < Loop || mode > Auto) {
                return Auto;
            }
            return mode;
        }

        public static int ResolveEffectiveMode(double value, double stretchRatio) {
            int mode = Normalize(value);
            if (mode != Auto) {
                return mode;
            }
            if (stretchRatio < 1.5) {
                return Natural;
            }
            if (stretchRatio < 2.0) {
                return Texture;
            }
            return Loop;
        }

        public static string StrategyName(double value, double stretchRatio) {
            int mode = Normalize(value);
            int effectiveMode = ResolveEffectiveMode(mode, stretchRatio);
            string name = effectiveMode switch {
                Texture => "texture",
                Natural => "natural",
                Loop => "loop",
                _ => "auto",
            };
            return mode == Auto ? $"he_auto_{name}" : $"he_{name}";
        }
    }
}
