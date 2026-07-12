using System;
using System.IO;

namespace OpenUtau.Core.Neutrino {
    internal static class NeutrinoInferenceUtil {
        public static bool IsExtensionLyric(string lyric) {
            return lyric == "-" || lyric?.StartsWith("+", StringComparison.Ordinal) == true;
        }

        public static float[] RequireLength(float[] values, int expectedLength, string outputName) {
            if (values.Length != expectedLength) {
                throw new InvalidDataException(
                    $"{outputName} length mismatch: actual {values.Length}, expected {expectedLength}.");
            }
            return values;
        }

        public static float[] RequireTimingBoundaryLength(
            float[] values,
            int phonemeCount,
            string outputName) {

            return RequireLength(values, checked(phonemeCount + 1), outputName);
        }
    }
}
