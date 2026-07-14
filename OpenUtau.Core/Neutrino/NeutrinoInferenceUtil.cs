using System;
using System.IO;

namespace OpenUtau.Core.Neutrino {
    internal readonly struct NeutrinoPhoneChunk {
        public int PhoneStart { get; }
        public int PhoneCount { get; }
        public bool IsActive { get; }

        public NeutrinoPhoneChunk(int phoneStart, int phoneCount, bool isActive) {
            PhoneStart = phoneStart;
            PhoneCount = phoneCount;
            IsActive = isActive;
        }
    }

    internal readonly struct NeutrinoFrameChunk {
        public int PhoneStart { get; }
        public int PhoneCount { get; }
        public int FrameStart { get; }
        public int FrameCount { get; }
        public bool IsActive { get; }

        public NeutrinoFrameChunk(
            int phoneStart,
            int phoneCount,
            int frameStart,
            int frameCount,
            bool isActive) {

            PhoneStart = phoneStart;
            PhoneCount = phoneCount;
            FrameStart = frameStart;
            FrameCount = frameCount;
            IsActive = isActive;
        }
    }

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

        public static NeutrinoPhoneChunk[] BuildPhoneChunks(long[] phonemeIds) {
            var chunks = new System.Collections.Generic.List<NeutrinoPhoneChunk>();
            if (phonemeIds.Length == 0) {
                return chunks.ToArray();
            }

            int chunkStart = 0;
            bool chunkIsActive = true;
            bool inPause = false;
            bool afterBreath = false;
            for (int phone = 0; phone < phonemeIds.Length; phone++) {
                if (phonemeIds[phone] == NeutrinoPhoneme.PAU) {
                    if (!inPause) {
                        if (phone > chunkStart) {
                            chunks.Add(new NeutrinoPhoneChunk(
                                chunkStart, phone - chunkStart, chunkIsActive));
                        }
                        chunkStart = phone;
                        chunkIsActive = false;
                        inPause = true;
                        afterBreath = false;
                    }
                    continue;
                }

                if (phonemeIds[phone] == NeutrinoPhoneme.BR) {
                    inPause = false;
                    afterBreath = true;
                    continue;
                }

                if (inPause || afterBreath) {
                    chunks.Add(new NeutrinoPhoneChunk(
                        chunkStart, phone - chunkStart, chunkIsActive));
                    chunkStart = phone;
                    chunkIsActive = true;
                    inPause = false;
                    afterBreath = false;
                }
            }
            chunks.Add(new NeutrinoPhoneChunk(
                chunkStart, phonemeIds.Length - chunkStart, chunkIsActive));
            return chunks.ToArray();
        }

        public static double[] BuildTimingBoundaries(
            float[] scoreDurations,
            long[] phonePositions,
            NeutrinoPhoneChunk[] chunks,
            double frameSeconds,
            Func<NeutrinoPhoneChunk, float[]> predictBoundaryShifts) {

            if (scoreDurations.Length != phonePositions.Length) {
                throw new ArgumentException("Score duration and phone position lengths must match.");
            }

            var baseBoundaries = BuildBaseBoundaryTimes(scoreDurations, phonePositions);
            var globalBoundaryShifts = new float[baseBoundaries.Length];
            foreach (var chunk in chunks) {
                if (!chunk.IsActive) {
                    continue;
                }
                var chunkShifts = predictBoundaryShifts(chunk);
                if (chunkShifts == null || chunkShifts.Length < chunk.PhoneCount) {
                    throw new InvalidDataException(
                        $"Timing chunk output is too short: actual {chunkShifts?.Length ?? 0}, " +
                        $"expected at least {chunk.PhoneCount}.");
                }

                // The official loader copies one value per phone and discards the
                // model's extra final value before applying shifts globally.
                Array.Copy(
                    chunkShifts,
                    0,
                    globalBoundaryShifts,
                    chunk.PhoneStart,
                    chunk.PhoneCount);
            }
            return ApplyTimingBoundaryShifts(baseBoundaries, globalBoundaryShifts, frameSeconds);
        }

        public static NeutrinoFrameChunk[] BuildFrameChunks(
            NeutrinoPhoneChunk[] phoneChunks,
            double[] boundaries,
            int totalFrames,
            double frameSeconds) {

            var chunks = new NeutrinoFrameChunk[phoneChunks.Length];
            for (int i = 0; i < chunks.Length; i++) {
                var chunk = phoneChunks[i];
                int frameStart = Math.Clamp(
                    (int)Math.Round(boundaries[chunk.PhoneStart] / frameSeconds),
                    0,
                    totalFrames);
                int frameEnd = Math.Clamp(
                    (int)Math.Round(boundaries[chunk.PhoneStart + chunk.PhoneCount] / frameSeconds),
                    frameStart,
                    totalFrames);
                chunks[i] = new NeutrinoFrameChunk(
                    chunk.PhoneStart,
                    chunk.PhoneCount,
                    frameStart,
                    frameEnd - frameStart,
                    chunk.IsActive);
            }
            return chunks;
        }

        public static T[] Slice<T>(T[] values, int start, int length) {
            var result = new T[length];
            Array.Copy(values, start, result, 0, length);
            return result;
        }

        static double[] BuildBaseBoundaryTimes(float[] scoreDurations, long[] phonePositions) {
            int numPhones = scoreDurations.Length;
            var boundaries = new double[numPhones + 1];
            double time = 0;
            for (int i = 0; i < numPhones; i++) {
                boundaries[i] = time;
                long nextPosition = i + 1 < numPhones ? phonePositions[i + 1] : -1;
                if (i == numPhones - 1 || nextPosition <= phonePositions[i]) {
                    time += scoreDurations[i];
                }
            }
            boundaries[numPhones] = time;
            return boundaries;
        }

        static double[] ApplyTimingBoundaryShifts(
            double[] baseBoundaries,
            float[] boundaryShifts,
            double frameSeconds) {

            var boundaries = (double[])baseBoundaries.Clone();
            for (int i = 1; i < boundaries.Length - 1; i++) {
                double shifted = baseBoundaries[i] + boundaryShifts[i];
                boundaries[i] = Math.Round(
                    Math.Max(shifted, boundaries[i - 1] + frameSeconds) * 1000.0) / 1000.0;
            }
            for (int i = 1; i < boundaries.Length; i++) {
                if (boundaries[i] <= boundaries[i - 1]) {
                    boundaries[i] = Math.Round(
                        (boundaries[i - 1] + frameSeconds) * 1000.0) / 1000.0;
                }
            }
            return boundaries;
        }
    }
}
