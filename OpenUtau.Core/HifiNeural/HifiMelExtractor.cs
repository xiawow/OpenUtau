using System;
using System.Collections.Generic;
using System.Numerics;
using System.Linq;
using System.Threading.Tasks;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using OpenUtau.Core.Format;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiMelExtractor {
        public const int SampleRate = 44100;
        public const int Nfft = 2048;
        public const int WinSize = 2048;
        public const int OriginHopSize = 128;
        public const int NMels = 128;
        public const double FMin = 40;
        public const double FMax = 16000;

        // Shared singleton: all fields are readonly after construction, so concurrent use is safe.
        public static readonly HifiMelExtractor Shared = new HifiMelExtractor();

        readonly float[] hann;
        readonly float[,] melFilterbank;
        // Sparse projection ranges: each mel triangular filter is non-zero only over a small,
        // contiguous span of FFT bins. Precomputing [start, end] per mel turns the projection
        // inner loop from O(NMels * Nfft/2) into O(sum of band widths) ~ a few thousand vs 130k.
        readonly int[] melBinStart;
        readonly int[] melBinEnd; // inclusive
        // Precomputed twiddle factor table for the FFT. Each stage of the Cooley-Tukey butterfly
        // needs len/2 twiddle factors; they are stored contiguously so the inner loop indexes
        // into the table instead of computing cos/sin per iteration.
        readonly Complex[] twiddles;

        public HifiMelExtractor() {
            hann = new float[WinSize];
            for (int i = 0; i < WinSize; i++) {
                hann[i] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / WinSize));
            }
            melFilterbank = BuildMelFilterbank();
            (melBinStart, melBinEnd) = BuildMelBinRanges(melFilterbank);
            twiddles = BuildTwiddles(Nfft);
        }

        static (int[] Start, int[] End) BuildMelBinRanges(float[,] filterbank) {
            int bins = filterbank.GetLength(1);
            var start = new int[NMels];
            var end = new int[NMels];
            for (int m = 0; m < NMels; m++) {
                int first = -1;
                int last = -1;
                for (int k = 0; k < bins; k++) {
                    if (filterbank[m, k] > 0f) {
                        if (first < 0) {
                            first = k;
                        }
                        last = k;
                    }
                }
                if (first < 0) {
                    // Degenerate empty band: keep an empty range (start > end) so the loop skips it.
                    start[m] = 1;
                    end[m] = 0;
                } else {
                    start[m] = first;
                    end[m] = last;
                }
            }
            return (start, end);
        }

        public float[] LoadMono(string path) {
            using var waveStream = Wave.OpenFile(path);
            ISampleProvider provider = waveStream.ToSampleProvider();
            if (provider.WaveFormat.Channels != 1) {
                provider = provider.ToMono(1f, 0f);
            }
            if (provider.WaveFormat.SampleRate != SampleRate) {
                provider = new WdlResamplingSampleProvider(provider, SampleRate);
            }
            var samples = Wave.GetSamples(provider);
            for (int i = 0; i < samples.Length; i++) {
                samples[i] = Math.Clamp(samples[i], -1f, 1f);
            }
            return samples;
        }

        public float[,] ExtractFromFile(string path) {
            return Extract(LoadMono(path));
        }

        public static int EstimateFrameCount(int sampleCount) {
            if (sampleCount <= 0) {
                return 0;
            }
            return Math.Max(1, 1 + Math.Max(0, sampleCount - OriginHopSize) / OriginHopSize);
        }

        // Below this frame count the Parallel.For scheduling overhead outweighs the work, so we
        // run the STFT loop serially.
        const int ParallelFrameThreshold = 16;

        sealed class FftScratch {
            public readonly Complex[] Fft = new Complex[Nfft];
            public readonly float[] Magnitude = new float[Nfft / 2 + 1];
            public readonly float[] WarpedMagnitude = new float[Nfft / 2 + 1];
        }

        static ParallelOptions MelParallelOptions() {
            return new ParallelOptions {
                MaxDegreeOfParallelism = Math.Max(1, Preferences.Default.NumRenderThreads),
            };
        }

        public float[,] Extract(float[] samples) {
            return Extract(samples, 0);
        }

        public float[,] Extract(float[] samples, double keyShiftSemitones) {
            if (samples.Length == 0) {
                return new float[NMels, 0];
            }

            var padded = ReflectPad(samples, (WinSize - OriginHopSize) / 2, (WinSize - OriginHopSize + 1) / 2);
            int frames = EstimateFrameCount(samples.Length);
            var mel = new float[NMels, frames];

            // Each STFT frame is independent and writes its own column of `mel`, and the filterbank /
            // hann / bin ranges are read-only after construction, so the frame loop parallelizes
            // cleanly. Thread-local FFT scratch avoids per-frame allocation and write contention.
            if (frames < ParallelFrameThreshold) {
                var scratch = new FftScratch();
                for (int frame = 0; frame < frames; frame++) {
                    ExtractFrame(padded, frame, mel, scratch, keyShiftSemitones);
                }
            } else {
                Parallel.For(
                    0,
                    frames,
                    MelParallelOptions(),
                    () => new FftScratch(),
                    (frame, _, scratch) => {
                        ExtractFrame(padded, frame, mel, scratch, keyShiftSemitones);
                        return scratch;
                    },
                    _ => { });
            }

            ValidateMel(mel);
            if (Log.IsEnabled(Serilog.Events.LogEventLevel.Debug)) {
                LogStats(mel);
            }
            return mel;
        }

        void ExtractFrame(float[] padded, int frame, float[,] mel, FftScratch scratch, double keyShiftSemitones) {
            int start = frame * OriginHopSize;
            var fft = scratch.Fft;
            Array.Clear(fft, 0, fft.Length);
            for (int n = 0; n < WinSize; n++) {
                fft[n] = new Complex(padded[start + n] * hann[n], 0);
            }
            ForwardFft(fft, twiddles);
            ComputeMagnitudes(fft, scratch.Magnitude);
            ProjectMel(ResolveMagnitudeForKeyShift(scratch, keyShiftSemitones), mel, frame);
        }

        public float[,] ExtractAtPositions(float[] samples, IReadOnlyList<double> centerSamplePositions) {
            return ExtractAtPositions(samples, centerSamplePositions, 0);
        }

        public float[,] ExtractAtPositions(float[] samples, IReadOnlyList<double> centerSamplePositions, double keyShiftSemitones) {
            int frames = centerSamplePositions.Count;
            var mel = new float[NMels, frames];
            if (frames == 0) {
                return mel;
            }
            if (samples.Length == 0) {
                FillConstant(mel, (float)Math.Log(1e-5));
                return mel;
            }

            if (frames < ParallelFrameThreshold) {
                var scratch = new FftScratch();
                for (int frame = 0; frame < frames; frame++) {
                    ExtractFrameAtPosition(samples, centerSamplePositions[frame], frame, mel, scratch, keyShiftSemitones);
                }
            } else {
                Parallel.For(
                    0,
                    frames,
                    MelParallelOptions(),
                    () => new FftScratch(),
                    (frame, _, scratch) => {
                        ExtractFrameAtPosition(samples, centerSamplePositions[frame], frame, mel, scratch, keyShiftSemitones);
                        return scratch;
                    },
                    _ => { });
            }

            Log.Debug(
                "HifiMelExtractor variable_position_mel shape=[{MelBins},{Frames}] source_samples={SourceSamples}",
                NMels,
                frames,
                samples.Length);
            ValidateMel(mel);
            if (Log.IsEnabled(Serilog.Events.LogEventLevel.Debug)) {
                LogStats(mel);
            }
            return mel;
        }

        void ExtractFrameAtPosition(float[] samples, double center, int frame, float[,] mel, FftScratch scratch, double keyShiftSemitones) {
            if (double.IsNaN(center) || double.IsInfinity(center)) {
                center = 0;
            }
            center = Math.Clamp(center, 0, samples.Length - 1);
            double start = center - (WinSize - 1) * 0.5;
            var fft = scratch.Fft;
            Array.Clear(fft, 0, fft.Length);
            for (int n = 0; n < WinSize; n++) {
                fft[n] = new Complex(SampleReflectedLinear(samples, start + n) * hann[n], 0);
            }
            ForwardFft(fft, twiddles);
            ComputeMagnitudes(fft, scratch.Magnitude);
            ProjectMel(ResolveMagnitudeForKeyShift(scratch, keyShiftSemitones), mel, frame);
        }

        float[] ResolveMagnitudeForKeyShift(FftScratch scratch, double keyShiftSemitones) {
            if (Math.Abs(keyShiftSemitones) < 1e-4 || double.IsNaN(keyShiftSemitones) || double.IsInfinity(keyShiftSemitones)) {
                return scratch.Magnitude;
            }
            double factor = Math.Pow(2.0, Math.Clamp(keyShiftSemitones, -24.0, 24.0) / 12.0);
            WarpMagnitude(scratch.Magnitude, scratch.WarpedMagnitude, factor);
            return scratch.WarpedMagnitude;
        }

        static void WarpMagnitude(float[] source, float[] output, double factor) {
            if (source.Length == 0) {
                return;
            }
            factor = Math.Clamp(factor, 0.25, 4.0);
            double sourceMean = 0;
            double outputMean = 0;
            for (int k = 0; k < output.Length; k++) {
                sourceMean += source[k];
                // Mirrors hifisampler's PitchAdjustableMelSpectrogram: increasing key_shift uses
                // a larger analysis FFT/window, so original mel bin k reads approximately source
                // energy at k / factor before the fixed mel filterbank projects it.
                double sourceIndex = k / factor;
                float value = SampleLinear(source, sourceIndex) * (float)(1.0 / factor);
                output[k] = value;
                outputMean += value;
            }
            sourceMean /= output.Length;
            outputMean /= output.Length;
            if (sourceMean > 1e-9 && outputMean > 1e-9) {
                float gain = (float)Math.Clamp(sourceMean / outputMean, 0.5, 2.0);
                for (int k = 0; k < output.Length; k++) {
                    output[k] *= gain;
                }
            }
        }

        static float SampleLinear(float[] values, double index) {
            if (values.Length == 0) {
                return 0;
            }
            index = Math.Clamp(index, 0, values.Length - 1);
            int left = (int)Math.Floor(index);
            int right = Math.Min(values.Length - 1, left + 1);
            double alpha = index - left;
            return (float)(values[left] + (values[right] - values[left]) * alpha);
        }

        void ComputeMagnitudes(Complex[] fft, float[] magnitude) {
            for (int k = 0; k < magnitude.Length; k++) {
                double re = fft[k].Real;
                double im = fft[k].Imaginary;
                magnitude[k] = (float)Math.Sqrt(re * re + im * im);
            }
        }

        void ProjectMel(float[] magnitude, float[,] mel, int frame) {
            for (int m = 0; m < NMels; m++) {
                int kStart = melBinStart[m];
                int kEnd = melBinEnd[m];
                double value = 0;
                for (int k = kStart; k <= kEnd; k++) {
                    value += melFilterbank[m, k] * magnitude[k];
                }
                mel[m, frame] = (float)Math.Log(Math.Max(value, 1e-5));
            }
        }

        static float SampleReflectedLinear(float[] samples, double index) {
            if (samples.Length == 0) {
                return 0;
            }
            if (samples.Length == 1) {
                return samples[0];
            }
            int left = (int)Math.Floor(index);
            double alpha = index - left;
            float v0 = samples[ReflectIndex(left, samples.Length)];
            float v1 = samples[ReflectIndex(left + 1, samples.Length)];
            return (float)(v0 + (v1 - v0) * alpha);
        }

        static int ReflectIndex(int index, int length) {
            if (length <= 1) {
                return 0;
            }
            while (index < 0 || index >= length) {
                if (index < 0) {
                    index = -index;
                } else {
                    index = 2 * length - 2 - index;
                }
            }
            return index;
        }

        static void ForwardFft(Complex[] buffer, Complex[] twiddles) {
            int n = buffer.Length;
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
            int twiddleIdx = 0;
            for (int len = 2; len <= n; len <<= 1) {
                int halfLen = len / 2;
                for (int i = 0; i < n; i += len) {
                    int twBase = twiddleIdx;
                    for (int j = 0; j < halfLen; j++) {
                        var u = buffer[i + j];
                        var v = buffer[i + j + halfLen] * twiddles[twBase + j];
                        buffer[i + j] = u + v;
                        buffer[i + j + halfLen] = u - v;
                    }
                }
                twiddleIdx += halfLen;
            }
        }

        static Complex[] BuildTwiddles(int n) {
            // Precompute all twiddle factors for a radix-2 Cooley-Tukey FFT.
            // Stage with length len needs len/2 factors; they are stored contiguously.
            int count = 0;
            for (int len = 2; len <= n; len <<= 1) {
                count += len / 2;
            }
            var table = new Complex[count];
            int idx = 0;
            for (int len = 2; len <= n; len <<= 1) {
                double angle = -2.0 * Math.PI / len;
                double wRe = Math.Cos(angle);
                double wIm = Math.Sin(angle);
                double curRe = 1.0, curIm = 0.0;
                for (int j = 0; j < len / 2; j++) {
                    table[idx++] = new Complex(curRe, curIm);
                    double nextRe = curRe * wRe - curIm * wIm;
                    double nextIm = curRe * wIm + curIm * wRe;
                    curRe = nextRe;
                    curIm = nextIm;
                }
            }
            return table;
        }

        static float[] ReflectPad(float[] samples, int left, int right) {
            var padded = new float[left + samples.Length + right];
            for (int i = 0; i < left; i++) {
                int source = Math.Clamp(left - i, 0, samples.Length - 1);
                padded[i] = samples[source];
            }
            Array.Copy(samples, 0, padded, left, samples.Length);
            for (int i = 0; i < right; i++) {
                int source = Math.Clamp(samples.Length - 2 - i, 0, samples.Length - 1);
                padded[left + samples.Length + i] = samples[source];
            }
            return padded;
        }

        static void FillConstant(float[,] values, float value) {
            for (int m = 0; m < values.GetLength(0); m++) {
                for (int t = 0; t < values.GetLength(1); t++) {
                    values[m, t] = value;
                }
            }
        }

        static double HzToMel(double hz) {
            const double fMin = 0;
            const double fSp = 200.0 / 3;
            double mel = (hz - fMin) / fSp;
            const double minLogHz = 1000;
            const double minLogMel = (minLogHz - fMin) / fSp;
            double logStep = Math.Log(6.4) / 27;
            if (hz >= minLogHz) {
                mel = minLogMel + Math.Log(hz / minLogHz) / logStep;
            }
            return mel;
        }

        static double MelToHz(double mel) {
            const double fMin = 0;
            const double fSp = 200.0 / 3;
            const double minLogHz = 1000;
            const double minLogMel = minLogHz / fSp;
            double logStep = Math.Log(6.4) / 27;
            if (mel >= minLogMel) {
                return minLogHz * Math.Exp(logStep * (mel - minLogMel));
            }
            return fMin + fSp * mel;
        }

        static float[,] BuildMelFilterbank() {
            var filterbank = new float[NMels, Nfft / 2 + 1];
            double minMel = HzToMel(FMin);
            double maxMel = HzToMel(FMax);
            var melPoints = Enumerable.Range(0, NMels + 2)
                .Select(i => minMel + (maxMel - minMel) * i / (NMels + 1))
                .Select(MelToHz)
                .ToArray();
            var fftFreqs = Enumerable.Range(0, Nfft / 2 + 1)
                .Select(i => (double)i * SampleRate / Nfft)
                .ToArray();

            for (int m = 0; m < NMels; m++) {
                double lower = melPoints[m];
                double center = melPoints[m + 1];
                double upper = melPoints[m + 2];
                double enorm = 2.0 / (upper - lower);
                for (int k = 0; k < fftFreqs.Length; k++) {
                    double freq = fftFreqs[k];
                    double weight = freq < center
                        ? (freq - lower) / (center - lower)
                        : (upper - freq) / (upper - center);
                    filterbank[m, k] = (float)(Math.Max(0, weight) * enorm);
                }
            }
            return filterbank;
        }

        static void ValidateMel(float[,] mel) {
            foreach (var v in mel) {
                if (float.IsNaN(v) || float.IsInfinity(v)) {
                    throw new InvalidOperationException("HifiMelExtractor produced NaN or Inf.");
                }
            }
        }

        static void LogStats(float[,] mel) {
            if (mel.Length == 0) {
                Log.Debug("HifiMelExtractor mel shape=[128,0]");
                return;
            }
            float min = float.PositiveInfinity;
            float max = float.NegativeInfinity;
            double sum = 0;
            foreach (var v in mel) {
                min = Math.Min(min, v);
                max = Math.Max(max, v);
                sum += v;
            }
            Log.Debug("HifiMelExtractor mel shape=[{MelBins},{Frames}] min={Min:F4} max={Max:F4} mean={Mean:F4}",
                mel.GetLength(0), mel.GetLength(1), min, max, sum / mel.Length);
        }
    }
}
