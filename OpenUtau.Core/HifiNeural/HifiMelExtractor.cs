using System;
using System.Collections.Generic;
using System.Numerics;
using System.Linq;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using OpenUtau.Core.Format;
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

        readonly float[] hann;
        readonly float[,] melFilterbank;

        public HifiMelExtractor() {
            hann = Enumerable.Range(0, WinSize)
                .Select(i => (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / WinSize)))
                .ToArray();
            melFilterbank = BuildMelFilterbank();
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
            return Wave.GetSamples(provider).Select(s => Math.Clamp(s, -1f, 1f)).ToArray();
        }

        public float[,] ExtractFromFile(string path) {
            return Extract(LoadMono(path));
        }

        public float[,] Extract(float[] samples) {
            if (samples.Length == 0) {
                return new float[NMels, 0];
            }

            var padded = ReflectPad(samples, (WinSize - OriginHopSize) / 2, (WinSize - OriginHopSize + 1) / 2);
            int frames = Math.Max(1, 1 + Math.Max(0, padded.Length - WinSize) / OriginHopSize);
            var mel = new float[NMels, frames];
            var fft = new Complex[Nfft];

            for (int frame = 0; frame < frames; frame++) {
                int start = frame * OriginHopSize;
                Array.Clear(fft, 0, fft.Length);
                for (int n = 0; n < WinSize; n++) {
                    fft[n] = new Complex(padded[start + n] * hann[n], 0);
                }
                ForwardFft(fft);

                for (int m = 0; m < NMels; m++) {
                    double value = 0;
                    for (int k = 0; k <= Nfft / 2; k++) {
                        value += melFilterbank[m, k] * fft[k].Magnitude;
                    }
                    mel[m, frame] = (float)Math.Log(Math.Max(value, 1e-5));
                }
            }

            LogStats(mel);
            return mel;
        }

        public float[,] ExtractAtPositions(float[] samples, IReadOnlyList<double> centerSamplePositions) {
            int frames = centerSamplePositions.Count;
            var mel = new float[NMels, frames];
            if (frames == 0) {
                LogStats(mel);
                return mel;
            }
            if (samples.Length == 0) {
                FillConstant(mel, (float)Math.Log(1e-5));
                LogStats(mel);
                return mel;
            }

            var fft = new Complex[Nfft];
            for (int frame = 0; frame < frames; frame++) {
                double center = centerSamplePositions[frame];
                if (double.IsNaN(center) || double.IsInfinity(center)) {
                    center = 0;
                }
                center = Math.Clamp(center, 0, samples.Length - 1);
                double start = center - (WinSize - 1) * 0.5;

                Array.Clear(fft, 0, fft.Length);
                for (int n = 0; n < WinSize; n++) {
                    fft[n] = new Complex(SampleReflectedLinear(samples, start + n) * hann[n], 0);
                }
                ForwardFft(fft);
                WriteMelFrame(fft, mel, frame);
            }

            Log.Information(
                "HifiMelExtractor variable_position_mel shape=[{MelBins},{Frames}] source_samples={SourceSamples}",
                NMels,
                frames,
                samples.Length);
            LogStats(mel);
            return mel;
        }

        void WriteMelFrame(Complex[] fft, float[,] mel, int frame) {
            for (int m = 0; m < NMels; m++) {
                double value = 0;
                for (int k = 0; k <= Nfft / 2; k++) {
                    value += melFilterbank[m, k] * fft[k].Magnitude;
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

        static void ForwardFft(Complex[] buffer) {
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
            for (int len = 2; len <= n; len <<= 1) {
                double angle = -2.0 * Math.PI / len;
                var wlen = new Complex(Math.Cos(angle), Math.Sin(angle));
                for (int i = 0; i < n; i += len) {
                    var w = Complex.One;
                    for (int j = 0; j < len / 2; j++) {
                        var u = buffer[i + j];
                        var v = buffer[i + j + len / 2] * w;
                        buffer[i + j] = u + v;
                        buffer[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
            }
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

        static void LogStats(float[,] mel) {
            if (mel.Length == 0) {
                Log.Information("HifiMelExtractor mel shape=[128,0]");
                return;
            }
            float min = float.PositiveInfinity;
            float max = float.NegativeInfinity;
            double sum = 0;
            foreach (var v in mel) {
                if (float.IsNaN(v) || float.IsInfinity(v)) {
                    throw new InvalidOperationException("HifiMelExtractor produced NaN or Inf.");
                }
                min = Math.Min(min, v);
                max = Math.Max(max, v);
                sum += v;
            }
            Log.Information("HifiMelExtractor mel shape=[{MelBins},{Frames}] min={Min:F4} max={Max:F4} mean={Mean:F4}",
                mel.GetLength(0), mel.GetLength(1), min, max, sum / mel.Length);
        }
    }
}
