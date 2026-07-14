using System;
using System.Collections.Generic;
using System.Numerics;

namespace OpenUtau.Core.HifiNeural {
    internal static class HifiHnSpectralProcessor {
        const int Nfft = 2048;
        const int Hop = 512;
        const int Bins = Nfft / 2 + 1;
        const double Floor = 1e-12;
        static readonly double[] window = BuildHannWindow();

        public static float[] Process(
            float[] baseline,
            float[] harmonic,
            float[] noise,
            HifiHnSpectralProfile profile) {
            if (!profile.HasAudibleEffect) {
                return baseline;
            }
            return Process(
                baseline,
                harmonic,
                noise,
                HifiHnSpectralProfileTrack.Constant(profile));
        }

        public static float[] Process(
            float[] baseline,
            float[] harmonic,
            float[] noise,
            HifiHnSpectralProfileTrack profileTrack) {
            if (!profileTrack.HasAudibleEffect
                || baseline.Length == 0
                || harmonic.Length != baseline.Length
                || noise.Length != baseline.Length) {
                return baseline;
            }

            float[] shapedHarmonic = profileTrack.NeedsHarmonicProcessing
                ? FilterComponent(harmonic, profileTrack, harmonicComponent: true)
                : harmonic;
            float[] shapedNoise = profileTrack.NeedsNoiseProcessing
                ? FilterComponent(noise, profileTrack, harmonicComponent: false)
                : noise;

            var shaped = new float[baseline.Length];
            for (int i = 0; i < shaped.Length; i++) {
                shaped[i] = FiniteOrZero(shapedHarmonic[i] + shapedNoise[i]);
            }
            return shaped;
        }

        static float[] FilterComponent(
            float[] input,
            HifiHnSpectralProfileTrack profileTrack,
            bool harmonicComponent) {
            int centerPad = Nfft / 2;
            int paddedLength = centerPad * 2 + Math.Max(1, input.Length);
            int frames = Math.Max(1, (int)Math.Ceiling(Math.Max(0, paddedLength - Nfft) / (double)Hop) + 1);
            int requiredLength = (frames - 1) * Hop + Nfft;
            var padded = new float[requiredLength];
            Array.Copy(input, 0, padded, centerPad, input.Length);
            var output = new double[requiredLength];
            var windowSum = new double[requiredLength];
            var fft = new Complex[Nfft];
            double analysisScale = Math.Max(1.0, WindowSum() * 0.5);
            var runtimes = new Dictionary<HifiHnSpectralProfile, ProfileRuntime>();

            for (int frame = 0; frame < frames; frame++) {
                int start = frame * Hop;
                Array.Clear(fft, 0, fft.Length);
                for (int i = 0; i < Nfft; i++) {
                    fft[i] = new Complex(padded[start + i] * window[i], 0);
                }
                HifiHnsepSourceProcessor.ForwardFft(fft, inverse: false);

                var profile = profileTrack.ProfileAtSourceSample(frame * Hop);
                if (profile != null) {
                    if (!runtimes.TryGetValue(profile, out var runtime)) {
                        runtime = new ProfileRuntime(profile);
                        runtimes[profile] = runtime;
                    }
                    bool applyDynamics = harmonicComponent
                        ? runtime.HarmonicDynamics
                        : runtime.NoiseDynamics;
                    if (applyDynamics) {
                        runtime.BeginFrame(frame);
                        Array.Clear(runtime.BandPower, 0, runtime.BandPower.Length);
                        for (int bin = 1; bin < Bins; bin++) {
                            double magnitude = fft[bin].Magnitude;
                            runtime.BandPower[runtime.BandByBin[bin]] += magnitude * magnitude;
                        }
                        for (int band = 0; band < runtime.BandPower.Length; band++) {
                            double level = Math.Sqrt(runtime.BandPower[band]) / analysisScale;
                            double levelDb = 20.0 * Math.Log10(Math.Max(level, Floor));
                            double reduction = 0;
                            if (levelDb > profile.ThresholdDb) {
                                reduction = (profile.ThresholdDb
                                    + (levelDb - profile.ThresholdDb) / profile.Ratio)
                                    - levelDb;
                            }
                            reduction = Math.Clamp(reduction, -profile.MaxReductionDb, 0);
                            double coefficient = reduction < runtime.SmoothedReductionDb[band]
                                ? runtime.Attack
                                : runtime.Release;
                            runtime.SmoothedReductionDb[band] = coefficient * runtime.SmoothedReductionDb[band]
                                + (1.0 - coefficient) * reduction;
                            runtime.BandReductionDb[band] = runtime.SmoothedReductionDb[band];
                        }
                    }

                    for (int bin = 1; bin < Bins; bin++) {
                        double frequency = bin * HifiMelExtractor.SampleRate / (double)Nfft;
                        double balanceDb = runtime.BalanceActive
                            ? InterpolateBands(profile.BalanceDb, profile.FrequenciesHz, frequency)
                            : 0;
                        double componentDb = harmonicComponent ? balanceDb * 0.5 : -balanceDb * 0.5;
                        double dynamicsDb = applyDynamics
                            ? InterpolateBands(runtime.BandReductionDb, profile.FrequenciesHz, frequency)
                            : 0;
                        double gain = Math.Pow(10.0, (componentDb + dynamicsDb) / 20.0);
                        fft[bin] *= gain;
                    }
                    double dcBalanceDb = runtime.BalanceActive
                        ? InterpolateBands(profile.BalanceDb, profile.FrequenciesHz, 0)
                        : 0;
                    double dcComponentDb = harmonicComponent ? dcBalanceDb * 0.5 : -dcBalanceDb * 0.5;
                    double dcDynamicsDb = applyDynamics
                        ? InterpolateBands(runtime.BandReductionDb, profile.FrequenciesHz, 0)
                        : 0;
                    fft[0] *= Math.Pow(10.0, (dcComponentDb + dcDynamicsDb) / 20.0);
                }
                for (int bin = 1; bin < Bins - 1; bin++) {
                    fft[Nfft - bin] = Complex.Conjugate(fft[bin]);
                }
                fft[0] = new Complex(fft[0].Real, 0);
                fft[Bins - 1] = new Complex(fft[Bins - 1].Real, 0);

                HifiHnsepSourceProcessor.ForwardFft(fft, inverse: true);
                for (int i = 0; i < Nfft; i++) {
                    double w = window[i];
                    output[start + i] += fft[i].Real * w;
                    windowSum[start + i] += w * w;
                }
            }

            var result = new float[input.Length];
            for (int i = 0; i < result.Length; i++) {
                int index = i + centerPad;
                double value = windowSum[index] > Floor ? output[index] / windowSum[index] : 0;
                result[i] = FiniteOrZero(value);
            }
            return result;
        }

        sealed class ProfileRuntime {
            public int[] BandByBin { get; }
            public double[] BandPower { get; }
            public double[] BandReductionDb { get; }
            public double[] SmoothedReductionDb { get; }
            public bool BalanceActive { get; }
            public bool HarmonicDynamics { get; }
            public bool NoiseDynamics { get; }
            public double Attack { get; }
            public double Release { get; }
            int lastFrame = -2;

            public ProfileRuntime(HifiHnSpectralProfile profile) {
                BandByBin = BuildBandMap(profile.FrequenciesHz);
                BandPower = new double[profile.FrequenciesHz.Length];
                BandReductionDb = new double[profile.FrequenciesHz.Length];
                SmoothedReductionDb = new double[profile.FrequenciesHz.Length];
                BalanceActive = Array.Exists(profile.BalanceDb, value => Math.Abs(value) >= 0.01);
                HarmonicDynamics = profile.DynamicsEnabled
                    && profile.DynamicsTarget is HifiHnDynamicsTarget.Harmonic or HifiHnDynamicsTarget.Both;
                NoiseDynamics = profile.DynamicsEnabled
                    && profile.DynamicsTarget is HifiHnDynamicsTarget.Noise or HifiHnDynamicsTarget.Both;
                Attack = SmoothingCoefficient(profile.AttackMs);
                Release = SmoothingCoefficient(profile.ReleaseMs);
            }

            public void BeginFrame(int frame) {
                if (frame != lastFrame + 1) {
                    Array.Clear(SmoothedReductionDb, 0, SmoothedReductionDb.Length);
                    Array.Clear(BandReductionDb, 0, BandReductionDb.Length);
                }
                lastFrame = frame;
            }
        }

        static double InterpolateBands(double[] values, double[] frequencies, double frequency) {
            if (frequency <= frequencies[0]) {
                return values[0];
            }
            if (frequency >= frequencies[^1]) {
                return values[^1];
            }
            double logFrequency = Math.Log(Math.Max(1.0, frequency));
            for (int i = 0; i < frequencies.Length - 1; i++) {
                if (frequency <= frequencies[i + 1]) {
                    double left = Math.Log(frequencies[i]);
                    double right = Math.Log(frequencies[i + 1]);
                    double t = Math.Clamp((logFrequency - left) / Math.Max(Floor, right - left), 0, 1);
                    t = SmoothStep(t);
                    return values[i] + (values[i + 1] - values[i]) * t;
                }
            }
            return values[^1];
        }

        static int[] BuildBandMap(double[] frequencies) {
            var result = new int[Bins];
            for (int bin = 0; bin < result.Length; bin++) {
                double frequency = Math.Max(1.0, bin * HifiMelExtractor.SampleRate / (double)Nfft);
                double logFrequency = Math.Log(frequency);
                int closest = 0;
                double bestDistance = double.MaxValue;
                for (int band = 0; band < frequencies.Length; band++) {
                    double distance = Math.Abs(logFrequency - Math.Log(frequencies[band]));
                    if (distance < bestDistance) {
                        closest = band;
                        bestDistance = distance;
                    }
                }
                result[bin] = closest;
            }
            return result;
        }

        static double[] BuildHannWindow() {
            var result = new double[Nfft];
            for (int i = 0; i < result.Length; i++) {
                result[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / Nfft);
            }
            return result;
        }

        static double WindowSum() {
            double sum = 0;
            for (int i = 0; i < window.Length; i++) {
                sum += window[i];
            }
            return sum;
        }

        static double SmoothingCoefficient(double milliseconds) {
            double seconds = Math.Max(0.001, milliseconds / 1000.0);
            return Math.Exp(-Hop / (HifiMelExtractor.SampleRate * seconds));
        }

        static double SmoothStep(double value) {
            value = Math.Clamp(value, 0, 1);
            return value * value * (3.0 - 2.0 * value);
        }

        static float FiniteOrZero(double value) {
            return double.IsFinite(value) ? (float)value : 0f;
        }
    }
}
