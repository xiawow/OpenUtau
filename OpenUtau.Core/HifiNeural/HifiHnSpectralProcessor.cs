using System;
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
            if (!profile.HasAudibleEffect
                || baseline.Length == 0
                || harmonic.Length != baseline.Length
                || noise.Length != baseline.Length) {
                return baseline;
            }
            profile = profile.Clone().Normalize();
            int[] bandByBin = BuildBandMap(profile.FrequenciesHz);

            bool balanceActive = false;
            for (int i = 0; i < profile.BalanceDb.Length; i++) {
                balanceActive |= Math.Abs(profile.BalanceDb[i]) >= 0.01;
            }
            bool harmonicDynamics = profile.DynamicsEnabled
                && profile.DynamicsTarget is HifiHnDynamicsTarget.Harmonic or HifiHnDynamicsTarget.Both;
            bool noiseDynamics = profile.DynamicsEnabled
                && profile.DynamicsTarget is HifiHnDynamicsTarget.Noise or HifiHnDynamicsTarget.Both;

            float[] shapedHarmonic = balanceActive || harmonicDynamics
                ? FilterComponent(harmonic, profile, bandByBin, harmonicComponent: true, harmonicDynamics)
                : harmonic;
            float[] shapedNoise = balanceActive || noiseDynamics
                ? FilterComponent(noise, profile, bandByBin, harmonicComponent: false, noiseDynamics)
                : noise;

            var shaped = new float[baseline.Length];
            for (int i = 0; i < shaped.Length; i++) {
                shaped[i] = FiniteOrZero(shapedHarmonic[i] + shapedNoise[i]);
            }
            return shaped;
        }

        static float[] FilterComponent(
            float[] input,
            HifiHnSpectralProfile profile,
            int[] bandByBin,
            bool harmonicComponent,
            bool applyDynamics) {
            int centerPad = Nfft / 2;
            int paddedLength = centerPad * 2 + Math.Max(1, input.Length);
            int frames = Math.Max(1, (int)Math.Ceiling(Math.Max(0, paddedLength - Nfft) / (double)Hop) + 1);
            int requiredLength = (frames - 1) * Hop + Nfft;
            var padded = new float[requiredLength];
            Array.Copy(input, 0, padded, centerPad, input.Length);
            var output = new double[requiredLength];
            var windowSum = new double[requiredLength];
            var fft = new Complex[Nfft];
            var bandPower = new double[HifiHnSpectralProfile.BandCount];
            var bandReductionDb = new double[HifiHnSpectralProfile.BandCount];
            var smoothedReductionDb = new double[HifiHnSpectralProfile.BandCount];
            double analysisScale = Math.Max(1.0, WindowSum() * 0.5);
            double attack = SmoothingCoefficient(profile.AttackMs);
            double release = SmoothingCoefficient(profile.ReleaseMs);

            for (int frame = 0; frame < frames; frame++) {
                int start = frame * Hop;
                Array.Clear(fft, 0, fft.Length);
                for (int i = 0; i < Nfft; i++) {
                    fft[i] = new Complex(padded[start + i] * window[i], 0);
                }
                HifiHnsepSourceProcessor.ForwardFft(fft, inverse: false);

                if (applyDynamics) {
                    Array.Clear(bandPower, 0, bandPower.Length);
                    for (int bin = 1; bin < Bins; bin++) {
                        double magnitude = fft[bin].Magnitude;
                        bandPower[bandByBin[bin]] += magnitude * magnitude;
                    }
                    for (int band = 0; band < bandPower.Length; band++) {
                        double level = Math.Sqrt(bandPower[band]) / analysisScale;
                        double levelDb = 20.0 * Math.Log10(Math.Max(level, Floor));
                        double reduction = 0;
                        if (levelDb > profile.ThresholdDb) {
                            reduction = (profile.ThresholdDb
                                + (levelDb - profile.ThresholdDb) / profile.Ratio)
                                - levelDb;
                        }
                        reduction = Math.Clamp(reduction, -profile.MaxReductionDb, 0);
                        double coefficient = reduction < smoothedReductionDb[band] ? attack : release;
                        smoothedReductionDb[band] = coefficient * smoothedReductionDb[band]
                            + (1.0 - coefficient) * reduction;
                        bandReductionDb[band] = smoothedReductionDb[band];
                    }
                } else {
                    Array.Clear(bandReductionDb, 0, bandReductionDb.Length);
                }

                for (int bin = 0; bin < Bins; bin++) {
                    double frequency = bin * HifiMelExtractor.SampleRate / (double)Nfft;
                    double balanceDb = InterpolateBands(
                        profile.BalanceDb,
                        profile.FrequenciesHz,
                        frequency);
                    double componentDb = harmonicComponent ? balanceDb * 0.5 : -balanceDb * 0.5;
                    double dynamicsDb = applyDynamics
                        ? InterpolateBands(bandReductionDb, profile.FrequenciesHz, frequency)
                        : 0;
                    double gain = Math.Pow(10.0, (componentDb + dynamicsDb) / 20.0);
                    fft[bin] *= gain;
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
