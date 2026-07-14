using System;
using System.IO;
using System.Numerics;

namespace OpenUtau.Core.Neutrino {
    internal sealed class NeutrinoLegacyTimingModel {
        const int rawFeatureSize = 243;
        const int phonemeClasses = 40;
        const int embeddedPhonemeSize = 8;
        const int embeddedFeatureSize = 83;
        const int hiddenSize = 1024;
        const int recurrentStateSize = 5;
        const int recurrentHiddenSize = 64;
        const int finalFeatureSize = hiddenSize + recurrentHiddenSize;

        readonly Layer[] layers;
        readonly float[] inputMean;
        readonly float[] inputStd;
        readonly float outputMean;
        readonly float outputStd;

        NeutrinoLegacyTimingModel(
            Layer[] layers,
            float[] inputMean,
            float[] inputStd,
            float outputMean,
            float outputStd) {

            this.layers = layers;
            this.inputMean = inputMean;
            this.inputStd = inputStd;
            this.outputMean = outputMean;
            this.outputStd = outputStd;
        }

        public static NeutrinoLegacyTimingModel Load(string modelPath, string statsPath) {
            var layers = LoadModel(modelPath);
            var (inputMean, inputStd, outputMean, outputStd) = LoadStats(statsPath);
            return new NeutrinoLegacyTimingModel(layers, inputMean, inputStd, outputMean, outputStd);
        }

        public float[] PredictDeltasMs(float[] rawFeatures, int phones) {
            if (phones <= 0) {
                return Array.Empty<float>();
            }
            if (rawFeatures.Length != phones * rawFeatureSize) {
                throw new ArgumentException(
                    $"NEUTRINO v2 timing feature length mismatch: {rawFeatures.Length} != {phones} * {rawFeatureSize}");
            }

            var normalized = new float[rawFeatures.Length];
            for (int phone = 0; phone < phones; phone++) {
                int offset = phone * rawFeatureSize;
                for (int i = 0; i < rawFeatureSize; i++) {
                    float scale = inputStd[i];
                    normalized[offset + i] = scale == 0
                        ? rawFeatures[offset + i] - inputMean[i]
                        : (rawFeatures[offset + i] - inputMean[i]) / scale;
                }
            }

            var baseHidden = new float[phones * hiddenSize];
            var embedded = new float[embeddedFeatureSize];
            var gateA = new float[hiddenSize];
            var gateB = new float[hiddenSize];
            var hidden = new float[hiddenSize];
            for (int phone = 0; phone < phones; phone++) {
                BuildEmbeddedInput(normalized, phone, embedded);
                Gated(layers[2], layers[1], embedded, hidden);
                ResidualGated(layers[4], layers[3], hidden);
                ResidualGated(layers[6], layers[5], hidden);
                ResidualGated(layers[8], layers[7], hidden);
                ResidualGated(layers[10], layers[9], hidden);
                ResidualGated(layers[12], layers[11], hidden);
                Array.Copy(hidden, 0, baseHidden, phone * hiddenSize, hiddenSize);
            }

            var state = new float[recurrentStateSize];
            var stateHidden = new float[recurrentHiddenSize];
            var finalInput = new float[finalFeatureSize];
            var output = new float[phones];
            float zeroDeltaNorm = (0 - outputMean) / outputStd;
            for (int phone = 0; phone < phones; phone++) {
                Gated(layers[14], layers[13], state, stateHidden);
                Gated(layers[16], layers[15], stateHidden, stateHidden);

                Array.Copy(baseHidden, phone * hiddenSize, hidden, 0, hiddenSize);
                Array.Copy(hidden, 0, finalInput, 0, hiddenSize);
                Array.Copy(stateHidden, 0, finalInput, hiddenSize, recurrentHiddenSize);
                FinalResidualGated(layers[18], layers[17], hidden, finalInput);
                FinalResidualGated(layers[20], layers[19], hidden, finalInput);
                FinalResidualGated(layers[22], layers[21], hidden, finalInput);

                Linear(layers[23], hidden, gateA);
                float yNorm = gateA[0];
                if (rawFeatures[phone * rawFeatureSize + 111] == 1f) {
                    yNorm = zeroDeltaNorm;
                }
                output[phone] = yNorm * outputStd + outputMean;

                Array.Copy(state, 1, state, 0, recurrentStateSize - 1);
                state[recurrentStateSize - 1] = yNorm;
            }
            return output;
        }

        void BuildEmbeddedInput(float[] normalized, int phone, float[] embedded) {
            int rawOffset = phone * rawFeatureSize;
            for (int group = 0; group < 5; group++) {
                var layer = layers[0];
                int inputOffset = rawOffset + group * phonemeClasses;
                int outputOffset = group * embeddedPhonemeSize;
                for (int o = 0; o < embeddedPhonemeSize; o++) {
                    float sum = layer.Bias[o];
                    int weightOffset = o * phonemeClasses;
                    for (int i = 0; i < phonemeClasses; i++) {
                        sum += layer.Weight[weightOffset + i] * normalized[inputOffset + i];
                    }
                    embedded[outputOffset + o] = sum;
                }
            }
            Array.Copy(
                normalized,
                rawOffset + 5 * phonemeClasses,
                embedded,
                5 * embeddedPhonemeSize,
                rawFeatureSize - 5 * phonemeClasses);
        }

        static void Gated(Layer tanhLayer, Layer sigmoidLayer, float[] input, float[] output) {
            var a = new float[tanhLayer.OutputSize];
            var b = new float[sigmoidLayer.OutputSize];
            Linear(tanhLayer, input, a);
            Linear(sigmoidLayer, input, b);
            for (int i = 0; i < output.Length; i++) {
                output[i] = MathF.Tanh(a[i]) * Sigmoid(b[i]);
            }
        }

        static void ResidualGated(Layer tanhLayer, Layer sigmoidLayer, float[] hidden) {
            var update = new float[hidden.Length];
            Gated(tanhLayer, sigmoidLayer, hidden, update);
            for (int i = 0; i < hidden.Length; i++) {
                hidden[i] += update[i];
            }
        }

        static void FinalResidualGated(Layer tanhLayer, Layer sigmoidLayer, float[] hidden, float[] finalInput) {
            Array.Copy(hidden, 0, finalInput, 0, hidden.Length);
            var update = new float[hidden.Length];
            Gated(tanhLayer, sigmoidLayer, finalInput, update);
            for (int i = 0; i < hidden.Length; i++) {
                hidden[i] += update[i];
            }
        }

        static void Linear(Layer layer, float[] input, float[] output) {
            for (int o = 0; o < layer.OutputSize; o++) {
                float sum = layer.Bias[o];
                sum += Dot(layer.Weight, o * layer.InputSize, input, 0, layer.InputSize);
                output[o] = sum;
            }
        }

        static float Dot(float[] a, int aOffset, float[] b, int bOffset, int length) {
            int width = Vector<float>.Count;
            int i = 0;
            var acc = Vector<float>.Zero;
            for (; i <= length - width; i += width) {
                acc += new Vector<float>(a, aOffset + i) * new Vector<float>(b, bOffset + i);
            }
            float sum = 0;
            for (int j = 0; j < width; j++) {
                sum += acc[j];
            }
            for (; i < length; i++) {
                sum += a[aOffset + i] * b[bOffset + i];
            }
            return sum;
        }

        static float Sigmoid(float value) {
            if (value >= 0) {
                float z = MathF.Exp(-value);
                return 1f / (1f + z);
            } else {
                float z = MathF.Exp(value);
                return z / (1f + z);
            }
        }

        static Layer[] LoadModel(string path) {
            var values = ReadSingleFile(path);
            if (values.Length < 538 || Math.Abs(values[0] - 1234f) > 0.001f || Math.Abs(values[1] - 512f) > 0.001f) {
                throw new InvalidDataException($"Invalid NEUTRINO v2 timing model header: {path}");
            }

            var shapes = new[] {
                (8, 40),
                (1024, 83),
                (1024, 83),
                (1024, 1024),
                (1024, 1024),
                (1024, 1024),
                (1024, 1024),
                (1024, 1024),
                (1024, 1024),
                (1024, 1024),
                (1024, 1024),
                (1024, 1024),
                (1024, 1024),
                (64, 5),
                (64, 5),
                (64, 64),
                (64, 64),
                (1024, 1088),
                (1024, 1088),
                (1024, 1088),
                (1024, 1088),
                (1024, 1088),
                (1024, 1088),
                (1, 1024),
            };
            var layers = new Layer[shapes.Length];
            int offset = 538;
            for (int i = 0; i < shapes.Length; i++) {
                var (outputSize, inputSize) = shapes[i];
                int weightCount = outputSize * inputSize;
                int required = weightCount + outputSize;
                if (offset + required > values.Length) {
                    throw new InvalidDataException($"NEUTRINO v2 timing model ended while reading layer {i}: {path}");
                }
                var weight = new float[weightCount];
                for (int input = 0; input < inputSize; input++) {
                    for (int output = 0; output < outputSize; output++) {
                        weight[output * inputSize + input] = values[offset + input * outputSize + output];
                    }
                }
                offset += weightCount;
                var bias = new float[outputSize];
                Array.Copy(values, offset, bias, 0, bias.Length);
                offset += bias.Length;
                layers[i] = new Layer(inputSize, outputSize, weight, bias);
            }
            if (offset != values.Length) {
                throw new InvalidDataException(
                    $"Unexpected NEUTRINO v2 timing model payload size: consumed {offset}, file has {values.Length} floats.");
            }
            return layers;
        }

        static (float[] inputMean, float[] inputStd, float outputMean, float outputStd) LoadStats(string path) {
            var values = ReadSingleFile(path);
            if (values.Length < 492
                || (int)MathF.Round(values[0]) != rawFeatureSize
                || (int)MathF.Round(values[1]) != rawFeatureSize
                || (int)MathF.Round(values[2]) != 1
                || (int)MathF.Round(values[3]) != 1) {
                throw new InvalidDataException($"Invalid NEUTRINO v2 timing stats header: {path}");
            }

            var inputMean = new float[rawFeatureSize];
            var inputStd = new float[rawFeatureSize];
            Array.Copy(values, 4, inputMean, 0, rawFeatureSize);
            Array.Copy(values, 4 + rawFeatureSize, inputStd, 0, rawFeatureSize);
            float outputMean = values[4 + rawFeatureSize * 2];
            float outputStd = MathF.Sqrt(values[5 + rawFeatureSize * 2]);
            return (inputMean, inputStd, outputMean, outputStd);
        }

        static float[] ReadSingleFile(string path) {
            var bytes = File.ReadAllBytes(path);
            if (bytes.Length % sizeof(float) != 0) {
                throw new InvalidDataException($"File length is not float32-aligned: {path}");
            }
            var values = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        sealed class Layer {
            public int InputSize { get; }
            public int OutputSize { get; }
            public float[] Weight { get; }
            public float[] Bias { get; }

            public Layer(int inputSize, int outputSize, float[] weight, float[] bias) {
                InputSize = inputSize;
                OutputSize = outputSize;
                Weight = weight;
                Bias = bias;
            }
        }
    }
}
