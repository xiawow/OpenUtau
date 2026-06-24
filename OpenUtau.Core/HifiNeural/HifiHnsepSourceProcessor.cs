using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using K4os.Hash.xxHash;
using OpenUtau.Core.Render;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiHnsepSourceCache {
        readonly Dictionary<string, HifiHnsepResult?> cache = new(StringComparer.OrdinalIgnoreCase);
        HifiHnsepOnnx? model;
        bool modelResolved;
        bool missingLogged;
        string lastFailureReason = string.Empty;

        public string LastFailureReason => lastFailureReason;

        public bool TryGetSlice(string sourcePath, RenderPhone phone, float[] fullSamples, float[] sourceSlice, out HifiHnsepResult result) {
            result = null!;
            if (fullSamples.Length == 0 || sourceSlice.Length == 0) {
                lastFailureReason = "empty_source";
                return false;
            }
            if (!TryGetOtoSliceBounds(phone, fullSamples.Length, out int sliceStart, out int sliceLength)
                || sliceLength != sourceSlice.Length) {
                lastFailureReason = $"slice_bounds_mismatch:source={fullSamples.Length}:slice={sourceSlice.Length}";
                return false;
            }
            if (!modelResolved) {
                modelResolved = true;
                if (!HifiHnsepOnnx.TryCreate(out model, out var diagnostic)) {
                    Log.Information("Hifi HNSEP disabled: {Diagnostic}", diagnostic);
                    lastFailureReason = "model_unavailable:" + diagnostic;
                    missingLogged = true;
                }
            }
            if (model == null) {
                if (!missingLogged) {
                    Log.Information("Hifi HNSEP disabled: model not available.");
                    missingLogged = true;
                }
                if (string.IsNullOrWhiteSpace(lastFailureReason)) {
                    lastFailureReason = "model_unavailable";
                }
                return false;
            }

            int contextSamples = HifiMelExtractor.SampleRate / 4;
            int contextStart = Math.Max(0, sliceStart - contextSamples);
            int contextEnd = Math.Min(fullSamples.Length, sliceStart + sliceLength + contextSamples);
            int contextLength = Math.Max(0, contextEnd - contextStart);
            if (contextLength <= 0) {
                lastFailureReason = "empty_context";
                return false;
            }

            string key = SliceCacheKey(sourcePath, fullSamples.Length, sliceStart, sliceLength, contextStart, contextLength);
            if (!cache.TryGetValue(key, out var cached)) {
                string diskPath = HifiHnsepDiskCache.GetPath(key);
                if (!HifiHnsepDiskCache.TryLoad(diskPath, sliceLength, out cached)) {
                    try {
                        var context = new float[contextLength];
                        Array.Copy(fullSamples, contextStart, context, 0, contextLength);
                        var separatedContext = model.Separate(context);
                        int cropStart = sliceStart - contextStart;
                        if (separatedContext.Harmonic.Length < cropStart + sliceLength) {
                            throw new InvalidDataException(
                                $"HNSEP context output too short context={separatedContext.Harmonic.Length} cropStart={cropStart} slice={sliceLength}");
                        }
                        var harmonic = new float[sliceLength];
                        Array.Copy(separatedContext.Harmonic, cropStart, harmonic, 0, harmonic.Length);
                        cached = new HifiHnsepResult { Harmonic = harmonic };
                        HifiHnsepDiskCache.TrySave(diskPath, cached);
                        Log.Debug(
                            "Hifi HNSEP slice separated source={Source} slice_start={SliceStart} slice_samples={SliceSamples} context_samples={ContextSamples}",
                            sourcePath,
                            sliceStart,
                            sliceLength,
                            contextLength);
                    } catch (Exception e) {
                        Log.Warning(e, "Hifi HNSEP separation failed source={Source}", sourcePath);
                        lastFailureReason = "separation_failed:" + e.GetType().Name + ":" + e.Message;
                        cached = null;
                    }
                }
                cache[key] = cached;
            }
            if (cached == null || cached.Harmonic.Length != sliceLength) {
                if (cached != null) {
                    lastFailureReason = $"separation_length_mismatch:slice={sliceLength}:harmonic={cached.Harmonic.Length}";
                } else if (string.IsNullOrWhiteSpace(lastFailureReason)) {
                    lastFailureReason = "separation_failed";
                }
                return false;
            }
            lastFailureReason = string.Empty;
            result = cached;
            return true;
        }

        static string SliceCacheKey(string sourcePath, int sourceLength, int sliceStart, int sliceLength, int contextStart, int contextLength) {
            try {
                var info = new FileInfo(sourcePath);
                return string.Concat(
                    info.FullName,
                    "|", info.Length,
                    "|", info.LastWriteTimeUtc.Ticks,
                    "|source_samples=", sourceLength,
                    "|slice=", sliceStart, "+", sliceLength,
                    "|context=", contextStart, "+", contextLength,
                    "|", HifiHnsepOnnx.CacheKeyOrDisabled());
            } catch {
                return string.Concat(
                    sourcePath,
                    "|source_samples=", sourceLength,
                    "|slice=", sliceStart, "+", sliceLength,
                    "|context=", contextStart, "+", contextLength,
                    "|", HifiHnsepOnnx.CacheKeyOrDisabled());
            }
        }

        internal static bool TryGetOtoSliceBounds(RenderPhone phone, int sourceLength, out int offset, out int length) {
            offset = 0;
            length = 0;
            if (sourceLength <= 0 || phone.oto == null) {
                return false;
            }
            offset = Math.Clamp(MsToSamples(phone.oto.Offset), 0, sourceLength);
            int available = Math.Max(0, sourceLength - offset);
            if (available == 0) {
                return false;
            }
            int cutoff = MsToSamples(phone.oto.Cutoff);
            length = cutoff >= 0
                ? available - cutoff
                : Math.Min(available, -cutoff);
            length = Math.Clamp(length, 0, available);
            return length > 0;
        }

        static int MsToSamples(double ms) {
            return (int)Math.Round(ms * HifiMelExtractor.SampleRate / 1000.0);
        }
    }

    public readonly record struct HifiHnsepProcessingReport(
        bool Requested,
        bool Applied,
        string Reason);

    public static class HifiHnsepDiskCache {
        const string Magic = "HNSP1";

        public static string GetPath(string key) {
            string hash = $"{XXH64.DigestOf(Encoding.UTF8.GetBytes(key)):x16}";
            return Path.Combine(PathManager.Inst.CachePath, "hifi-hnsep", $"{hash}.f32");
        }

        public static bool TryLoad(string path, int expectedLength, out HifiHnsepResult? result) {
            result = null;
            try {
                if (!File.Exists(path)) {
                    return false;
                }
                using var stream = File.OpenRead(path);
                using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);
                if (reader.ReadString() != Magic) {
                    return false;
                }
                int length = reader.ReadInt32();
                if (length != expectedLength || length < 0) {
                    return false;
                }
                var harmonic = new float[length];
                for (int i = 0; i < harmonic.Length; i++) {
                    harmonic[i] = reader.ReadSingle();
                }
                result = new HifiHnsepResult { Harmonic = harmonic };
                Log.Debug("Hifi HNSEP source cache hit path={Path} samples={Samples}", path, length);
                return true;
            } catch (Exception e) {
                Log.Warning(e, "Failed to read Hifi HNSEP source cache path={Path}", path);
                result = null;
                return false;
            }
        }

        public static void TrySave(string path, HifiHnsepResult? result) {
            if (result == null || result.Harmonic.Length == 0) {
                return;
            }
            try {
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                string tempPath = path + ".tmp";
                using (var stream = File.Open(tempPath, FileMode.Create, FileAccess.Write, FileShare.None))
                using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false)) {
                    writer.Write(Magic);
                    writer.Write(result.Harmonic.Length);
                    for (int i = 0; i < result.Harmonic.Length; i++) {
                        float value = result.Harmonic[i];
                        writer.Write(float.IsFinite(value) ? value : 0f);
                    }
                }
                if (File.Exists(path)) {
                    File.Delete(path);
                }
                File.Move(tempPath, path);
                Log.Debug("Hifi HNSEP source cache saved path={Path} samples={Samples}", path, result.Harmonic.Length);
            } catch (Exception e) {
                Log.Warning(e, "Failed to write Hifi HNSEP source cache path={Path}", path);
            }
        }
    }

    public static class HifiHnsepSourceProcessor {
        const int TensionNfft = HifiMelExtractor.Nfft;
        const int TensionHop = HifiOnnxVocoder.HopSize;

        /// <summary>
        /// Applies HNSEP-based harmonic/noise controls to a synthesized waveform.
        /// This is shared by HIFI-NEURA source processing and neural renderers that
        /// produce a waveform directly instead of sampling a source oto.
        /// </summary>
        public static float[] ApplyGeneratedWaveform(
            float[] waveform,
            HifiFrameParameterTrack parameterTrack,
            string? separationCacheKey,
            out HifiHnsepProcessingReport report) {
            if (!parameterTrack.NeedsHnsep || waveform.Length == 0) {
                report = new HifiHnsepProcessingReport(false, false, "neutral_parameters_or_empty_waveform");
                return waveform;
            }

            try {
                HifiHnsepResult? separated = null;
                string? diskPath = null;
                if (!string.IsNullOrWhiteSpace(separationCacheKey)) {
                    diskPath = HifiHnsepDiskCache.GetPath(separationCacheKey);
                    HifiHnsepDiskCache.TryLoad(diskPath, waveform.Length, out separated);
                }
                if (separated == null) {
                    if (!HifiHnsepOnnx.TryCreate(out var model, out var diagnostic) || model == null) {
                        report = new HifiHnsepProcessingReport(true, false, "no_model_or_separation_failed");
                        Log.Debug(
                            "HNSEP waveform parameters skipped brec={Brec:F2} voic={Voic:F2} tenc={Tenc:F2} reason={Reason}",
                            parameterTrack.Average.Breathiness,
                            parameterTrack.Average.Voicing,
                            parameterTrack.Average.Tension,
                            diagnostic);
                        return waveform;
                    }
                    separated = model.Separate(waveform);
                    if (diskPath != null) {
                        HifiHnsepDiskCache.TrySave(diskPath, separated);
                    }
                }
                if (separated.Harmonic.Length != waveform.Length) {
                    report = new HifiHnsepProcessingReport(true, false, "waveform_length_mismatch");
                    Log.Warning(
                        "HNSEP waveform length mismatch source={SourceLength} harmonic={HarmonicLength}",
                        waveform.Length,
                        separated.Harmonic.Length);
                    return waveform;
                }

                report = new HifiHnsepProcessingReport(true, true, "applied");
                return RemixSeparatedWaveform(waveform, separated.Harmonic, parameterTrack);
            } catch (Exception e) {
                report = new HifiHnsepProcessingReport(true, false, "separation_failed");
                Log.Warning(e, "HNSEP waveform separation failed");
                return waveform;
            }
        }

        public static float[] Apply(
            RenderPhone phone,
            string sourcePath,
            float[] fullSamples,
            float[] sourceSlice,
            HifiFrameParameterAverages parameters,
            HifiHnsepSourceCache cache,
            out HifiHnsepProcessingReport report) {
            return Apply(
                phone,
                sourcePath,
                fullSamples,
                sourceSlice,
                HifiFrameParameterTrack.Constant(parameters),
                cache,
                out report);
        }

        public static float[] Apply(
            RenderPhone phone,
            string sourcePath,
            float[] fullSamples,
            float[] sourceSlice,
            HifiFrameParameterTrack parameterTrack,
            HifiHnsepSourceCache cache,
            out HifiHnsepProcessingReport report) {
            var parameters = parameterTrack.Average;
            if (!parameterTrack.NeedsHnsep || sourceSlice.Length == 0) {
                report = new HifiHnsepProcessingReport(false, false, "neutral_parameters_or_empty_slice");
                return sourceSlice;
            }
            if (!cache.TryGetSlice(sourcePath, phone, fullSamples, sourceSlice, out var separated)) {
                string reason = string.IsNullOrWhiteSpace(cache.LastFailureReason)
                    ? "no_model_or_separation_failed"
                    : cache.LastFailureReason;
                Log.Debug(
                    "Hifi HNSEP parameters skipped phoneme={Phoneme} brec={Brec:F2} voic={Voic:F2} tenc={Tenc:F2} reason={Reason}",
                    phone.phoneme,
                    parameters.Breathiness,
                    parameters.Voicing,
                    parameters.Tension,
                    reason);
                report = new HifiHnsepProcessingReport(true, false, reason);
                return sourceSlice;
            }

            if (separated.Harmonic.Length != sourceSlice.Length) {
                Log.Warning(
                    "Hifi HNSEP slice length mismatch phoneme={Phoneme} source={SourceLength} harmonic={HarmonicLength}",
                    phone.phoneme,
                    sourceSlice.Length,
                    separated.Harmonic.Length);
                report = new HifiHnsepProcessingReport(true, false, "slice_length_mismatch");
                return sourceSlice;
            }

            var output = RemixSeparatedWaveform(sourceSlice, separated.Harmonic, parameterTrack);
            Log.Debug(
                "Hifi HNSEP applied phoneme={Phoneme} mode=source_frame_aware frames={Frames} brec_avg={Brec:F2} noise_gain_avg={NoiseGain:F3} voic_avg={Voic:F2} harmonic_gain_avg={HarmonicGain:F3} tenc_avg={Tenc:F2}",
                phone.phoneme,
                parameterTrack.FrameCount,
                parameters.Breathiness,
                parameters.BreathNoiseGain,
                parameters.Voicing,
                parameters.VoicingGain,
                parameters.Tension);
            report = new HifiHnsepProcessingReport(true, true, "applied");
            return output;
        }

        static float[] RemixSeparatedWaveform(
            float[] waveform,
            float[] originalHarmonic,
            HifiFrameParameterTrack parameterTrack) {
            float[] processedHarmonic = PrepareHarmonicForRemix(originalHarmonic, parameterTrack);
            return RemixHarmonicNoiseWithSourceEnergy(
                waveform,
                originalHarmonic,
                processedHarmonic,
                parameterTrack);
        }

        internal static float[] PrepareHarmonicForRemix(float[] cachedHarmonic, double tension) {
            return PrepareHarmonicForRemix(cachedHarmonic, HifiFrameParameterTrack.Constant(new HifiFrameParameterAverages(0, 0, tension, 100)));
        }

        internal static float[] PrepareHarmonicForRemix(float[] cachedHarmonic, HifiFrameParameterTrack parameterTrack) {
            var harmonic = new float[cachedHarmonic.Length];
            Array.Copy(cachedHarmonic, harmonic, cachedHarmonic.Length);
            if (parameterTrack.HasTension) {
                ApplyTensionInPlace(harmonic, parameterTrack);
            }
            return harmonic;
        }

        internal static float[] RemixHarmonicNoiseWithSourceEnergy(float[] sourceSlice, float[] harmonic, double noiseGain, double harmonicGain) {
            return RemixHarmonicNoiseWithSourceEnergy(
                sourceSlice,
                harmonic,
                HifiFrameParameterTrack.Constant(new HifiFrameParameterAverages(0, (noiseGain - 1.0) / 0.02, 0, harmonicGain * 100.0)));
        }

        internal static float[] RemixHarmonicNoiseWithSourceEnergy(float[] sourceSlice, float[] harmonic, HifiFrameParameterTrack parameterTrack) {
            return RemixHarmonicNoiseWithSourceEnergy(sourceSlice, harmonic, harmonic, parameterTrack);
        }

        internal static float[] RemixHarmonicNoiseWithSourceEnergy(
            float[] sourceSlice,
            float[] originalHarmonic,
            float[] processedHarmonic,
            HifiFrameParameterTrack parameterTrack) {
            var output = new float[sourceSlice.Length];
            for (int i = 0; i < output.Length; i++) {
                double original = i < originalHarmonic.Length ? originalHarmonic[i] : 0;
                double processed = i < processedHarmonic.Length ? processedHarmonic[i] : original;
                double noise = sourceSlice[i] - original;
                double noiseGain = parameterTrack.BreathNoiseGainAtSourceSample(i, sourceSlice.Length);
                double harmonicGain = parameterTrack.VoicingGainAtSourceSample(i, sourceSlice.Length);
                output[i] = (float)(noise * noiseGain + processed * harmonicGain);
            }
            MatchRmsInPlace(output, sourceSlice);
            LimitPeakInPlace(output);
            return output;
        }

        static void ApplyTensionInPlace(float[] harmonic, double tension) {
            ApplyTensionInPlace(harmonic, HifiFrameParameterTrack.Constant(new HifiFrameParameterAverages(0, 0, tension, 100)));
        }

        static void ApplyTensionInPlace(float[] harmonic, HifiFrameParameterTrack parameterTrack) {
            if (!parameterTrack.HasTension || harmonic.Length <= 1) {
                return;
            }
            var filtered = ApplyHifisamplerStyleSpectralTension(harmonic, parameterTrack);
            for (int i = 0; i < harmonic.Length; i++) {
                harmonic[i] = filtered[i];
            }
            LimitPeakInPlace(harmonic);
        }

        static float[] ApplyHifisamplerStyleSpectralTension(float[] wave, double b) {
            double tension = -b * 50.0;
            return ApplyHifisamplerStyleSpectralTension(
                wave,
                HifiFrameParameterTrack.Constant(new HifiFrameParameterAverages(0, 0, tension, 100)));
        }

        static float[] ApplyHifisamplerStyleSpectralTension(float[] wave, HifiFrameParameterTrack parameterTrack) {
            if (!parameterTrack.HasTension) {
                var copy = new float[wave.Length];
                Array.Copy(wave, copy, wave.Length);
                return copy;
            }
            int originalLength = wave.Length;
            int centerPad = TensionNfft / 2;
            int paddedLength = centerPad * 2 + Math.Max(1, originalLength);
            int frames = Math.Max(1, 1 + Math.Max(0, paddedLength - TensionNfft) / TensionHop);
            int requiredLength = (frames - 1) * TensionHop + TensionNfft;
            var padded = new float[requiredLength];
            Array.Copy(wave, 0, padded, centerPad, wave.Length);

            var output = new double[requiredLength];
            var windowSum = new double[requiredLength];
            var fft = new Complex[TensionNfft];
            var window = TensionWindow.Value;
            int bins = TensionNfft / 2 + 1;
            double x0 = bins / ((HifiMelExtractor.SampleRate / 2.0) / 1500.0);

            for (int frame = 0; frame < frames; frame++) {
                Array.Clear(fft, 0, fft.Length);
                int start = frame * TensionHop;
                for (int i = 0; i < TensionNfft; i++) {
                    fft[i] = new Complex(padded[start + i] * window[i], 0);
                }
                ForwardFft(fft, inverse: false);

                double frameCenterSample = Math.Clamp(start + TensionNfft * 0.5 - centerPad, 0, Math.Max(0, originalLength - 1));
                double tension = parameterTrack.TensionAtSourceSample(frameCenterSample, originalLength);
                double b = -Math.Clamp(tension, -100.0, 100.0) / 50.0;
                for (int k = 0; k < bins; k++) {
                    double amp = fft[k].Magnitude;
                    double filter = Math.Clamp((-b / x0) * k + b, -2.0, 2.0);
                    double newAmp = Math.Exp(Math.Log(Math.Max(amp, 1e-9)) + filter);
                    double phase = Math.Atan2(fft[k].Imaginary, fft[k].Real);
                    fft[k] = Complex.FromPolarCoordinates(newAmp, phase);
                }
                for (int k = 1; k < bins - 1; k++) {
                    fft[TensionNfft - k] = Complex.Conjugate(fft[k]);
                }

                ForwardFft(fft, inverse: true);
                for (int i = 0; i < TensionNfft; i++) {
                    double w = window[i];
                    output[start + i] += fft[i].Real * w;
                    windowSum[start + i] += w * w;
                }
            }

            var result = new float[originalLength];
            for (int i = 0; i < result.Length; i++) {
                int src = i + centerPad;
                double value = output[src];
                if (windowSum[src] > 1e-9) {
                    value /= windowSum[src];
                }
                result[i] = (float)value;
            }

            double originalMax = Peak(wave);
            double filteredMax = Peak(result);
            if (originalMax > 1e-8 && filteredMax > 1e-8) {
                double averageB = -Math.Clamp(parameterTrack.Average.Tension, -100.0, 100.0) / 50.0;
                double extraGain = Math.Clamp(averageB / -15.0, 0.0, 0.33) + 1.0;
                double gain = originalMax / filteredMax * extraGain;
                for (int i = 0; i < result.Length; i++) {
                    result[i] = (float)(result[i] * gain);
                }
            }
            return result;
        }

        static readonly Lazy<float[]> TensionWindow = new(() => {
            var window = new float[TensionNfft];
            for (int i = 0; i < window.Length; i++) {
                window[i] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / window.Length));
            }
            return window;
        });

        static void ForwardFft(Complex[] buffer, bool inverse) {
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
                double angle = 2.0 * Math.PI / len * (inverse ? 1 : -1);
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
            if (inverse) {
                for (int i = 0; i < n; i++) {
                    buffer[i] /= n;
                }
            }
        }

        static double Peak(float[] samples) {
            double peak = 0;
            for (int i = 0; i < samples.Length; i++) {
                peak = Math.Max(peak, Math.Abs(samples[i]));
            }
            return peak;
        }

        static double Rms(float[] samples) {
            if (samples.Length == 0) {
                return 0;
            }
            double sum = 0;
            int count = 0;
            for (int i = 0; i < samples.Length; i++) {
                float value = samples[i];
                if (!float.IsFinite(value)) {
                    continue;
                }
                sum += value * value;
                count++;
            }
            return count > 0 ? Math.Sqrt(sum / count) : 0;
        }

        static void MatchRmsInPlace(float[] samples, float[] reference) {
            double targetRms = Rms(reference);
            double currentRms = Rms(samples);
            if (targetRms <= 1e-5 || currentRms <= 1e-5) {
                return;
            }
            double gain = Math.Clamp(targetRms / currentRms, 0.5, 2.0);
            if (!double.IsFinite(gain) || Math.Abs(gain - 1.0) < 1e-4) {
                return;
            }
            for (int i = 0; i < samples.Length; i++) {
                samples[i] = (float)(samples[i] * gain);
            }
        }

        static void LimitPeakInPlace(float[] samples) {
            double peak = 0;
            for (int i = 0; i < samples.Length; i++) {
                if (float.IsNaN(samples[i]) || float.IsInfinity(samples[i])) {
                    samples[i] = 0;
                    continue;
                }
                peak = Math.Max(peak, Math.Abs(samples[i]));
            }
            if (peak <= 1.0 || peak <= 1e-9) {
                return;
            }
            double gain = 1.0 / peak;
            for (int i = 0; i < samples.Length; i++) {
                samples[i] = (float)(samples[i] * gain);
            }
        }
    }
}
