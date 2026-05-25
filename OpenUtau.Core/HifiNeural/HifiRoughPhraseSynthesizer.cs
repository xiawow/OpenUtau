using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using NAudio.Wave;
using OpenUtau.Classic;
using OpenUtau.Core.Format;
using OpenUtau.Core.Render;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiRoughPhraseSynthesizer {
        const int SampleRate = 44100;

        public float[] Synthesize(RenderPhrase phrase, CancellationTokenSource cancellation) {
            var resamplerItems = BuildResamplerItems(phrase);
            if (resamplerItems.Count == 0 || cancellation.IsCancellationRequested) {
                return Array.Empty<float>();
            }

            Log.Information("HifiRoughPhraseSynthesizer overlap_only using SharpWavtool convergence path items={Count}", resamplerItems.Count);
            var sourceCache = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);
            int underLengthCount = 0;
            int extendedCount = 0;

            foreach (var item in resamplerItems) {
                if (cancellation.IsCancellationRequested) {
                    return Array.Empty<float>();
                }
                try {
                    var segment = LoadSegmentFromSource(item, sourceCache);
                    if (segment.Length == 0) {
                        continue;
                    }
                    int targetSamples = ComputeTargetSegmentSamples(item);
                    int sourceSamples = segment.Length;
                    if (sourceSamples < targetSamples) {
                        underLengthCount++;
                        segment = ExtendSegmentToTarget(segment, targetSamples, item.phone);
                        extendedCount++;
                    }
                    double sourceMs = sourceSamples * 1000.0 / SampleRate;
                    double targetMs = targetSamples * 1000.0 / SampleRate;
                    if (sourceMs + 1e-3 < targetMs * 0.85) {
                        Log.Warning(
                            "Hifi overlap_only timing risk phoneme={Phoneme} source_ms={SourceMs:F2} target_ms={TargetMs:F2} skip_over_ms={SkipOverMs:F2} dur_required_ms={DurRequiredMs:F2}",
                            item.phone.phoneme,
                            sourceMs,
                            targetMs,
                            item.skipOver,
                            item.durRequired);
                    }

                    string segmentPath = GetOverlapSegmentCachePath(item);
                    item.phrase.AddCacheFile(segmentPath);
                    item.outputFile = segmentPath;
                    SaveWave(segmentPath, segment);
                } catch (Exception ex) {
                    Log.Warning(ex, "Hifi overlap_only source compose failed phoneme={Phoneme} file={File}", item.phone.phoneme, item.inputFile);
                }
            }

            Log.Information(
                "HifiRoughPhraseSynthesizer overlap_only summary under_length_items={UnderLengthItems} extended_items={ExtendedItems} total_items={TotalItems}",
                underLengthCount,
                extendedCount,
                resamplerItems.Count);

            var wavtool = new SharpWavtool(true);
            var samples = wavtool.Concatenate(resamplerItems, string.Empty, cancellation);
            return samples ?? Array.Empty<float>();
        }

        static List<ResamplerItem> BuildResamplerItems(RenderPhrase phrase) {
            var resamplerItems = new List<ResamplerItem>(phrase.phones.Length);
            foreach (var phone in phrase.phones) {
                if (phone.oto == null || string.IsNullOrWhiteSpace(phone.oto.File)) {
                    continue;
                }
                resamplerItems.Add(new ResamplerItem(phrase, phone));
            }
            return resamplerItems;
        }

        static float[] LoadSegmentFromSource(ResamplerItem item, Dictionary<string, float[]> sourceCache) {
            if (string.IsNullOrWhiteSpace(item.inputFile) || !File.Exists(item.inputFile)) {
                return Array.Empty<float>();
            }
            if (!sourceCache.TryGetValue(item.inputFile, out var full)) {
                using var waveStream = Wave.OpenFile(item.inputFile);
                full = Wave.GetSamples(waveStream.ToSampleProvider().ToMono(1, 0))
                    .Select(s => Math.Clamp(s, -1f, 1f))
                    .ToArray();
                sourceCache[item.inputFile] = full;
            }
            return SliceWithOto(full, item.phone);
        }

        static float[] SliceWithOto(float[] source, RenderPhone phone) {
            if (source.Length == 0 || phone.oto == null) {
                return Array.Empty<float>();
            }
            int offset = Math.Clamp(MsToSamples(phone.oto.Offset), 0, source.Length);
            int available = Math.Max(0, source.Length - offset);
            if (available == 0) {
                return Array.Empty<float>();
            }
            int cutoff = MsToSamples(phone.oto.Cutoff);
            int length = cutoff >= 0
                ? available - cutoff
                : Math.Min(available, -cutoff);
            length = Math.Clamp(length, 0, available);
            if (length == 0) {
                return Array.Empty<float>();
            }
            var result = new float[length];
            Array.Copy(source, offset, result, 0, length);
            return result;
        }

        static int MsToSamples(double ms) {
            return (int)Math.Round(ms * SampleRate / 1000.0);
        }

        static int ComputeTargetSegmentSamples(ResamplerItem item) {
            int durRequiredSamples = Math.Max(1, MsToSamples(item.durRequired));
            int envelopeTailSamples = 1;
            var envelope = item.EnvelopeMsToSamples();
            if (envelope.Count > 0) {
                envelopeTailSamples = Math.Max(1, (int)Math.Ceiling(envelope[envelope.Count - 1].X));
            }
            return Math.Max(durRequiredSamples, envelopeTailSamples);
        }

        static float[] ExtendSegmentToTarget(float[] source, int targetSamples, RenderPhone phone) {
            if (source.Length >= targetSamples) {
                return source;
            }
            if (source.Length == 0) {
                return new float[targetSamples];
            }

            var output = new float[targetSamples];
            ResolveSustainLoop(source.Length, phone, out int loopStart, out int loopEnd);
            int loopLength = Math.Max(1, loopEnd - loopStart);
            int crossfade = Math.Min(MsToSamples(28), Math.Min(loopLength / 3, Math.Max(0, loopEnd / 3)));
            if (crossfade < MsToSamples(4) && loopLength > MsToSamples(12)) {
                crossfade = Math.Min(MsToSamples(8), loopLength / 3);
            }

            loopStart = FindSmoothLoopStart(source, loopStart, loopEnd, crossfade);
            loopEnd = Math.Min(source.Length, loopStart + loopLength);
            int prefixEnd = Math.Clamp(loopEnd, 1, Math.Min(source.Length, targetSamples));
            Array.Copy(source, output, prefixEnd);

            int write = prefixEnd;
            int loopIndex = 0;
            while (write < targetSamples) {
                int dstStart = crossfade > 1 ? Math.Max(0, write - crossfade) : write;
                int copy = Math.Min(loopLength, targetSamples - dstStart);
                for (int i = 0; i < copy; i++) {
                    int dst = dstStart + i;
                    float sample = source[loopStart + i % loopLength];
                    if (crossfade > 1 && dst < write && i < crossfade) {
                        float t = (float)(i + 1) / (crossfade + 1);
                        float fadeIn = (float)(0.5 - 0.5 * Math.Cos(Math.PI * t));
                        float fadeOut = 1f - fadeIn;
                        output[dst] = output[dst] * fadeOut + sample * fadeIn;
                    } else {
                        output[dst] = sample;
                    }
                }
                write = Math.Max(write + 1, dstStart + copy);
                loopIndex++;
                if (loopIndex > 0 && loopIndex % 8 == 0) {
                    Log.Debug(
                        "Hifi overlap_only sustain loop phoneme={Phoneme} target_samples={TargetSamples} source_samples={SourceSamples} loop_start={LoopStart} loop_end={LoopEnd} crossfade={Crossfade}",
                        phone.phoneme,
                        targetSamples,
                        source.Length,
                        loopStart,
                        loopEnd,
                        crossfade);
                }
            }

            return output;
        }

        static void ResolveSustainLoop(int sourceLength, RenderPhone phone, out int loopStart, out int loopEnd) {
            int guard = Math.Min(MsToSamples(35), Math.Max(0, sourceLength / 6));
            int stableStart = Math.Clamp(MsToSamples(Math.Max(0, phone.oto?.Consonant ?? phone.preutterMs)) + MsToSamples(15), 0, Math.Max(0, sourceLength - 1));
            int stableEnd = Math.Clamp(sourceLength - guard, stableStart + 1, sourceLength);
            int stableLength = Math.Max(1, stableEnd - stableStart);
            int preferredLength = Math.Clamp(stableLength, MsToSamples(120), MsToSamples(420));
            if (stableLength < MsToSamples(80)) {
                stableStart = Math.Clamp(Math.Min(stableStart, sourceLength / 3), 0, Math.Max(0, sourceLength - 1));
                stableEnd = sourceLength;
                stableLength = Math.Max(1, stableEnd - stableStart);
                preferredLength = Math.Clamp(stableLength, Math.Min(MsToSamples(45), stableLength), Math.Min(MsToSamples(260), stableLength));
            }
            loopEnd = stableEnd;
            loopStart = Math.Clamp(loopEnd - preferredLength, stableStart, Math.Max(stableStart, loopEnd - 1));
        }

        static int FindSmoothLoopStart(float[] source, int loopStart, int loopEnd, int crossfade) {
            int loopLength = loopEnd - loopStart;
            if (crossfade < 16 || loopLength <= crossfade * 2) {
                return loopStart;
            }
            int searchRadius = Math.Min(MsToSamples(40), Math.Max(0, loopStart));
            int first = Math.Max(0, loopStart - searchRadius);
            int last = Math.Min(loopStart + searchRadius, source.Length - loopLength);
            if (last <= first) {
                return loopStart;
            }

            int step = Math.Max(8, crossfade / 8);
            int best = loopStart;
            double bestError = double.PositiveInfinity;
            for (int candidate = first; candidate <= last; candidate += step) {
                double error = 0;
                int tail = candidate + loopLength - crossfade;
                for (int i = 0; i < crossfade; i += 4) {
                    double diff = source[tail + i] - source[candidate + i];
                    error += diff * diff;
                }
                if (error < bestError) {
                    bestError = error;
                    best = candidate;
                }
            }
            return best;
        }

        static string GetOverlapSegmentCachePath(ResamplerItem item) {
            return Path.Join(PathManager.Inst.CachePath, $"hifi-overlap-src-{item.hash:x16}.wav");
        }

        static void SaveWave(string path, float[] samples) {
            using var writer = new WaveFileWriter(path, WaveFormat.CreateIeeeFloatWaveFormat(SampleRate, 1));
            if (samples.Length > 0) {
                writer.WriteSamples(samples, 0, samples.Length);
            }
        }
    }
}
