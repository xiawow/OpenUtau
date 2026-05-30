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
    /// <summary>
    /// DEPRECATED / unused by the live render path. Previously this used SharpWavtool to
    /// concatenate per-phone slices into one continuous rough wav, which was then re-analyzed by
    /// variable-position mel sampling. As of the mel-domain refactor (CacheKey v18), each phone's
    /// oto slice is turned into mel independently and the mels are concatenated with overlap
    /// cross-fades (see <see cref="HifiMelPhraseAssembler"/>), so this class is no longer called.
    /// Kept for reference / A-B comparison only.
    /// </summary>
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
                    }
                    double sourceMs = sourceSamples * 1000.0 / SampleRate;
                    double targetMs = targetSamples * 1000.0 / SampleRate;
                    if (sourceMs + 1e-3 < targetMs * 0.85) {
                        Log.Warning(
                            "Hifi overlap_only rough_vowel_loop_disabled phoneme={Phoneme} source_ms={SourceMs:F2} target_ms={TargetMs:F2} skip_over_ms={SkipOverMs:F2} dur_required_ms={DurRequiredMs:F2}",
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
                "HifiRoughPhraseSynthesizer overlap_only summary under_length_items={UnderLengthItems} rough_vowel_loop_disabled=True total_items={TotalItems}",
                underLengthCount,
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
