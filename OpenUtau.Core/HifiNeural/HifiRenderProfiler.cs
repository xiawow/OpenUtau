using System;
using System.Diagnostics;
using System.Threading;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    internal enum HifiRenderStage {
        QueueWait,
        CacheRead,
        FeatureBuild,
        SourceDecode,
        Hnsep,
        HnsepStft,
        HnsepInference,
        HnsepIstft,
        SourceMel,
        SustainMel,
        Assembly,
        FeaturePost,
        VocoderPack,
        VocoderInference,
        VocoderOutput,
        PostLeveler,
        PostGrowl,
        PostAmplitude,
        PostNormalize,
        CacheWrite,
        Count,
    }

    internal enum HifiRenderCounter {
        PcmHit,
        PcmMiss,
        SourceMelHit,
        SourceMelMiss,
        SustainMelHit,
        SustainMelMiss,
        HnsepMemoryHit,
        HnsepMemoryMiss,
        HnsepDiskHit,
        HnsepDiskMiss,
        Count,
    }

    internal sealed class HifiRenderProfiler {
        static readonly AsyncLocal<HifiRenderProfiler?> current = new AsyncLocal<HifiRenderProfiler?>();

        readonly long[] stageTicks = new long[(int)HifiRenderStage.Count];
        readonly long[] counters = new long[(int)HifiRenderCounter.Count];
        readonly long started = Stopwatch.GetTimestamp();
        readonly int notes;
        readonly int phones;
        readonly double audioMs;
        string backend = "unknown";

        public HifiRenderProfiler(int notes, int phones, double audioMs) {
            this.notes = notes;
            this.phones = phones;
            this.audioMs = audioMs;
        }

        public static HifiRenderProfiler? Current => current.Value;

        public IDisposable Bind() {
            var previous = current.Value;
            current.Value = this;
            return new Binding(previous);
        }

        public void SetBackend(string value) {
            backend = string.IsNullOrWhiteSpace(value) ? "unknown" : value;
        }

        public static Measurement Measure(HifiRenderStage stage) {
            var profiler = current.Value;
            return profiler == null ? default : new Measurement(profiler, stage, Stopwatch.GetTimestamp());
        }

        public static T Time<T>(HifiRenderStage stage, Func<T> action) {
            using var timing = Measure(stage);
            return action();
        }

        public static void Count(HifiRenderCounter counter) {
            var profiler = current.Value;
            if (profiler != null) {
                Interlocked.Increment(ref profiler.counters[(int)counter]);
            }
        }

        public void LogSummary(string status) {
            double totalMs = ElapsedMs(Stopwatch.GetTimestamp() - started);
            double rtf = audioMs > 0 ? totalMs / audioMs : 0;
            Log.Information(
                "Hifi performance status={Status} backend={Backend} notes={Notes} phones={Phones} audio_ms={AudioMs:F1} total_ms={TotalMs:F1} rtf={Rtf:F3} queue_ms={QueueMs:F1} cache_read_ms={CacheReadMs:F1} feature_ms={FeatureMs:F1} source_decode_ms={SourceDecodeMs:F1} hnsep_ms={HnsepMs:F1} hnsep_stft_ms={HnsepStftMs:F1} hnsep_infer_ms={HnsepInferMs:F1} hnsep_istft_ms={HnsepIstftMs:F1} source_mel_ms={SourceMelMs:F1} sustain_mel_ms={SustainMelMs:F1} assembly_ms={AssemblyMs:F1} feature_post_ms={FeaturePostMs:F1} vocoder_pack_ms={VocoderPackMs:F1} vocoder_infer_ms={VocoderInferMs:F1} vocoder_output_ms={VocoderOutputMs:F1} leveler_ms={LevelerMs:F1} growl_ms={GrowlMs:F1} amplitude_ms={AmplitudeMs:F1} normalize_ms={NormalizeMs:F1} cache_write_ms={CacheWriteMs:F1} pcm_cache={PcmHit}/{PcmMiss} source_mel_cache={SourceMelHit}/{SourceMelMiss} sustain_mel_cache={SustainMelHit}/{SustainMelMiss} hnsep_memory_cache={HnsepMemoryHit}/{HnsepMemoryMiss} hnsep_disk_cache={HnsepDiskHit}/{HnsepDiskMiss} memory_cache_mb={MemoryCacheMb:F1}",
                status,
                backend,
                notes,
                phones,
                audioMs,
                totalMs,
                rtf,
                StageMs(HifiRenderStage.QueueWait),
                StageMs(HifiRenderStage.CacheRead),
                StageMs(HifiRenderStage.FeatureBuild),
                StageMs(HifiRenderStage.SourceDecode),
                StageMs(HifiRenderStage.Hnsep),
                StageMs(HifiRenderStage.HnsepStft),
                StageMs(HifiRenderStage.HnsepInference),
                StageMs(HifiRenderStage.HnsepIstft),
                StageMs(HifiRenderStage.SourceMel),
                StageMs(HifiRenderStage.SustainMel),
                StageMs(HifiRenderStage.Assembly),
                StageMs(HifiRenderStage.FeaturePost),
                StageMs(HifiRenderStage.VocoderPack),
                StageMs(HifiRenderStage.VocoderInference),
                StageMs(HifiRenderStage.VocoderOutput),
                StageMs(HifiRenderStage.PostLeveler),
                StageMs(HifiRenderStage.PostGrowl),
                StageMs(HifiRenderStage.PostAmplitude),
                StageMs(HifiRenderStage.PostNormalize),
                StageMs(HifiRenderStage.CacheWrite),
                Counter(HifiRenderCounter.PcmHit),
                Counter(HifiRenderCounter.PcmMiss),
                Counter(HifiRenderCounter.SourceMelHit),
                Counter(HifiRenderCounter.SourceMelMiss),
                Counter(HifiRenderCounter.SustainMelHit),
                Counter(HifiRenderCounter.SustainMelMiss),
                Counter(HifiRenderCounter.HnsepMemoryHit),
                Counter(HifiRenderCounter.HnsepMemoryMiss),
                Counter(HifiRenderCounter.HnsepDiskHit),
                Counter(HifiRenderCounter.HnsepDiskMiss),
                HifiRenderMemoryCache.Shared.UsedBytes / (1024.0 * 1024.0));
        }

        double StageMs(HifiRenderStage stage) => ElapsedMs(Interlocked.Read(ref stageTicks[(int)stage]));
        long Counter(HifiRenderCounter counter) => Interlocked.Read(ref counters[(int)counter]);
        static double ElapsedMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

        public readonly struct Measurement : IDisposable {
            readonly HifiRenderProfiler? profiler;
            readonly HifiRenderStage stage;
            readonly long start;

            internal Measurement(HifiRenderProfiler profiler, HifiRenderStage stage, long start) {
                this.profiler = profiler;
                this.stage = stage;
                this.start = start;
            }

            public void Dispose() {
                if (profiler != null) {
                    Interlocked.Add(ref profiler.stageTicks[(int)stage], Stopwatch.GetTimestamp() - start);
                }
            }
        }

        sealed class Binding : IDisposable {
            readonly HifiRenderProfiler? previous;
            bool disposed;

            public Binding(HifiRenderProfiler? previous) {
                this.previous = previous;
            }

            public void Dispose() {
                if (!disposed) {
                    current.Value = previous;
                    disposed = true;
                }
            }
        }
    }
}
