using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;

namespace OpenUtau.Core.HifiNeural {
    /// <summary>
    /// Process-wide exact-result cache for immutable HIFI source features. A single byte budget
    /// prevents the PCM, mel and HNSEP caches from independently consuming unbounded memory.
    /// </summary>
    internal sealed class HifiBoundedMemoryCache {
        sealed class Entry {
            public required object Value { get; init; }
            public required long SizeBytes { get; init; }
            public required LinkedListNode<string> Node { get; init; }
        }

        readonly object gate = new object();
        readonly Dictionary<string, Entry> entries = new Dictionary<string, Entry>(StringComparer.Ordinal);
        readonly LinkedList<string> lru = new LinkedList<string>();
        readonly ConcurrentDictionary<string, Lazy<object?>> pending = new ConcurrentDictionary<string, Lazy<object?>>(StringComparer.Ordinal);
        readonly long maxBytes;
        long usedBytes;

        public HifiBoundedMemoryCache(long maxBytes) {
            this.maxBytes = Math.Max(1, maxBytes);
        }

        public long MaxBytes => maxBytes;
        public long UsedBytes {
            get {
                lock (gate) {
                    return usedBytes;
                }
            }
        }
        public int Count {
            get {
                lock (gate) {
                    return entries.Count;
                }
            }
        }

        public bool TryGet<T>(string key, out T value) where T : class {
            lock (gate) {
                if (entries.TryGetValue(key, out var entry) && entry.Value is T typed) {
                    lru.Remove(entry.Node);
                    lru.AddFirst(entry.Node);
                    value = typed;
                    return true;
                }
            }
            value = null!;
            return false;
        }

        public T GetOrAdd<T>(
            string key,
            Func<T> factory,
            Func<T, long> sizeEstimator,
            out bool cacheHit) where T : class {
            if (TryGet(key, out T cached)) {
                cacheHit = true;
                return cached;
            }

            var candidate = new Lazy<object?>(() => factory(), LazyThreadSafetyMode.ExecutionAndPublication);
            var lazy = pending.GetOrAdd(key, candidate);
            bool ownsComputation = ReferenceEquals(candidate, lazy);
            try {
                var value = (T)lazy.Value!;
                AddOrRefresh(key, value, sizeEstimator(value));
                cacheHit = !ownsComputation;
                return value;
            } finally {
                if (ownsComputation) {
                    pending.TryRemove(key, out _);
                }
            }
        }

        public void AddOrRefresh<T>(string key, T value, long sizeBytes) where T : class {
            sizeBytes = Math.Max(0, sizeBytes);
            if (sizeBytes <= 0 || sizeBytes > maxBytes) {
                return;
            }
            lock (gate) {
                if (entries.TryGetValue(key, out var existing)) {
                    usedBytes -= existing.SizeBytes;
                    lru.Remove(existing.Node);
                    entries.Remove(key);
                }
                var node = lru.AddFirst(key);
                entries[key] = new Entry {
                    Value = value,
                    SizeBytes = sizeBytes,
                    Node = node,
                };
                usedBytes += sizeBytes;
                while (usedBytes > maxBytes && lru.Last != null) {
                    string victimKey = lru.Last.Value;
                    lru.RemoveLast();
                    if (entries.Remove(victimKey, out var victim)) {
                        usedBytes -= victim.SizeBytes;
                    }
                }
            }
        }

        public void Clear() {
            lock (gate) {
                entries.Clear();
                lru.Clear();
                usedBytes = 0;
            }
            pending.Clear();
        }
    }

    internal static class HifiRenderMemoryCache {
        const long MiB = 1024L * 1024L;
        public static readonly HifiBoundedMemoryCache Shared = new HifiBoundedMemoryCache(ResolveBudgetBytes());

        public static long FloatBytes(float[] values) => values.LongLength * sizeof(float);
        public static long FloatBytes(float[,] values) => values.LongLength * sizeof(float);

        public static string FileVersionKey(string path) {
            try {
                var info = new FileInfo(path);
                return string.Concat(
                    info.FullName,
                    "|", info.Exists ? info.Length : -1,
                    "|", info.Exists ? info.LastWriteTimeUtc.Ticks : 0);
            } catch {
                return path ?? string.Empty;
            }
        }

        static long ResolveBudgetBytes() {
            long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            if (available <= 0) {
                return 192 * MiB;
            }
            return Math.Clamp(available / 16, 64 * MiB, 256 * MiB);
        }
    }
}
