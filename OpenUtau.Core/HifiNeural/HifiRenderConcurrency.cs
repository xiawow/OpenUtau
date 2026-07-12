using System;
using System.Threading;
using OpenUtau.Core.Util;

namespace OpenUtau.Core.HifiNeural {
    internal static class HifiRenderConcurrency {
        static readonly int cpuPhraseLimit = Environment.ProcessorCount >= 8 ? 2 : 1;
        static readonly int dmlPhraseLimit = Environment.ProcessorCount >= 4 ? 2 : 1;
        static readonly SemaphoreSlim cpuGate = new SemaphoreSlim(cpuPhraseLimit, cpuPhraseLimit);
        static readonly SemaphoreSlim dmlGate = new SemaphoreSlim(dmlPhraseLimit, dmlPhraseLimit);
        static int activeFeatureBuilds;
        static int activeCpuInferences;

        public static int CpuPhraseLimit => cpuPhraseLimit;
        public static int DmlPhraseLimit => dmlPhraseLimit;
        public static int ActiveFeatureBuilds => Math.Max(0, Volatile.Read(ref activeFeatureBuilds));
        public static int ActiveCpuInferences => Math.Max(0, Volatile.Read(ref activeCpuInferences));

        public static SemaphoreSlim PhraseGate(bool usesDirectML) => usesDirectML ? dmlGate : cpuGate;

        public static int MelParallelism {
            get {
                int configured = Math.Max(1, Preferences.Default.NumRenderThreads);
                return ResolveMelParallelism(configured, ActiveFeatureBuilds, ActiveCpuInferences);
            }
        }

        internal static int ResolveMelParallelism(int configured, int featureBuilds, int cpuInferences) {
            configured = Math.Max(1, configured);
            int active = Math.Max(1, Math.Max(0, featureBuilds) + Math.Max(0, cpuInferences));
            return Math.Max(1, configured / active);
        }

        public static IDisposable EnterFeatureBuild() {
            Interlocked.Increment(ref activeFeatureBuilds);
            return new CounterScope(() => Interlocked.Decrement(ref activeFeatureBuilds));
        }

        public static IDisposable EnterInference(bool usesDirectML) {
            if (usesDirectML) {
                return NoopScope.Instance;
            }
            Interlocked.Increment(ref activeCpuInferences);
            return new CounterScope(() => Interlocked.Decrement(ref activeCpuInferences));
        }

        sealed class CounterScope : IDisposable {
            readonly Action release;
            bool disposed;

            public CounterScope(Action release) {
                this.release = release;
            }

            public void Dispose() {
                if (!disposed) {
                    release();
                    disposed = true;
                }
            }
        }

        sealed class NoopScope : IDisposable {
            public static readonly NoopScope Instance = new NoopScope();
            public void Dispose() { }
        }
    }
}
