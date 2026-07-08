using System;
using System.Diagnostics;
using System.Threading;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    static class HifiDmlInferenceGate {
        const int WaitLogMs = 250;
        static readonly object sync = new();
        static bool occupied;
        static int interactiveWaiters;

        public static T Run<T>(bool useDml, string owner, Func<T> action) {
            return Run(useDml, owner, HifiRenderContext.None, action);
        }

        public static T Run<T>(bool useDml, string owner, HifiRenderContext context, Func<T> action) {
            if (!useDml) {
                return action();
            }

            var sw = Stopwatch.StartNew();
            using var lease = Enter(owner, context, sw);
            context.ThrowIfCancellationRequested();
            return action();
        }

        static GateLease Enter(string owner, HifiRenderContext context, Stopwatch sw) {
            context.ThrowIfCancellationRequested();
            bool interactive = !context.IsPreRender;
            bool countedInteractive = false;
            var registration = default(CancellationTokenRegistration);
            try {
                if (context.CancellationToken.CanBeCanceled) {
                    registration = context.CancellationToken.Register(static state => {
                        var gateSync = (object)state!;
                        lock (gateSync) {
                            Monitor.PulseAll(gateSync);
                        }
                    }, sync);
                }

                lock (sync) {
                    if (interactive) {
                        interactiveWaiters++;
                        countedInteractive = true;
                    }
                    while (occupied || (context.IsPreRender && interactiveWaiters > 0)) {
                        if (context.CancellationToken.IsCancellationRequested) {
                            throw new OperationCanceledException(context.CancellationToken);
                        }
                        Monitor.Wait(sync, 50);
                    }
                    occupied = true;
                    if (countedInteractive) {
                        interactiveWaiters--;
                        countedInteractive = false;
                    }
                }

                if (sw.ElapsedMilliseconds >= WaitLogMs) {
                    Log.Debug(
                        "Hifi DirectML inference waited owner={Owner} pre_render={PreRender} wait_ms={WaitMs}",
                        owner,
                        context.IsPreRender,
                        sw.ElapsedMilliseconds);
                }
                return new GateLease(registration);
            } catch {
                if (countedInteractive) {
                    lock (sync) {
                        interactiveWaiters--;
                        Monitor.PulseAll(sync);
                    }
                }
                registration.Dispose();
                throw;
            }
        }

        sealed class GateLease : IDisposable {
            readonly CancellationTokenRegistration registration;
            bool disposed;

            public GateLease(CancellationTokenRegistration registration) {
                this.registration = registration;
            }

            public void Dispose() {
                if (disposed) {
                    return;
                }
                disposed = true;
                lock (sync) {
                    occupied = false;
                    Monitor.PulseAll(sync);
                }
                registration.Dispose();
            }
        }
    }
}
