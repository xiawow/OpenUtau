using System.Threading;

namespace OpenUtau.Core.HifiNeural {
    public readonly record struct HifiRenderContext(bool IsPreRender, CancellationToken CancellationToken) {
        public static readonly HifiRenderContext None = new(false, CancellationToken.None);

        public void ThrowIfCancellationRequested() {
            CancellationToken.ThrowIfCancellationRequested();
        }
    }
}
