namespace OpenUtau.Core.HifiNeural {
    public interface IHifiTransitionRefiner {
        string Name { get; }
        HifiPhraseFeatures Refine(HifiPhraseFeatures features);
    }

    public sealed class IdentityHifiTransitionRefiner : IHifiTransitionRefiner {
        public string Name => "Identity";

        public HifiPhraseFeatures Refine(HifiPhraseFeatures features) {
            return features;
        }
    }
}
