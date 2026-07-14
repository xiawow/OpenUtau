using System;
using System.Collections.Generic;
using System.Linq;
using OpenUtau.Core.Render;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiHnSpectralProfileTrack {
        readonly HifiHnSpectralProfile?[] profiles;

        public int FrameCount => profiles.Length;
        public bool HasAudibleEffect { get; }
        public bool NeedsHarmonicProcessing { get; }
        public bool NeedsNoiseProcessing { get; }

        internal HifiHnSpectralProfileTrack(IReadOnlyList<HifiHnSpectralProfile?> frameProfiles) {
            int count = Math.Max(1, frameProfiles.Count);
            profiles = new HifiHnSpectralProfile?[count];
            var normalizedByKey = new Dictionary<string, HifiHnSpectralProfile>(StringComparer.Ordinal);
            bool hasAudibleEffect = false;
            bool needsHarmonic = false;
            bool needsNoise = false;
            for (int i = 0; i < frameProfiles.Count; i++) {
                HifiHnSpectralProfile? source = frameProfiles[i];
                if (source?.HasAudibleEffect != true) {
                    continue;
                }
                var normalized = source.Clone().Normalize();
                string key = normalized.CacheKey();
                if (!normalizedByKey.TryGetValue(key, out var profile)) {
                    profile = normalized;
                    normalizedByKey[key] = profile;
                }
                profiles[i] = profile;
                bool balanceActive = profile.BalanceDb.Any(value => Math.Abs(value) >= 0.01);
                hasAudibleEffect = true;
                needsHarmonic |= balanceActive
                    || (profile.DynamicsEnabled
                        && profile.DynamicsTarget is HifiHnDynamicsTarget.Harmonic or HifiHnDynamicsTarget.Both);
                needsNoise |= balanceActive
                    || (profile.DynamicsEnabled
                        && profile.DynamicsTarget is HifiHnDynamicsTarget.Noise or HifiHnDynamicsTarget.Both);
            }
            HasAudibleEffect = hasAudibleEffect;
            NeedsHarmonicProcessing = needsHarmonic;
            NeedsNoiseProcessing = needsNoise;
        }

        public static HifiHnSpectralProfileTrack Constant(HifiHnSpectralProfile profile) {
            return new HifiHnSpectralProfileTrack(new[] { profile });
        }

        public static HifiHnSpectralProfileTrack ForPhrase(
            RenderPhrase phrase,
            double phraseStartMs,
            int frameCount) {
            frameCount = Math.Max(1, frameCount);
            var frameProfiles = new HifiHnSpectralProfile?[frameCount];
            var noteEntries = new List<(double StartMs, HifiHnSpectralProfile? Profile)>();
            var seenNotes = new HashSet<int>();
            foreach (var phone in phrase.phones) {
                if (phone.noteIndex < 0
                    || phone.noteIndex >= phrase.notes.Length
                    || !seenNotes.Add(phone.noteIndex)) {
                    continue;
                }
                noteEntries.Add((
                    phrase.notes[phone.noteIndex].positionMs,
                    phone.hifiHnSpectralProfileIsPostEffect
                        ? phone.hifiHnSpectralProfile
                        : null));
            }
            if (noteEntries.Count == 0) {
                return new HifiHnSpectralProfileTrack(frameProfiles);
            }
            noteEntries.Sort((left, right) => left.StartMs.CompareTo(right.StartMs));

            int note = 0;
            for (int frame = 0; frame < frameCount; frame++) {
                double frameCenterMs = phraseStartMs + (frame + 0.5) * HifiF0Builder.FrameMs;
                while (note + 1 < noteEntries.Count
                    && frameCenterMs >= noteEntries[note + 1].StartMs) {
                    note++;
                }
                frameProfiles[frame] = noteEntries[note].Profile;
            }
            return new HifiHnSpectralProfileTrack(frameProfiles);
        }

        internal HifiHnSpectralProfile? ProfileAtSourceSample(double sourceSample) {
            int frame = (int)Math.Floor(Math.Max(0, sourceSample) / HifiOnnxVocoder.HopSize);
            return profiles[Math.Clamp(frame, 0, profiles.Length - 1)];
        }
    }
}
