using System;
using System.Collections.Generic;
using System.Linq;
using OpenUtau.Core.Render;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    /// <summary>
    /// Builds the phrase mel by extracting a mel spectrogram from each phone's oto source
    /// slice independently, time-stretching it per phone (reusing the exact stretch logic in
    /// <see cref="HifiRoughFeatureBuilder"/>), then assembling the stretched phone mels onto the
    /// phrase frame grid with overlap cross-fades.
    ///
    /// This replaces the previous "SharpWavtool concatenates a rough wav, then variable-position
    /// mel is sampled over it" path. By keeping each phone's source in its own local coordinate
    /// space and explicitly cross-fading the overlap region (driven by oto.overlap/preutter), the
    /// VCV/CVVC vowel-to-vowel boundaries stay continuous even under large stretch, which is what
    /// broke before.
    /// </summary>
    public sealed class HifiMelPhraseAssembler {
        const float LogFloor = -11.512925f; // log(1e-5), matches HifiMelExtractor floor.
        const int SampleRate = HifiMelExtractor.SampleRate;

        readonly HifiMelExtractor melExtractor = new HifiMelExtractor();

        sealed class PhoneMelSegment {
            public int PhoneIndex;
            public string Phoneme = string.Empty;
            public RenderPhone Phone = null!;
            public float[,] Mel = new float[HifiMelExtractor.NMels, 0];
            public int StartFrame;
            public int FrameCount;
            public int OverlapFramesWithPrev;
            public int ConsonantFrames;
            public string Strategy = string.Empty;
        }

        /// <summary>
        /// Build the full phrase mel [NMels, targetFrames] from per-phone source slices.
        /// <paramref name="consonantFrameRanges"/> receives, per phone that has a fixed consonant
        /// region, the absolute phrase frame span [start, end) of that consonant — used by the
        /// caller to mask F0 to 0 there (the NSF vocoder then drives the consonant with noise
        /// excitation instead of harmonic, which is correct for unvoiced consonants and avoids the
        /// buzzy/stretched feel on Japanese-VCV consonants).
        /// </summary>
        public float[,] Build(
            RenderPhrase phrase,
            double phraseStartMs,
            int targetFrames,
            float[] targetF0,
            Dictionary<string, float[]> sourceCache,
            out HifiMelAssemblyReport report) {
            report = new HifiMelAssemblyReport();
            var output = new float[HifiMelExtractor.NMels, Math.Max(0, targetFrames)];
            FillConstant(output, LogFloor);
            if (targetFrames <= 0 || phrase.phones.Length == 0) {
                return output;
            }

            var segments = new List<PhoneMelSegment>(phrase.phones.Length);
            // Source mels are cached per oto slice (File + Offset + Cutoff) so a phone/oto that
            // recurs within the phrase reuses its STFT instead of re-extracting it.
            var sliceMelCache = new Dictionary<string, float[,]>(StringComparer.Ordinal);
            for (int i = 0; i < phrase.phones.Length; i++) {
                var phone = phrase.phones[i];
                var segment = BuildPhoneSegment(phrase, phone, i, phraseStartMs, targetFrames, targetF0, sourceCache, sliceMelCache);
                if (segment != null) {
                    segments.Add(segment);
                }
            }

            if (segments.Count == 0) {
                return output;
            }

            AssembleWithOverlapCrossfade(output, segments, targetFrames);
            BuildAssemblyReport(segments, targetFrames, phraseStartMs, report);
            foreach (var seg in segments) {
                if (seg.ConsonantFrames > 0) {
                    int start = Math.Clamp(seg.StartFrame, 0, targetFrames);
                    int end = Math.Clamp(seg.StartFrame + seg.ConsonantFrames, start, targetFrames);
                    if (end > start) {
                        report.ConsonantFrameRanges.Add((start, end));
                    }
                }
            }
            LogSummary(phrase, segments, targetFrames);
            return output;
        }

        PhoneMelSegment? BuildPhoneSegment(
            RenderPhrase phrase,
            RenderPhone phone,
            int phoneIndex,
            double phraseStartMs,
            int targetFrames,
            float[] targetF0,
            Dictionary<string, float[]> sourceCache,
            Dictionary<string, float[,]> sliceMelCache) {
            if (phone.oto == null || string.IsNullOrWhiteSpace(phone.oto.File)) {
                return null;
            }

            float[] sourceSamples = LoadSourceSlice(phone, sourceCache);
            float[,] sourceMel = LoadSliceMel(phone, sourceSamples, sliceMelCache);
            int sourceFrames = sourceMel.GetLength(1);
            if (sourceFrames <= 0) {
                return null;
            }

            // Placement on the phrase frame grid. The phone audibly begins at its pre-utterance
            // point (positionMs - preutterMs), the same anchor SharpWavtool used (posMs =
            // phone.positionMs - phone.leadingMs). The segment runs until the next phone's anchor
            // (or phrase end), plus a short overlap tail so adjacent segments share frames.
            int startFrame = MsToFrame(phone.positionMs - phone.preutterMs - phraseStartMs);
            startFrame = Math.Clamp(startFrame, 0, Math.Max(0, targetFrames - 1));

            int nextAnchorFrame;
            if (phoneIndex + 1 < phrase.phones.Length) {
                var next = phrase.phones[phoneIndex + 1];
                nextAnchorFrame = MsToFrame(next.positionMs - next.preutterMs - phraseStartMs);
            } else {
                nextAnchorFrame = targetFrames;
            }
            nextAnchorFrame = Math.Clamp(nextAnchorFrame, startFrame + 1, targetFrames);

            // Overlap with the next segment: the next phone's overlap window (overlapMs) is the
            // region where both phones sound. We extend this segment past the next anchor by that
            // overlap so the cross-fade has frames to work with.
            int overlapTailFrames = 0;
            if (phoneIndex + 1 < phrase.phones.Length) {
                double nextOverlapMs = Math.Max(0, phrase.phones[phoneIndex + 1].overlapMs);
                overlapTailFrames = Math.Clamp(
                    (int)Math.Round(nextOverlapMs / HifiF0Builder.FrameMs),
                    0,
                    Math.Max(0, targetFrames - nextAnchorFrame));
            }

            int frameCount = Math.Max(1, nextAnchorFrame - startFrame + overlapTailFrames);
            frameCount = Math.Min(frameCount, targetFrames - startFrame);
            if (frameCount <= 0) {
                return null;
            }

            var phoneMel = new float[HifiMelExtractor.NMels, frameCount];
            // Local target F0 slice so the (F0-aware) stretch logic sees the right pitch motion.
            float[] localTargetF0 = SliceTargetF0(targetF0, startFrame, frameCount);
            var report = HifiRoughFeatureBuilder.WritePhoneMappedSegment(
                sourceMel,
                0,
                sourceFrames,
                phoneMel,
                0,
                frameCount,
                phone,
                localTargetF0,
                sourceSamples);

            return new PhoneMelSegment {
                PhoneIndex = phoneIndex,
                Phoneme = phone.phoneme,
                Phone = phone,
                Mel = phoneMel,
                StartFrame = startFrame,
                FrameCount = frameCount,
                ConsonantFrames = report.ConsonantTargetFrames,
                Strategy = report.Strategy,
            };
        }

        static void AssembleWithOverlapCrossfade(float[,] output, List<PhoneMelSegment> segments, int targetFrames) {
            int bins = output.GetLength(0);
            // accumulated[t] tracks how many segments have already contributed to frame t, so we
            // know when we are inside an overlap region and must cross-fade instead of overwrite.
            var occupiedUntil = new int[1]; // boundary of the previously placed segment (exclusive).
            occupiedUntil[0] = 0;

            for (int s = 0; s < segments.Count; s++) {
                var seg = segments[s];
                int segStart = seg.StartFrame;
                int segEnd = Math.Min(targetFrames, seg.StartFrame + seg.FrameCount);
                int prevEnd = occupiedUntil[0];

                int overlapStart = Math.Max(segStart, 0);
                int overlapEnd = Math.Min(segEnd, prevEnd); // frames shared with the previous segment.
                int overlapFrames = Math.Max(0, overlapEnd - overlapStart);
                seg.OverlapFramesWithPrev = s == 0 ? 0 : overlapFrames;

                for (int t = segStart; t < segEnd; t++) {
                    int local = t - segStart;
                    bool inOverlap = s > 0 && t < prevEnd && overlapFrames > 0;
                    if (inOverlap) {
                        // Equal-power cross-fade in linear (power) domain. The previous segment is
                        // already written into output[t]; blend it with this segment.
                        double u = CrossfadeProgress(t - overlapStart, overlapFrames);
                        for (int m = 0; m < bins; m++) {
                            output[m, t] = CrossfadeLogMel(output[m, t], seg.Mel[m, local], u);
                        }
                    } else {
                        for (int m = 0; m < bins; m++) {
                            output[m, t] = seg.Mel[m, local];
                        }
                    }
                }
                occupiedUntil[0] = Math.Max(prevEnd, segEnd);
            }
        }

        static void BuildAssemblyReport(
            List<PhoneMelSegment> segments,
            int targetFrames,
            double phraseStartMs,
            HifiMelAssemblyReport report) {
            for (int i = 0; i < segments.Count; i++) {
                var seg = segments[i];
                var phone = seg.Phone;
                int start = Math.Clamp(seg.StartFrame, 0, targetFrames);
                int end = Math.Clamp(seg.StartFrame + seg.FrameCount, start + 1, Math.Max(start + 1, targetFrames));
                report.Phones.Add(new HifiPhoneMetadata {
                    Index = seg.PhoneIndex,
                    Phoneme = phone.phoneme,
                    Tone = phone.tone,
                    SourceFile = phone.oto?.File ?? string.Empty,
                    PositionMs = phone.positionMs,
                    DurationMs = phone.durationMs,
                    LeadingMs = phone.leadingMs,
                    StartFrame = start,
                    FrameCount = Math.Max(1, end - start),
                    SourceSkipOverMs = 0,
                    SourceStartOffsetFrames = 0,
                });
                if (i > 0) {
                    var left = segments[i - 1];
                    report.Boundaries.Add(new HifiBoundaryMetadata {
                        Index = i - 1,
                        LeftPhoneIndex = left.PhoneIndex,
                        RightPhoneIndex = seg.PhoneIndex,
                        LeftPhone = left.Phoneme,
                        RightPhone = seg.Phoneme,
                        Frame = start,
                        PositionMs = phraseStartMs + start * HifiF0Builder.FrameMs,
                        TransitionType = seg.OverlapFramesWithPrev > 0 ? "oto-overlap" : "phone",
                    });
                }
            }
        }

        /// <summary>
        /// Equal-power cross-fade between two log-mel values at normalized position u in [0,1],
        /// where u=0 is fully the previous (old) value and u=1 is fully the new value. The blend is
        /// done in the linear power domain (exp of log-mel) so a cross-fade between two equal-energy
        /// voiced segments preserves energy through the overlap — this is what keeps VCV/CVVC vowel
        /// boundaries from dipping or jumping under stretch.
        /// </summary>
        internal static double CrossfadeProgress(int overlapOffset, int overlapFrames) {
            if (overlapFrames <= 1) {
                return 0.5;
            }
            return Math.Clamp(overlapOffset / (double)(overlapFrames - 1), 0.0, 1.0);
        }

        internal static float CrossfadeLogMel(float logOld, float logNew, double u) {
            u = Math.Clamp(u, 0.0, 1.0);
            double wNew = Math.Sin(0.5 * Math.PI * u);
            double wOld = Math.Cos(0.5 * Math.PI * u);
            double pOld = Math.Exp(logOld);
            double pNew = Math.Exp(logNew);
            double mixed = pOld * wOld * wOld + pNew * wNew * wNew;
            return (float)Math.Log(Math.Max(mixed, 1e-5));
        }

        float[,] LoadSliceMel(
            RenderPhone phone,
            float[] sourceSamples,
            Dictionary<string, float[,]> sliceMelCache) {
            string key = SliceCacheKey(phone);
            if (key.Length > 0 && sliceMelCache.TryGetValue(key, out var cachedMel)) {
                return cachedMel;
            }
            float[,] mel = sourceSamples.Length == 0
                ? new float[HifiMelExtractor.NMels, 0]
                : melExtractor.Extract(sourceSamples);
            if (key.Length > 0) {
                sliceMelCache[key] = mel;
            }
            return mel;
        }

        static float[] SliceTargetF0(float[] targetF0, int startFrame, int frameCount) {
            var result = new float[Math.Max(0, frameCount)];
            if (targetF0.Length == 0 || result.Length == 0) {
                return result;
            }
            for (int i = 0; i < result.Length; i++) {
                int index = Math.Clamp(startFrame + i, 0, targetF0.Length - 1);
                result[i] = targetF0[index];
            }
            return result;
        }

        static string SliceCacheKey(RenderPhone phone) {
            if (phone.oto == null || string.IsNullOrWhiteSpace(phone.oto.File)) {
                return string.Empty;
            }
            // Offset+Cutoff fully determine the sample slice taken from the file, so two phones
            // sharing them (same oto entry) share the extracted mel.
            return string.Concat(
                phone.oto.File,
                "|", phone.oto.Offset.ToString("R"),
                "|", phone.oto.Cutoff.ToString("R"));
        }

        static float[] LoadSourceSlice(RenderPhone phone, Dictionary<string, float[]> sourceCache) {
            string file = phone.oto.File;
            if (string.IsNullOrWhiteSpace(file) || !System.IO.File.Exists(file)) {
                return Array.Empty<float>();
            }
            if (!sourceCache.TryGetValue(file, out var full)) {
                full = melExtractorLoad(file);
                sourceCache[file] = full;
            }
            return SliceWithOto(full, phone);
        }

        static float[] melExtractorLoad(string file) {
            var extractor = new HifiMelExtractor();
            return extractor.LoadMono(file).Select(s => Math.Clamp(s, -1f, 1f)).ToArray();
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

        static int MsToFrame(double ms) {
            return (int)Math.Round(ms / HifiF0Builder.FrameMs);
        }

        static void FillConstant(float[,] values, float value) {
            for (int m = 0; m < values.GetLength(0); m++) {
                for (int t = 0; t < values.GetLength(1); t++) {
                    values[m, t] = value;
                }
            }
        }

        static void LogSummary(RenderPhrase phrase, List<PhoneMelSegment> segments, int targetFrames) {
            Log.Debug(
                "HifiMelPhraseAssembler mel_domain_concat phones={Phones} segments={Segments} target_frames={TargetFrames}",
                phrase.phones.Length,
                segments.Count,
                targetFrames);
            foreach (var seg in segments) {
                Log.Debug(
                    "HifiMelPhraseAssembler segment phone_index={Index} phoneme={Phoneme} start_frame={Start} frame_count={Count} overlap_prev={Overlap} strategy={Strategy}",
                    seg.PhoneIndex,
                    seg.Phoneme,
                    seg.StartFrame,
                    seg.FrameCount,
                    seg.OverlapFramesWithPrev,
                    seg.Strategy);
            }
        }
    }
}
