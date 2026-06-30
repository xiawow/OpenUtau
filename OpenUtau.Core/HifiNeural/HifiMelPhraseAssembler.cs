using System;
using System.Collections.Generic;
using System.Linq;
using OpenUtau.Core.Render;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    /// <summary>
    /// Builds the phrase mel by extracting a mel spectrogram from each phone's oto source
    /// slice independently, time-stretching it per phone (reusing the exact stretch logic in
    /// <see cref="HifiPhraseFeatureBuilder"/>), then assembling the stretched phone mels onto the
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
        const double RestGapToleranceMs = 8.0;
        const double RestReleaseGuardMs = 18.0;
        const double IsolatedLeadCatchupMaxMs = 80.0;
        const double IsolatedLeadCatchupPreutterRatio = 0.55;

        readonly HifiMelExtractor melExtractor = HifiMelExtractor.Shared;

        sealed class PhoneMelSegment {
            public int PhoneIndex;
            public string Phoneme = string.Empty;
            public RenderPhone Phone = null!;
            public float[,] Mel = new float[HifiMelExtractor.NMels, 0];
            public int StartFrame;
            public int FrameCount;
            public int OverlapFramesWithPrev;
            public int FixedFrames;
            public int F0MaskFrames;
            public double SourceSkipOverMs;
            public int SourceStartOffsetFrames;
            public string Strategy = string.Empty;
            public HifiPhoneFeatureDiagnostic? Diagnostic;
            public HifiFrameParameterAverages Parameters;
            public HifiHnsepProcessingReport HnsepReport;
        }

        /// <summary>
        /// Build the full phrase mel [NMels, targetFrames] from per-phone source slices.
        /// The assembly report records each phone's fixed leading span for diagnostics only.
        /// HIFI-NEURA does not zero F0 on consonants because this NSF vocoder treats F0=0 as
        /// silence, not noise excitation.
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
            var hnsepCache = new HifiHnsepSourceCache();
            for (int i = 0; i < phrase.phones.Length; i++) {
                var phone = phrase.phones[i];
                var segment = BuildPhoneSegment(phrase, phone, i, phraseStartMs, targetFrames, targetF0, sourceCache, sliceMelCache, hnsepCache);
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
                if (seg.FixedFrames > 0) {
                    int start = Math.Clamp(seg.StartFrame, 0, targetFrames);
                    int end = Math.Clamp(seg.StartFrame + seg.FixedFrames, start, targetFrames);
                    if (end > start) {
                        report.ConsonantFrameRanges.Add((start, end));
                    }
                }
                if (seg.Diagnostic != null) {
                    report.PhoneDiagnostics.Add(seg.Diagnostic);
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
            Dictionary<string, float[,]> sliceMelCache,
            HifiHnsepSourceCache hnsepCache) {
            if (phone.oto == null || string.IsNullOrWhiteSpace(phone.oto.File)) {
                return null;
            }

            // Placement stays on OpenUtau's raw preutter anchor so the phrase grid remains
            // continuous. The shortened target lead is only used inside the phone mel mapper.
            int startFrame = MsToFrame(phone.positionMs - phone.preutterMs - phraseStartMs);
            startFrame = Math.Clamp(startFrame, 0, Math.Max(0, targetFrames - 1));

            bool hasNextPhone = phoneIndex + 1 < phrase.phones.Length;
            double nextAnchorMs;
            if (hasNextPhone) {
                var next = phrase.phones[phoneIndex + 1];
                nextAnchorMs = next.positionMs - next.preutterMs;
            } else {
                nextAnchorMs = phraseStartMs + targetFrames * HifiF0Builder.FrameMs;
            }
            int nextAnchorFrame = MsToFrame(nextAnchorMs - phraseStartMs);
            nextAnchorFrame = Math.Clamp(nextAnchorFrame, startFrame + 1, targetFrames);

            // Overlap with the next segment: the next phone's overlap window (overlapMs) is the
            // region where both phones sound. We extend this segment past the next anchor by that
            // overlap so the cross-fade has frames to work with.
            int overlapTailFrames = 0;
            if (hasNextPhone) {
                double nextOverlapMs = Math.Max(0, phrase.phones[phoneIndex + 1].overlapMs);
                overlapTailFrames = Math.Clamp(
                    (int)Math.Round(nextOverlapMs / HifiF0Builder.FrameMs),
                    0,
                    Math.Max(0, targetFrames - nextAnchorFrame));
            }

            bool hasRestGap = hasNextPhone && nextAnchorMs - phone.endMs > RestGapToleranceMs;
            int segmentEndFrame = ResolveSegmentEndFrame(
                startFrame,
                nextAnchorFrame,
                overlapTailFrames,
                targetFrames,
                hasNextPhone,
                hasRestGap,
                ResolvePhoneReleaseEndFrame(phone, phraseStartMs),
                ResolveCorrectedEnvelopeEndFrame(phone, phraseStartMs));
            int frameCount = Math.Max(1, segmentEndFrame - startFrame);
            if (frameCount <= 0) {
                return null;
            }

            var parameterTrack = HifiParameterCurves.TrackForFrames(phrase, phraseStartMs, startFrame, frameCount);
            var parameters = parameterTrack.Average;
            float[] localTargetF0 = SliceTargetF0(targetF0, startFrame, frameCount);
            float[] fullSourceSamples = LoadSourceFile(phone.oto.File, sourceCache);
            float[] sourceSamples = SliceWithOto(fullSourceSamples, phone);
            double autoLeadCatchupMs = ResolveIsolatedLeadCatchupMs(phrase, phone, phoneIndex);
            var sourceParameterTrack = BuildHnsepSourceParameterTrack(parameterTrack, sourceSamples.Length, frameCount, phone, autoLeadCatchupMs);
            sourceSamples = HifiHnsepSourceProcessor.Apply(phone, phone.oto.File, fullSourceSamples, sourceSamples, sourceParameterTrack, hnsepCache, out var hnsepReport);
            float[,] sourceMel = LoadSliceMel(phone, sourceSamples, sliceMelCache, parameterTrack, sourceParameterTrack);
            int sourceFrames = sourceMel.GetLength(1);
            if (sourceFrames <= 0) {
                return null;
            }

            var phoneMel = new float[HifiMelExtractor.NMels, frameCount];
            // Local target F0 slice so the (F0-aware) stretch logic sees the right pitch motion.
            var report = HifiPhraseFeatureBuilder.WritePhoneMappedSegment(
                sourceMel,
                0,
                sourceFrames,
                phoneMel,
                0,
                frameCount,
                phone,
                localTargetF0,
                sourceSamples,
                parameters.GenderKeyShiftSemitones,
                phone.hifiSustainMode,
                autoLeadCatchupMs);
            var diagnostic = HifiClickDiagnostic.BuildPhoneFeatureDiagnostic(
                phoneIndex,
                phone.phoneme,
                startFrame,
                frameCount,
                sourceSamples,
                sourceMel,
                phoneMel,
                localTargetF0,
                report.Strategy);

            return new PhoneMelSegment {
                PhoneIndex = phoneIndex,
                Phoneme = phone.phoneme,
                Phone = phone,
                Mel = phoneMel,
                StartFrame = startFrame,
                FrameCount = frameCount,
                FixedFrames = report.FixedTargetFrames,
                F0MaskFrames = report.F0MaskFrames,
                SourceSkipOverMs = report.SourceSkipOverMs,
                SourceStartOffsetFrames = report.SourceStartOffsetFrames,
                Strategy = report.Strategy,
                Diagnostic = diagnostic,
                Parameters = parameters,
                HnsepReport = hnsepReport,
            };
        }

        static HifiFrameParameterTrack BuildHnsepSourceParameterTrack(
            HifiFrameParameterTrack targetTrack,
            int sourceSampleCount,
            int targetFrameCount,
            RenderPhone phone,
            double autoLeadCatchupMs) {
            if (!targetTrack.NeedsHnsep || sourceSampleCount <= 0 || targetFrameCount <= 1) {
                return targetTrack;
            }
            int sourceFrameCount = HifiMelExtractor.EstimateFrameCount(sourceSampleCount);
            if (sourceFrameCount <= 1) {
                return targetTrack;
            }
            var targetToSourceFrameMap = HifiPhraseFeatureBuilder.BuildPhoneTargetToSourceFrameMap(
                sourceFrameCount,
                targetFrameCount,
                phone,
                autoLeadCatchupMs);
            if (targetToSourceFrameMap.Length != targetTrack.FrameCount) {
                return targetTrack;
            }
            var sourceTrack = targetTrack.ProjectToSourceFrames(targetToSourceFrameMap, sourceFrameCount);
            Log.Debug(
                "HifiMelPhraseAssembler hnsep_nonlinear_source_params phoneme={Phoneme} target_frames={TargetFrames} source_frames={SourceFrames} source_samples={SourceSamples}",
                phone.phoneme,
                targetFrameCount,
                sourceFrameCount,
                sourceSampleCount);
            return sourceTrack;
        }

        static double ResolveIsolatedLeadCatchupMs(RenderPhrase phrase, RenderPhone phone, int phoneIndex) {
            if (phone.oto == null || phone.oto.Preutter <= 0) {
                return 0;
            }
            double sourcePreutterMs = Math.Max(0, phone.oto.Preutter);
            double sourceOverlapMs = Math.Max(0, phone.oto.Overlap);
            if (sourcePreutterMs <= HifiF0Builder.FrameMs || sourceOverlapMs <= HifiF0Builder.FrameMs * 0.25) {
                return 0;
            }
            if (HasPreviousAcousticOverlap(phrase, phone, phoneIndex)) {
                return 0;
            }

            double catchupMs = Math.Min(sourceOverlapMs, sourcePreutterMs * IsolatedLeadCatchupPreutterRatio);
            catchupMs = Math.Min(catchupMs, IsolatedLeadCatchupMaxMs);
            if (catchupMs <= HifiF0Builder.FrameMs * 0.5) {
                return 0;
            }
            Log.Debug(
                "HifiMelPhraseAssembler isolated_lead_catchup phone_index={Index} phoneme={Phoneme} source_preutter_ms={PreutterMs:F2} source_overlap_ms={OverlapMs:F2} catchup_ms={CatchupMs:F2}",
                phoneIndex,
                phone.phoneme,
                sourcePreutterMs,
                sourceOverlapMs,
                catchupMs);
            return catchupMs;
        }

        static bool HasPreviousAcousticOverlap(RenderPhrase phrase, RenderPhone phone, int phoneIndex) {
            if (phoneIndex <= 0 || phoneIndex >= phrase.phones.Length) {
                return false;
            }
            var previous = phrase.phones[phoneIndex - 1];
            double noteGapMs = phone.positionMs - previous.endMs;
            if (noteGapMs > RestGapToleranceMs) {
                return false;
            }
            double targetOverlapMs = Math.Max(0, phone.overlapMs);
            return targetOverlapMs > HifiF0Builder.FrameMs * 0.5;
        }

        internal static int ResolveSegmentEndFrame(
            int startFrame,
            int nextAnchorFrame,
            int overlapTailFrames,
            int targetFrames,
            bool hasNextPhone,
            bool hasRestGap,
            int phoneReleaseEndFrame,
            int correctedEnvelopeEndFrame) {
            targetFrames = Math.Max(0, targetFrames);
            if (targetFrames == 0) {
                return 0;
            }
            startFrame = Math.Clamp(startFrame, 0, Math.Max(0, targetFrames - 1));
            int overlapEndFrame = Math.Clamp(nextAnchorFrame + Math.Max(0, overlapTailFrames), startFrame + 1, targetFrames);
            if (!hasNextPhone || !hasRestGap) {
                return overlapEndFrame;
            }

            // In a real rest gap, do not fill silence up to the next phone anchor. Keep a short
            // release guard, and let the corrected OTO envelope end participate only as an upper
            // bound for rest handling. Connected phones must not be hard-clipped by envelope[4].
            int restEndFrame = Math.Clamp(nextAnchorFrame, startFrame + 1, targetFrames);
            if (phoneReleaseEndFrame > 0) {
                restEndFrame = Math.Min(restEndFrame, Math.Clamp(phoneReleaseEndFrame, startFrame + 1, targetFrames));
            }
            if (correctedEnvelopeEndFrame > 0) {
                restEndFrame = Math.Min(restEndFrame, Math.Clamp(correctedEnvelopeEndFrame, startFrame + 1, targetFrames));
            }
            return Math.Clamp(restEndFrame, startFrame + 1, targetFrames);
        }

        static int ResolvePhoneReleaseEndFrame(RenderPhone phone, double phraseStartMs) {
            return MsToFrame(phone.endMs + RestReleaseGuardMs - phraseStartMs);
        }

        static int ResolveCorrectedEnvelopeEndFrame(RenderPhone phone, double phraseStartMs) {
            if (phone.envelope == null || phone.envelope.Length < 5) {
                return -1;
            }
            double envelopeLengthMs = phone.envelope[4].X - phone.envelope[0].X;
            if (envelopeLengthMs <= 0) {
                return -1;
            }
            double segmentStartMs = phone.positionMs - phone.leadingMs;
            return MsToFrame(segmentStartMs + envelopeLengthMs + RestReleaseGuardMs - phraseStartMs);
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
                        // Weights depend only on progress u, so compute once for all bins.
                        double u = Math.Clamp(CrossfadeProgress(t - overlapStart, overlapFrames), 0.0, 1.0);
                        double wNew = Math.Sin(0.5 * Math.PI * u);
                        double wOld = Math.Cos(0.5 * Math.PI * u);
                        double wOld2 = wOld * wOld;
                        double wNew2 = wNew * wNew;
                        for (int m = 0; m < bins; m++) {
                            double pOld = Math.Exp(output[m, t]);
                            double pNew = Math.Exp(seg.Mel[m, local]);
                            double mixed = pOld * wOld2 + pNew * wNew2;
                            output[m, t] = (float)Math.Log(Math.Max(mixed, 1e-5));
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
                int frameCount = Math.Max(1, end - start);
                int fixedFrames = Math.Clamp(seg.FixedFrames, 0, frameCount);
                int f0MaskFrames = Math.Clamp(seg.F0MaskFrames, 0, fixedFrames);
                report.Phones.Add(new HifiPhoneMetadata {
                    Index = seg.PhoneIndex,
                    Phoneme = phone.phoneme,
                    Tone = phone.tone,
                    SourceFile = phone.oto?.File ?? string.Empty,
                    PositionMs = phone.positionMs,
                    DurationMs = phone.durationMs,
                    LeadingMs = phone.leadingMs,
                    StartFrame = start,
                    FrameCount = frameCount,
                    FixedFrames = fixedFrames,
                    F0MaskFrames = f0MaskFrames,
                    ConsonantStartFrame = start,
                    ConsonantFrameCount = fixedFrames,
                    SourceSkipOverMs = seg.SourceSkipOverMs,
                    SourceStartOffsetFrames = seg.SourceStartOffsetFrames,
                    Parameters = new HifiPhoneParameterMetadata {
                        Gender = seg.Parameters.Gender,
                        Breathiness = seg.Parameters.Breathiness,
                        Tension = seg.Parameters.Tension,
                        Voicing = seg.Parameters.Voicing,
                        GenderKeyShiftSemitones = seg.Parameters.GenderKeyShiftSemitones,
                        BreathNoiseGain = seg.Parameters.BreathNoiseGain,
                        VoicingGain = seg.Parameters.VoicingGain,
                        HnsepRequested = seg.HnsepReport.Requested,
                        HnsepApplied = seg.HnsepReport.Applied,
                        HnsepReason = seg.HnsepReport.Reason,
                    },
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
        /// voiced segments preserves energy through the overlap; this keeps VCV/CVVC vowel
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
            Dictionary<string, float[,]> sliceMelCache,
            HifiFrameParameterTrack parameterTrack,
            HifiFrameParameterTrack sourceParameterTrack) {
            string key = SliceCacheKey(phone, parameterTrack, sourceParameterTrack);
            if (key.Length > 0 && sliceMelCache.TryGetValue(key, out var cachedMel)) {
                return cachedMel;
            }
            var parameters = parameterTrack.Average;
            float[,] mel = sourceSamples.Length == 0
                ? new float[HifiMelExtractor.NMels, 0]
                : melExtractor.Extract(sourceSamples, parameters.GenderKeyShiftSemitones);
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

        static string SliceCacheKey(
            RenderPhone phone,
            HifiFrameParameterTrack parameterTrack,
            HifiFrameParameterTrack sourceParameterTrack) {
            if (phone.oto == null || string.IsNullOrWhiteSpace(phone.oto.File)) {
                return string.Empty;
            }
            var parameters = parameterTrack.Average;
            // Offset+Cutoff fully determine the sample slice taken from the file, so two phones
            // sharing them (same oto entry) share the extracted mel.
            return string.Concat(
                phone.oto.File,
                "|", phone.oto.Offset.ToString("R"),
                "|", phone.oto.Cutoff.ToString("R"),
                "|g", Quantize(parameters.GenderKeyShiftSemitones),
                "|trk", parameterTrack.CacheKey,
                "|src", sourceParameterTrack.CacheKey);
        }

        static string Quantize(double value) {
            return Math.Round(value, 3).ToString("R");
        }

        static float[] LoadSourceFile(string file, Dictionary<string, float[]> sourceCache) {
            if (string.IsNullOrWhiteSpace(file) || !System.IO.File.Exists(file)) {
                return Array.Empty<float>();
            }
            if (!sourceCache.TryGetValue(file, out var full)) {
                full = melExtractorLoad(file);
                sourceCache[file] = full;
            }
            return full;
        }

        static float[] melExtractorLoad(string file) {
            return HifiMelExtractor.Shared.LoadMono(file);
        }

        internal static float[] SliceWithOto(float[] source, RenderPhone phone) {
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
