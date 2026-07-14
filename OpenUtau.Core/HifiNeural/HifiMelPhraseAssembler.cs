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
        const float LinearFloor = 1e-5f;
        const int SampleRate = HifiMelExtractor.SampleRate;
        const double RestGapToleranceMs = 8.0;
        const double RestReleaseGuardMs = 18.0;
        const double IsolatedLeadCatchupMaxMs = 80.0;
        const double IsolatedLeadCatchupPreutterRatio = 0.55;
        // The phrase is assembled on a fine grid (one fine frame per source hop, 4x the vocoder
        // hop) so phone anchors land within ~1.5ms of their true position instead of being
        // quantized to the 11.6ms vocoder grid, and boundary cross-fades get sub-frame resolution.
        // The fine buffer is mean-pooled 4:1 back to the vocoder grid at the end.
        const int FineRatio = HifiF0Builder.HopSize / HifiMelExtractor.OriginHopSize;
        const double FineFrameMs = HifiF0Builder.FrameMs / FineRatio;
        // Connected joints can need a tiny safety fade when OTO overlap is near zero, but forcing a
        // 3-frame fade on every boundary smears fast consonants. The minimum is resolved per joint.
        const int MaxAdaptiveCrossfadeFrames = 3;
        // Boundary spectral bias matching: measure the average log-mel step between the two sides
        // of a joint over this window, band-smooth it, and remove half of it from each side over
        // a short vowel-only ramp so timbre/level differences between source recordings become
        // glides without flattening fast consonant articulation.
        const int BoundaryBiasMeasureFineFrames = 16;
        const int BoundaryBiasDefaultRampFineFrames = 10;
        const int BoundaryBiasLongRampFineFrames = 16;
        const int BoundaryBiasMaxRampFineFrames = 20;
        const double BoundaryBiasMaxLog = 0.35; // per side, natural log (~3.0dB)
        const int BoundaryBiasBandRadius = 6;
        const double BoundaryBiasActiveFloor = LogFloor + 2.0;

        readonly HifiMelExtractor melExtractor = HifiMelExtractor.Shared;
        // Estimated recording pitch per oto slice; keyed by file identity + slice bounds so it
        // survives across phrases and only re-runs when the sample or oto timing changes.
        static readonly System.Collections.Concurrent.ConcurrentDictionary<string, double> sliceF0Cache = new();

        sealed class PhoneMelSegment {
            public int PhoneIndex;
            public string Phoneme = string.Empty;
            public RenderPhone Phone = null!;
            public float[,] Mel = new float[HifiMelExtractor.NMels, 0];
            public int StartFrame;
            public int FrameCount;
            // Sub-frame placement anchor on the fine (source-hop) grid; StartFrame stays the
            // coarse anchor used by timing plans, reports, and the leveler.
            public int StartFineFrame;
            public int OverlapFramesWithPrev;
            public int FixedFrames;
            public int F0MaskFrames;
            public double SourceSkipOverMs;
            public int SourceStartOffsetFrames;
            public double SourceF0Hz;
            public string Strategy = string.Empty;
            public HifiPhoneFeatureDiagnostic? Diagnostic;
            public HifiFrameParameterAverages Parameters;
            public HifiHnsepProcessingReport HnsepReport;
        }

        sealed class SourceMelCacheEntry {
            public required float[] Samples;
            public required float[,] Mel;
            public required HifiHnsepProcessingReport HnsepReport;
            public required bool CanShare;
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
            return Build(phrase, phraseStartMs, targetFrames, targetF0, sourceCache, out report, HifiRenderContext.None);
        }

        public float[,] Build(
            RenderPhrase phrase,
            double phraseStartMs,
            int targetFrames,
            float[] targetF0,
            Dictionary<string, float[]> sourceCache,
            out HifiMelAssemblyReport report,
            HifiRenderContext context) {
            report = new HifiMelAssemblyReport();
            var output = new float[HifiMelExtractor.NMels, Math.Max(0, targetFrames)];
            FillConstant(output, LogFloor);
            if (targetFrames <= 0 || phrase.phones.Length == 0) {
                return output;
            }

            var segments = new List<PhoneMelSegment>(phrase.phones.Length);
            // Source mels are cached per oto slice (File + Offset + Cutoff) so a phone/oto that
            // recurs within the phrase reuses its STFT instead of re-extracting it.
            var sliceMelCache = new Dictionary<string, SourceMelCacheEntry>(StringComparer.Ordinal);
            var hnsepCache = new HifiHnsepSourceCache();
            bool collectDiagnostics = HifiRenderConfig.DebugExportEnabled;
            for (int i = 0; i < phrase.phones.Length; i++) {
                context.ThrowIfCancellationRequested();
                var phone = phrase.phones[i];
                var segment = BuildPhoneSegment(
                    phrase,
                    phone,
                    i,
                    phraseStartMs,
                    targetFrames,
                    targetF0,
                    sourceCache,
                    sliceMelCache,
                    hnsepCache,
                    collectDiagnostics,
                    context);
                if (segment != null) {
                    segments.Add(segment);
                }
            }

            if (segments.Count == 0) {
                return output;
            }

            using (HifiRenderProfiler.Measure(HifiRenderStage.Assembly)) {
                AssembleWithOverlapCrossfade(output, segments, targetFrames);
            }
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
            Dictionary<string, SourceMelCacheEntry> sliceMelCache,
            HifiHnsepSourceCache hnsepCache,
            bool collectDiagnostics,
            HifiRenderContext context) {
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

            bool hasRestGap = hasNextPhone && nextAnchorMs - phone.endMs > RestGapToleranceMs;

            // Overlap with the next segment: the next phone's overlap window (overlapMs) is the
            // region where both phones sound. We extend this segment past the next anchor by that
            // overlap so the cross-fade has frames to work with. Only add a small adaptive safety
            // overlap when the boundary is vowel-like; fast consonant/short-note boundaries can
            // stay as hard joins because a forced fade smears articulation.
            int overlapTailFrames = 0;
            if (hasNextPhone) {
                var next = phrase.phones[phoneIndex + 1];
                double nextOverlapMs = Math.Max(0, next.overlapMs);
                overlapTailFrames = (int)Math.Round(nextOverlapMs / HifiF0Builder.FrameMs);
                if (!hasRestGap) {
                    int minOverlapFrames = ResolveAdaptiveMinimumCrossfadeFrames(phone, next);
                    overlapTailFrames = Math.Max(overlapTailFrames, minOverlapFrames);
                }
                overlapTailFrames = Math.Clamp(overlapTailFrames, 0, Math.Max(0, targetFrames - nextAnchorFrame));
            }
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
            // Waveform-domain slice analysis on the pre-HNSEP slice: the recording pitch feeds the
            // pitch-mismatch compensation, and the shared active-frame count keeps the HNSEP
            // parameter projection and the mel mapping trimming the same dead tail.
            double sourceF0Hz = EstimateSliceF0(phone, sourceSamples);
            int activeSourceFrames = HifiSourceAnalysis.EstimateActiveFrameCount(sourceSamples);
            var sourceParameterTrack = BuildHnsepSourceParameterTrack(parameterTrack, sourceSamples.Length, frameCount, phone, autoLeadCatchupMs, activeSourceFrames);
            var sourceFeatures = LoadSliceMel(
                phone,
                fullSourceSamples,
                sourceSamples,
                sliceMelCache,
                parameterTrack,
                sourceParameterTrack,
                hnsepCache,
                context);
            sourceSamples = sourceFeatures.Samples;
            float[,] sourceMel = sourceFeatures.Mel;
            var hnsepReport = sourceFeatures.HnsepReport;
            int sourceFrames = sourceMel.GetLength(1);
            if (sourceFrames <= 0) {
                return null;
            }

            var phoneMel = new float[HifiMelExtractor.NMels, frameCount];
            int sourceToneOverride = sourceF0Hz > 0
                ? Math.Max(1, (int)Math.Round(MusicMath.FreqToTone(sourceF0Hz)))
                : 0;
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
                autoLeadCatchupMs,
                activeSourceFrames,
                sourceToneOverride);
            HifiPhoneFeatureDiagnostic? diagnostic = collectDiagnostics
                ? HifiClickDiagnostic.BuildPhoneFeatureDiagnostic(
                    phoneIndex,
                    phone.phoneme,
                    startFrame,
                    frameCount,
                    sourceSamples,
                    sourceMel,
                    phoneMel,
                    localTargetF0,
                    report.Strategy)
                : null;

            return new PhoneMelSegment {
                PhoneIndex = phoneIndex,
                Phoneme = phone.phoneme,
                Phone = phone,
                Mel = phoneMel,
                StartFrame = startFrame,
                FrameCount = frameCount,
                StartFineFrame = ResolveStartFineFrame(phone, phraseStartMs, startFrame, targetFrames),
                FixedFrames = report.FixedTargetFrames,
                F0MaskFrames = report.F0MaskFrames,
                SourceSkipOverMs = report.SourceSkipOverMs,
                SourceStartOffsetFrames = report.SourceStartOffsetFrames,
                SourceF0Hz = sourceF0Hz,
                Strategy = report.Strategy,
                Diagnostic = diagnostic,
                Parameters = parameters,
                HnsepReport = hnsepReport,
            };
        }

        static int ResolveAdaptiveMinimumCrossfadeFrames(RenderPhone left, RenderPhone right) {
            int leftFrames = Math.Max(1, MsToFrame(left.durationMs));
            int rightFrames = Math.Max(1, MsToFrame(right.durationMs));
            int shorterFrames = Math.Min(leftFrames, rightFrames);
            if (shorterFrames <= 2) {
                return 0;
            }

            int rightFixedFrames = EstimateTargetFixedFrames(right);
            bool rightHasTransientHead = rightFixedFrames >= 2
                || HifiPhraseFeatureBuilder.ResolveTargetFixedMs(right) >= HifiF0Builder.FrameMs * 1.25;
            if (rightHasTransientHead) {
                return shorterFrames <= 5 ? 0 : 1;
            }

            bool leftHasVowelBody = leftFrames - EstimateTargetFixedFrames(left) >= 4;
            bool rightHasVowelBody = rightFrames - rightFixedFrames >= 4;
            if (leftHasVowelBody && rightHasVowelBody) {
                if (shorterFrames >= 12) {
                    return MaxAdaptiveCrossfadeFrames;
                }
                return shorterFrames >= 7 ? 2 : 1;
            }
            return shorterFrames <= 5 ? 0 : 1;
        }

        static int EstimateTargetFixedFrames(RenderPhone phone) {
            double fixedMs = HifiPhraseFeatureBuilder.ResolveTargetFixedMs(phone);
            if (fixedMs <= 0 || double.IsNaN(fixedMs) || double.IsInfinity(fixedMs)) {
                return 0;
            }
            return Math.Max(0, (int)Math.Round(fixedMs / HifiF0Builder.FrameMs));
        }

        static int ResolveStartFineFrame(RenderPhone phone, double phraseStartMs, int startFrame, int targetFrames) {
            int fineFrames = targetFrames * FineRatio;
            double anchorMs = phone.positionMs - phone.preutterMs - phraseStartMs;
            int fine = (int)Math.Round(anchorMs / FineFrameMs);
            // Stay within one coarse frame of the coarse anchor so the segment content and the
            // coarse timing plan (FrameCount) cannot drift apart.
            fine = Math.Clamp(fine, (startFrame - 1) * FineRatio, (startFrame + 1) * FineRatio);
            return Math.Clamp(fine, 0, Math.Max(0, fineFrames - 1));
        }

        static double EstimateSliceF0(RenderPhone phone, float[] sourceSamples) {
            if (phone.oto == null || string.IsNullOrWhiteSpace(phone.oto.File) || sourceSamples.Length == 0) {
                return 0;
            }
            string key;
            try {
                var info = new System.IO.FileInfo(phone.oto.File);
                key = string.Concat(
                    info.FullName,
                    "|", info.Length,
                    "|", info.LastWriteTimeUtc.Ticks,
                    "|", phone.oto.Offset.ToString("R"),
                    "|", phone.oto.Cutoff.ToString("R"));
            } catch {
                key = string.Concat(
                    phone.oto.File,
                    "|", phone.oto.Offset.ToString("R"),
                    "|", phone.oto.Cutoff.ToString("R"));
            }
            return sliceF0Cache.GetOrAdd(key, _ => HifiSourceAnalysis.EstimateF0Hz(sourceSamples));
        }

        static HifiFrameParameterTrack BuildHnsepSourceParameterTrack(
            HifiFrameParameterTrack targetTrack,
            int sourceSampleCount,
            int targetFrameCount,
            RenderPhone phone,
            double autoLeadCatchupMs,
            int activeSourceFrames) {
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
                autoLeadCatchupMs,
                activeSourceFrames);
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
            int fineFrames = targetFrames * FineRatio;
            if (fineFrames <= 0 || segments.Count == 0) {
                return;
            }

            // Fine buffer holds linear magnitudes: blending and pooling stay physical, and the
            // final log happens once per pooled frame.
            var fine = new float[bins, fineFrames];
            FillConstant(fine, LinearFloor);

            int prevFineEnd = 0;
            int prevFineStart = 0;
            var newColumn = new float[bins];
            var biasHalf = new double[bins];

            for (int s = 0; s < segments.Count; s++) {
                var seg = segments[s];
                int segFineLen = seg.FrameCount * FineRatio;
                int fineStart = Math.Clamp(seg.StartFineFrame, 0, fineFrames);
                bool connected = s > 0
                    && seg.StartFrame <= segments[s - 1].StartFrame + segments[s - 1].FrameCount;
                if (connected && fineStart > prevFineEnd) {
                    // Sub-frame rounding opened a hole at a connected joint; pull the segment back
                    // so the fade never crosses a silent sliver.
                    fineStart = prevFineEnd;
                }
                int fineEnd = Math.Min(fineFrames, fineStart + segFineLen);
                if (fineEnd <= fineStart) {
                    seg.OverlapFramesWithPrev = 0;
                    continue;
                }

                int overlapEnd = s == 0 ? fineStart : Math.Min(fineEnd, prevFineEnd);
                int overlapLen = Math.Max(0, overlapEnd - fineStart);
                // A segment fully inside the previous one blends in and back out (bump) instead of
                // ending on 100% new content one frame before the old content resumes.
                bool contained = s > 0 && overlapLen > 0 && fineEnd <= prevFineEnd;
                seg.OverlapFramesWithPrev = (int)Math.Round(overlapLen / (double)FineRatio);

                var leftSeg = s > 0 ? segments[s - 1] : null;
                int biasRampFrames = leftSeg != null
                    ? ResolveBoundaryBiasRampFineFrames(leftSeg, seg, overlapLen, fineStart - prevFineStart, fineEnd - fineStart)
                    : 0;
                bool hasBias = connected && biasRampFrames > 0
                    && TryMeasureBoundaryBias(fine, seg, fineStart, overlapLen, biasHalf);
                if (hasBias) {
                    int oldRampFrames = Math.Clamp(fineStart - prevFineStart, 0, biasRampFrames);
                    for (int t = Math.Max(0, fineStart - oldRampFrames); t < overlapEnd; t++) {
                        double w = t >= fineStart
                            ? 1.0
                            : (t - (fineStart - oldRampFrames) + 1) / (double)(oldRampFrames + 1);
                        for (int m = 0; m < bins; m++) {
                            fine[m, t] = (float)Math.Max(LinearFloor, fine[m, t] * Math.Exp(-biasHalf[m] * w));
                        }
                    }
                }
                int newRampFrames = hasBias
                    ? Math.Min(biasRampFrames, (fineEnd - fineStart) / 2)
                    : 0;

                int lastLocalIdx = -1;
                for (int t = fineStart; t < fineEnd; t++) {
                    // Nearest-neighbour along time: aligned segments pool back to their exact
                    // coarse frames, misaligned ones become a clean sub-frame shift after pooling.
                    int localIdx = Math.Min(seg.FrameCount - 1, (t - fineStart) / FineRatio);
                    if (localIdx != lastLocalIdx) {
                        for (int m = 0; m < bins; m++) {
                            newColumn[m] = (float)Math.Exp(seg.Mel[m, localIdx]);
                        }
                        lastLocalIdx = localIdx;
                    }
                    double newBiasW = newRampFrames > 0 && t - fineStart < newRampFrames
                        ? 1.0 - (t - fineStart) / (double)newRampFrames
                        : 0;
                    bool inOverlap = s > 0 && t < overlapEnd && overlapLen > 0;
                    double wOld = 0;
                    double wNew = 1;
                    if (inOverlap) {
                        // (offset+1)/(len+1) keeps every overlap frame an actual mix; the frames
                        // just outside the overlap supply the pure endpoints, so even a 1-2 frame
                        // overlap fades instead of butt-splicing.
                        double u = (t - fineStart + 1) / (double)(overlapLen + 1);
                        if (contained) {
                            u = Math.Sin(Math.PI * u);
                        }
                        wNew = Math.Sin(0.5 * Math.PI * u);
                        wOld = Math.Cos(0.5 * Math.PI * u);
                    }
                    for (int m = 0; m < bins; m++) {
                        double value = newColumn[m];
                        if (newBiasW > 0) {
                            value *= Math.Exp(biasHalf[m] * newBiasW);
                        }
                        if (inOverlap) {
                            double old = fine[m, t];
                            // Equal-power blend of magnitudes: two uncorrelated equal-level vowels
                            // keep constant energy through the fade instead of dipping mid-way.
                            value = Math.Sqrt(wOld * wOld * old * old + wNew * wNew * value * value);
                        }
                        fine[m, t] = (float)Math.Max(LinearFloor, value);
                    }
                }
                prevFineStart = fineStart;
                prevFineEnd = Math.Max(prevFineEnd, fineEnd);
            }

            // 4:1 mean-pool (linear domain) back onto the vocoder grid.
            for (int frame = 0; frame < targetFrames; frame++) {
                int first = frame * FineRatio;
                for (int m = 0; m < bins; m++) {
                    double sum = 0;
                    for (int k = 0; k < FineRatio; k++) {
                        sum += fine[m, first + k];
                    }
                    output[m, frame] = (float)Math.Log(Math.Max(sum / FineRatio, 1e-5));
                }
            }
        }

        // Returns the fine-frame ramp length for boundary bias matching. Zero means the boundary is
        // too short or too consonant/transient-heavy and should only use normal overlap handling.
        static int ResolveBoundaryBiasRampFineFrames(
            PhoneMelSegment left,
            PhoneMelSegment right,
            int overlapLen,
            int leftFineFrames,
            int rightFineFrames) {
            if (overlapLen < Math.Max(2, FineRatio / 2) || !IsStableVowelBoundary(left, right)) {
                return 0;
            }

            int leftVowelFrames = Math.Max(0, left.FrameCount - Math.Clamp(left.FixedFrames, 0, left.FrameCount));
            int rightVowelFrames = Math.Max(0, right.FrameCount - Math.Clamp(right.FixedFrames, 0, right.FrameCount));
            int baseRamp = Math.Min(leftVowelFrames, rightVowelFrames) >= 12
                ? BoundaryBiasLongRampFineFrames
                : BoundaryBiasDefaultRampFineFrames;
            if (Math.Min(leftVowelFrames, rightVowelFrames) >= 20 && overlapLen >= FineRatio * 2) {
                baseRamp = BoundaryBiasMaxRampFineFrames;
            }

            int durationLimit = Math.Max(0, Math.Min(leftFineFrames, rightFineFrames) / 4);
            int overlapLimit = Math.Max(0, overlapLen * 2);
            int ramp = Math.Min(baseRamp, Math.Min(durationLimit, overlapLimit));
            return ramp >= 4 ? Math.Clamp(ramp, 0, BoundaryBiasMaxRampFineFrames) : 0;
        }

        static bool IsStableVowelBoundary(PhoneMelSegment left, PhoneMelSegment right) {
            if (left.FrameCount <= 5 || right.FrameCount <= 5) {
                return false;
            }
            int leftFixed = Math.Clamp(left.FixedFrames, 0, left.FrameCount);
            int rightFixed = Math.Clamp(right.FixedFrames, 0, right.FrameCount);
            int leftVowelFrames = left.FrameCount - leftFixed;
            int rightVowelFrames = right.FrameCount - rightFixed;
            if (leftVowelFrames < 4 || rightVowelFrames < 4) {
                return false;
            }
            // If the incoming phone still has a visible fixed/transient lead, keep its articulation
            // intact and leave only the overlap cross-fade to handle the joint.
            if (rightFixed >= 2 || rightFixed * 3 > right.FrameCount) {
                return false;
            }
            return true;
        }

        /// <summary>
        /// Measures the average per-bin log-mel step between the already-written old content and
        /// the incoming segment over the head of the overlap, band-smooths it so only the broad
        /// timbre/level trend remains, and returns half of it (each side of the joint absorbs
        /// half). Returns false when either side is close to silence.
        /// </summary>
        static bool TryMeasureBoundaryBias(
            float[,] fine,
            PhoneMelSegment seg,
            int fineStart,
            int overlapLen,
            double[] biasHalf) {
            int bins = fine.GetLength(0);
            int measure = Math.Min(BoundaryBiasMeasureFineFrames, overlapLen);
            if (measure < 2) {
                return false;
            }
            var oldMean = new double[bins];
            var newMean = new double[bins];
            for (int t = 0; t < measure; t++) {
                int localIdx = Math.Min(seg.FrameCount - 1, t / FineRatio);
                for (int m = 0; m < bins; m++) {
                    oldMean[m] += Math.Log(Math.Max(fine[m, fineStart + t], LinearFloor));
                    newMean[m] += seg.Mel[m, localIdx];
                }
            }
            double oldOverall = 0;
            double newOverall = 0;
            for (int m = 0; m < bins; m++) {
                oldMean[m] /= measure;
                newMean[m] /= measure;
                oldOverall += oldMean[m];
                newOverall += newMean[m];
            }
            oldOverall /= Math.Max(1, bins);
            newOverall /= Math.Max(1, bins);
            if (oldOverall <= BoundaryBiasActiveFloor || newOverall <= BoundaryBiasActiveFloor) {
                return false;
            }
            for (int m = 0; m < bins; m++) {
                double sum = 0;
                double weightSum = 0;
                int first = Math.Max(0, m - BoundaryBiasBandRadius);
                int last = Math.Min(bins - 1, m + BoundaryBiasBandRadius);
                for (int k = first; k <= last; k++) {
                    double distance = Math.Abs(k - m) / (double)(BoundaryBiasBandRadius + 1);
                    double weight = 0.5 + 0.5 * Math.Cos(Math.PI * distance);
                    sum += (oldMean[k] - newMean[k]) * weight;
                    weightSum += weight;
                }
                double step = weightSum > 0 ? sum / weightSum : 0;
                biasHalf[m] = Math.Clamp(step * 0.5, -BoundaryBiasMaxLog, BoundaryBiasMaxLog);
            }
            return true;
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
                    SourceF0Hz = seg.SourceF0Hz,
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
                    int boundaryFrame = Math.Clamp(
                        (int)Math.Round(seg.StartFineFrame / (double)FineRatio),
                        0,
                        Math.Max(0, targetFrames - 1));
                    report.Boundaries.Add(new HifiBoundaryMetadata {
                        Index = i - 1,
                        LeftPhoneIndex = left.PhoneIndex,
                        RightPhoneIndex = seg.PhoneIndex,
                        LeftPhone = left.Phoneme,
                        RightPhone = seg.Phoneme,
                        Frame = boundaryFrame,
                        PositionMs = phraseStartMs + seg.StartFineFrame * FineFrameMs,
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

        SourceMelCacheEntry LoadSliceMel(
            RenderPhone phone,
            float[] fullSourceSamples,
            float[] sourceSamples,
            Dictionary<string, SourceMelCacheEntry> sliceMelCache,
            HifiFrameParameterTrack parameterTrack,
            HifiFrameParameterTrack sourceParameterTrack,
            HifiHnsepSourceCache hnsepCache,
            HifiRenderContext context) {
            bool processingRequested = sourceParameterTrack.NeedsHnsep
                || phone.hifiHnSpectralProfile.HasAudibleEffect;
            string key = SliceCacheKey(phone, parameterTrack, sourceParameterTrack, processingRequested);
            if (key.Length > 0 && sliceMelCache.TryGetValue(key, out var localEntry)) {
                HifiRenderProfiler.Count(HifiRenderCounter.SourceMelHit);
                return localEntry;
            }
            var parameters = parameterTrack.Average;
            SourceMelCacheEntry BuildEntry() {
                context.ThrowIfCancellationRequested();
                float[] processedSamples = HifiHnsepSourceProcessor.Apply(
                    phone,
                    phone.oto.File,
                    fullSourceSamples,
                    sourceSamples,
                    sourceParameterTrack,
                    hnsepCache,
                    phone.hifiHnSpectralProfile,
                    out var hnsepReport,
                    context);
                using var timing = HifiRenderProfiler.Measure(HifiRenderStage.SourceMel);
                float[,] mel = processedSamples.Length == 0
                    ? new float[HifiMelExtractor.NMels, 0]
                    : melExtractor.Extract(processedSamples, parameters.GenderKeyShiftSemitones);
                return new SourceMelCacheEntry {
                    Samples = processedSamples,
                    Mel = mel,
                    HnsepReport = hnsepReport,
                    // A failed separation may recover on a later render. Reuse it inside this
                    // phrase, but do not let the fallback waveform poison the process-wide cache.
                    CanShare = !processingRequested || hnsepReport.Applied,
                };
            }

            SourceMelCacheEntry entry;
            if (key.Length > 0) {
                string sharedKey = "source-mel-v2|" + key;
                if (HifiRenderMemoryCache.Shared.TryGet(sharedKey, out SourceMelCacheEntry sharedEntry)) {
                    entry = sharedEntry;
                    HifiRenderProfiler.Count(HifiRenderCounter.SourceMelHit);
                } else {
                    entry = BuildEntry();
                    HifiRenderProfiler.Count(HifiRenderCounter.SourceMelMiss);
                    if (entry.CanShare) {
                        HifiRenderMemoryCache.Shared.AddOrRefresh(
                            sharedKey,
                            entry,
                            HifiRenderMemoryCache.FloatBytes(entry.Samples)
                                + HifiRenderMemoryCache.FloatBytes(entry.Mel));
                    }
                }
                sliceMelCache[key] = entry;
            } else {
                HifiRenderProfiler.Count(HifiRenderCounter.SourceMelMiss);
                entry = BuildEntry();
            }
            return entry;
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
            HifiFrameParameterTrack sourceParameterTrack,
            bool processingRequested) {
            if (phone.oto == null || string.IsNullOrWhiteSpace(phone.oto.File)) {
                return string.Empty;
            }
            var parameters = parameterTrack.Average;
            // Offset+Cutoff fully determine the sample slice taken from the file, so two phones
            // sharing them (same oto entry) share the extracted mel.
            string key = string.Concat(
                "v2-nfft", HifiMelExtractor.Nfft,
                "-hop", HifiMelExtractor.OriginHopSize,
                "-mels", HifiMelExtractor.NMels,
                "|", HifiRenderMemoryCache.FileVersionKey(phone.oto.File),
                "|", phone.oto.Offset.ToString("R"),
                "|", phone.oto.Cutoff.ToString("R"),
                "|g", Quantize(parameters.GenderKeyShiftSemitones));
            if (sourceParameterTrack.NeedsHnsep) {
                key += "|src" + sourceParameterTrack.CacheKey;
            }
            if (phone.hifiHnSpectralProfile.HasAudibleEffect) {
                key += "|hnprofile" + phone.hifiHnSpectralProfile.CacheKey();
            }
            if (processingRequested) {
                key += "|hnmodel" + HifiHnsepOnnx.CacheKeyOrDisabled();
            }
            return key;
        }

        static string Quantize(double value) {
            return Math.Round(value, 3).ToString("R");
        }

        static float[] LoadSourceFile(string file, Dictionary<string, float[]> sourceCache) {
            if (string.IsNullOrWhiteSpace(file) || !System.IO.File.Exists(file)) {
                return Array.Empty<float>();
            }
            if (sourceCache.TryGetValue(file, out var full)) {
                HifiRenderProfiler.Count(HifiRenderCounter.PcmHit);
                return full;
            }
            string key = "source-pcm-v1-sr" + HifiMelExtractor.SampleRate + "|" + HifiRenderMemoryCache.FileVersionKey(file);
            full = HifiRenderMemoryCache.Shared.GetOrAdd(
                key,
                () => {
                    using var timing = HifiRenderProfiler.Measure(HifiRenderStage.SourceDecode);
                    return melExtractorLoad(file);
                },
                HifiRenderMemoryCache.FloatBytes,
                out bool cacheHit);
            HifiRenderProfiler.Count(cacheHit ? HifiRenderCounter.PcmHit : HifiRenderCounter.PcmMiss);
            sourceCache[file] = full;
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
