using System;
using System.Collections.Generic;
using System.Linq;
using OpenUtau.Core.Render;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiPhraseFeatureBuilder {
        const int MinSourceFrames = 8;
        const int MinVowelSourceFrames = 4;
        const int MinVowelTargetFrames = 1;
        const double TargetFixedMaxMs = 72.0;
        const double TargetFixedShortNoteRatio = 0.60;
        const double TargetFixedDynamicShortNoteRatioRange = 0.22;
        const double TargetFixedSourceOnsetBonusMaxMs = 12.0;
        const double TargetFixedSourceOnsetBonusRatio = 0.25;
        const double TargetFixedOverlapRelease = 0.35;
        const double LeadSoftSkipBoostRatio = 0.12;
        const double LeadSoftSkipMaxSourceRatio = 0.08;
        const int LeadSoftSkipGraceFrames = 1;
        // Historical diagnostic ceiling for the old F0-mask span. The mask is not applied because
        // this NSF vocoder treats F0=0 as silence.
        const int ConsonantF0MaskMaxFrames = 5;
        // Disabled by default: OpenUtau pitch curves already carry intentional vibrato.
        const bool EnableF0MicroJitter = false;
        // Optional synthetic spectral wobble. Current default callers keep this off; extreme
        // sustains use real stable source frames instead of synthetic motion.
        const double SustainMicroVarLogAmp = 0.06;
        // How many distinct slowly-drifting wobble bands modulate the mel bins. Each band covers a
        // contiguous group of bins so neighbouring bins move together (formant-like), not as noise.
        const int SustainMicroVarBands = 6;
        // Peak F0 micro-jitter in cents if EnableF0MicroJitter is turned on for A/B testing.
        const double F0MicroJitterCents = 5.0;
        const double SustainTemplateStretchThreshold = 2.0;
        const double StableSustainEnergyClamp = 0.25;
        const int SustainTextureMinStableFrames = 4;
        const double SustainTextureNaturalStepFrames = HifiF0Builder.HopSize / (double)HifiMelExtractor.OriginHopSize;
        const double SustainTextureMaxStepFrames = SustainTextureNaturalStepFrames * 1.35;
        const double SustainTextureBodyAmount = 0.88;
        const double SustainTextureBodyClamp = 0.22;
        const double SustainTextureBodyLowBandResidual = 0.42;
        const int WaveformSustainMinOutputFrames = 8;
        const int WaveformSustainMinStableFrames = 8;
        const double WaveformSustainEnergyOffsetClamp = 0.10;
        const double WaveformSustainEnergyOutlierTolerance = 0.45;
        const double WaveformSustainEnergyOutlierClamp = 0.14;
        const double F0MismatchHighBandStart = 54;
        const double F0MismatchMaxCutDb = 0.75;
        const double F0MismatchMaxContrast = 0.06;
        const int PhraseEdgeMelFadeInFrames = 3;
        const int PhraseEdgeMelFadeOutFrames = 5;
        const double PhraseEdgeMelMinGain = 0.18;
        const int InactiveTailGuardFrames = 6;
        const double InactiveTailLogDrop = 2.8;
        const double F0MelCompDbPerOctave = 0.95;
        const double F0MelCompHighBandExtra = 0.30;
        const double F0MelCompMaxCutDb = 2.4;
        const double F0MelCompReferenceCapHz = 330.0;
        const int F0MelCompSmoothHalfFrames = 4;
        const double SourceFrameMs = 1000.0 * HifiMelExtractor.OriginHopSize / HifiMelExtractor.SampleRate;
        const double SourceSampleMs = 1000.0 / HifiMelExtractor.SampleRate;
        const int MinSourceSamples = MinSourceFrames * HifiMelExtractor.OriginHopSize;
        const int MinVowelSourceSamples = MinVowelSourceFrames * HifiMelExtractor.OriginHopSize;
        const int InactiveTailGuardSamples = InactiveTailGuardFrames * HifiMelExtractor.OriginHopSize;
        const double F0AwareMaxSlopeCentsPerFrame = 160.0;

        readonly HifiMelExtractor melExtractor = new HifiMelExtractor();
        readonly HifiF0Builder f0Builder = new HifiF0Builder();
        readonly HifiMelPhraseAssembler melAssembler = new HifiMelPhraseAssembler();
        readonly IHifiMelEnhancer melEnhancer;
        static readonly HifiMelExtractor sustainMelExtractor = new HifiMelExtractor();

        readonly record struct VowelMapReport(
            int LoopStartFrame,
            int LoopEndFrame,
            int LoopCount,
            int CrossfadeFrames,
            string Strategy,
            int TargetOnsetFrames,
            int TargetSustainFrames,
            int TargetReleaseFrames);

        readonly record struct TransientAnchorPlan(
            bool Enabled,
            int SourceAnchorFrame,
            int TargetAnchorFrame,
            double PeakFlux,
            double MedianFlux,
            string Reason);

        readonly record struct HifiPhoneTimingPlan(
            double? ConsonantMs,
            double TargetPreutterMs,
            double SourcePreutterMs,
            double SourceSoftSkipMs,
            int SourceSoftSkipFrames,
            int SourceLeadFrames,
            int SourceStableStartFrames,
            int SourceVowelOnsetFrames,
            int SourceVowelFrames,
            int TargetLeadFrames,
            int TargetFixedFrames,
            int TargetLeadOnsetFrames,
            int SourceFixedFrames,
            int SourceLeadOnsetFrames,
            int TargetVowelFrames);

        public HifiPhraseFeatureBuilder(IHifiMelEnhancer melEnhancer) {
            this.melEnhancer = melEnhancer;
        }

        public HifiPhraseFeatures Build(RenderPhrase phrase, RenderResult layout) {
            double phraseStartMs = layout.positionMs - layout.leadingMs;
            int targetFrames = Math.Max(1, (int)Math.Ceiling(layout.estimatedLengthMs / HifiF0Builder.FrameMs));
            float[] f0 = f0Builder.Build(phrase, targetFrames, phraseStartMs);
            if (EnableF0MicroJitter) {
                ApplyF0MicroJitter(f0);
            }

            // Mel-domain assembly: extract each phone's oto slice into its own mel, stretch it per
            // phone with the existing warp logic, then concatenate onto the phrase grid with
            // overlap cross-fades. Replaces the previous SharpWavtool rough + variable-position
            // mel sampling, which broke VCV/CVVC vowel boundaries under stretch.
            var sourceCache = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);
            float[,] alignedMel = melAssembler.Build(phrase, phraseStartMs, targetFrames, f0, sourceCache, out var assemblyReport);

            // NOTE: F0 is kept continuous across consonants on purpose. This NSF vocoder produces
            // silence (not noise) when F0 == 0, so masking the consonant F0 made those frames go
            // dead/mute. The consonant's unvoiced character comes from its mel spectrum; the F0
            // there just needs to be a smooth continuation of the neighbouring vowels' pitch.
            ApplyF0MelCompensation(alignedMel, f0);

            double targetDurationMs = targetFrames * HifiF0Builder.FrameMs;
            Log.Information(
                "HifiPhraseFeatureBuilder mel_domain_concat mode=overlap_neural target_duration_ms={TargetDurationMs:F3} target_frames={TargetFrames} phones={Phones}",
                targetDurationMs,
                targetFrames,
                phrase.phones.Length);

            float[,] enhancedMel = melEnhancer.Enhance(alignedMel, f0);
            ApplyPhraseEdgeMelGuard(enhancedMel);
            Validate(enhancedMel, f0);

            return new HifiPhraseFeatures {
                Mel = enhancedMel,
                F0 = f0,
                Metadata = BuildMetadata(phrase, layout, targetFrames, phraseStartMs, assemblyReport),
            };
        }

        // Adds a slow, deterministic +/-F0MicroJitterCents drift to voiced (F0>0) frames. Unvoiced
        // frames (F0==0) are left untouched. Deterministic so renders stay cache-stable.
        static void ApplyF0MicroJitter(float[] f0) {
            if (f0.Length == 0 || F0MicroJitterCents <= 0) {
                return;
            }
            double seed = SeedFromInts(f0.Length, 0x5f3759df);
            for (int i = 0; i < f0.Length; i++) {
                float hz = f0[i];
                if (hz <= 0 || float.IsNaN(hz) || float.IsInfinity(hz)) {
                    continue;
                }
                double cents = F0MicroJitterCents * SmoothWobble(i * 0.13, seed);
                f0[i] = (float)(hz * Math.Pow(2.0, cents / 1200.0));
            }
        }

        static void ApplyPhraseEdgeMelGuard(float[,] mel) {
            int frames = mel.GetLength(1);
            if (frames <= 1) {
                return;
            }
            int fadeIn = Math.Min(PhraseEdgeMelFadeInFrames, frames);
            int fadeOut = Math.Min(PhraseEdgeMelFadeOutFrames, frames);
            for (int t = 0; t < fadeIn; t++) {
                double progress = (t + 1) / (double)fadeIn;
                double gain = PhraseEdgeMelMinGain + (1 - PhraseEdgeMelMinGain) * SmoothStep(progress);
                ApplyMelGain(mel, t, gain);
            }
            for (int i = 0; i < fadeOut; i++) {
                int frame = frames - 1 - i;
                double progress = (i + 1) / (double)fadeOut;
                double gain = PhraseEdgeMelMinGain + (1 - PhraseEdgeMelMinGain) * SmoothStep(progress);
                ApplyMelGain(mel, frame, gain);
            }
        }

        static void ApplyMelGain(float[,] mel, int frame, double gain) {
            if (!IsFinite(gain) || gain <= 0) {
                return;
            }
            float logGain = (float)Math.Log(gain);
            int bins = mel.GetLength(0);
            frame = Math.Clamp(frame, 0, Math.Max(0, mel.GetLength(1) - 1));
            for (int m = 0; m < bins; m++) {
                mel[m, frame] += logGain;
            }
        }

        internal readonly record struct PhoneMapReport(
            bool SplitApplied,
            string Strategy,
            int FixedTargetFrames,
            int F0MaskFrames,
            double SourceSkipOverMs,
            int SourceStartOffsetFrames);

        internal static double[] BuildPhoneTargetToSourceFrameMap(int sourceFrames, int outputFrames, RenderPhone phone) {
            var map = new double[Math.Max(0, outputFrames)];
            if (map.Length == 0) {
                return map;
            }
            sourceFrames = Math.Max(1, sourceFrames);
            var timing = BuildPhoneTimingPlan(sourceFrames, outputFrames, phone);
            if (outputFrames <= 2) {
                WriteCompactPhoneFrameMap(map, 0, outputFrames, sourceFrames, timing.SourceLeadFrames);
                return map;
            }

            if (sourceFrames < MinSourceFrames) {
                WriteFrameMapRegion(map, 0, outputFrames, 0, sourceFrames);
                return map;
            }
            if (timing.SourceVowelFrames < MinVowelSourceFrames) {
                WriteFrameMapRegion(map, 0, outputFrames, 0, sourceFrames);
                return map;
            }

            if (timing.TargetLeadFrames > 0 && timing.SourceLeadFrames > 0) {
                WriteLeadFrameMap(
                    map,
                    0,
                    timing.TargetLeadFrames,
                    0,
                    timing.SourceLeadFrames,
                    timing.SourceSoftSkipFrames);
            }
            WriteVowelFrameMap(
                map,
                timing.TargetLeadFrames,
                timing.TargetVowelFrames,
                timing.SourceLeadFrames,
                timing.SourceVowelFrames,
                timing.SourceVowelOnsetFrames);
            return map;
        }

        internal static PhoneMapReport WritePhoneMappedSegment(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            RenderPhone phone,
            float[] targetF0) {
            return WritePhoneMappedSegment(
                sourceMel,
                sourceStart,
                sourceFrames,
                output,
                outputStart,
                outputFrames,
                phone,
                targetF0,
                null,
                0);
        }

        static void WriteCompactPhoneFrameMap(double[] map, int outputStart, int outputFrames, int sourceFrames, int sourceConsonantFrames) {
            if (outputFrames <= 0) {
                return;
            }
            sourceFrames = Math.Max(1, sourceFrames);
            int nucleus = Math.Clamp(sourceConsonantFrames > 0 ? sourceConsonantFrames : sourceFrames / 2, 0, sourceFrames - 1);
            if (outputFrames == 1) {
                map[outputStart] = nucleus;
                return;
            }
            int onset = sourceConsonantFrames > 0
                ? Math.Clamp(sourceConsonantFrames / 2, 0, sourceFrames - 1)
                : Math.Clamp((int)Math.Round(sourceFrames * 0.25), 0, sourceFrames - 1);
            map[outputStart] = onset;
            map[outputStart + 1] = nucleus;
        }

        static void WriteVowelFrameMap(
            double[] map,
            int outputStart,
            int outputFrames,
            int sourceStart,
            int sourceFrames,
            int preferredOnsetFrames) {
            if (outputFrames <= 0) {
                return;
            }
            ResolveVowelSections(sourceFrames, preferredOnsetFrames, out int onsetFrames, out int releaseFrames, out int sustainFrames);
            var (targetOnsetFrames, targetSustainFrames, targetReleaseFrames) = AllocateVowelTargetSections(
                onsetFrames,
                sustainFrames,
                releaseFrames,
                outputFrames);

            if (targetOnsetFrames > 0) {
                WriteFrameMapRegion(map, outputStart, targetOnsetFrames, sourceStart, onsetFrames);
            }
            if (targetSustainFrames > 0) {
                WriteSustainFrameMap(
                    map,
                    outputStart + targetOnsetFrames,
                    targetSustainFrames,
                    sourceStart + onsetFrames,
                    sustainFrames);
            }
            if (targetReleaseFrames > 0) {
                WriteFrameMapRegion(
                    map,
                    outputStart + targetOnsetFrames + targetSustainFrames,
                    targetReleaseFrames,
                    sourceStart + sourceFrames - releaseFrames,
                    releaseFrames);
            }
        }

        static void WriteLeadFrameMap(
            double[] map,
            int outputStart,
            int outputFrames,
            int sourceStart,
            int sourceFrames,
            int sourceSoftSkipFrames) {
            if (outputFrames <= 0) {
                return;
            }
            sourceFrames = Math.Max(1, sourceFrames);
            outputFrames = Math.Max(1, Math.Min(outputFrames, map.Length - outputStart));
            double previous = sourceStart;
            for (int t = 0; t < outputFrames; t++) {
                double sourceIndex = sourceStart + ResolveLeadSourceIndex(t, outputFrames, sourceFrames, sourceSoftSkipFrames);
                if (t > 0) {
                    sourceIndex = Math.Max(previous, sourceIndex);
                }
                previous = sourceIndex;
                map[outputStart + t] = sourceIndex;
            }
        }

        static double ResolveLeadSourceIndex(
            int targetFrame,
            int targetFrames,
            int sourceFrames,
            int sourceSoftSkipFrames) {
            sourceFrames = Math.Max(1, sourceFrames);
            if (targetFrames <= 1 || sourceFrames <= 1) {
                return 0;
            }
            double sourceMax = sourceFrames - 1.0;
            double u = Math.Clamp(targetFrame / (double)(targetFrames - 1), 0.0, 1.0);
            double linear = u * sourceMax;
            if (sourceSoftSkipFrames <= 0 || targetFrame <= LeadSoftSkipGraceFrames) {
                return linear;
            }
            double graceU = Math.Clamp(LeadSoftSkipGraceFrames / (double)(targetFrames - 1), 0.0, 0.8);
            double x = Math.Clamp((u - graceU) / Math.Max(1e-6, 1.0 - graceU), 0.0, 1.0);
            double boostShape = Math.Pow(Math.Sin(Math.PI * x), 2.0);
            double maxBoost = Math.Min(
                sourceSoftSkipFrames * LeadSoftSkipBoostRatio,
                sourceMax * LeadSoftSkipMaxSourceRatio);
            double boost = maxBoost * boostShape;
            return Math.Clamp(linear + boost, 0.0, sourceMax);
        }

        static void WriteSustainFrameMap(
            double[] map,
            int outputStart,
            int outputFrames,
            int sourceStart,
            int sourceFrames) {
            double stretchRatio = SourceToTargetDurationRatio(sourceFrames, outputFrames);
            if (sourceFrames <= 2 || stretchRatio <= 1.05) {
                WriteFrameMapRegion(map, outputStart, outputFrames, sourceStart, sourceFrames);
                return;
            }
            if (stretchRatio < SustainTemplateStretchThreshold) {
                WriteNaturalFrameMapRegion(map, outputStart, outputFrames, sourceStart, sourceFrames);
                return;
            }
            // Long sustain mel uses directContour as the main body and texture only as a residual.
            // Keep source-frame-aware parameters aligned to that main contour instead of the residual
            // wander path, otherwise HNSEP parameters drift away from the audible vowel body.
            WriteFrameMapRegion(map, outputStart, outputFrames, sourceStart, sourceFrames);
        }

        static void WriteNaturalFrameMapRegion(
            double[] map,
            int outputStart,
            int outputFrames,
            int sourceStart,
            int sourceFrames) {
            var transientAnchor = new TransientAnchorPlan(false, 0, 0, 0, 0, "parameter_map");
            double stretchRatio = SourceToTargetDurationRatio(sourceFrames, outputFrames);
            for (int t = 0; t < outputFrames; t++) {
                double targetNorm = outputFrames == 1 ? 0 : t / (double)(outputFrames - 1);
                double sourceNorm = MapSourceNormalized(targetNorm, stretchRatio, transientAnchor, sourceFrames, outputFrames);
                map[outputStart + t] = sourceStart + Math.Clamp(sourceNorm, 0, 1) * Math.Max(0, sourceFrames - 1);
            }
        }

        static void WriteSustainTextureFrameMap(
            double[] map,
            int outputStart,
            int outputFrames,
            int sourceStart,
            int sourceFrames,
            double stretchRatio) {
            int stableStart = Math.Max(0, sourceFrames / 5);
            int stableEnd = sourceFrames - Math.Max(0, sourceFrames / 5);
            if (stableEnd <= stableStart) {
                stableStart = 0;
                stableEnd = sourceFrames;
            }
            int stableFrames = Math.Max(1, stableEnd - stableStart);
            if (stableFrames < SustainTextureMinStableFrames) {
                WriteNaturalFrameMapRegion(map, outputStart, outputFrames, sourceStart, sourceFrames);
                return;
            }

            int edgeFrames = ResolveSustainEdgeFrames(outputFrames);
            double seed = SeedFromInts(sourceStart, outputFrames);
            double previousTextureIndex = stableStart + (stableFrames - 1) * 0.5;
            for (int t = 0; t < outputFrames; t++) {
                double u = outputFrames == 1 ? 0 : t / (double)(outputFrames - 1);
                double directIndex = outputFrames == 1 || sourceFrames == 1
                    ? 0
                    : u * (sourceFrames - 1);
                double textureIndex = ResolveSustainTextureSourceIndex(
                    t,
                    outputFrames,
                    stableStart,
                    stableFrames,
                    stretchRatio,
                    seed);
                if (t > 0) {
                    textureIndex = LimitTextureIndexStep(previousTextureIndex, textureIndex, stretchRatio);
                }
                previousTextureIndex = textureIndex;
                double coreWeight = SustainCoreTextureWeight(t, outputFrames, edgeFrames);
                map[outputStart + t] = sourceStart + directIndex * (1.0 - coreWeight) + textureIndex * coreWeight;
            }
        }

        static void WriteFrameMapRegion(
            double[] map,
            int outputStart,
            int outputFrames,
            int sourceStart,
            int sourceFrames) {
            if (outputFrames <= 0) {
                return;
            }
            sourceFrames = Math.Max(1, sourceFrames);
            outputFrames = Math.Max(1, Math.Min(outputFrames, map.Length - outputStart));
            for (int t = 0; t < outputFrames; t++) {
                double sourceIndex = outputFrames == 1 || sourceFrames == 1
                    ? 0
                    : t * (sourceFrames - 1.0) / (outputFrames - 1);
                map[outputStart + t] = sourceStart + sourceIndex;
            }
        }

        internal static PhoneMapReport WritePhoneMappedSegment(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            RenderPhone phone,
            float[] targetF0,
            float[]? sourceSamples,
            double sourceKeyShiftSemitones = 0) {
            sourceFrames = Math.Max(1, Math.Min(sourceFrames, sourceMel.GetLength(1) - sourceStart));
            outputFrames = Math.Max(1, Math.Min(outputFrames, output.GetLength(1) - outputStart));
            var initialTiming = BuildPhoneTimingPlan(sourceFrames, outputFrames, phone);
            sourceFrames = TrimInactiveTailFrames(sourceMel, sourceStart, sourceFrames, initialTiming.SourceLeadFrames, phone.phoneme);
            var timing = BuildPhoneTimingPlan(sourceFrames, outputFrames, phone);
            if (outputFrames <= 2) {
                WriteCompactPhoneRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames, timing.SourceLeadFrames);
                return new PhoneMapReport(false, "compact_short_target", 0, 0, timing.SourceSoftSkipMs, 0);
            }
            if (sourceFrames < MinSourceFrames) {
                WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return new PhoneMapReport(false, "simple_short_source", 0, 0, timing.SourceSoftSkipMs, 0);
            }
            if (timing.SourceVowelFrames < MinVowelSourceFrames) {
                WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return new PhoneMapReport(false, "simple_no_vowel_room", 0, 0, timing.SourceSoftSkipMs, 0);
            }

            if (timing.SourceLeadFrames > 0 && timing.TargetLeadFrames > 0) {
                WriteLeadMappedRegion(
                    sourceMel,
                    sourceStart,
                    timing.SourceLeadFrames,
                    output,
                    outputStart,
                    timing.TargetLeadFrames,
                    timing.SourceSoftSkipFrames);
            }

            var vowelMap = WriteVowelSourceToTargetMap(
                sourceMel,
                sourceStart + timing.SourceLeadFrames,
                timing.SourceVowelFrames,
                output,
                outputStart + timing.TargetLeadFrames,
                timing.TargetVowelFrames,
                targetF0,
                phone.phoneme,
                timing.SourceVowelOnsetFrames,
                sourceSamples,
                phone.tone,
                sourceKeyShiftSemitones);

            double consonantRatio = timing.SourceLeadFrames > 0
                ? (timing.TargetFixedFrames * HifiF0Builder.FrameMs) / Math.Max(SourceFrameMs, timing.SourceLeadFrames * SourceFrameMs)
                : 0;
            double vowelRatio = (timing.TargetVowelFrames * HifiF0Builder.FrameMs) / Math.Max(SourceFrameMs, timing.SourceVowelFrames * SourceFrameMs);
            Log.Debug(
                "HifiPhraseFeatureBuilder phone_timewarp phoneme={Phoneme} source_frames={SourceFrames} target_frames={TargetFrames} source_preutter_ms={SourcePreutterMs:F2} target_preutter_ms={TargetPreutterMs:F2} source_soft_skip_ms={SourceSoftSkipMs:F2} source_soft_skip_frames={SourceSoftSkipFrames} source_fixed_frames={SourceFixedFrames} target_fixed_frames={TargetFixedFrames} source_vowel_onset_frames={SourceVowelOnsetFrames} target_vowel_onset_frames={TargetVowelOnsetFrames} source_vowel_frames={SourceVowelFrames} target_vowel_frames={TargetVowelFrames} fixed_ratio={FixedRatio:F4} vowel_ratio={VowelRatio:F4} strategy={Strategy}",
                phone.phoneme,
                sourceFrames,
                outputFrames,
                timing.SourcePreutterMs,
                timing.TargetPreutterMs,
                timing.SourceSoftSkipMs,
                timing.SourceSoftSkipFrames,
                timing.SourceFixedFrames,
                timing.TargetFixedFrames,
                timing.SourceLeadOnsetFrames + timing.SourceVowelOnsetFrames,
                vowelMap.TargetOnsetFrames,
                timing.SourceVowelFrames,
                timing.TargetVowelFrames,
                consonantRatio,
                vowelRatio,
                vowelMap.Strategy);
            // F0 mask is intentionally NARROWER than the fixed consonant region. The consonant
            // region (targetConsonantFrames) also contains the onset of the target vowel, which is
            // voiced; zeroing F0 over the whole region made that vowel onset lose its harmonic
            // source and the note sounded muffled/"dead". So we only mask the leading pure-consonant
            // portion, bounded by min(preutter, consonant) and kept strictly inside the region.
            int f0MaskFrames = ResolveConsonantF0MaskFrames(phone, timing.ConsonantMs, timing.TargetFixedFrames);
            return new PhoneMapReport(true, vowelMap.Strategy, timing.TargetFixedFrames, f0MaskFrames, timing.SourceSoftSkipMs, 0);
        }

        static void WriteCompactPhoneRegion(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            int sourceConsonantFrames) {
            if (outputFrames <= 0) {
                return;
            }
            sourceFrames = Math.Max(1, Math.Min(sourceFrames, sourceMel.GetLength(1) - sourceStart));
            outputFrames = Math.Max(1, Math.Min(outputFrames, output.GetLength(1) - outputStart));
            int nucleus = PickVowelNucleusFrame(sourceMel, sourceStart, sourceFrames, sourceConsonantFrames);
            if (outputFrames == 1) {
                CopyFrame(sourceMel, sourceStart + nucleus, output, outputStart);
                return;
            }

            int onset = sourceConsonantFrames > 0
                ? Math.Clamp(sourceConsonantFrames / 2, 0, sourceFrames - 1)
                : Math.Clamp((int)Math.Round(sourceFrames * 0.25), 0, sourceFrames - 1);
            CopyFrame(sourceMel, sourceStart + onset, output, outputStart);
            CopyFrame(sourceMel, sourceStart + nucleus, output, outputStart + 1);
        }

        static int PickVowelNucleusFrame(float[,] sourceMel, int sourceStart, int sourceFrames, int sourceConsonantFrames) {
            int bins = sourceMel.GetLength(0);
            int first = Math.Clamp(sourceConsonantFrames, 0, Math.Max(0, sourceFrames - 1));
            int last = sourceFrames - 1;
            if (last < first) {
                return Math.Clamp(sourceFrames / 2, 0, Math.Max(0, sourceFrames - 1));
            }

            int best = first;
            double bestEnergy = double.NegativeInfinity;
            for (int t = first; t <= last; t++) {
                double energy = 0;
                for (int m = 0; m < bins; m++) {
                    energy += sourceMel[m, sourceStart + t];
                }
                energy /= Math.Max(1, bins);
                if (energy > bestEnergy) {
                    bestEnergy = energy;
                    best = t;
                }
            }
            return best;
        }

        static void CopyFrame(float[,] sourceMel, int sourceFrame, float[,] output, int outputFrame) {
            int bins = sourceMel.GetLength(0);
            sourceFrame = Math.Clamp(sourceFrame, 0, sourceMel.GetLength(1) - 1);
            outputFrame = Math.Clamp(outputFrame, 0, output.GetLength(1) - 1);
            for (int m = 0; m < bins; m++) {
                output[m, outputFrame] = sourceMel[m, sourceFrame];
            }
        }

        static VowelMapReport WriteVowelSourceToTargetMap(
            float[,] sourceMel,
            int vowelStart,
            int vowelSourceFrames,
            float[,] output,
            int dstStart,
            int dstFrames,
            float[] targetF0,
            string phoneme,
            int preferredOnsetFrames = -1,
            float[]? sourceSamples = null,
            int sourceTone = 0,
            double sourceKeyShiftSemitones = 0) {
            if (dstFrames <= 0) {
                return new VowelMapReport(0, 0, 0, 0, "empty_vowel_target", 0, 0, 0);
            }

            ResolveVowelSections(vowelSourceFrames, preferredOnsetFrames, out int onsetFrames, out int releaseFrames, out int sourceSustainFrames);
            var (targetOnsetFrames, targetSustainFrames, targetReleaseFrames) = AllocateVowelTargetSections(
                onsetFrames,
                sourceSustainFrames,
                releaseFrames,
                dstFrames);
            var transientAnchor = new TransientAnchorPlan(false, 0, 0, 0, 0, "disabled_for_safe_timewarp");

            if (targetOnsetFrames > 0) {
                WriteMappedRegion(sourceMel, vowelStart, onsetFrames, output, dstStart, targetOnsetFrames);
            }
            bool sustainMicroVariationApplied = false;
            if (targetSustainFrames > 0) {
                sustainMicroVariationApplied = WriteSustainWithNaturalTimeMap(
                    sourceMel,
                    vowelStart + onsetFrames,
                    sourceSustainFrames,
                    output,
                    dstStart + targetOnsetFrames,
                    targetSustainFrames,
                    transientAnchor,
                    allowMicroVariation: false,
                    sourceSamples: sourceSamples,
                    targetF0: targetF0,
                    targetF0Start: dstStart + targetOnsetFrames,
                    sourceTone: sourceTone,
                    sourceKeyShiftSemitones: sourceKeyShiftSemitones);
            }
            if (targetReleaseFrames > 0) {
                WriteMappedRegion(
                    sourceMel,
                    vowelStart + vowelSourceFrames - releaseFrames,
                    releaseFrames,
                    output,
                    dstStart + targetOnsetFrames + targetSustainFrames,
                    targetReleaseFrames);
            }

            double vowelRatio = SourceToTargetDurationRatio(vowelSourceFrames, dstFrames);
            double sustainRatio = SourceToTargetDurationRatio(sourceSustainFrames, targetSustainFrames);
            string strategy = Math.Abs(vowelRatio - 1.0) < 0.05
                ? "continuous_equal"
                : vowelRatio > 1.0
                    ? "continuous_sustain_stretch"
                    : "area_compress";
            if (sustainRatio > 1.05) {
                strategy += "_natural";
                if (transientAnchor.Enabled) {
                    strategy += "_transient_lock";
                }
                if (sustainMicroVariationApplied) {
                    strategy += "_microvar";
                }
            }
            if (HasFastF0Motion(targetF0, dstStart, dstFrames)) {
                strategy += "_f0_motion";
            }
            return new VowelMapReport(
                vowelStart + onsetFrames,
                vowelStart + onsetFrames + sourceSustainFrames,
                0,
                0,
                strategy,
                targetOnsetFrames,
                targetSustainFrames,
                targetReleaseFrames);
        }

        static void ResolveVowelSections(int vowelSourceFrames, out int onsetFrames, out int releaseFrames, out int sustainFrames) {
            ResolveVowelSections(vowelSourceFrames, preferredOnsetFrames: -1, out onsetFrames, out releaseFrames, out sustainFrames);
        }

        static void ApplyF0MelCompensation(float[,] mel, float[] f0) {
            int frames = Math.Min(mel.GetLength(1), f0.Length);
            if (frames <= 0) {
                return;
            }

            var voiced = new List<double>(frames);
            for (int i = 0; i < frames; i++) {
                float hz = f0[i];
                if (hz >= 55 && hz <= 1400 && !float.IsNaN(hz) && !float.IsInfinity(hz)) {
                    voiced.Add(hz);
                }
            }
            if (voiced.Count < 4) {
                return;
            }

            double referenceF0 = Math.Min(Percentile(voiced, 0.45), F0MelCompReferenceCapHz);
            if (referenceF0 <= 0 || !IsFinite(referenceF0)) {
                return;
            }

            var desiredCutDb = new double[frames];
            for (int t = 0; t < frames; t++) {
                float hz = f0[t];
                if (hz <= referenceF0 * 1.04 || hz < 55 || hz > 1400 || float.IsNaN(hz) || float.IsInfinity(hz)) {
                    continue;
                }
                double octaves = Math.Log(hz / referenceF0, 2.0);
                desiredCutDb[t] = Math.Clamp(octaves * F0MelCompDbPerOctave, 0, F0MelCompMaxCutDb);
            }

            double[] cutDb = SmoothPositiveEnvelope(desiredCutDb, F0MelCompSmoothHalfFrames);
            int bins = mel.GetLength(0);
            int cutFrames = 0;
            double maxCut = 0;
            for (int t = 0; t < frames; t++) {
                double baseCut = cutDb[t];
                if (baseCut <= 1e-4) {
                    continue;
                }
                cutFrames++;
                maxCut = Math.Max(maxCut, baseCut);
                for (int m = 0; m < bins; m++) {
                    double highWeight = SmoothStep((m - 42) / (double)Math.Max(1, bins - 42));
                    double totalCutDb = Math.Min(F0MelCompMaxCutDb + 0.6, baseCut * (1.0 + F0MelCompHighBandExtra * highWeight));
                    mel[m, t] -= (float)(totalCutDb * Math.Log(10.0) / 20.0);
                }
            }
            if (cutFrames > 0) {
                Log.Information(
                    "HifiPhraseFeatureBuilder f0_mel_compensation reference_f0={ReferenceF0:F2} cut_frames={CutFrames} max_cut_db={MaxCutDb:F2}",
                    referenceF0,
                    cutFrames,
                    maxCut);
            }
        }

        static double[] SmoothPositiveEnvelope(double[] values, int halfWindow) {
            var output = new double[values.Length];
            if (values.Length == 0) {
                return output;
            }
            for (int i = 0; i < values.Length; i++) {
                double sum = 0;
                double weightSum = 0;
                int start = Math.Max(0, i - halfWindow);
                int end = Math.Min(values.Length - 1, i + halfWindow);
                for (int j = start; j <= end; j++) {
                    double distance = Math.Abs(i - j) / Math.Max(1.0, halfWindow + 1.0);
                    double weight = 0.5 + 0.5 * Math.Cos(Math.PI * Math.Clamp(distance, 0, 1));
                    sum += Math.Max(0, values[j]) * weight;
                    weightSum += weight;
                }
                output[i] = weightSum > 0 ? sum / weightSum : Math.Max(0, values[i]);
            }
            return output;
        }

        static double Percentile(List<double> values, double percentile) {
            if (values.Count == 0) {
                return 0;
            }
            values.Sort();
            double index = Math.Clamp(percentile, 0, 1) * (values.Count - 1);
            int left = (int)Math.Floor(index);
            int right = Math.Min(values.Count - 1, left + 1);
            double alpha = index - left;
            return values[left] + (values[right] - values[left]) * alpha;
        }

        static void ResolveVowelSections(int vowelSourceFrames, int preferredOnsetFrames, out int onsetFrames, out int releaseFrames, out int sustainFrames) {
            if (vowelSourceFrames <= 3) {
                onsetFrames = Math.Max(1, vowelSourceFrames - 1);
                releaseFrames = 0;
                sustainFrames = Math.Max(1, vowelSourceFrames - onsetFrames);
                return;
            }

            if (preferredOnsetFrames >= 0) {
                int maxOnset = Math.Max(0, vowelSourceFrames - 1);
                onsetFrames = Math.Clamp(preferredOnsetFrames, 0, maxOnset);
                int remaining = Math.Max(1, vowelSourceFrames - onsetFrames);
                releaseFrames = Math.Clamp((int)Math.Round(remaining * 0.08), 0, Math.Min(5, remaining - 1));
            } else {
                onsetFrames = Math.Clamp((int)Math.Round(vowelSourceFrames * 0.10), 1, 5);
                releaseFrames = Math.Clamp((int)Math.Round(vowelSourceFrames * 0.08), 1, 5);
            }
            while (onsetFrames + releaseFrames > vowelSourceFrames - 2 && (onsetFrames > 1 || releaseFrames > 0)) {
                if (onsetFrames >= releaseFrames && onsetFrames > 1) {
                    onsetFrames--;
                } else if (releaseFrames > 0) {
                    releaseFrames--;
                } else {
                    break;
                }
            }
            sustainFrames = Math.Max(1, vowelSourceFrames - onsetFrames - releaseFrames);
        }

        static (int Onset, int Sustain, int Release) AllocateVowelTargetSections(
            int sourceOnsetFrames,
            int sourceSustainFrames,
            int sourceReleaseFrames,
            int targetTotalFrames) {
            if (targetTotalFrames <= 0) {
                return (0, 0, 0);
            }
            int sourceTotal = Math.Max(1, sourceOnsetFrames + sourceSustainFrames + sourceReleaseFrames);
            int naturalOnsetFrames = SourceFramesToTargetFrames(sourceOnsetFrames);
            int naturalSustainFrames = SourceFramesToTargetFrames(sourceSustainFrames);
            int naturalReleaseFrames = SourceFramesToTargetFrames(sourceReleaseFrames);
            int naturalTotalFrames = Math.Max(1, naturalOnsetFrames + naturalSustainFrames + naturalReleaseFrames);
            bool compressed = targetTotalFrames < naturalTotalFrames;
            int minNucleusFrames = ResolveShortNoteNucleusFloor(sourceSustainFrames, naturalTotalFrames, targetTotalFrames, compressed);
            if (targetTotalFrames == 1) {
                return sourceOnsetFrames > 0 ? (1, 0, 0) : (0, 1, 0);
            }
            if (targetTotalFrames <= 5) {
                int quickOnsetFrames = sourceOnsetFrames > 0 && targetTotalFrames >= 3 ? 1 : 0;
                int quickReleaseFrames = sourceReleaseFrames > 0 && targetTotalFrames >= 5 ? 1 : 0;
                int quickSustainFrames = Math.Max(1, targetTotalFrames - quickOnsetFrames - quickReleaseFrames);
                if (minNucleusFrames > 0 && quickSustainFrames < minNucleusFrames) {
                    int needed = minNucleusFrames - quickSustainFrames;
                    int takeFromRelease = Math.Min(needed, quickReleaseFrames);
                    quickReleaseFrames -= takeFromRelease;
                    needed -= takeFromRelease;
                    int takeFromOnset = Math.Min(needed, quickOnsetFrames);
                    quickOnsetFrames -= takeFromOnset;
                    needed -= takeFromOnset;
                    quickSustainFrames = Math.Max(quickSustainFrames, minNucleusFrames - needed);
                }
                int quickSum = quickOnsetFrames + quickSustainFrames + quickReleaseFrames;
                if (quickSum != targetTotalFrames) {
                    quickSustainFrames += targetTotalFrames - quickSum;
                }
                return (Math.Max(0, quickOnsetFrames), Math.Max(0, quickSustainFrames), Math.Max(0, quickReleaseFrames));
            }

            // VCV/CVVC samples often have a real vowel onset between preutter and consonant
            // boundaries. Keep that onset close to its natural duration and let the stable sustain
            // absorb most length changes; scaling all three regions together makes the onset feel
            // swallowed on short notes and over-stretched on long notes.
            int minSustainFrames = sourceSustainFrames > 0 ? Math.Max(1, minNucleusFrames) : 0;
            int onsetFrames = Math.Min(naturalOnsetFrames, Math.Max(0, targetTotalFrames - minSustainFrames));
            int releaseFrames = Math.Min(naturalReleaseFrames, Math.Max(0, targetTotalFrames - onsetFrames - minSustainFrames));
            int sustainFrames = Math.Max(0, targetTotalFrames - onsetFrames - releaseFrames);

            if (minSustainFrames > 0 && sustainFrames < minSustainFrames) {
                int needed = minSustainFrames - sustainFrames;
                int takeRelease = Math.Min(needed, releaseFrames);
                releaseFrames -= takeRelease;
                needed -= takeRelease;
                int minOnsetFrames = sourceOnsetFrames > 0 ? 1 : 0;
                int takeOnset = Math.Min(needed, Math.Max(0, onsetFrames - minOnsetFrames));
                onsetFrames -= takeOnset;
                needed -= takeOnset;
                sustainFrames = Math.Max(0, targetTotalFrames - onsetFrames - releaseFrames);
            }
            if (minNucleusFrames > 0 && sustainFrames < minNucleusFrames) {
                int needed = minNucleusFrames - sustainFrames;
                int takeRelease = Math.Min(needed, releaseFrames);
                releaseFrames -= takeRelease;
                needed -= takeRelease;
                int takeOnset = Math.Min(needed, Math.Max(0, onsetFrames - (sourceOnsetFrames > 0 ? 1 : 0)));
                onsetFrames -= takeOnset;
                needed -= takeOnset;
                sustainFrames = Math.Max(sustainFrames, minNucleusFrames - needed);
            }
            int sum = onsetFrames + sustainFrames + releaseFrames;
            if (sum != targetTotalFrames) {
                sustainFrames += targetTotalFrames - sum;
            }
            return (Math.Max(0, onsetFrames), Math.Max(0, sustainFrames), Math.Max(0, releaseFrames));
        }

        static int ResolveShortNoteNucleusFloor(int sourceSustainFrames, int sourceTotalFrames, int targetTotalFrames, bool compressed) {
            if (!compressed || sourceSustainFrames <= 0) {
                return 0;
            }
            if (targetTotalFrames <= 2) {
                return 1;
            }
            if (targetTotalFrames <= 5) {
                return 2;
            }
            if (targetTotalFrames <= 10) {
                return 3;
            }
            if (targetTotalFrames <= 14 && targetTotalFrames < sourceTotalFrames) {
                return 2;
            }
            return 0;
        }

        static int SourceFramesToTargetFrames(int sourceFrames) {
            if (sourceFrames <= 0) {
                return 0;
            }
            return Math.Max(1, (int)Math.Round(sourceFrames * SourceFrameMs / HifiF0Builder.FrameMs));
        }

        static double SourceToTargetDurationRatio(int sourceFrames, int targetFrames) {
            double sourceMs = Math.Max(SourceFrameMs, sourceFrames * SourceFrameMs);
            double targetMs = Math.Max(0, targetFrames * HifiF0Builder.FrameMs);
            return targetMs / sourceMs;
        }

        static bool WriteSustainWithNaturalTimeMap(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            TransientAnchorPlan transientAnchor,
            bool allowMicroVariation,
            float[]? sourceSamples = null,
            float[]? targetF0 = null,
            int targetF0Start = 0,
            int sourceTone = 0,
            double sourceKeyShiftSemitones = 0) {
            if (outputFrames <= 0) {
                return false;
            }
            double stretchRatio = SourceToTargetDurationRatio(sourceFrames, outputFrames);
            if (sourceFrames <= 2 || stretchRatio <= 1.05) {
                WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return false;
            }
            if (stretchRatio < SustainTemplateStretchThreshold) {
                return WriteMappedRegionNaturalStretch(
                    sourceMel,
                    sourceStart,
                    sourceFrames,
                    output,
                    outputStart,
                    outputFrames,
                    transientAnchor,
                    allowMicroVariation);
            }
            return WriteSustainTemplateExtension(
                sourceMel,
                sourceStart,
                sourceFrames,
                output,
                outputStart,
                outputFrames,
                allowMicroVariation,
                sourceSamples,
                targetF0,
                targetF0Start,
                sourceTone,
                sourceKeyShiftSemitones);
        }

        static bool WriteSustainTemplateExtension(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            bool allowMicroVariation,
            float[]? sourceSamples = null,
            float[]? targetF0 = null,
            int targetF0Start = 0,
            int sourceTone = 0,
            double sourceKeyShiftSemitones = 0) {
            int bins = sourceMel.GetLength(0);
            int totalSourceFrames = sourceMel.GetLength(1);
            sourceStart = Math.Clamp(sourceStart, 0, Math.Max(0, totalSourceFrames - 1));
            sourceFrames = Math.Max(1, Math.Min(sourceFrames, totalSourceFrames - sourceStart));
            outputFrames = Math.Min(outputFrames, output.GetLength(1) - outputStart);
            if (outputFrames <= 0) {
                return false;
            }
            if (sourceFrames <= 2) {
                WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return false;
            }
            if (outputFrames <= 2) {
                WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return true;
            }

            int stableStart = sourceStart + Math.Max(0, sourceFrames / 5);
            int stableEnd = sourceStart + sourceFrames - Math.Max(0, sourceFrames / 5);
            if (stableEnd <= stableStart) {
                stableStart = sourceStart;
                stableEnd = sourceStart + sourceFrames;
            }
            int stableFrames = Math.Max(1, stableEnd - stableStart);
            if (stableFrames < SustainTextureMinStableFrames) {
                WriteMappedRegionNaturalStretch(
                    sourceMel,
                    sourceStart,
                    sourceFrames,
                    output,
                    outputStart,
                    outputFrames,
                    new TransientAnchorPlan(false, 0, 0, 0, 0, "sustain_texture_too_short"),
                    allowMicroVariation);
                return false;
            }
            double stretchRatio = SourceToTargetDurationRatio(sourceFrames, outputFrames);
            int directSmoothRadius = Math.Clamp(sourceFrames / 5, 1, 10);
            int textureSmoothRadius = Math.Clamp(stableFrames / 10, 1, 3);
            int edgeFrames = ResolveSustainEdgeFrames(outputFrames);
            if (TryWriteWaveformSustainTexture(
                    sourceMel,
                    sourceSamples,
                    sourceStart,
                    sourceFrames,
                    stableStart,
                    stableFrames,
                    output,
                    outputStart,
                    outputFrames,
                    directSmoothRadius,
                    edgeFrames,
                    stretchRatio,
                    allowMicroVariation,
                    targetF0,
                    targetF0Start,
                    sourceTone,
                    sourceKeyShiftSemitones)) {
                return true;
            }
            var directFrame = new float[bins];
            var directContourFrame = new float[bins];
            var textureFrame = new float[bins];
            var textureEnvelopeFrame = new float[bins];
            double seed = SeedFromInts(sourceStart, outputFrames);
            double previousTextureIndex = stableStart - sourceStart + (stableFrames - 1) * 0.5;
            int textureEnvelopeRadius = Math.Clamp(stableFrames / 4, textureSmoothRadius + 1, 14);

            for (int t = 0; t < outputFrames; t++) {
                double u = outputFrames == 1 ? 0 : t / (double)(outputFrames - 1);
                double sourceIndex = outputFrames == 1 || sourceFrames == 1
                    ? 0
                    : u * (sourceFrames - 1);
                SampleInterpolatedFrame(sourceMel, sourceStart, sourceFrames, sourceIndex, directFrame);
                SampleSmoothedFrame(sourceMel, sourceStart, sourceFrames, sourceIndex, directSmoothRadius, directContourFrame);
                double textureIndex = ResolveSustainTextureSourceIndex(
                    t,
                    outputFrames,
                    stableStart - sourceStart,
                    stableFrames,
                    stretchRatio,
                    seed);
                if (t > 0) {
                    textureIndex = LimitTextureIndexStep(previousTextureIndex, textureIndex, stretchRatio);
                }
                previousTextureIndex = textureIndex;
                SampleInterpolatedFrame(sourceMel, sourceStart, sourceFrames, textureIndex, textureFrame);
                SampleSmoothedFrame(sourceMel, sourceStart, sourceFrames, textureIndex, textureEnvelopeRadius, textureEnvelopeFrame);

                double coreWeight = SustainCoreTextureWeight(t, outputFrames, edgeFrames);
                double residualMean = TextureResidualMean(textureFrame, textureEnvelopeFrame);
                double microAmp = allowMicroVariation
                    ? SustainMicroVarLogAmp * coreWeight * Math.Clamp(stretchRatio - 1.0, 0, 1)
                    : 0;
                double framePhase = t * 0.20; // slow drift across the sustain
                for (int m = 0; m < bins; m++) {
                    double micro = 0;
                    if (microAmp > 1e-4) {
                        int band = m * SustainMicroVarBands / Math.Max(1, bins);
                        double bandSeed = SeedFromInts(sourceStart, band);
                        micro = microAmp * SmoothWobble(framePhase, bandSeed);
                    }
                    output[m, outputStart + t] = ComposeSustainTextureBodyValue(
                        directFrame[m],
                        directContourFrame[m],
                        textureFrame[m],
                        textureEnvelopeFrame[m],
                        residualMean,
                        m,
                        bins,
                        coreWeight,
                        micro);
                }
                ApplySustainF0MelMismatchCompensation(
                    output,
                    outputStart + t,
                    TargetF0At(targetF0, targetF0Start + t),
                    sourceTone,
                    coreWeight);
            }
            SmoothInternalSustain(output, outputStart, outputFrames, strength: 0.04f);

            Log.Debug(
                "HifiPhraseFeatureBuilder sustain_texture_trajectory source_frames={SourceFrames} output_frames={OutputFrames} stable_frames={StableFrames} edge_frames={EdgeFrames} direct_smooth_radius={DirectSmoothRadius} texture_smooth_radius={TextureSmoothRadius} micro_variation={MicroVariation}",
                sourceFrames,
                outputFrames,
                stableFrames,
                edgeFrames,
                directSmoothRadius,
                textureSmoothRadius,
                allowMicroVariation);
            return true;
        }

        static bool TryWriteWaveformSustainTexture(
            float[,] sourceMel,
            float[]? sourceSamples,
            int sourceStart,
            int sourceFrames,
            int stableStart,
            int stableFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            int directSmoothRadius,
            int edgeFrames,
            double stretchRatio,
            bool allowMicroVariation,
            float[]? targetF0,
            int targetF0Start,
            int sourceTone,
            double sourceKeyShiftSemitones) {
            if (sourceSamples == null
                    || sourceSamples.Length < HifiMelExtractor.WinSize
                    || outputFrames < WaveformSustainMinOutputFrames
                    || stableFrames < WaveformSustainMinStableFrames) {
                return false;
            }

            int bins = sourceMel.GetLength(0);
            var positions = new double[outputFrames];
            double seed = SeedFromInts(sourceStart, outputFrames);
            double previousTextureIndex = stableStart - sourceStart + (stableFrames - 1) * 0.5;
            for (int t = 0; t < outputFrames; t++) {
                double textureIndex = ResolveSustainTextureSourceIndex(
                    t,
                    outputFrames,
                    stableStart - sourceStart,
                    stableFrames,
                    stretchRatio,
                    seed);
                if (t > 0) {
                    textureIndex = LimitTextureIndexStep(previousTextureIndex, textureIndex, stretchRatio);
                }
                previousTextureIndex = textureIndex;
                double absoluteSourceFrame = sourceStart + textureIndex;
                positions[t] = Math.Clamp(MelFrameToSampleCenter(absoluteSourceFrame), 0, sourceSamples.Length - 1);
            }

            float[,] waveformTextureMel = sustainMelExtractor.ExtractAtPositions(sourceSamples, positions, sourceKeyShiftSemitones);
            if (waveformTextureMel.GetLength(1) != outputFrames) {
                return false;
            }

            var directFrame = new float[bins];
            var directContourFrame = new float[bins];
            var textureFrame = new float[bins];
            var textureEnvelopeFrame = new float[bins];
            int textureEnvelopeRadius = Math.Clamp(stableFrames / 4, directSmoothRadius + 1, 14);
            for (int t = 0; t < outputFrames; t++) {
                double u = outputFrames == 1 ? 0 : t / (double)(outputFrames - 1);
                double sourceIndex = outputFrames == 1 || sourceFrames == 1
                    ? 0
                    : u * (sourceFrames - 1);
                SampleInterpolatedFrame(sourceMel, sourceStart, sourceFrames, sourceIndex, directFrame);
                SampleSmoothedFrame(sourceMel, sourceStart, sourceFrames, sourceIndex, directSmoothRadius, directContourFrame);
                CopyMelColumn(waveformTextureMel, t, textureFrame);
                double textureIndex = MelSampleCenterToFrameIndex(positions[t]) - sourceStart;
                SampleSmoothedFrame(sourceMel, sourceStart, sourceFrames, textureIndex, textureEnvelopeRadius, textureEnvelopeFrame);

                double coreWeight = SustainCoreTextureWeight(t, outputFrames, edgeFrames);
                double residualMean = TextureResidualMean(textureFrame, textureEnvelopeFrame);
                double microAmp = allowMicroVariation
                    ? SustainMicroVarLogAmp * coreWeight * Math.Clamp(stretchRatio - 1.0, 0, 1)
                    : 0;
                double framePhase = t * 0.20;
                for (int m = 0; m < bins; m++) {
                    double micro = 0;
                    if (microAmp > 1e-4) {
                        int band = m * SustainMicroVarBands / Math.Max(1, bins);
                        double bandSeed = SeedFromInts(sourceStart, band);
                        micro = microAmp * SmoothWobble(framePhase, bandSeed);
                    }
                    output[m, outputStart + t] = ComposeSustainTextureBodyValue(
                        directFrame[m],
                        directContourFrame[m],
                        textureFrame[m],
                        textureEnvelopeFrame[m],
                        residualMean,
                        m,
                        bins,
                        coreWeight,
                        micro);
                }
                ApplySustainF0MelMismatchCompensation(
                    output,
                    outputStart + t,
                    TargetF0At(targetF0, targetF0Start + t),
                    sourceTone,
                    coreWeight);
            }
            Log.Debug(
                "HifiPhraseFeatureBuilder sustain_waveform_texture_body source_frames={SourceFrames} output_frames={OutputFrames} stable_frames={StableFrames} edge_frames={EdgeFrames} stretch_ratio={StretchRatio:F3} source_samples={SourceSamples}",
                sourceFrames,
                outputFrames,
                stableFrames,
                edgeFrames,
                stretchRatio,
                sourceSamples.Length);
            return true;
        }

        static double ResolveWaveformSustainGlobalEnergyOffset(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] waveformTextureMel,
            int outputFrames,
            int directSmoothRadius) {
            if (outputFrames <= 0 || waveformTextureMel.GetLength(1) <= 0) {
                return 0;
            }
            int bins = sourceMel.GetLength(0);
            var directContourFrame = new float[bins];
            double directSum = 0;
            double textureSum = 0;
            int count = Math.Min(outputFrames, waveformTextureMel.GetLength(1));
            for (int t = 0; t < count; t++) {
                double u = count == 1 ? 0 : t / (double)(count - 1);
                double sourceIndex = sourceFrames <= 1 ? 0 : u * (sourceFrames - 1);
                SampleSmoothedFrame(sourceMel, sourceStart, sourceFrames, sourceIndex, directSmoothRadius, directContourFrame);
                directSum += FrameMean(directContourFrame);
                textureSum += FrameMean(waveformTextureMel, t);
            }
            double directMean = directSum / count;
            double textureMean = textureSum / count;
            if (!IsFinite(directMean) || !IsFinite(textureMean)) {
                return 0;
            }
            return Math.Clamp(directMean - textureMean, -WaveformSustainEnergyOffsetClamp, WaveformSustainEnergyOffsetClamp);
        }

        static double ResolveWaveformSustainEnergyDelta(double targetEnergy, double currentEnergy, double globalOffset) {
            if (!IsFinite(targetEnergy) || !IsFinite(currentEnergy)) {
                return globalOffset;
            }
            double correctedEnergy = currentEnergy + globalOffset;
            double remaining = targetEnergy - correctedEnergy;
            if (Math.Abs(remaining) <= WaveformSustainEnergyOutlierTolerance) {
                return globalOffset;
            }
            double outlierCorrection = Math.Clamp(
                remaining - Math.Sign(remaining) * WaveformSustainEnergyOutlierTolerance,
                -WaveformSustainEnergyOutlierClamp,
                WaveformSustainEnergyOutlierClamp);
            return globalOffset + outlierCorrection;
        }

        static double MelFrameToSampleCenter(double frameIndex) {
            return frameIndex * HifiMelExtractor.OriginHopSize + HifiMelExtractor.OriginHopSize * 0.5;
        }

        static double MelSampleCenterToFrameIndex(double sampleCenter) {
            return (sampleCenter - HifiMelExtractor.OriginHopSize * 0.5) / HifiMelExtractor.OriginHopSize;
        }

        static float ComposeSustainTextureBodyValue(
            float directFrameValue,
            float directContourValue,
            float textureValue,
            float textureEnvelopeValue,
            double residualMean,
            int bin,
            int bins,
            double coreWeight,
            double micro) {
            double residual = textureValue - textureEnvelopeValue - residualMean;
            if (!IsFinite(residual)) {
                residual = 0;
            }
            residual = Math.Clamp(residual, -SustainTextureBodyClamp, SustainTextureBodyClamp);
            double residualWeight = SustainTextureBodyResidualWeight(bin, bins);

            // Keep the long-sustain loudness/formant envelope on the slow contour, but let the
            // stable source texture provide most of the fine spectral motion. This avoids both
            // static-template metal and repeated source vibrato/energy envelopes.
            double coreValue = directContourValue
                + residual * SustainTextureBodyAmount * residualWeight
                + micro;
            return (float)(directFrameValue * (1.0 - coreWeight) + coreValue * coreWeight);
        }

        static double TextureResidualMean(float[] textureFrame, float[] textureEnvelopeFrame) {
            int count = Math.Min(textureFrame.Length, textureEnvelopeFrame.Length);
            if (count <= 0) {
                return 0;
            }
            double sum = 0;
            int valid = 0;
            for (int i = 0; i < count; i++) {
                double residual = textureFrame[i] - textureEnvelopeFrame[i];
                if (!IsFinite(residual)) {
                    continue;
                }
                sum += residual;
                valid++;
            }
            return valid > 0 ? sum / valid : 0;
        }

        static double SustainTextureBodyResidualWeight(int bin, int bins) {
            if (bins <= 1) {
                return SustainTextureBodyLowBandResidual;
            }
            double high = SmoothStep((bin - 12) / (double)Math.Max(1, bins - 12));
            return SustainTextureBodyLowBandResidual + (1.0 - SustainTextureBodyLowBandResidual) * high;
        }

        static void CopyMelColumn(float[,] mel, int frame, float[] output) {
            int bins = mel.GetLength(0);
            frame = Math.Clamp(frame, 0, Math.Max(0, mel.GetLength(1) - 1));
            for (int m = 0; m < bins; m++) {
                output[m] = mel[m, frame];
            }
        }

        static float TargetF0At(float[]? targetF0, int frame) {
            if (targetF0 == null || targetF0.Length == 0) {
                return 0;
            }
            return targetF0[Math.Clamp(frame, 0, targetF0.Length - 1)];
        }

        static void ApplySustainF0MelMismatchCompensation(
            float[,] output,
            int frame,
            float targetF0,
            int sourceTone,
            double strength) {
            if (strength <= 1e-4 || sourceTone <= 0 || targetF0 < 55 || targetF0 > 1400 || !IsFinite(targetF0)) {
                return;
            }
            double sourceF0 = MusicMath.ToneToFreq(sourceTone);
            if (sourceF0 <= 0 || !IsFinite(sourceF0) || targetF0 <= sourceF0 * 1.06) {
                return;
            }

            int bins = output.GetLength(0);
            frame = Math.Clamp(frame, 0, Math.Max(0, output.GetLength(1) - 1));
            double octaves = Math.Log(targetF0 / sourceF0, 2.0);
            double highCutDb = Math.Clamp(octaves * 0.65, 0, F0MismatchMaxCutDb) * strength;
            double contrast = Math.Clamp(octaves * 0.09, 0, F0MismatchMaxContrast) * strength;
            double mean = FrameMean(output, frame);
            double newMean = 0;
            for (int m = 0; m < bins; m++) {
                double highWeight = HighBandWeight(m, bins, (int)F0MismatchHighBandStart);
                double value = mean + (output[m, frame] - mean) * (1.0 - contrast * (0.35 + 0.65 * highWeight));
                value -= highCutDb * Math.Log(10.0) / 20.0 * highWeight;
                output[m, frame] = (float)value;
                newMean += value;
            }
            newMean /= Math.Max(1, bins);
            float correction = (float)((mean - newMean) * 0.85);
            for (int m = 0; m < bins; m++) {
                output[m, frame] += correction;
            }
        }

        static void SmoothInternalSustain(float[,] output, int outputStart, int outputFrames, float strength) {
            int bins = output.GetLength(0);
            if (outputFrames < 5 || strength <= 0) {
                return;
            }
            var previous = new float[bins];
            for (int t = 1; t < outputFrames - 1; t++) {
                int frame = outputStart + t;
                float edgeDistance = Math.Min(t, outputFrames - 1 - t) / (float)Math.Max(1, outputFrames / 6);
                float localStrength = strength * Math.Clamp(edgeDistance, 0f, 1f);
                if (localStrength <= 0) {
                    continue;
                }
                for (int m = 0; m < bins; m++) {
                    previous[m] = output[m, frame];
                }
                for (int m = 0; m < bins; m++) {
                    float smoothed = (output[m, frame - 1] + previous[m] * 2f + output[m, frame + 1]) * 0.25f;
                    output[m, frame] = previous[m] * (1f - localStrength) + smoothed * localStrength;
                }
            }
        }

        static int ResolveSustainEdgeFrames(int outputFrames) {
            if (outputFrames <= 1) {
                return 0;
            }
            int maxEdgeFrames = Math.Min(14, Math.Max(1, outputFrames / 2));
            int preferred = Math.Max(1, outputFrames / 6);
            if (maxEdgeFrames >= 2) {
                preferred = Math.Max(2, preferred);
            }
            return Math.Clamp(preferred, 1, maxEdgeFrames);
        }

        static double SustainCoreTextureWeight(int frame, int totalFrames, int edgeFrames) {
            if (totalFrames <= 2 || edgeFrames <= 0) {
                return 1.0;
            }
            int distance = Math.Min(frame, totalFrames - 1 - frame);
            if (distance >= edgeFrames) {
                return 1.0;
            }
            return SmoothStep(distance / (double)Math.Max(1, edgeFrames));
        }

        static double ResolveSustainTextureSourceIndex(
            int targetFrame,
            int targetFrames,
            int stableStart,
            int stableFrames,
            double stretchRatio,
            double seed) {
            if (stableFrames <= 1) {
                return stableStart;
            }
            double span = stableFrames - 1;
            double center = stableStart + span * 0.5;
            double wanderRange = Math.Max(0.5, span * Math.Clamp(0.18 + 0.05 * Math.Log(Math.Max(1.0, stretchRatio), 2.0), 0.18, 0.36));
            double slow = SmoothWobble(targetFrame * 0.021, seed + 0.37);
            double slower = SmoothWobble(targetFrame * 0.0067, seed + 1.11);
            double detail = SmoothWobble(targetFrame * 0.113, seed + 2.73);
            double wander = (slow * 0.55 + slower * 0.35 + detail * 0.10) * wanderRange;
            return Math.Clamp(center + wander, stableStart, stableStart + span);
        }

        static double LimitTextureIndexStep(double previous, double current, double stretchRatio) {
            double maxStep = SustainTextureMaxStepFrames + Math.Clamp((stretchRatio - 2.0) * 0.12, 0, 0.55);
            return previous + Math.Clamp(current - previous, -maxStep, maxStep);
        }

        static void SampleInterpolatedFrame(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            double sourceIndex,
            float[] output) {
            int bins = sourceMel.GetLength(0);
            sourceIndex = Math.Clamp(sourceIndex, 0, Math.Max(0, sourceFrames - 1));
            int left = sourceStart + (int)Math.Floor(sourceIndex);
            int right = Math.Min(sourceStart + sourceFrames - 1, left + 1);
            float alpha = (float)(sourceIndex - Math.Floor(sourceIndex));
            for (int m = 0; m < bins; m++) {
                float v0 = sourceMel[m, left];
                float v1 = sourceMel[m, right];
                output[m] = v0 + (v1 - v0) * alpha;
            }
        }

        static void SampleSmoothedFrame(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            double sourceIndex,
            int radius,
            float[] output) {
            int bins = sourceMel.GetLength(0);
            sourceIndex = Math.Clamp(sourceIndex, 0, Math.Max(0, sourceFrames - 1));
            int center = (int)Math.Round(sourceIndex);
            int first = Math.Clamp(center - radius, 0, sourceFrames - 1);
            int last = Math.Clamp(center + radius, first, sourceFrames - 1);
            Array.Clear(output, 0, output.Length);
            double weightSum = 0;
            for (int local = first; local <= last; local++) {
                double distance = Math.Abs(local - sourceIndex) / Math.Max(1.0, radius + 1.0);
                double weight = 0.5 + 0.5 * Math.Cos(Math.PI * Math.Clamp(distance, 0, 1));
                weightSum += weight;
                int frame = sourceStart + local;
                for (int m = 0; m < bins; m++) {
                    output[m] += (float)(sourceMel[m, frame] * weight);
                }
            }
            if (weightSum <= 1e-6) {
                SampleInterpolatedFrame(sourceMel, sourceStart, sourceFrames, sourceIndex, output);
                return;
            }
            float scale = (float)(1.0 / weightSum);
            for (int m = 0; m < bins; m++) {
                output[m] *= scale;
            }
        }

        static double ResolveSustainTargetEnergy(double stableMeanEnergy, double sourceEnergy) {
            if (!IsFinite(stableMeanEnergy)) {
                return sourceEnergy;
            }
            if (!IsFinite(sourceEnergy)) {
                return stableMeanEnergy;
            }
            double delta = sourceEnergy - stableMeanEnergy;
            // Stable pool supplies the spectral shape; slow source energy supplies the loudness
            // contour. Limit boosts more than drops so long sustains do not inflate in volume.
            return stableMeanEnergy + Math.Clamp(delta, -0.55, 0.16);
        }

        static double HighBandWeight(int bin, int bins, int startBin) {
            if (bins <= 1) {
                return 0;
            }
            int start = Math.Clamp(startBin, 0, bins - 1);
            if (bin <= start || start >= bins - 1) {
                return 0;
            }
            double u = (bin - start) / (double)(bins - 1 - start);
            return SmoothStep(Math.Clamp(u, 0, 1));
        }

        static double FrameMean(float[] frame) {
            if (frame.Length == 0) {
                return 0;
            }
            double sum = 0;
            for (int i = 0; i < frame.Length; i++) {
                sum += frame[i];
            }
            return sum / frame.Length;
        }

        static double FrameMean(float[,] mel, int frame) {
            int bins = mel.GetLength(0);
            double sum = 0;
            for (int m = 0; m < bins; m++) {
                sum += mel[m, frame];
            }
            return sum / Math.Max(1, bins);
        }

        static TransientAnchorPlan DetectTransientAnchor(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            int targetFrames) {
            if (sourceFrames < 10 || targetFrames < 8) {
                return new TransientAnchorPlan(false, 0, 0, 0, 0, "too_short");
            }
            int bins = sourceMel.GetLength(0);
            var flux = new double[sourceFrames];
            for (int t = 1; t < sourceFrames; t++) {
                int current = sourceStart + t;
                int previous = current - 1;
                if (current >= sourceMel.GetLength(1)) {
                    break;
                }
                double sum = 0;
                for (int m = 0; m < bins; m++) {
                    sum += Math.Abs(sourceMel[m, current] - sourceMel[m, previous]);
                }
                flux[t] = sum / Math.Max(1, bins);
            }

            int first = 2;
            int last = sourceFrames - 3;
            if (last <= first) {
                return new TransientAnchorPlan(false, 0, 0, 0, 0, "too_short_interior");
            }
            int peakFrame = first;
            double peakFlux = flux[first];
            var region = new List<double>(last - first + 1);
            for (int t = first; t <= last; t++) {
                double value = flux[t];
                region.Add(value);
                if (value > peakFlux) {
                    peakFlux = value;
                    peakFrame = t;
                }
            }
            if (region.Count == 0) {
                return new TransientAnchorPlan(false, 0, 0, 0, 0, "empty_flux_region");
            }
            region.Sort();
            double medianFlux = region[region.Count / 2];
            if (!IsFinite(peakFlux) || !IsFinite(medianFlux) || medianFlux <= 1e-6 || peakFlux < medianFlux * 1.8) {
                return new TransientAnchorPlan(false, 0, 0, peakFlux, medianFlux, "no_strong_transient");
            }

            int sourceMinAnchor = Math.Max(2, (int)Math.Round(sourceFrames * 0.15));
            int sourceMaxAnchor = Math.Min(sourceFrames - 3, (int)Math.Round(sourceFrames * 0.85));
            if (sourceMaxAnchor <= sourceMinAnchor) {
                return new TransientAnchorPlan(false, 0, 0, peakFlux, medianFlux, "no_anchor_room");
            }
            int clampedSourceAnchor = Math.Clamp(peakFrame, sourceMinAnchor, sourceMaxAnchor);
            double sourceAnchorNorm = clampedSourceAnchor / (double)Math.Max(1, sourceFrames - 1);
            int targetAnchor = (int)Math.Round(sourceAnchorNorm * Math.Max(1, targetFrames - 1));
            int targetMinAnchor = Math.Max(2, (int)Math.Round(targetFrames * 0.12));
            int targetMaxAnchor = Math.Min(targetFrames - 3, (int)Math.Round(targetFrames * 0.88));
            if (targetMaxAnchor <= targetMinAnchor) {
                return new TransientAnchorPlan(false, 0, 0, peakFlux, medianFlux, "no_target_anchor_room");
            }
            return new TransientAnchorPlan(
                true,
                clampedSourceAnchor,
                Math.Clamp(targetAnchor, targetMinAnchor, targetMaxAnchor),
                peakFlux,
                medianFlux,
                string.Empty);
        }

        static bool WriteMappedRegionNaturalStretch(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            TransientAnchorPlan transientAnchor,
            bool allowMicroVariation) {
            int bins = sourceMel.GetLength(0);
            int totalSourceFrames = sourceMel.GetLength(1);
            sourceStart = Math.Clamp(sourceStart, 0, Math.Max(0, totalSourceFrames - 1));
            sourceFrames = Math.Max(1, Math.Min(sourceFrames, totalSourceFrames - sourceStart));
            outputFrames = Math.Min(outputFrames, output.GetLength(1) - outputStart);
            if (outputFrames <= 0) {
                return false;
            }
            double stretchRatio = SourceToTargetDurationRatio(sourceFrames, outputFrames);
            bool microVariationEnabled = false;
            for (int t = 0; t < outputFrames; t++) {
                double targetNorm = outputFrames == 1 ? 0 : t / (double)(outputFrames - 1);
                double sourceNorm = MapSourceNormalized(targetNorm, stretchRatio, transientAnchor, sourceFrames, outputFrames);
                sourceNorm = Math.Clamp(sourceNorm, 0, 1);
                double sourceIndex = sourceNorm * Math.Max(0, sourceFrames - 1);
                int left = sourceStart + (int)Math.Floor(sourceIndex);
                int right = Math.Min(sourceStart + sourceFrames - 1, left + 1);
                float alpha = (float)(sourceIndex - Math.Floor(sourceIndex));
                for (int m = 0; m < bins; m++) {
                    float v0 = sourceMel[m, left];
                    float v1 = sourceMel[m, right];
                    output[m, outputStart + t] = v0 + (v1 - v0) * alpha;
                }
            }
            return microVariationEnabled;
        }

        static double MapSourceNormalized(
            double targetNorm,
            double stretchRatio,
            TransientAnchorPlan transientAnchor,
            int sourceFrames,
            int targetFrames) {
            targetNorm = Math.Clamp(targetNorm, 0, 1);
            if (sourceFrames <= 1 || targetFrames <= 1) {
                return targetNorm;
            }
            if (!transientAnchor.Enabled) {
                return ApplyNaturalStretchWarp(targetNorm, stretchRatio);
            }
            double sourceAnchorNorm = Math.Clamp(transientAnchor.SourceAnchorFrame / (double)(sourceFrames - 1), 1e-4, 1 - 1e-4);
            double targetAnchorNorm = Math.Clamp(transientAnchor.TargetAnchorFrame / (double)(targetFrames - 1), 1e-4, 1 - 1e-4);
            if (targetNorm <= targetAnchorNorm) {
                double local = targetNorm / targetAnchorNorm;
                double segmentStretchRatio = targetAnchorNorm / sourceAnchorNorm;
                return sourceAnchorNorm * ApplyNaturalStretchWarp(local, segmentStretchRatio);
            }
            double rightLocal = (targetNorm - targetAnchorNorm) / (1 - targetAnchorNorm);
            double rightStretchRatio = (1 - targetAnchorNorm) / (1 - sourceAnchorNorm);
            return sourceAnchorNorm + (1 - sourceAnchorNorm) * ApplyNaturalStretchWarp(rightLocal, rightStretchRatio);
        }

        static double ApplyNaturalStretchWarp(double u, double stretchRatio) {
            u = Math.Clamp(u, 0, 1);
            if (!IsFinite(stretchRatio) || stretchRatio <= 0) {
                return u;
            }
            double magnitude = Math.Abs(Math.Log(stretchRatio, 2.0));
            double strength = Math.Clamp(0.08 + 0.20 * magnitude, 0, 0.38);
            if (stretchRatio > 1.05) {
                return Math.Clamp(u + strength * (u - SmoothStep(u)), 0, 1);
            }
            if (stretchRatio < 0.95) {
                return Math.Clamp(u + strength * (SmoothStep(u) - u), 0, 1);
            }
            return u;
        }

        static void WriteMappedRegion(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames) {
            if (outputFrames <= 0) {
                return;
            }
            int bins = sourceMel.GetLength(0);
            int totalSourceFrames = sourceMel.GetLength(1);
            sourceStart = Math.Clamp(sourceStart, 0, Math.Max(0, totalSourceFrames - 1));
            sourceFrames = Math.Max(1, Math.Min(sourceFrames, totalSourceFrames - sourceStart));
            outputFrames = Math.Min(outputFrames, output.GetLength(1) - outputStart);
            if (outputFrames <= 0) {
                return;
            }
            double sourceMs = sourceFrames * SourceFrameMs;
            double outputMs = outputFrames * HifiF0Builder.FrameMs;
            if (outputMs < sourceMs * 0.98) {
                WriteAreaResampledRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return;
            }
            for (int t = 0; t < outputFrames; t++) {
                double sourceIndex = outputFrames == 1 || sourceFrames == 1
                    ? 0
                    : (double)t * (sourceFrames - 1) / (outputFrames - 1);
                int left = sourceStart + (int)Math.Floor(sourceIndex);
                int right = Math.Min(sourceStart + sourceFrames - 1, left + 1);
                float alpha = (float)(sourceIndex - Math.Floor(sourceIndex));
                for (int m = 0; m < bins; m++) {
                    float v0 = sourceMel[m, left];
                    float v1 = sourceMel[m, right];
                    output[m, outputStart + t] = v0 + (v1 - v0) * alpha;
                }
            }
        }

        static void WriteLeadMappedRegion(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            int sourceSoftSkipFrames) {
            if (outputFrames <= 0) {
                return;
            }
            int bins = sourceMel.GetLength(0);
            int totalSourceFrames = sourceMel.GetLength(1);
            sourceStart = Math.Clamp(sourceStart, 0, Math.Max(0, totalSourceFrames - 1));
            sourceFrames = Math.Max(1, Math.Min(sourceFrames, totalSourceFrames - sourceStart));
            outputFrames = Math.Min(outputFrames, output.GetLength(1) - outputStart);
            if (outputFrames <= 0) {
                return;
            }
            double previous = 0;
            for (int t = 0; t < outputFrames; t++) {
                double sourceIndex = ResolveLeadSourceIndex(t, outputFrames, sourceFrames, sourceSoftSkipFrames);
                if (t > 0) {
                    sourceIndex = Math.Max(previous, sourceIndex);
                }
                previous = sourceIndex;
                int left = sourceStart + (int)Math.Floor(sourceIndex);
                int right = Math.Min(sourceStart + sourceFrames - 1, left + 1);
                float alpha = (float)(sourceIndex - Math.Floor(sourceIndex));
                for (int m = 0; m < bins; m++) {
                    float v0 = sourceMel[m, left];
                    float v1 = sourceMel[m, right];
                    output[m, outputStart + t] = v0 + (v1 - v0) * alpha;
                }
            }
        }

        static void WriteAreaResampledRegion(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames) {
            int bins = sourceMel.GetLength(0);
            for (int t = 0; t < outputFrames; t++) {
                double rangeStart = t * sourceFrames / (double)outputFrames;
                double rangeEnd = (t + 1) * sourceFrames / (double)outputFrames;
                int first = Math.Clamp((int)Math.Floor(rangeStart), 0, sourceFrames - 1);
                int last = Math.Clamp((int)Math.Ceiling(rangeEnd) - 1, first, sourceFrames - 1);
                for (int m = 0; m < bins; m++) {
                    double sum = 0;
                    double weightSum = 0;
                    for (int s = first; s <= last; s++) {
                        double overlap = Math.Min(rangeEnd, s + 1.0) - Math.Max(rangeStart, s);
                        if (overlap <= 0) {
                            continue;
                        }
                        sum += sourceMel[m, sourceStart + s] * overlap;
                        weightSum += overlap;
                    }
                    output[m, outputStart + t] = weightSum > 0
                        ? (float)(sum / weightSum)
                        : sourceMel[m, sourceStart + first];
                }
            }
        }

        static int TrimInactiveTailFrames(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            int sourceConsonantFrames,
            string phoneme) {
            int bins = sourceMel.GetLength(0);
            int totalSourceFrames = sourceMel.GetLength(1);
            sourceStart = Math.Clamp(sourceStart, 0, Math.Max(0, totalSourceFrames - 1));
            sourceFrames = Math.Max(1, Math.Min(sourceFrames, totalSourceFrames - sourceStart));
            if (sourceFrames <= MinSourceFrames * 2) {
                return sourceFrames;
            }

            var frameEnergy = new double[sourceFrames];
            double maxEnergy = double.NegativeInfinity;
            for (int t = 0; t < sourceFrames; t++) {
                double sum = 0;
                for (int m = 0; m < bins; m++) {
                    float value = sourceMel[m, sourceStart + t];
                    if (float.IsNaN(value) || float.IsInfinity(value)) {
                        continue;
                    }
                    sum += value;
                }
                double energy = sum / Math.Max(1, bins);
                frameEnergy[t] = energy;
                if (IsFinite(energy)) {
                    maxEnergy = Math.Max(maxEnergy, energy);
                }
            }
            if (!IsFinite(maxEnergy)) {
                return sourceFrames;
            }

            double threshold = maxEnergy - InactiveTailLogDrop;
            int minKeep = Math.Clamp(sourceConsonantFrames + MinVowelSourceFrames, 1, sourceFrames);
            int lastActive = sourceFrames - 1;
            for (int t = sourceFrames - 1; t >= minKeep - 1; t--) {
                if (frameEnergy[t] >= threshold) {
                    lastActive = t;
                    break;
                }
            }
            int trimmedFrames = Math.Clamp(lastActive + 1 + InactiveTailGuardFrames, minKeep, sourceFrames);
            if (trimmedFrames < sourceFrames - InactiveTailGuardFrames) {
                Log.Information(
                    "HifiPhraseFeatureBuilder source_inactive_tail_trimmed phoneme={Phoneme} source_frames={SourceFrames} trimmed_source_frames={TrimmedSourceFrames} max_energy={MaxEnergy:F4} threshold={Threshold:F4}",
                    phoneme,
                    sourceFrames,
                    trimmedFrames,
                    maxEnergy,
                    threshold);
            }
            return trimmedFrames;
        }

        static int NormalizeSourceConsonantFrames(int sourceConsonantFrames, int sourceFrames, string phoneme) {
            if (sourceFrames <= 0) {
                return 0;
            }
            sourceConsonantFrames = Math.Clamp(sourceConsonantFrames, 0, sourceFrames - 1);
            int maxConsonant = Math.Max(0, sourceFrames - MinVowelSourceFrames);
            if (sourceConsonantFrames > maxConsonant) {
                Log.Information(
                    "HifiPhraseFeatureBuilder source_consonant_clamped phoneme={Phoneme} source_frames={SourceFrames} original_source_consonant_frames={OriginalSourceConsonantFrames} clamped_source_consonant_frames={ClampedSourceConsonantFrames}",
                    phoneme,
                    sourceFrames,
                    sourceConsonantFrames,
                    maxConsonant);
                sourceConsonantFrames = maxConsonant;
            }
            return sourceConsonantFrames;
        }

        static int NormalizeSourceStableStartFrames(int sourceStableStartFrames, int sourceFixedFrames, int sourceFrames) {
            if (sourceFrames <= 0) {
                return 0;
            }
            sourceFixedFrames = Math.Clamp(sourceFixedFrames, 0, Math.Max(0, sourceFrames - 1));
            sourceStableStartFrames = Math.Clamp(sourceStableStartFrames, sourceFixedFrames, sourceFrames);
            int maxStableStart = Math.Max(sourceFixedFrames, sourceFrames - MinVowelSourceFrames);
            if (sourceStableStartFrames > maxStableStart) {
                sourceStableStartFrames = maxStableStart;
            }
            return Math.Clamp(sourceStableStartFrames, sourceFixedFrames, sourceFrames);
        }

        static HifiPhoneTimingPlan BuildPhoneTimingPlan(int sourceFrames, int outputFrames, RenderPhone phone) {
            sourceFrames = Math.Max(1, sourceFrames);
            outputFrames = Math.Max(1, outputFrames);
            double? consonantMs = EffectiveConsonantMs(phone);
            // VCV/CVVC timing has two separate coordinate systems:
            // - target preutter: OpenUtau's possibly capped/overlap-adjusted note-grid lead.
            // - source preutter/consonant: the original oto coordinates inside the raw sample.
            // ClassicRenderer bridges these with skipOver; HIFI-NEURA must not use the capped
            // target preutter as a source-sample boundary, or the vowel/onset arrives late.
            double targetPreutterMs = consonantMs.HasValue ? Math.Max(0, phone.preutterMs) : 0;
            double sourcePreutterMs = consonantMs.HasValue ? SourcePreutterMs(phone) : targetPreutterMs;
            double sourceOverlapMs = consonantMs.HasValue ? SourceOverlapMs(phone) : Math.Max(0, phone.overlapMs);
            double stableStartMs = consonantMs.HasValue ? SourceStableStartMs(phone, sourcePreutterMs) : sourcePreutterMs;

            int sourceLeadFrames = sourcePreutterMs > 0
                ? (int)Math.Round(sourcePreutterMs / SourceFrameMs)
                : 0;
            int sourceStableStartFrames = stableStartMs > 0
                ? (int)Math.Round(stableStartMs / SourceFrameMs)
                : sourceLeadFrames;
            sourceLeadFrames = NormalizeSourceConsonantFrames(sourceLeadFrames, sourceFrames, phone.phoneme);
            sourceStableStartFrames = NormalizeSourceStableStartFrames(sourceStableStartFrames, sourceLeadFrames, sourceFrames);
            int sourceVowelOnsetFrames = Math.Max(0, sourceStableStartFrames - sourceLeadFrames);
            int sourceVowelFrames = sourceFrames - sourceLeadFrames;

            int targetLeadFrames = ResolveTargetLeadFrames(phone, consonantMs, outputFrames);
            int sourceSoftSkipFrames = ResolveSourceSoftSkipFrames(phone, sourceLeadFrames, targetLeadFrames);
            double sourceSoftSkipMs = sourceSoftSkipFrames * SourceFrameMs;
            int targetFixedFrames = Math.Min(
                ResolveTargetFixedFrames(phone, consonantMs, outputFrames, sourcePreutterMs, sourceOverlapMs, stableStartMs),
                targetLeadFrames);
            int targetLeadOnsetFrames = targetLeadFrames - targetFixedFrames;
            int sourceFixedFrames = ResolveSourceFixedFrames(sourceLeadFrames, targetFixedFrames, targetLeadFrames);
            int sourceLeadOnsetFrames = Math.Max(0, sourceLeadFrames - sourceFixedFrames);
            int targetVowelFrames = outputFrames - targetLeadFrames;
            if (targetVowelFrames < MinVowelTargetFrames) {
                targetVowelFrames = MinVowelTargetFrames;
                targetLeadFrames = Math.Max(0, outputFrames - targetVowelFrames);
                targetFixedFrames = Math.Min(targetFixedFrames, targetLeadFrames);
                targetLeadOnsetFrames = targetLeadFrames - targetFixedFrames;
                sourceFixedFrames = ResolveSourceFixedFrames(sourceLeadFrames, targetFixedFrames, targetLeadFrames);
                sourceLeadOnsetFrames = Math.Max(0, sourceLeadFrames - sourceFixedFrames);
            }

            return new HifiPhoneTimingPlan(
                consonantMs,
                targetPreutterMs,
                sourcePreutterMs,
                sourceSoftSkipMs,
                sourceSoftSkipFrames,
                sourceLeadFrames,
                sourceStableStartFrames,
                sourceVowelOnsetFrames,
                sourceVowelFrames,
                targetLeadFrames,
                targetFixedFrames,
                targetLeadOnsetFrames,
                sourceFixedFrames,
                sourceLeadOnsetFrames,
                targetVowelFrames);
        }

        internal static double ResolveTargetFixedMs(RenderPhone phone) {
            double sourcePreutterMs = SourcePreutterMs(phone);
            return ResolveTargetFixedMs(
                phone.preutterMs,
                phone.overlapMs,
                phone.durationMs,
                EffectiveConsonantMs(phone).HasValue,
                sourcePreutterMs,
                SourceOverlapMs(phone),
                SourceStableStartMs(phone, sourcePreutterMs));
        }

        internal static double ResolveTargetFixedMs(
            double preutterMs,
            double overlapMs,
            double durationMs,
            bool hasReliableConsonant) {
            return ResolveTargetFixedMs(
                preutterMs,
                overlapMs,
                durationMs,
                hasReliableConsonant,
                preutterMs,
                overlapMs,
                preutterMs);
        }

        internal static double ResolveTargetFixedMs(
            double preutterMs,
            double overlapMs,
            double durationMs,
            bool hasReliableConsonant,
            double sourcePreutterMs,
            double sourceOverlapMs,
            double sourceStableStartMs) {
            if (!hasReliableConsonant || preutterMs <= 0) {
                return 0;
            }
            preutterMs = Math.Max(0, preutterMs);
            overlapMs = Math.Clamp(overlapMs, 0, preutterMs);
            durationMs = Math.Max(0, durationMs);
            sourcePreutterMs = Math.Max(0, sourcePreutterMs);
            sourceOverlapMs = Math.Clamp(sourceOverlapMs, 0, Math.Max(0, sourcePreutterMs));
            sourceStableStartMs = Math.Max(sourcePreutterMs, sourceStableStartMs);
            if (preutterMs <= HifiF0Builder.FrameMs * 3.0) {
                return preutterMs;
            }

            double overlapAwareMs = Math.Max(
                HifiF0Builder.FrameMs,
                preutterMs - overlapMs * TargetFixedOverlapRelease);
            double sourceLeadBudgetMs = Math.Max(
                HifiF0Builder.FrameMs,
                (sourcePreutterMs > 0 ? sourcePreutterMs : preutterMs) - sourceOverlapMs * TargetFixedOverlapRelease);
            double sourceOnsetMs = Math.Max(0, sourceStableStartMs - sourcePreutterMs);
            double sourceOnsetBonusMs = Math.Clamp(
                sourceOnsetMs * TargetFixedSourceOnsetBonusRatio,
                0,
                TargetFixedSourceOnsetBonusMaxMs);
            double sourceAwareMaxMs = Math.Min(
                TargetFixedMaxMs,
                Math.Max(HifiF0Builder.FrameMs, sourceLeadBudgetMs + sourceOnsetBonusMs));
            double shortNoteRatio = ResolveTargetFixedShortNoteRatio(sourceLeadBudgetMs);
            double durationCapMs = Math.Max(
                HifiF0Builder.FrameMs * 2.0,
                durationMs * shortNoteRatio);
            double upperMs = Math.Min(preutterMs, overlapAwareMs);
            upperMs = Math.Min(upperMs, durationCapMs);
            upperMs = Math.Min(upperMs, sourceAwareMaxMs);
            return Math.Clamp(upperMs, HifiF0Builder.FrameMs, preutterMs);
        }

        static int ResolveTargetFixedFrames(
            RenderPhone phone,
            double? consonantMs,
            int outputFrames,
            double sourcePreutterMs,
            double sourceOverlapMs,
            double sourceStableStartMs) {
            int maxFixedFrames = ResolveMaxTargetFixedFrames(outputFrames);
            if (maxFixedFrames <= 0) {
                return 0;
            }
            double fixedMs = ResolveTargetFixedMs(
                phone.preutterMs,
                phone.overlapMs,
                phone.durationMs,
                consonantMs.HasValue,
                sourcePreutterMs,
                sourceOverlapMs,
                sourceStableStartMs);
            if (fixedMs <= 0) {
                return 0;
            }
            int frames = Math.Max(1, (int)Math.Round(fixedMs / HifiF0Builder.FrameMs));
            if (frames == 1 && outputFrames >= 4 && phone.preutterMs >= HifiF0Builder.FrameMs * 1.5) {
                frames = 2;
            }
            return Math.Clamp(frames, 0, maxFixedFrames);
        }

        static double ResolveTargetFixedShortNoteRatio(double sourceLeadBudgetMs) {
            double sourceWeight = sourceLeadBudgetMs / Math.Max(HifiF0Builder.FrameMs, sourceLeadBudgetMs + 80.0);
            return TargetFixedShortNoteRatio + (sourceWeight - 0.5) * TargetFixedDynamicShortNoteRatioRange;
        }

        static int ResolveTargetLeadFrames(RenderPhone phone, double? consonantMs, int outputFrames) {
            int maxLeadFrames = ResolveMaxTargetFixedFrames(outputFrames);
            if (!consonantMs.HasValue || phone.preutterMs <= 0 || maxLeadFrames <= 0) {
                return 0;
            }
            int frames = Math.Max(1, (int)Math.Round(phone.preutterMs / HifiF0Builder.FrameMs));
            return Math.Clamp(frames, 0, maxLeadFrames);
        }

        internal static int ResolveSourceFixedFrames(int sourceLeadFrames, int targetFixedFrames, int targetLeadFrames) {
            if (sourceLeadFrames <= 0 || targetFixedFrames <= 0 || targetLeadFrames <= 0) {
                return 0;
            }
            int frames = (int)Math.Round(sourceLeadFrames * targetFixedFrames / (double)targetLeadFrames);
            return Math.Clamp(frames, 1, sourceLeadFrames);
        }

        static double SourcePreutterMs(RenderPhone phone) {
            if (phone.oto != null && phone.oto.Preutter > 0) {
                return Math.Max(0, phone.oto.Preutter);
            }
            return Math.Max(0, phone.preutterMs);
        }

        static int ResolveSourceSoftSkipFrames(RenderPhone phone, int sourceLeadFrames, int targetLeadFrames) {
            if (sourceLeadFrames <= 2 || targetLeadFrames <= 2 || sourceLeadFrames <= targetLeadFrames) {
                return 0;
            }
            double skipMs = ResolveClassicSkipOverCandidateMs(phone);
            if (skipMs <= 0 || !IsFinite(skipMs)) {
                return 0;
            }
            int requestedFrames = (int)Math.Round(skipMs / SourceFrameMs);
            int maxFrames = Math.Max(0, sourceLeadFrames - Math.Max(2, targetLeadFrames / 2));
            return Math.Clamp(requestedFrames, 0, maxFrames);
        }

        static double ResolveClassicSkipOverCandidateMs(RenderPhone phone) {
            if (phone.oto == null || phone.oto.Preutter <= 0) {
                return 0;
            }
            double velocityPercent = phone.velocity > 0
                ? phone.velocity * 100.0
                : 100.0;
            if (!IsFinite(velocityPercent)) {
                velocityPercent = 100.0;
            }
            velocityPercent = Math.Clamp(velocityPercent, 0.0, 200.0);
            double stretchRatio = Math.Pow(2.0, 1.0 - velocityPercent * 0.01);
            double pitchLeadingMs = phone.oto.Preutter * stretchRatio;
            double targetLeadingMs = phone.leadingMs > 0
                ? phone.leadingMs
                : phone.preutterMs;
            double skipMs = pitchLeadingMs - Math.Max(0, targetLeadingMs);
            return IsFinite(skipMs) && skipMs > 0 ? skipMs : 0;
        }

        static double SourceOverlapMs(RenderPhone phone) {
            if (phone.oto != null && phone.oto.Overlap > 0) {
                return Math.Max(0, phone.oto.Overlap);
            }
            return Math.Max(0, phone.overlapMs);
        }

        static double SourceStableStartMs(RenderPhone phone, double sourcePreutterMs) {
            double consonantMs = phone.oto != null && phone.oto.Consonant > 0
                ? phone.oto.Consonant
                : sourcePreutterMs;
            return Math.Max(sourcePreutterMs, consonantMs);
        }

        static int ResolveMaxTargetFixedFrames(int outputFrames) {
            if (outputFrames <= 1) {
                return 0;
            }
            return Math.Max(0, outputFrames - ResolveMinimumTargetVowelFrames(outputFrames));
        }

        static int ResolveMinimumTargetVowelFrames(int outputFrames) {
            if (outputFrames <= 1) {
                return 0;
            }
            if (outputFrames <= 3) {
                return 1;
            }
            if (outputFrames <= 7) {
                return 2;
            }
            return 3;
        }

        // Frames at the START of a phone's fixed region that were previously F0-masked. Kept for
        // reference / potential reuse, but no longer applied: this vocoder muted F0==0 frames, so
        // F0 is now kept continuous across consonants (see Build).
        static int ResolveConsonantF0MaskFrames(RenderPhone phone, double? consonantMs, int targetConsonantFrames) {
            if (!consonantMs.HasValue || targetConsonantFrames <= 0) {
                return 0;
            }
            double maskMs = consonantMs.Value;
            if (phone.preutterMs > 0) {
                maskMs = Math.Min(maskMs, phone.preutterMs);
            }
            int maskFrames = (int)Math.Round(maskMs / HifiF0Builder.FrameMs);
            int upper = Math.Min(targetConsonantFrames - 1, ConsonantF0MaskMaxFrames);
            return Math.Clamp(maskFrames, 0, Math.Max(0, upper));
        }

        static double? EffectiveConsonantMs(RenderPhone phone) {
            if (phone.oto == null || phone.oto.Consonant <= 0) {
                return null;
            }
            // The oto Consonant marks where the stable vowel starts. The fixed alignment span is
            // still based on preutter; the preutter->consonant interval is treated as vowel onset
            // and kept near its source duration before the stable sustain absorbs most scaling.
            double consonantMs = phone.oto.Consonant;
            double durationCapMs = Math.Max(HifiF0Builder.FrameMs * 3.0, phone.durationMs * 0.80);
            if (durationCapMs > 0 && consonantMs > durationCapMs) {
                consonantMs = durationCapMs;
            }
            return consonantMs > 0 ? consonantMs : null;
        }

        static double PhoneSegmentStartMs(RenderPhone phone) {
            return phone.positionMs - Math.Max(0, phone.leadingMs);
        }

        static bool HasFastF0Motion(float[] f0, int start, int frames) {
            if (f0.Length < 2 || frames <= 1) {
                return false;
            }
            int s = Math.Clamp(start, 0, f0.Length - 1);
            int e = Math.Clamp(start + frames, s + 1, f0.Length);
            for (int i = s + 1; i < e; i++) {
                if (F0SlopeCents(f0[i - 1], f0[i]) > F0AwareMaxSlopeCentsPerFrame) {
                    return true;
                }
            }
            return false;
        }

        static double F0SlopeCents(float left, float right) {
            if (left <= 0 || right <= 0 || float.IsNaN(left) || float.IsNaN(right) || float.IsInfinity(left) || float.IsInfinity(right)) {
                return double.PositiveInfinity;
            }
            return Math.Abs(1200.0 * Math.Log(right / left, 2.0));
        }

        static double SmoothStep(double x) {
            x = Math.Clamp(x, 0, 1);
            return x * x * (3 - 2 * x);
        }

        // Deterministic, smooth, non-periodic wobble in roughly [-1, 1]. Built from a few sines at
        // incommensurable (irrational-ratio) frequencies so the sum never repeats over the sustain,
        // which is what keeps the stretched vowel from sounding like a fixed loop. Deterministic
        // (seed-driven, no clock) so renders are reproducible and cache-stable.
        static double SmoothWobble(double phase, double seed) {
            // Irrational frequency ratios: no common period.
            double a = Math.Sin(phase * 1.0 + seed * 1.7);
            double b = Math.Sin(phase * 1.6180339887 + seed * 2.3 + 1.3);
            double c = Math.Sin(phase * 2.7182818285 + seed * 0.7 + 2.6);
            return (a * 0.5 + b * 0.33 + c * 0.17);
        }

        static double SeedFromInts(int a, int b) {
            // Cheap deterministic hash: a stable fractional seed in [0, 2*pi).
            unchecked {
                uint h = 2166136261u;
                h = (h ^ (uint)a) * 16777619u;
                h = (h ^ (uint)b) * 16777619u;
                return (h % 100000) / 100000.0 * (2.0 * Math.PI);
            }
        }

        static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

        static void FillConstant(float[,] values, float value) {
            for (int m = 0; m < values.GetLength(0); m++) {
                for (int t = 0; t < values.GetLength(1); t++) {
                    values[m, t] = value;
                }
            }
        }

        static HifiPhraseMetadata BuildMetadata(RenderPhrase phrase, RenderResult layout, int frames, double phraseStartMs, HifiMelAssemblyReport assemblyReport) {
            var metadata = new HifiPhraseMetadata {
                SampleRate = HifiMelExtractor.SampleRate,
                HopSize = HifiF0Builder.HopSize,
                FrameMs = HifiF0Builder.FrameMs,
                PhraseStartMs = phraseStartMs,
                LeadingMs = layout.leadingMs,
                EstimatedLengthMs = layout.estimatedLengthMs,
            };
            metadata.Notes.AddRange(phrase.notes.Select((n, i) => new HifiNoteMetadata {
                Index = i,
                Lyric = n.lyric,
                Tone = n.tone,
                AdjustedTone = n.adjustedTone,
                PositionMs = n.positionMs,
                DurationMs = n.durationMs,
            }));

            metadata.Phones.AddRange(assemblyReport.Phones);
            metadata.Boundaries.AddRange(assemblyReport.Boundaries);
            foreach (var phone in metadata.Phones) {
                if (phone.ConsonantFrameCount <= 0) {
                    continue;
                }
                metadata.ConsonantFrameRanges.Add(new HifiFrameRangeMetadata {
                    PhoneIndex = phone.Index,
                    Phoneme = phone.Phoneme,
                    StartFrame = phone.ConsonantStartFrame,
                    FrameCount = phone.ConsonantFrameCount,
                    Kind = phone.F0MaskFrames > 0 ? "fixed+f0-mask" : "fixed",
                });
            }
            metadata.PhoneDiagnostics.AddRange(assemblyReport.PhoneDiagnostics);
            return metadata;
        }

        static void Validate(float[,] mel, float[] f0) {
            foreach (float value in mel) {
                if (float.IsNaN(value) || float.IsInfinity(value)) {
                    throw new InvalidOperationException("hifi phrase_mel contains NaN or Inf.");
                }
            }
            foreach (float value in f0) {
                if (float.IsNaN(value) || float.IsInfinity(value)) {
                    throw new InvalidOperationException("hifi phrase_f0 contains NaN or Inf.");
                }
            }
        }
    }
}
