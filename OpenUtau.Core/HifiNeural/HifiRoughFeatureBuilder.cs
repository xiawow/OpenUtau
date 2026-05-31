using System;
using System.Collections.Generic;
using System.Linq;
using OpenUtau.Core.Render;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiRoughFeatureBuilder {
        const int MinSourceFrames = 8;
        const int MinVowelSourceFrames = 4;
        const int MinVowelTargetFrames = 1;
        // Absolute ceiling on how many leading frames get F0-masked (noise excitation). ~5 frames
        // at ~11.6ms/frame covers a plosive/fricative burst without reaching the vowel.
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
        const int StableSustainPoolMaxFrames = 32;
        const double StableSustainEnergyFloorLog = 1.20;
        const double StableSustainEnergyClamp = 0.25;
        const double StableSustainResidualAmp = 0.11;
        const double StableSustainResidualClamp = 0.018;
        const double StableSustainDeDenseMax = 0.24;
        const int StableSustainDeDenseHighBandStart = 58;
        const double StableSustainDeDenseHighBandExtra = 0.35;
        const int PhraseEdgeMelFadeInFrames = 3;
        const int PhraseEdgeMelFadeOutFrames = 5;
        const double PhraseEdgeMelMinGain = 0.18;
        const int InactiveTailGuardFrames = 6;
        const double InactiveTailLogDrop = 2.8;
        // At larger stretch ratios the source is "scanned" more slowly; if the slow-scanned source
        // trajectory keeps a high weight, every micro-motion inside the vowel (formant drift,
        // vibrato) gets stretched proportionally and the note audibly slows down. But freezing it
        // too hard (very low blend) collapses the sustain to a near-static template, which the
        // vocoder turns into a perfectly periodic, mechanical-sounding loop. 0.28 keeps enough real
        // spectral motion to avoid the loop without bringing back the slow-down.
        const double TemplateTrajectoryMinBlend = 0.28;
        const double TemplateTrajectoryMaxBlend = 0.70;
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

        readonly record struct StableSustainPool(int[] Frames, double MeanEnergy);

        public HifiRoughFeatureBuilder(IHifiMelEnhancer melEnhancer) {
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
                "HifiRoughFeatureBuilder mel_domain_concat mode=overlap_neural target_duration_ms={TargetDurationMs:F3} target_frames={TargetFrames} phones={Phones}",
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

        static double[] BuildVariableSourcePositions(RenderPhrase phrase, double phraseStartMs, float[] roughSamples, int targetFrames) {
            var positions = BuildLinearSourcePositions(targetFrames, roughSamples.Length);
            if (targetFrames <= 0 || roughSamples.Length <= 0 || phrase.phones.Length == 0) {
                return positions;
            }

            int[] sourceStarts = BuildPhoneStarts(phrase, phraseStartMs, roughSamples.Length, SourceSampleMs);
            int[] targetStarts = BuildPhoneStarts(phrase, phraseStartMs, targetFrames, HifiF0Builder.FrameMs);
            var written = new bool[targetFrames];
            int splitCount = 0;
            int fallbackCount = 0;
            int trimmedCount = 0;

            for (int i = 0; i < phrase.phones.Length; i++) {
                int sourceStart = Math.Clamp(sourceStarts[i], 0, Math.Max(0, roughSamples.Length - 1));
                int sourceEnd = i + 1 < sourceStarts.Length ? sourceStarts[i + 1] : roughSamples.Length;
                sourceEnd = Math.Clamp(sourceEnd, sourceStart + 1, Math.Max(sourceStart + 1, roughSamples.Length));
                int targetStart = Math.Clamp(targetStarts[i], 0, Math.Max(0, targetFrames - 1));
                int targetEnd = i + 1 < targetStarts.Length ? targetStarts[i + 1] : targetFrames;
                targetEnd = Math.Clamp(targetEnd, targetStart + 1, Math.Max(targetStart + 1, targetFrames));

                int sourceCount = Math.Max(1, sourceEnd - sourceStart);
                int targetCount = Math.Max(1, targetEnd - targetStart);
                double? consonantMs = EffectiveConsonantMs(phrase.phones[i]);
                int sourceConsonantSamples = consonantMs.HasValue
                    ? (int)Math.Round(consonantMs.Value * HifiMelExtractor.SampleRate / 1000.0)
                    : 0;
                sourceConsonantSamples = NormalizeSourceConsonantSamples(sourceConsonantSamples, sourceCount, phrase.phones[i].phoneme);
                int trimmedSourceCount = TrimInactiveTailSamples(
                    roughSamples,
                    sourceStart,
                    sourceCount,
                    sourceConsonantSamples,
                    phrase.phones[i].phoneme);
                if (trimmedSourceCount < sourceCount) {
                    trimmedCount++;
                    sourceCount = trimmedSourceCount;
                    sourceConsonantSamples = NormalizeSourceConsonantSamples(sourceConsonantSamples, sourceCount, phrase.phones[i].phoneme);
                }

                bool splitApplied = WritePhoneSourcePositionMap(
                    positions,
                    targetStart,
                    targetCount,
                    sourceStart,
                    sourceCount,
                    sourceConsonantSamples,
                    consonantMs,
                    phrase.phones[i]);
                if (splitApplied) {
                    splitCount++;
                } else {
                    fallbackCount++;
                }
                for (int t = targetStart; t < targetEnd && t < written.Length; t++) {
                    written[t] = true;
                }
            }

            FillUnwrittenSourcePositions(positions, written, roughSamples.Length);
            Log.Information(
                "HifiRoughFeatureBuilder variable_position_time_map phones={Phones} split_segments={SplitSegments} fallback_segments={FallbackSegments} trimmed_segments={TrimmedSegments} rough_samples={RoughSamples} target_frames={TargetFrames}",
                phrase.phones.Length,
                splitCount,
                fallbackCount,
                trimmedCount,
                roughSamples.Length,
                targetFrames);
            return positions;
        }

        static double[] BuildLinearSourcePositions(int targetFrames, int sourceSamples) {
            var positions = new double[Math.Max(0, targetFrames)];
            if (positions.Length == 0 || sourceSamples <= 0) {
                return positions;
            }
            for (int t = 0; t < positions.Length; t++) {
                positions[t] = positions.Length == 1 || sourceSamples == 1
                    ? 0
                    : t * (sourceSamples - 1.0) / (positions.Length - 1);
            }
            return positions;
        }

        static void FillUnwrittenSourcePositions(double[] positions, bool[] written, int sourceSamples) {
            var fallback = BuildLinearSourcePositions(positions.Length, sourceSamples);
            for (int t = 0; t < positions.Length; t++) {
                if (!written[t]) {
                    positions[t] = fallback[t];
                }
            }
        }

        static bool WritePhoneSourcePositionMap(
            double[] positions,
            int outputStart,
            int outputFrames,
            int sourceStart,
            int sourceSamples,
            int sourceConsonantSamples,
            double? consonantMs,
            RenderPhone phone) {
            outputFrames = Math.Max(1, Math.Min(outputFrames, positions.Length - outputStart));
            if (outputFrames <= 0) {
                return false;
            }
            sourceSamples = Math.Max(1, sourceSamples);
            if (outputFrames <= 2 || sourceSamples < MinSourceSamples) {
                WriteSourcePositionRegion(positions, outputStart, outputFrames, sourceStart, sourceSamples);
                return false;
            }

            sourceConsonantSamples = NormalizeSourceConsonantSamples(sourceConsonantSamples, sourceSamples, phone.phoneme);
            int sourceVowelSamples = sourceSamples - sourceConsonantSamples;
            if (!consonantMs.HasValue || sourceConsonantSamples <= 0 || sourceVowelSamples < MinVowelSourceSamples) {
                WriteSourcePositionRegion(positions, outputStart, outputFrames, sourceStart, sourceSamples);
                return false;
            }

            int targetConsonantFrames = (int)Math.Round(consonantMs.Value / HifiF0Builder.FrameMs);
            targetConsonantFrames = Math.Clamp(targetConsonantFrames, 0, Math.Max(0, outputFrames - MinVowelTargetFrames));
            int targetVowelFrames = outputFrames - targetConsonantFrames;
            if (targetVowelFrames < MinVowelTargetFrames) {
                targetVowelFrames = MinVowelTargetFrames;
                targetConsonantFrames = Math.Max(0, outputFrames - targetVowelFrames);
            }

            if (targetConsonantFrames > 0) {
                WriteSourcePositionRegion(
                    positions,
                    outputStart,
                    targetConsonantFrames,
                    sourceStart,
                    sourceConsonantSamples);
            }
            var vowelReport = WriteVowelSourcePositionMap(
                positions,
                outputStart + targetConsonantFrames,
                targetVowelFrames,
                sourceStart + sourceConsonantSamples,
                sourceVowelSamples);

            double consonantRatio = sourceConsonantSamples > 0
                ? (targetConsonantFrames * HifiF0Builder.FrameMs) / Math.Max(SourceSampleMs, sourceConsonantSamples * SourceSampleMs)
                : 0;
            double vowelRatio = (targetVowelFrames * HifiF0Builder.FrameMs) / Math.Max(SourceSampleMs, sourceVowelSamples * SourceSampleMs);
            Log.Information(
                "HifiRoughFeatureBuilder variable_position_phone phoneme={Phoneme} source_samples={SourceSamples} target_frames={TargetFrames} source_consonant_samples={SourceConsonantSamples} target_consonant_frames={TargetConsonantFrames} source_vowel_samples={SourceVowelSamples} target_vowel_frames={TargetVowelFrames} consonant_ratio={ConsonantRatio:F4} vowel_ratio={VowelRatio:F4} strategy={Strategy}",
                phone.phoneme,
                sourceSamples,
                outputFrames,
                sourceConsonantSamples,
                targetConsonantFrames,
                sourceVowelSamples,
                targetVowelFrames,
                consonantRatio,
                vowelRatio,
                vowelReport.Strategy);
            return true;
        }

        static VowelMapReport WriteVowelSourcePositionMap(
            double[] positions,
            int outputStart,
            int outputFrames,
            int sourceStart,
            int sourceSamples) {
            if (outputFrames <= 0) {
                return new VowelMapReport(0, 0, 0, 0, "empty_vowel_target", 0, 0, 0);
            }
            ResolveVowelSectionsSamples(sourceSamples, out int onsetSamples, out int releaseSamples, out int sustainSamples);
            var (targetOnsetFrames, targetSustainFrames, targetReleaseFrames) = AllocateVowelTargetSections(
                SamplesToSourceFrames(onsetSamples),
                SamplesToSourceFrames(sustainSamples),
                SamplesToSourceFrames(releaseSamples),
                outputFrames);

            if (targetOnsetFrames > 0) {
                WriteSourcePositionRegion(positions, outputStart, targetOnsetFrames, sourceStart, onsetSamples);
            }
            if (targetSustainFrames > 0) {
                WriteSustainSourcePositionRegion(
                    positions,
                    outputStart + targetOnsetFrames,
                    targetSustainFrames,
                    sourceStart + onsetSamples,
                    sustainSamples);
            }
            if (targetReleaseFrames > 0) {
                WriteSourcePositionRegion(
                    positions,
                    outputStart + targetOnsetFrames + targetSustainFrames,
                    targetReleaseFrames,
                    sourceStart + sourceSamples - releaseSamples,
                    releaseSamples);
            }

            double vowelRatio = (outputFrames * HifiF0Builder.FrameMs) / Math.Max(SourceSampleMs, sourceSamples * SourceSampleMs);
            string strategy = Math.Abs(vowelRatio - 1.0) < 0.05
                ? "variable_position_equal"
                : vowelRatio > 1.0
                    ? "variable_position_sustain_stretch"
                    : "variable_position_area_compress";
            return new VowelMapReport(
                sourceStart + onsetSamples,
                sourceStart + onsetSamples + sustainSamples,
                0,
                0,
                strategy,
                targetOnsetFrames,
                targetSustainFrames,
                targetReleaseFrames);
        }

        static void WriteSourcePositionRegion(
            double[] positions,
            int outputStart,
            int outputFrames,
            int sourceStart,
            int sourceSamples) {
            if (outputFrames <= 0) {
                return;
            }
            sourceSamples = Math.Max(1, sourceSamples);
            outputFrames = Math.Max(1, Math.Min(outputFrames, positions.Length - outputStart));
            for (int t = 0; t < outputFrames; t++) {
                double sourceOffset = outputFrames == 1 || sourceSamples == 1
                    ? 0
                    : t * (sourceSamples - 1.0) / (outputFrames - 1);
                positions[outputStart + t] = sourceStart + sourceOffset;
            }
        }

        static void WriteSustainSourcePositionRegion(
            double[] positions,
            int outputStart,
            int outputFrames,
            int sourceStart,
            int sourceSamples) {
            if (outputFrames <= 0) {
                return;
            }
            sourceSamples = Math.Max(1, sourceSamples);
            outputFrames = Math.Max(1, Math.Min(outputFrames, positions.Length - outputStart));
            double stretchRatio = (outputFrames * HifiF0Builder.FrameMs) / Math.Max(SourceSampleMs, sourceSamples * SourceSampleMs);
            for (int t = 0; t < outputFrames; t++) {
                double u = outputFrames == 1 || sourceSamples == 1
                    ? 0
                    : t / (double)(outputFrames - 1);
                double sourceNorm = ApplyNaturalStretchWarp(u, stretchRatio);
                positions[outputStart + t] = sourceStart + sourceNorm * (sourceSamples - 1);
            }
        }

        static void ResolveVowelSectionsSamples(int sourceSamples, out int onsetSamples, out int releaseSamples, out int sustainSamples) {
            sourceSamples = Math.Max(1, sourceSamples);
            if (sourceSamples <= HifiMelExtractor.OriginHopSize * 3) {
                onsetSamples = Math.Max(1, sourceSamples - 1);
                releaseSamples = 0;
                sustainSamples = Math.Max(1, sourceSamples - onsetSamples);
                return;
            }
            onsetSamples = Math.Clamp((int)Math.Round(sourceSamples * 0.10), HifiMelExtractor.OriginHopSize, HifiMelExtractor.OriginHopSize * 5);
            releaseSamples = Math.Clamp((int)Math.Round(sourceSamples * 0.08), HifiMelExtractor.OriginHopSize, HifiMelExtractor.OriginHopSize * 5);
            while (onsetSamples + releaseSamples > sourceSamples - HifiMelExtractor.OriginHopSize * 2
                    && (onsetSamples > HifiMelExtractor.OriginHopSize || releaseSamples > 0)) {
                if (onsetSamples >= releaseSamples && onsetSamples > HifiMelExtractor.OriginHopSize) {
                    onsetSamples -= HifiMelExtractor.OriginHopSize;
                } else if (releaseSamples > 0) {
                    releaseSamples = Math.Max(0, releaseSamples - HifiMelExtractor.OriginHopSize);
                } else {
                    break;
                }
            }
            sustainSamples = Math.Max(1, sourceSamples - onsetSamples - releaseSamples);
        }

        static int SamplesToSourceFrames(int samples) {
            if (samples <= 0) {
                return 0;
            }
            return Math.Max(1, (int)Math.Round(samples / (double)HifiMelExtractor.OriginHopSize));
        }

        static int TrimInactiveTailSamples(
            float[] samples,
            int sourceStart,
            int sourceSamples,
            int sourceConsonantSamples,
            string phoneme) {
            sourceStart = Math.Clamp(sourceStart, 0, Math.Max(0, samples.Length - 1));
            sourceSamples = Math.Max(1, Math.Min(sourceSamples, samples.Length - sourceStart));
            if (sourceSamples <= MinSourceSamples * 2) {
                return sourceSamples;
            }

            const int energyHop = HifiF0Builder.HopSize / 2;
            int windows = Math.Max(1, (sourceSamples + energyHop - 1) / energyHop);
            var rms = new double[windows];
            double maxRms = 0;
            for (int w = 0; w < windows; w++) {
                int start = sourceStart + w * energyHop;
                int end = Math.Min(sourceStart + sourceSamples, start + energyHop);
                double sum = 0;
                int count = Math.Max(1, end - start);
                for (int i = start; i < end; i++) {
                    sum += samples[i] * samples[i];
                }
                double value = Math.Sqrt(sum / count);
                rms[w] = value;
                maxRms = Math.Max(maxRms, value);
            }
            if (maxRms <= 1e-6 || !IsFinite(maxRms)) {
                return sourceSamples;
            }

            double threshold = maxRms * 0.08;
            int minKeep = Math.Clamp(sourceConsonantSamples + MinVowelSourceSamples, 1, sourceSamples);
            int minWindow = Math.Clamp(minKeep / energyHop, 0, windows - 1);
            int lastActiveWindow = windows - 1;
            for (int w = windows - 1; w >= minWindow; w--) {
                if (rms[w] >= threshold) {
                    lastActiveWindow = w;
                    break;
                }
            }

            int trimmedSamples = Math.Clamp((lastActiveWindow + 1) * energyHop + InactiveTailGuardSamples, minKeep, sourceSamples);
            if (trimmedSamples < sourceSamples - InactiveTailGuardSamples) {
                Log.Information(
                    "HifiRoughFeatureBuilder source_inactive_tail_trimmed_wave phoneme={Phoneme} source_samples={SourceSamples} trimmed_source_samples={TrimmedSourceSamples} max_rms={MaxRms:F6} threshold={Threshold:F6}",
                    phoneme,
                    sourceSamples,
                    trimmedSamples,
                    maxRms,
                    threshold);
            }
            return trimmedSamples;
        }

        static int NormalizeSourceConsonantSamples(int sourceConsonantSamples, int sourceSamples, string phoneme) {
            if (sourceSamples <= 0) {
                return 0;
            }
            sourceConsonantSamples = Math.Clamp(sourceConsonantSamples, 0, sourceSamples - 1);
            int maxConsonant = Math.Max(0, sourceSamples - MinVowelSourceSamples);
            if (sourceConsonantSamples > maxConsonant) {
                Log.Information(
                    "HifiRoughFeatureBuilder source_consonant_clamped_wave phoneme={Phoneme} source_samples={SourceSamples} original_source_consonant_samples={OriginalSourceConsonantSamples} clamped_source_consonant_samples={ClampedSourceConsonantSamples}",
                    phoneme,
                    sourceSamples,
                    sourceConsonantSamples,
                    maxConsonant);
                sourceConsonantSamples = maxConsonant;
            }
            return sourceConsonantSamples;
        }

        static float[,] AlignMelFrames(float[,] sourceMel, int targetFrames) {
            int bins = sourceMel.GetLength(0);
            int sourceFrames = sourceMel.GetLength(1);
            var output = new float[bins, targetFrames];
            if (sourceFrames <= 0) {
                FillConstant(output, (float)Math.Log(1e-5));
                return output;
            }
            if (sourceFrames == targetFrames) {
                BufferCopy(sourceMel, output);
                return output;
            }
            for (int t = 0; t < targetFrames; t++) {
                double sourceIndex = targetFrames <= 1 || sourceFrames <= 1
                    ? 0
                    : (double)t * (sourceFrames - 1) / (targetFrames - 1);
                int left = (int)Math.Floor(sourceIndex);
                int right = Math.Min(sourceFrames - 1, left + 1);
                float alpha = (float)(sourceIndex - left);
                for (int m = 0; m < bins; m++) {
                    float v0 = sourceMel[m, left];
                    float v1 = sourceMel[m, right];
                    output[m, t] = v0 + (v1 - v0) * alpha;
                }
            }
            return output;
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

        static float[,] AlignMelFramesByPhones(
            RenderPhrase phrase,
            double phraseStartMs,
            float[,] sourceMel,
            int targetFrames,
            float[] targetF0) {
            int bins = sourceMel.GetLength(0);
            int sourceFrames = sourceMel.GetLength(1);
            if (phrase.phones.Length == 0 || sourceFrames <= 0 || targetFrames <= 0) {
                return AlignMelFrames(sourceMel, targetFrames);
            }

            var output = new float[bins, targetFrames];
            var written = new bool[targetFrames];
            int[] sourceStarts = BuildPhoneStarts(phrase, phraseStartMs, sourceFrames, SourceFrameMs);
            int[] targetStarts = BuildPhoneStarts(phrase, phraseStartMs, targetFrames, HifiF0Builder.FrameMs);
            int splitCount = 0;
            int fallbackCount = 0;

            for (int i = 0; i < phrase.phones.Length; i++) {
                int sourceStart = sourceStarts[i];
                int sourceEnd = i + 1 < sourceStarts.Length ? sourceStarts[i + 1] : sourceFrames;
                int targetStart = targetStarts[i];
                int targetEnd = i + 1 < targetStarts.Length ? targetStarts[i + 1] : targetFrames;

                sourceEnd = Math.Clamp(sourceEnd, sourceStart + 1, Math.Max(sourceStart + 1, sourceFrames));
                targetEnd = Math.Clamp(targetEnd, targetStart + 1, Math.Max(targetStart + 1, targetFrames));
                int sourceCount = Math.Max(1, sourceEnd - sourceStart);
                int targetCount = Math.Max(1, targetEnd - targetStart);

                var report = WritePhoneMappedSegment(
                    sourceMel,
                    sourceStart,
                    sourceCount,
                    output,
                    targetStart,
                    targetCount,
                    phrase.phones[i],
                    targetF0);
                if (report.SplitApplied) {
                    splitCount++;
                } else {
                    fallbackCount++;
                }
                for (int t = targetStart; t < targetEnd && t < written.Length; t++) {
                    written[t] = true;
                }
            }

            FillUnwrittenFrames(output, written, AlignMelFrames(sourceMel, targetFrames));
            Log.Information(
                "HifiRoughFeatureBuilder phone_timewarp phones={Phones} split_segments={SplitSegments} fallback_segments={FallbackSegments} source_frames={SourceFrames} target_frames={TargetFrames}",
                phrase.phones.Length,
                splitCount,
                fallbackCount,
                sourceFrames,
                targetFrames);
            return output;
        }

        internal readonly record struct PhoneMapReport(bool SplitApplied, string Strategy, int ConsonantTargetFrames);

        internal static PhoneMapReport WritePhoneMappedSegment(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            RenderPhone phone,
            float[] targetF0) {
            double? consonantMs = EffectiveConsonantMs(phone);
            // VCV/CVVC timing has two important boundaries:
            // - preutter: the alignment point where the target vowel begins on the note grid.
            // - consonant: the end of the fixed transition/onset region in the source oto.
            // Keep preutter-leading material fixed, keep preutter->consonant as vowel onset, and
            // let only the stable vowel after consonant carry most stretch/compression.
            double preutterMs = consonantMs.HasValue ? Math.Max(0, phone.preutterMs) : 0;
            double stableStartMs = consonantMs.HasValue ? Math.Max(preutterMs, consonantMs.Value) : preutterMs;
            int sourceConsonantFrames = preutterMs > 0
                ? (int)Math.Round(preutterMs / SourceFrameMs)
                : 0;
            int sourceStableStartFrames = stableStartMs > 0
                ? (int)Math.Round(stableStartMs / SourceFrameMs)
                : sourceConsonantFrames;
            sourceFrames = Math.Max(1, Math.Min(sourceFrames, sourceMel.GetLength(1) - sourceStart));
            outputFrames = Math.Max(1, Math.Min(outputFrames, output.GetLength(1) - outputStart));
            sourceFrames = TrimInactiveTailFrames(sourceMel, sourceStart, sourceFrames, sourceConsonantFrames, phone.phoneme);
            sourceConsonantFrames = NormalizeSourceConsonantFrames(sourceConsonantFrames, sourceFrames, phone.phoneme);
            sourceStableStartFrames = NormalizeSourceStableStartFrames(sourceStableStartFrames, sourceConsonantFrames, sourceFrames);
            int sourceVowelOnsetFrames = Math.Max(0, sourceStableStartFrames - sourceConsonantFrames);
            if (outputFrames <= 2) {
                WriteCompactPhoneRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames, sourceConsonantFrames);
                return new PhoneMapReport(false, "compact_short_target", 0);
            }
            if (sourceFrames < MinSourceFrames) {
                WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return new PhoneMapReport(false, "simple_short_source", 0);
            }
            int vowelSourceFrames = sourceFrames - sourceConsonantFrames;
            if (vowelSourceFrames < MinVowelSourceFrames) {
                WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return new PhoneMapReport(false, "simple_no_vowel_room", 0);
            }

            // Target-axis fixed length uses the SAME preutter boundary as the source fixed split.
            // The segment starts at (positionMs - preutter), so this lands the vowel on positionMs.
            int targetConsonantFrames = preutterMs > 0
                ? (int)Math.Round(preutterMs / HifiF0Builder.FrameMs)
                : 0;
            targetConsonantFrames = Math.Clamp(targetConsonantFrames, 0, Math.Max(0, outputFrames - MinVowelTargetFrames));
            int vowelTargetFrames = outputFrames - targetConsonantFrames;
            if (vowelTargetFrames < MinVowelTargetFrames) {
                vowelTargetFrames = MinVowelTargetFrames;
                targetConsonantFrames = Math.Max(0, outputFrames - vowelTargetFrames);
            }

            if (sourceConsonantFrames > 0 && targetConsonantFrames > 0) {
                WriteMappedRegion(sourceMel, sourceStart, sourceConsonantFrames, output, outputStart, targetConsonantFrames);
            }

            var vowelMap = WriteVowelSourceToTargetMap(
                sourceMel,
                sourceStart + sourceConsonantFrames,
                vowelSourceFrames,
                output,
                outputStart + targetConsonantFrames,
                vowelTargetFrames,
                targetF0,
                phone.phoneme,
                sourceVowelOnsetFrames);

            double consonantRatio = sourceConsonantFrames > 0
                ? (targetConsonantFrames * HifiF0Builder.FrameMs) / Math.Max(SourceFrameMs, sourceConsonantFrames * SourceFrameMs)
                : 0;
            double vowelRatio = (vowelTargetFrames * HifiF0Builder.FrameMs) / Math.Max(SourceFrameMs, vowelSourceFrames * SourceFrameMs);
            Log.Debug(
                "HifiRoughFeatureBuilder phone_timewarp phoneme={Phoneme} source_frames={SourceFrames} target_frames={TargetFrames} source_fixed_frames={SourceFixedFrames} target_fixed_frames={TargetFixedFrames} source_vowel_onset_frames={SourceVowelOnsetFrames} target_vowel_onset_frames={TargetVowelOnsetFrames} source_vowel_frames={SourceVowelFrames} target_vowel_frames={TargetVowelFrames} fixed_ratio={FixedRatio:F4} vowel_ratio={VowelRatio:F4} strategy={Strategy}",
                phone.phoneme,
                sourceFrames,
                outputFrames,
                sourceConsonantFrames,
                targetConsonantFrames,
                sourceVowelOnsetFrames,
                vowelMap.TargetOnsetFrames,
                vowelSourceFrames,
                vowelTargetFrames,
                consonantRatio,
                vowelRatio,
                vowelMap.Strategy);
            // F0 mask is intentionally NARROWER than the fixed consonant region. The consonant
            // region (targetConsonantFrames) also contains the onset of the target vowel, which is
            // voiced; zeroing F0 over the whole region made that vowel onset lose its harmonic
            // source and the note sounded muffled/"dead". So we only mask the leading pure-consonant
            // portion, bounded by min(preutter, consonant) and kept strictly inside the region.
            int f0MaskFrames = ResolveConsonantF0MaskFrames(phone, consonantMs, targetConsonantFrames);
            return new PhoneMapReport(true, vowelMap.Strategy, f0MaskFrames);
        }

        static int[] BuildPhoneStarts(RenderPhrase phrase, double phraseStartMs, int totalFrames, double frameMs) {
            int phoneCount = phrase.phones.Length;
            var starts = new int[phoneCount];
            if (phoneCount == 0 || totalFrames <= 0) {
                return starts;
            }

            var durationWeights = new double[phoneCount];
            double previousMs = phraseStartMs;
            for (int i = 0; i < phoneCount; i++) {
                double currentMs = i == 0 ? phraseStartMs : PhoneSegmentStartMs(phrase.phones[i]);
                currentMs = Math.Max(currentMs, previousMs);
                double nextMs = i + 1 < phoneCount
                    ? Math.Max(currentMs, PhoneSegmentStartMs(phrase.phones[i + 1]))
                    : phraseStartMs + totalFrames * frameMs;
                durationWeights[i] = Math.Max(0, nextMs - currentMs) / frameMs;
                previousMs = currentMs;
            }

            int[] durations = AllocatePhoneDurations(durationWeights, totalFrames);
            int cursor = 0;
            for (int i = 0; i < phoneCount; i++) {
                starts[i] = Math.Clamp(cursor, 0, Math.Max(0, totalFrames - 1));
                cursor += durations[i];
            }
            return starts;
        }

        static int[] AllocatePhoneDurations(double[] weights, int totalFrames) {
            int count = weights.Length;
            var durations = new int[count];
            if (count == 0 || totalFrames <= 0) {
                return durations;
            }

            int activeCount = weights.Count(w => w > 1e-6);
            bool canKeepEveryPhoneAudible = activeCount > 0 && totalFrames >= activeCount;
            int used = 0;
            if (canKeepEveryPhoneAudible) {
                for (int i = 0; i < count; i++) {
                    if (weights[i] > 1e-6) {
                        durations[i] = 1;
                        used++;
                    }
                }
            }

            int remaining = Math.Max(0, totalFrames - used);
            double weightSum = weights.Sum();
            if (remaining > 0 && weightSum > 1e-6) {
                var fractional = new List<(int Index, double Fraction)>(count);
                int allocated = 0;
                for (int i = 0; i < count; i++) {
                    double exact = remaining * weights[i] / weightSum;
                    int whole = (int)Math.Floor(exact);
                    durations[i] += whole;
                    allocated += whole;
                    fractional.Add((i, exact - whole));
                }
                int left = remaining - allocated;
                foreach (var item in fractional.OrderByDescending(f => f.Fraction)) {
                    if (left <= 0) {
                        break;
                    }
                    durations[item.Index]++;
                    left--;
                }
            } else if (remaining > 0) {
                durations[count - 1] += remaining;
            }

            int diff = totalFrames - durations.Sum();
            if (diff != 0) {
                durations[count - 1] = Math.Max(0, durations[count - 1] + diff);
            }
            return durations;
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
            int preferredOnsetFrames = -1) {
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
                    allowMicroVariation: false);
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
                    "HifiRoughFeatureBuilder f0_mel_compensation reference_f0={ReferenceF0:F2} cut_frames={CutFrames} max_cut_db={MaxCutDb:F2}",
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
            bool allowMicroVariation) {
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
                allowMicroVariation);
        }

        static bool WriteSustainTemplateExtension(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            bool allowMicroVariation) {
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

            int stableStart = sourceStart + Math.Max(0, sourceFrames / 5);
            int stableEnd = sourceStart + sourceFrames - Math.Max(0, sourceFrames / 5);
            if (stableEnd <= stableStart) {
                stableStart = sourceStart;
                stableEnd = sourceStart + sourceFrames;
            }
            int stableFrames = Math.Max(1, stableEnd - stableStart);
            var stablePool = BuildStableSustainPool(sourceMel, stableStart, stableFrames);
            if (stablePool.Frames.Length == 0) {
                WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return false;
            }
            double stretchRatio = SourceToTargetDurationRatio(sourceFrames, outputFrames);
            double trajectoryBlend = ResolveTemplateTrajectoryBlend(stretchRatio);
            int smoothRadius = Math.Clamp(sourceFrames / 5, 1, 10);
            int edgeFrames = ResolveSustainEdgeFrames(outputFrames);
            var trajectoryFrame = new float[bins];
            var poolFrame = new float[bins];
            var directFrame = new float[bins];
            var residualFrame = new float[bins];
            double seed = SeedFromInts(sourceStart, outputFrames);

            for (int t = 0; t < outputFrames; t++) {
                double u = outputFrames == 1 ? 0 : t / (double)(outputFrames - 1);
                double anchorWeight = EdgeAnchorWeight(t, outputFrames, edgeFrames);
                double sourceIndex = outputFrames == 1 || sourceFrames == 1
                    ? 0
                    : u * (sourceFrames - 1);
                SampleInterpolatedFrame(sourceMel, sourceStart, sourceFrames, sourceIndex, directFrame);
                SampleSmoothedFrame(sourceMel, sourceStart, sourceFrames, sourceIndex, smoothRadius, trajectoryFrame);
                double sourceEnergy = FrameMean(trajectoryFrame) * (1.0 - anchorWeight) + FrameMean(directFrame) * anchorWeight;
                double poolIndex = StablePoolIndex(u, t, stablePool.Frames.Length, seed);
                SampleStablePoolFrame(sourceMel, stablePool.Frames, poolIndex, poolFrame);
                double targetEnergy = ResolveSustainTargetEnergy(stablePool.MeanEnergy, sourceEnergy);
                double currentEnergy = 0;
                double textureAmount = ResolveResidualTextureAmount(t, outputFrames, edgeFrames, stretchRatio);
                for (int m = 0; m < bins; m++) {
                    float stableValue = poolFrame[m];
                    float edgeValue = (float)(poolFrame[m] * (1.0 - trajectoryBlend) + trajectoryFrame[m] * trajectoryBlend);
                    edgeValue = (float)(edgeValue * (1.0 - anchorWeight) + directFrame[m] * anchorWeight);
                    float value = (float)(stableValue * (1.0 - anchorWeight) + edgeValue * anchorWeight);
                    trajectoryFrame[m] = value;
                    currentEnergy += value;
                }
                currentEnergy /= Math.Max(1, bins);
                double deDenseAmount = ResolveSustainDeDenseAmount(t, outputFrames, edgeFrames, stretchRatio);
                if (deDenseAmount > 1e-6) {
                    currentEnergy = ApplySustainSpectralDeDensity(trajectoryFrame, currentEnergy, deDenseAmount);
                }
                double residualMean = 0;
                for (int m = 0; m < bins; m++) {
                    float residual = (float)Math.Clamp(directFrame[m] - trajectoryFrame[m], -StableSustainResidualClamp, StableSustainResidualClamp);
                    residualFrame[m] = residual;
                    residualMean += residual;
                }
                residualMean /= Math.Max(1, bins);
                float energyDelta = (float)Math.Clamp(targetEnergy - currentEnergy, -StableSustainEnergyClamp, StableSustainEnergyClamp);
                // Deterministic, band-grouped spectral micro-variation. Faded out at the edges
                // (where frames anchor to the real source) and ramped in with stretch so only
                // genuinely held notes get it. Breaks the otherwise fixed period that the vocoder
                // would render as a mechanical loop.
                double microAmp = allowMicroVariation
                    ? SustainMicroVarLogAmp * (1.0 - anchorWeight) * Math.Clamp(stretchRatio - 1.0, 0, 1)
                    : 0;
                double framePhase = t * 0.20; // slow drift across the sustain
                for (int m = 0; m < bins; m++) {
                    double micro = 0;
                    if (microAmp > 1e-4) {
                        int band = m * SustainMicroVarBands / Math.Max(1, bins);
                        double bandSeed = SeedFromInts(sourceStart, band);
                        micro = microAmp * SmoothWobble(framePhase, bandSeed);
                    }
                    float residual = (float)((residualFrame[m] - residualMean) * textureAmount);
                    output[m, outputStart + t] = trajectoryFrame[m] + energyDelta + residual + (float)micro;
                }
            }
            SmoothInternalSustain(output, outputStart, outputFrames, strength: 0.16f);

            Log.Debug(
                "HifiRoughFeatureBuilder sustain_stable_pool_extension source_frames={SourceFrames} output_frames={OutputFrames} stable_frames={StableFrames} pool_frames={PoolFrames} edge_frames={EdgeFrames} smooth_radius={SmoothRadius} trajectory_blend={TrajectoryBlend:F3} micro_variation={MicroVariation}",
                sourceFrames,
                outputFrames,
                stableFrames,
                stablePool.Frames.Length,
                edgeFrames,
                smoothRadius,
                trajectoryBlend,
                allowMicroVariation);
            return true;
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

        static double ResolveTemplateTrajectoryBlend(double stretchRatio) {
            if (!IsFinite(stretchRatio) || stretchRatio <= 1.0) {
                return TemplateTrajectoryMaxBlend;
            }
            // Reach the min blend around ~2.5x stretch. Below that we keep more of the real source
            // motion so moderately-held notes don't sound frozen/looped.
            double amount = Math.Clamp((stretchRatio - 1.0) / 1.5, 0, 1);
            return TemplateTrajectoryMaxBlend + (TemplateTrajectoryMinBlend - TemplateTrajectoryMaxBlend) * amount;
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

        static double ResolveResidualTextureAmount(int frame, int totalFrames, int edgeFrames, double stretchRatio) {
            if (totalFrames < 32 || stretchRatio < 2.0 || edgeFrames <= 0) {
                return 0;
            }
            int distance = Math.Min(frame, totalFrames - 1 - frame);
            if (distance <= edgeFrames) {
                return 0;
            }
            double interior = Math.Clamp((distance - edgeFrames) / (double)Math.Max(1, edgeFrames), 0, 1);
            double stretch = Math.Clamp((stretchRatio - 2.0) / 2.0, 0, 1);
            return StableSustainResidualAmp * SmoothStep(interior) * stretch;
        }

        static double ResolveSustainDeDenseAmount(int frame, int totalFrames, int edgeFrames, double stretchRatio) {
            if (totalFrames < 32 || stretchRatio <= 1.6 || edgeFrames <= 0) {
                return 0;
            }
            int distance = Math.Min(frame, totalFrames - 1 - frame);
            if (distance <= edgeFrames) {
                return 0;
            }
            double interior = Math.Clamp((distance - edgeFrames) / (double)Math.Max(1, edgeFrames * 2), 0, 1);
            double stretch = Math.Clamp(Math.Log(stretchRatio / 1.6, 2.0) / 1.8, 0, 1);
            return StableSustainDeDenseMax * SmoothStep(interior) * stretch;
        }

        static double ApplySustainSpectralDeDensity(float[] frame, double frameMean, double amount) {
            if (frame.Length == 0 || amount <= 0) {
                return frameMean;
            }
            int bins = frame.Length;
            double newMean = 0;
            for (int m = 0; m < bins; m++) {
                double highWeight = HighBandWeight(m, bins, StableSustainDeDenseHighBandStart);
                double binAmount = Math.Clamp(amount * (1.0 + StableSustainDeDenseHighBandExtra * highWeight), 0, 0.55);
                frame[m] = (float)(frameMean + (frame[m] - frameMean) * (1.0 - binAmount));
                newMean += frame[m];
            }
            newMean /= bins;
            float correction = (float)(frameMean - newMean);
            double correctedMean = 0;
            for (int m = 0; m < bins; m++) {
                frame[m] += correction;
                correctedMean += frame[m];
            }
            return correctedMean / bins;
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

        static StableSustainPool BuildStableSustainPool(float[,] sourceMel, int start, int frames) {
            int totalFrames = sourceMel.GetLength(1);
            if (totalFrames <= 0 || frames <= 0) {
                return new StableSustainPool(Array.Empty<int>(), 0);
            }
            start = Math.Clamp(start, 0, Math.Max(0, totalFrames - 1));
            frames = Math.Max(1, Math.Min(frames, totalFrames - start));

            int margin = Math.Max(0, frames / 5);
            int first = start + margin;
            int end = start + frames - margin;
            if (end <= first) {
                first = start;
                end = start + frames;
            }

            var candidates = new List<(int Frame, double Energy)>();
            double maxEnergy = double.NegativeInfinity;
            for (int frame = first; frame < end; frame++) {
                double energy = FrameMean(sourceMel, frame);
                if (!IsFinite(energy)) {
                    continue;
                }
                candidates.Add((frame, energy));
                maxEnergy = Math.Max(maxEnergy, energy);
            }
            if (candidates.Count == 0) {
                return new StableSustainPool(Array.Empty<int>(), 0);
            }

            double floor = maxEnergy - StableSustainEnergyFloorLog;
            var selected = candidates
                .Where(candidate => candidate.Energy >= floor)
                .OrderBy(candidate => candidate.Frame)
                .ToList();
            int minFrames = Math.Min(candidates.Count, Math.Max(4, Math.Min(8, candidates.Count)));
            if (selected.Count < minFrames) {
                selected = candidates
                    .OrderByDescending(candidate => candidate.Energy)
                    .Take(minFrames)
                    .OrderBy(candidate => candidate.Frame)
                    .ToList();
            }
            if (selected.Count > StableSustainPoolMaxFrames) {
                selected = SelectEvenly(selected, StableSustainPoolMaxFrames);
            }

            double meanEnergy = selected.Average(candidate => candidate.Energy);
            return new StableSustainPool(selected.Select(candidate => candidate.Frame).ToArray(), meanEnergy);
        }

        static List<(int Frame, double Energy)> SelectEvenly(List<(int Frame, double Energy)> frames, int count) {
            if (frames.Count <= count) {
                return frames;
            }
            var selected = new List<(int Frame, double Energy)>(count);
            for (int i = 0; i < count; i++) {
                int index = count == 1
                    ? 0
                    : (int)Math.Round(i * (frames.Count - 1) / (double)(count - 1));
                selected.Add(frames[Math.Clamp(index, 0, frames.Count - 1)]);
            }
            return selected;
        }

        static double StablePoolIndex(double u, int targetFrame, int poolFrames, double seed) {
            if (poolFrames <= 1) {
                return 0;
            }
            double index = Math.Clamp(u, 0, 1) * (poolFrames - 1);
            if (poolFrames > 3) {
                index += 0.35 * SmoothWobble(targetFrame * 0.047, seed);
            }
            return Math.Clamp(index, 0, poolFrames - 1);
        }

        static void SampleStablePoolFrame(float[,] sourceMel, int[] poolFrames, double poolIndex, float[] output) {
            int bins = sourceMel.GetLength(0);
            if (poolFrames.Length == 0) {
                Array.Clear(output, 0, output.Length);
                return;
            }
            poolIndex = Math.Clamp(poolIndex, 0, Math.Max(0, poolFrames.Length - 1));
            int leftIndex = (int)Math.Floor(poolIndex);
            int rightIndex = Math.Min(poolFrames.Length - 1, leftIndex + 1);
            float alpha = (float)(poolIndex - leftIndex);
            int leftFrame = poolFrames[leftIndex];
            int rightFrame = poolFrames[rightIndex];
            for (int m = 0; m < bins; m++) {
                float left = sourceMel[m, leftFrame];
                float right = sourceMel[m, rightFrame];
                output[m] = left + (right - left) * alpha;
            }
        }

        static double FrameMean(float[,] mel, int frame) {
            int bins = mel.GetLength(0);
            double sum = 0;
            for (int m = 0; m < bins; m++) {
                sum += mel[m, frame];
            }
            return sum / Math.Max(1, bins);
        }

        static double EdgeAnchorWeight(int frame, int totalFrames, int edgeFrames) {
            if (edgeFrames <= 0 || totalFrames <= 1) {
                return 0;
            }
            int distance = Math.Min(frame, totalFrames - 1 - frame);
            if (distance >= edgeFrames) {
                return 0;
            }
            double edge = 1.0 - distance / (double)edgeFrames;
            return 0.35 * SmoothStep(edge);
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
                    "HifiRoughFeatureBuilder source_inactive_tail_trimmed phoneme={Phoneme} source_frames={SourceFrames} trimmed_source_frames={TrimmedSourceFrames} max_energy={MaxEnergy:F4} threshold={Threshold:F4}",
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
                    "HifiRoughFeatureBuilder source_consonant_clamped phoneme={Phoneme} source_frames={SourceFrames} original_source_consonant_frames={OriginalSourceConsonantFrames} clamped_source_consonant_frames={ClampedSourceConsonantFrames}",
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
            // The oto Consonant marks the fixed (non-stretched) region: the leading vowel tail,
            // the consonant, and the onset of the target vowel. It is almost always LONGER than
            // preutter (which is only the timing-alignment lead), so we must NOT clamp it down to
            // preutter; doing that pushed the consonant/transition into the vowel stretch region,
            // which is exactly why long Japanese-VCV notes audibly stretched their consonants.
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

        static void FillUnwrittenFrames(float[,] output, bool[] written, float[,] fallback) {
            int bins = output.GetLength(0);
            int frames = output.GetLength(1);
            for (int t = 0; t < frames; t++) {
                if (written[t]) {
                    continue;
                }
                for (int m = 0; m < bins; m++) {
                    output[m, t] = fallback[m, t];
                }
            }
        }

        static void BufferCopy(float[,] source, float[,] destination) {
            for (int m = 0; m < source.GetLength(0); m++) {
                for (int t = 0; t < source.GetLength(1); t++) {
                    destination[m, t] = source[m, t];
                }
            }
        }

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
            return metadata;
        }

        static void Validate(float[,] mel, float[] f0) {
            foreach (float value in mel) {
                if (float.IsNaN(value) || float.IsInfinity(value)) {
                    throw new InvalidOperationException("rough phrase_mel contains NaN or Inf.");
                }
            }
            foreach (float value in f0) {
                if (float.IsNaN(value) || float.IsInfinity(value)) {
                    throw new InvalidOperationException("rough phrase_f0 contains NaN or Inf.");
                }
            }
        }
    }
}
