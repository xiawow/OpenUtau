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
        const int PhraseEdgeMelFadeInFrames = 3;
        const int PhraseEdgeMelFadeOutFrames = 5;
        const double PhraseEdgeMelMinGain = 0.18;
        const int InactiveTailGuardFrames = 6;
        const double InactiveTailLogDrop = 2.8;
        const double TemplateEnergyClamp = 0.70;
        const double TemplateTrajectoryMinBlend = 0.38;
        const double TemplateTrajectoryMaxBlend = 0.70;
        const double SourceFrameMs = 1000.0 * HifiMelExtractor.OriginHopSize / HifiMelExtractor.SampleRate;
        const double SourceSampleMs = 1000.0 / HifiMelExtractor.SampleRate;
        const int MinSourceSamples = MinSourceFrames * HifiMelExtractor.OriginHopSize;
        const int MinVowelSourceSamples = MinVowelSourceFrames * HifiMelExtractor.OriginHopSize;
        const int InactiveTailGuardSamples = InactiveTailGuardFrames * HifiMelExtractor.OriginHopSize;
        const double F0AwareMaxSlopeCentsPerFrame = 160.0;

        readonly HifiMelExtractor melExtractor = new HifiMelExtractor();
        readonly HifiF0Builder f0Builder = new HifiF0Builder();
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

        public HifiRoughFeatureBuilder(IHifiMelEnhancer melEnhancer) {
            this.melEnhancer = melEnhancer;
        }

        public HifiPhraseFeatures Build(RenderPhrase phrase, RenderResult layout, float[] roughSamples) {
            double phraseStartMs = layout.positionMs - layout.leadingMs;
            int targetFrames = Math.Max(1, (int)Math.Ceiling(layout.estimatedLengthMs / HifiF0Builder.FrameMs));
            float[] f0 = f0Builder.Build(phrase, targetFrames, phraseStartMs);
            double[] sourcePositions = BuildVariableSourcePositions(phrase, phraseStartMs, roughSamples, targetFrames);
            float[,] alignedMel = melExtractor.ExtractAtPositions(roughSamples, sourcePositions);

            double roughDurationMs = roughSamples.Length * 1000.0 / HifiMelExtractor.SampleRate;
            double targetDurationMs = targetFrames * HifiF0Builder.FrameMs;
            double stretchRatio = roughDurationMs > 1e-6 ? targetDurationMs / roughDurationMs : 1.0;
            int nominalSourceFrames = roughSamples.Length <= 0
                ? 0
                : Math.Max(1, (int)Math.Ceiling(roughSamples.Length / (double)HifiF0Builder.HopSize));
            Log.Information(
                "HifiRoughFeatureBuilder variable_position_mel mode=overlap_neural rough_duration_ms={RoughDurationMs:F3} target_duration_ms={TargetDurationMs:F3} stretch_ratio={StretchRatio:F4} nominal_source_frames={Before} mel_frames_after={After} source_pos_min={SourcePosMin:F1} source_pos_max={SourcePosMax:F1}",
                roughDurationMs,
                targetDurationMs,
                stretchRatio,
                nominalSourceFrames,
                targetFrames,
                sourcePositions.Length == 0 ? 0 : sourcePositions.Min(),
                sourcePositions.Length == 0 ? 0 : sourcePositions.Max());
            if (stretchRatio > 1.5 || stretchRatio < 0.67) {
                Log.Warning(
                    "HifiRoughFeatureBuilder variable_position_mel ratio_out_of_range mode=overlap_neural stretch_ratio={StretchRatio:F4} rough_duration_ms={RoughDurationMs:F3} target_duration_ms={TargetDurationMs:F3}",
                    stretchRatio,
                    roughDurationMs,
                    targetDurationMs);
            }

            float[,] enhancedMel = melEnhancer.Enhance(alignedMel, f0);
            ApplyPhraseEdgeMelGuard(enhancedMel);
            Validate(enhancedMel, f0);

            return new HifiPhraseFeatures {
                Mel = enhancedMel,
                F0 = f0,
                Metadata = BuildMetadata(phrase, layout, targetFrames, phraseStartMs),
            };
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

        readonly record struct PhoneMapReport(bool SplitApplied, string Strategy);

        static PhoneMapReport WritePhoneMappedSegment(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames,
            RenderPhone phone,
            float[] targetF0) {
            double? consonantMs = EffectiveConsonantMs(phone);
            int sourceConsonantFrames = consonantMs.HasValue
                ? (int)Math.Round(consonantMs.Value / SourceFrameMs)
                : 0;
            sourceFrames = Math.Max(1, Math.Min(sourceFrames, sourceMel.GetLength(1) - sourceStart));
            outputFrames = Math.Max(1, Math.Min(outputFrames, output.GetLength(1) - outputStart));
            sourceFrames = TrimInactiveTailFrames(sourceMel, sourceStart, sourceFrames, sourceConsonantFrames, phone.phoneme);
            sourceConsonantFrames = NormalizeSourceConsonantFrames(sourceConsonantFrames, sourceFrames, phone.phoneme);
            if (outputFrames <= 2) {
                WriteCompactPhoneRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames, sourceConsonantFrames);
                return new PhoneMapReport(false, "compact_short_target");
            }
            if (sourceFrames < MinSourceFrames) {
                WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return new PhoneMapReport(false, "simple_short_source");
            }
            int vowelSourceFrames = sourceFrames - sourceConsonantFrames;
            if (vowelSourceFrames < MinVowelSourceFrames) {
                WriteMappedRegion(sourceMel, sourceStart, sourceFrames, output, outputStart, outputFrames);
                return new PhoneMapReport(false, "simple_no_vowel_room");
            }

            int targetConsonantFrames = consonantMs.HasValue
                ? (int)Math.Round(consonantMs.Value / HifiF0Builder.FrameMs)
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
                phone.phoneme);

            double consonantRatio = sourceConsonantFrames > 0
                ? (targetConsonantFrames * HifiF0Builder.FrameMs) / Math.Max(SourceFrameMs, sourceConsonantFrames * SourceFrameMs)
                : 0;
            double vowelRatio = (vowelTargetFrames * HifiF0Builder.FrameMs) / Math.Max(SourceFrameMs, vowelSourceFrames * SourceFrameMs);
            Log.Information(
                "HifiRoughFeatureBuilder phone_timewarp phoneme={Phoneme} source_frames={SourceFrames} target_frames={TargetFrames} source_consonant_frames={SourceConsonantFrames} target_consonant_frames={TargetConsonantFrames} source_vowel_frames={SourceVowelFrames} target_vowel_frames={TargetVowelFrames} consonant_ratio={ConsonantRatio:F4} vowel_ratio={VowelRatio:F4} strategy={Strategy}",
                phone.phoneme,
                sourceFrames,
                outputFrames,
                sourceConsonantFrames,
                targetConsonantFrames,
                vowelSourceFrames,
                vowelTargetFrames,
                consonantRatio,
                vowelRatio,
                vowelMap.Strategy);
            return new PhoneMapReport(true, vowelMap.Strategy);
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
            string phoneme) {
            if (dstFrames <= 0) {
                return new VowelMapReport(0, 0, 0, 0, "empty_vowel_target", 0, 0, 0);
            }

            ResolveVowelSections(vowelSourceFrames, out int onsetFrames, out int releaseFrames, out int sourceSustainFrames);
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
            if (vowelSourceFrames <= 3) {
                onsetFrames = Math.Max(1, vowelSourceFrames - 1);
                releaseFrames = 0;
                sustainFrames = Math.Max(1, vowelSourceFrames - onsetFrames);
                return;
            }
            onsetFrames = Math.Clamp((int)Math.Round(vowelSourceFrames * 0.10), 1, 5);
            releaseFrames = Math.Clamp((int)Math.Round(vowelSourceFrames * 0.08), 1, 5);
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

            double ratio = Math.Min(1.0, targetTotalFrames / (double)naturalTotalFrames);
            int onsetFrames = Math.Clamp(
                (int)Math.Round(naturalOnsetFrames * ratio),
                sourceOnsetFrames > 0 ? 1 : 0,
                Math.Min(naturalOnsetFrames, targetTotalFrames));
            int releaseFrames = Math.Clamp(
                (int)Math.Round(naturalReleaseFrames * ratio),
                0,
                Math.Min(naturalReleaseFrames, Math.Max(0, targetTotalFrames - onsetFrames)));

            while (targetTotalFrames >= 3 && onsetFrames + releaseFrames > targetTotalFrames - 1) {
                if (onsetFrames >= releaseFrames && onsetFrames > 1) {
                    onsetFrames--;
                } else if (releaseFrames > 0) {
                    releaseFrames--;
                } else {
                    break;
                }
            }
            int sustainFrames = Math.Max(0, targetTotalFrames - onsetFrames - releaseFrames);
            if (targetTotalFrames >= 3 && sourceSustainFrames > 0 && sustainFrames <= 0) {
                if (releaseFrames > 0) {
                    releaseFrames--;
                } else if (onsetFrames > 1) {
                    onsetFrames--;
                }
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
            return WriteSustainTemplateExtension(
                sourceMel,
                sourceStart,
                sourceFrames,
                output,
                outputStart,
                outputFrames);
        }

        static bool WriteSustainTemplateExtension(
            float[,] sourceMel,
            int sourceStart,
            int sourceFrames,
            float[,] output,
            int outputStart,
            int outputFrames) {
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
            int splitFrame = stableStart + stableFrames / 2;
            var leftTemplate = BuildMedianTemplate(sourceMel, stableStart, Math.Max(1, splitFrame - stableStart));
            var rightTemplate = BuildMedianTemplate(sourceMel, splitFrame, Math.Max(1, stableEnd - splitFrame));
            var centerTemplate = BuildMedianTemplate(sourceMel, stableStart, stableFrames);
            double leftEnergy = Mean(leftTemplate);
            double rightEnergy = Mean(rightTemplate);
            double centerEnergy = Mean(centerTemplate);
            double stretchRatio = SourceToTargetDurationRatio(sourceFrames, outputFrames);
            double trajectoryBlend = ResolveTemplateTrajectoryBlend(stretchRatio);
            int smoothRadius = Math.Clamp(sourceFrames / 5, 1, 10);
            int edgeFrames = ResolveSustainEdgeFrames(outputFrames);
            var trajectoryFrame = new float[bins];
            var templateFrame = new float[bins];
            var directFrame = new float[bins];

            for (int t = 0; t < outputFrames; t++) {
                double u = outputFrames == 1 ? 0 : t / (double)(outputFrames - 1);
                double drift = SmoothStep(u);
                double anchorWeight = EdgeAnchorWeight(t, outputFrames, edgeFrames);
                double sourceIndex = outputFrames == 1 || sourceFrames == 1
                    ? 0
                    : u * (sourceFrames - 1);
                SampleInterpolatedFrame(sourceMel, sourceStart, sourceFrames, sourceIndex, directFrame);
                SampleSmoothedFrame(sourceMel, sourceStart, sourceFrames, sourceIndex, smoothRadius, trajectoryFrame);
                double targetEnergy = centerEnergy + ((leftEnergy + (rightEnergy - leftEnergy) * drift) - centerEnergy) * 0.35;
                double currentEnergy = 0;
                for (int m = 0; m < bins; m++) {
                    templateFrame[m] = (float)(leftTemplate[m] + (rightTemplate[m] - leftTemplate[m]) * drift);
                    float value = (float)(templateFrame[m] * (1.0 - trajectoryBlend) + trajectoryFrame[m] * trajectoryBlend);
                    value = (float)(value * (1.0 - anchorWeight) + directFrame[m] * anchorWeight);
                    trajectoryFrame[m] = value;
                    currentEnergy += value;
                }
                currentEnergy /= Math.Max(1, bins);
                float energyDelta = (float)Math.Clamp(targetEnergy - currentEnergy, -TemplateEnergyClamp, TemplateEnergyClamp);
                for (int m = 0; m < bins; m++) {
                    output[m, outputStart + t] = trajectoryFrame[m] + energyDelta;
                }
            }
            SmoothInternalSustain(output, outputStart, outputFrames, strength: 0.16f);

            Log.Information(
                "HifiRoughFeatureBuilder sustain_template_extension source_frames={SourceFrames} output_frames={OutputFrames} stable_frames={StableFrames} edge_frames={EdgeFrames} smooth_radius={SmoothRadius} trajectory_blend={TrajectoryBlend:F3}",
                sourceFrames,
                outputFrames,
                stableFrames,
                edgeFrames,
                smoothRadius,
                trajectoryBlend);
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
            double amount = Math.Clamp((stretchRatio - 1.0) / 3.0, 0, 1);
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

        static float[] BuildMedianTemplate(float[,] sourceMel, int start, int frames) {
            int bins = sourceMel.GetLength(0);
            int totalFrames = sourceMel.GetLength(1);
            start = Math.Clamp(start, 0, Math.Max(0, totalFrames - 1));
            frames = Math.Max(1, Math.Min(frames, totalFrames - start));
            var template = new float[bins];
            var values = new float[frames];
            for (int m = 0; m < bins; m++) {
                for (int t = 0; t < frames; t++) {
                    values[t] = sourceMel[m, start + t];
                }
                Array.Sort(values);
                template[m] = values[frames / 2];
            }
            return template;
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

        static double Mean(float[] values) {
            if (values.Length == 0) {
                return 0;
            }
            double sum = 0;
            foreach (float value in values) {
                sum += value;
            }
            return sum / values.Length;
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

        static double? EffectiveConsonantMs(RenderPhone phone) {
            if (phone.oto == null || phone.oto.Consonant <= 0) {
                return null;
            }
            double consonantMs = phone.oto.Consonant;
            if (phone.preutterMs > 0 && consonantMs > phone.preutterMs) {
                consonantMs = phone.preutterMs;
            }
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

        static HifiPhraseMetadata BuildMetadata(RenderPhrase phrase, RenderResult layout, int frames, double phraseStartMs) {
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

            int[] phoneStarts = BuildPhoneStarts(phrase, phraseStartMs, frames, HifiF0Builder.FrameMs);

            for (int i = 0; i < phrase.phones.Length; i++) {
                var p = phrase.phones[i];
                int start = phoneStarts[i];
                int end = i + 1 < phoneStarts.Length ? phoneStarts[i + 1] : frames;
                end = Math.Clamp(end, start + 1, Math.Max(start + 1, frames));
                metadata.Phones.Add(new HifiPhoneMetadata {
                    Index = i,
                    Phoneme = p.phoneme,
                    Tone = p.tone,
                    SourceFile = p.oto?.File ?? string.Empty,
                    PositionMs = p.positionMs,
                    DurationMs = p.durationMs,
                    LeadingMs = p.leadingMs,
                    StartFrame = start,
                    FrameCount = Math.Max(1, end - start),
                    SourceSkipOverMs = 0,
                    SourceStartOffsetFrames = 0,
                });
                if (i > 0) {
                    metadata.Boundaries.Add(new HifiBoundaryMetadata {
                        Index = i - 1,
                        LeftPhoneIndex = i - 1,
                        RightPhoneIndex = i,
                        LeftPhone = phrase.phones[i - 1].phoneme,
                        RightPhone = p.phoneme,
                        Frame = start,
                        PositionMs = phraseStartMs + start * HifiF0Builder.FrameMs,
                        TransitionType = "phone",
                    });
                }
            }
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
