using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenUtau.Core.Render;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiPhraseFeatureBuilder {
        readonly HifiMelExtractor melExtractor = new HifiMelExtractor();
        readonly HifiF0Builder f0Builder = new HifiF0Builder();

        public HifiPhraseFeatures Build(RenderPhrase phrase, RenderResult layout) {
            var stretchMode = HifiNeuralConfig.StretchMode;
            int frames = Math.Max(1, (int)Math.Ceiling(layout.estimatedLengthMs / HifiF0Builder.FrameMs));
            var phraseMel = new float[HifiMelExtractor.NMels, frames];
            var weights = new float[frames];
            double phraseStartMs = layout.positionMs - layout.leadingMs;
            var durationPlans = BuildDurationPlans(phrase, phraseStartMs, frames);
            var durationResult = HifiDurationRedistributor.Redistribute(
                durationPlans,
                HifiNeuralConfig.EnableDurationBorrow && stretchMode == HifiStretchMode.ConsonantVowelSplit,
                frames,
                HifiNeuralConfig.MinConsonantMs,
                HifiNeuralConfig.MinVowelMs,
                HifiNeuralConfig.MaxBorrowMs,
                HifiNeuralConfig.MaxBorrowRatio,
                HifiNeuralConfig.ConsonantLockMode,
                HifiNeuralConfig.EnableCodaProtection,
                HifiNeuralConfig.MinCodaMs);
            durationPlans = durationResult.Plans;
            var metadata = BuildMetadata(phrase, layout, frames, phraseStartMs, durationPlans);
            var preparedPhones = new List<HifiPreparedPhoneMel>();
            var f0 = f0Builder.Build(phrase, frames, phraseStartMs);

            foreach (var phone in phrase.phones.Select((value, index) => new { value, index })) {
                var p = phone.value;
                var durationPlan = durationPlans[phone.index];
                if (p.oto == null || string.IsNullOrWhiteSpace(p.oto.File) || !File.Exists(p.oto.File)) {
                    Log.Warning("HifiPhraseFeatureBuilder missing source wav for phone {Phone}", p.phoneme);
                    continue;
                }

                float[] source = melExtractor.LoadMono(p.oto.File);
                source = SliceOto(source, p.oto.Offset, p.oto.Cutoff);
                var sourceAlignment = CalculateSourceAlignment(p);
                source = ApplySourceSkip(source, sourceAlignment, p.phoneme);
                int maxStartOffsetFrames = MaxSourceStartOffsetForPhone(durationPlan);
                int maxSanitizeOffsetFrames = Math.Max(0, maxStartOffsetFrames - sourceAlignment.StartOffsetFrames);
                var leadingSanitizer = HifiSourceLeadingSanitizer.Analyze(
                    p.phoneme,
                    phone.index > 0 ? phrase.phones[phone.index - 1].phoneme : null,
                    IsFirstOrAfterSilence(phrase.phones, phone.index),
                    sourceAlignment.SourcePreutterMs,
                    maxSanitizeOffsetFrames);
                source = HifiSourceLeadingSanitizer.Apply(source, leadingSanitizer);
                if (source.Length == 0) {
                    continue;
                }
                var phoneMel = melExtractor.Extract(source);
                int sourceStartOffsetFrames = ClampSourceStartOffsetForShortPhone(
                    sourceAlignment.StartOffsetFrames,
                    durationPlan,
                    p.phoneme);
                sourceStartOffsetFrames = ClampSourceStartOffsetForShortPhoneVowelContinuity(
                    sourceStartOffsetFrames,
                    durationPlan,
                    p.phoneme);
                if (sourceStartOffsetFrames > 0) {
                    phoneMel = TrimLeadingMelFrames(phoneMel, sourceStartOffsetFrames);
                }
                int targetFrames = Math.Max(1, durationPlan.AdjustedDurationFrames);
                int targetConsonantFrames = Math.Max(0, durationPlan.TargetConsonantFrames - sourceStartOffsetFrames);
                int alignedStartFrame = durationPlan.AdjustedStartFrame;
                double? effectiveConsonantMs = EffectiveConsonantMs(
                    p,
                    HifiSourceLeadingSanitizer.SourcePreutterAfterSanitize(leadingSanitizer));
                var targetF0 = SliceF0(f0, alignedStartFrame, targetFrames);
                var stretched = HifiPhoneMelStretcher.Stretch(
                    phoneMel,
                    targetFrames,
                    p.phoneme,
                    effectiveConsonantMs,
                    stretchMode,
                    targetConsonantFrames,
                    durationPlan.BorrowedFromPreviousFrames + durationPlan.BorrowedFromNextFrames,
                    durationPlan.BorrowedFramesAppliedTo,
                    HifiNeuralConfig.ConsonantLockMode,
                    targetF0);
                double sourceF0 = HifiPitchMelCompensator.EstimateSourceF0(source, effectiveConsonantMs);
                HifiPitchMelCompensator.Apply(
                    stretched.Mel,
                    targetF0,
                    p.phoneme,
                    targetConsonantFrames,
                    sourceF0);
                metadata.PhoneDiagnostics.Add(HifiClickDiagnostic.BuildPhoneFeatureDiagnostic(
                    phone.index,
                    p.phoneme,
                    alignedStartFrame,
                    targetFrames,
                    source,
                    phoneMel,
                    stretched.Mel,
                    targetF0,
                    leadingSanitizer.Reason));
                int phoneCrossfadeFrames = Math.Max(HifiNeuralConfig.BoundaryCrossfadeFrames, MsToFrame(Math.Max(0, p.overlapMs)));
                preparedPhones.Add(new HifiPreparedPhoneMel {
                    Index = phone.index,
                    Phoneme = p.phoneme,
                    Mel = stretched.Mel,
                    NominalStartFrame = durationPlan.AdjustedStartFrame,
                    StartFrame = alignedStartFrame,
                    TargetConsonantFrames = targetConsonantFrames,
                    CrossfadeFrames = phoneCrossfadeFrames,
                });
            }

            HifiMelPhraseComposer.Compose(phraseMel, weights, preparedPhones, frames, f0);
            HifiMelPostProcessor.NormalizeWeightedMel(phraseMel, weights);
            FillGaps(phraseMel, weights);
            if (HifiNeuralConfig.EnableVoicedOnsetDipRepair) {
                HifiMelPostProcessor.RepairVoicedOnsetDips(
                    phraseMel,
                    f0,
                    durationPlans,
                    metadata.PhoneDiagnostics,
                    HifiNeuralConfig.VoicedOnsetDipRepairFrames,
                    HifiNeuralConfig.VoicedOnsetDipMinDelta,
                    HifiNeuralConfig.VoicedOnsetDipMaxLift);
            }
            HifiMelPostProcessor.SmoothBoundaries(
                phraseMel,
                StableVowelBoundaryFrames(metadata.Boundaries),
                HifiNeuralConfig.BoundarySmoothFrames,
                "phone_boundary",
                ConsonantRanges(durationPlans));
            HifiMelPostProcessor.SmoothBoundaries(
                phraseMel,
                StableVowelDurationBorrowBoundaryFrames(durationPlans),
                HifiNeuralConfig.DurationBorrowSmoothFrames,
                "duration_borrow",
                ConsonantRanges(durationPlans));
            if (HifiNeuralConfig.EnableBoundaryEnergyMatching
                && HifiNeuralConfig.BoundaryEnergyPreset == HifiNeuralConfig.EnergyConservative) {
                HifiBoundaryEnergyMatcher.ApplyConservative(
                    phraseMel,
                    metadata.Boundaries,
                    durationPlans,
                    HifiNeuralConfig.BoundaryEnergyWindowFrames,
                    HifiNeuralConfig.BoundaryEnergyMaxGainDb);
            }
            HifiMelPostProcessor.ApplyVoicedContinuity(
                phraseMel,
                durationPlans,
                ConsonantRanges(durationPlans),
                HifiNeuralConfig.VoicedContinuityRadius,
                HifiNeuralConfig.VoicedContinuityStrength);
            HifiMelPostProcessor.ApplyVoicingMask(f0, durationPlans);
            if (HifiNeuralConfig.EnableVoicedIslandContinuity) {
                HifiVoicedIslandSmoother.Apply(
                    phraseMel,
                    f0,
                    durationPlans,
                    HifiNeuralConfig.VoicedIslandMorphFrames,
                    HifiNeuralConfig.VoicedIslandSmoothRadiusFrames,
                    HifiNeuralConfig.VoicedIslandMorphStrength,
                    HifiNeuralConfig.VoicedIslandSmoothStrength,
                    HifiNeuralConfig.VoicedIslandMaxF0BridgeFrames,
                    HifiNeuralConfig.VoicedIslandMaxF0BridgeCents,
                    HifiNeuralConfig.RapidShortPhoneFrames,
                    HifiNeuralConfig.VoicedIslandMinStableFrames);
            }
            HifiMelPostProcessor.SmoothInternalF0Boundaries(
                f0,
                metadata,
                HifiNeuralConfig.F0BoundarySmoothFrames,
                HifiNeuralConfig.F0BoundarySmoothMinJumpCents);
            HifiMelPostProcessor.ApplyHeadAttack(
                phraseMel,
                f0,
                durationPlans,
                HifiNeuralConfig.HeadAttackFrames,
                HifiNeuralConfig.HeadAttackGainDb,
                HifiNeuralConfig.HeadAttackF0Frames);
            HifiMelPostProcessor.ApplyTailRelease(
                phraseMel,
                f0,
                durationPlans,
                HifiNeuralConfig.TailReleaseFrames,
                HifiNeuralConfig.TailReleaseGainDb,
                HifiNeuralConfig.TailReleaseF0Frames);
            Validate(phraseMel, f0);
            LogStats(phraseMel, f0, metadata);
            return new HifiPhraseFeatures {
                Mel = phraseMel,
                F0 = f0,
                Metadata = metadata,
            };
        }

        static float[] SliceOto(float[] samples, double offsetMs, double cutoffMs) {
            int offset = Math.Clamp((int)Math.Round(offsetMs * HifiMelExtractor.SampleRate / 1000.0), 0, samples.Length);
            int end = cutoffMs >= 0
                ? samples.Length - Math.Max(0, (int)Math.Round(cutoffMs * HifiMelExtractor.SampleRate / 1000.0))
                : offset + Math.Max(0, (int)Math.Round(-cutoffMs * HifiMelExtractor.SampleRate / 1000.0));
            end = Math.Clamp(end, offset, samples.Length);
            return samples.Skip(offset).Take(end - offset).ToArray();
        }

        static HifiSourceAlignment CalculateSourceAlignment(RenderPhone phone, bool log = true) {
            double stretchRatio = Math.Pow(2, 1.0 - phone.velocity);
            double pitchLeadingMs = Math.Max(0, (phone.oto?.Preutter ?? 0) * stretchRatio);
            double leadingMs = Math.Max(0, phone.leadingMs);
            double skipOverMs = pitchLeadingMs - leadingMs;
            int skipSamples = skipOverMs > 0
                ? (int)Math.Round(skipOverMs * HifiMelExtractor.SampleRate / 1000.0)
                : 0;
            int startOffsetFrames = skipOverMs < 0
                ? MsToFrame(-skipOverMs)
                : 0;
            double sourcePreutterMs = Math.Max(0, pitchLeadingMs - Math.Max(0, skipOverMs));
            if (log) {
                Log.Information(
                    "HifiPhraseFeatureBuilder source_alignment phoneme={Phoneme} oto_preutter_ms={OtoPreutterMs:F3} pitch_leading_ms={PitchLeadingMs:F3} leading_ms={LeadingMs:F3} velocity={Velocity:F3} stretch_ratio={StretchRatio:F4} skip_over_ms={SkipOverMs:F3} skip_samples={SkipSamples} start_offset_frames={StartOffsetFrames} source_preutter_ms={SourcePreutterMs:F3}",
                    phone.phoneme,
                    phone.oto?.Preutter ?? 0,
                    pitchLeadingMs,
                    leadingMs,
                    phone.velocity,
                    stretchRatio,
                    skipOverMs,
                    skipSamples,
                    startOffsetFrames,
                    sourcePreutterMs);
            }
            return new HifiSourceAlignment(skipOverMs, skipSamples, startOffsetFrames, sourcePreutterMs);
        }

        static float[] ApplySourceSkip(float[] samples, HifiSourceAlignment alignment, string phoneme) {
            if (samples.Length == 0 || alignment.SkipSamples <= 0) {
                return samples;
            }
            int skip = Math.Min(alignment.SkipSamples, Math.Max(0, samples.Length - 1));
            if (skip <= 0) {
                return samples;
            }
            if (skip >= samples.Length - 1) {
                Log.Warning(
                    "HifiPhraseFeatureBuilder source_skip_near_empty phoneme={Phoneme} skip_over_ms={SkipOverMs:F3} skip_samples={SkipSamples} source_samples={SourceSamples}",
                    phoneme,
                    alignment.SkipOverMs,
                    alignment.SkipSamples,
                    samples.Length);
            }
            return samples.Skip(skip).ToArray();
        }

        static int ClampSourceStartOffsetForShortPhone(int sourceStartOffsetFrames, HifiPhoneDurationPlan plan, string phoneme) {
            if (sourceStartOffsetFrames <= 0) {
                return 0;
            }
            int maxOffset = MaxSourceStartOffsetForPhone(plan);
            int clamped = Math.Min(sourceStartOffsetFrames, maxOffset);
            if (clamped != sourceStartOffsetFrames) {
                Log.Information(
                    "HifiPhraseFeatureBuilder short_phone_start_offset_clamped phoneme={Phoneme} original_offset_frames={OriginalOffsetFrames} clamped_offset_frames={ClampedOffsetFrames} duration_frames={DurationFrames} min_rendered_frames={MinRenderedFrames}",
                    phoneme,
                    sourceStartOffsetFrames,
                    clamped,
                    plan.AdjustedDurationFrames,
                    HifiNeuralConfig.ShortPhoneMinRenderedFrames);
            }
            return clamped;
        }

        static int MaxSourceStartOffsetForPhone(HifiPhoneDurationPlan plan) {
            int maxOffset = Math.Max(0, plan.AdjustedDurationFrames - 1);
            if (plan.AdjustedDurationFrames <= HifiNeuralConfig.RapidShortPhoneFrames) {
                int minRenderedFrames = Math.Min(
                    plan.AdjustedDurationFrames,
                    Math.Max(1, HifiNeuralConfig.ShortPhoneMinRenderedFrames));
                maxOffset = Math.Max(0, plan.AdjustedDurationFrames - minRenderedFrames);
            }
            return Math.Min(maxOffset, HifiNeuralConfig.SourceStartOffsetMaxFrames);
        }

        static int ClampSourceStartOffsetForShortPhoneVowelContinuity(
            int sourceStartOffsetFrames,
            HifiPhoneDurationPlan plan,
            string phoneme) {
            if (sourceStartOffsetFrames <= 0 || !IsStableVowel(phoneme)) {
                return Math.Max(0, sourceStartOffsetFrames);
            }
            int durationFrames = Math.Max(1, plan.AdjustedDurationFrames);
            if (durationFrames > Math.Max(1, HifiNeuralConfig.RapidShortPhoneFrames)) {
                return sourceStartOffsetFrames;
            }
            int minVowelFrames = ShortPhoneMinVowelFrames(durationFrames);
            int maxOffsetByVowel = Math.Max(0, durationFrames - minVowelFrames);
            int clamped = Math.Min(sourceStartOffsetFrames, maxOffsetByVowel);
            if (clamped != sourceStartOffsetFrames) {
                Log.Information(
                    "HifiPhraseFeatureBuilder short_phone_vowel_guard phoneme={Phoneme} original_offset_frames={OriginalOffsetFrames} clamped_offset_frames={ClampedOffsetFrames} duration_frames={DurationFrames} min_vowel_frames={MinVowelFrames}",
                    phoneme,
                    sourceStartOffsetFrames,
                    clamped,
                    durationFrames,
                    minVowelFrames);
            }
            return clamped;
        }

        static int ShortPhoneMinVowelFrames(int durationFrames) {
            if (durationFrames <= 1) {
                return 1;
            }
            int floorByConfig = Math.Max(2, HifiNeuralConfig.ShortPhoneMinRenderedFrames);
            int floorByDuration = Math.Max(1, (durationFrames + 1) / 2);
            int minVowel = Math.Min(floorByConfig, floorByDuration);
            return Math.Clamp(minVowel, 1, durationFrames);
        }

        static List<HifiPhoneDurationPlan> BuildDurationPlans(RenderPhrase phrase, double phraseStartMs, int phraseFrames) {
            if (phrase.phones.Length == 0) {
                return new List<HifiPhoneDurationPlan>();
            }
            var starts = new int[phrase.phones.Length];
            for (int i = 0; i < phrase.phones.Length; i++) {
                int start = MsToFrame(AcousticStartMs(phrase.phones[i]) - phraseStartMs);
                if (i > 0) {
                    start = Math.Max(start, starts[i - 1]);
                }
                starts[i] = Math.Clamp(start, 0, Math.Max(0, phraseFrames - 1));
            }
            return phrase.phones.Select((phone, index) => {
                double? consonantMs = EffectiveConsonantMs(phone);
                int consonantFrames = consonantMs.HasValue
                    ? HifiDurationRedistributor.MsToFrames(consonantMs.Value)
                    : 0;
                int start = starts[index];
                int end;
                if (index + 1 < starts.Length) {
                    end = starts[index + 1];
                } else {
                    int byPhoneEnd = MsToFrame(phone.endMs - phraseStartMs);
                    end = Math.Max(byPhoneEnd, phraseFrames);
                }
                end = Math.Clamp(end, start + 1, Math.Max(start + 1, phraseFrames));
                int duration = Math.Max(1, end - start);
                int targetConsonantFrames = Math.Clamp(consonantFrames, 0, Math.Max(0, duration - 1));
                var plan = new HifiPhoneDurationPlan {
                    Index = index,
                    Phoneme = phone.phoneme,
                    OriginalStartFrame = start,
                    OriginalDurationFrames = duration,
                    AdjustedStartFrame = start,
                    AdjustedDurationFrames = duration,
                    ConsonantFrames = targetConsonantFrames,
                    TargetConsonantFrames = targetConsonantFrames,
                    CanDonateVowel = IsVowelDonorCandidate(phone, targetConsonantFrames, duration),
                };
                Log.Information(
                    "HifiPhraseFeatureBuilder timing phoneme={Phoneme} position_ms={PositionMs:F3} preutter_ms={PreutterMs:F3} overlap_ms={OverlapMs:F3} acoustic_start_ms={AcousticStartMs:F3} start_frame={StartFrame} end_frame={EndFrame} duration_frames={DurationFrames} oto_consonant_ms={OtoConsonantMs:F3} effective_consonant_ms={EffectiveConsonantMs:F3} target_consonant_frames={TargetConsonantFrames}",
                    phone.phoneme,
                    phone.positionMs,
                    phone.preutterMs,
                    phone.overlapMs,
                    AcousticStartMs(phone),
                    start,
                    end,
                    duration,
                    phone.oto?.Consonant ?? 0,
                    consonantMs ?? 0,
                    targetConsonantFrames);
                return plan;
            }).ToList();
        }

        static bool IsVowelDonorCandidate(RenderPhone phone, int consonantFrames, int durationFrames) {
            if (durationFrames <= consonantFrames) {
                return false;
            }
            if (string.IsNullOrWhiteSpace(phone.phoneme)) {
                return false;
            }
            string phoneme = phone.phoneme.Trim().ToLowerInvariant();
            if (phoneme == "r" || phoneme == "rest" || phoneme == "sil" || phoneme == "pau" || phoneme == "-" || phoneme.Contains("cl")) {
                return false;
            }
            return true;
        }

        static IReadOnlyList<(int start, int end)> ConsonantRanges(IReadOnlyList<HifiPhoneDurationPlan> plans) {
            return plans.Select(plan => (
                start: plan.AdjustedStartFrame,
                end: plan.AdjustedStartFrame + Math.Max(0, plan.TargetConsonantFrames)))
                .Where(range => range.end > range.start)
                .ToArray();
        }

        static IReadOnlyList<int> StableVowelBoundaryFrames(IReadOnlyList<HifiBoundaryMetadata> boundaries) {
            return boundaries
                .Where(boundary => IsStableVowel(boundary.LeftPhone) && IsStableVowel(boundary.RightPhone))
                .Select(boundary => boundary.Frame)
                .ToArray();
        }

        static IReadOnlyList<int> StableVowelDurationBorrowBoundaryFrames(IReadOnlyList<HifiPhoneDurationPlan> plans) {
            return plans
                .Where(plan => plan.BorrowedFromPreviousFrames > 0 || plan.BorrowedFromNextFrames > 0 || plan.DonatedFrames > 0)
                .Where(plan => IsStableVowel(plan.Phoneme))
                .SelectMany(plan => new[] { plan.AdjustedStartFrame, plan.AdjustedEndFrame })
                .Where(frame => frame >= 0)
                .Distinct()
                .ToArray();
        }

        static bool IsStableVowel(string phoneme) {
            if (string.IsNullOrWhiteSpace(phoneme)) {
                return false;
            }
            string p = phoneme.Trim().ToLowerInvariant();
            if (p == "r" || p == "rest" || p == "sil" || p == "pau" || p == "-" || p == "br" || p.Contains("cl")) {
                return false;
            }
            if (p is "p" or "t" or "k" or "q" or "s" or "sh" or "ch" or "ts" or "f" or "h" or "hh" or "th") {
                return false;
            }
            return p.Any(c => "aeiou".Contains(c)) || p is "m" or "n" or "ng";
        }

        static bool IsFirstOrAfterSilence(IReadOnlyList<RenderPhone> phones, int index) {
            if (index <= 0 || index >= phones.Count) {
                return true;
            }
            var previous = phones[index - 1];
            var current = phones[index];
            if (current.positionMs - previous.endMs > 1.0) {
                return true;
            }
            string p = (previous.phoneme ?? string.Empty).Trim().ToLowerInvariant();
            return p == "r" || p == "rest" || p == "sil" || p == "pau" || p == "-" || p == "br";
        }

        static void FillGaps(float[,] mel, float[] weights) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            for (int t = 0; t < frames; t++) {
                if (weights[t] > 0) {
                    continue;
                }
                int left = t - 1;
                while (left >= 0 && weights[left] <= 0) left--;
                int right = t + 1;
                while (right < frames && weights[right] <= 0) right++;
                for (int m = 0; m < bins; m++) {
                    if (left >= 0 && right < frames) {
                        float alpha = (float)(t - left) / (right - left);
                        mel[m, t] = mel[m, left] + (mel[m, right] - mel[m, left]) * alpha;
                    } else if (left >= 0) {
                        mel[m, t] = mel[m, left];
                    } else if (right < frames) {
                        mel[m, t] = mel[m, right];
                    } else {
                        mel[m, t] = (float)Math.Log(1e-5);
                    }
                }
            }
        }

        static float[] SliceF0(float[] f0, int startFrame, int frames) {
            var result = new float[Math.Max(0, frames)];
            for (int i = 0; i < result.Length; i++) {
                int src = startFrame + i;
                result[i] = src >= 0 && src < f0.Length ? f0[src] : 0f;
            }
            return result;
        }

        static float[,] TrimLeadingMelFrames(float[,] mel, int trimFrames) {
            int bins = mel.GetLength(0);
            int frames = mel.GetLength(1);
            trimFrames = Math.Clamp(trimFrames, 0, Math.Max(0, frames - 2));
            if (trimFrames <= 0) {
                return mel;
            }
            int outFrames = frames - trimFrames;
            var trimmed = new float[bins, outFrames];
            for (int t = 0; t < outFrames; t++) {
                int src = t + trimFrames;
                for (int m = 0; m < bins; m++) {
                    trimmed[m, t] = mel[m, src];
                }
            }
            return trimmed;
        }

        static int MsToFrame(double ms) {
            return (int)Math.Round(ms / HifiF0Builder.FrameMs);
        }

        static HifiPhraseMetadata BuildMetadata(
            RenderPhrase phrase,
            RenderResult layout,
            int frames,
            double phraseStartMs,
            IReadOnlyList<HifiPhoneDurationPlan> durationPlans) {
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
            metadata.Phones.AddRange(phrase.phones.Select((p, i) => {
                var sourceAlignment = CalculateSourceAlignment(p, log: false);
                return new HifiPhoneMetadata {
                    Index = i,
                    Phoneme = p.phoneme,
                    Tone = p.tone,
                    SourceFile = p.oto?.File ?? string.Empty,
                    PositionMs = p.positionMs,
                    DurationMs = p.durationMs,
                    LeadingMs = p.leadingMs,
                    StartFrame = i < durationPlans.Count ? durationPlans[i].AdjustedStartFrame : MsToFrame(AcousticStartMs(p) - phraseStartMs),
                    FrameCount = i < durationPlans.Count ? durationPlans[i].AdjustedDurationFrames : Math.Max(1, MsToFrame(p.durationMs)),
                    SourceSkipOverMs = sourceAlignment.SkipOverMs,
                    SourceStartOffsetFrames = sourceAlignment.StartOffsetFrames,
                };
            }));
            for (int i = 1; i < phrase.phones.Length; i++) {
                var left = phrase.phones[i - 1];
                var right = phrase.phones[i];
                int boundaryFrame = i < durationPlans.Count
                    ? durationPlans[i].AdjustedStartFrame
                    : Math.Clamp(MsToFrame(AcousticStartMs(right) - phraseStartMs), 0, frames - 1);
                metadata.Boundaries.Add(new HifiBoundaryMetadata {
                    Index = i - 1,
                    LeftPhoneIndex = i - 1,
                    RightPhoneIndex = i,
                    LeftPhone = left.phoneme,
                    RightPhone = right.phoneme,
                    PositionMs = phraseStartMs + boundaryFrame * HifiF0Builder.FrameMs,
                    Frame = Math.Clamp(boundaryFrame, 0, frames - 1),
                });
            }
            return metadata;
        }

        static double AcousticStartMs(RenderPhone phone) {
            return phone.positionMs - Math.Max(0, phone.leadingMs);
        }

        static double? EffectiveConsonantMs(RenderPhone phone, double? maxPreutterMs = null) {
            if (phone.oto == null || phone.oto.Consonant <= 0) {
                return null;
            }
            double consonantMs = phone.oto.Consonant;
            double preutterLimit = maxPreutterMs ?? phone.preutterMs;
            if (preutterLimit > 0 && consonantMs > preutterLimit) {
                Log.Information(
                    "HifiPhraseFeatureBuilder consonant_clamped_to_preutter phoneme={Phoneme} oto_consonant_ms={OtoConsonantMs:F3} preutter_ms={PreutterMs:F3}",
                    phone.phoneme,
                    consonantMs,
                    preutterLimit);
                consonantMs = preutterLimit;
            }
            double durationCapMs = Math.Max(HifiF0Builder.FrameMs * 3.0, phone.durationMs * 0.80);
            if (durationCapMs > 0 && consonantMs > durationCapMs) {
                Log.Information(
                    "HifiPhraseFeatureBuilder consonant_clamped_to_duration phoneme={Phoneme} consonant_ms_before={ConsonantBefore:F3} duration_ms={DurationMs:F3} duration_cap_ms={DurationCapMs:F3}",
                    phone.phoneme,
                    consonantMs,
                    phone.durationMs,
                    durationCapMs);
                consonantMs = durationCapMs;
            }
            return consonantMs;
        }

        readonly record struct HifiSourceAlignment(
            double SkipOverMs,
            int SkipSamples,
            int StartOffsetFrames,
            double SourcePreutterMs);

        static void Validate(float[,] mel, float[] f0) {
            foreach (var v in mel) {
                if (float.IsNaN(v) || float.IsInfinity(v)) {
                    throw new InvalidOperationException("phrase_mel contains NaN or Inf.");
                }
            }
            foreach (var v in f0) {
                if (float.IsNaN(v) || float.IsInfinity(v)) {
                    throw new InvalidOperationException("phrase_f0 contains NaN or Inf.");
                }
            }
        }

        static void LogStats(float[,] mel, float[] f0, HifiPhraseMetadata metadata) {
            float melMin = float.PositiveInfinity;
            float melMax = float.NegativeInfinity;
            double melSum = 0;
            foreach (var value in mel) {
                melMin = Math.Min(melMin, value);
                melMax = Math.Max(melMax, value);
                melSum += value;
            }

            var voiced = f0.Where(value => value > 0).ToArray();
            float f0Min = voiced.Length > 0 ? voiced.Min() : 0;
            float f0Max = voiced.Length > 0 ? voiced.Max() : 0;
            double f0Mean = voiced.Length > 0 ? voiced.Average() : 0;

            Log.Information(
                "HifiPhraseFeatureBuilder phrase_mel=[{MelBins},{Frames}] min={MelMin:F4} max={MelMax:F4} mean={MelMean:F4}; phrase_f0=[{F0Frames}] voiced={Voiced} min={F0Min:F2} max={F0Max:F2} mean={F0Mean:F2}; notes={Notes} phones={Phones} boundaries={Boundaries}",
                mel.GetLength(0),
                mel.GetLength(1),
                melMin,
                melMax,
                melSum / mel.Length,
                f0.Length,
                voiced.Length,
                f0Min,
                f0Max,
                f0Mean,
                metadata.Notes.Count,
                metadata.Phones.Count,
                metadata.Boundaries.Count);
        }
    }
}
