using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiClickDiagnosticReport {
        public int SampleRate { get; init; }
        public int HopSize { get; init; }
        public int SampleCount { get; init; }
        public int SuspectPhones { get; init; }
        public int SuspectNotes { get; init; }
        public List<HifiPhoneClickDiagnostic> Phones { get; init; } = new List<HifiPhoneClickDiagnostic>();
        public List<HifiNoteClickDiagnostic> Notes { get; init; } = new List<HifiNoteClickDiagnostic>();
    }

    public sealed class HifiPhoneClickDiagnostic {
        public int Index { get; init; }
        public string Phoneme { get; init; } = string.Empty;
        public int StartFrame { get; init; }
        public int StartSample { get; init; }
        public double WavJump { get; init; }
        public double WavLocalRms { get; init; }
        public double WavPeak { get; init; }
        public double MelBoundaryDelta { get; init; }
        public double MelOnsetEnergy { get; init; }
        public float F0Before { get; init; }
        public float F0At { get; init; }
        public float F0After { get; init; }
        public bool F0VoicedOnset { get; init; }
        public bool Suspect { get; init; }
        public string SuspectReason { get; init; } = string.Empty;
        public HifiPhoneFeatureDiagnostic? Feature { get; init; }
    }

    public sealed class HifiNoteClickDiagnostic {
        public int Index { get; init; }
        public string Lyric { get; init; } = string.Empty;
        public int StartFrame { get; init; }
        public int StartSample { get; init; }
        public double WavJump { get; init; }
        public double WavLocalRms { get; init; }
        public double WavPeak { get; init; }
        public double MelBoundaryDelta { get; init; }
        public bool Suspect { get; init; }
        public string SuspectReason { get; init; } = string.Empty;
    }

    public static class HifiClickDiagnostic {
        public static HifiPhoneFeatureDiagnostic BuildPhoneFeatureDiagnostic(
            int index,
            string phoneme,
            int startFrame,
            int frameCount,
            float[] source,
            float[,] sourceMel,
            float[,] stretchedMel,
            float[] targetF0,
            string leadingSanitizerReason) {
            var sourceStats = SourceOnsetStats(source, milliseconds: 5.0);
            var sourceMelStats = MelOnsetStats(sourceMel);
            var stretchedStats = MelOnsetStats(stretchedMel);
            return new HifiPhoneFeatureDiagnostic {
                Index = index,
                Phoneme = phoneme,
                StartFrame = startFrame,
                FrameCount = frameCount,
                SourceOnsetPeak = sourceStats.Peak,
                SourceOnsetRms = sourceStats.Rms,
                SourceOnsetDc = sourceStats.Dc,
                SourceOnsetMaxJump = sourceStats.MaxJump,
                SourceMelOnsetMean = sourceMelStats.OnsetMean,
                SourceMelAfterMean = sourceMelStats.AfterMean,
                SourceMelOnsetDelta = sourceMelStats.Delta,
                StretchedMelOnsetMean = stretchedStats.OnsetMean,
                StretchedMelAfterMean = stretchedStats.AfterMean,
                StretchedMelOnsetDelta = stretchedStats.Delta,
                TargetF0AtStart = targetF0.Length > 0 ? targetF0[0] : 0,
                TargetF0AfterStart = targetF0.Length > 1 ? targetF0[Math.Min(targetF0.Length - 1, 2)] : 0,
                TargetF0OnsetJumpCents = targetF0.Length > 1 ? F0Cents(targetF0[0], targetF0[Math.Min(targetF0.Length - 1, 2)]) : 0,
                LeadingSanitizerReason = leadingSanitizerReason,
            };
        }

        public static string Export(string key, HifiPhraseFeatures features, float[] samples) {
            string dir = Path.Combine(PathManager.Inst.CachePath, "hifi-debug", key);
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, "click_diagnostic.json");
            var report = Analyze(features, samples);
            File.WriteAllText(path, JsonConvert.SerializeObject(report, Formatting.Indented));
            Log.Information(
                "HifiClickDiagnostic exported path={Path} suspect_phones={SuspectPhones} suspect_notes={SuspectNotes}",
                path,
                report.SuspectPhones,
                report.SuspectNotes);
            return path;
        }

        public static HifiClickDiagnosticReport Analyze(HifiPhraseFeatures features, float[] samples) {
            var metadata = features.Metadata;
            var phoneFeatureMap = metadata.PhoneDiagnostics.ToDictionary(d => d.Index, d => d);
            var phoneReports = new List<HifiPhoneClickDiagnostic>();
            foreach (var phone in metadata.Phones) {
                int startSample = phone.StartFrame * metadata.HopSize;
                var wav = WavBoundaryStats(samples, startSample, metadata.SampleRate);
                double melDelta = MelBoundaryDelta(features.Mel, phone.StartFrame);
                double melEnergy = MeanMelFrame(features.Mel, phone.StartFrame);
                float f0Before = F0At(features.F0, phone.StartFrame - 1);
                float f0At = F0At(features.F0, phone.StartFrame);
                float f0After = F0At(features.F0, phone.StartFrame + 1);
                bool f0VoicedOnset = f0Before <= 0 && (f0At > 0 || f0After > 0);
                phoneFeatureMap.TryGetValue(phone.Index, out var feature);
                string reason = SuspectReason(wav.Jump, wav.LocalRms, melDelta, f0VoicedOnset, feature);
                var report = new HifiPhoneClickDiagnostic {
                    Index = phone.Index,
                    Phoneme = phone.Phoneme,
                    StartFrame = phone.StartFrame,
                    StartSample = startSample,
                    WavJump = wav.Jump,
                    WavLocalRms = wav.LocalRms,
                    WavPeak = wav.Peak,
                    MelBoundaryDelta = melDelta,
                    MelOnsetEnergy = melEnergy,
                    F0Before = f0Before,
                    F0At = f0At,
                    F0After = f0After,
                    F0VoicedOnset = f0VoicedOnset,
                    Suspect = !string.IsNullOrEmpty(reason),
                    SuspectReason = reason,
                    Feature = feature,
                };
                phoneReports.Add(report);
                if (report.Suspect) {
                    Log.Warning(
                        "HifiClickDiagnostic phone_suspect index={Index} phoneme={Phoneme} start_frame={StartFrame} wav_jump={WavJump:F6} local_rms={LocalRms:F6} mel_delta={MelDelta:F4} f0_before={F0Before:F2} f0_at={F0At:F2} reason={Reason}",
                        report.Index,
                        report.Phoneme,
                        report.StartFrame,
                        report.WavJump,
                        report.WavLocalRms,
                        report.MelBoundaryDelta,
                        report.F0Before,
                        report.F0At,
                        report.SuspectReason);
                }
            }

            var noteReports = metadata.Notes.Select(note => {
                int frame = MsToFrame(note.PositionMs - metadata.PhraseStartMs, metadata.FrameMs);
                int sample = frame * metadata.HopSize;
                var wav = WavBoundaryStats(samples, sample, metadata.SampleRate);
                double melDelta = MelBoundaryDelta(features.Mel, frame);
                string reason = SuspectReason(wav.Jump, wav.LocalRms, melDelta, f0VoicedOnset: false, feature: null);
                return new HifiNoteClickDiagnostic {
                    Index = note.Index,
                    Lyric = note.Lyric,
                    StartFrame = frame,
                    StartSample = sample,
                    WavJump = wav.Jump,
                    WavLocalRms = wav.LocalRms,
                    WavPeak = wav.Peak,
                    MelBoundaryDelta = melDelta,
                    Suspect = !string.IsNullOrEmpty(reason),
                    SuspectReason = reason,
                };
            }).ToList();
            foreach (var note in noteReports.Where(n => n.Suspect)) {
                Log.Warning(
                    "HifiClickDiagnostic note_suspect index={Index} lyric={Lyric} start_frame={StartFrame} wav_jump={WavJump:F6} local_rms={LocalRms:F6} mel_delta={MelDelta:F4} reason={Reason}",
                    note.Index,
                    note.Lyric,
                    note.StartFrame,
                    note.WavJump,
                    note.WavLocalRms,
                    note.MelBoundaryDelta,
                    note.SuspectReason);
            }

            return new HifiClickDiagnosticReport {
                SampleRate = metadata.SampleRate,
                HopSize = metadata.HopSize,
                SampleCount = samples.Length,
                SuspectPhones = phoneReports.Count(p => p.Suspect),
                SuspectNotes = noteReports.Count(n => n.Suspect),
                Phones = phoneReports,
                Notes = noteReports,
            };
        }

        static string SuspectReason(
            double wavJump,
            double localRms,
            double melDelta,
            bool f0VoicedOnset,
            HifiPhoneFeatureDiagnostic? feature) {
            var reasons = new List<string>();
            if (localRms > 1e-5 && wavJump > Math.Max(0.02, localRms * 0.9)) {
                reasons.Add("wav_jump");
            }
            if (melDelta > HifiNeuralConfig.ClickDiagnosticMelDeltaThreshold) {
                reasons.Add("phrase_mel_boundary_delta");
            }
            if (f0VoicedOnset) {
                reasons.Add("f0_voiced_onset");
            }
            if (feature != null) {
                if (feature.SourceOnsetMaxJump > Math.Max(0.04, feature.SourceOnsetRms * 1.4)) {
                    reasons.Add("source_onset_jump");
                }
                if (Math.Abs(feature.SourceOnsetDc) > 0.03) {
                    reasons.Add("source_onset_dc");
                }
                if (feature.StretchedMelOnsetDelta > HifiNeuralConfig.ClickDiagnosticMelDeltaThreshold) {
                    reasons.Add("stretched_mel_onset_delta");
                }
            }
            return string.Join(",", reasons.Distinct());
        }

        static (double Peak, double Rms, double Dc, double MaxJump) SourceOnsetStats(float[] samples, double milliseconds) {
            int count = Math.Clamp((int)Math.Round(milliseconds * HifiMelExtractor.SampleRate / 1000.0), 0, samples.Length);
            if (count <= 0) {
                return (0, 0, 0, 0);
            }
            double peak = 0;
            double sum = 0;
            double sumSq = 0;
            double maxJump = 0;
            for (int i = 0; i < count; i++) {
                double value = samples[i];
                peak = Math.Max(peak, Math.Abs(value));
                sum += value;
                sumSq += value * value;
                if (i > 0) {
                    maxJump = Math.Max(maxJump, Math.Abs(samples[i] - samples[i - 1]));
                }
            }
            return (peak, Math.Sqrt(sumSq / count), sum / count, maxJump);
        }

        static (double OnsetMean, double AfterMean, double Delta) MelOnsetStats(float[,] mel) {
            int frames = mel.GetLength(1);
            if (frames <= 0) {
                return (0, 0, 0);
            }
            double onset = MeanMelRange(mel, 0, Math.Min(2, frames));
            double after = MeanMelRange(mel, Math.Min(2, frames), Math.Min(6, frames));
            if (double.IsNaN(after)) {
                after = onset;
            }
            return (onset, after, Math.Abs(after - onset));
        }

        static (double Jump, double LocalRms, double Peak) WavBoundaryStats(float[] samples, int center, int sampleRate) {
            if (samples.Length == 0 || center <= 0 || center >= samples.Length) {
                return (0, 0, 0);
            }
            int halfWindow = Math.Clamp((int)Math.Round(sampleRate * 0.003), 16, 256);
            int start = Math.Clamp(center - halfWindow, 0, samples.Length);
            int end = Math.Clamp(center + halfWindow, start, samples.Length);
            double peak = 0;
            double sumSq = 0;
            int count = 0;
            for (int i = start; i < end; i++) {
                double value = samples[i];
                peak = Math.Max(peak, Math.Abs(value));
                sumSq += value * value;
                count++;
            }
            return (
                Math.Abs(samples[center] - samples[center - 1]),
                count > 0 ? Math.Sqrt(sumSq / count) : 0,
                peak);
        }

        static double MelBoundaryDelta(float[,] mel, int frame) {
            int frames = mel.GetLength(1);
            if (frame <= 0 || frame >= frames) {
                return 0;
            }
            int bins = mel.GetLength(0);
            double sum = 0;
            for (int m = 0; m < bins; m++) {
                sum += Math.Abs(mel[m, frame] - mel[m, frame - 1]);
            }
            return sum / Math.Max(1, bins);
        }

        static double MeanMelFrame(float[,] mel, int frame) {
            return MeanMelRange(mel, frame, frame + 1);
        }

        static double MeanMelRange(float[,] mel, int start, int end) {
            int frames = mel.GetLength(1);
            int bins = mel.GetLength(0);
            start = Math.Clamp(start, 0, frames);
            end = Math.Clamp(end, start, frames);
            if (end <= start) {
                return double.NaN;
            }
            double sum = 0;
            int count = 0;
            for (int t = start; t < end; t++) {
                for (int m = 0; m < bins; m++) {
                    sum += mel[m, t];
                    count++;
                }
            }
            return count > 0 ? sum / count : 0;
        }

        static float F0At(float[] f0, int frame) {
            return frame >= 0 && frame < f0.Length ? f0[frame] : 0;
        }

        static int MsToFrame(double ms, double frameMs) {
            return Math.Clamp((int)Math.Round(ms / frameMs), 0, int.MaxValue);
        }

        static double F0Cents(float left, float right) {
            if (left <= 0 || right <= 0 || float.IsNaN(left) || float.IsNaN(right) || float.IsInfinity(left) || float.IsInfinity(right)) {
                return 0;
            }
            return Math.Abs(1200.0 * Math.Log(right / left, 2.0));
        }
    }
}
