using System.Collections.Generic;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiPhraseFeatures {
        public required float[,] Mel { get; init; }
        public required float[] F0 { get; init; }
        public required HifiPhraseMetadata Metadata { get; init; }

        public int MelBins => Mel.GetLength(0);
        public int Frames => Mel.GetLength(1);
    }

    public sealed class HifiPhraseMetadata {
        public int SampleRate { get; init; }
        public int HopSize { get; init; }
        public double FrameMs { get; init; }
        public double PhraseStartMs { get; init; }
        public double LeadingMs { get; init; }
        public double EstimatedLengthMs { get; init; }
        public List<HifiPhoneMetadata> Phones { get; init; } = new List<HifiPhoneMetadata>();
        public List<HifiNoteMetadata> Notes { get; init; } = new List<HifiNoteMetadata>();
        public List<HifiBoundaryMetadata> Boundaries { get; init; } = new List<HifiBoundaryMetadata>();
        public List<HifiPhoneFeatureDiagnostic> PhoneDiagnostics { get; init; } = new List<HifiPhoneFeatureDiagnostic>();
    }

    public sealed class HifiPhoneMetadata {
        public int Index { get; init; }
        public string Phoneme { get; init; } = string.Empty;
        public int Tone { get; init; }
        public string SourceFile { get; init; } = string.Empty;
        public double PositionMs { get; init; }
        public double DurationMs { get; init; }
        public double LeadingMs { get; init; }
        public int StartFrame { get; init; }
        public int FrameCount { get; init; }
        public double SourceSkipOverMs { get; init; }
        public int SourceStartOffsetFrames { get; init; }
        public HifiPhoneParameterMetadata Parameters { get; init; } = new HifiPhoneParameterMetadata();
    }

    public sealed class HifiPhoneParameterMetadata {
        public double Gender { get; init; }
        public double Breathiness { get; init; }
        public double Tension { get; init; }
        public double Voicing { get; init; } = 100;
        public double GenderKeyShiftSemitones { get; init; }
        public double BreathNoiseGain { get; init; } = 1;
        public double VoicingGain { get; init; } = 1;
        public bool HnsepRequested { get; init; }
        public bool HnsepApplied { get; init; }
        public string HnsepReason { get; init; } = string.Empty;
    }

    public sealed class HifiNoteMetadata {
        public int Index { get; init; }
        public string Lyric { get; init; } = string.Empty;
        public int Tone { get; init; }
        public float AdjustedTone { get; init; }
        public double PositionMs { get; init; }
        public double DurationMs { get; init; }
    }

    public sealed class HifiBoundaryMetadata {
        public int Index { get; init; }
        public int LeftPhoneIndex { get; init; }
        public int RightPhoneIndex { get; init; }
        public string LeftPhone { get; init; } = string.Empty;
        public string RightPhone { get; init; } = string.Empty;
        public double PositionMs { get; init; }
        public int Frame { get; init; }
        public string TransitionType { get; init; } = "phone";
    }

    public sealed class HifiPhoneFeatureDiagnostic {
        public int Index { get; init; }
        public string Phoneme { get; init; } = string.Empty;
        public int StartFrame { get; init; }
        public int FrameCount { get; init; }
        public double SourceOnsetPeak { get; init; }
        public double SourceOnsetRms { get; init; }
        public double SourceOnsetDc { get; init; }
        public double SourceOnsetMaxJump { get; init; }
        public double SourceMelOnsetMean { get; init; }
        public double SourceMelAfterMean { get; init; }
        public double SourceMelOnsetDelta { get; init; }
        public double StretchedMelOnsetMean { get; init; }
        public double StretchedMelAfterMean { get; init; }
        public double StretchedMelOnsetDelta { get; init; }
        public float TargetF0AtStart { get; init; }
        public float TargetF0AfterStart { get; init; }
        public double TargetF0OnsetJumpCents { get; init; }
        public string LeadingSanitizerReason { get; init; } = string.Empty;
    }

    public sealed class HifiMelAssemblyReport {
        public List<HifiPhoneMetadata> Phones { get; } = new List<HifiPhoneMetadata>();
        public List<HifiBoundaryMetadata> Boundaries { get; } = new List<HifiBoundaryMetadata>();
        public List<(int Start, int End)> ConsonantFrameRanges { get; } = new List<(int Start, int End)>();
        public List<HifiPhoneFeatureDiagnostic> PhoneDiagnostics { get; } = new List<HifiPhoneFeatureDiagnostic>();
    }
}
