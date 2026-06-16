using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Api;
using OpenUtau.Core.Ustx;
using Serilog;

namespace OpenUtau.Core.Neutrino {
    /// <summary>
    /// NEUTRINO phonemizer: converts lyrics to model phonemes and uses t.bin
    /// to place default phoneme boundaries when the singer is available.
    /// </summary>
    [Phonemizer("NEUTRINO Phonemizer", "NEUTRINO", language: "JA")]
    public class NeutrinoPhonemizer : Phonemizer {
        const double defaultConsonantMs = 60;
        const int minPhonemeTicks = 10;
        const int sampleRate = 48000;
        const int hopSize = 480;

        NeutrinoSinger neutrinoSinger;
        readonly Dictionary<int, Phoneme[]> timedPhonemes = new Dictionary<int, Phoneme[]>();

        public override void SetSinger(USinger singer) {
            neutrinoSinger = singer as NeutrinoSinger;
        }

        public override void SetUp(Note[][] notes, UProject project, UTrack track) {
            timedPhonemes.Clear();
            if (neutrinoSinger == null || notes == null || notes.Length == 0 || timeAxis == null) {
                return;
            }
            try {
                neutrinoSinger.EnsureTimingSession();
                foreach (var phrase in SplitPhrases(notes)) {
                    BuildTimedPhonemes(phrase);
                }
            } catch (Exception e) {
                timedPhonemes.Clear();
                Log.Warning(e, "Failed to run NEUTRINO timing model for phoneme panel; using estimated phoneme positions.");
            }
        }

        public override Result Process(Note[] notes, Note? prev, Note? next,
            Note? prevNeighbour, Note? nextNeighbour, Note[] prevs) {

            if (timedPhonemes.TryGetValue(notes[0].position, out var timed)) {
                return new Result {
                    phonemes = timed.ToArray(),
                };
            }

            var lyric = string.IsNullOrWhiteSpace(notes[0].phoneticHint)
                ? notes[0].lyric ?? "R"
                : notes[0].phoneticHint;
            var phonemes = NeutrinoPhoneme.KanaToPhonemes(lyric);
            var positions = DistributePhonemes(phonemes, notes);
            return new Result {
                phonemes = phonemes
                    .Select((phoneme, index) => new Phoneme {
                        index = index,
                        phoneme = phoneme,
                        position = positions[index],
                    })
                    .ToArray(),
            };
        }

        List<List<Note[]>> SplitPhrases(Note[][] noteGroups) {
            var phrases = new List<List<Note[]>>();
            List<Note[]> phrase = null;
            int previousEnd = int.MinValue;
            foreach (var group in noteGroups.Where(group => group.Length > 0)) {
                int start = group[0].position;
                int end = group[^1].position + group[^1].duration;
                if (phrase == null || start > previousEnd) {
                    phrase = new List<Note[]>();
                    phrases.Add(phrase);
                }
                phrase.Add(group);
                previousEnd = Math.Max(previousEnd, end);
            }
            return phrases;
        }

        void BuildTimedPhonemes(List<Note[]> noteGroups) {
            var phonemeIds = new List<long>();
            var scorePitchesHz = new List<float>();
            var scoreDurations = new List<float>();
            var phonePositions = new List<long>();
            var phoneRefs = new List<TimedPhoneRef>();
            var groupedPhonemes = noteGroups.ToDictionary(group => group[0].position, _ => new List<Phoneme>());

            foreach (var group in noteGroups) {
                var lyric = string.IsNullOrWhiteSpace(group[0].phoneticHint)
                    ? group[0].lyric ?? "R"
                    : group[0].phoneticHint;
                var phonemes = NeutrinoPhoneme.KanaToPhonemes(lyric);
                float notePitchHz = phonemes.All(p => NeutrinoPhoneme.GetPhonemeId(p) == NeutrinoPhoneme.PAU)
                    ? 0
                    : (float)NeutrinoConfig.MidiToFreq(group[0].tone);
                float durationSec = Math.Max(0.001f, (float)(GetGroupDurationMs(group) / 1000.0));

                for (int i = 0; i < phonemes.Length; i++) {
                    int id = NeutrinoPhoneme.GetPhonemeId(phonemes[i]);
                    phonemeIds.Add(id);
                    scorePitchesHz.Add(id == NeutrinoPhoneme.PAU ? 0 : notePitchHz);
                    scoreDurations.Add(durationSec);
                    phonePositions.Add(i);
                    phoneRefs.Add(new TimedPhoneRef {
                        groupPosition = group[0].position,
                        index = i,
                        phoneme = phonemes[i],
                    });
                }
            }

            int numPhones = phonemeIds.Count;
            if (numPhones == 0) {
                return;
            }

            var timingInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("electron",
                    new DenseTensor<long>(phonemeIds.ToArray(), new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("muon",
                    new DenseTensor<float>(scorePitchesHz.ToArray(), new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("tau",
                    new DenseTensor<float>(scoreDurations.ToArray(), new[] { 1, numPhones })),
                NamedOnnxValue.CreateFromTensor("selectron",
                    new DenseTensor<long>(phonePositions.ToArray(), new[] { 1, numPhones })),
            };
            var boundaryShifts = neutrinoSinger.RunTiming(timingInputs);
            var baseBoundaries = BuildBaseBoundaryTimes(scoreDurations.ToArray(), phonePositions.ToArray());
            var boundaries = ApplyTimingBoundaryShifts(baseBoundaries, boundaryShifts);
            double phraseStartMs = timeAxis.TickPosToMsPos(noteGroups[0][0].position);

            for (int i = 0; i < phoneRefs.Count; i++) {
                var phoneRef = phoneRefs[i];
                double positionMs = phraseStartMs + boundaries[i] * 1000.0;
                int position = timeAxis.MsPosToTickPos(positionMs) - phoneRef.groupPosition;
                groupedPhonemes[phoneRef.groupPosition].Add(new Phoneme {
                    index = phoneRef.index,
                    phoneme = phoneRef.phoneme,
                    position = position,
                });
            }

            foreach (var pair in groupedPhonemes) {
                if (pair.Value.Count > 0) {
                    timedPhonemes[pair.Key] = pair.Value.ToArray();
                }
            }
        }

        double GetGroupDurationMs(Note[] notes) {
            double startMs = timeAxis.TickPosToMsPos(notes[0].position);
            var last = notes[^1];
            double endMs = timeAxis.TickPosToMsPos(last.position + last.duration);
            return Math.Max(1, endMs - startMs);
        }

        double[] BuildBaseBoundaryTimes(float[] scoreDurations, long[] phonePositions) {
            int numPhones = scoreDurations.Length;
            var boundaries = new double[numPhones + 1];
            double time = 0;
            for (int i = 0; i < numPhones; i++) {
                boundaries[i] = time;
                long nextPosition = i + 1 < numPhones ? phonePositions[i + 1] : -1;
                if (i == numPhones - 1 || nextPosition <= phonePositions[i]) {
                    time += scoreDurations[i];
                }
            }
            boundaries[numPhones] = time;
            return boundaries;
        }

        double[] ApplyTimingBoundaryShifts(double[] baseBoundaries, float[] boundaryShifts) {
            var boundaries = (double[])baseBoundaries.Clone();
            double frameSec = (double)hopSize / sampleRate;
            for (int i = 1; i < boundaries.Length - 1; i++) {
                double shift = i < boundaryShifts.Length ? boundaryShifts[i] : 0;
                double shifted = baseBoundaries[i] + shift;
                boundaries[i] = Math.Round(Math.Max(shifted, boundaries[i - 1] + frameSec) * 1000.0) / 1000.0;
            }
            for (int i = 1; i < boundaries.Length; i++) {
                if (boundaries[i] <= boundaries[i - 1]) {
                    boundaries[i] = Math.Round((boundaries[i - 1] + frameSec) * 1000.0) / 1000.0;
                }
            }
            return boundaries;
        }

        struct TimedPhoneRef {
            public int groupPosition;
            public int index;
            public string phoneme;
        }

        int[] DistributePhonemes(string[] phonemes, Note[] notes) {
            if (phonemes.Length == 0) {
                return Array.Empty<int>();
            }
            if (phonemes.Length == 1) {
                return new[] { 0 };
            }

            int noteStart = notes[0].position;
            int noteEnd = notes.Last().position + notes.Last().duration;
            int totalDuration = Math.Max(minPhonemeTicks, noteEnd - noteStart);
            int lastStart = Math.Max(0, totalDuration - minPhonemeTicks);
            int consonantTicks = Math.Clamp(DefaultConsonantTicks(noteStart), minPhonemeTicks, lastStart);
            int firstVowel = Array.FindIndex(phonemes, NeutrinoPhoneme.IsVowelPhoneme);

            var positions = new int[phonemes.Length];
            if (firstVowel > 0) {
                for (int i = 0; i < phonemes.Length; i++) {
                    if (i <= firstVowel) {
                        positions[i] = (int)Math.Round((double)consonantTicks * i / firstVowel);
                    } else {
                        int restCount = phonemes.Length - firstVowel;
                        positions[i] = consonantTicks
                            + (int)Math.Round((double)(lastStart - consonantTicks) * (i - firstVowel) / restCount);
                    }
                }
            } else {
                for (int i = 0; i < phonemes.Length; i++) {
                    positions[i] = (int)Math.Round((double)lastStart * i / (phonemes.Length - 1));
                }
            }

            positions[0] = 0;
            for (int i = 1; i < positions.Length; i++) {
                int maxPosition = Math.Max(positions[i - 1] + minPhonemeTicks,
                    totalDuration - minPhonemeTicks * (positions.Length - i));
                positions[i] = Math.Min(Math.Max(positions[i], positions[i - 1] + minPhonemeTicks), maxPosition);
            }
            return positions;
        }

        int DefaultConsonantTicks(int notePosition) {
            if (timeAxis == null) {
                return 60;
            }
            double noteMs = timeAxis.TickPosToMsPos(notePosition);
            return Math.Max(minPhonemeTicks,
                timeAxis.TicksBetweenMsPos(noteMs, noteMs + defaultConsonantMs));
        }
    }
}
