using System;
using System.Linq;
using OpenUtau.Api;
using OpenUtau.Core.Neutrino;
using Xunit;

namespace OpenUtau.Core.Test.Neutrino {
    public class NeutrinoChinesePhonemizerTest {
        [Theory]
        [InlineData("zhong", new[] {"j", "o", "N"})]
        [InlineData("xiao", new[] {"sh", "a", "o"})]
        [InlineData("quan", new[] {"ch", "e", "N"})]
        [InlineData("ju", new[] {"j", "u"})]
        [InlineData("xue", new[] {"sh", "e"})]
        [InlineData("qun", new[] {"ch", "u", "N"})]
        [InlineData("xiong", new[] {"sh", "o", "N"})]
        [InlineData("biao", new[] {"by", "a", "o"})]
        [InlineData("miao", new[] {"my", "a", "o"})]
        [InlineData("liang", new[] {"ry", "a", "N"})]
        [InlineData("liu", new[] {"ry", "o", "u"})]
        [InlineData("guo", new[] {"g", "w", "o"})]
        [InlineData("shui", new[] {"sh", "w", "e", "i"})]
        [InlineData("hua", new[] {"h", "w", "a"})]
        [InlineData("yuan", new[] {"y", "e", "N"})]
        [InlineData("yan", new[] {"y", "e", "N"})]
        [InlineData("tian", new[] {"ty", "e", "N"})]
        [InlineData("zhi", new[] {"j", "u"})]
        [InlineData("ci", new[] {"ts", "u"})]
        [InlineData("lv", new[] {"ry", "u"})]
        [InlineData("lu:", new[] {"ry", "u"})]
        [InlineData("l\u00fc", new[] {"ry", "u"})]
        [InlineData("n\u01da", new[] {"ny", "u"})]
        public void MapsPinyinToNeutrinoPhonemes(string lyric, string[] expected) {
            Assert.Equal(expected, NeutrinoChinesePhonemizer.ChineseLyricToNeutrinoPhonemes(lyric));
        }

        [Fact]
        public void MapsHanziThroughPinyin() {
            Assert.Equal(
                new[] {"j", "o", "N"},
                NeutrinoChinesePhonemizer.ChineseLyricToNeutrinoPhonemes("\u4e2d"));
        }

        [Fact]
        public void RomanizesAdjacentHanziWithPhraseContext() {
            var groups = new[] {
                new[] { Note("\u94f6", position: 0) },
                new[] { Note("\u884c", position: 480) },
            };

            new NeutrinoChinesePhonemizer().SetUp(groups, null, null);

            Assert.Equal("yin", groups[0][0].lyric);
            Assert.Equal("hang", groups[1][0].lyric);
        }

        [Fact]
        public void DoesNotCarryHanziContextAcrossAGap() {
            var groups = new[] {
                new[] { Note("\u94f6", position: 0) },
                new[] { Note("\u884c", position: 960) },
            };

            new NeutrinoChinesePhonemizer().SetUp(groups, null, null);

            Assert.Equal("yin", groups[0][0].lyric);
            Assert.Equal("xing", groups[1][0].lyric);
        }

        [Fact]
        public void KeepsDirectNeutrinoPhonemeInput() {
            Assert.Equal(
                new[] {"b", "o"},
                NeutrinoChinesePhonemizer.ChineseLyricToNeutrinoPhonemes("b o"));
            Assert.Equal(
                new[] {"v"},
                NeutrinoChinesePhonemizer.ChineseLyricToNeutrinoPhonemes("v"));
        }

        [Fact]
        public void UsesChineseDefaultTimingForCompoundFinals() {
            var result = Process("xiao", duration: 480);

            Assert.Equal(new[] {"sh", "a", "o"}, result.phonemes.Select(phoneme => phoneme.phoneme));
            Assert.Equal(new[] {0, 67, 346}, result.phonemes.Select(phoneme => phoneme.position));
        }

        [Fact]
        public void KeepsTimingModelPositionsForPanel() {
            var phonemes = new[] {
                new Phonemizer.Phoneme { index = 0, phoneme = "sh", position = 0 },
                new Phonemizer.Phoneme { index = 1, phoneme = "a", position = 81 },
                new Phonemizer.Phoneme { index = 2, phoneme = "o", position = 327 },
            };
            var notes = new[] { Note("xiao", position: 0) };

            var result = new InspectableChinesePhonemizer().PostProcessTimed(phonemes, notes);

            Assert.Same(phonemes, result);
            Assert.Equal(new[] {0, 81, 327}, result.Select(phoneme => phoneme.position));
        }

        [Fact]
        public void KeepsNasalCodaNearTheEnd() {
            var result = Process("zhong", duration: 480);

            Assert.Equal(new[] {"j", "o", "N"}, result.phonemes.Select(phoneme => phoneme.phoneme));
            Assert.Equal(new[] {0, 67, 394}, result.phonemes.Select(phoneme => phoneme.position));
        }

        static Phonemizer.Result Process(string lyric, int duration) {
            var phonemizer = new NeutrinoChinesePhonemizer();
            return phonemizer.Process(
                new[] {
                    new Phonemizer.Note {
                        lyric = lyric,
                        tone = 60,
                        position = 0,
                        duration = duration,
                    },
                },
                null,
                null,
                null,
                null,
                Array.Empty<Phonemizer.Note>());
        }

        static Phonemizer.Note Note(string lyric, int position, int duration = 480) {
            return new Phonemizer.Note {
                lyric = lyric,
                tone = 60,
                position = position,
                duration = duration,
            };
        }

        sealed class InspectableChinesePhonemizer : NeutrinoChinesePhonemizer {
            public Phonemizer.Phoneme[] PostProcessTimed(
                Phonemizer.Phoneme[] phonemes,
                Phonemizer.Note[] notes) {

                return PostProcessTimedPhonemePositions(phonemes, notes);
            }
        }
    }
}
