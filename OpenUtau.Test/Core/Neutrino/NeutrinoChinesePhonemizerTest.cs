using System;
using System.Linq;
using OpenUtau.Api;
using OpenUtau.Core.Neutrino;
using Xunit;

namespace OpenUtau.Core.Test.Neutrino {
    public class NeutrinoChinesePhonemizerTest {
        [Theory]
        [InlineData("zhong", new[] {"j", "o", "N"})]
        [InlineData("xiao", new[] {"sh", "y", "a", "o"})]
        [InlineData("quan", new[] {"ch", "y", "e", "N"})]
        [InlineData("ju", new[] {"j", "y", "u"})]
        [InlineData("xue", new[] {"sh", "y", "e"})]
        [InlineData("qun", new[] {"ch", "y", "u", "N"})]
        [InlineData("xiong", new[] {"sh", "y", "o", "N"})]
        [InlineData("liang", new[] {"r", "y", "a", "N"})]
        [InlineData("liu", new[] {"r", "y", "o", "u"})]
        [InlineData("guo", new[] {"g", "w", "o"})]
        [InlineData("shui", new[] {"sh", "w", "e", "i"})]
        [InlineData("hua", new[] {"h", "w", "a"})]
        [InlineData("yuan", new[] {"y", "e", "N"})]
        [InlineData("yan", new[] {"y", "e", "N"})]
        [InlineData("tian", new[] {"t", "y", "e", "N"})]
        [InlineData("zhi", new[] {"j", "u"})]
        [InlineData("ci", new[] {"ts", "u"})]
        [InlineData("lv", new[] {"r", "y", "u"})]
        [InlineData("lu:", new[] {"r", "y", "u"})]
        [InlineData("l\u00fc", new[] {"r", "y", "u"})]
        [InlineData("n\u01da", new[] {"n", "y", "u"})]
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

            Assert.Equal(new[] {"sh", "y", "a", "o"}, result.phonemes.Select(phoneme => phoneme.phoneme));
            Assert.Equal(new[] {0, 48, 96, 346}, result.phonemes.Select(phoneme => phoneme.position));
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
    }
}
