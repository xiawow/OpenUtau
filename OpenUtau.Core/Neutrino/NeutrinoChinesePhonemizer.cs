using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using OpenUtau.Api;
using Pinyin;

namespace OpenUtau.Core.Neutrino {
    [Phonemizer("NEUTRINO Chinese Phonemizer", "NEUTRINO ZH", language: "ZH")]
    public class NeutrinoChinesePhonemizer : NeutrinoPhonemizer {
        const int minChinesePhonemeTicks = 10;
        const int maxPrefixTicks = 96;
        const int maxTailTicks = 180;
        const int maxNasalTailTicks = 144;

        static readonly Dictionary<string, string[]> WholeSyllables =
            new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase) {
                {"zhi", new[] {"j", "u"}},
                {"chi", new[] {"ch", "u"}},
                {"shi", new[] {"sh", "u"}},
                {"ri", new[] {"r", "u"}},
                {"zi", new[] {"z", "u"}},
                {"ci", new[] {"ts", "u"}},
                {"si", new[] {"s", "u"}},
                {"yi", new[] {"i"}},
                {"yin", new[] {"i", "N"}},
                {"ying", new[] {"i", "N"}},
                {"yan", new[] {"y", "e", "N"}},
                {"wu", new[] {"u"}},
                {"yu", new[] {"y", "u"}},
                {"yue", new[] {"y", "e"}},
                {"yuan", new[] {"y", "e", "N"}},
                {"yun", new[] {"y", "u", "N"}},
                {"ng", new[] {"N"}},
            };

        static readonly Dictionary<string, string[]> Initials =
            new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase) {
                {"b", new[] {"b"}},
                {"p", new[] {"p"}},
                {"m", new[] {"m"}},
                {"f", new[] {"f"}},
                {"d", new[] {"d"}},
                {"t", new[] {"t"}},
                {"n", new[] {"n"}},
                {"l", new[] {"r"}},
                {"g", new[] {"g"}},
                {"k", new[] {"k"}},
                {"h", new[] {"h"}},
                {"j", new[] {"j"}},
                {"q", new[] {"ch"}},
                {"x", new[] {"sh"}},
                {"zh", new[] {"j"}},
                {"ch", new[] {"ch"}},
                {"sh", new[] {"sh"}},
                {"r", new[] {"r"}},
                {"z", new[] {"z"}},
                {"c", new[] {"ts"}},
                {"s", new[] {"s"}},
                {"y", new[] {"y"}},
                {"w", new[] {"w"}},
            };

        static readonly Dictionary<string, string[]> Finals =
            new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase) {
                {"a", new[] {"a"}},
                {"ai", new[] {"a", "i"}},
                {"an", new[] {"a", "N"}},
                {"ang", new[] {"a", "N"}},
                {"ao", new[] {"a", "o"}},
                {"o", new[] {"o"}},
                {"ong", new[] {"o", "N"}},
                {"ou", new[] {"o", "u"}},
                {"e", new[] {"e"}},
                {"ei", new[] {"e", "i"}},
                {"en", new[] {"e", "N"}},
                {"eng", new[] {"e", "N"}},
                {"er", new[] {"e", "r"}},
                {"i", new[] {"i"}},
                {"ia", new[] {"y", "a"}},
                {"ian", new[] {"y", "e", "N"}},
                {"iang", new[] {"y", "a", "N"}},
                {"iao", new[] {"y", "a", "o"}},
                {"ie", new[] {"y", "e"}},
                {"in", new[] {"i", "N"}},
                {"ing", new[] {"i", "N"}},
                {"iong", new[] {"y", "o", "N"}},
                {"iu", new[] {"y", "o", "u"}},
                {"u", new[] {"u"}},
                {"ua", new[] {"w", "a"}},
                {"uai", new[] {"w", "a", "i"}},
                {"uan", new[] {"w", "a", "N"}},
                {"uang", new[] {"w", "a", "N"}},
                {"ue", new[] {"w", "e"}},
                {"ui", new[] {"w", "e", "i"}},
                {"un", new[] {"w", "e", "N"}},
                {"uo", new[] {"w", "o"}},
                {"v", new[] {"y", "u"}},
                {"ve", new[] {"y", "e"}},
                {"van", new[] {"y", "e", "N"}},
                {"vn", new[] {"y", "u", "N"}},
            };

        static readonly string[] InitialKeys = {
            "zh", "ch", "sh",
            "b", "p", "m", "f",
            "d", "t", "n", "l",
            "g", "k", "h",
            "j", "q", "x",
            "r", "z", "c", "s",
            "y", "w",
        };

        static readonly char[] TokenSeparators = {
            ' ', '\t', '\r', '\n',
            '\'', '\u2019', '-', '_',
            ',', '\uff0c', '.', '\u3002', ';', '\uff1b',
            '/', '\\', '|', '\u3001',
        };

        static readonly HashSet<string> NeutrinoPhonemeNames =
            new HashSet<string>(NeutrinoPhoneme.AllPhonemes, StringComparer.OrdinalIgnoreCase);

        protected override string[] LyricToPhonemes(string lyric) {
            return ChineseLyricToNeutrinoPhonemes(lyric);
        }

        protected override Phoneme[] PostProcessPhonemePositions(Phoneme[] phonemes, Note[] notes) {
            if (phonemes.Length <= 1 || notes == null || notes.Length == 0) {
                return phonemes;
            }

            var adjusted = phonemes
                .Select(phoneme => new Phoneme {
                    index = phoneme.index,
                    phoneme = phoneme.phoneme,
                    position = phoneme.position,
                })
                .ToArray();
            var positions = DistributeChinesePhonemes(
                adjusted.Select(phoneme => phoneme.phoneme).ToArray(),
                notes);
            if (positions.Length != adjusted.Length) {
                return phonemes;
            }
            for (int i = 0; i < adjusted.Length; i++) {
                adjusted[i].position = positions[i];
            }
            return adjusted;
        }

        internal static string[] ChineseLyricToNeutrinoPhonemes(string lyric) {
            if (IsRestLyric(lyric)) {
                return NeutrinoPhoneme.KanaToPhonemes(lyric);
            }

            var romanized = RomanizeHanziLyric(lyric);
            var tokens = SplitTokens(romanized).ToArray();
            if (tokens.Length == 0) {
                return NeutrinoPhoneme.KanaToPhonemes(lyric);
            }
            if (tokens.Length == 1 && IsNeutrinoPhonemeToken(tokens[0])) {
                return NeutrinoPhoneme.KanaToPhonemes(romanized);
            }
            if (tokens.Length > 1 && tokens.All(IsNeutrinoPhonemeToken)) {
                return NeutrinoPhoneme.KanaToPhonemes(romanized);
            }

            var phonemes = new List<string>();
            foreach (var token in tokens) {
                var mapped = PinyinSyllableToPhonemes(token);
                if (mapped.Length > 0) {
                    phonemes.AddRange(mapped);
                    continue;
                }
                phonemes.AddRange(NeutrinoPhoneme.KanaToPhonemes(token));
            }
            return phonemes.Count > 0 ? phonemes.ToArray() : NeutrinoPhoneme.KanaToPhonemes(lyric);
        }

        static string[] PinyinSyllableToPhonemes(string syllable) {
            syllable = NormalizePinyin(syllable);
            if (string.IsNullOrEmpty(syllable)) {
                return Array.Empty<string>();
            }
            if (WholeSyllables.TryGetValue(syllable, out var whole)) {
                return whole;
            }

            string? initial = null;
            string final = syllable;
            foreach (var key in InitialKeys) {
                if (syllable.StartsWith(key, StringComparison.OrdinalIgnoreCase)) {
                    initial = key;
                    final = syllable.Substring(key.Length);
                    break;
                }
            }

            if ((initial == "j" || initial == "q" || initial == "x")
                && final.StartsWith("u", StringComparison.OrdinalIgnoreCase)) {
                final = "v" + final.Substring(1);
            }

            if (string.IsNullOrEmpty(final) || !Finals.TryGetValue(final, out var finalPhonemes)) {
                return Array.Empty<string>();
            }
            if (initial == null) {
                return finalPhonemes;
            }
            if (!Initials.TryGetValue(initial, out var initialPhonemes)) {
                return Array.Empty<string>();
            }
            return initialPhonemes.Concat(finalPhonemes).ToArray();
        }

        static int[] DistributeChinesePhonemes(string[] phonemes, Note[] notes) {
            if (phonemes.Length == 0) {
                return Array.Empty<int>();
            }
            if (phonemes.Length == 1) {
                return new[] { 0 };
            }

            int totalDuration = Math.Max(
                minChinesePhonemeTicks,
                notes[^1].position + notes[^1].duration - notes[0].position);
            int lastStart = Math.Max(0, totalDuration - minChinesePhonemeTicks);
            int mainIndex = FindMainVowelIndex(phonemes);
            if (mainIndex < 0) {
                return EvenlyDistribute(phonemes.Length, lastStart);
            }

            var positions = new int[phonemes.Length];
            positions[0] = 0;

            int prefixSpan = GetPrefixSpanTicks(totalDuration, mainIndex);
            for (int i = 1; i < phonemes.Length; i++) {
                if (i < mainIndex) {
                    positions[i] = mainIndex == 0
                        ? 0
                        : (int)Math.Round((double)prefixSpan * i / mainIndex);
                } else if (i == mainIndex) {
                    positions[i] = prefixSpan;
                } else {
                    positions[i] = GetTailStartTicks(phonemes, mainIndex, i, totalDuration);
                }
            }

            for (int i = 1; i < positions.Length; i++) {
                int maxPosition = Math.Max(
                    positions[i - 1] + minChinesePhonemeTicks,
                    totalDuration - minChinesePhonemeTicks * (positions.Length - i));
                positions[i] = Math.Min(
                    Math.Max(positions[i], positions[i - 1] + minChinesePhonemeTicks),
                    maxPosition);
            }
            for (int i = positions.Length - 1; i >= 1; i--) {
                int maxPosition = totalDuration - minChinesePhonemeTicks * (positions.Length - i);
                positions[i] = Math.Min(positions[i], maxPosition);
                if (positions[i] <= positions[i - 1]) {
                    positions[i - 1] = Math.Max(0, positions[i] - minChinesePhonemeTicks);
                }
            }
            positions[0] = 0;
            return positions;
        }

        static int GetPrefixSpanTicks(int totalDuration, int mainIndex) {
            if (mainIndex <= 0) {
                return 0;
            }
            double ratio = mainIndex switch {
                1 => 0.14,
                2 => 0.23,
                _ => 0.30,
            };
            int desired = (int)Math.Round(totalDuration * ratio);
            int minSpan = minChinesePhonemeTicks * mainIndex;
            return Math.Clamp(desired, minSpan, Math.Min(maxPrefixTicks, totalDuration - minChinesePhonemeTicks));
        }

        static int GetTailStartTicks(string[] phonemes, int mainIndex, int index, int totalDuration) {
            bool finalNasal = IsNasalCoda(phonemes[^1]);
            if (finalNasal && index == phonemes.Length - 1) {
                return Math.Max(
                    (int)Math.Round(totalDuration * 0.82),
                    totalDuration - maxNasalTailTicks);
            }

            int tailCount = phonemes.Length - mainIndex - 1;
            int tailIndex = index - mainIndex;
            double endRatio = finalNasal ? 0.66 : 0.72;
            int ratioPosition = (int)Math.Round(totalDuration * endRatio * tailIndex / Math.Max(1, tailCount));
            int cappedPosition = totalDuration - maxTailTicks * (tailCount - tailIndex + 1);
            return Math.Max(ratioPosition, cappedPosition);
        }

        static int FindMainVowelIndex(string[] phonemes) {
            var oralVowels = phonemes
                .Select((phoneme, index) => (phoneme, index))
                .Where(pair => IsOralVowel(pair.phoneme))
                .ToArray();
            if (oralVowels.Length == 0) {
                return -1;
            }
            if (oralVowels.Length == 1) {
                return oralVowels[0].index;
            }

            var openVowel = oralVowels.FirstOrDefault(pair =>
                pair.phoneme == "a" || pair.phoneme == "e" || pair.phoneme == "o");
            return openVowel.phoneme != null ? openVowel.index : oralVowels[0].index;
        }

        static int[] EvenlyDistribute(int count, int lastStart) {
            var positions = new int[count];
            for (int i = 0; i < count; i++) {
                positions[i] = count <= 1
                    ? 0
                    : (int)Math.Round((double)lastStart * i / (count - 1));
            }
            return positions;
        }

        static bool IsOralVowel(string phoneme) {
            return phoneme == "a"
                || phoneme == "i"
                || phoneme == "u"
                || phoneme == "e"
                || phoneme == "o";
        }

        static bool IsNasalCoda(string phoneme) {
            return phoneme == "N";
        }

        static string RomanizeHanziLyric(string lyric) {
            if (string.IsNullOrWhiteSpace(lyric)) {
                return lyric ?? string.Empty;
            }

            var elements = EnumerateTextElements(lyric.Trim()).ToArray();
            var hanzi = elements.Where(Pinyin.Pinyin.Instance.IsHanzi).ToList();
            if (hanzi.Count == 0) {
                return lyric.Trim();
            }

            var pinyinResult = Pinyin.Pinyin.Instance.HanziToPinyin(
                hanzi,
                ManTone.Style.NORMAL,
                Pinyin.Error.Default,
                false,
                false,
                false).ToStrList();

            if (pinyinResult == null) {
                return lyric.Trim();
            }

            var tokens = new List<string>();
            var run = new StringBuilder();
            int pinyinIndex = 0;
            foreach (var element in elements) {
                if (Pinyin.Pinyin.Instance.IsHanzi(element)) {
                    FlushRun(tokens, run);
                    tokens.Add(pinyinResult[pinyinIndex]);
                    pinyinIndex++;
                } else if (string.IsNullOrWhiteSpace(element) || TokenSeparators.Contains(element[0])) {
                    FlushRun(tokens, run);
                } else {
                    run.Append(element);
                }
            }
            FlushRun(tokens, run);
            return string.Join(" ", tokens);
        }

        static IEnumerable<string> SplitTokens(string text) {
            return (text ?? string.Empty)
                .Split(TokenSeparators, StringSplitOptions.RemoveEmptyEntries)
                .Select(token => token.Trim())
                .Where(token => token.Length > 0);
        }

        static IEnumerable<string> EnumerateTextElements(string text) {
            var enumerator = StringInfo.GetTextElementEnumerator(text);
            while (enumerator.MoveNext()) {
                yield return enumerator.GetTextElement();
            }
        }

        static string NormalizePinyin(string text) {
            if (string.IsNullOrWhiteSpace(text)) {
                return string.Empty;
            }

            var builder = new StringBuilder();
            var lower = text.Trim().ToLowerInvariant();
            for (int i = 0; i < lower.Length; i++) {
                char c = lower[i];
                if (c >= '1' && c <= '5') {
                    continue;
                }
                if (c == 'u' && i + 1 < lower.Length && lower[i + 1] == ':') {
                    builder.Append('v');
                    i++;
                    continue;
                }
                switch (c) {
                    case '\u0101':
                    case '\u00e1':
                    case '\u01ce':
                    case '\u00e0':
                        builder.Append('a');
                        break;
                    case '\u0113':
                    case '\u00e9':
                    case '\u011b':
                    case '\u00e8':
                        builder.Append('e');
                        break;
                    case '\u012b':
                    case '\u00ed':
                    case '\u01d0':
                    case '\u00ec':
                        builder.Append('i');
                        break;
                    case '\u014d':
                    case '\u00f3':
                    case '\u01d2':
                    case '\u00f2':
                        builder.Append('o');
                        break;
                    case '\u016b':
                    case '\u00fa':
                    case '\u01d4':
                    case '\u00f9':
                        builder.Append('u');
                        break;
                    case '\u01d6':
                    case '\u01d8':
                    case '\u01da':
                    case '\u01dc':
                    case '\u00fc':
                        builder.Append('v');
                        break;
                    default:
                        if (c >= 'a' && c <= 'z') {
                            builder.Append(c);
                        }
                        break;
                }
            }
            return builder.ToString();
        }

        static void FlushRun(List<string> tokens, StringBuilder run) {
            if (run.Length == 0) {
                return;
            }
            tokens.Add(run.ToString());
            run.Clear();
        }

        static bool IsRestLyric(string lyric) {
            return string.IsNullOrWhiteSpace(lyric)
                || lyric == "R"
                || lyric.Equals("SP", StringComparison.OrdinalIgnoreCase)
                || lyric.Equals("rest", StringComparison.OrdinalIgnoreCase)
                || lyric.Equals("pau", StringComparison.OrdinalIgnoreCase)
                || lyric.Equals("sil", StringComparison.OrdinalIgnoreCase);
        }

        static bool IsNeutrinoPhonemeToken(string token) {
            return NeutrinoPhonemeNames.Contains(token);
        }
    }
}
