using System;
using System.Collections.Generic;
using System.Linq;
using Serilog;

namespace OpenUtau.Core.HifiNeural {
    public sealed class HifiSourceLeadingSanitizerReport {
        public string Phoneme { get; init; } = string.Empty;
        public string PreviousPhoneme { get; init; } = string.Empty;
        public string LeadingToken { get; init; } = string.Empty;
        public string PreviousStableToken { get; init; } = string.Empty;
        public double SourcePreutterMs { get; init; }
        public double ExtraSkipMs { get; init; }
        public int ExtraSkipSamples { get; init; }
        public int ExtraStartOffsetFrames { get; init; }
        public bool DeclickApplied { get; init; }
        public string Reason { get; init; } = string.Empty;
        public bool Applied => ExtraSkipSamples > 0;
    }

    public static class HifiSourceLeadingSanitizer {
        public static HifiSourceLeadingSanitizerReport Analyze(
            string phoneme,
            string? previousPhoneme,
            bool isFirstOrAfterSilence,
            double sourcePreutterMs,
            int maxExtraStartOffsetFrames = int.MaxValue) {
            string[] tokens = SplitTokens(phoneme);
            string leading = tokens.Length >= 2 ? Normalize(tokens[0]) : string.Empty;
            string previousStable = ExtractStableToken(previousPhoneme);

            if (!HifiNeuralConfig.EnableSourceLeadingSanitizer) {
                return Skip(phoneme, previousPhoneme, leading, previousStable, sourcePreutterMs, "disabled");
            }
            if (sourcePreutterMs <= HifiNeuralConfig.SourceLeadingSanitizeGuardMs + 1) {
                return Skip(phoneme, previousPhoneme, leading, previousStable, sourcePreutterMs, "preutter_too_short");
            }
            if (tokens.Length < 2 || string.IsNullOrWhiteSpace(leading)) {
                return Skip(phoneme, previousPhoneme, leading, previousStable, sourcePreutterMs, "no_leading_context");
            }
            if (IsSilenceOrRest(phoneme)) {
                return Skip(phoneme, previousPhoneme, leading, previousStable, sourcePreutterMs, "silence");
            }
            if (leading == "-") {
                return Skip(phoneme, previousPhoneme, leading, previousStable, sourcePreutterMs, "explicit_start_alias");
            }
            if (!IsLikelyLeadingContextToken(leading)) {
                return Skip(phoneme, previousPhoneme, leading, previousStable, sourcePreutterMs, "leading_token_not_vowel_context");
            }
            if (!isFirstOrAfterSilence) {
                bool matchesPrevious = LeadingMatchesPrevious(leading, previousStable);
                var connectedTrim = TryConnectedVcvTrim(
                    phoneme,
                    previousPhoneme,
                    leading,
                    previousStable,
                    sourcePreutterMs,
                    maxExtraStartOffsetFrames,
                    matchesPrevious);
                if (connectedTrim != null) {
                    return connectedTrim;
                }
                return Skip(
                    phoneme,
                    previousPhoneme,
                    leading,
                    previousStable,
                    sourcePreutterMs,
                    matchesPrevious ? "previous_context_matches" : "connected_context_preserved");
            }
            if (maxExtraStartOffsetFrames <= 0) {
                return Skip(phoneme, previousPhoneme, leading, previousStable, sourcePreutterMs, "no_target_room_for_sanitize");
            }

            double extraSkipMs = Math.Min(
                sourcePreutterMs - HifiNeuralConfig.SourceLeadingSanitizeGuardMs,
                HifiNeuralConfig.SourceLeadingSanitizeMaxSkipMs);
            extraSkipMs = Math.Min(extraSkipMs, maxExtraStartOffsetFrames * HifiF0Builder.FrameMs);
            if (extraSkipMs < HifiF0Builder.FrameMs * 0.5) {
                return Skip(phoneme, previousPhoneme, leading, previousStable, sourcePreutterMs, "skip_too_small");
            }
            int extraSkipSamples = Math.Max(0, (int)Math.Round(extraSkipMs * HifiMelExtractor.SampleRate / 1000.0));
            var report = new HifiSourceLeadingSanitizerReport {
                Phoneme = phoneme,
                PreviousPhoneme = previousPhoneme ?? string.Empty,
                LeadingToken = leading,
                PreviousStableToken = previousStable,
                SourcePreutterMs = sourcePreutterMs,
                ExtraSkipMs = extraSkipMs,
                ExtraSkipSamples = extraSkipSamples,
                ExtraStartOffsetFrames = 0,
                DeclickApplied = HifiNeuralConfig.SourceLeadingSanitizeDeclickMs > 0,
                Reason = isFirstOrAfterSilence ? "leading_context_after_silence" : "leading_context_mismatch",
            };
            Log.Information(
                "HifiSourceLeadingSanitizer applied phoneme={Phoneme} previous={PreviousPhoneme} leading_token={LeadingToken} previous_stable={PreviousStableToken} source_preutter_ms={SourcePreutterMs:F3} extra_skip_ms={ExtraSkipMs:F3} extra_skip_samples={ExtraSkipSamples} extra_start_offset_frames={ExtraStartOffsetFrames} reason={Reason}",
                report.Phoneme,
                report.PreviousPhoneme,
                report.LeadingToken,
                report.PreviousStableToken,
                report.SourcePreutterMs,
                report.ExtraSkipMs,
                report.ExtraSkipSamples,
                report.ExtraStartOffsetFrames,
                report.Reason);
            return report;
        }

        static HifiSourceLeadingSanitizerReport? TryConnectedVcvTrim(
            string phoneme,
            string? previousPhoneme,
            string leading,
            string previousStable,
            double sourcePreutterMs,
            int maxExtraStartOffsetFrames,
            bool matchesPrevious) {
            if (maxExtraStartOffsetFrames <= 0) {
                return null;
            }
            double keepMs = Math.Max(HifiNeuralConfig.SourceLeadingSanitizeGuardMs, 10.0);
            double maxTrimMs = HifiNeuralConfig.SourceLeadingSanitizeMaxSkipMs * (matchesPrevious ? 0.45 : 0.25);
            double extraSkipMs = Math.Min(sourcePreutterMs - keepMs, maxTrimMs);
            extraSkipMs = Math.Min(extraSkipMs, maxExtraStartOffsetFrames * HifiF0Builder.FrameMs);
            if (extraSkipMs < HifiF0Builder.FrameMs * 0.5) {
                return null;
            }
            int extraSkipSamples = Math.Max(0, (int)Math.Round(extraSkipMs * HifiMelExtractor.SampleRate / 1000.0));
            var report = new HifiSourceLeadingSanitizerReport {
                Phoneme = phoneme,
                PreviousPhoneme = previousPhoneme ?? string.Empty,
                LeadingToken = leading,
                PreviousStableToken = previousStable,
                SourcePreutterMs = sourcePreutterMs,
                ExtraSkipMs = extraSkipMs,
                ExtraSkipSamples = extraSkipSamples,
                ExtraStartOffsetFrames = 0,
                DeclickApplied = HifiNeuralConfig.SourceLeadingSanitizeDeclickMs > 0,
                Reason = matchesPrevious ? "connected_vcv_trim_match" : "connected_vcv_trim",
            };
            Log.Information(
                "HifiSourceLeadingSanitizer connected_trim phoneme={Phoneme} previous={PreviousPhoneme} leading_token={LeadingToken} previous_stable={PreviousStableToken} source_preutter_ms={SourcePreutterMs:F3} extra_skip_ms={ExtraSkipMs:F3} extra_skip_samples={ExtraSkipSamples} reason={Reason}",
                report.Phoneme,
                report.PreviousPhoneme,
                report.LeadingToken,
                report.PreviousStableToken,
                report.SourcePreutterMs,
                report.ExtraSkipMs,
                report.ExtraSkipSamples,
                report.Reason);
            return report;
        }

        public static float[] Apply(float[] samples, HifiSourceLeadingSanitizerReport report) {
            if (samples.Length == 0 || report.ExtraSkipSamples <= 0) {
                return samples;
            }
            int skip = Math.Clamp(report.ExtraSkipSamples, 0, Math.Max(0, samples.Length - 1));
            var sanitized = samples.Skip(skip).ToArray();
            ApplyDeclick(sanitized, HifiNeuralConfig.SourceLeadingSanitizeDeclickMs);
            return sanitized;
        }

        public static double SourcePreutterAfterSanitize(HifiSourceLeadingSanitizerReport report) {
            return Math.Max(0, report.SourcePreutterMs - report.ExtraSkipMs);
        }

        static HifiSourceLeadingSanitizerReport Skip(
            string phoneme,
            string? previousPhoneme,
            string leading,
            string previousStable,
            double sourcePreutterMs,
            string reason) {
            return new HifiSourceLeadingSanitizerReport {
                Phoneme = phoneme,
                PreviousPhoneme = previousPhoneme ?? string.Empty,
                LeadingToken = leading,
                PreviousStableToken = previousStable,
                SourcePreutterMs = sourcePreutterMs,
                Reason = reason,
            };
        }

        static void ApplyDeclick(float[] samples, double fadeMs) {
            if (samples.Length == 0 || fadeMs <= 0) {
                return;
            }
            int fadeSamples = Math.Clamp((int)Math.Round(fadeMs * HifiMelExtractor.SampleRate / 1000.0), 1, Math.Min(samples.Length, 256));
            for (int i = 0; i < fadeSamples; i++) {
                float gain = (float)(0.5 - 0.5 * Math.Cos(Math.PI * (i + 1) / (fadeSamples + 1)));
                samples[i] *= gain;
            }
        }

        static bool LeadingMatchesPrevious(string leading, string previousStable) {
            if (string.IsNullOrWhiteSpace(leading) || string.IsNullOrWhiteSpace(previousStable)) {
                return false;
            }
            if (leading == previousStable) {
                return true;
            }
            return NormalizeVowel(leading) == NormalizeVowel(previousStable);
        }

        static string ExtractStableToken(string? phoneme) {
            string[] tokens = SplitTokens(phoneme ?? string.Empty);
            for (int i = tokens.Length - 1; i >= 0; i--) {
                string token = Normalize(tokens[i]);
                string jpVowel = TryExtractJapaneseVowel(token);
                if (!string.IsNullOrEmpty(jpVowel)) {
                    return jpVowel;
                }
                if (IsLikelyLeadingContextToken(token)) {
                    return NormalizeVowel(token);
                }
            }
            string normalized = Normalize(phoneme ?? string.Empty);
            for (int i = normalized.Length - 1; i >= 0; i--) {
                char c = normalized[i];
                if ("aeiou".Contains(c)) {
                    return c.ToString();
                }
            }
            if (normalized.EndsWith("ng", StringComparison.Ordinal)) {
                return "ng";
            }
            if (normalized.EndsWith("n", StringComparison.Ordinal)) {
                return "n";
            }
            return string.Empty;
        }

        static bool IsLikelyLeadingContextToken(string token) {
            token = Normalize(token);
            if (string.IsNullOrWhiteSpace(token)) {
                return false;
            }
            if (!string.IsNullOrEmpty(TryExtractJapaneseVowel(token))) {
                return true;
            }
            if (token is "a" or "i" or "u" or "e" or "o" or "n" or "ng" or "m") {
                return true;
            }
            return token.Length <= 3 && token.Any(c => "aeiou".Contains(c));
        }

        static bool IsSilenceOrRest(string phoneme) {
            string p = Normalize(phoneme);
            return p == "r" || p == "rest" || p == "sil" || p == "pau" || p == "-" || p == "br";
        }

        static string NormalizeVowel(string token) {
            token = Normalize(token);
            string jpVowel = TryExtractJapaneseVowel(token);
            if (!string.IsNullOrEmpty(jpVowel)) {
                return jpVowel;
            }
            if (token is "ng" or "n" or "m") {
                return token;
            }
            foreach (char c in token) {
                if ("aeiou".Contains(c)) {
                    return c.ToString();
                }
            }
            return token;
        }

        static string[] SplitTokens(string phoneme) {
            return (phoneme ?? string.Empty)
                .Split(new[] { ' ', '\t', '-', '_', '+', '/' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(Normalize)
                .Where(token => token.Length > 0)
                .ToArray();
        }

        static string Normalize(string value) => (value ?? string.Empty).Trim().ToLowerInvariant();

        static string TryExtractJapaneseVowel(string token) {
            if (string.IsNullOrWhiteSpace(token)) {
                return string.Empty;
            }
            for (int i = token.Length - 1; i >= 0; i--) {
                char c = token[i];
                string vowel = JapaneseKanaVowel(c);
                if (!string.IsNullOrEmpty(vowel)) {
                    return vowel;
                }
                if (JapaneseA.Contains(c)) return "a";
                if (JapaneseI.Contains(c)) return "i";
                if (JapaneseU.Contains(c)) return "u";
                if (JapaneseE.Contains(c)) return "e";
                if (JapaneseO.Contains(c)) return "o";
                if (c == 'ん' || c == 'ン') return "n";
            }
            return string.Empty;
        }

        static string JapaneseKanaVowel(char c) {
            return c switch {
                '\u3042' or '\u304b' or '\u304c' or '\u3055' or '\u3056' or '\u305f' or '\u3060' or '\u306a' or '\u306f' or '\u3070' or '\u3071' or '\u307e' or '\u3084' or '\u3089' or '\u308f'
                    or '\u30a2' or '\u30ab' or '\u30ac' or '\u30b5' or '\u30b6' or '\u30bf' or '\u30c0' or '\u30ca' or '\u30cf' or '\u30d0' or '\u30d1' or '\u30de' or '\u30e4' or '\u30e9' or '\u30ef' => "a",
                '\u3044' or '\u304d' or '\u304e' or '\u3057' or '\u3058' or '\u3061' or '\u3062' or '\u306b' or '\u3072' or '\u3073' or '\u3074' or '\u307f' or '\u308a'
                    or '\u30a4' or '\u30ad' or '\u30ae' or '\u30b7' or '\u30b8' or '\u30c1' or '\u30c2' or '\u30cb' or '\u30d2' or '\u30d3' or '\u30d4' or '\u30df' or '\u30ea' => "i",
                '\u3046' or '\u304f' or '\u3050' or '\u3059' or '\u305a' or '\u3064' or '\u3065' or '\u306c' or '\u3075' or '\u3076' or '\u3077' or '\u3080' or '\u3086' or '\u308b'
                    or '\u30a6' or '\u30af' or '\u30b0' or '\u30b9' or '\u30ba' or '\u30c4' or '\u30c5' or '\u30cc' or '\u30d5' or '\u30d6' or '\u30d7' or '\u30e0' or '\u30e6' or '\u30eb' => "u",
                '\u3048' or '\u3051' or '\u3052' or '\u305b' or '\u305c' or '\u3066' or '\u3067' or '\u306d' or '\u3078' or '\u3079' or '\u307a' or '\u3081' or '\u308c'
                    or '\u30a8' or '\u30b1' or '\u30b2' or '\u30bb' or '\u30bc' or '\u30c6' or '\u30c7' or '\u30cd' or '\u30d8' or '\u30d9' or '\u30da' or '\u30e1' or '\u30ec' => "e",
                '\u304a' or '\u3053' or '\u3054' or '\u305d' or '\u305e' or '\u3068' or '\u3069' or '\u306e' or '\u307b' or '\u307c' or '\u307d' or '\u3082' or '\u3088' or '\u308d' or '\u3092'
                    or '\u30aa' or '\u30b3' or '\u30b4' or '\u30bd' or '\u30be' or '\u30c8' or '\u30c9' or '\u30ce' or '\u30db' or '\u30dc' or '\u30dd' or '\u30e2' or '\u30e8' or '\u30ed' or '\u30f2' => "o",
                '\u3093' or '\u30f3' => "n",
                _ => string.Empty,
            };
        }

        static readonly HashSet<char> JapaneseA = new(new[] {
            'あ','ぁ','か','が','さ','ざ','た','だ','な','は','ば','ぱ','ま','ゃ','や','ら','ゎ','わ',
            'ア','ァ','カ','ガ','サ','ザ','タ','ダ','ナ','ハ','バ','パ','マ','ャ','ヤ','ラ','ヮ','ワ',
        });
        static readonly HashSet<char> JapaneseI = new(new[] {
            'い','ぃ','き','ぎ','し','じ','ち','ぢ','に','ひ','び','ぴ','み','り',
            'イ','ィ','キ','ギ','シ','ジ','チ','ヂ','ニ','ヒ','ビ','ピ','ミ','リ',
        });
        static readonly HashSet<char> JapaneseU = new(new[] {
            'う','ぅ','く','ぐ','す','ず','つ','づ','ぬ','ふ','ぶ','ぷ','む','ゅ','ゆ','る','ゔ',
            'ウ','ゥ','ク','グ','ス','ズ','ツ','ヅ','ヌ','フ','ブ','プ','ム','ュ','ユ','ル','ヴ',
        });
        static readonly HashSet<char> JapaneseE = new(new[] {
            'え','ぇ','け','げ','せ','ぜ','て','で','ね','へ','べ','ぺ','め','れ',
            'エ','ェ','ケ','ゲ','セ','ゼ','テ','デ','ネ','ヘ','ベ','ペ','メ','レ',
        });
        static readonly HashSet<char> JapaneseO = new(new[] {
            'お','ぉ','こ','ご','そ','ぞ','と','ど','の','ほ','ぼ','ぽ','も','ょ','よ','ろ','を',
            'オ','ォ','コ','ゴ','ソ','ゾ','ト','ド','ノ','ホ','ボ','ポ','モ','ョ','ヨ','ロ','ヲ',
        });
    }
}
