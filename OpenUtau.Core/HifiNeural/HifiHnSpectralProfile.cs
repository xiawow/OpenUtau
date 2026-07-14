using System;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using K4os.Hash.xxHash;
using OpenUtau.Core.Ustx;

namespace OpenUtau.Core.HifiNeural {
    public enum HifiHnDynamicsTarget {
        Harmonic,
        Noise,
        Both,
    }

    public sealed class HifiHnSpectralProfile {
        public const string RendererSettingKey = "hifi-neura.hn-spectral.v1";
        public const int BandCount = 5;
        public const int MinBandCount = 2;
        public const int MaxBandCount = 16;
        public const double MaxBalanceDb = 18.0;
        public const double MaxBalancePercent = 100.0;
        public const double BalancePercentExponent = 1.25;
        public const double MinFrequencyHz = 40.0;
        public const double MaxFrequencyHz = 20000.0;
        public const double MinFrequencyRatio = 1.05;
        public static readonly double[] DefaultFrequenciesHz = { 120, 350, 1200, 4200, 11000 };

        public bool Enabled { get; set; } = true;
        public double[] BalanceDb { get; set; } = new double[BandCount];
        public double[] FrequenciesHz { get; set; } = (double[])DefaultFrequenciesHz.Clone();
        public bool DynamicsEnabled { get; set; }
        public HifiHnDynamicsTarget DynamicsTarget { get; set; } = HifiHnDynamicsTarget.Both;
        public double ThresholdDb { get; set; } = -30.0;
        public double Ratio { get; set; } = 2.0;
        public double AttackMs { get; set; } = 15.0;
        public double ReleaseMs { get; set; } = 120.0;
        public double MaxReductionDb { get; set; } = 6.0;

        [JsonIgnore]
        public bool HasAudibleEffect => Enabled
            && ((BalanceDb?.Any(value => Math.Abs(value) >= 0.01) ?? false)
                || (DynamicsEnabled && Ratio > 1.01 && MaxReductionDb >= 0.01));

        public HifiHnSpectralProfile Clone() {
            var clone = (HifiHnSpectralProfile)MemberwiseClone();
            clone.BalanceDb = BalanceDb == null ? new double[BandCount] : (double[])BalanceDb.Clone();
            clone.FrequenciesHz = FrequenciesHz == null
                ? (double[])DefaultFrequenciesHz.Clone()
                : (double[])FrequenciesHz.Clone();
            return clone;
        }

        public HifiHnSpectralProfile Normalize() {
            NormalizeBands();
            ThresholdDb = FiniteClamp(ThresholdDb, -60, 0, -30);
            Ratio = FiniteClamp(Ratio, 1, 12, 2);
            AttackMs = FiniteClamp(AttackMs, 1, 100, 15);
            ReleaseMs = FiniteClamp(ReleaseMs, 10, 600, 120);
            MaxReductionDb = FiniteClamp(MaxReductionDb, 0, 18, 6);
            if (!Enum.IsDefined(DynamicsTarget)) {
                DynamicsTarget = HifiHnDynamicsTarget.Both;
            }
            return this;
        }

        void NormalizeBands() {
            int requestedCount = Math.Max(BalanceDb?.Length ?? 0, FrequenciesHz?.Length ?? 0);
            int count = requestedCount == 0
                ? BandCount
                : Math.Clamp(requestedCount, MinBandCount, MaxBandCount);
            var values = new double[count];
            var fallbackFrequencies = BuildDefaultFrequencies(count);
            var frequencies = (double[])fallbackFrequencies.Clone();
            if (BalanceDb != null) {
                Array.Copy(BalanceDb, values, Math.Min(BalanceDb.Length, values.Length));
            }
            if (FrequenciesHz != null) {
                Array.Copy(FrequenciesHz, frequencies, Math.Min(FrequenciesHz.Length, frequencies.Length));
            }

            var bands = Enumerable.Range(0, count)
                .Select(i => (
                    Frequency: FiniteClamp(frequencies[i], MinFrequencyHz, MaxFrequencyHz, fallbackFrequencies[i]),
                    Balance: FiniteClamp(values[i], -MaxBalanceDb, MaxBalanceDb, 0)))
                .OrderBy(band => band.Frequency)
                .ToArray();
            for (int i = 0; i < bands.Length; i++) {
                double minimum = i == 0
                    ? MinFrequencyHz
                    : frequencies[i - 1] * MinFrequencyRatio;
                double maximum = MaxFrequencyHz
                    / Math.Pow(MinFrequencyRatio, bands.Length - 1 - i);
                frequencies[i] = Math.Clamp(bands[i].Frequency, minimum, maximum);
                values[i] = bands[i].Balance;
            }
            FrequenciesHz = frequencies;
            BalanceDb = values;
        }

        static double[] BuildDefaultFrequencies(int count) {
            if (count == BandCount) {
                return (double[])DefaultFrequenciesHz.Clone();
            }
            var frequencies = new double[count];
            double logStart = Math.Log(DefaultFrequenciesHz[0]);
            double logEnd = Math.Log(DefaultFrequenciesHz[^1]);
            for (int i = 0; i < count; i++) {
                double t = i / (double)(count - 1);
                frequencies[i] = Math.Exp(logStart + (logEnd - logStart) * t);
            }
            return frequencies;
        }

        public string Serialize() {
            Normalize();
            return JsonSerializer.Serialize(this);
        }

        public string CacheKey() {
            string serialized = Serialize();
            return XXH64.DigestOf(System.Text.Encoding.UTF8.GetBytes(serialized)).ToString("x16");
        }

        public static HifiHnSpectralProfile FromNote(UNote note) {
            return Deserialize(note.GetRendererSetting(RendererSettingKey));
        }

        public static HifiHnSpectralProfile Deserialize(string? value) {
            if (string.IsNullOrWhiteSpace(value)) {
                return new HifiHnSpectralProfile();
            }
            try {
                return (JsonSerializer.Deserialize<HifiHnSpectralProfile>(value)
                    ?? new HifiHnSpectralProfile()).Normalize();
            } catch {
                return new HifiHnSpectralProfile();
            }
        }

        public static double PercentToBalanceDb(double percent) {
            if (!double.IsFinite(percent)) {
                return 0;
            }
            double normalized = Math.Clamp(
                Math.Abs(percent) / MaxBalancePercent,
                0,
                1);
            return Math.CopySign(
                MaxBalanceDb * Math.Pow(normalized, BalancePercentExponent),
                percent);
        }

        public static double BalanceDbToPercent(double balanceDb) {
            if (!double.IsFinite(balanceDb)) {
                return 0;
            }
            double normalized = Math.Clamp(Math.Abs(balanceDb) / MaxBalanceDb, 0, 1);
            return Math.CopySign(
                MaxBalancePercent * Math.Pow(normalized, 1.0 / BalancePercentExponent),
                balanceDb);
        }

        static double FiniteClamp(double value, double min, double max, double fallback) {
            return double.IsFinite(value) ? Math.Clamp(value, min, max) : fallback;
        }
    }
}
