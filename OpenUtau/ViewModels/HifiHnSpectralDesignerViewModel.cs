using System;
using System.Collections.ObjectModel;
using OpenUtau.Core.HifiNeural;
using ReactiveUI;
using ReactiveUI.Fody.Helpers;

namespace OpenUtau.App.ViewModels {
    public sealed class HifiHnSpectralDesignerViewModel : ViewModelBase {
        readonly int noteCount;

        [Reactive] public bool Enabled { get; set; }
        double bodyDb;
        double warmthDb;
        double presenceDb;
        double clarityDb;
        double airDb;
        public double BodyDb { get => bodyDb; set => SetBalanceValue(0, value); }
        public double WarmthDb { get => warmthDb; set => SetBalanceValue(1, value); }
        public double PresenceDb { get => presenceDb; set => SetBalanceValue(2, value); }
        public double ClarityDb { get => clarityDb; set => SetBalanceValue(3, value); }
        public double AirDb { get => airDb; set => SetBalanceValue(4, value); }
        double bodyHz = HifiHnSpectralProfile.DefaultFrequenciesHz[0];
        double warmthHz = HifiHnSpectralProfile.DefaultFrequenciesHz[1];
        double presenceHz = HifiHnSpectralProfile.DefaultFrequenciesHz[2];
        double clarityHz = HifiHnSpectralProfile.DefaultFrequenciesHz[3];
        double airHz = HifiHnSpectralProfile.DefaultFrequenciesHz[4];
        public double BodyHz { get => bodyHz; set => SetFrequency(0, value); }
        public double WarmthHz { get => warmthHz; set => SetFrequency(1, value); }
        public double PresenceHz { get => presenceHz; set => SetFrequency(2, value); }
        public double ClarityHz { get => clarityHz; set => SetFrequency(3, value); }
        public double AirHz { get => airHz; set => SetFrequency(4, value); }
        [Reactive] public bool DynamicsEnabled { get; set; }
        [Reactive] public int DynamicsTargetIndex { get; set; }
        [Reactive] public double ThresholdDb { get; set; }
        [Reactive] public double Ratio { get; set; }
        [Reactive] public double AttackMs { get; set; }
        [Reactive] public double ReleaseMs { get; set; }
        [Reactive] public double MaxReductionDb { get; set; }
        int selectedBandIndex = -1;
        bool dynamicsPanelExpanded;

        public int SelectedBandIndex {
            get => selectedBandIndex;
            set => this.RaiseAndSetIfChanged(
                ref selectedBandIndex,
                Math.Clamp(value, -1, HifiHnSpectralProfile.BandCount - 1));
        }
        public bool HasSelectedBand => SelectedBandIndex >= 0;
        public double SelectedBandOpacity => HasSelectedBand ? 1.0 : 0.0;
        public string SelectedBandTitle => HasSelectedBand
            ? string.Format(ThemeManager.GetString("hifi.hn.point"), SelectedBandIndex + 1)
            : string.Empty;
        public double SelectedFrequencyHz {
            get => SelectedBandIndex switch {
                0 => BodyHz,
                1 => WarmthHz,
                2 => PresenceHz,
                3 => ClarityHz,
                4 => AirHz,
                _ => 0,
            };
            set {
                switch (SelectedBandIndex) {
                    case 0: BodyHz = value; break;
                    case 1: WarmthHz = value; break;
                    case 2: PresenceHz = value; break;
                    case 3: ClarityHz = value; break;
                    case 4: AirHz = value; break;
                }
            }
        }
        public double SelectedBalanceDb {
            get => SelectedBandIndex switch {
                0 => BodyDb,
                1 => WarmthDb,
                2 => PresenceDb,
                3 => ClarityDb,
                4 => AirDb,
                _ => 0,
            };
            set {
                switch (SelectedBandIndex) {
                    case 0: BodyDb = value; break;
                    case 1: WarmthDb = value; break;
                    case 2: PresenceDb = value; break;
                    case 3: ClarityDb = value; break;
                    case 4: AirDb = value; break;
                }
            }
        }
        public bool DynamicsPanelExpanded {
            get => dynamicsPanelExpanded;
            set {
                if (dynamicsPanelExpanded == value) {
                    return;
                }
                this.RaiseAndSetIfChanged(ref dynamicsPanelExpanded, value);
                this.RaisePropertyChanged(nameof(DynamicsPanelHeight));
                this.RaisePropertyChanged(nameof(DynamicsPanelOpacity));
                this.RaisePropertyChanged(nameof(DynamicsPanelChevron));
            }
        }
        public double DynamicsPanelHeight => DynamicsPanelExpanded ? 124.0 : 0.0;
        public double DynamicsPanelOpacity => DynamicsPanelExpanded ? 1.0 : 0.0;
        public string DynamicsPanelChevron => DynamicsPanelExpanded ? "▲" : "▼";

        public string SelectionText => string.Format(
            ThemeManager.GetString("hifi.hn.selected"),
            noteCount);
        public string SelectionCompactText => string.Format(
            ThemeManager.GetString("hifi.hn.selected.compact"),
            noteCount);
        public ObservableCollection<string> DynamicsTargets { get; }

        public HifiHnSpectralDesignerViewModel(HifiHnSpectralProfile profile, int noteCount) {
            this.noteCount = Math.Max(1, noteCount);
            PropertyChanged += (_, args) => {
                if (IsSelectedBandDependency(args.PropertyName)) {
                    RaiseSelectedBandProperties();
                }
            };
            DynamicsTargets = new ObservableCollection<string> {
                ThemeManager.GetString("hifi.hn.target.harmonic"),
                ThemeManager.GetString("hifi.hn.target.noise"),
                ThemeManager.GetString("hifi.hn.target.both"),
            };
            Load(profile);
        }

        public void Reset() => Load(new HifiHnSpectralProfile());

        public void SelectPreviousBand() {
            SelectedBandIndex = SelectedBandIndex <= 0
                ? HifiHnSpectralProfile.BandCount - 1
                : SelectedBandIndex - 1;
        }

        public void SelectNextBand() {
            SelectedBandIndex = SelectedBandIndex < 0
                || SelectedBandIndex >= HifiHnSpectralProfile.BandCount - 1
                ? 0
                : SelectedBandIndex + 1;
        }

        public void ResetSelectedBalance() {
            SelectedBalanceDb = 0;
        }

        public void ToggleDynamicsPanel() {
            DynamicsPanelExpanded = !DynamicsPanelExpanded;
        }

        public HifiHnSpectralProfile BuildProfile() {
            return new HifiHnSpectralProfile {
                Enabled = Enabled,
                BalanceDb = new[] { BodyDb, WarmthDb, PresenceDb, ClarityDb, AirDb },
                FrequenciesHz = new[] { BodyHz, WarmthHz, PresenceHz, ClarityHz, AirHz },
                DynamicsEnabled = DynamicsEnabled,
                DynamicsTarget = Enum.IsDefined(typeof(HifiHnDynamicsTarget), DynamicsTargetIndex)
                    ? (HifiHnDynamicsTarget)DynamicsTargetIndex
                    : HifiHnDynamicsTarget.Both,
                ThresholdDb = ThresholdDb,
                Ratio = Ratio,
                AttackMs = AttackMs,
                ReleaseMs = ReleaseMs,
                MaxReductionDb = MaxReductionDb,
            }.Normalize();
        }

        void Load(HifiHnSpectralProfile profile) {
            profile = profile.Clone().Normalize();
            Enabled = profile.Enabled;
            SetBalance(profile.BalanceDb);
            SetFrequencies(profile.FrequenciesHz);
            DynamicsEnabled = profile.DynamicsEnabled;
            DynamicsTargetIndex = (int)profile.DynamicsTarget;
            ThresholdDb = profile.ThresholdDb;
            Ratio = profile.Ratio;
            AttackMs = profile.AttackMs;
            ReleaseMs = profile.ReleaseMs;
            MaxReductionDb = profile.MaxReductionDb;
        }

        void SetBalance(double[] values) {
            BodyDb = values.Length > 0 ? values[0] : 0;
            WarmthDb = values.Length > 1 ? values[1] : 0;
            PresenceDb = values.Length > 2 ? values[2] : 0;
            ClarityDb = values.Length > 3 ? values[3] : 0;
            AirDb = values.Length > 4 ? values[4] : 0;
        }

        void SetBalanceValue(int band, double value) {
            double requested = value;
            value = double.IsFinite(value)
                ? Math.Clamp(value, -HifiHnSpectralProfile.MaxBalanceDb, HifiHnSpectralProfile.MaxBalanceDb)
                : 0;
            switch (band) {
                case 0: this.RaiseAndSetIfChanged(ref bodyDb, value, nameof(BodyDb)); break;
                case 1: this.RaiseAndSetIfChanged(ref warmthDb, value, nameof(WarmthDb)); break;
                case 2: this.RaiseAndSetIfChanged(ref presenceDb, value, nameof(PresenceDb)); break;
                case 3: this.RaiseAndSetIfChanged(ref clarityDb, value, nameof(ClarityDb)); break;
                case 4: this.RaiseAndSetIfChanged(ref airDb, value, nameof(AirDb)); break;
            }
            if (!double.IsFinite(requested) || Math.Abs(requested - value) > 0.001) {
                this.RaisePropertyChanged(band switch {
                    0 => nameof(BodyDb),
                    1 => nameof(WarmthDb),
                    2 => nameof(PresenceDb),
                    3 => nameof(ClarityDb),
                    _ => nameof(AirDb),
                });
            }
        }

        void SetFrequencies(double[] values) {
            var defaults = HifiHnSpectralProfile.DefaultFrequenciesHz;
            bodyHz = values.Length > 0 ? values[0] : defaults[0];
            warmthHz = values.Length > 1 ? values[1] : defaults[1];
            presenceHz = values.Length > 2 ? values[2] : defaults[2];
            clarityHz = values.Length > 3 ? values[3] : defaults[3];
            airHz = values.Length > 4 ? values[4] : defaults[4];
            this.RaisePropertyChanged(nameof(BodyHz));
            this.RaisePropertyChanged(nameof(WarmthHz));
            this.RaisePropertyChanged(nameof(PresenceHz));
            this.RaisePropertyChanged(nameof(ClarityHz));
            this.RaisePropertyChanged(nameof(AirHz));
        }

        void SetFrequency(int band, double value) {
            double requested = value;
            var frequencies = new[] { bodyHz, warmthHz, presenceHz, clarityHz, airHz };
            double minimum = band == 0
                ? HifiHnSpectralProfile.MinFrequencyHz
                : frequencies[band - 1] * HifiHnSpectralProfile.MinFrequencyRatio;
            double maximum = band == HifiHnSpectralProfile.BandCount - 1
                ? HifiHnSpectralProfile.MaxFrequencyHz
                : frequencies[band + 1] / HifiHnSpectralProfile.MinFrequencyRatio;
            value = double.IsFinite(value)
                ? Math.Clamp(value, minimum, maximum)
                : HifiHnSpectralProfile.DefaultFrequenciesHz[band];
            switch (band) {
                case 0: this.RaiseAndSetIfChanged(ref bodyHz, value, nameof(BodyHz)); break;
                case 1: this.RaiseAndSetIfChanged(ref warmthHz, value, nameof(WarmthHz)); break;
                case 2: this.RaiseAndSetIfChanged(ref presenceHz, value, nameof(PresenceHz)); break;
                case 3: this.RaiseAndSetIfChanged(ref clarityHz, value, nameof(ClarityHz)); break;
                case 4: this.RaiseAndSetIfChanged(ref airHz, value, nameof(AirHz)); break;
            }
            if (!double.IsFinite(requested) || Math.Abs(requested - value) > 0.01) {
                this.RaisePropertyChanged(band switch {
                    0 => nameof(BodyHz),
                    1 => nameof(WarmthHz),
                    2 => nameof(PresenceHz),
                    3 => nameof(ClarityHz),
                    _ => nameof(AirHz),
                });
            }
        }

        static bool IsSelectedBandDependency(string? propertyName) {
            return propertyName is nameof(SelectedBandIndex)
                or nameof(BodyDb) or nameof(WarmthDb) or nameof(PresenceDb) or nameof(ClarityDb) or nameof(AirDb)
                or nameof(BodyHz) or nameof(WarmthHz) or nameof(PresenceHz) or nameof(ClarityHz) or nameof(AirHz);
        }

        void RaiseSelectedBandProperties() {
            this.RaisePropertyChanged(nameof(HasSelectedBand));
            this.RaisePropertyChanged(nameof(SelectedBandOpacity));
            this.RaisePropertyChanged(nameof(SelectedBandTitle));
            this.RaisePropertyChanged(nameof(SelectedFrequencyHz));
            this.RaisePropertyChanged(nameof(SelectedBalanceDb));
        }
    }
}
