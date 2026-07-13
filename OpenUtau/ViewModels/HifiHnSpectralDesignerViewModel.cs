using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Linq;
using OpenUtau.Core.HifiNeural;
using ReactiveUI;
using ReactiveUI.Fody.Helpers;

namespace OpenUtau.App.ViewModels {
    public sealed class HifiHnBandViewModel : ReactiveObject {
        double frequencyHz;
        double balanceDb;

        public double FrequencyHz {
            get => frequencyHz;
            set => this.RaiseAndSetIfChanged(
                ref frequencyHz,
                double.IsFinite(value)
                    ? Math.Clamp(value, HifiHnSpectralProfile.MinFrequencyHz, HifiHnSpectralProfile.MaxFrequencyHz)
                    : HifiHnSpectralProfile.MinFrequencyHz);
        }

        public double BalanceDb {
            get => balanceDb;
            set => this.RaiseAndSetIfChanged(
                ref balanceDb,
                double.IsFinite(value)
                    ? Math.Clamp(value, -HifiHnSpectralProfile.MaxBalanceDb, HifiHnSpectralProfile.MaxBalanceDb)
                    : 0);
        }

        public HifiHnBandViewModel(double frequencyHz, double balanceDb) {
            this.frequencyHz = double.IsFinite(frequencyHz)
                ? Math.Clamp(frequencyHz, HifiHnSpectralProfile.MinFrequencyHz, HifiHnSpectralProfile.MaxFrequencyHz)
                : HifiHnSpectralProfile.MinFrequencyHz;
            this.balanceDb = double.IsFinite(balanceDb)
                ? Math.Clamp(balanceDb, -HifiHnSpectralProfile.MaxBalanceDb, HifiHnSpectralProfile.MaxBalanceDb)
                : 0;
        }
    }

    public sealed class HifiHnSpectralDesignerViewModel : ViewModelBase {
        readonly record struct BandClipboard(double FrequencyHz, double BalanceDb);
        sealed record BandSnapshot(double[] FrequenciesHz, double[] BalanceDb, int SelectedIndex);
        const int MaxHistoryEntries = 64;
        static BandClipboard? clipboard;

        readonly int noteCount;
        readonly System.Collections.Generic.List<BandSnapshot> undoHistory = new();
        readonly System.Collections.Generic.List<BandSnapshot> redoHistory = new();
        bool updatingBands;
        int bandEditDepth;
        BandSnapshot? pendingBandEdit;
        int selectedBandIndex = -1;
        bool dynamicsPanelExpanded;

        [Reactive] public bool Enabled { get; set; }
        [Reactive] public bool DynamicsEnabled { get; set; }
        [Reactive] public int DynamicsTargetIndex { get; set; }
        [Reactive] public double ThresholdDb { get; set; }
        [Reactive] public double Ratio { get; set; }
        [Reactive] public double AttackMs { get; set; }
        [Reactive] public double ReleaseMs { get; set; }
        [Reactive] public double MaxReductionDb { get; set; }

        public ObservableCollection<HifiHnBandViewModel> Bands { get; } = new();

        public int SelectedBandIndex {
            get => selectedBandIndex;
            set {
                int maximum = Bands.Count - 1;
                int clamped = maximum < 0 ? -1 : Math.Clamp(value, -1, maximum);
                if (selectedBandIndex == clamped) {
                    return;
                }
                this.RaiseAndSetIfChanged(ref selectedBandIndex, clamped);
                RaiseSelectedBandProperties();
            }
        }

        public bool HasSelectedBand => SelectedBandIndex >= 0 && SelectedBandIndex < Bands.Count;
        public bool CanDeleteSelectedBand => HasSelectedBand && Bands.Count > HifiHnSpectralProfile.MinBandCount;
        public bool CanPasteBand => clipboard.HasValue && Bands.Count < HifiHnSpectralProfile.MaxBandCount;
        public bool CanUndoBandEdit => undoHistory.Count > 0;
        public bool CanRedoBandEdit => redoHistory.Count > 0;
        public double SelectedBandOpacity => HasSelectedBand ? 1.0 : 0.0;
        public string SelectedBandTitle => HasSelectedBand
            ? string.Format(ThemeManager.GetString("hifi.hn.point"), SelectedBandIndex + 1)
            : string.Empty;
        public double SelectedFrequencyHz {
            get => HasSelectedBand ? Bands[SelectedBandIndex].FrequencyHz : 0;
            set {
                if (HasSelectedBand) {
                    RunBandEdit(() =>
                        Bands[SelectedBandIndex].FrequencyHz = ClampFrequency(SelectedBandIndex, value));
                }
            }
        }
        public double SelectedBalanceDb {
            get => HasSelectedBand ? Bands[SelectedBandIndex].BalanceDb : 0;
            set {
                if (HasSelectedBand) {
                    RunBandEdit(() => Bands[SelectedBandIndex].BalanceDb = value);
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
        public string DynamicsPanelChevron => DynamicsPanelExpanded ? "\u25B2" : "\u25BC";

        public string SelectionText => string.Format(
            ThemeManager.GetString("hifi.hn.selected"),
            noteCount);
        public string SelectionCompactText => string.Format(
            ThemeManager.GetString("hifi.hn.selected.compact"),
            noteCount);
        public ObservableCollection<string> DynamicsTargets { get; }

        public HifiHnSpectralDesignerViewModel(HifiHnSpectralProfile profile, int noteCount) {
            this.noteCount = Math.Max(1, noteCount);
            Bands.CollectionChanged += OnBandsChanged;
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
                ? Bands.Count - 1
                : SelectedBandIndex - 1;
        }

        public void SelectNextBand() {
            SelectedBandIndex = SelectedBandIndex < 0 || SelectedBandIndex >= Bands.Count - 1
                ? 0
                : SelectedBandIndex + 1;
        }

        public void ResetSelectedBalance() {
            SelectedBalanceDb = 0;
        }

        public void ToggleDynamicsPanel() {
            DynamicsPanelExpanded = !DynamicsPanelExpanded;
        }

        public int AddBand(double frequencyHz, double balanceDb) {
            return RunBandEdit(() => AddBandCore(frequencyHz, balanceDb));
        }

        int AddBandCore(double frequencyHz, double balanceDb) {
            if (Bands.Count >= HifiHnSpectralProfile.MaxBandCount) {
                return -1;
            }
            frequencyHz = double.IsFinite(frequencyHz)
                ? Math.Clamp(frequencyHz, HifiHnSpectralProfile.MinFrequencyHz, HifiHnSpectralProfile.MaxFrequencyHz)
                : HifiHnSpectralProfile.MinFrequencyHz;
            int bestIndex = -1;
            double bestFrequency = frequencyHz;
            double bestDistance = double.MaxValue;
            for (int index = 0; index <= Bands.Count; index++) {
                double minimum = index == 0
                    ? HifiHnSpectralProfile.MinFrequencyHz
                    : Bands[index - 1].FrequencyHz * HifiHnSpectralProfile.MinFrequencyRatio;
                double maximum = index == Bands.Count
                    ? HifiHnSpectralProfile.MaxFrequencyHz
                    : Bands[index].FrequencyHz / HifiHnSpectralProfile.MinFrequencyRatio;
                if (minimum > maximum) {
                    continue;
                }
                double candidate = Math.Clamp(frequencyHz, minimum, maximum);
                double distance = Math.Abs(Math.Log(candidate) - Math.Log(frequencyHz));
                if (distance < bestDistance) {
                    bestIndex = index;
                    bestFrequency = candidate;
                    bestDistance = distance;
                }
            }
            if (bestIndex < 0) {
                return -1;
            }
            Bands.Insert(bestIndex, new HifiHnBandViewModel(bestFrequency, balanceDb));
            SelectedBandIndex = bestIndex;
            return bestIndex;
        }

        public void DeleteSelectedBand() {
            if (!CanDeleteSelectedBand) {
                return;
            }
            RunBandEdit(() => {
                int removedIndex = SelectedBandIndex;
                Bands.RemoveAt(removedIndex);
                SelectedBandIndex = Math.Min(removedIndex, Bands.Count - 1);
            });
        }

        public void CopySelectedBand() {
            if (HasSelectedBand) {
                var band = Bands[SelectedBandIndex];
                clipboard = new BandClipboard(band.FrequencyHz, band.BalanceDb);
            }
        }

        public void CutSelectedBand() {
            if (!CanDeleteSelectedBand) {
                return;
            }
            CopySelectedBand();
            DeleteSelectedBand();
        }

        public void PasteBand() {
            if (!CanPasteBand || !clipboard.HasValue) {
                return;
            }
            BandClipboard value = clipboard.Value;
            double frequency = CanInsertAtFrequency(value.FrequencyHz)
                ? value.FrequencyHz
                : FrequencyNearSelectedBand();
            AddBand(frequency, value.BalanceDb);
        }

        public void BeginBandEdit() {
            if (bandEditDepth++ == 0) {
                pendingBandEdit = CaptureBandSnapshot();
            }
        }

        public void CommitBandEdit() {
            if (bandEditDepth <= 0) {
                return;
            }
            bandEditDepth--;
            if (bandEditDepth > 0) {
                return;
            }
            BandSnapshot? before = pendingBandEdit;
            pendingBandEdit = null;
            if (before != null && !BandSnapshotsEqual(before, CaptureBandSnapshot())) {
                PushHistory(undoHistory, before);
                redoHistory.Clear();
                RaiseHistoryProperties();
            }
        }

        public void UndoBandEdit() {
            FinishPendingBandEdit();
            if (undoHistory.Count == 0) {
                return;
            }
            BandSnapshot target = PopHistory(undoHistory);
            PushHistory(redoHistory, CaptureBandSnapshot());
            RestoreBandSnapshot(target);
            RaiseHistoryProperties();
        }

        public void RedoBandEdit() {
            FinishPendingBandEdit();
            if (redoHistory.Count == 0) {
                return;
            }
            BandSnapshot target = PopHistory(redoHistory);
            PushHistory(undoHistory, CaptureBandSnapshot());
            RestoreBandSnapshot(target);
            RaiseHistoryProperties();
        }

        public HifiHnSpectralProfile BuildProfile() {
            return new HifiHnSpectralProfile {
                Enabled = Enabled,
                BalanceDb = Bands.Select(band => band.BalanceDb).ToArray(),
                FrequenciesHz = Bands.Select(band => band.FrequencyHz).ToArray(),
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
            updatingBands = true;
            try {
                foreach (var band in Bands) {
                    band.PropertyChanged -= OnBandPropertyChanged;
                }
                Bands.Clear();
                for (int i = 0; i < profile.FrequenciesHz.Length; i++) {
                    Bands.Add(new HifiHnBandViewModel(profile.FrequenciesHz[i], profile.BalanceDb[i]));
                }
            } finally {
                updatingBands = false;
            }
            SelectedBandIndex = -1;
            DynamicsEnabled = profile.DynamicsEnabled;
            DynamicsTargetIndex = (int)profile.DynamicsTarget;
            ThresholdDb = profile.ThresholdDb;
            Ratio = profile.Ratio;
            AttackMs = profile.AttackMs;
            ReleaseMs = profile.ReleaseMs;
            MaxReductionDb = profile.MaxReductionDb;
            undoHistory.Clear();
            redoHistory.Clear();
            pendingBandEdit = null;
            bandEditDepth = 0;
            RaiseSelectedBandProperties();
            RaiseHistoryProperties();
        }

        void OnBandsChanged(object? sender, NotifyCollectionChangedEventArgs e) {
            if (e.OldItems != null) {
                foreach (HifiHnBandViewModel band in e.OldItems) {
                    band.PropertyChanged -= OnBandPropertyChanged;
                }
            }
            if (e.NewItems != null) {
                foreach (HifiHnBandViewModel band in e.NewItems) {
                    band.PropertyChanged += OnBandPropertyChanged;
                }
            }
            if (!updatingBands && SelectedBandIndex >= Bands.Count) {
                SelectedBandIndex = Bands.Count - 1;
            }
            RaiseSelectedBandProperties();
        }

        void OnBandPropertyChanged(object? sender, PropertyChangedEventArgs e) {
            if (sender is not HifiHnBandViewModel band) {
                return;
            }
            int index = Bands.IndexOf(band);
            if (index < 0) {
                return;
            }
            if (!updatingBands && e.PropertyName == nameof(HifiHnBandViewModel.FrequencyHz)) {
                double clamped = ClampFrequency(index, band.FrequencyHz);
                if (Math.Abs(clamped - band.FrequencyHz) > 0.001) {
                    updatingBands = true;
                    try {
                        band.FrequencyHz = clamped;
                    } finally {
                        updatingBands = false;
                    }
                }
            }
            if (index == SelectedBandIndex) {
                RaiseSelectedBandProperties();
            }
        }

        double ClampFrequency(int index, double value) {
            double minimum = index == 0
                ? HifiHnSpectralProfile.MinFrequencyHz
                : Bands[index - 1].FrequencyHz * HifiHnSpectralProfile.MinFrequencyRatio;
            double maximum = index == Bands.Count - 1
                ? HifiHnSpectralProfile.MaxFrequencyHz
                : Bands[index + 1].FrequencyHz / HifiHnSpectralProfile.MinFrequencyRatio;
            value = double.IsFinite(value) ? value : Bands[index].FrequencyHz;
            return Math.Clamp(value, minimum, maximum);
        }

        bool CanInsertAtFrequency(double frequencyHz) {
            int index = 0;
            while (index < Bands.Count && Bands[index].FrequencyHz < frequencyHz) {
                index++;
            }
            double minimum = index == 0
                ? HifiHnSpectralProfile.MinFrequencyHz
                : Bands[index - 1].FrequencyHz * HifiHnSpectralProfile.MinFrequencyRatio;
            double maximum = index == Bands.Count
                ? HifiHnSpectralProfile.MaxFrequencyHz
                : Bands[index].FrequencyHz / HifiHnSpectralProfile.MinFrequencyRatio;
            return frequencyHz >= minimum && frequencyHz <= maximum;
        }

        double FrequencyNearSelectedBand() {
            if (!HasSelectedBand) {
                return clipboard?.FrequencyHz ?? HifiHnSpectralProfile.DefaultFrequenciesHz[0];
            }
            double selected = Bands[SelectedBandIndex].FrequencyHz;
            if (SelectedBandIndex < Bands.Count - 1) {
                return Math.Sqrt(selected * Bands[SelectedBandIndex + 1].FrequencyHz);
            }
            if (SelectedBandIndex > 0) {
                return Math.Sqrt(Bands[SelectedBandIndex - 1].FrequencyHz * selected);
            }
            return Math.Clamp(
                selected * 1.5,
                HifiHnSpectralProfile.MinFrequencyHz,
                HifiHnSpectralProfile.MaxFrequencyHz);
        }

        void RunBandEdit(Action action) {
            bool ownsTransaction = bandEditDepth == 0;
            if (ownsTransaction) {
                BeginBandEdit();
            }
            try {
                action();
            } finally {
                if (ownsTransaction) {
                    CommitBandEdit();
                }
            }
        }

        T RunBandEdit<T>(Func<T> action) {
            bool ownsTransaction = bandEditDepth == 0;
            if (ownsTransaction) {
                BeginBandEdit();
            }
            try {
                return action();
            } finally {
                if (ownsTransaction) {
                    CommitBandEdit();
                }
            }
        }

        void FinishPendingBandEdit() {
            if (bandEditDepth <= 0) {
                return;
            }
            bandEditDepth = 1;
            CommitBandEdit();
        }

        BandSnapshot CaptureBandSnapshot() {
            return new BandSnapshot(
                Bands.Select(band => band.FrequencyHz).ToArray(),
                Bands.Select(band => band.BalanceDb).ToArray(),
                SelectedBandIndex);
        }

        void RestoreBandSnapshot(BandSnapshot snapshot) {
            updatingBands = true;
            try {
                foreach (var band in Bands) {
                    band.PropertyChanged -= OnBandPropertyChanged;
                }
                Bands.Clear();
                for (int i = 0; i < snapshot.FrequenciesHz.Length; i++) {
                    Bands.Add(new HifiHnBandViewModel(snapshot.FrequenciesHz[i], snapshot.BalanceDb[i]));
                }
            } finally {
                updatingBands = false;
            }
            SelectedBandIndex = Math.Clamp(snapshot.SelectedIndex, -1, Bands.Count - 1);
            RaiseSelectedBandProperties();
        }

        static bool BandSnapshotsEqual(BandSnapshot left, BandSnapshot right) {
            return left.FrequenciesHz.SequenceEqual(right.FrequenciesHz)
                && left.BalanceDb.SequenceEqual(right.BalanceDb);
        }

        static BandSnapshot PopHistory(System.Collections.Generic.List<BandSnapshot> history) {
            int index = history.Count - 1;
            BandSnapshot value = history[index];
            history.RemoveAt(index);
            return value;
        }

        static void PushHistory(System.Collections.Generic.List<BandSnapshot> history, BandSnapshot snapshot) {
            history.Add(snapshot);
            if (history.Count > MaxHistoryEntries) {
                history.RemoveAt(0);
            }
        }

        void RaiseHistoryProperties() {
            this.RaisePropertyChanged(nameof(CanUndoBandEdit));
            this.RaisePropertyChanged(nameof(CanRedoBandEdit));
        }

        void RaiseSelectedBandProperties() {
            this.RaisePropertyChanged(nameof(HasSelectedBand));
            this.RaisePropertyChanged(nameof(CanDeleteSelectedBand));
            this.RaisePropertyChanged(nameof(CanPasteBand));
            this.RaisePropertyChanged(nameof(SelectedBandOpacity));
            this.RaisePropertyChanged(nameof(SelectedBandTitle));
            this.RaisePropertyChanged(nameof(SelectedFrequencyHz));
            this.RaisePropertyChanged(nameof(SelectedBalanceDb));
        }
    }
}
