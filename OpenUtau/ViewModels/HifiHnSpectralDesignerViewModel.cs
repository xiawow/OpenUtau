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
        double balancePercent;

        public double FrequencyHz {
            get => frequencyHz;
            set => this.RaiseAndSetIfChanged(
                ref frequencyHz,
                double.IsFinite(value)
                    ? Math.Clamp(value, HifiHnSpectralProfile.MinFrequencyHz, HifiHnSpectralProfile.MaxFrequencyHz)
                    : HifiHnSpectralProfile.MinFrequencyHz);
        }

        public double BalancePercent {
            get => balancePercent;
            set => this.RaiseAndSetIfChanged(
                ref balancePercent,
                double.IsFinite(value)
                    ? Math.Clamp(value, -HifiHnSpectralProfile.MaxBalancePercent, HifiHnSpectralProfile.MaxBalancePercent)
                    : 0);
        }

        public HifiHnBandViewModel(double frequencyHz, double balancePercent) {
            this.frequencyHz = double.IsFinite(frequencyHz)
                ? Math.Clamp(frequencyHz, HifiHnSpectralProfile.MinFrequencyHz, HifiHnSpectralProfile.MaxFrequencyHz)
                : HifiHnSpectralProfile.MinFrequencyHz;
            this.balancePercent = double.IsFinite(balancePercent)
                ? Math.Clamp(balancePercent, -HifiHnSpectralProfile.MaxBalancePercent, HifiHnSpectralProfile.MaxBalancePercent)
                : 0;
        }
    }

    public sealed class HifiHnSpectralDesignerViewModel : ViewModelBase {
        readonly record struct BandClipboard(double FrequencyHz, double BalancePercent);
        sealed record ProfileSnapshot(
            bool Enabled,
            double[] FrequenciesHz,
            double[] BalancePercent,
            int SelectedIndex,
            bool DynamicsEnabled,
            int DynamicsTargetIndex,
            double ThresholdDb,
            double Ratio,
            double AttackMs,
            double ReleaseMs,
            double MaxReductionDb);
        const int MaxHistoryEntries = 64;
        static BandClipboard? clipboard;

        readonly int noteCount;
        readonly System.Collections.Generic.List<ProfileSnapshot> undoHistory = new();
        readonly System.Collections.Generic.List<ProfileSnapshot> redoHistory = new();
        bool updatingBands;
        bool loadingProfile;
        int bandEditDepth;
        ProfileSnapshot? pendingBandEdit;
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
        public event EventHandler? ProfileChanged;
        public bool IsBandEditInProgress => bandEditDepth > 0;

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
        public double SelectedBalancePercent {
            get => HasSelectedBand ? Bands[SelectedBandIndex].BalancePercent : 0;
            set {
                if (HasSelectedBand) {
                    RunBandEdit(() => Bands[SelectedBandIndex].BalancePercent = value);
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
            PropertyChanged += OnProfilePropertyChanged;
        }

        public void Reset() {
            FinishPendingBandEdit();
            ProfileSnapshot before = CaptureProfileSnapshot();
            ProfileSnapshot defaults = SnapshotFromProfile(new HifiHnSpectralProfile(), -1);
            if (ProfileSnapshotsEqual(before, defaults)) {
                SelectedBandIndex = -1;
                return;
            }
            PushHistory(undoHistory, before);
            redoHistory.Clear();
            RestoreProfileSnapshot(defaults);
            RaiseHistoryProperties();
        }

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
            SelectedBalancePercent = 0;
        }

        public void ToggleDynamicsPanel() {
            DynamicsPanelExpanded = !DynamicsPanelExpanded;
        }

        public int AddBand(double frequencyHz, double balancePercent) {
            return RunBandEdit(() => AddBandCore(frequencyHz, balancePercent));
        }

        int AddBandCore(double frequencyHz, double balancePercent) {
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
            Bands.Insert(bestIndex, new HifiHnBandViewModel(bestFrequency, balancePercent));
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
                clipboard = new BandClipboard(band.FrequencyHz, band.BalancePercent);
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
            AddBand(frequency, value.BalancePercent);
        }

        public void BeginBandEdit() {
            if (bandEditDepth++ == 0) {
                pendingBandEdit = CaptureProfileSnapshot();
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
            ProfileSnapshot? before = pendingBandEdit;
            pendingBandEdit = null;
            if (before != null && !ProfileSnapshotsEqual(before, CaptureProfileSnapshot())) {
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
            ProfileSnapshot target = PopHistory(undoHistory);
            PushHistory(redoHistory, CaptureProfileSnapshot());
            RestoreProfileSnapshot(target);
            RaiseHistoryProperties();
        }

        public void RedoBandEdit() {
            FinishPendingBandEdit();
            if (redoHistory.Count == 0) {
                return;
            }
            ProfileSnapshot target = PopHistory(redoHistory);
            PushHistory(undoHistory, CaptureProfileSnapshot());
            RestoreProfileSnapshot(target);
            RaiseHistoryProperties();
        }

        public HifiHnSpectralProfile BuildProfile() {
            return new HifiHnSpectralProfile {
                Enabled = Enabled,
                BalanceDb = Bands
                    .Select(band => HifiHnSpectralProfile.PercentToBalanceDb(band.BalancePercent))
                    .ToArray(),
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
            loadingProfile = true;
            try {
                Enabled = profile.Enabled;
                updatingBands = true;
                try {
                    foreach (var band in Bands) {
                        band.PropertyChanged -= OnBandPropertyChanged;
                    }
                    Bands.Clear();
                    for (int i = 0; i < profile.FrequenciesHz.Length; i++) {
                        Bands.Add(new HifiHnBandViewModel(
                            profile.FrequenciesHz[i],
                            HifiHnSpectralProfile.BalanceDbToPercent(profile.BalanceDb[i])));
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
            } finally {
                loadingProfile = false;
            }
            RaiseProfileChanged();
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
            if (!updatingBands) {
                RaiseProfileChanged();
            }
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
            if (!updatingBands) {
                RaiseProfileChanged();
            }
        }

        void OnProfilePropertyChanged(object? sender, PropertyChangedEventArgs e) {
            if (e.PropertyName is nameof(Enabled)
                or nameof(DynamicsEnabled)
                or nameof(DynamicsTargetIndex)
                or nameof(ThresholdDb)
                or nameof(Ratio)
                or nameof(AttackMs)
                or nameof(ReleaseMs)
                or nameof(MaxReductionDb)) {
                RaiseProfileChanged();
            }
        }

        void RaiseProfileChanged() {
            if (!loadingProfile) {
                ProfileChanged?.Invoke(this, EventArgs.Empty);
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

        ProfileSnapshot CaptureProfileSnapshot() {
            return new ProfileSnapshot(
                Enabled,
                Bands.Select(band => band.FrequencyHz).ToArray(),
                Bands.Select(band => band.BalancePercent).ToArray(),
                SelectedBandIndex,
                DynamicsEnabled,
                DynamicsTargetIndex,
                ThresholdDb,
                Ratio,
                AttackMs,
                ReleaseMs,
                MaxReductionDb);
        }

        static ProfileSnapshot SnapshotFromProfile(HifiHnSpectralProfile profile, int selectedIndex) {
            profile = profile.Clone().Normalize();
            return new ProfileSnapshot(
                profile.Enabled,
                (double[])profile.FrequenciesHz.Clone(),
                profile.BalanceDb.Select(HifiHnSpectralProfile.BalanceDbToPercent).ToArray(),
                selectedIndex,
                profile.DynamicsEnabled,
                (int)profile.DynamicsTarget,
                profile.ThresholdDb,
                profile.Ratio,
                profile.AttackMs,
                profile.ReleaseMs,
                profile.MaxReductionDb);
        }

        void RestoreProfileSnapshot(ProfileSnapshot snapshot) {
            loadingProfile = true;
            try {
                Enabled = snapshot.Enabled;
                updatingBands = true;
                try {
                    foreach (var band in Bands) {
                        band.PropertyChanged -= OnBandPropertyChanged;
                    }
                    Bands.Clear();
                    for (int i = 0; i < snapshot.FrequenciesHz.Length; i++) {
                        Bands.Add(new HifiHnBandViewModel(snapshot.FrequenciesHz[i], snapshot.BalancePercent[i]));
                    }
                } finally {
                    updatingBands = false;
                }
                SelectedBandIndex = Math.Clamp(snapshot.SelectedIndex, -1, Bands.Count - 1);
                DynamicsEnabled = snapshot.DynamicsEnabled;
                DynamicsTargetIndex = snapshot.DynamicsTargetIndex;
                ThresholdDb = snapshot.ThresholdDb;
                Ratio = snapshot.Ratio;
                AttackMs = snapshot.AttackMs;
                ReleaseMs = snapshot.ReleaseMs;
                MaxReductionDb = snapshot.MaxReductionDb;
            } finally {
                loadingProfile = false;
            }
            RaiseSelectedBandProperties();
            RaiseProfileChanged();
        }

        static bool ProfileSnapshotsEqual(ProfileSnapshot left, ProfileSnapshot right) {
            return left.Enabled == right.Enabled
                && left.FrequenciesHz.SequenceEqual(right.FrequenciesHz)
                && left.BalancePercent.SequenceEqual(right.BalancePercent)
                && left.DynamicsEnabled == right.DynamicsEnabled
                && left.DynamicsTargetIndex == right.DynamicsTargetIndex
                && left.ThresholdDb == right.ThresholdDb
                && left.Ratio == right.Ratio
                && left.AttackMs == right.AttackMs
                && left.ReleaseMs == right.ReleaseMs
                && left.MaxReductionDb == right.MaxReductionDb;
        }

        static ProfileSnapshot PopHistory(System.Collections.Generic.List<ProfileSnapshot> history) {
            int index = history.Count - 1;
            ProfileSnapshot value = history[index];
            history.RemoveAt(index);
            return value;
        }

        static void PushHistory(System.Collections.Generic.List<ProfileSnapshot> history, ProfileSnapshot snapshot) {
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
            this.RaisePropertyChanged(nameof(SelectedBalancePercent));
        }
    }
}
