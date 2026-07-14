using System;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Threading;
using OpenUtau.App.ViewModels;
using OpenUtau.Core.HifiNeural;

namespace OpenUtau.App.Views {
    public partial class HifiHnSpectralDesignerDialog : Window {
        static readonly TimeSpan AutoApplyDelay = TimeSpan.FromMilliseconds(500);
        static readonly TimeSpan ActiveEditPollDelay = TimeSpan.FromMilliseconds(75);

        readonly DispatcherTimer autoApplyTimer;
        HifiHnSpectralDesignerViewModel? subscribedViewModel;
        bool hasPendingChanges;

        public event EventHandler? ApplyRequested;
        public HifiHnSpectralProfile ResultProfile => ViewModel.BuildProfile();
        HifiHnSpectralDesignerViewModel ViewModel => (HifiHnSpectralDesignerViewModel)DataContext!;

        public HifiHnSpectralDesignerDialog() {
            InitializeComponent();
            autoApplyTimer = new DispatcherTimer(
                AutoApplyDelay,
                DispatcherPriority.Background,
                OnAutoApplyTimerTick);
            DataContextChanged += OnDataContextChanged;
            Deactivated += OnDeactivated;
            Closing += OnClosing;
        }

        void OnReset(object? sender, RoutedEventArgs e) => ViewModel.Reset();
        void OnPreviousBand(object? sender, RoutedEventArgs e) => ViewModel.SelectPreviousBand();
        void OnNextBand(object? sender, RoutedEventArgs e) => ViewModel.SelectNextBand();
        void OnResetSelectedBand(object? sender, RoutedEventArgs e) => ViewModel.ResetSelectedBalance();
        void OnToggleDynamics(object? sender, RoutedEventArgs e) => ViewModel.ToggleDynamicsPanel();
        void OnClose(object? sender, RoutedEventArgs e) => Close();

        void OnApply(object? sender, RoutedEventArgs e) {
            FlushPendingChanges(force: true, allowDuringEdit: true);
        }

        void OnDataContextChanged(object? sender, EventArgs e) {
            if (subscribedViewModel != null) {
                subscribedViewModel.ProfileChanged -= OnProfileChanged;
            }
            subscribedViewModel = DataContext as HifiHnSpectralDesignerViewModel;
            if (subscribedViewModel != null) {
                subscribedViewModel.ProfileChanged += OnProfileChanged;
            }
            hasPendingChanges = false;
            autoApplyTimer.Stop();
        }

        void OnProfileChanged(object? sender, EventArgs e) {
            hasPendingChanges = true;
            ScheduleAutoApply(AutoApplyDelay);
        }

        void OnAutoApplyTimerTick(object? sender, EventArgs e) {
            autoApplyTimer.Stop();
            FlushPendingChanges(force: false, allowDuringEdit: false);
        }

        void OnDeactivated(object? sender, EventArgs e) {
            FlushPendingChanges(force: false, allowDuringEdit: false);
        }

        void OnClosing(object? sender, WindowClosingEventArgs e) {
            FlushPendingChanges(force: false, allowDuringEdit: true);
        }

        void ScheduleAutoApply(TimeSpan delay) {
            autoApplyTimer.Stop();
            autoApplyTimer.Interval = delay;
            autoApplyTimer.Start();
        }

        void FlushPendingChanges(bool force, bool allowDuringEdit) {
            if (subscribedViewModel == null || (!force && !hasPendingChanges)) {
                return;
            }
            if (!allowDuringEdit && subscribedViewModel.IsBandEditInProgress) {
                ScheduleAutoApply(ActiveEditPollDelay);
                return;
            }
            autoApplyTimer.Stop();
            hasPendingChanges = false;
            ApplyRequested?.Invoke(this, EventArgs.Empty);
        }

        protected override void OnClosed(EventArgs e) {
            autoApplyTimer.Stop();
            if (subscribedViewModel != null) {
                subscribedViewModel.ProfileChanged -= OnProfileChanged;
                subscribedViewModel = null;
            }
            base.OnClosed(e);
        }

        protected override void OnKeyDown(KeyEventArgs e) {
            if (e.Key == Key.Escape) {
                e.Handled = true;
                Close();
                return;
            }
            if (FocusManager?.GetFocusedElement() is TextBox) {
                base.OnKeyDown(e);
                return;
            }

            bool shortcut = e.KeyModifiers.HasFlag(KeyModifiers.Control)
                || e.KeyModifiers.HasFlag(KeyModifiers.Meta);
            bool shift = e.KeyModifiers.HasFlag(KeyModifiers.Shift);
            if (shortcut && e.Key == Key.C && ViewModel.HasSelectedBand) {
                ViewModel.CopySelectedBand();
                e.Handled = true;
            } else if (shortcut && e.Key == Key.X && ViewModel.CanDeleteSelectedBand) {
                ViewModel.CutSelectedBand();
                e.Handled = true;
            } else if (shortcut && e.Key == Key.V && ViewModel.CanPasteBand) {
                ViewModel.PasteBand();
                e.Handled = true;
            } else if (shortcut && e.Key == Key.Z && shift && ViewModel.CanRedoBandEdit) {
                ViewModel.RedoBandEdit();
                e.Handled = true;
            } else if (shortcut && e.Key == Key.Z && ViewModel.CanUndoBandEdit) {
                ViewModel.UndoBandEdit();
                e.Handled = true;
            } else if (shortcut && e.Key == Key.Y && ViewModel.CanRedoBandEdit) {
                ViewModel.RedoBandEdit();
                e.Handled = true;
            } else if (!shortcut && e.Key == Key.Delete && ViewModel.CanDeleteSelectedBand) {
                ViewModel.DeleteSelectedBand();
                e.Handled = true;
            }
            if (!e.Handled) {
                base.OnKeyDown(e);
            }
        }
    }
}
