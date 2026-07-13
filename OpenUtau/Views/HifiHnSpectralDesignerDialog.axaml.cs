using System;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using OpenUtau.App.ViewModels;
using OpenUtau.Core.HifiNeural;

namespace OpenUtau.App.Views {
    public partial class HifiHnSpectralDesignerDialog : Window {
        public event EventHandler? ApplyRequested;
        public HifiHnSpectralProfile ResultProfile => ViewModel.BuildProfile();
        HifiHnSpectralDesignerViewModel ViewModel => (HifiHnSpectralDesignerViewModel)DataContext!;

        public HifiHnSpectralDesignerDialog() {
            InitializeComponent();
        }

        void OnReset(object? sender, RoutedEventArgs e) => ViewModel.Reset();
        void OnPreviousBand(object? sender, RoutedEventArgs e) => ViewModel.SelectPreviousBand();
        void OnNextBand(object? sender, RoutedEventArgs e) => ViewModel.SelectNextBand();
        void OnResetSelectedBand(object? sender, RoutedEventArgs e) => ViewModel.ResetSelectedBalance();
        void OnToggleDynamics(object? sender, RoutedEventArgs e) => ViewModel.ToggleDynamicsPanel();
        void OnClose(object? sender, RoutedEventArgs e) => Close();

        void OnApply(object? sender, RoutedEventArgs e) {
            ApplyRequested?.Invoke(this, EventArgs.Empty);
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
