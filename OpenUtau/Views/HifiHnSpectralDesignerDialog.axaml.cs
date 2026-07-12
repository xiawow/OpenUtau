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
            } else {
                base.OnKeyDown(e);
            }
        }
    }
}
