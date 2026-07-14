using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Media;
using OpenUtau.App.ViewModels;
using OpenUtau.Core.HifiNeural;

namespace OpenUtau.App.Controls {
    public sealed class HifiHnBalanceGraph : Control {
        public static readonly DirectProperty<HifiHnBalanceGraph, ObservableCollection<HifiHnBandViewModel>?> BandsProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, ObservableCollection<HifiHnBandViewModel>?>(
                nameof(Bands), control => control.Bands, (control, value) => control.Bands = value,
                defaultBindingMode: BindingMode.OneWay);
        public static readonly DirectProperty<HifiHnBalanceGraph, int> SelectedBandIndexProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, int>(
                nameof(SelectedBandIndex),
                control => control.SelectedBandIndex,
                (control, value) => control.SelectedBandIndex = value,
                defaultBindingMode: BindingMode.TwoWay);
        public static readonly StyledProperty<IBrush?> PlotBackgroundProperty =
            AvaloniaProperty.Register<HifiHnBalanceGraph, IBrush?>(nameof(PlotBackground));
        public static readonly StyledProperty<IBrush?> GridLineBrushProperty =
            AvaloniaProperty.Register<HifiHnBalanceGraph, IBrush?>(nameof(GridLineBrush));
        public static readonly StyledProperty<IBrush?> ZeroLineBrushProperty =
            AvaloniaProperty.Register<HifiHnBalanceGraph, IBrush?>(nameof(ZeroLineBrush));
        public static readonly StyledProperty<IBrush?> CurveBrushProperty =
            AvaloniaProperty.Register<HifiHnBalanceGraph, IBrush?>(nameof(CurveBrush));
        public static readonly StyledProperty<IBrush?> HarmonicBrushProperty =
            AvaloniaProperty.Register<HifiHnBalanceGraph, IBrush?>(nameof(HarmonicBrush));
        public static readonly StyledProperty<IBrush?> NoiseBrushProperty =
            AvaloniaProperty.Register<HifiHnBalanceGraph, IBrush?>(nameof(NoiseBrush));
        public static readonly StyledProperty<IBrush?> SelectionBrushProperty =
            AvaloniaProperty.Register<HifiHnBalanceGraph, IBrush?>(nameof(SelectionBrush));
        public static readonly StyledProperty<IBrush?> FocusGuideBrushProperty =
            AvaloniaProperty.Register<HifiHnBalanceGraph, IBrush?>(nameof(FocusGuideBrush));

        static readonly double[] GridFrequenciesHz = { 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000 };

        readonly HashSet<HifiHnBandViewModel> subscribedBands = new();
        ObservableCollection<HifiHnBandViewModel>? bands;
        ContextMenu? bandContextMenu;
        int activeBand = -1;
        int selectedBandIndex = -1;
        double dragOffsetX;
        double dragOffsetY;
        Point hoverPosition;
        bool pointerInside;

        public ObservableCollection<HifiHnBandViewModel>? Bands {
            get => bands;
            set {
                if (ReferenceEquals(bands, value)) {
                    return;
                }
                UnsubscribeBands();
                SetAndRaise(BandsProperty, ref bands, value);
                SubscribeBands();
                SelectedBandIndex = Math.Min(SelectedBandIndex, (Bands?.Count ?? 0) - 1);
                InvalidateVisual();
            }
        }
        public int SelectedBandIndex {
            get => selectedBandIndex;
            set {
                int maximum = (Bands?.Count ?? 0) - 1;
                int clamped = maximum < 0 ? -1 : Math.Clamp(value, -1, maximum);
                SetAndRaise(SelectedBandIndexProperty, ref selectedBandIndex, clamped);
            }
        }
        public IBrush? PlotBackground { get => GetValue(PlotBackgroundProperty); set => SetValue(PlotBackgroundProperty, value); }
        public IBrush? GridLineBrush { get => GetValue(GridLineBrushProperty); set => SetValue(GridLineBrushProperty, value); }
        public IBrush? ZeroLineBrush { get => GetValue(ZeroLineBrushProperty); set => SetValue(ZeroLineBrushProperty, value); }
        public IBrush? CurveBrush { get => GetValue(CurveBrushProperty); set => SetValue(CurveBrushProperty, value); }
        public IBrush? HarmonicBrush { get => GetValue(HarmonicBrushProperty); set => SetValue(HarmonicBrushProperty, value); }
        public IBrush? NoiseBrush { get => GetValue(NoiseBrushProperty); set => SetValue(NoiseBrushProperty, value); }
        public IBrush? SelectionBrush { get => GetValue(SelectionBrushProperty); set => SetValue(SelectionBrushProperty, value); }
        public IBrush? FocusGuideBrush { get => GetValue(FocusGuideBrushProperty); set => SetValue(FocusGuideBrushProperty, value); }

        public HifiHnBalanceGraph() {
            ClipToBounds = true;
            DoubleTapped += OnGraphDoubleTapped;
        }

        protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change) {
            base.OnPropertyChanged(change);
            InvalidateVisual();
        }

        protected override void OnPointerPressed(PointerPressedEventArgs e) {
            base.OnPointerPressed(e);
            var point = e.GetCurrentPoint(this);
            hoverPosition = point.Position;
            pointerInside = true;

            if (point.Properties.IsRightButtonPressed) {
                int hitBand = ClosestPoint(point.Position, out double distanceSquared);
                if (hitBand >= 0 && distanceSquared <= 28 * 28) {
                    SelectedBandIndex = hitBand;
                    OpenBandContextMenu();
                }
                e.Handled = true;
                return;
            }
            if (!point.Properties.IsLeftButtonPressed) {
                return;
            }
            activeBand = ClosestPoint(point.Position, out double leftDistanceSquared);
            if (activeBand < 0 || leftDistanceSquared > 28 * 28) {
                activeBand = -1;
                SelectedBandIndex = -1;
                e.Handled = true;
                return;
            }
            SelectedBandIndex = activeBand;
            Point bandPoint = PointForBand(activeBand);
            dragOffsetX = point.Position.X - bandPoint.X;
            dragOffsetY = point.Position.Y - bandPoint.Y;
            if (DataContext is HifiHnSpectralDesignerViewModel viewModel) {
                viewModel.BeginBandEdit();
            }
            e.Pointer.Capture(this);
            e.Handled = true;
        }

        protected override void OnPointerMoved(PointerEventArgs e) {
            base.OnPointerMoved(e);
            hoverPosition = e.GetPosition(this);
            pointerInside = true;
            InvalidateVisual();
            if (activeBand < 0 || e.Pointer.Captured != this) {
                return;
            }
            Point position = e.GetPosition(this);
            SetBandFromPosition(
                activeBand,
                new Point(position.X - dragOffsetX, position.Y - dragOffsetY));
            e.Handled = true;
        }

        protected override void OnPointerReleased(PointerReleasedEventArgs e) {
            base.OnPointerReleased(e);
            if (activeBand >= 0 && DataContext is HifiHnSpectralDesignerViewModel viewModel) {
                viewModel.CommitBandEdit();
            }
            activeBand = -1;
            if (e.Pointer.Captured == this) {
                e.Pointer.Capture(null);
            }
            dragOffsetX = 0;
            dragOffsetY = 0;
            InvalidateVisual();
        }

        protected override void OnPointerCaptureLost(PointerCaptureLostEventArgs e) {
            base.OnPointerCaptureLost(e);
            if (activeBand >= 0 && DataContext is HifiHnSpectralDesignerViewModel viewModel) {
                viewModel.CommitBandEdit();
            }
            activeBand = -1;
            dragOffsetX = 0;
            dragOffsetY = 0;
            InvalidateVisual();
        }

        protected override void OnPointerEntered(PointerEventArgs e) {
            base.OnPointerEntered(e);
            hoverPosition = e.GetPosition(this);
            pointerInside = true;
            InvalidateVisual();
        }

        protected override void OnPointerExited(PointerEventArgs e) {
            base.OnPointerExited(e);
            if (e.Pointer.Captured != this) {
                pointerInside = false;
                InvalidateVisual();
            }
        }

        public override void Render(DrawingContext context) {
            base.Render(context);
            var bounds = new Rect(0, 0, Bounds.Width, Bounds.Height);
            IBrush plotBackground = PlotBackground ?? Brushes.Transparent;
            IBrush gridBrush = GridLineBrush ?? Brushes.Gray;
            IBrush zeroBrush = ZeroLineBrush ?? Brushes.DimGray;
            IBrush curveBrush = CurveBrush ?? Brushes.DodgerBlue;
            IBrush harmonicBrush = HarmonicBrush ?? Brushes.DodgerBlue;
            IBrush noiseBrush = NoiseBrush ?? Brushes.DeepPink;
            IBrush selectionBrush = SelectionBrush ?? Brushes.Black;
            IBrush focusGuideBrush = FocusGuideBrush ?? gridBrush;
            context.DrawRectangle(plotBackground, null, bounds);
            if (Bounds.Width < 40 || Bounds.Height < 40) {
                return;
            }

            var plot = PlotBounds();
            var gridPen = new Pen(gridBrush, 1);
            var zeroPen = new Pen(zeroBrush, 1.4);
            for (int line = 0; line <= 4; line++) {
                double y = plot.Y + plot.Height * line / 4.0;
                context.DrawLine(line == 2 ? zeroPen : gridPen, new Point(plot.X, y), new Point(plot.Right, y));
            }
            foreach (double frequency in GridFrequenciesHz) {
                double x = FrequencyX(frequency, plot);
                if (x > plot.X + 1 && x < plot.Right - 1) {
                    context.DrawLine(gridPen, new Point(x, plot.Y), new Point(x, plot.Bottom));
                }
            }
            DrawHoverGuides(context, plot, focusGuideBrush);

            if (Bands == null || Bands.Count == 0) {
                return;
            }
            var points = new Point[Bands.Count];
            for (int i = 0; i < Bands.Count; i++) {
                points[i] = new Point(
                    FrequencyX(Bands[i].FrequencyHz, plot),
                    ValueY(Bands[i].BalancePercent, plot));
            }

            // The DSP holds the outer values to the plot edges. Keep those shelves visually
            // subordinate to the editable curve while still showing their real behavior.
            DrawEdgeShelf(
                context,
                plot.X,
                points[0].X,
                points[0].Y,
                fadeAtStart: true,
                curveBrush,
                SelectedBandIndex == 0,
                Bands[0].BalancePercent);
            DrawEdgeShelf(
                context,
                points[^1].X,
                plot.Right,
                points[^1].Y,
                fadeAtStart: false,
                curveBrush,
                SelectedBandIndex == Bands.Count - 1,
                Bands[^1].BalancePercent);

            var geometry = new PathGeometry();
            var figure = new PathFigure { StartPoint = points[0], IsClosed = false };
            for (int i = 0; i < points.Length - 1; i++) {
                double oneThird = points[i].X + (points[i + 1].X - points[i].X) / 3.0;
                double twoThirds = points[i].X + (points[i + 1].X - points[i].X) * 2.0 / 3.0;
                figure.Segments!.Add(new BezierSegment {
                    Point1 = new Point(oneThird, points[i].Y),
                    Point2 = new Point(twoThirds, points[i + 1].Y),
                    Point3 = points[i + 1],
                });
            }
            geometry.Figures!.Add(figure);
            context.DrawGeometry(null, new Pen(curveBrush, 2.5), geometry);

            for (int i = 0; i < points.Length; i++) {
                bool harmonic = Bands[i].BalancePercent >= 0;
                IBrush fill = harmonic ? harmonicBrush : noiseBrush;
                bool selected = i == SelectedBandIndex;
                if (selected) {
                    context.DrawEllipse(null, new Pen(selectionBrush, 2), points[i], 11, 11);
                }
                double radius = i == activeBand ? 9 : selected ? 8 : 6;
                context.DrawEllipse(fill, new Pen(plotBackground, 1.5), points[i], radius, radius);
            }
        }

        static void DrawEdgeShelf(
            DrawingContext context,
            double startX,
            double endX,
            double y,
            bool fadeAtStart,
            IBrush brush,
            bool selected,
            double valuePercent) {
            const double lineWidth = 1.6;
            const double edgeFadePixels = 28;
            const int fadeSegments = 8;
            double length = endX - startX;
            if (length <= 0.5) {
                return;
            }

            double maxOpacity = selected ? 0.62 : 0.40;
            double zeroLineFactor = Math.Clamp(Math.Abs(valuePercent) / 12.0, 0.55, 1.0);
            maxOpacity *= zeroLineFactor;
            double fadeLength = Math.Min(edgeFadePixels, length);
            double fadeStart = fadeAtStart ? startX : endX - fadeLength;
            double fadeEnd = fadeAtStart ? startX + fadeLength : endX;
            double solidStart = fadeAtStart ? fadeEnd : startX;
            double solidEnd = fadeAtStart ? endX : fadeStart;

            if (solidEnd - solidStart > 0.5) {
                using (context.PushOpacity(maxOpacity)) {
                    context.DrawLine(
                        new Pen(brush, lineWidth),
                        new Point(solidStart, y),
                        new Point(solidEnd, y));
                }
            }

            for (int i = 0; i < fadeSegments; i++) {
                double t0 = i / (double)fadeSegments;
                double t1 = (i + 1) / (double)fadeSegments;
                double x0 = fadeStart + (fadeEnd - fadeStart) * t0;
                double x1 = fadeStart + (fadeEnd - fadeStart) * t1;
                double towardCurve = fadeAtStart
                    ? (t0 + t1) * 0.5
                    : 1.0 - (t0 + t1) * 0.5;
                double smoothOpacity = towardCurve * towardCurve * (3.0 - 2.0 * towardCurve);
                using (context.PushOpacity(maxOpacity * smoothOpacity)) {
                    context.DrawLine(
                        new Pen(brush, lineWidth),
                        new Point(x0, y),
                        new Point(x1, y));
                }
            }
        }

        void OnGraphDoubleTapped(object? sender, TappedEventArgs e) {
            if (DataContext is not HifiHnSpectralDesignerViewModel viewModel
                || Bands == null
                || Bands.Count == 0
                || Bands.Count >= HifiHnSpectralProfile.MaxBandCount) {
                return;
            }
            Point position = e.GetPosition(this);
            Rect plot = PlotBounds();
            if (!plot.Contains(position)) {
                return;
            }
            int nearest = ClosestPoint(position, out double pointDistanceSquared);
            if (nearest >= 0 && pointDistanceSquared <= 18 * 18) {
                SelectedBandIndex = nearest;
                e.Handled = true;
                return;
            }

            double frequency = FrequencyAtX(position.X, plot);
            double curveValue = InterpolateCurveValue(frequency);
            if (Math.Abs(position.Y - ValueY(curveValue, plot)) > 18) {
                return;
            }
            int added = viewModel.AddBand(Math.Round(frequency), curveValue);
            if (added >= 0) {
                SelectedBandIndex = added;
                e.Handled = true;
            }
        }

        void OpenBandContextMenu() {
            if (DataContext is not HifiHnSpectralDesignerViewModel viewModel) {
                return;
            }
            viewModel.SelectedBandIndex = SelectedBandIndex;
            var reset = CreateMenuItem("hifi.hn.zero", viewModel.ResetSelectedBalance);
            var cut = CreateMenuItem("menu.edit.cut", viewModel.CutSelectedBand, viewModel.CanDeleteSelectedBand);
            var copy = CreateMenuItem("menu.edit.copy", viewModel.CopySelectedBand);
            var paste = CreateMenuItem("menu.edit.paste", viewModel.PasteBand, viewModel.CanPasteBand);
            var delete = CreateMenuItem("menu.edit.delete", viewModel.DeleteSelectedBand, viewModel.CanDeleteSelectedBand);
            bandContextMenu = new ContextMenu {
                Placement = PlacementMode.Pointer,
                ItemsSource = new object[] {
                    reset,
                    new Separator(),
                    cut,
                    copy,
                    paste,
                    new Separator(),
                    delete,
                },
            };
            bandContextMenu.Open(this);
        }

        static MenuItem CreateMenuItem(string resourceKey, Action action, bool enabled = true) {
            var item = new MenuItem {
                Header = ThemeManager.GetString(resourceKey),
                IsEnabled = enabled,
            };
            item.Click += (_, _) => action();
            return item;
        }

        int ClosestPoint(Point pointer, out double distanceSquared) {
            int closest = -1;
            distanceSquared = double.MaxValue;
            if (Bands == null) {
                return closest;
            }
            for (int i = 0; i < Bands.Count; i++) {
                Point candidate = PointForBand(i);
                double dx = pointer.X - candidate.X;
                double dy = pointer.Y - candidate.Y;
                double candidateDistance = dx * dx + dy * dy;
                if (candidateDistance < distanceSquared) {
                    distanceSquared = candidateDistance;
                    closest = i;
                }
            }
            return closest;
        }

        Point PointForBand(int band) {
            if (Bands == null || band < 0 || band >= Bands.Count) {
                return default;
            }
            var plot = PlotBounds();
            return new Point(
                FrequencyX(Bands[band].FrequencyHz, plot),
                ValueY(Bands[band].BalancePercent, plot));
        }

        void DrawHoverGuides(DrawingContext context, Rect plot, IBrush brush) {
            if (!pointerInside || !plot.Contains(hoverPosition)) {
                return;
            }
            const double radiusX = 116;
            const double radiusY = 88;
            const double dotSpacing = 12;
            double left = Math.Max(plot.X, hoverPosition.X - radiusX);
            double right = Math.Min(plot.Right, hoverPosition.X + radiusX);
            double top = Math.Max(plot.Y, hoverPosition.Y - radiusY);
            double bottom = Math.Min(plot.Bottom, hoverPosition.Y + radiusY);
            double startX = Math.Ceiling(left / dotSpacing) * dotSpacing;
            double startY = Math.Ceiling(top / dotSpacing) * dotSpacing;
            for (double x = startX; x <= right; x += dotSpacing) {
                for (double y = startY; y <= bottom; y += dotSpacing) {
                    double normalizedX = (x - hoverPosition.X) / radiusX;
                    double normalizedY = (y - hoverPosition.Y) / radiusY;
                    double distance = Math.Sqrt(normalizedX * normalizedX + normalizedY * normalizedY);
                    double angle = Math.Atan2(normalizedY, normalizedX);
                    double boundary = 0.92
                        + Math.Sin(angle * 3 + 0.8) * 0.07
                        + Math.Sin(angle * 5 - 1.1) * 0.045;
                    int column = (int)Math.Round(x / dotSpacing);
                    int row = (int)Math.Round(y / dotSpacing);
                    boundary += (Hash01(column, row, 0) - 0.5) * 0.07;
                    if (distance >= boundary) {
                        continue;
                    }

                    double fade = Math.Clamp((boundary - distance) / 0.48, 0, 1);
                    fade = fade * fade * (3 - 2 * fade);
                    double dropout = 0.05 + (1 - fade) * 0.25;
                    if (Hash01(column, row, 1) < dropout) {
                        continue;
                    }

                    double opacity = fade * (0.42 + Hash01(column, row, 2) * 0.20);
                    double edgeScale = 0.38 + Math.Sqrt(fade) * 0.62;
                    double radius = edgeScale * (0.78 + Hash01(column, row, 3) * 0.42);
                    double horizontalScale = 0.72 + Hash01(column, row, 4) * 0.28;
                    double verticalScale = 0.68 + Hash01(column, row, 5) * 0.32;
                    using (context.PushOpacity(opacity)) {
                        context.DrawEllipse(
                            brush,
                            null,
                            new Point(x, y),
                            radius * horizontalScale,
                            radius * verticalScale);
                    }
                }
            }
        }

        static double Hash01(int x, int y, int salt) {
            uint value = unchecked((uint)(x * 374761393 + y * 668265263 + salt * 1442695041));
            value = (value ^ (value >> 13)) * 1274126177u;
            value ^= value >> 16;
            return value / (double)uint.MaxValue;
        }

        void SetBandFromPosition(int band, Point position) {
            if (Bands == null || band < 0 || band >= Bands.Count) {
                return;
            }
            var plot = PlotBounds();
            double frequency = FrequencyAtX(position.X, plot);
            Bands[band].FrequencyHz = Math.Round(ClampFrequencyForBand(band, frequency));
            double value = (plot.Center.Y - position.Y) / (plot.Height * 0.5)
                * HifiHnSpectralProfile.MaxBalancePercent;
            Bands[band].BalancePercent = Math.Round(ClampPercent(value));
        }

        double ClampFrequencyForBand(int band, double value) {
            if (Bands == null || band < 0 || band >= Bands.Count) {
                return HifiHnSpectralProfile.MinFrequencyHz;
            }
            double minimum = band == 0
                ? HifiHnSpectralProfile.MinFrequencyHz
                : Bands[band - 1].FrequencyHz * HifiHnSpectralProfile.MinFrequencyRatio;
            double maximum = band == Bands.Count - 1
                ? HifiHnSpectralProfile.MaxFrequencyHz
                : Bands[band + 1].FrequencyHz / HifiHnSpectralProfile.MinFrequencyRatio;
            return Math.Clamp(value, minimum, maximum);
        }

        double InterpolateCurveValue(double frequency) {
            if (Bands == null || Bands.Count == 0) {
                return 0;
            }
            if (frequency <= Bands[0].FrequencyHz) {
                return Bands[0].BalancePercent;
            }
            if (frequency >= Bands[^1].FrequencyHz) {
                return Bands[^1].BalancePercent;
            }
            double logFrequency = Math.Log(frequency);
            for (int i = 0; i < Bands.Count - 1; i++) {
                if (frequency <= Bands[i + 1].FrequencyHz) {
                    double left = Math.Log(Bands[i].FrequencyHz);
                    double right = Math.Log(Bands[i + 1].FrequencyHz);
                    double t = Math.Clamp((logFrequency - left) / Math.Max(1e-9, right - left), 0, 1);
                    t = t * t * (3 - 2 * t);
                    return Bands[i].BalancePercent + (Bands[i + 1].BalancePercent - Bands[i].BalancePercent) * t;
                }
            }
            return Bands[^1].BalancePercent;
        }

        void SubscribeBands() {
            if (Bands == null) {
                return;
            }
            Bands.CollectionChanged += OnBandsCollectionChanged;
            RefreshBandSubscriptions();
        }

        void UnsubscribeBands() {
            if (Bands != null) {
                Bands.CollectionChanged -= OnBandsCollectionChanged;
            }
            foreach (var band in subscribedBands) {
                band.PropertyChanged -= OnBandPropertyChanged;
            }
            subscribedBands.Clear();
        }

        void OnBandsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e) {
            RefreshBandSubscriptions();
            if (activeBand >= (Bands?.Count ?? 0)) {
                activeBand = -1;
            }
            SelectedBandIndex = Math.Min(SelectedBandIndex, (Bands?.Count ?? 0) - 1);
            InvalidateVisual();
        }

        void RefreshBandSubscriptions() {
            foreach (var band in subscribedBands) {
                band.PropertyChanged -= OnBandPropertyChanged;
            }
            subscribedBands.Clear();
            if (Bands == null) {
                return;
            }
            foreach (var band in Bands) {
                band.PropertyChanged += OnBandPropertyChanged;
                subscribedBands.Add(band);
            }
        }

        void OnBandPropertyChanged(object? sender, PropertyChangedEventArgs e) {
            InvalidateVisual();
        }

        Rect PlotBounds() => new(
            24,
            14,
            Math.Max(1, Bounds.Width - 48),
            Math.Max(1, Bounds.Height - 28));

        static double FrequencyAtX(double x, Rect plot) {
            double normalizedX = Math.Clamp((x - plot.X) / plot.Width, 0, 1);
            double logMin = Math.Log(HifiHnSpectralProfile.MinFrequencyHz);
            double logMax = Math.Log(HifiHnSpectralProfile.MaxFrequencyHz);
            return Math.Exp(logMin + normalizedX * (logMax - logMin));
        }

        static double FrequencyX(double frequency, Rect plot) {
            double min = Math.Log(HifiHnSpectralProfile.MinFrequencyHz);
            double max = Math.Log(HifiHnSpectralProfile.MaxFrequencyHz);
            frequency = Math.Clamp(frequency, HifiHnSpectralProfile.MinFrequencyHz, HifiHnSpectralProfile.MaxFrequencyHz);
            return plot.X + (Math.Log(frequency) - min) / (max - min) * plot.Width;
        }

        static double ValueY(double value, Rect plot) {
            return plot.Center.Y - ClampPercent(value) / HifiHnSpectralProfile.MaxBalancePercent * plot.Height * 0.5;
        }

        static double ClampPercent(double value) {
            return double.IsFinite(value)
                ? Math.Clamp(value, -HifiHnSpectralProfile.MaxBalancePercent, HifiHnSpectralProfile.MaxBalancePercent)
                : 0;
        }
    }
}
