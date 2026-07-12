using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Media;
using OpenUtau.Core.HifiNeural;

namespace OpenUtau.App.Controls {
    public sealed class HifiHnBalanceGraph : Control {
        public static readonly DirectProperty<HifiHnBalanceGraph, double> BodyDbProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, double>(
                nameof(BodyDb), control => control.BodyDb, (control, value) => control.BodyDb = value,
                defaultBindingMode: BindingMode.TwoWay);
        public static readonly DirectProperty<HifiHnBalanceGraph, double> WarmthDbProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, double>(
                nameof(WarmthDb), control => control.WarmthDb, (control, value) => control.WarmthDb = value,
                defaultBindingMode: BindingMode.TwoWay);
        public static readonly DirectProperty<HifiHnBalanceGraph, double> PresenceDbProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, double>(
                nameof(PresenceDb), control => control.PresenceDb, (control, value) => control.PresenceDb = value,
                defaultBindingMode: BindingMode.TwoWay);
        public static readonly DirectProperty<HifiHnBalanceGraph, double> ClarityDbProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, double>(
                nameof(ClarityDb), control => control.ClarityDb, (control, value) => control.ClarityDb = value,
                defaultBindingMode: BindingMode.TwoWay);
        public static readonly DirectProperty<HifiHnBalanceGraph, double> AirDbProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, double>(
                nameof(AirDb), control => control.AirDb, (control, value) => control.AirDb = value,
                defaultBindingMode: BindingMode.TwoWay);
        public static readonly DirectProperty<HifiHnBalanceGraph, double> BodyHzProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, double>(
                nameof(BodyHz), control => control.BodyHz, (control, value) => control.BodyHz = value,
                defaultBindingMode: BindingMode.TwoWay);
        public static readonly DirectProperty<HifiHnBalanceGraph, double> WarmthHzProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, double>(
                nameof(WarmthHz), control => control.WarmthHz, (control, value) => control.WarmthHz = value,
                defaultBindingMode: BindingMode.TwoWay);
        public static readonly DirectProperty<HifiHnBalanceGraph, double> PresenceHzProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, double>(
                nameof(PresenceHz), control => control.PresenceHz, (control, value) => control.PresenceHz = value,
                defaultBindingMode: BindingMode.TwoWay);
        public static readonly DirectProperty<HifiHnBalanceGraph, double> ClarityHzProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, double>(
                nameof(ClarityHz), control => control.ClarityHz, (control, value) => control.ClarityHz = value,
                defaultBindingMode: BindingMode.TwoWay);
        public static readonly DirectProperty<HifiHnBalanceGraph, double> AirHzProperty =
            AvaloniaProperty.RegisterDirect<HifiHnBalanceGraph, double>(
                nameof(AirHz), control => control.AirHz, (control, value) => control.AirHz = value,
                defaultBindingMode: BindingMode.TwoWay);
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

        double bodyDb;
        double warmthDb;
        double presenceDb;
        double clarityDb;
        double airDb;
        double bodyHz = HifiHnSpectralProfile.DefaultFrequenciesHz[0];
        double warmthHz = HifiHnSpectralProfile.DefaultFrequenciesHz[1];
        double presenceHz = HifiHnSpectralProfile.DefaultFrequenciesHz[2];
        double clarityHz = HifiHnSpectralProfile.DefaultFrequenciesHz[3];
        double airHz = HifiHnSpectralProfile.DefaultFrequenciesHz[4];
        int activeBand = -1;
        int selectedBandIndex = -1;
        double dragOffsetX;
        double dragOffsetY;
        Point hoverPosition;
        bool pointerInside;

        public double BodyDb { get => bodyDb; set => SetAndRaise(BodyDbProperty, ref bodyDb, ClampDb(value)); }
        public double WarmthDb { get => warmthDb; set => SetAndRaise(WarmthDbProperty, ref warmthDb, ClampDb(value)); }
        public double PresenceDb { get => presenceDb; set => SetAndRaise(PresenceDbProperty, ref presenceDb, ClampDb(value)); }
        public double ClarityDb { get => clarityDb; set => SetAndRaise(ClarityDbProperty, ref clarityDb, ClampDb(value)); }
        public double AirDb { get => airDb; set => SetAndRaise(AirDbProperty, ref airDb, ClampDb(value)); }
        public double BodyHz { get => bodyHz; set => SetFrequency(BodyHzProperty, ref bodyHz, value, 0); }
        public double WarmthHz { get => warmthHz; set => SetFrequency(WarmthHzProperty, ref warmthHz, value, 1); }
        public double PresenceHz { get => presenceHz; set => SetFrequency(PresenceHzProperty, ref presenceHz, value, 2); }
        public double ClarityHz { get => clarityHz; set => SetFrequency(ClarityHzProperty, ref clarityHz, value, 3); }
        public double AirHz { get => airHz; set => SetFrequency(AirHzProperty, ref airHz, value, 4); }
        public int SelectedBandIndex {
            get => selectedBandIndex;
            set => SetAndRaise(
                SelectedBandIndexProperty,
                ref selectedBandIndex,
                Math.Clamp(value, -1, HifiHnSpectralProfile.BandCount - 1));
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
            Cursor = new Cursor(StandardCursorType.SizeAll);
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
            if (!point.Properties.IsLeftButtonPressed) {
                return;
            }
            activeBand = ClosestPoint(point.Position, out double distanceSquared);
            if (distanceSquared > 28 * 28) {
                activeBand = -1;
                SelectedBandIndex = -1;
                e.Handled = true;
                return;
            }
            SelectedBandIndex = activeBand;
            Point bandPoint = PointForBand(activeBand);
            dragOffsetX = point.Position.X - bandPoint.X;
            dragOffsetY = point.Position.Y - bandPoint.Y;
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
            if (e.Pointer.Captured == this) {
                e.Pointer.Capture(null);
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

            var values = Values();
            var frequencies = Frequencies();
            var points = new Point[values.Length];
            for (int i = 0; i < values.Length; i++) {
                points[i] = new Point(FrequencyX(frequencies[i], plot), ValueY(values[i], plot));
            }

            var geometry = new PathGeometry();
            var figure = new PathFigure { StartPoint = points[0], IsClosed = false };
            for (int i = 0; i < points.Length - 1; i++) {
                double midpoint = (points[i].X + points[i + 1].X) * 0.5;
                figure.Segments!.Add(new BezierSegment {
                    Point1 = new Point(midpoint, points[i].Y),
                    Point2 = new Point(midpoint, points[i + 1].Y),
                    Point3 = points[i + 1],
                });
            }
            geometry.Figures!.Add(figure);
            context.DrawGeometry(null, new Pen(curveBrush, 2.5), geometry);

            for (int i = 0; i < points.Length; i++) {
                bool harmonic = values[i] >= 0;
                IBrush fill = harmonic ? harmonicBrush : noiseBrush;
                bool selected = i == SelectedBandIndex;
                if (selected) {
                    context.DrawEllipse(
                        null,
                        new Pen(selectionBrush, 2),
                        points[i],
                        11,
                        11);
                }
                double radius = i == activeBand ? 9 : selected ? 8 : 6;
                context.DrawEllipse(fill, new Pen(plotBackground, 1.5), points[i], radius, radius);
            }
        }

        int ClosestPoint(Point pointer, out double distanceSquared) {
            int closest = 0;
            distanceSquared = double.MaxValue;
            for (int i = 0; i < HifiHnSpectralProfile.BandCount; i++) {
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
            var plot = PlotBounds();
            return new Point(
                FrequencyX(Frequencies()[band], plot),
                ValueY(Values()[band], plot));
        }

        void DrawHoverGuides(DrawingContext context, Rect plot, IBrush brush) {
            if (!pointerInside || !plot.Contains(hoverPosition)) {
                return;
            }
            const double radius = 72;
            const double dotSpacing = 12;
            double left = Math.Max(plot.X, hoverPosition.X - radius);
            double right = Math.Min(plot.Right, hoverPosition.X + radius);
            double top = Math.Max(plot.Y, hoverPosition.Y - radius);
            double bottom = Math.Min(plot.Bottom, hoverPosition.Y + radius);
            var guidePen = new Pen(brush, 0.8);
            context.DrawLine(
                guidePen,
                new Point(hoverPosition.X, top),
                new Point(hoverPosition.X, bottom));
            context.DrawLine(
                guidePen,
                new Point(left, hoverPosition.Y),
                new Point(right, hoverPosition.Y));

            double startX = Math.Ceiling(left / dotSpacing) * dotSpacing;
            double startY = Math.Ceiling(top / dotSpacing) * dotSpacing;
            double radiusSquared = radius * radius;
            for (double x = startX; x <= right; x += dotSpacing) {
                for (double y = startY; y <= bottom; y += dotSpacing) {
                    double dx = x - hoverPosition.X;
                    double dy = y - hoverPosition.Y;
                    if (dx * dx + dy * dy <= radiusSquared) {
                        context.DrawEllipse(brush, null, new Point(x, y), 0.9, 0.9);
                    }
                }
            }
        }

        void SetBandFromPosition(int band, Point position) {
            var plot = PlotBounds();
            double normalizedX = Math.Clamp((position.X - plot.X) / plot.Width, 0, 1);
            double logMin = Math.Log(HifiHnSpectralProfile.MinFrequencyHz);
            double logMax = Math.Log(HifiHnSpectralProfile.MaxFrequencyHz);
            double frequency = Math.Exp(logMin + normalizedX * (logMax - logMin));
            SetFrequencyByBand(band, ClampFrequencyForBand(band, Math.Round(frequency)));

            double value = (plot.Center.Y - position.Y) / (plot.Height * 0.5)
                * HifiHnSpectralProfile.MaxBalanceDb;
            SetValueByBand(band, Math.Round(ClampDb(value) * 2.0) / 2.0);
        }

        void SetFrequencyByBand(int band, double value) {
            switch (band) {
                case 0: BodyHz = value; break;
                case 1: WarmthHz = value; break;
                case 2: PresenceHz = value; break;
                case 3: ClarityHz = value; break;
                case 4: AirHz = value; break;
            }
        }

        void SetValueByBand(int band, double value) {
            switch (band) {
                case 0: BodyDb = value; break;
                case 1: WarmthDb = value; break;
                case 2: PresenceDb = value; break;
                case 3: ClarityDb = value; break;
                case 4: AirDb = value; break;
            }
        }

        void SetFrequency(
            DirectProperty<HifiHnBalanceGraph, double> property,
            ref double field,
            double value,
            int band) {
            double fallback = HifiHnSpectralProfile.DefaultFrequenciesHz[band];
            value = double.IsFinite(value)
                ? Math.Clamp(value, HifiHnSpectralProfile.MinFrequencyHz, HifiHnSpectralProfile.MaxFrequencyHz)
                : fallback;
            SetAndRaise(property, ref field, value);
        }

        double ClampFrequencyForBand(int band, double value) {
            var frequencies = Frequencies();
            double minimum = band == 0
                ? HifiHnSpectralProfile.MinFrequencyHz
                : frequencies[band - 1] * HifiHnSpectralProfile.MinFrequencyRatio;
            double maximum = band == HifiHnSpectralProfile.BandCount - 1
                ? HifiHnSpectralProfile.MaxFrequencyHz
                : frequencies[band + 1] / HifiHnSpectralProfile.MinFrequencyRatio;
            return Math.Clamp(value, minimum, maximum);
        }

        Rect PlotBounds() => new(
            24,
            14,
            Math.Max(1, Bounds.Width - 48),
            Math.Max(1, Bounds.Height - 28));

        double[] Values() => new[] { BodyDb, WarmthDb, PresenceDb, ClarityDb, AirDb };
        double[] Frequencies() => new[] { BodyHz, WarmthHz, PresenceHz, ClarityHz, AirHz };

        static double FrequencyX(double frequency, Rect plot) {
            double min = Math.Log(HifiHnSpectralProfile.MinFrequencyHz);
            double max = Math.Log(HifiHnSpectralProfile.MaxFrequencyHz);
            frequency = Math.Clamp(frequency, HifiHnSpectralProfile.MinFrequencyHz, HifiHnSpectralProfile.MaxFrequencyHz);
            return plot.X + (Math.Log(frequency) - min) / (max - min) * plot.Width;
        }

        static double ValueY(double value, Rect plot) {
            return plot.Center.Y - ClampDb(value) / HifiHnSpectralProfile.MaxBalanceDb * plot.Height * 0.5;
        }

        static double ClampDb(double value) {
            return double.IsFinite(value)
                ? Math.Clamp(value, -HifiHnSpectralProfile.MaxBalanceDb, HifiHnSpectralProfile.MaxBalanceDb)
                : 0;
        }

    }
}
