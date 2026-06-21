# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenUtau is a cross-platform (Windows/macOS/Linux) singing voice synthesis editor for the UTAU community. It is a .NET 8 desktop application built with Avalonia (UI) and a large core library that handles voicebanks, phonemization, and multiple audio rendering backends. This fork ("HifiNeura") is actively developing the **HIFI-NEURA** neural phrase renderer (see `OpenUtau.Core/HifiNeural/`).

## Build, Test, Run

```bash
# Restore dependencies (run from repo root)
dotnet restore OpenUtau

# Run the app (Debug)
dotnet run --project OpenUtau

# Run all tests
dotnet test OpenUtau.Test

# Run a single test class / method (xUnit filter)
dotnet test OpenUtau.Test --filter "FullyQualifiedName~ClassName"
dotnet test OpenUtau.Test --filter "FullyQualifiedName~ClassName.MethodName"

# Publish a self-contained build for a specific runtime (rid: win-x64, osx-arm64, linux-x64, ...)
dotnet publish OpenUtau -c Release -r win-x64 --self-contained true -o bin/win-x64/
```

Notes:
- `dotnet test` builds the whole solution first, so it is the quickest way to verify a change compiles end-to-end.
- The `OpenUtau` app project compiles with `TreatWarningsAsErrors=true` 鈥?warnings will fail the build. `OpenUtau.Core` does not.
- On Windows the app targets `net8.0-windows` and uses `Microsoft.ML.OnnxRuntime.DirectML`; elsewhere it targets `net8.0` and uses the plain `Microsoft.ML.OnnxRuntime`. Code guarded by Windows-only APIs uses the `WINDOWS` compile constant.
- Native libraries (resamplers, worldline, etc.) live under `runtimes/<rid>/native/` and are copied to output per-RID.

## Project Structure

The solution (`OpenUtau.sln`) has four projects:

- **OpenUtau.Core** 鈥?Engine and business logic. No UI dependency. Contains the document model, command system, rendering backends, phonemizer API, and format I/O.
- **OpenUtau** 鈥?Avalonia desktop UI (MVVM via ReactiveUI). Views (`.axaml`) + ViewModels. Depends on Core and Plugin.Builtin.
- **OpenUtau.Plugin.Builtin** 鈥?Built-in phonemizers (one class per language/method, e.g. `JapaneseVCVPhonemizer`, `ArpasingPhonemizer`). Loaded by reflection at runtime.
- **OpenUtau.Test** 鈥?xUnit tests. Uses `Avalonia.Headless.XUnit` for UI-touching tests. Test fixtures live in `Usts/` and `Files/`.

## Core Architecture

### Document model + command pattern (the heart of the app)
- `DocManager` (`OpenUtau.Core/DocManager.cs`) is a singleton holding the current `UProject`. It owns the **undo/redo command queue** and a pub/sub notification bus.
- All mutations to the project go through `UCommand` subclasses (`OpenUtau.Core/Commands/`). A command implements `Execute()`/`Unexecute()`. UI and editing code never mutate the model directly.
- Mutations are wrapped in undo groups: `docManager.StartUndoGroup()` 鈫?`docManager.ExecuteCmd(cmd)` (one or more) 鈫?`docManager.EndUndoGroup()`.
- Components implement `ICmdSubscriber.OnNext(cmd, isUndo)` to react to changes (ViewModels subscribe to keep the UI in sync). Non-mutating `Notifications` are also dispatched through the same bus.
- The project model (`OpenUtau.Core/Ustx/`): `UProject` 鈫?`UTrack` / `UVoicePart` 鈫?`UNote` 鈫?`UPhoneme`, plus `UExpression`, `UCurve`, `USinger`. The native file format is `.ustx` (YAML); see `Format/USTx.cs`.

### Phonemizers (lyric 鈫?phonemes)
- API in `OpenUtau.Core/Api/` (see `Api/README.md` and `Phonemizer.cs`). The key method is `Phoneme[] Process(Note[] notes, Note? prev, Note? next)`.
- Concrete phonemizers live in `OpenUtau.Plugin.Builtin/`. They are discovered by reflection (`DocManager.SearchAllPlugins`) from the builtin DLL and from user plugin folders 鈥?adding a `Phonemizer` subclass is enough to register it.
- A phonemizer turns notes/lyrics into positioned phonemes; the renderer then turns phonemes into audio.

### Rendering backends (phonemes 鈫?audio)
- `OpenUtau.Core/Render/Renderers.cs` is the registry. Each singer type (`Classic`, `Enunu`, `Vogen`, `DiffSinger`, `Voicevox`) maps to one or more renderer ids; `CreateRenderer(id)` instantiates the `IRenderer`.
- All renderers implement `IRenderer` (`Render/IRenderer.cs`): `Layout()` estimates timing, `Render()` produces a `RenderResult` (float samples) per `RenderPhrase`. Rendering is phrase-based and cached (`RenderCache.cs`); `RenderEngine.cs` drives pre-render and playback.
- Backends are in their own directories: `Classic/` (UTAU resamplers/wavtools), `DiffSinger/`, `Enunu/`, `Vogen/`, `Voicevox/`, and `HifiNeural/`. Many use ONNX models via `Microsoft.ML.OnnxRuntime`.
- **HIFI-NEURA** (`HifiNeural/`) is the renderer under active development in this fork. `HifiNeuralPhraseRenderer` registers as renderer id `HIFI-NEURA` (legacy alias `HIFI-NEURAL-PHRASE`) for `USingerType.Classic`. The pipeline extracts a mel spectrogram from each phone's oto slice independently (`HifiMelExtractor`), time-stretches each per phone reusing `HifiPhraseFeatureBuilder.WritePhoneMappedSegment` (onset/sustain/release split + natural-stretch warp), then concatenates the per-phone mels onto the phrase frame grid with equal-power overlap cross-fades (`HifiMelPhraseAssembler`, anchored by `oto.preutter`/`overlap`). Target F0 comes from the phrase pitch curve (`HifiF0Builder`), and the mel+F0 run through an ONNX PC-NSF-HiFiGAN vocoder (`HifiOnnxVocoder`). Output is cached by phrase hash + config (`HifiRenderConfig.CacheKey`). The old SharpWavtool rough-wav prototype path has been removed from the active code.

### Batch edits / editing macros
- `OpenUtau.Core/Editing/` (`BatchEdit`) 鈥?macros that mutate notes. Follow the same undo-group + command pattern (see `Editing/README.md`). They get a localized name and are discovered by reflection like phonemizers.

### UI (OpenUtau project)
- Avalonia + ReactiveUI MVVM. `ViewModels/` mirror `Views/`. `MainWindowViewModel` and `NotesViewModel` are the central editor view models; they subscribe to `DocManager` and issue `UCommand`s.
- `Program.cs` is the entry point; `ViewLocator` maps view models to views.

## Conventions

- Both Core and app projects have `<Nullable>enable</Nullable>` 鈥?respect nullable annotations.
- Localized UI strings live in `.resx` files (`OpenUtau/Strings/`, `OpenUtau/Resources/`); translations are managed via Crowdin (`crowdin.yml`). Use resource keys, not hardcoded display strings, for user-facing text.
- Logging uses Serilog (`Log.Information/Warning/Error`).
- `.editorconfig` defines formatting (notably 4-space indentation, file-scoped conventions) and is enforced consistently across the codebase.

## CI

- GitHub Actions (`.github/workflows/build.yml`) is `workflow_dispatch` only: runs `dotnet test OpenUtau.Test`, then publishes per-RID self-contained builds and packages installers/dmg/AppImage. AppVeyor (`appveyor.yml`) builds the `stable` branch.
