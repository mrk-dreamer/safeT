# SAFE-T — Standalone Distribution

This package contains a ready-to-run copy of **SAFE-T** (Semi-automatic formant recognition tool) for Windows 11 (64-bit).

Everything required by the application is already bundled inside this folder:
- Python runtime & GUI (`SAFE-T.exe`, `_internal\`)
- FFmpeg (`ffmpeg\`)
- libVLC audio engine (`vlc\`)
- Montreal Forced Aligner (MFA) environment (`mfa_env.tar`, unpacked automatically to `%LOCALAPPDATA%\SAFE-T\mfa_env` on first install / first alignment)
- Pre-trained acoustic models, pronunciation dictionaries, Piper TTS voices, and cached ASR/TTS models (`pretrained_models\`, `model_cache\`, `*.onnx`)

**No additional software (no Python, no Conda, no VLC, no FFmpeg) and no `PATH` edits are required.**

---

## How to Run

1. If you downloaded the **Installer** (`SAFE-T-Setup.exe`), run it and launch **SAFE-T** from the Desktop or Start Menu shortcut.
2. If you downloaded the **portable `.zip`** (`SAFE-T.zip`), extract the entire `SAFE-T` folder to a writable folder (for example on your Desktop or under `Documents`) and double-click **`SAFE-T.exe`**.

> **Note:** On the very first MFA alignment run (if you used the portable `.zip` instead of the installer), the bundled MFA environment is unpacked to `%LOCALAPPDATA%\SAFE-T\mfa_env`. That one-time step takes few minutes; subsequent runs start immediately.

---

## Offline vs. Online Use

- **WhisperX ASR + Wav2Vec2 alignment**, **MFA phoneme alignment**, **Praat formant analysis**, **Piper TTS**, and **KittenTTS** all work **100% offline**.
- Only the **gTTS** (Google Text-to-Speech) option inside Subsequence DTW requires an internet connection, because gTTS queries Google's online synthesis endpoint.

---

## Basic Workflow

1. **Load Audio**: Use `File -> Load Audio` to open a `.wav` recording.
2. **Load Stimulus**: Click **Load Stimulus** to select the text file containing expected sentences.
3. **Run ASR + Run MFA**: Choose WhisperX or Subsequence DTW and run sentence segmentation + MFA phoneme alignment.
4. **Load Landmark Info / Formant Analysis**: Inspect, clean, and export formant targets and plots.
