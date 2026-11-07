# SAFE-T
Semi-automatic formant recognition tool

## Operating System
Tested on Windows 11.

## Installation

### Option 1: Use Pre-built Standalone Package / Installer (Zero Setup for End Users)
Download the latest release (`SAFE-T-Setup.exe` or `SAFE-T.zip`) from the [Releases page](https://github.com/kamsur/SAFE-T/releases) and follow the instructions in `README.md` (same as `packaging/README_DISTRIBUTION.md`) included in the package.

All runtime dependencies — Python, PyTorch/WhisperX, Wav2Vec2, Piper & KittenTTS models, FFmpeg, libVLC, and the Montreal Forced Aligner (MFA) environment + acoustic models — are bundled inside the package. End users do **not** need to install Python, Conda, MFA, VLC, or FFmpeg.

### Option 2: Build from Source (For Developers)

**You will first need to install these external dependencies:**
- VLC Media Player
- FFmpeg
- Conda + Montreal Forced Aligner (MFA)

Installation instructions below.

Clone the repository using git:
```bash
git clone https://github.com/kamsur/SAFE-T.git
```

### Setup Montreal Forced Aligner (MFA)

Reference: https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html

1. Install Conda-installation[https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html]. Python versions tested: 3.13.7.
    - Add directory of Conda executable to system PATH
    - Windows: Add `C:\Users\<YourUsername>\anaconda3\Scripts` to the PATH environment variable, for example.
    - Restart your terminal or IDE after modifying the PATH variable.
    - Check installation by running:
    ```bash
    conda --version
    ```
    - If path to `conda.exe`, cannot be added to system PATH (generally when conda is not in C:\ drive), then in `player_pyqt6.py`, under `self.CONDA_EXE_PATH`, set the full absolute path to `conda.exe` as in:
    ```python
    self.CONDA_EXE_PATH = r"C:\Users\<YourUsername>\anaconda3\Scripts\conda.exe"
    ```
    - If path to `conda.exe`, can be added to system PATH, then in `player_pyqt6.py`, under `self.CONDA_EXE_PATH`, set as:
    ```python
    self.CONDA_EXE_PATH = "conda"
    ```

2. Setup new virtual environment for MFA through Anaconda Prompt/Terminal. (MFA versions tested: 3.3.7. Python versions tested: Same as Conda. Use version-specific installation if future versions cause issues):
```bash
conda config --add channels conda-forge
conda create -n mfa_env1 montreal-forced-aligner
conda activate mfa_env1
mfa --help
```
The name of the environment (as in example above: `mfa_env1`) should match the name used in `player_pyqt6.py` under `self.MFA_ENVIRONMENT_NAME`.

3. Download pre-trained acoustic models and pronunciation dictionaries (already present in this repository).

Test data belonged to Australian English accent. With no acoustic model for Australian English available, the British English acoustic model and dictionary were used.

Download British English acoustic model from its release page(`english_mfa v3.1.0` tested):
1. Download link here[https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-english_mfa-v3.1.0/english_mfa.zip]. If link is not reachable, go to the download page and find the file. Download page here[https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-english_mfa-v3.1.0]. Downloadable file: `english_mfa.zip`.
2. Place the downloaded `.zip` file in the following location, without unzipping:
   `<path_to_project_directory>/pretrained_models/acoustic/`

The file location of acoustic model is placed in `mfa_align.py` as `DEFAULT_MODEL_PATH = "pretrained_models/acoustic/english_mfa.zip"`

Download British English pronunciation dictionary from its release page (`english_uk_mfa v3.1.0` tested):
1. Download link here[https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_uk_mfa-v3.1.0/english_uk_mfa.dict] If link is not reachable, go to the download page and find the file. Download page here[https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-english_uk_mfa-v3.1.0]. Downloadable file: `english_uk_mfa.dict`. 
2. Place the downloaded `.dict` file in the following location:
   `<path_to_project_directory>/pretrained_models/dictionary/`

The file location of dictionary is placed in `mfa_align.py` as `DEFAULT_DICTIONARY_PATH = "pretrained_models/dictionary/english_uk_mfa.dict"`

### Download vlc media player
1. Download and install VLC media player from https://www.videolan.org/vlc/. Version 3.0.21,64-bit, with 'minimum' installation preset tested. Needed for python-vlc package to work.
2. (Optional, if vlc import still fails) Add VLC installation directory to system PATH variable:
   - Windows: Add `C:\Program Files\VideoLAN\VLC` to the PATH environment variable.
   - Restart your terminal or IDE after modifying the PATH variable.
   - Check installation by running:
   ```bash
   vlc --version
   ```

### Setup FFmpeg
1. Download a pre-built FFmpeg `.zip` package for Windows, download link here[https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-full_build.zip] If link is not reachable, go to the download page and find the file. Download page here[https://github.com/GyanD/codexffmpeg/releases/]. Version `version 7.1-full_build-www.gyan.dev` tested. Downloadable file: `ffmpeg-7.1-full_build.zip`.
2. Extract the downloaded `.zip` file into `ffmpeg` directory, in a way that the `bin` directory is directly under `ffmpeg`. Move this folder to `C:\` drive so that the path to `ffmpeg.exe` is `C:\ffmpeg\bin\ffmpeg.exe`.
3. Add the `bin` directory of `ffmpeg` to your system PATH:
   - Windows: Add `C:\ffmpeg\bin` to the PATH environment variable.
4. Restart your terminal or IDE after modifying the PATH variable. Check installation by running:
```bash
ffmpeg -version
```

### Setup Python environment
1. Python 3.12.10 (non-Conda) tested for main application.
2. Create a virtual environment:
```bash
python -m venv .venv
```
3. Activate the virtual environment:
   - Windows:
   ```bash
   .venv\Scripts\activate
   ```
4. Install required packages:
```bash
pip install -r requirements.txt
```

## Run the application
1. Ensure the virtual environment is activated.
2. Run the application:
```bash
python player_pyqt6.py
```
First run may take some time as WhisperX ASR model and Wav2Vec2 are downloaded.

## Building Standalone Zero-Setup Executable / Installer

To build a self-contained Windows distribution (`dist/SAFE-T/SAFE-T.exe` and `dist/SAFE-T.zip`) that bundles Python, FFmpeg, libVLC, the MFA Conda environment (`conda-pack`), and all pre-downloaded ASR/TTS models:

1. Ensure the developer prerequisites above (`.venv`, the `mfa_env1` Conda environment, VLC, FFmpeg, and `pretrained_models/`) are set up on the build machine.
2. Install build dependencies in `.venv` and `conda-pack` in Conda:
```bash
pip install -r requirements_build.txt
conda install -n base -c conda-forge conda-pack
```
3. Run the build script (creates `dist/SAFE-T/SAFE-T.exe` and `dist/SAFE-T.zip`):
```bash
build_exe.bat
```
   Or invoke the PowerShell build script directly (pass `-Installer` if [Inno Setup 6](https://jrsoftware.org/isinfo.php) is installed to also build `dist/SAFE-T-Setup.exe`):
```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_bundle.ps1 -Zip -Installer
```

Do NOT cancel the build process when intermediate failures and warnings may occur or if no progress is seen for a long time. The build takes considerable time to complete as it downloads all the required dependencies.

### Build Script Arguments

Any argument supported by `packaging\build_bundle.ps1` can also be passed directly to `build_exe.bat` (note that `build_exe.bat` passes `-Zip` by default). When a step is skipped, the build script verifies up front that the skipped item already exists in `dist/SAFE-T/` from a previous build, preserves it across the PyInstaller rebuild, and verifies full bundle completeness before packaging.

**Incremental / Skip Flags:**
- `-CodeOnly`: Rebuilds only the application executable and Python code (`mfa_align.py`, `verify_all.py`, `LICENSE`, `README.md`), skipping all external tools, CSVs, MFA Conda packing, and model bundling while preserving and verifying existing bundled assets in `dist/SAFE-T/`. Equivalent to `-SkipMfa -SkipFfmpeg -SkipVlc -SkipCsv -SkipPiperModels -SkipMfaModels -SkipHfTorchModels`.
- `-SkipPyInstaller`: Skips running PyInstaller (`SAFE-T.exe` / `_internal/`) and only updates bundled tools, models, or archives.
- `-SkipMfa`: Skips packing the Conda MFA environment (`mfa_env.tar`).
- `-SkipFfmpeg`: Skips locating and bundling `ffmpeg/` (`ffmpeg.exe`, `ffprobe.exe`, DLLs).
- `-SkipVlc`: Skips locating and bundling `vlc/` (`libvlc.dll`, `libvlccore.dll`, `plugins/`).
- `-SkipCsv`: Skips re-copying `landmark_identification_ground_truth.csv` and `landmark_identification_phoneme_mapping.csv`.
- `-SkipPiperModels`: Skips re-copying the Piper TTS `.onnx` and `.onnx.json` voice models (`en_GB-alan-medium.onnx*`, `en_GB-alba-medium.onnx*`).
- `-SkipMfaModels`: Skips re-copying `pretrained_models/` (`english_mfa.zip` and `english_uk_mfa.dict`).
- `-SkipHfTorchModels`: Skips running `packaging/prefetch_models.py` (`model_cache/` with WhisperX, `wav2vec2`, and KittenTTS).
- `-SkipModels`: Shortcut for `-SkipPiperModels -SkipMfaModels -SkipHfTorchModels`.

**Verification, Output & Tool Path Flags:**
- `-Verify`: Runs the full functional verification suite (`verify_all.py --bundle dist\SAFE-T`) after building the bundle.
- `-Zip`: Creates `dist/SAFE-T.zip` after building the bundle (enabled by default in `build_exe.bat`).
- `-Installer`: Builds `dist/SAFE-T-Setup.exe` using Inno Setup 6 (`ISCC.exe`).
- `-MfaEnvName <name>`: Name of the MFA Conda environment to pack (default: `mfa_env1`).
- `-Python <path>`: Path to the virtual environment's `python.exe` (default: `.venv\Scripts\python.exe`).
- `-Conda <path>`: Path to `conda.exe` if not in standard locations or `PATH`.
- `-VlcDir <path>`: Path to the 64-bit VLC installation directory containing `libvlc.dll`.
- `-FfmpegExe <path>`: Path to `ffmpeg.exe`.

**Examples:**
```bash
# Full first-time build (bundle + zip)
build_exe.bat

# Fast rebuild after changing Python code (preserves & verifies existing models, MFA env, VLC, FFmpeg, CSVs)
build_exe.bat -CodeOnly

# Rebuild without re-packing MFA or re-running model prefetch, then run functional verification
build_exe.bat -SkipMfa -SkipHfTorchModels -Verify
```

### All-in-One Functional Verification (`verify_all.bat` / `verify_all.py`)

To verify that all external-dependent functionalities work end-to-end (required CSV/ONNX/MFA files, FFmpeg audio decoding, libVLC media player initialization, Praat formant extraction & cleaning, Piper + KittenTTS synthesis, Subsequence DTW alignment, WhisperX ASR + Pyannote VAD + Wav2Vec2 alignment, Montreal Forced Aligner alignment, and PyQt6/PyQtGraph GUI initialization):

```bash
# Verify developer environment
verify_all.bat

# Verify the built standalone bundle in dist\SAFE-T (runs offline using bundled tools & model_cache)
verify_all.bat --bundle dist\SAFE-T

# Or run the self-test directly from the built executable (not functional yet)
dist\SAFE-T\SAFE-T.exe --self-test
```

## Usage
Refer USAGE.md for detailed usage instructions.

## Data
Used sample data is available in `Test_data` directory.

## Algorithm documentation
1. **ASR_ALGORITHMS.md:** Describes the `Whisper AI` algorithm's implementation, used for sentence segmentation.
2. **SUBSEQUENCE_DTW_ALGORITHMS.md:** Describes the `Subsequence DTW` algorithm's implementation, used for sentence segmentation.
3. **FORMANT_ALGORITHMS.md:** Describes the algorithmic implementation used for cleaning of Praat formant readings and identification of formant targets.
4. **MAIN_ALGORITHMS.md:** Describes the algorithmic implementation behind all UI functionalities.