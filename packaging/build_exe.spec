# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the SAFE-T standalone bundle (one-folder build).

Do not run this directly; use packaging/build_bundle.ps1, which also adds FFmpeg,
libVLC, the packed MFA environment, the pre-downloaded models and the data files
next to the executable.

Manual use (from the project root, with .venv active):
    pyinstaller --noconfirm --distpath dist --workpath build packaging/build_exe.spec
"""

import os

from PyInstaller.utils.hooks import collect_all

project_root = os.path.abspath(os.path.join(SPECPATH, os.pardir))

datas = []
binaries = []
hiddenimports = []

# Packages without (complete) PyInstaller hooks that ship data files, native
# libraries or import submodules dynamically. Size is not a concern, so collect
# everything they contain.
for package in (
    # ASR
    "whisperx",
    "faster_whisper",
    "ctranslate2",
    "pyannote.audio",
    "pyannote.core",
    "pyannote.database",
    "pyannote.metrics",
    "pyannote.pipeline",
    "speechbrain",
    "asteroid_filterbanks",
    "torch_audiomentations",
    "torch_pitch_shift",
    "torchmetrics",
    "lightning",
    "lightning_fabric",
    "lightning_utilities",
    # TTS used by the subsequence DTW method
    "kittentts",
    "misaki",
    "phonemizer",
    "segments",
    "csvw",
    "language_tags",
    "espeakng_loader",
    "piper",
    "gtts",
    "num2words",
    # Praat / text tools
    "parselmouth",
    "praatio",
    "textgrid",
    "jiwer",
    "rapidfuzz",
):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception as exc:  # package missing from the environment
        print(f"[build_exe.spec] Skipping {package}: {exc}")
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports += ["vlc", "pkg_resources", "verify_all"]

# Packages that call torch.jit.script / inspect.getsource at import time need
# their .py sources next to the bytecode.
source_packages = [
    "whisperx",
    "transformers",
    "pyannote",
    "speechbrain",
    "asteroid_filterbanks",
    "torch_audiomentations",
    "torch_pitch_shift",
    "torchmetrics",
    "lightning",
    "lightning_fabric",
    "julius",
]

a = Analysis(
    [os.path.join(project_root, "player_pyqt6.py")],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPECPATH, "runtime_hook.py")],
    excludes=["tkinter"],
    noarchive=False,
    module_collection_mode={name: "pyz+py" for name in source_packages},
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SAFE-T",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX is known to corrupt torch / Qt / CUDA DLLs
    # The application reports progress and errors via print(); keep the console.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SAFE-T",
)
