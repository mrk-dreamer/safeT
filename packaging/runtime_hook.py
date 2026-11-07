"""
PyInstaller runtime hook for the SAFE-T standalone bundle.

Runs before player_pyqt6.py and wires up the tools shipped next to the executable,
so that end users do not need to install or configure anything:

    SAFE-T.exe
    ffmpeg/        ffmpeg.exe (used by whisperx.load_audio)
    vlc/           libvlc.dll, libvlccore.dll, plugins/ (used by python-vlc)
    model_cache/   pre-downloaded HuggingFace and torch hub models
    mfa_env.tar    conda-pack archive of the MFA environment (see mfa_runner.py)
"""

import os
import sys

if getattr(sys, "frozen", False):
    base_dir = os.path.dirname(os.path.abspath(sys.executable))

    # The application resolves data files and writes results relative to the
    # working directory (pretrained_models/, *.csv, *.onnx, PyPraat_sentences/).
    os.chdir(base_dir)

    def _prepend_to_path(directory):
        if os.path.isdir(directory):
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")

    # FFmpeg: whisperx calls the `ffmpeg` executable.
    _prepend_to_path(os.path.join(base_dir, "ffmpeg"))

    # libVLC: python-vlc honours these variables before searching the system.
    vlc_dir = os.path.join(base_dir, "vlc")
    if os.path.isfile(os.path.join(vlc_dir, "libvlc.dll")):
        os.environ.setdefault("PYTHON_VLC_LIB_PATH", os.path.join(vlc_dir, "libvlc.dll"))
        os.environ.setdefault("PYTHON_VLC_MODULE_PATH", os.path.join(vlc_dir, "plugins"))
        os.environ.setdefault("VLC_PLUGIN_PATH", os.path.join(vlc_dir, "plugins"))
        os.add_dll_directory(vlc_dir)

    # Pyannote VAD checkpoints used by WhisperX contain OmegaConf/Pyannote globals
    # that require weights_only=False under PyTorch 2.6+.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    # Guard transformers.utils.auto_docstring against short relative paths in PyInstaller PYZ.
    try:
        import transformers.utils.auto_docstring as _ad

        _orig_get_model_name = getattr(_ad, "get_model_name", None)
        if callable(_orig_get_model_name):
            def _safe_get_model_name(obj):
                try:
                    return _orig_get_model_name(obj)
                except IndexError:
                    return None

            _ad.get_model_name = _safe_get_model_name
    except Exception:
        pass

    # Pre-downloaded models (WhisperX, wav2vec2 alignment, KittenTTS).
    model_cache = os.path.join(base_dir, "model_cache")
    if os.path.isdir(model_cache):
        os.environ.setdefault("HF_HOME", os.path.join(model_cache, "huggingface"))
        os.environ.setdefault("TORCH_HOME", os.path.join(model_cache, "torch"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        if os.path.isfile(os.path.join(model_cache, ".complete")):
            # Everything is cached: never wait for the network.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # `SAFE-T.exe --prepare-mfa` unpacks the bundled MFA environment
    # without starting the GUI (called by the installer).
    if "--prepare-mfa" in sys.argv:
        from mfa_runner import prepare_bundled_mfa_env

        try:
            prepare_bundled_mfa_env()
        except Exception as exc:
            print(f"Could not prepare the bundled MFA environment: {exc}")
            sys.exit(1)
        sys.exit(0)

    # `SAFE-T.exe --self-test` runs the all-in-one verification suite inside
    # the frozen executable using its bundled tools, models, and MFA archive.
    if "--self-test" in sys.argv:
        ext_verify = os.path.join(base_dir, "verify_all.py")
        if os.path.isfile(ext_verify):
            import importlib.util

            spec = importlib.util.spec_from_file_location("verify_all", ext_verify)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            run_verification = mod.run_verification
        else:
            from verify_all import run_verification

        skip_mfa = "--skip-mfa-align" in sys.argv
        sys.exit(run_verification(bundle_dir=base_dir, skip_mfa_align=skip_mfa))
