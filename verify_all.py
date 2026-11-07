# -*- coding: utf-8 -*-
"""
All-in-one verification script for SAFE-T external dependencies and runtime functionalities.

Verifies every component that depends on external tools, models, environments, or data files:
  1. Required data & model files (CSVs, Piper .onnx/.json, MFA pretrained_models, mfa_align.py)
  2. Bundle-specific artifacts when testing a built bundle (SAFE-T.exe, ffmpeg/, vlc/, mfa_env.tar, model_cache/)
  3. FFmpeg executable & audio loading (whisperx.load_audio)
  4. libVLC / python-vlc initialization (vlc.Instance & MediaPlayer)
  5. Praat (parselmouth) & FormantData extraction + Viterbi/Z-score cleaning + CSV loading
  6. SubsequenceDTW & TTS synthesis (Piper male, Piper female, KittenTTS nano + espeak-ng)
  7. WhisperX ASR ("tiny") + Pyannote VAD + Wav2Vec2 alignment on synthesized speech
  8. Montreal Forced Aligner (MFA) end-to-end alignment (via conda env or bundled mfa_env.tar)
  9. PyQt6 + PyQtGraph GUI initialization smoke test

Usage:
    # Verify developer environment (from project root):
    python verify_all.py

    # Verify a built standalone bundle directory:
    python verify_all.py --bundle dist/SAFE-T

    # Fast mode (skip full MFA alignment subprocess):
    python verify_all.py --skip-mfa-align
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile


REQUIRED_FILES = [
    "mfa_align.py",
    "landmark_identification_ground_truth.csv",
    "landmark_identification_phoneme_mapping.csv",
    "en_GB-alan-medium.onnx",
    "en_GB-alan-medium.onnx.json",
    "en_GB-alba-medium.onnx",
    "en_GB-alba-medium.onnx.json",
    os.path.join("pretrained_models", "acoustic", "english_mfa.zip"),
    os.path.join("pretrained_models", "dictionary", "english_uk_mfa.dict"),
    "LICENSE",
]

BUNDLE_REQUIRED_ITEMS = [
    "SAFE-T.exe",
    os.path.join("ffmpeg", "ffmpeg.exe"),
    os.path.join("vlc", "libvlc.dll"),
    os.path.join("vlc", "libvlccore.dll"),
    os.path.join("vlc", "plugins"),
    "mfa_env.tar",
    os.path.join("model_cache", ".complete"),
    os.path.join("model_cache", "huggingface"),
    os.path.join("model_cache", "torch"),
]


def _patch_transformers_auto_docstring() -> None:
    """Guard transformers.utils.auto_docstring against short relative paths in PyInstaller PYZ."""
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


def _configure_bundle_env(bundle_dir: str) -> None:
    """Configure environment variables to point exclusively at a bundle's local assets."""
    bundle_dir = os.path.abspath(bundle_dir)
    os.chdir(bundle_dir)

    ffmpeg_dir = os.path.join(bundle_dir, "ffmpeg")
    if os.path.isdir(ffmpeg_dir):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    vlc_dir = os.path.join(bundle_dir, "vlc")
    if os.path.isfile(os.path.join(vlc_dir, "libvlc.dll")):
        os.environ["PYTHON_VLC_LIB_PATH"] = os.path.join(vlc_dir, "libvlc.dll")
        os.environ["PYTHON_VLC_MODULE_PATH"] = os.path.join(vlc_dir, "plugins")
        os.environ["VLC_PLUGIN_PATH"] = os.path.join(vlc_dir, "plugins")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(vlc_dir)

    model_cache = os.path.join(bundle_dir, "model_cache")
    if os.path.isdir(model_cache):
        os.environ["HF_HOME"] = os.path.join(model_cache, "huggingface")
        os.environ["TORCH_HOME"] = os.path.join(model_cache, "torch")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        if os.path.isfile(os.path.join(model_cache, ".complete")):
            os.environ["HF_HUB_OFFLINE"] = "1"

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def check_required_files(root_dir: str, is_bundle: bool) -> str:
    missing = []
    for rel in REQUIRED_FILES:
        full = os.path.join(root_dir, rel)
        if not os.path.exists(full):
            missing.append(rel)
        elif os.path.isfile(full) and os.path.getsize(full) == 0:
            missing.append(f"{rel} (0 bytes)")

    if is_bundle:
        for rel in BUNDLE_REQUIRED_ITEMS:
            full = os.path.join(root_dir, rel)
            if not os.path.exists(full):
                missing.append(rel)

    if missing:
        raise FileNotFoundError(
            "Missing required files/folders in "
            f"'{root_dir}': {', '.join(missing)}"
        )

    # Validate english_mfa.zip is a valid zip archive
    mfa_zip = os.path.join(root_dir, "pretrained_models", "acoustic", "english_mfa.zip")
    if not zipfile.is_zipfile(mfa_zip):
        raise RuntimeError(f"'{mfa_zip}' is not a valid zip archive.")

    # Validate CSV files parse cleanly
    import pandas as pd

    gt_df = pd.read_csv(os.path.join(root_dir, "landmark_identification_ground_truth.csv"))
    pm_df = pd.read_csv(os.path.join(root_dir, "landmark_identification_phoneme_mapping.csv"))
    if gt_df.empty or pm_df.empty:
        raise RuntimeError("One or more landmark CSV files are empty.")

    return (
        f"{len(REQUIRED_FILES)} core files OK"
        + (f", {len(BUNDLE_REQUIRED_ITEMS)} bundle items OK" if is_bundle else "")
        + f" (GT rows={len(gt_df)}, phoneme mappings={len(pm_df)})"
    )


def check_ffmpeg(temp_dir: str) -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError("ffmpeg executable not found on PATH.")

    proc = subprocess.run(
        [ffmpeg_path, "-version"],
        capture_output=True,
        text=True,
        check=True,
    )
    first_line = (proc.stdout or "").splitlines()[0] if proc.stdout else "ffmpeg OK"

    # Generate a short sine wave WAV and decode it via whisperx.load_audio (which shells out to ffmpeg)
    _patch_transformers_auto_docstring()
    import numpy as np
    import soundfile as sf
    import whisperx

    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    tone = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    wav_path = os.path.join(temp_dir, "ffmpeg_test.wav")
    sf.write(wav_path, tone, sr)

    loaded = whisperx.load_audio(wav_path)
    if loaded is None or len(loaded) == 0:
        raise RuntimeError("whisperx.load_audio returned empty audio.")

    return f"{ffmpeg_path} ({first_line})"


def check_vlc() -> str:
    import vlc

    instance = vlc.Instance("--no-video", "--quiet")
    if instance is None:
        raise RuntimeError("vlc.Instance() returned None (libvlc failed to initialize).")
    player = instance.media_player_new()
    if player is None:
        raise RuntimeError("instance.media_player_new() returned None.")
    version = vlc.libvlc_get_version()
    version_str = version.decode("utf-8", errors="replace") if isinstance(version, bytes) else str(version)
    player.release()
    instance.release()
    return f"libVLC {version_str}"


def check_praat_and_formants(temp_dir: str) -> str:
    import numpy as np
    import soundfile as sf
    from parselmouth import Sound
    from formants import FormantData

    # Create a synthetic vowel-like signal (F0=120Hz + resonances near 500, 1500, 2500 Hz)
    sr = 16000
    duration = 0.4
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (
        0.4 * np.sin(2 * np.pi * 500.0 * t)
        + 0.3 * np.sin(2 * np.pi * 1500.0 * t)
        + 0.2 * np.sin(2 * np.pi * 2500.0 * t)
        + 0.1 * np.sin(2 * np.pi * 3500.0 * t)
    ).astype(np.float32)
    wav_path = os.path.join(temp_dir, "praat_test.wav")
    sf.write(wav_path, signal, sr)

    # Run FormantData in temp_dir or clean up side-effect log files in cwd
    side_effect_files = [
        FormantData.DEBUG_LOG_FILE,
        FormantData.OUTLIER_FIX_LOG_FILE,
        FormantData.FORMANT_TARGET_RECORD_FILE,
        "formants_values.csv",
    ]
    pre_existing = {f: os.path.exists(f) for f in side_effect_files}

    try:
        snd = Sound(wav_path)
        fd = FormantData(sound=snd)
        fd.extract_formant_data_from_sound(snd)
        times = fd.get_formant_times()
        vals = fd.get_formant_values()
        if times.size == 0 or vals.size == 0:
            raise RuntimeError("Praat formant extraction returned no frames.")

        cleaned_z = FormantData.formant_clean_shift_outliers(
            formant_values=vals, formant_times=times
        )

        # Build a realistic target window from landmark_identification_ground_truth.csv
        import pandas as pd

        gt_df = pd.read_csv("landmark_identification_ground_truth.csv")
        row0 = gt_df.iloc[0]
        sample_targets = [
            {
                "start_frac": 0.0,
                "end_frac": 1.0,
                "mean": np.array(
                    [
                        FormantData.safe_float_convert(row0["T1_F1_Mean"], 500.0),
                        FormantData.safe_float_convert(row0["T1_F2_Mean"], 1500.0),
                        np.nan,
                        np.nan,
                    ]
                ),
                "sd": np.array(
                    [
                        FormantData.safe_float_convert(row0["T1_F1_SD"], 50.0),
                        FormantData.safe_float_convert(row0["T1_F2_SD"], 150.0),
                        np.nan,
                        np.nan,
                    ]
                ),
                "weight": FormantData.VITERBI_EMISSION_MONOPHTHONG_WEIGHT,
            }
        ]
        cleaned_v = FormantData.formant_clean_viterbi_fb(
            formant_values=vals,
            formant_times=times,
            targets=sample_targets,
        )
        if cleaned_z.shape != vals.shape or cleaned_v.shape != vals.shape:
            raise RuntimeError("Formant cleaning output shape mismatch.")
    finally:
        for f, existed in pre_existing.items():
            if not existed and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    return f"Extracted & cleaned {vals.shape[0]} frames x {vals.shape[1]} formants"


def check_subsequence_dtw_and_tts(temp_dir: str) -> tuple[str, str]:
    """
    Verify SubsequenceDTW initialization, Piper Male, Piper Female, and KittenTTS synthesis,
    and DTW sentence matching. Returns (status_message, path_to_synthesized_wav).
    """
    from subsequence_dtw import SubsequenceDTW

    dtw = SubsequenceDTW(hop_length=512, sr=16000)
    sentence = "The quick brown fox jumps over the lazy dog."

    wav_paths = {}
    for model_name in ("piper_male", "piper_female", "kitten_female"):
        out_wav = os.path.join(temp_dir, f"tts_{model_name}.wav")
        audio = dtw.generate_reference_audio(sentence, out_wav, voice_model=model_name)
        if audio is None or len(audio) < 1600:
            raise RuntimeError(f"TTS model '{model_name}' generated invalid/short audio.")
        wav_paths[model_name] = (out_wav, audio)

    # Verify DTW alignment between piper_male and piper_female
    ref_audio = wav_paths["piper_male"][1]
    long_audio = wav_paths["piper_female"][1]
    match = dtw.find_sentence_match(ref_audio, long_audio)
    if not match or match["end"] <= match["start"]:
        raise RuntimeError(f"SubsequenceDTW failed to find valid match: {match}")

    return (
        f"Piper (male/female) + KittenTTS OK; DTW match={match['start']:.2f}s-{match['end']:.2f}s",
        wav_paths["piper_male"][0],
    )


def check_whisperx_asr(temp_dir: str, sample_wav: str) -> str:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    _patch_transformers_auto_docstring()
    from asr import WhisperX_ASR

    asr_engine = WhisperX_ASR("tiny")
    stim_path = os.path.join(temp_dir, "stimulus.txt")
    with open(stim_path, "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog.\n")

    tg_path = os.path.join(temp_dir, "asr_out.TextGrid")
    asr_engine.set_asr_target(sample_wav, stimulus_path=stim_path, textgrid_path=tg_path)
    timestamps = asr_engine.get_valid_sentence_timestamps(shuffle_chunk_size=False)
    if not timestamps:
        raise RuntimeError("WhisperX_ASR returned no sentence timestamps.")

    matched = [ts for ts in timestamps if ts.start is not None and ts.end is not None]
    if not matched:
        raise RuntimeError("WhisperX_ASR did not match the synthesized test sentence.")

    asr_engine.save_sentences_as_textgrid(timestamps)
    if not os.path.isfile(tg_path):
        raise RuntimeError("WhisperX_ASR failed to save output TextGrid.")

    return (
        f"WhisperX ('tiny', device={asr_engine.device}) + Pyannote VAD + Wav2Vec2 OK "
        f"({matched[0].start:.2f}s-{matched[0].end:.2f}s)"
    )


def check_mfa_alignment(temp_dir: str, sample_wav: str, mfa_env_name: str, skip_align: bool) -> str:
    import tarfile
    import mfa_runner

    archive = mfa_runner.bundled_mfa_archive()
    conda_exe = None
    if archive:
        if skip_align:
            if not tarfile.is_tarfile(archive):
                raise RuntimeError(f"'{archive}' is not a valid tar archive.")
            size_mb = os.path.getsize(archive) / (1024 * 1024)
            return f"Bundled mfa_env.tar valid ({size_mb:.1f} MB); alignment skipped via --skip-mfa-align"
        env_dir = mfa_runner.prepare_bundled_mfa_env(archive)
        mode_desc = f"bundled mfa_env.tar -> {env_dir}"
    else:
        conda_exe = shutil.which("conda") or os.environ.get("CONDA_EXE")
        if not conda_exe:
            for candidate in (
                os.path.expanduser(r"~\anaconda3\Scripts\conda.exe"),
                os.path.expanduser(r"~\miniconda3\Scripts\conda.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\anaconda3\Scripts\conda.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"),
                r"C:\ProgramData\anaconda3\Scripts\conda.exe",
                r"C:\ProgramData\miniconda3\Scripts\conda.exe",
            ):
                if os.path.isfile(candidate):
                    conda_exe = candidate
                    break
        if not conda_exe:
            raise FileNotFoundError(
                "Neither bundled mfa_env.tar nor conda executable found for MFA verification."
            )
        mode_desc = f"conda env '{mfa_env_name}' via {conda_exe}"
        if skip_align:
            return f"MFA environment ready ({mode_desc}); alignment skipped via --skip-mfa-align"

    corpus_dir = os.path.join(temp_dir, "mfa_corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    shutil.copy2(sample_wav, os.path.join(corpus_dir, "utt1.wav"))
    with open(os.path.join(corpus_dir, "utt1.lab"), "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog")

    script_path = os.path.abspath("mfa_align.py")
    rc = mfa_runner.run_mfa_alignment(
        script_path=script_path,
        input_dir=corpus_dir,
        conda_exe=conda_exe if not archive else "conda",
        env_name=mfa_env_name,
    )
    if rc != 0:
        raise RuntimeError(f"mfa_runner.run_mfa_alignment exited with code {rc}.")

    aligned_tg = os.path.join(corpus_dir, "aligned", "utt1.TextGrid")
    if not os.path.isfile(aligned_tg):
        raise FileNotFoundError(
            f"MFA finished with code 0 but '{aligned_tg}' was not created."
        )

    return f"MFA alignment produced '{os.path.basename(aligned_tg)}' ({mode_desc})"


def check_pyqt6_gui() -> str:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    import pyqtgraph as pg

    app = QApplication.instance() or QApplication([])
    plot = pg.PlotWidget()
    plot.close()
    return f"PyQt6 + PyQtGraph initialized ({app.platformName()})"


def run_verification(
    *,
    bundle_dir: str | None = None,
    mfa_env_name: str = "mfa_env1",
    skip_mfa_align: bool = False,
) -> int:
    is_bundle = bool(bundle_dir) or getattr(sys, "frozen", False)
    if bundle_dir:
        _configure_bundle_env(bundle_dir)
    elif getattr(sys, "frozen", False):
        bundle_dir = os.path.dirname(os.path.abspath(sys.executable))
        _configure_bundle_env(bundle_dir)
    else:
        project_root = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_root)
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    root_dir = os.getcwd()
    print(f"SAFE-T Verification Root : {root_dir}")
    print(f"Mode                     : {'Standalone Bundle' if is_bundle else 'Developer Environment'}")
    print(f"Python                   : {sys.executable}")

    results: list[tuple[str, bool, str, float]] = []

    with tempfile.TemporaryDirectory(prefix="safet_verify_") as temp_dir:
        sample_wav_holder = [None]

        def _step_files() -> str:
            return check_required_files(root_dir, is_bundle=is_bundle)

        def _step_ffmpeg() -> str:
            return check_ffmpeg(temp_dir)

        def _step_vlc() -> str:
            return check_vlc()

        def _step_praat() -> str:
            return check_praat_and_formants(temp_dir)

        def _step_dtw() -> str:
            msg, wav_path = check_subsequence_dtw_and_tts(temp_dir)
            sample_wav_holder[0] = wav_path
            return msg

        def _step_asr() -> str:
            if not sample_wav_holder[0]:
                raise RuntimeError("No sample WAV available from TTS step.")
            return check_whisperx_asr(temp_dir, sample_wav_holder[0])

        def _step_mfa() -> str:
            if not sample_wav_holder[0]:
                raise RuntimeError("No sample WAV available from TTS step.")
            return check_mfa_alignment(
                temp_dir,
                sample_wav_holder[0],
                mfa_env_name=mfa_env_name,
                skip_align=skip_mfa_align,
            )

        def _step_gui() -> str:
            return check_pyqt6_gui()

        steps = [
            ("1. Data & Model Files", _step_files),
            ("2. FFmpeg & Audio Loading", _step_ffmpeg),
            ("3. libVLC / python-vlc", _step_vlc),
            ("4. Praat & Formant Extraction/Cleaning", _step_praat),
            ("5. SubsequenceDTW & TTS (Piper + KittenTTS)", _step_dtw),
            ("6. WhisperX ASR + Pyannote VAD + Wav2Vec2", _step_asr),
            ("7. Montreal Forced Aligner (MFA)", _step_mfa),
            ("8. PyQt6 + PyQtGraph GUI", _step_gui),
        ]

        for name, fn in steps:
            print(f"\n==> {name} ...", flush=True)
            t0 = time.perf_counter()
            try:
                detail = fn()
                elapsed = time.perf_counter() - t0
                print(f"    [PASS] ({elapsed:.1f}s) {detail}", flush=True)
                results.append((name, True, detail, elapsed))
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                tb = traceback.format_exc()
                print(f"    [FAIL] ({elapsed:.1f}s) {exc}\n{tb}", flush=True)
                results.append((name, False, str(exc), elapsed))

    print("\n" + "=" * 72)
    print("  VERIFICATION SUMMARY")
    print("=" * 72)
    failed = 0
    for name, ok, detail, elapsed in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{status}] {name:<45} ({elapsed:5.1f}s)  {detail}")
    print("=" * 72)

    if failed:
        print(f"Verification FAILED ({failed} of {len(results)} checks failed).")
        return 1
    print(f"All {len(results)} checks PASSED!")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify all external-dependent SAFE-T functionalities."
    )
    parser.add_argument(
        "--bundle",
        metavar="DIR",
        default=None,
        help="Path to a built bundle directory (e.g. dist/SAFE-T) to verify.",
    )
    parser.add_argument(
        "--mfa-env",
        default="mfa_env1",
        help="Conda MFA environment name when testing dev setup (default: mfa_env1).",
    )
    parser.add_argument(
        "--skip-mfa-align",
        action="store_true",
        help="Only check MFA environment availability without running a full alignment.",
    )
    args = parser.parse_args(argv)
    return run_verification(
        bundle_dir=args.bundle,
        mfa_env_name=args.mfa_env,
        skip_mfa_align=args.skip_mfa_align,
    )


if __name__ == "__main__":
    sys.exit(main())
