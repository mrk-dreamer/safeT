"""
Download every model the application fetches at runtime into a self-contained cache
folder that is shipped with the standalone bundle (see packaging/runtime_hook.py).

Usage (from the project root, with .venv active):
    python packaging/prefetch_models.py dist/SAFE-T/model_cache

It instantiates the same ASR classes the GUI creates at startup, so the cache
contains exactly what the application needs:
    * WhisperX "tiny" (faster-whisper, HuggingFace)
    * wav2vec2 English alignment model (torchaudio / torch hub)
    * KittenTTS nano (HuggingFace)
"""

import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    cache_dir = os.path.abspath(sys.argv[1])
    os.makedirs(cache_dir, exist_ok=True)
    complete_marker = os.path.join(cache_dir, ".complete")
    if os.path.exists(complete_marker):
        os.remove(complete_marker)

    # Must be set before huggingface_hub / torch are imported.
    os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.pop("HF_HUB_OFFLINE", None)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.insert(0, project_root)
    os.chdir(project_root)  # SubsequenceDTW loads the Piper .onnx voices from the cwd

    from asr import WhisperX_ASR
    from subsequence_dtw import SubsequenceDTW

    print(f"Downloading models into {cache_dir} ...")
    WhisperX_ASR("tiny")
    SubsequenceDTW(hop_length=512, sr=16000)

    with open(complete_marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    print("All models cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
