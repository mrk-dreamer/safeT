"""
Launch Montreal Forced Aligner (MFA) alignment from the GUI.

Two modes are supported:

* Development (default when running from source): MFA lives in a named conda
  environment and is started through
  ``conda run -n <env_name> python mfa_align.py ...``.
* Standalone bundle: the MFA conda environment is shipped as a conda-pack
  archive (``mfa_env.tar``) next to the executable. On first use it is
  extracted to a per-user folder, relocated with ``conda-unpack`` and then
  used directly, so end users need neither conda nor any PATH changes.
"""

import os
import shutil
import subprocess
import sys
import tarfile

BUNDLED_MFA_ARCHIVE_NAME = "mfa_env.tar"
_READY_MARKER_NAME = ".autospeech_mfa_ready"


def app_base_dir() -> str:
    """Directory of the executable (frozen) or of this source file."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def user_data_dir() -> str:
    """Per-user, always-writable folder for unpacked runtime data."""
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(root, "SAFE-T")


def bundled_mfa_archive() -> str | None:
    """Path of the bundled conda-pack archive, or None when running from source."""
    for base in (app_base_dir(), os.getcwd()):
        path = os.path.join(base, BUNDLED_MFA_ARCHIVE_NAME)
        if os.path.isfile(path):
            return path
    return None


def _archive_signature(archive: str) -> str:
    stat = os.stat(archive)
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def _env_search_dirs(env_dir: str) -> list[str]:
    """Same directories that `conda activate` puts on PATH on Windows."""
    return [
        env_dir,
        os.path.join(env_dir, "Library", "mingw-w64", "bin"),
        os.path.join(env_dir, "Library", "usr", "bin"),
        os.path.join(env_dir, "Library", "bin"),
        os.path.join(env_dir, "Scripts"),
        os.path.join(env_dir, "bin"),
    ]


def _mfa_subprocess_env(env_dir: str) -> dict[str, str]:
    """Environment variables emulating an activated conda environment."""
    env = os.environ.copy()
    # Do not leak the frozen app's Python/PyInstaller settings into MFA's Python.
    for key in list(env):
        if key in ("PYTHONHOME", "PYTHONPATH", "_MEIPASS2") or key.startswith("_PYI"):
            env.pop(key)
    env["PATH"] = os.pathsep.join(_env_search_dirs(env_dir) + [env.get("PATH", "")])
    env["CONDA_PREFIX"] = env_dir
    env["CONDA_DEFAULT_ENV"] = env_dir
    # Keep MFA's temporary files/databases out of ~/Documents/MFA.
    env.setdefault("MFA_ROOT_DIR", os.path.join(user_data_dir(), "MFA"))
    return env


def _run_conda_unpack(env_dir: str) -> None:
    """Rewrite the prefixes embedded by conda-pack to the current location."""
    python_exe = os.path.join(env_dir, "python.exe")
    unpack_script = os.path.join(env_dir, "Scripts", "conda-unpack-script.py")
    unpack_exe = os.path.join(env_dir, "Scripts", "conda-unpack.exe")
    if os.path.isfile(python_exe) and os.path.isfile(unpack_script):
        command = [python_exe, unpack_script]
    elif os.path.isfile(unpack_exe):
        command = [unpack_exe]
    else:
        raise FileNotFoundError(
            f"conda-unpack not found in '{env_dir}'. Was mfa_env.tar created with conda-pack?"
        )
    subprocess.run(command, check=True, cwd=env_dir, env=_mfa_subprocess_env(env_dir))


def prepare_bundled_mfa_env(archive: str | None = None) -> str:
    """
    Extract and relocate the bundled MFA environment (only once per archive version).

    Returns the path of the ready-to-use environment.
    """
    archive = archive or bundled_mfa_archive()
    if not archive:
        raise FileNotFoundError(f"{BUNDLED_MFA_ARCHIVE_NAME} not found next to the application.")

    target = os.path.join(user_data_dir(), "mfa_env")
    marker = os.path.join(target, _READY_MARKER_NAME)
    signature = _archive_signature(archive)
    if os.path.isfile(marker) and os.path.isfile(os.path.join(target, "python.exe")):
        with open(marker, encoding="utf-8") as f:
            if f.read().strip() == signature:
                return target

    print(
        "Preparing the bundled Montreal Forced Aligner environment "
        f"in '{target}' (one-time setup, this can take a few minutes)..."
    )
    os.makedirs(user_data_dir(), exist_ok=True)
    staging = target + ".partial"
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(staging, filter="tar")
        except TypeError:  # Python without extraction filters
            tar.extractall(staging)

    # conda-unpack must run at the final location, so move first, then relocate.
    shutil.rmtree(target, ignore_errors=True)
    if os.path.exists(target):
        raise RuntimeError(
            f"Could not replace '{target}'. Close programs using it and try again."
        )
    os.replace(staging, target)
    _run_conda_unpack(target)

    with open(marker, "w", encoding="utf-8") as f:
        f.write(signature)
    print("Bundled Montreal Forced Aligner environment is ready.")
    return target


def run_mfa_alignment(
    *,
    script_path: str,
    input_dir: str,
    conda_exe: str = "conda",
    env_name: str = "mfa_env1",
) -> int | None:
    """
    Run `script_path` (mfa_align.py) with the Python interpreter of the MFA environment.

    Uses the bundled environment when available, otherwise `conda run -n env_name`.
    Returns the process exit code, or None if the bundled environment could not be prepared.
    """
    script_path = os.path.abspath(script_path)
    archive = bundled_mfa_archive()
    if archive:
        try:
            env_dir = prepare_bundled_mfa_env(archive)
        except Exception as e:
            print(f"Could not prepare the bundled MFA environment: {e}")
            return None
        command = [os.path.join(env_dir, "python.exe"), script_path, "--input_dir", input_dir]
        env = _mfa_subprocess_env(env_dir)
    else:
        command = [conda_exe, "run", "-n", env_name, "python", script_path, "--input_dir", input_dir]
        env = None

    completed = subprocess.run(command, cwd=os.getcwd(), env=env)
    if completed.returncode != 0:
        print(f"MFA alignment process exited with code {completed.returncode}.")
    return completed.returncode


if __name__ == "__main__":
    # `python mfa_runner.py --prepare` (or `SAFE-T.exe --prepare-mfa`, see
    # packaging/runtime_hook.py) unpacks the bundled environment ahead of time.
    if "--prepare" in sys.argv:
        print(prepare_bundled_mfa_env())
