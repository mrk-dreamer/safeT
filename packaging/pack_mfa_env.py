"""
Pack a Conda environment via conda-pack, taking files that exist in the live
environment whenever a package's files were modified/overwritten by pip.

Why this exists:
`conda-pack` normally reads conda-managed files from the global Conda package
cache (`info['link']['source']`) rather than from the environment folder itself.
If a package in the environment was later downgraded/upgraded via `pip` (for
example `numpy 1.26.4` installed via pip over `conda`'s `numpy 2.2.5`), using
`conda pack --ignore-missing-files` mixes files from the old conda cache with
files from the live pip install, corrupting the packed environment.

By clearing `link.source` in memory for any conda-meta package that has missing
or modified files, `conda-pack` falls back to its built-in uncached-package
path, which packs the exact files present in the live environment and still
creates `conda-unpack` to relocate prefixes on the target machine.
"""

import json
import os
import sys

import conda_pack
import conda_pack.core as _cp_core


def _patch_conda_pack_to_use_live_env() -> None:
    """
    Ensure conda-pack always reads file contents from the live environment
    (`prefix`) rather than from the global `pkgs/` cache (`link.source`), while
    preserving `prefix_placeholder` and `file_mode` metadata from `paths.json`.
    """
    orig_load_managed = _cp_core.load_managed_package

    def _load_managed_from_live_env(info, prefix, site_packages, all_files):
        files = orig_load_managed(info, prefix, site_packages, all_files)
        targets = {os.path.normcase(f.target) for f in files}
        clobbered = bool(targets.difference(all_files))
        if clobbered:
            print(
                f"[pack_mfa_env] Package {info.get('name')} {info.get('version')} "
                "was modified by pip; packing live environment files instead of conda cache."
            )
        return [
            _cp_core.File(
                os.path.join(prefix, all_files.get(os.path.normcase(f.target), f.target)),
                all_files.get(os.path.normcase(f.target), f.target),
                is_conda=False if clobbered else f.is_conda,
                prefix_placeholder=None if clobbered else f.prefix_placeholder,
                file_mode="unknown" if clobbered else f.file_mode,
            )
            for f in files
        ]

    _cp_core.load_managed_package = _load_managed_from_live_env


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python pack_mfa_env.py <env_name_or_prefix> <output_tar_path>")
        return 2

    env_name_or_prefix = sys.argv[1]
    output_tar = os.path.abspath(sys.argv[2])

    _patch_conda_pack_to_use_live_env()

    kwargs = {"output": output_tar, "force": True, "ignore_missing_files": True}
    if os.path.isdir(env_name_or_prefix):
        kwargs["prefix"] = env_name_or_prefix
    else:
        kwargs["name"] = env_name_or_prefix

    print(f"Packing '{env_name_or_prefix}' -> '{output_tar}' ...")
    conda_pack.pack(**kwargs)
    print("MFA environment packed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
