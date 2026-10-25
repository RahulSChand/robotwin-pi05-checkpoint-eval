#!/usr/bin/env python3
"""Download RoboTwin assets and apply its pinned runtime compatibility patches."""

from __future__ import annotations

import re
import site
import subprocess
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
ROBOTWIN = ROOT / "RoboTwin"


def patch_installed_packages() -> None:
    patched = []
    for directory in map(Path, site.getsitepackages()):
        mplib_planner = directory / "mplib" / "planner.py"
        if mplib_planner.exists():
            source = mplib_planner.read_text()
            updated = re.sub(r"\bor collide\b", "", source, count=1)
            if updated != source:
                mplib_planner.write_text(updated)
                patched.append(str(mplib_planner))

        sapien_loader = directory / "sapien" / "wrapper" / "urdf_loader.py"
        if sapien_loader.exists():
            source = sapien_loader.read_text()
            updated = source.replace(
                "with open(srdf_path) as f:",
                'with open(srdf_path, encoding="utf-8") as f:',
            ).replace('"srdf"', '".srdf"')
            if updated != source:
                sapien_loader.write_text(updated)
                patched.append(str(sapien_loader))
    print(f"patched_files={patched}")


def download_assets() -> None:
    assets = ROBOTWIN / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for filename in ("embodiments.zip", "objects.zip"):
        archive = Path(
            hf_hub_download(
                repo_id="TianxingChen/RoboTwin2.0",
                repo_type="dataset",
                filename=filename,
                local_dir=assets,
            )
        )
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(assets)
        archive.unlink()
        print(f"extracted={filename}")


def update_embodiment_paths() -> None:
    subprocess.run(
        [sys.executable, "script/update_embodiment_config_path.py"],
        cwd=ROBOTWIN,
        check=True,
    )


def main() -> None:
    if not (ROBOTWIN / "script" / "update_embodiment_config_path.py").is_file():
        raise FileNotFoundError(f"Pinned RoboTwin checkout not found at {ROBOTWIN}")
    patch_installed_packages()
    download_assets()
    update_embodiment_paths()


if __name__ == "__main__":
    main()
