"""Prepare writable runtime assets and start the single-worker API process."""

import os
import shutil
from pathlib import Path


def prepare_chroma_runtime_copy() -> Path:
    seed = Path(os.environ["FINGUARD_CHROMA_SEED_PATH"])
    runtime = Path(os.environ["FINGUARD_CHROMA_PATH"])
    if not seed.is_dir():
        raise RuntimeError("Chroma seed directory is unavailable.")
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(seed, runtime)
    return runtime


def main() -> None:
    prepare_chroma_runtime_copy()
    os.execvp(
        "uvicorn",
        [
            "uvicorn",
            "src.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--workers",
            "1",
        ],
    )


if __name__ == "__main__":
    main()
