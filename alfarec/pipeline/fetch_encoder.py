"""Fetch only revision-pinned public model files at image build time."""

import argparse
import json
from pathlib import Path

from alfarec.recsys.retrieval import digest


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/ml"))
    args = parser.parse_args()
    manifest = json.loads((args.artifacts / "manifest.json").read_text("utf-8"))
    destination = args.artifacts / "model"
    snapshot_download(
        manifest["model_id"],
        revision=manifest["revision"],
        allow_patterns=list(manifest["model_files"]),
        local_dir=destination,
        token=False,
    )
    for name, checksum in manifest["model_files"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or digest(destination / relative) != checksum
        ):
            raise ValueError("Downloaded encoder checksum mismatch")


if __name__ == "__main__":
    main()
