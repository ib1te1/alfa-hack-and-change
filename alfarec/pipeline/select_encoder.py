"""Select the deployable neural encoder using only tune metrics and resource limits."""

import argparse
import json
from pathlib import Path

from alfarec.recsys.retrieval import write_json


def select(manifests):
    candidates = [
        m
        for m in manifests
        if m["candidate"] != "tfidf" and m["weights_bytes"] <= 500_000_000 and m["cpu_p95_ms"] < 150
    ]
    if not candidates:
        raise ValueError("No neural candidate meets CPU and size targets")
    return max(
        candidates,
        key=lambda m: (m["metrics"]["tune_ndcg"]["mean"], -m["weights_bytes"], m["candidate"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    chosen = select([json.loads(p.read_text("utf-8")) for p in a.manifests])
    write_json(
        a.output,
        {
            "candidate": chosen["candidate"],
            "model_id": chosen["model_id"],
            "revision": chosen["revision"],
            "rule": "max neural tune nDCG, CPU p95 <150ms, files <=500MB; holdout not consulted",
        },
    )


if __name__ == "__main__":
    main()
