"""Checked, catalog-bound ranking weights shared by serving and offline evaluation."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

from alfarec.recsys.retrieval import digest

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "weights.json"


def validate_weights(value: object, *, complete: bool = False) -> dict[str, float]:
    from alfarec.recsys.ranking import DEFAULT_WEIGHTS

    if not isinstance(value, Mapping) or set(value) - DEFAULT_WEIGHTS.keys():
        raise ValueError("Unknown or invalid ranking weights")
    if complete and set(value) != DEFAULT_WEIGHTS.keys():
        raise ValueError("Weight artifact must contain every ranking feature")
    result = {}
    for name, weight in value.items():
        if type(weight) not in (int, float) or not 0 <= weight <= 1000 or not math.isfinite(weight):
            raise ValueError(f"Weight {name} must be finite and between 0 and 1000")
        result[name] = float(weight)
    if complete and not any(result.values()):
        raise ValueError("At least one ranking weight must be positive")
    return result


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate weight artifact key: {key}")
        result[key] = value
    return result


def load_weights(path: Path, catalog_path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text("utf-8"), object_pairs_hook=_unique_keys)
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "weights",
        "catalog_sha256",
        "provenance",
    }:
        raise ValueError("Invalid weight artifact fields")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("Unsupported weight artifact version")
    if payload["catalog_sha256"] != digest(catalog_path):
        raise ValueError("Weights belong to a different catalog")
    if not isinstance(payload["provenance"], dict) or not payload["provenance"]:
        raise ValueError("Weights must record their selection provenance")
    return validate_weights(payload["weights"], complete=True)


def weight_artifact(weights: Mapping[str, float], catalog_path: Path, provenance: dict) -> dict:
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Weights must record their selection provenance")
    return {
        "version": 1,
        "weights": validate_weights(weights, complete=True),
        "catalog_sha256": digest(catalog_path),
        "provenance": provenance,
    }
