"""
Utility helpers for configuring optional graph-builder features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

FEATURE_ALIASES = {
    "density": "enable_density",
    "quadratic": "enable_quadratic_ratio",
    "bucket": "enable_bucket_ranks",
}


def parse_graph_feature_spec(spec: str | None, bucket_count: int | None = None) -> dict[str, Any]:
    """
    Parse a comma-separated feature spec into builder kwargs.

    Args:
        spec: String such as ``"density,bucket"`` or keywords ``none`` / ``all``.
        bucket_count: Number of buckets when bucketized ranks are enabled.
    """
    normalized = (spec or "").strip().lower()
    flags: Dict[str, bool] = {name: False for name in FEATURE_ALIASES.values()}

    if normalized in ("", "none", "basic"):
        pass
    elif normalized in ("all", "extended"):
        for key in flags:
            flags[key] = True
    else:
        tokens = [token.strip() for token in normalized.split(",") if token.strip()]
        for token in tokens:
            if token not in FEATURE_ALIASES:
                raise ValueError(
                    f"Unknown graph feature '{token}'. "
                    f"Supported: {', '.join(sorted(FEATURE_ALIASES))}"
                )
            flags[FEATURE_ALIASES[token]] = True

    buckets = bucket_count if bucket_count and bucket_count > 0 else 4

    kwargs: dict[str, Any] = {
        "enable_density": flags["enable_density"],
        "enable_quadratic_ratio": flags["enable_quadratic_ratio"],
        "enable_bucket_ranks": flags["enable_bucket_ranks"],
        "buckets": buckets,
    }
    return kwargs


def feature_flags_to_spec(flags: dict[str, Any]) -> str:
    """
    Convert builder kwargs back to a comma-separated spec string.
    """
    parts: list[str] = []
    if flags.get("enable_density"):
        parts.append("density")
    if flags.get("enable_quadratic_ratio"):
        parts.append("quadratic")
    if flags.get("enable_bucket_ranks"):
        parts.append("bucket")
    return ",".join(parts) if parts else "none"


def read_graph_feature_config(checkpoint_dir: str | Path) -> tuple[str | None, int | None]:
    """
    Read the graph-feature specification stored in ``config.txt`` inside a checkpoint directory.
    """
    base = Path(checkpoint_dir)
    config_path = base / "config.txt"
    spec: str | None = None
    buckets: int | None = None

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "graph_features":
                    spec = value or None
                elif key == "graph_feature_buckets":
                    try:
                        buckets = int(value)
                    except ValueError:
                        buckets = None
        if spec or buckets is not None:
            return spec, buckets

    json_path = base / "config.json"
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            spec = data.get("graph_features", spec)
            bucket_value = data.get("graph_feature_buckets")
            if bucket_value is not None:
                try:
                    buckets = int(bucket_value)
                except (TypeError, ValueError):
                    buckets = buckets
        except Exception:
            pass

    return spec, buckets


def resolve_graph_feature_kwargs(
    requested_spec: str | None,
    bucket_count: int | None,
    *,
    checkpoint_dir: str | Path | None = None,
    fallback: str = "none",
) -> Tuple[dict[str, Any], str]:
    """
    Resolve the feature spec to use (honouring ``auto``/``config`` requests) and return builder kwargs.

    Args:
        requested_spec: Spec string from CLI/config (may be ``auto``).
        bucket_count: Number of buckets when bucketized ranks are enabled.
        checkpoint_dir: Optional checkpoint directory whose ``config.txt`` stores defaults.
        fallback: Spec to use when nothing else is provided (defaults to ``none``).
    """
    config_spec: str | None = None
    config_buckets: int | None = None
    if checkpoint_dir is not None:
        config_spec, config_buckets = read_graph_feature_config(checkpoint_dir)

    normalized = (requested_spec or "").strip().lower()
    if normalized in ("", "auto", "config", "same"):
        resolved_spec = config_spec or fallback
    else:
        resolved_spec = requested_spec or fallback

    resolved_buckets = bucket_count if bucket_count and bucket_count > 0 else config_buckets

    kwargs = parse_graph_feature_spec(resolved_spec, resolved_buckets)
    return kwargs, resolved_spec or "none"
