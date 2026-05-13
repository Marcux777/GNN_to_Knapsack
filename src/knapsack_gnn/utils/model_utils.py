"""
Utilities for loading trained models and datasets for inference/export tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import torch

from knapsack_gnn.data.generator import KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.models.pna import create_model
from knapsack_gnn.utils.feature_flags import parse_graph_feature_spec


def _parse_config_file(config_path: Path) -> dict[str, Any]:
    """Parse a simple key:value config file written by the training pipeline."""
    config: dict[str, Any] = {}
    if not config_path.exists():
        return config
    with config_path.open() as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = value.strip()
    return config


def _model_hyperparams_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract model hyperparameters with safe fallbacks."""
    hidden_dim = int(config.get("hidden_dim", 64))
    num_layers = int(config.get("num_layers", 3))
    dropout = float(config.get("dropout", 0.1))
    return {"hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout}


def load_graph_dataset(
    split: str,
    data_dir: str,
    normalize_features: bool = True,
    graph_features: dict[str, Any] | None = None,
) -> KnapsackGraphDataset:
    """
    Load a serialized knapsack dataset split and wrap it as a graph dataset.

    Args:
        split: One of ``train``, ``val``, ``test``.
        data_dir: Directory that contains ``<split>.pkl`` files.
        normalize_features: Whether to normalize node features (default: True).

    Returns:
        KnapsackGraphDataset ready for PyG models.
    """
    dataset_path = Path(data_dir) / f"{split}.pkl"
    knapsack_dataset = KnapsackDataset.load(str(dataset_path))
    return KnapsackGraphDataset(
        knapsack_dataset,
        normalize_features=normalize_features,
        graph_features=graph_features,
    )


def load_model_from_checkpoint(
    checkpoint_dir: str | Path,
    checkpoint_name: str,
    data_dir: str,
    device: str,
) -> Tuple[torch.nn.Module, KnapsackGraphDataset]:
    """
    Load a trained KnapsackPNA model from a checkpoint directory.

    Args:
        checkpoint_dir: Directory containing trained model artifacts.
        checkpoint_name: Name of the checkpoint file (e.g., ``best_model.pt``).
        data_dir: Directory where ``train.pkl`` resides (used for degree histogram).
        device: Target device (``cpu`` or ``cuda``).

    Returns:
        Tuple of ``(model, train_graph_dataset)`` ready for inference.
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_file = checkpoint_path / checkpoint_name
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    config = _parse_config_file(checkpoint_path / "config.txt")
    hyperparams = _model_hyperparams_from_config(config)
    graph_spec = config.get("graph_features")
    bucket_cfg = config.get("graph_feature_buckets")

    if not graph_spec or graph_spec.lower() == "auto":
        json_path = checkpoint_path / "config.json"
        if json_path.exists():
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    json_config = json.load(handle)
                graph_spec = json_config.get("graph_features", graph_spec)
                if bucket_cfg is None:
                    bucket_cfg = json_config.get("graph_feature_buckets")
            except Exception:
                pass

    bucket_value = None
    if isinstance(bucket_cfg, str):
        bucket_value = int(bucket_cfg) if bucket_cfg.isdigit() else None
    elif bucket_cfg is not None:
        try:
            bucket_value = int(bucket_cfg)
        except (TypeError, ValueError):
            bucket_value = None
    graph_feature_kwargs = parse_graph_feature_spec(graph_spec, bucket_value)

    train_graph_dataset = load_graph_dataset(
        "train",
        data_dir,
        graph_features=graph_feature_kwargs,
    )

    model = create_model(
        dataset=train_graph_dataset,
        hidden_dim=hyperparams["hidden_dim"],
        num_layers=hyperparams["num_layers"],
        dropout=hyperparams["dropout"],
    )

    checkpoint = torch.load(checkpoint_file, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, train_graph_dataset


def apply_dynamic_quantization(model: torch.nn.Module, dtype: torch.dtype = torch.qint8) -> torch.nn.Module:
    """Apply dynamic quantization to Linear layers, if torch.ao.quantization is available."""
    try:
        from torch.ao.quantization import quantize_dynamic
    except Exception:  # pragma: no cover - optional dependency
        return model

    quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=dtype)
    return quantized


def maybe_compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """Compile the model with torch.compile if available."""
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:  # pragma: no cover - depends on torch version
        return model
    try:
        compiled = compile_fn(model, mode="reduce-overhead", fullgraph=False)
        return compiled
    except Exception:  # pragma: no cover - compilation can fail on unsupported ops
        return model
