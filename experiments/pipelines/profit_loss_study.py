"""
Profit-loss weight × dropout calibration study.

Runs a grid over profit_loss_weight and dropout values across multiple seeds,
training the GNN and collecting evaluation metrics for each combination.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from knapsack_gnn.data.generator import KnapsackDataset, create_datasets
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import evaluate_model
from knapsack_gnn.eval.reporting import save_results_to_json
from knapsack_gnn.models.pna import create_model
from knapsack_gnn.training.loop import train_model
from knapsack_gnn.training.utils import set_seed, validate_seed
from knapsack_gnn.utils.feature_flags import resolve_graph_feature_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profit loss weight calibration study")
    parser.add_argument("--data_dir", type=str, default="data/datasets")
    parser.add_argument("--generate_data", action="store_true")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--n_items_min", type=int, default=10)
    parser.add_argument("--n_items_max", type=int, default=50)
    parser.add_argument(
        "--graph_features",
        type=str,
        default="none",
        help="Optional graph feature spec (density,quadratic,bucket,all,none).",
    )
    parser.add_argument(
        "--graph_feature_buckets",
        type=int,
        default=4,
        help="Bucket count for bucketized ranks (default: 4).",
    )

    parser.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 47, 59])

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument(
        "--strategy",
        type=str,
        default="sampling",
        choices=["threshold", "sampling", "adaptive", "lagrangian", "warm_start"],
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sampling_schedule", type=str, default="32,64,128")
    parser.add_argument("--sampling_tolerance", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--output_dir", type=str, default="results/profit_loss_study")
    return parser.parse_args()


def load_or_create_datasets(args: argparse.Namespace) -> Tuple[KnapsackDataset, ...]:
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.pkl"
    val_path = data_dir / "val.pkl"
    test_path = data_dir / "test.pkl"

    if (
        args.generate_data
        or not train_path.exists()
        or not val_path.exists()
        or not test_path.exists()
    ):
        print("Generating datasets...")
        train_ds, val_ds, test_ds = create_datasets(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            n_items_range=(args.n_items_min, args.n_items_max),
            seed=args.seeds[0],
            output_dir=str(data_dir),
        )
        return train_ds, val_ds, test_ds

    print("Loading cached datasets...")
    return (
        KnapsackDataset.load(str(train_path)),
        KnapsackDataset.load(str(val_path)),
        KnapsackDataset.load(str(test_path)),
    )


def build_graph_datasets(
    datasets: Tuple[KnapsackDataset, KnapsackDataset, KnapsackDataset],
    graph_features: dict[str, Any],
) -> Tuple[KnapsackGraphDataset, KnapsackGraphDataset, KnapsackGraphDataset]:
    train, val, test = datasets
    return (
        KnapsackGraphDataset(train, normalize_features=True, graph_features=graph_features),
        KnapsackGraphDataset(val, normalize_features=True, graph_features=graph_features),
        KnapsackGraphDataset(test, normalize_features=True, graph_features=graph_features),
    )


def train_and_evaluate(
    weight: float,
    dropout: float,
    seed: int,
    graph_datasets: Tuple[KnapsackGraphDataset, ...],
    args: argparse.Namespace,
    sampler_kwargs: dict,
) -> dict:
    train_graph, val_graph, test_graph = graph_datasets

    validate_seed(seed)
    set_seed(seed, deterministic=True)

    model = create_model(
        dataset=train_graph,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=dropout,
    ).to(args.device)

    combo_dir = Path(args.output_dir) / f"w{weight:.3f}_d{dropout:.2f}" / f"seed_{seed}"
    combo_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    trained_model, history = train_model(
        model=model,
        train_dataset=train_graph,
        val_dataset=val_graph,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=str(combo_dir),
        device=args.device,
        seed=seed,
        profit_loss_weight=weight,
        plot_curves=False,
    )
    train_time = time.perf_counter() - start

    strategy_kwargs: dict[str, Any] = {}
    if args.strategy == "sampling":
        strategy_kwargs = {
            "temperature": args.temperature,
            "sampling_schedule": tuple(int(x) for x in args.sampling_schedule.split(",")),
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
        }
    elif args.strategy == "threshold":
        strategy_kwargs = {"threshold": 0.5}
    elif args.strategy == "adaptive":
        strategy_kwargs = {"n_trials": args.n_samples}
    elif args.strategy == "lagrangian":
        strategy_kwargs = {
            "lagrangian_iters": 30,
            "lagrangian_tol": 1e-4,
            "lagrangian_bias": 0.0,
        }
    elif args.strategy == "warm_start":
        strategy_kwargs = {
            "temperature": args.temperature,
            "sampling_schedule": tuple(int(x) for x in args.sampling_schedule.split(",")),
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
            "fix_threshold": 0.9,
            "ilp_time_limit": 1.0,
        }

    eval_results = evaluate_model(
        model=trained_model,
        dataset=test_graph,
        strategy=args.strategy,
        device=args.device,
        sampler_kwargs=sampler_kwargs,
        **strategy_kwargs,
    )

    return {
        "weight": weight,
        "dropout": dropout,
        "seed": seed,
        "train_time_s": train_time,
        "train_loss_final": history["train_loss"][-1],
        "val_loss_final": history["val_loss"][-1],
        "mean_gap": eval_results.get("mean_gap"),
        "median_gap": eval_results.get("median_gap"),
        "max_gap": eval_results.get("max_gap"),
        "feasibility_rate": eval_results.get("feasibility_rate"),
        "mean_inference_time": eval_results.get("mean_inference_time"),
    }


def aggregate_by_combo(records: Iterable[dict]) -> List[dict]:
    grouped: Dict[tuple, List[dict]] = {}
    for record in records:
        key = (record["weight"], record["dropout"])
        grouped.setdefault(key, []).append(record)

    summary: List[dict] = []
    for (weight, dropout), items in grouped.items():
        gaps = [it["mean_gap"] for it in items if it["mean_gap"] is not None]
        medians = [it["median_gap"] for it in items if it["median_gap"] is not None]
        maxes = [it["max_gap"] for it in items if it["max_gap"] is not None]
        times = [it["train_time_s"] for it in items]
        summary.append(
            {
                "weight": weight,
                "dropout": dropout,
                "n_runs": len(items),
                "mean_gap_mean": float(np.mean(gaps)) if gaps else None,
                "mean_gap_std": float(np.std(gaps)) if gaps else None,
                "median_gap_mean": float(np.mean(medians)) if medians else None,
                "max_gap_mean": float(np.mean(maxes)) if maxes else None,
                "train_time_mean": float(np.mean(times)),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    graph_feature_kwargs, graph_feature_spec = resolve_graph_feature_kwargs(
        args.graph_features,
        args.graph_feature_buckets,
    )
    print(f"Using graph feature spec: {graph_feature_spec}")

    datasets = load_or_create_datasets(args)
    graph_datasets = build_graph_datasets(datasets, graph_features=graph_feature_kwargs)

    sampler_kwargs = {"num_threads": None, "compile_model": False, "quantize": False}
    records: list[dict] = []

    combos = list(itertools.product(args.weights, args.dropouts))
    for weight, dropout in combos:
        print(f"\n=== Combination: weight={weight} | dropout={dropout} ===")
        for seed in args.seeds:
            record = train_and_evaluate(weight, dropout, seed, graph_datasets, args, sampler_kwargs)
            records.append(record)

    results_path = Path(args.output_dir) / "raw_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as handle:
        json.dump(records, handle, indent=2)
    print(f"\nRaw results saved to {results_path}")

    summary = aggregate_by_combo(records)
    summary_path = Path(args.output_dir) / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
