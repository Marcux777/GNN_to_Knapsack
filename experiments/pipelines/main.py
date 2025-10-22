"""
Unified experiment runner for the Knapsack GNN project.

Provides a single entry-point to:
    - prepare/generate datasets
    - train the GNN
    - evaluate multiple decoding strategies (sampling, lagrangian, warm-start ILP, etc.)
    - generate plots and consolidated summaries

Usage examples:
    python experiments/main.py full --device cpu
    python experiments/main.py train --generate-data
    python experiments/main.py evaluate --checkpoint-dir checkpoints/run_20251020_104533 --strategies sampling warm_start
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# Ensure project root is on sys.path for absolute imports when running as a module.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knapsack_gnn.data.generator import create_datasets, KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import KnapsackSampler, evaluate_model
from knapsack_gnn.models.pna import create_model
from knapsack_gnn.training.loop import train_model
from knapsack_gnn.eval.reporting import (
    print_evaluation_summary,
    save_results_to_json,
)
from experiments.visualization import (
    plot_optimality_gaps,
    plot_strategy_comparison,
)

# --------------------------------------------------------------------------- #
# Helper dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class DatasetBundle:
    train: KnapsackDataset
    val: KnapsackDataset
    test: KnapsackDataset


@dataclass
class GraphDatasetBundle:
    train: KnapsackGraphDataset
    val: KnapsackGraphDataset
    test: KnapsackGraphDataset


# --------------------------------------------------------------------------- #
# Dataset and model preparation utilities
# --------------------------------------------------------------------------- #


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_datasets(
    data_dir: Path,
    generate: bool,
    train_size: int,
    val_size: int,
    test_size: int,
    n_items_min: int,
    n_items_max: int,
    seed: int,
) -> DatasetBundle:
    """
    Ensure train/val/test knapsack datasets exist and return them.
    """
    ensure_dir(data_dir)
    train_path = data_dir / "train.pkl"
    val_path = data_dir / "val.pkl"
    test_path = data_dir / "test.pkl"

    need_generate = generate or not (
        train_path.exists() and val_path.exists() and test_path.exists()
    )

    if need_generate:
        print("\n[DATA] Generating fresh datasets...")
        create_datasets(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            n_items_range=(n_items_min, n_items_max),
            seed=seed,
            output_dir=str(data_dir),
        )
    else:
        print("\n[DATA] Using cached datasets from", data_dir)

    train = KnapsackDataset.load(str(train_path))
    val = KnapsackDataset.load(str(val_path))
    test = KnapsackDataset.load(str(test_path))

    return DatasetBundle(train=train, val=val, test=test)


def build_graph_datasets(
    bundle: DatasetBundle, normalize_features: bool = True
) -> GraphDatasetBundle:
    """
    Build PyG graph datasets. This step is cached inside KnapsackGraphDataset.
    """
    print("\n[DATA] Building graph datasets...")
    train_graph = KnapsackGraphDataset(bundle.train, normalize_features=normalize_features)
    val_graph = KnapsackGraphDataset(bundle.val, normalize_features=normalize_features)
    test_graph = KnapsackGraphDataset(bundle.test, normalize_features=normalize_features)
    return GraphDatasetBundle(train=train_graph, val=val_graph, test=test_graph)


def load_best_model(
    checkpoint_dir: Path,
    graph_bundle: GraphDatasetBundle,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: str,
) -> torch.nn.Module:
    """
    Recreate the model architecture, load the best checkpoint and move to device.
    """
    print(f"\n[MODEL] Loading best model from {checkpoint_dir}")
    checkpoint_path = checkpoint_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find best_model.pt in {checkpoint_dir}")

    model = create_model(
        dataset=graph_bundle.train,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def run_training(
    datasets: DatasetBundle,
    device: str,
    checkpoint_root: Path,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Tuple[Path, GraphDatasetBundle]:
    """
    Train the model and return (checkpoint_dir, graph_dataset_bundle).
    """
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    graph_bundle = build_graph_datasets(datasets)
    checkpoint_root = ensure_dir(checkpoint_root)
    run_dir = ensure_dir(checkpoint_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    model = create_model(
        dataset=graph_bundle.train,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    print("\n[TRAIN] Starting training loop...")
    model, history = train_model(
        model=model,
        train_dataset=graph_bundle.train,
        val_dataset=graph_bundle.val,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_dir=str(run_dir),
        device=device,
    )

    # Reload best weights for downstream evaluation
    best_ckpt = run_dir / "best_model.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Persist configuration
    config_path = run_dir / "config.json"
    with open(config_path, "w") as fp:
        json.dump(
            {
                "device": device,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
            },
            fp,
            indent=2,
        )
    print(f"[TRAIN] Configuration saved to {config_path}")
    print(f"[TRAIN] Checkpoints stored in {run_dir}")
    return run_dir, graph_bundle


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #


def build_sampler_kwargs(args: argparse.Namespace) -> Dict:
    return {
        "num_threads": args.threads,
        "compile_model": args.compile,
        "quantize": args.quantize,
    }


def build_strategy_kwargs(args: argparse.Namespace, strategy: str) -> Dict:
    if strategy == "sampling":
        return {
            "temperature": args.temperature,
            "sampling_schedule": tuple(args.sampling_schedule),
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
        }
    if strategy == "threshold":
        return {"threshold": args.threshold}
    if strategy == "adaptive":
        return {"n_trials": args.n_samples}
    if strategy == "lagrangian":
        return {
            "lagrangian_iters": args.lagrangian_iters,
            "lagrangian_tol": args.lagrangian_tol,
            "lagrangian_bias": args.lagrangian_bias,
        }
    if strategy == "warm_start":
        return {
            "temperature": args.temperature,
            "sampling_schedule": tuple(args.sampling_schedule),
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
            "fix_threshold": args.fix_threshold,
            "ilp_time_limit": args.ilp_time_limit,
            "max_hint_items": args.max_hint_items,
            "ilp_threads": args.ilp_threads,
        }
    if strategy == "sampling_repair":
        return {
            "temperature": args.temperature,
            "sampling_schedule": tuple(args.sampling_schedule),
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
        }
    if strategy == "warm_start_repair":
        return {
            "temperature": args.temperature,
            "sampling_schedule": tuple(args.sampling_schedule),
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
            "fix_threshold": args.fix_threshold,
            "ilp_time_limit": args.ilp_time_limit,
            "max_hint_items": args.max_hint_items,
            "ilp_threads": args.ilp_threads,
        }
    raise ValueError(f"Unknown strategy '{strategy}'")


def evaluate_strategy(
    model: torch.nn.Module,
    dataset: KnapsackGraphDataset,
    strategy: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict:
    sampler_kwargs = build_sampler_kwargs(args)
    strategy_kwargs = build_strategy_kwargs(args, strategy)

    print(f"\n[EVAL] Strategy = {strategy}")
    results = evaluate_model(
        model=model,
        dataset=dataset,
        strategy=strategy,
        device=args.device,
        sampler_kwargs=sampler_kwargs,
        **strategy_kwargs,
    )

    print_evaluation_summary(results)

    save_dir = ensure_dir(output_dir)
    result_path = save_dir / f"results_{strategy}.json"
    save_results_to_json(results, result_path)

    if results.get("gaps"):
        plot_path = save_dir / f"gaps_{strategy}.png"
        plot_optimality_gaps(
            results["gaps"],
            title=f"Optimality Gaps ({strategy})",
            save_path=str(plot_path),
        )
        print(f"[EVAL] Gap plot saved to {plot_path}")

    return results


def generate_solution_examples(
    model: torch.nn.Module,
    dataset: KnapsackGraphDataset,
    strategy: str,
    args: argparse.Namespace,
    output_dir: Path,
    num_examples: int = 3,
) -> None:
    if num_examples <= 0:
        return

    sampler_kwargs = build_sampler_kwargs(args)
    strategy_kwargs = build_strategy_kwargs(args, strategy)
    sampler = KnapsackSampler(model, args.device, **sampler_kwargs)

    for idx in range(min(num_examples, len(dataset))):
        data = dataset[idx]
        result = sampler.solve(data, strategy=strategy, **strategy_kwargs)
        # TODO: Implement plot_solution_comparison for individual solution visualization
        # plot_path = output_dir / f"solution_{strategy}_{idx}.png"
        # plot_solution_comparison(
        #     instance_data=data,
        #     predicted_solution=result["solution"],
        #     probabilities=result["probabilities"],
        #     title=f"{strategy} - Instance {idx} (Gap {result.get('optimality_gap', 0):.2f}%)",
        #     save_path=str(plot_path),
        # )


def summarize_results(results_map: Dict[str, Dict]) -> Dict:
    summary = {}
    for strategy, metrics in results_map.items():
        summary[strategy] = {
            "mean_gap": metrics.get("mean_gap"),
            "median_gap": metrics.get("median_gap"),
            "std_gap": metrics.get("std_gap"),
            "max_gap": metrics.get("max_gap"),
            "feasibility_rate": metrics.get("feasibility_rate"),
            "mean_inference_time_ms": metrics.get("mean_inference_time", 0) * 1000
            if metrics.get("mean_inference_time") is not None
            else None,
            "median_inference_time_ms": metrics.get("median_inference_time", 0) * 1000
            if metrics.get("median_inference_time") is not None
            else None,
            "mean_samples_used": metrics.get("mean_samples_used"),
            "mean_ilp_time_ms": metrics.get("mean_ilp_time", 0) * 1000
            if metrics.get("mean_ilp_time") is not None
            else None,
            "ilp_success_rate": metrics.get("ilp_success_rate"),
        }
    return summary


# --------------------------------------------------------------------------- #
# Pipeline runners
# --------------------------------------------------------------------------- #


def run_full_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    checkpoint_root = Path(args.checkpoint_root)

    datasets = prepare_datasets(
        data_dir=data_dir,
        generate=args.generate_data,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        n_items_min=args.n_items_min,
        n_items_max=args.n_items_max,
        seed=args.seed,
    )

    checkpoint_dir: Optional[Path] = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    graph_bundle: Optional[GraphDatasetBundle] = None
    model: Optional[torch.nn.Module] = None

    if args.skip_train:
        if checkpoint_dir is None:
            raise ValueError("When --skip-train is set you must provide --checkpoint-dir.")
        graph_bundle = build_graph_datasets(datasets)
        model = load_best_model(
            checkpoint_dir=checkpoint_dir,
            graph_bundle=graph_bundle,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=args.device,
        )
    else:
        checkpoint_dir, graph_bundle = run_training(
            datasets=datasets,
            device=args.device,
            checkpoint_root=checkpoint_root,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        model = load_best_model(
            checkpoint_dir=checkpoint_dir,
            graph_bundle=graph_bundle,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=args.device,
        )

    evaluation_dir = ensure_dir(checkpoint_dir / "evaluation")
    strategies = [s.strip() for s in args.strategies]
    print(f"\n[PIPELINE] Running strategies: {strategies}")

    results_map: Dict[str, Dict] = {}
    for strategy in strategies:
        results = evaluate_strategy(
            model=model,
            dataset=graph_bundle.test,
            strategy=strategy,
            args=args,
            output_dir=evaluation_dir,
        )
        results_map[strategy] = results
        generate_solution_examples(
            model=model,
            dataset=graph_bundle.test,
            strategy=strategy,
            args=args,
            output_dir=evaluation_dir,
            num_examples=args.visualize_examples,
        )

    summary = summarize_results(results_map)
    summary_path = evaluation_dir / "pipeline_summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n[PIPELINE] Summary saved to {summary_path}")


def run_train_only(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    checkpoint_root = Path(args.checkpoint_root)

    datasets = prepare_datasets(
        data_dir=data_dir,
        generate=args.generate_data,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        n_items_min=args.n_items_min,
        n_items_max=args.n_items_max,
        seed=args.seed,
    )

    run_training(
        datasets=datasets,
        device=args.device,
        checkpoint_root=checkpoint_root,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


def run_evaluate_only(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    datasets = prepare_datasets(
        data_dir=data_dir,
        generate=False,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        n_items_min=args.n_items_min,
        n_items_max=args.n_items_max,
        seed=args.seed,
    )
    graph_bundle = build_graph_datasets(datasets)
    model = load_best_model(
        checkpoint_dir=checkpoint_dir,
        graph_bundle=graph_bundle,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device,
    )

    evaluation_dir = ensure_dir(checkpoint_dir / "evaluation")
    strategies = [s.strip() for s in args.strategies]
    results_map: Dict[str, Dict] = {}
    for strategy in strategies:
        results = evaluate_strategy(
            model=model,
            dataset=graph_bundle.test,
            strategy=strategy,
            args=args,
            output_dir=evaluation_dir,
        )
        results_map[strategy] = results

    summary = summarize_results(results_map)
    summary_path = (
        evaluation_dir / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n[EVAL] Summary saved to {summary_path}")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir", type=str, default="data/datasets", help="Directory containing the datasets."
    )
    parser.add_argument(
        "--checkpoint-root", type=str, default="checkpoints", help="Root directory for checkpoints."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Specific checkpoint directory for evaluation.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu, cuda, auto)."
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="Set torch.set_num_threads and sampler threads."
    )
    parser.add_argument(
        "--compile", action="store_true", help="Enable torch.compile for inference."
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Apply dynamic quantization to linear layers."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Dataset parameters
    parser.add_argument(
        "--generate-data", action="store_true", help="Force regeneration of datasets."
    )
    parser.add_argument("--train-size", type=int, default=1000, help="Training dataset size.")
    parser.add_argument("--val-size", type=int, default=200, help="Validation dataset size.")
    parser.add_argument("--test-size", type=int, default=200, help="Test dataset size.")
    parser.add_argument(
        "--n-items-min", type=int, default=10, help="Minimum number of items per instance."
    )
    parser.add_argument(
        "--n-items-max", type=int, default=50, help="Maximum number of items per instance."
    )

    # Model/training parameters
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for the GNN.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of PNA layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.002, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay.")

    # Evaluation parameters
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["sampling", "warm_start"],
        help="List of decoding strategies to evaluate.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="Number of samples for sampling/adaptive strategies.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for threshold strategy."
    )
    parser.add_argument(
        "--sampling-schedule",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="Batch sizes for vectorised sampling.",
    )
    parser.add_argument(
        "--sampling-tolerance",
        type=float,
        default=1e-3,
        help="Early-stopping tolerance for sampling.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=128, help="Maximum samples budget for adaptive sampling."
    )
    parser.add_argument(
        "--lagrangian-iters", type=int, default=30, help="Iterations for Lagrangian decoder."
    )
    parser.add_argument(
        "--lagrangian-tol", type=float, default=1e-4, help="Tolerance for Lagrangian decoder."
    )
    parser.add_argument(
        "--lagrangian-bias",
        type=float,
        default=0.0,
        help="Probability bias for Lagrangian decoder.",
    )
    parser.add_argument(
        "--fix-threshold",
        type=float,
        default=0.9,
        help="Probability threshold to fix variables in warm-start ILP.",
    )
    parser.add_argument(
        "--ilp-time-limit", type=float, default=1.0, help="Time limit (seconds) for warm-start ILP."
    )
    parser.add_argument(
        "--max-hint-items",
        type=int,
        default=None,
        help="Maximum number of hints passed to ILP solver.",
    )
    parser.add_argument(
        "--ilp-threads", type=int, default=None, help="Number of threads for ILP solver."
    )
    parser.add_argument(
        "--visualize-examples",
        type=int,
        default=0,
        help="Number of solution plots to generate per strategy.",
    )

    parser.add_argument(
        "--skip-train", action="store_true", help="Skip training step (requires --checkpoint-dir)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified experiment runner for Knapsack GNN.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Full pipeline
    full_parser = subparsers.add_parser("full", help="Run training + evaluation pipeline.")
    add_common_arguments(full_parser)

    # Train only
    train_parser = subparsers.add_parser("train", help="Only train the model.")
    add_common_arguments(train_parser)

    # Evaluate only
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate existing checkpoints.")
    add_common_arguments(eval_parser)
    args = parser.parse_args()

    # Tidy schedule list for further processing
    args.sampling_schedule = [int(v) for v in args.sampling_schedule]
    return args


# --------------------------------------------------------------------------- #
# Main entry-point
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()

    # Normalise device selection
    device = args.device.lower()
    if device in ("auto", "best"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    if args.command == "full":
        run_full_pipeline(args)
    elif args.command == "train":
        run_train_only(args)
    elif args.command == "evaluate":
        if args.checkpoint_dir is None:
            raise ValueError("Please provide --checkpoint-dir for evaluation.")
        run_evaluate_only(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
