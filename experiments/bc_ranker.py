"""
Supervised Behavioral Cloning (BC) ranker experiment for 0/1 Knapsack.

Trains a GNN ranker (PNA/GCN/GAT) with BCE + profit-gap hinge loss and evaluates
using greedy masked decoding. Designed as the first step of the L2O plan.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from knapsack_gnn.data.generator import KnapsackDataset, create_datasets
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.models.gat import create_gat_model
from knapsack_gnn.models.gcn import create_gcn_model
from knapsack_gnn.models.pna import create_model as create_pna_model
from knapsack_gnn.training.loop import KnapsackTrainer, greedy_masked_selection
from knapsack_gnn.training.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BC Ranker for 0/1 Knapsack")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--n_items_min", type=int, default=10)
    parser.add_argument("--n_items_max", type=int, default=50)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--generate_data", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data/datasets")

    parser.add_argument("--architecture", type=str, default="pna", choices=["pna", "gcn", "gat"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--profit_loss_weight", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 73])
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", type=str, default="results/bc_ranker")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints/bc_ranker")
    parser.add_argument(
        "--plot_curves", action="store_true", help="Save loss curves (requires display backend)"
    )

    return parser.parse_args()


def load_or_create_datasets(
    args: argparse.Namespace,
) -> tuple[KnapsackDataset, KnapsackDataset, KnapsackDataset]:
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.pkl"
    val_path = data_dir / "val.pkl"
    test_path = data_dir / "test.pkl"

    if args.generate_data or not (train_path.exists() and val_path.exists() and test_path.exists()):
        print("Generating fresh datasets...")
        return create_datasets(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            n_items_range=(args.n_items_min, args.n_items_max),
            seed=args.dataset_seed,
            output_dir=str(data_dir),
        )

    print("Loading datasets from disk...")
    return (
        KnapsackDataset.load(str(train_path)),
        KnapsackDataset.load(str(val_path)),
        KnapsackDataset.load(str(test_path)),
    )


def build_model(
    architecture: str, dataset: KnapsackGraphDataset, args: argparse.Namespace
) -> torch.nn.Module:
    if architecture == "pna":
        return create_pna_model(
            dataset=dataset,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    if architecture == "gcn":
        return create_gcn_model(
            hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout
        )
    if architecture == "gat":
        return create_gat_model(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_heads=4,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def evaluate_greedy(
    model: torch.nn.Module, dataset: KnapsackGraphDataset, device: str
) -> dict[str, float]:
    model.eval()
    gaps: list[float] = []
    feas_flags: list[bool] = []

    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            probs = model(data)
            solution = greedy_masked_selection(probs, data.item_weights, float(data.capacity))
            weights = data.item_weights.detach().cpu().numpy()
            values = data.item_values.detach().cpu().numpy()

            total_weight = float(np.dot(solution.numpy(), weights))
            total_value = float(np.dot(solution.numpy(), values))

            feasible = total_weight <= data.capacity + 1e-6
            feas_flags.append(feasible)

            if hasattr(data, "optimal_value") and data.optimal_value > 0:
                opt = float(data.optimal_value)
                gap = max(0.0, (opt - total_value) / opt * 100)
                gaps.append(gap)

    gaps_np = np.array(gaps) if gaps else np.array([0.0])
    return {
        "mean_gap": float(np.mean(gaps_np)),
        "median_gap": float(np.median(gaps_np)),
        "feasibility_rate": float(np.mean(feas_flags)) if feas_flags else 0.0,
        "num_instances": len(dataset),
    }


def run_seed(
    seed: int,
    args: argparse.Namespace,
    train_graphs: KnapsackGraphDataset,
    val_graphs: KnapsackGraphDataset,
    test_graphs: KnapsackGraphDataset,
) -> dict[str, float]:
    print(f"\n=== Seed {seed} ===")
    set_seed(seed)

    model = build_model(args.architecture, train_graphs, args)
    checkpoint_dir = Path(args.checkpoint_root) / f"{args.architecture}_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = KnapsackTrainer(
        model=model,
        train_dataset=train_graphs,
        val_dataset=val_graphs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        profit_loss_weight=args.profit_loss_weight,
    )
    trainer.train(num_epochs=args.epochs)
    if args.plot_curves:
        trainer.plot_training_curves(save_path=str(checkpoint_dir / "training_curves.png"))

    metrics = evaluate_greedy(model, test_graphs, args.device)
    metrics.update({"seed": seed})

    with open(checkpoint_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(trainer.history, f, indent=2)

    return metrics


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_root, exist_ok=True)

    train_ds, val_ds, test_ds = load_or_create_datasets(args)
    train_graphs = KnapsackGraphDataset(train_ds, normalize_features=True)
    val_graphs = KnapsackGraphDataset(val_ds, normalize_features=True)
    test_graphs = KnapsackGraphDataset(test_ds, normalize_features=True)

    all_metrics: list[dict[str, float]] = []
    for seed in args.seeds:
        metrics = run_seed(seed, args, train_graphs, val_graphs, test_graphs)
        all_metrics.append(metrics)

    mean_gap = float(np.mean([m["mean_gap"] for m in all_metrics]))
    median_gap = float(np.median([m["median_gap"] for m in all_metrics]))
    feas_rate = float(np.mean([m["feasibility_rate"] for m in all_metrics]))

    summary = {
        "architecture": args.architecture,
        "seeds": args.seeds,
        "mean_gap_mean": mean_gap,
        "median_gap_mean": median_gap,
        "feasibility_rate_mean": feas_rate,
    }
    with open(Path(args.output_dir) / f"summary_{args.architecture}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
