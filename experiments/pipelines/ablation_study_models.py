"""
Ablation Study: Model Architecture Comparison

Trains and compares different GNN architectures:
    - PNA (current baseline)
    - GCN (simpler alternative)
    - GAT (attention-based)
    - PNA with different depths (2/3/4 layers)

Goal: Prove that PNA-3 layers dominates alternatives in p95 gap with acceptable cost.

Usage:
    python experiments/pipelines/ablation_study_models.py \
        --data-dir data/datasets \
        --output-dir checkpoints/ablation \
        --models pna gcn gat \
        --depths 2 3 4 \
        --epochs 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from knapsack_gnn.data.generator import KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import evaluate_model
from knapsack_gnn.models.gat import KnapsackGAT
from knapsack_gnn.models.gcn import KnapsackGCN
from knapsack_gnn.models.pna import create_model as create_pna_model
from knapsack_gnn.training.loop import train_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_model_by_name(
    model_name: str,
    dataset: KnapsackGraphDataset,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
):
    """
    Create model by name.

    Args:
        model_name: 'pna', 'gcn', or 'gat'
        dataset: Graph dataset for degree histogram (PNA only)
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate

    Returns:
        Model instance
    """
    if model_name == "pna":
        return create_pna_model(
            dataset=dataset,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_name == "gcn":
        # Get feature dimensions from dataset
        dataset[0]
        item_dim = 2  # weight, value
        constraint_dim = 1  # capacity

        return KnapsackGCN(
            item_input_dim=item_dim,
            constraint_input_dim=constraint_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_name == "gat":
        dataset[0]
        item_dim = 2
        constraint_dim = 1

        return KnapsackGAT(
            item_input_dim=item_dim,
            constraint_input_dim=constraint_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate_model(
    model_name: str,
    num_layers: int,
    train_dataset: KnapsackGraphDataset,
    val_dataset: KnapsackGraphDataset,
    test_dataset: KnapsackGraphDataset,
    output_dir: Path,
    args,
):
    """
    Train and evaluate a single model configuration.

    Returns:
        Dictionary with training and evaluation results
    """
    run_name = f"{model_name}_L{num_layers}"
    print("\n" + "=" * 80)
    print(f"TRAINING: {run_name}")
    print("=" * 80)

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_model_by_name(
        model_name=model_name,
        dataset=train_dataset,
        hidden_dim=args.hidden_dim,
        num_layers=num_layers,
        dropout=args.dropout,
    )
    model = model.to(args.device)

    print(f"Model: {model_name}, Layers: {num_layers}, Hidden: {args.hidden_dim}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Train
    print("\nTraining...")
    model, history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=str(run_dir),
        device=args.device,
    )

    # Reload best weights
    best_ckpt = run_dir / "best_model.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=args.device)
        model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = {}

    for strategy in args.strategies:
        print(f"  Strategy: {strategy}")

        strategy_kwargs = {}
        if strategy == "sampling":
            strategy_kwargs = {
                "temperature": args.temperature,
                "sampling_schedule": tuple(args.sampling_schedule),
                "max_samples": args.max_samples,
            }
        elif strategy == "sampling_repair":
            strategy_kwargs = {
                "temperature": args.temperature,
                "sampling_schedule": tuple(args.sampling_schedule),
                "max_samples": args.max_samples,
            }

        eval_results = evaluate_model(
            model=model,
            dataset=test_dataset,
            strategy=strategy,
            device=args.device,
            **strategy_kwargs,
        )

        results[strategy] = {
            "mean_gap": eval_results.get("mean_gap"),
            "median_gap": eval_results.get("median_gap"),
            "std_gap": eval_results.get("std_gap"),
            "max_gap": eval_results.get("max_gap"),
            "p95": float(np.percentile(eval_results.get("gaps", [0]), 95)),
            "p99": float(np.percentile(eval_results.get("gaps", [0]), 99)),
            "feasibility_rate": eval_results.get("feasibility_rate"),
            "mean_time_ms": eval_results.get("mean_inference_time", 0) * 1000,
            "median_time_ms": eval_results.get("median_inference_time", 0) * 1000,
        }

        print(f"    Mean gap: {results[strategy]['mean_gap']:.2f}%")
        print(f"    p95: {results[strategy]['p95']:.2f}%")
        print(f"    Mean time: {results[strategy]['mean_time_ms']:.2f} ms")

    # Summary
    summary = {
        "model_name": model_name,
        "num_layers": num_layers,
        "hidden_dim": args.hidden_dim,
        "n_parameters": n_params,
        "training": {
            "epochs": args.epochs,
            "best_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "best_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        },
        "evaluation": results,
    }

    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print("=" * 80)

    return summary


def create_comparison_table(all_results: list, output_dir: Path):
    """
    Create comparison table of all model configurations.
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    # Flatten results for table
    rows = []
    for result in all_results:
        model_name = result["model_name"]
        num_layers = result["num_layers"]
        n_params = result["n_parameters"]

        for strategy, eval_res in result["evaluation"].items():
            rows.append(
                {
                    "Model": model_name.upper(),
                    "Layers": num_layers,
                    "Parameters": n_params,
                    "Strategy": strategy,
                    "Mean Gap (%)": eval_res["mean_gap"],
                    "Median Gap (%)": eval_res["median_gap"],
                    "p95 (%)": eval_res["p95"],
                    "p99 (%)": eval_res["p99"],
                    "Max Gap (%)": eval_res["max_gap"],
                    "Mean Time (ms)": eval_res["mean_time_ms"],
                    "Feasibility": eval_res["feasibility_rate"],
                }
            )

    df = pd.DataFrame(rows)

    # Print table
    print("\n" + df.to_string(index=False))
    print()

    # Save CSV
    csv_path = output_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Table saved to: {csv_path}")

    # Find best configuration
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)

    for strategy in df["Strategy"].unique():
        df_strategy = df[df["Strategy"] == strategy]

        # Best by p95
        best_p95 = df_strategy.loc[df_strategy["p95 (%)"].idxmin()]
        print(f"\n{strategy} - Best p95:")
        print(
            f"  {best_p95['Model']} L{best_p95['Layers']}: "
            f"p95={best_p95['p95 (%)']:.2f}%, "
            f"mean={best_p95['Mean Gap (%)']:.2f}%, "
            f"time={best_p95['Mean Time (ms)']:.2f}ms"
        )

        # Best by mean gap
        best_mean = df_strategy.loc[df_strategy["Mean Gap (%)"].idxmin()]
        if best_mean.name != best_p95.name:
            print(f"\n{strategy} - Best Mean Gap:")
            print(
                f"  {best_mean['Model']} L{best_mean['Layers']}: "
                f"mean={best_mean['Mean Gap (%)']:.2f}%, "
                f"p95={best_mean['p95 (%)']:.2f}%, "
                f"time={best_mean['Mean Time (ms)']:.2f}ms"
            )

    print("\n" + "=" * 80)

    # Create LaTeX table
    create_latex_table(df, output_dir)

    return df


def create_latex_table(df: pd.DataFrame, output_dir: Path):
    """Create LaTeX table for publication."""

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation Study: Model Architecture Comparison}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Layers & Params & Strategy & Mean Gap (\%) & p95 (\%) & Time (ms) \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(
            f"{row['Model']} & {row['Layers']} & {row['Parameters']:,} & "
            f"{row['Strategy']} & {row['Mean Gap (%)']:.2f} & "
            f"{row['p95 (%)']:.2f} & {row['Mean Time (ms)']:.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_path = output_dir / "ablation_table.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table saved to: {latex_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Model ablation study")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/datasets")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ablation")

    # Models to compare
    parser.add_argument(
        "--models", nargs="+", default=["pna", "gcn", "gat"], help="Models to compare"
    )
    parser.add_argument(
        "--depths", nargs="+", type=int, default=[2, 3, 4], help="Number of layers to try"
    )

    # Model hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-6)

    # Evaluation
    parser.add_argument("--strategies", nargs="+", default=["sampling", "sampling_repair"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-schedule", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--max-samples", type=int, default=128)

    # System
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ABLATION STUDY: MODEL ARCHITECTURE COMPARISON")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print(f"Models: {args.models}")
    print(f"Depths: {args.depths}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
    val_dataset = KnapsackDataset.load(f"{args.data_dir}/val.pkl")
    test_dataset = KnapsackDataset.load(f"{args.data_dir}/test.pkl")

    train_graph = KnapsackGraphDataset(train_dataset, normalize_features=True)
    val_graph = KnapsackGraphDataset(val_dataset, normalize_features=True)
    test_graph = KnapsackGraphDataset(test_dataset, normalize_features=True)

    print(f"Train: {len(train_graph)} instances")
    print(f"Val: {len(val_graph)} instances")
    print(f"Test: {len(test_graph)} instances")

    # Train and evaluate all configurations
    all_results = []

    for model_name in args.models:
        for num_layers in args.depths:
            result = train_and_evaluate_model(
                model_name=model_name,
                num_layers=num_layers,
                train_dataset=train_graph,
                val_dataset=val_graph,
                test_dataset=test_graph,
                output_dir=output_dir,
                args=args,
            )
            all_results.append(result)

    # Create comparison table
    create_comparison_table(all_results, output_dir)

    # Save all results
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {all_results_path}")
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
