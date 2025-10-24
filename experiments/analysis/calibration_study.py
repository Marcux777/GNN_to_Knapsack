# mypy: ignore-errors
"""
Calibration Study Script

Analyzes probability calibration of the GNN model.
Generates:
    - ECE, MCE, Brier scores
    - Reliability plots
    - Temperature-scaled calibration
    - Platt-scaled calibration
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from knapsack_gnn.analysis.calibration import (
    PlattScaling,
    TemperatureScaling,
    evaluate_calibration,
)
from knapsack_gnn.data.generator import KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.models.pna import create_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sns.set_style("whitegrid")


def plot_reliability_diagram(
    results_list: list,
    labels: list,
    save_path: str = None,
    title: str = "Reliability Diagram",
):
    """
    Plot reliability diagram comparing multiple calibration methods.

    Args:
        results_list: List of calibration result dictionaries
        labels: List of labels for each result
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_list)))

    for results, label, color in zip(results_list, labels, colors, strict=False):
        rel_curve = results["reliability_curve"]
        mean_pred = np.array(rel_curve["mean_predicted"])
        frac_pos = np.array(rel_curve["fraction_positive"])
        counts = np.array(rel_curve["counts"])

        # Filter out NaN values
        valid = ~np.isnan(mean_pred) & ~np.isnan(frac_pos)
        mean_pred = mean_pred[valid]
        frac_pos = frac_pos[valid]
        counts = counts[valid]

        if len(mean_pred) == 0:
            continue

        # Plot with marker size proportional to bin count
        sizes = (counts / counts.max()) * 200 + 20
        ax.scatter(
            mean_pred,
            frac_pos,
            s=sizes,
            alpha=0.6,
            color=color,
            label=f"{label} (ECE={results['ece']:.3f})",
            edgecolors="black",
            linewidths=0.5,
        )

        # Connect with line
        sorted_idx = np.argsort(mean_pred)
        ax.plot(
            mean_pred[sorted_idx],
            frac_pos[sorted_idx],
            linestyle="--",
            alpha=0.5,
            color=color,
            linewidth=1,
        )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration", alpha=0.7)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Reliability diagram saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def extract_probabilities_and_labels(model, dataset, device: str = "cpu"):
    """
    Extract predicted probabilities and ground truth labels from dataset.

    Args:
        model: Trained model
        dataset: KnapsackGraphDataset
        device: Device to run on

    Returns:
        Tuple of (probabilities, labels)
            probabilities: [n_instances * n_items] predicted probabilities
            labels: [n_instances * n_items] ground truth binary labels
    """
    model = model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.inference_mode():
        for data in dataset:
            data = data.to(device)
            probs = model(data).cpu().numpy()

            # Get ground truth solution
            instance_idx = data.instance_idx.item() if hasattr(data, "instance_idx") else 0
            true_solution = dataset.raw_dataset.instances[instance_idx].solution

            all_probs.append(probs)
            all_labels.append(true_solution)

    probabilities = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    return probabilities, labels


def run_calibration_study(
    model,
    val_dataset: KnapsackGraphDataset,
    test_dataset: KnapsackGraphDataset,
    output_dir: Path,
    device: str = "cpu",
    n_bins: int = 10,
):
    """
    Run full calibration study.

    Args:
        model: Trained model
        val_dataset: Validation dataset (for fitting calibration)
        test_dataset: Test dataset (for evaluating calibration)
        output_dir: Output directory
        device: Device to run on
        n_bins: Number of bins for ECE
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CALIBRATION STUDY")
    print("=" * 80)
    print()

    # Extract probabilities and labels
    print("Extracting validation probabilities...")
    val_probs, val_labels = extract_probabilities_and_labels(model, val_dataset, device)
    print(f"  Validation: {len(val_labels)} predictions")

    print("Extracting test probabilities...")
    test_probs, test_labels = extract_probabilities_and_labels(model, test_dataset, device)
    print(f"  Test: {len(test_labels)} predictions")
    print()

    # Convert to logits for calibration methods
    eps = 1e-7
    val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
    test_probs_clipped = np.clip(test_probs, eps, 1 - eps)
    val_logits = np.log(val_probs_clipped / (1 - val_probs_clipped))
    test_logits = np.log(test_probs_clipped / (1 - test_probs_clipped))

    # 1. Evaluate uncalibrated model
    print("=" * 80)
    print("1. UNCALIBRATED MODEL")
    print("=" * 80)
    results_uncalibrated = evaluate_calibration(test_labels, test_probs, n_bins=n_bins)
    print(f"ECE: {results_uncalibrated['ece']:.4f}")
    print(f"MCE: {results_uncalibrated['mce']:.4f}")
    print(f"Brier Score: {results_uncalibrated['brier_score']:.4f}")
    print(f"Accuracy: {results_uncalibrated['accuracy']:.4f}")
    print()

    # 2. Temperature scaling
    print("=" * 80)
    print("2. TEMPERATURE SCALING")
    print("=" * 80)
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(val_logits, val_labels, method="ece")
    test_probs_temp = temp_scaler.transform(test_logits)

    results_temp = evaluate_calibration(test_labels, test_probs_temp, n_bins=n_bins)
    print(f"Optimal Temperature: {optimal_temp:.4f}")
    print(
        f"ECE: {results_temp['ece']:.4f} (Δ = {results_temp['ece'] - results_uncalibrated['ece']:.4f})"
    )
    print(f"MCE: {results_temp['mce']:.4f}")
    print(f"Brier Score: {results_temp['brier_score']:.4f}")
    print(f"Accuracy: {results_temp['accuracy']:.4f}")

    status = "✓ PASS" if results_temp["ece"] < 0.1 else "✗ FAIL"
    print(f"ECE < 0.1: {status}")
    print()

    # 3. Platt scaling
    print("=" * 80)
    print("3. PLATT SCALING")
    print("=" * 80)
    platt_scaler = PlattScaling()
    A, B = platt_scaler.fit(val_logits, val_labels)
    test_probs_platt = platt_scaler.transform(test_logits)

    results_platt = evaluate_calibration(test_labels, test_probs_platt, n_bins=n_bins)
    print(f"Platt Parameters: A={A:.4f}, B={B:.4f}")
    print(
        f"ECE: {results_platt['ece']:.4f} (Δ = {results_platt['ece'] - results_uncalibrated['ece']:.4f})"
    )
    print(f"MCE: {results_platt['mce']:.4f}")
    print(f"Brier Score: {results_platt['brier_score']:.4f}")
    print(f"Accuracy: {results_platt['accuracy']:.4f}")

    status = "✓ PASS" if results_platt["ece"] < 0.1 else "✗ FAIL"
    print(f"ECE < 0.1: {status}")
    print()

    # Save results
    all_results = {
        "uncalibrated": results_uncalibrated,
        "temperature_scaled": {
            **results_temp,
            "optimal_temperature": optimal_temp,
        },
        "platt_scaled": {
            **results_platt,
            "A": A,
            "B": B,
        },
        "n_bins": n_bins,
        "n_val_samples": int(len(val_labels)),
        "n_test_samples": int(len(test_labels)),
    }

    results_path = output_dir / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Generate reliability plot
    print("\nGenerating reliability diagram...")
    plot_reliability_diagram(
        results_list=[results_uncalibrated, results_temp, results_platt],
        labels=["Uncalibrated", "Temperature Scaled", "Platt Scaled"],
        save_path=str(output_dir / "reliability_diagram.png"),
        title="Model Calibration: Reliability Diagram",
    )

    # Summary
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'ECE':<10} {'MCE':<10} {'Brier':<10} {'Status'}")
    print("-" * 80)

    methods = [
        ("Uncalibrated", results_uncalibrated),
        ("Temperature", results_temp),
        ("Platt", results_platt),
    ]

    for name, res in methods:
        status = "✓" if res["ece"] < 0.1 else "✗"
        print(
            f"{name:<20} {res['ece']:<10.4f} {res['mce']:<10.4f} {res['brier_score']:<10.4f} {status}"
        )

    print("=" * 80)
    print()

    # Determine best method
    best_method = min(methods, key=lambda x: x[1]["ece"])
    print(f"Best Method: {best_method[0]} (ECE = {best_method[1]['ece']:.4f})")

    if best_method[1]["ece"] < 0.1:
        print("✓ TARGET MET: ECE < 0.1")
    else:
        print("✗ TARGET NOT MET: ECE >= 0.1")
        print(f"  Recommendation: Use {best_method[0]} scaling for deployment")

    print()
    print("=" * 80)
    print("CALIBRATION STUDY COMPLETE")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibration study script")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True, help="Path to val.pkl")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test.pkl")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load datasets
    print("Loading datasets...")
    val_dataset_raw = KnapsackDataset.load(args.val_data)
    test_dataset_raw = KnapsackDataset.load(args.test_data)

    val_dataset = KnapsackGraphDataset(val_dataset_raw, normalize_features=True)
    test_dataset = KnapsackGraphDataset(test_dataset_raw, normalize_features=True)

    print(f"Validation: {len(val_dataset)} instances")
    print(f"Test: {len(test_dataset)} instances")

    # Load model
    print("\nLoading model...")
    checkpoint_path = Path(args.checkpoint_dir) / "best_model.pt"
    model = create_model(
        dataset=val_dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    state = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

    # Run calibration study
    run_calibration_study(
        model=model,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        output_dir=Path(args.output_dir),
        device=args.device,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
