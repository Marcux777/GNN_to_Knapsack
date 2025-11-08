"""Utility to inspect BC ranker outputs for interpretability plots and metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as MplPath

from knapsack_gnn.data.generator import KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.models.pna import create_model as create_pna_model
from knapsack_gnn.training.loop import greedy_masked_selection


def _path_deepcopy(self, memo):
    vertices = self.vertices.copy()
    codes = None if self.codes is None else self.codes.copy()
    return MplPath(vertices, codes)


MplPath.__deepcopy__ = _path_deepcopy


def load_model(
    checkpoint_path: Path, dataset: KnapsackGraphDataset, device: str
) -> torch.nn.Module:
    print(f"Loading model from {checkpoint_path}")
    model = create_pna_model(dataset=dataset, hidden_dim=64, num_layers=3, dropout=0.1)
    device_obj = torch.device(device)
    state = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(state["model_state_dict"])
    model.to(device_obj)
    model.eval()
    return model


def inspect_instances(
    model: torch.nn.Module,
    dataset: KnapsackGraphDataset,
    indices: list[int],
    device: str,
    output_dir: Path | str,
) -> list[dict]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    metrics: list[dict] = []

    for idx in indices:
        data = dataset[idx].to(device_obj)
        with torch.no_grad():
            probs = model(data)
        solution = greedy_masked_selection(probs, data.item_weights, float(data.capacity))
        probs_np = probs.cpu().numpy()
        solution_np = solution.cpu().numpy()
        weights_np = data.item_weights.cpu().numpy()
        values_np = data.item_values.cpu().numpy()

        plot_scores_vs_selection(
            data, probs_np, solution_np, out_path / f"instance_{idx}_scores.png"
        )
        plot_ratio_vs_score(data, probs_np, out_path / f"instance_{idx}_ratio.png")
        plot_cumulative(
            probs_np,
            weights_np,
            values_np,
            float(data.capacity),
            out_path / f"instance_{idx}_cumulative.png",
        )

        metrics.append(
            compute_metrics(
                idx=idx,
                scores=probs_np,
                solution=solution_np,
                weights=weights_np,
                values=values_np,
                capacity=float(data.capacity),
                optimal_value=float(data.optimal_value),
            )
        )
        with open(out_path / f"instance_{idx}_metrics.json", "w") as f:
            json.dump(metrics[-1], f, indent=2)

    write_report(out_path, metrics)
    return metrics


def compute_metrics(
    idx: int,
    scores: np.ndarray,
    solution: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    capacity: float,
    optimal_value: float,
) -> dict:
    ratio = values / np.maximum(weights, 1e-6)
    result = {
        "index": idx,
        "spearman_ratio": spearman_corr(scores, ratio),
        "spearman_value": spearman_corr(scores, values),
        "spearman_weight": spearman_corr(scores, weights),
    }
    base_value = float(np.dot(solution, values))
    base_weight = float(np.dot(solution, weights))
    result["base_value"] = base_value
    result["base_weight"] = base_weight
    result["base_gap"] = compute_gap(base_value, optimal_value)
    result["capacity_minus5"] = capacity_sensitivity(
        scores, weights, values, capacity * 0.95, solution
    )
    result["capacity_plus5"] = capacity_sensitivity(
        scores, weights, values, capacity * 1.05, solution
    )
    return result


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    std_a = np.std(ar)
    std_b = np.std(br)
    if std_a < 1e-6 or std_b < 1e-6:
        return 0.0
    corr = np.corrcoef(ar, br)[0, 1]
    return float(corr)


def compute_gap(value: float, optimal_value: float) -> float | None:
    if optimal_value <= 0:
        return None
    return max(0.0, (optimal_value - value) / optimal_value * 100)


def capacity_sensitivity(
    scores: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    new_capacity: float,
    base_solution: np.ndarray,
) -> dict:
    scores_t = torch.from_numpy(scores)
    weights_t = torch.from_numpy(weights)
    new_solution = greedy_masked_selection(scores_t, weights_t, float(new_capacity))
    new_solution_np = new_solution.cpu().numpy()
    value = float(np.dot(new_solution_np, values))
    weight = float(np.dot(new_solution_np, weights))
    hamming = int(np.sum(np.abs(new_solution_np - base_solution)))
    return {
        "value": value,
        "weight": weight,
        "hamming_distance": hamming,
    }


def plot_scores_vs_selection(data, scores: np.ndarray, solution: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(scores))
    ax.bar(x, scores, color="skyblue", label="Score")
    ax.scatter(x[solution == 1], scores[solution == 1], color="green", label="Selected")
    ax.set_xlabel("Item index")
    ax.set_ylabel("Score")
    ax.set_title("Scores vs Selection")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ratio_vs_score(data, scores: np.ndarray, path: Path) -> None:
    item_features = data.x[data.node_types == 0].cpu().numpy()
    ratio = item_features[:, 2]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ratio, scores, alpha=0.7)
    ax.set_xlabel("Value/Weight (normalized)")
    ax.set_ylabel("Score")
    ax.set_title("Score vs Density")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cumulative(
    scores: np.ndarray, weights: np.ndarray, values: np.ndarray, capacity: float, path: Path
) -> None:
    order = np.argsort(scores)[::-1]
    sorted_weights = weights[order]
    sorted_values = values[order]

    cum_weights = np.cumsum(sorted_weights)
    cum_values = np.cumsum(sorted_values)
    remaining_capacity = np.maximum(capacity - cum_weights, 0)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(cum_values, label="Cumulative Value", color="tab:blue")
    ax1.set_ylabel("Value", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(remaining_capacity, label="Remaining Capacity", color="tab:red")
    ax2.set_ylabel("Capacity", color="tab:red")
    ax1.set_xlabel("Items sorted by score")
    ax1.set_title("Cumulative Value and Capacity vs Score Rank")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(out_path: Path, metrics: list[dict]) -> None:
    if not metrics:
        return
    lines = ["# BC Ranker Interpretability Report", ""]
    mean_ratio = np.nanmean([m["spearman_ratio"] for m in metrics])
    lines.append(f"*Average Spearman(score, v/w):* {mean_ratio:.3f}")
    lines.append("")
    for metric in metrics:
        idx = metric["index"]
        lines.append(f"## Instance {idx}")
        lines.append(f"- Spearman(score, v/w): {metric['spearman_ratio']:.3f}")
        lines.append(f"- Spearman(score, value): {metric['spearman_value']:.3f}")
        lines.append(f"- Spearman(score, weight): {metric['spearman_weight']:.3f}")
        lines.append(f"- Base value: {metric['base_value']:.2f}")
        lines.append(
            f"- Capacity -5% value: {metric['capacity_minus5']['value']:.2f}, "
            f"hamming={metric['capacity_minus5']['hamming_distance']}"
        )
        lines.append(
            f"- Capacity +5% value: {metric['capacity_plus5']['value']:.2f}, "
            f"hamming={metric['capacity_plus5']['hamming_distance']}"
        )
        lines.append("")
        lines.append(f"![Scores](instance_{idx}_scores.png)")
        lines.append(f"![Density](instance_{idx}_ratio.png)")
        lines.append(f"![Cumulative](instance_{idx}_cumulative.png)")
        lines.append("")

    with open(out_path / "report.md", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BC ranker outputs")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output_dir", type=str, default="results/bc_ranker_inspect")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data_dir = Path("data/datasets")
    test_ds = KnapsackDataset.load(str(data_dir / "test.pkl"))
    test_graphs = KnapsackGraphDataset(test_ds, normalize_features=True)

    model = load_model(Path(args.checkpoint), test_graphs, args.device)
    metrics = inspect_instances(
        model, test_graphs, args.indices, args.device, Path(args.output_dir)
    )
    with open(Path(args.output_dir) / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()


def _path_deepcopy(self, memo):
    vertices = self.vertices.copy()
    codes = None if self.codes is None else self.codes.copy()
    return MplPath(vertices, codes)


MplPath.__deepcopy__ = _path_deepcopy
