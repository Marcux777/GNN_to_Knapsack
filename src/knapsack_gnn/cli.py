"""
Unified CLI for Knapsack GNN.

Provides subcommands for training, evaluation, and experiments.
"""

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

from knapsack_gnn.decoding.sampling import KnapsackSampler
from knapsack_gnn.utils.error_handler import (
    handle_cli_errors,
    validate_checkpoint_dir,
)
from knapsack_gnn.utils.feature_flags import resolve_graph_feature_kwargs
from knapsack_gnn.utils.model_utils import (
    apply_dynamic_quantization,
    load_graph_dataset,
    load_model_from_checkpoint,
    maybe_compile_model,
)


@click.group()
@click.version_option(version="1.0.0")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (show full stack traces on errors)",
    default=False,
)
@click.pass_context
def main(ctx: Any, debug: bool) -> None:
    """
    Knapsack GNN - Learning to Optimize.

    Graph Neural Network for solving the 0-1 Knapsack Problem.

    Examples:
        knapsack-gnn train --config experiments/configs/train_default.yaml
        knapsack-gnn eval --checkpoint checkpoints/run_001 --strategy sampling
        knapsack-gnn pipeline --config experiments/configs/pipeline.yaml

    Use --debug flag to see detailed error information.
    """
    # Store debug flag in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@main.command()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to training configuration YAML file"
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--device", type=str, default="cpu", help="Device (cpu/cuda)")
@click.option("--epochs", type=int, help="Number of epochs (overrides config)")
@click.option("--batch-size", type=int, help="Batch size (overrides config)")
@click.option("--lr", type=float, help="Learning rate (overrides config)")
@click.pass_context
@handle_cli_errors()
def train(
    ctx: Any, config: str, seed: int, device: str, epochs: int, batch_size: int, lr: float
) -> None:
    """Train a GNN model on knapsack instances."""
    ctx.obj.get("debug", False)

    # Import here to avoid slow startup
    from experiments.pipelines.train_pipeline import main as train_main

    # Construct args for train_pipeline
    args = ["--seed", str(seed), "--device", device]

    if config:
        args.extend(["--config", config])
    if epochs:
        args.extend(["--epochs", str(epochs)])
    if batch_size:
        args.extend(["--batch-size", str(batch_size)])
    if lr:
        args.extend(["--learning-rate", str(lr)])

    # Replace sys.argv and call train_main
    old_argv = sys.argv
    try:
        sys.argv = ["train"] + args
        train_main()
    finally:
        sys.argv = old_argv


@main.command()
@click.option(
    "--checkpoint", type=click.Path(exists=True), required=True, help="Path to checkpoint directory"
)
@click.option(
    "--strategy",
    type=click.Choice(["sampling", "warm_start", "lagrangian"], case_sensitive=False),
    default="sampling",
    help="Decoding strategy",
)
@click.option("--device", type=str, default="cpu", help="Device (cpu/cuda)")
@click.option("--test-only", is_flag=True, help="Evaluate only on test set")
@click.pass_context
@handle_cli_errors()
def eval(ctx: Any, checkpoint: str, strategy: str, device: str, test_only: bool) -> None:
    """Evaluate a trained model on knapsack instances."""
    ctx.obj.get("debug", False)

    # Validate checkpoint
    validate_checkpoint_dir(checkpoint)

    from experiments.pipelines.evaluate_pipeline import main as eval_main

    args = ["--checkpoint-dir", checkpoint, "--strategy", strategy, "--device", device]

    if test_only:
        args.append("--test-only")

    old_argv = sys.argv
    try:
        sys.argv = ["evaluate"] + args
        eval_main()
    finally:
        sys.argv = old_argv


@main.command()
@click.option(
    "--checkpoint", type=click.Path(exists=True), required=True, help="Path to checkpoint directory"
)
@click.option(
    "--sizes",
    type=str,
    default="100,150,200",
    help="Comma-separated OOD sizes (e.g., '100,150,200')",
)
@click.option("--strategy", type=str, default="sampling", help="Decoding strategy")
@click.option("--device", type=str, default="cpu", help="Device")
def ood(checkpoint: str, sizes: str, strategy: str, device: str) -> None:
    """Evaluate out-of-distribution generalization."""
    from experiments.pipelines.evaluate_ood_pipeline import main as ood_main

    args = [
        "--checkpoint-dir",
        checkpoint,
        "--ood-sizes",
        sizes,
        "--strategy",
        strategy,
        "--device",
        device,
    ]

    old_argv = sys.argv
    try:
        sys.argv = ["evaluate_ood"] + args
        ood_main()
    finally:
        sys.argv = old_argv


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Pipeline configuration YAML")
@click.option(
    "--strategies", type=str, default="sampling,warm_start", help="Comma-separated strategies"
)
@click.option("--skip-train", is_flag=True, help="Skip training phase")
@click.option("--checkpoint", type=click.Path(), help="Existing checkpoint to use")
@click.option("--seed", type=int, default=1337, help="Random seed")
@click.option("--device", type=str, default="cpu", help="Device")
def pipeline(
    config: str, strategies: str, skip_train: bool, checkpoint: str, seed: int, device: str
) -> None:
    """Run full experiment pipeline (train + evaluate)."""
    from experiments.pipelines.main import main as pipeline_main

    args = ["--pipeline-strategies", strategies, "--seed", str(seed), "--device", device]

    if config:
        args.extend(["--config", config])
    if skip_train:
        args.append("--skip-train")
    if checkpoint:
        args.extend(["--checkpoint-dir", checkpoint])

    old_argv = sys.argv
    try:
        sys.argv = ["pipeline"] + args
        pipeline_main()
    finally:
        sys.argv = old_argv


@main.command()
@click.option(
    "--mode",
    type=click.Choice(["features", "architecture"], case_sensitive=False),
    required=True,
    help="Ablation mode",
)
@click.option("--config", type=click.Path(), help="Config file")
@click.option("--device", type=str, default="cpu", help="Device")
def ablation(mode: str, config: str, device: str) -> None:
    """Run ablation studies (features or architecture)."""
    from experiments.pipelines.ablation_study import main as ablation_main

    args = ["--mode", mode, "--device", device]

    if config:
        args.extend(["--config", config])

    old_argv = sys.argv
    try:
        sys.argv = ["ablation"] + args
        ablation_main()
    finally:
        sys.argv = old_argv


@main.command()
@click.option(
    "--checkpoint", type=click.Path(exists=True), required=True, help="Checkpoint directory"
)
@click.option(
    "--baseline",
    type=click.Choice(["greedy", "random", "fptas", "meet"], case_sensitive=False),
    multiple=True,
    default=["greedy", "random", "fptas", "meet"],
    help="Baselines to compare (repeat to select a subset)",
)
@click.option(
    "--fptas-epsilon",
    type=float,
    default=0.05,
    show_default=True,
    help="Approximation factor ε for the FPTAS baseline.",
)
@click.option(
    "--meet-max-items",
    type=int,
    default=32,
    show_default=True,
    help="Maximum number of items solved exactly by meet-in-the-middle.",
)
@click.option(
    "--meet-fallback-epsilon",
    type=float,
    default=0.02,
    show_default=True,
    help="Fallback ε used when meet-in-the-middle falls back to FPTAS.",
)
@click.pass_context
@handle_cli_errors()
def compare(
    ctx: Any,
    checkpoint: str,
    baseline: tuple,
    fptas_epsilon: float,
    meet_max_items: int,
    meet_fallback_epsilon: float,
) -> None:
    """Compare GNN with classical baselines."""
    ctx.obj.get("debug", False)

    # Validate checkpoint
    validate_checkpoint_dir(checkpoint)

    from experiments.analysis.baseline_comparison import main as compare_main

    args = [
        "--checkpoint-dir",
        checkpoint,
        "--fptas-epsilon",
        str(fptas_epsilon),
        "--meet-max-items",
        str(meet_max_items),
        "--meet-fallback-epsilon",
        str(meet_fallback_epsilon),
    ]
    for b in baseline:
        args.extend(["--baseline", b])

    old_argv = sys.argv
    try:
        sys.argv = ["compare"] + args
        compare_main()
    finally:
        sys.argv = old_argv


@main.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Checkpoint directory containing trained weights",
)
@click.option(
    "--checkpoint-name",
    type=str,
    default="best_model.pt",
    show_default=True,
    help="Checkpoint filename to load",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="data/datasets",
    show_default=True,
    help="Directory containing train/val/test dataset pickles",
)
@click.option(
    "--onnx-path",
    type=click.Path(),
    default=None,
    help="Output path for the exported ONNX model (defaults to <checkpoint>/model.onnx)",
)
@click.option("--device", type=str, default="cpu", show_default=True, help="Device to use")
@click.option(
    "--sample-index",
    type=int,
    default=0,
    show_default=True,
    help="Sample graph index used to trace the ONNX graph",
)
@click.option(
    "--opset",
    type=int,
    default=17,
    show_default=True,
    help="ONNX opset version",
)
@click.option(
    "--quantize",
    is_flag=True,
    default=False,
    help="Apply dynamic quantization before exporting",
)
@handle_cli_errors()
def export(
    checkpoint: str,
    checkpoint_name: str,
    data_dir: str,
    onnx_path: str | None,
    device: str,
    sample_index: int,
    opset: int,
    quantize: bool,
) -> None:
    """Export a trained model to ONNX for deployment."""
    model, train_graph_dataset = load_model_from_checkpoint(
        checkpoint_dir=checkpoint,
        checkpoint_name=checkpoint_name,
        data_dir=data_dir,
        device=device,
    )

    if quantize:
        model = apply_dynamic_quantization(model)

    sample_count = len(train_graph_dataset)
    if sample_count == 0:
        raise click.ClickException("Training dataset is empty; cannot trace ONNX graph.")

    sample = train_graph_dataset[sample_index % sample_count]
    x = sample.x.to(device)
    edge_index = sample.edge_index.to(device)
    node_types = sample.node_types.to(device)

    wrapper = _OnnxWrapper(model).to(device)
    wrapper.eval()

    target_path = Path(onnx_path) if onnx_path else Path(checkpoint) / "model.onnx"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "x": {0: "num_nodes"},
        "edge_index": {1: "num_edges"},
        "node_types": {0: "num_nodes"},
        "probabilities": {0: "num_items"},
    }

    try:
        torch.onnx.export(
            wrapper,
            (x, edge_index, node_types),
            target_path,
            input_names=["x", "edge_index", "node_types"],
            output_names=["probabilities"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
        )
    except Exception as exc:  # pragma: no cover - depends on ONNX availability
        raise click.ClickException(f"Failed to export ONNX model: {exc}") from exc

    click.echo(f"ONNX model exported to {target_path}")


@main.command()
@click.argument("checkpoint", type=click.Path(exists=True))
def demo(checkpoint: str) -> None:
    """Run interactive demo with visualization."""
    from experiments.examples.demo import main as demo_main

    old_argv = sys.argv
    try:
        sys.argv = ["demo", checkpoint]
        demo_main()
    finally:
        sys.argv = old_argv


@main.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model checkpoint",
)
@click.option(
    "--output-dir",
    type=str,
    default="validation_report",
    help="Output directory for validation results",
)
@click.option(
    "--baselines",
    multiple=True,
    default=["greedy", "random"],
    help="Baseline methods to compare (can specify multiple times)",
)
@click.option(
    "--run-cv", is_flag=True, help="Run cross-validation (requires training, time-consuming)"
)
@click.option("--cv-folds", type=int, default=5, help="Number of cross-validation folds")
@click.option("--stratify-cv", is_flag=True, help="Stratify cross-validation by problem size")
@click.option("--alpha", type=float, default=0.05, help="Significance level for statistical tests")
@click.option(
    "--n-bootstrap",
    type=int,
    default=10000,
    help="Number of bootstrap samples for confidence intervals",
)
@click.option("--check-power", is_flag=True, help="Run statistical power analysis")
@click.option(
    "--strategy",
    type=click.Choice(["sampling", "warm_start", "lagrangian", "threshold", "adaptive"]),
    default="sampling",
    help="GNN inference strategy",
)
@click.option("--n-samples", type=int, default=200, help="Number of samples for sampling strategy")
@click.option("--latex", is_flag=True, default=True, help="Generate LaTeX tables for publication")
@click.option("--figures", is_flag=True, default=True, help="Generate publication-quality figures")
@click.option(
    "--config", type=click.Path(exists=True), help="Path to validation configuration YAML file"
)
@click.option("--device", type=str, default="cpu", help="Device (cpu/cuda)")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
def validate(
    checkpoint: str,
    output_dir: str,
    baselines: tuple,
    run_cv: bool,
    cv_folds: int,
    stratify_cv: bool,
    alpha: float,
    n_bootstrap: int,
    check_power: bool,
    strategy: str,
    n_samples: int,
    latex: bool,
    figures: bool,
    config: str,
    device: str,
    seed: int,
) -> None:
    """
    Run comprehensive publication-grade validation.

    Performs rigorous statistical validation including:
    - Baseline comparisons with statistical tests
    - Cross-validation for generalization estimates
    - Statistical power analysis
    - Assumption checking
    - Publication-ready LaTeX tables and figures

    Examples:
        # Quick validation with default settings
        knapsack-gnn validate --checkpoint checkpoints/run_001

        # Full validation with cross-validation
        knapsack-gnn validate --checkpoint checkpoints/run_001 --run-cv --check-power

        # Custom baselines and output directory
        knapsack-gnn validate --checkpoint checkpoints/run_001 \\
            --baselines greedy --baselines random \\
            --output-dir my_validation

        # Use configuration file
        knapsack-gnn validate --checkpoint checkpoints/run_001 \\
            --config experiments/configs/validation_config.yaml
    """
    from experiments.pipelines.publication_validation import main as validate_main

    # Build arguments
    args = [
        "--checkpoint",
        checkpoint,
        "--output-dir",
        output_dir,
        "--strategy",
        strategy,
        "--n-samples",
        str(n_samples),
        "--alpha",
        str(alpha),
        "--n-bootstrap",
        str(n_bootstrap),
        "--device",
        device,
        "--seed",
        str(seed),
        "--cv-folds",
        str(cv_folds),
    ]

    # Add baselines
    for baseline in baselines:
        args.extend(["--baselines", baseline])

    # Add flags
    if run_cv:
        args.append("--run_cv")
    if stratify_cv:
        args.append("--stratify_cv")
    if check_power:
        args.append("--check_power")
    if latex:
        args.append("--latex")
    if figures:
        args.append("--figures")

    old_argv = sys.argv
    try:
        sys.argv = ["validate"] + args
        validate_main()
    finally:
        sys.argv = old_argv


class _OnnxWrapper(torch.nn.Module):
    """Thin wrapper to expose Tensor inputs when exporting to ONNX."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        from torch_geometric.data import Data

        data = Data(
            x=x,
            edge_index=edge_index.long(),
            node_types=node_types.long(),
        )
        return self.model(data)


def _benchmark_sampler(
    sampler: KnapsackSampler,
    graphs: list,
    device: str,
) -> dict[str, float]:
    if not graphs:
        raise click.ClickException("No graphs available to benchmark.")

    # Warm-up
    sampler.get_probabilities(graphs[0])
    _sync_device(device)

    timings = []
    start_all = time.perf_counter()
    for data in graphs:
        start = time.perf_counter()
        sampler.get_probabilities(data)
        _sync_device(device)
        timings.append(time.perf_counter() - start)
    total_elapsed = time.perf_counter() - start_all

    arr = np.array(timings, dtype=np.float64)
    return {
        "mean_time_ms": float(arr.mean() * 1000),
        "median_time_ms": float(np.median(arr) * 1000),
        "p90_time_ms": float(np.percentile(arr, 90) * 1000),
        "p95_time_ms": float(np.percentile(arr, 95) * 1000),
        "p99_time_ms": float(np.percentile(arr, 99) * 1000),
        "std_time_ms": float(arr.std() * 1000),
        "throughput": len(graphs) / total_elapsed if total_elapsed > 0 else 0.0,
    }


def _sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():  # pragma: no cover - GPU only
        torch.cuda.synchronize()


@main.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Checkpoint directory to benchmark",
)
@click.option(
    "--checkpoint-name",
    type=str,
    default="best_model.pt",
    show_default=True,
    help="Checkpoint filename to load",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="data/datasets",
    show_default=True,
    help="Directory containing dataset pickles",
)
@click.option("--device", type=str, default="cpu", show_default=True, help="Device (cpu/cuda)")
@click.option(
    "--n-instances",
    type=int,
    default=64,
    show_default=True,
    help="Number of test instances to benchmark",
)
@click.option(
    "--threads",
    type=int,
    multiple=True,
    help="Thread counts to benchmark (repeatable). Defaults to [1, cpu_count].",
)
@click.option(
    "--compile",
    "compile_model",
    is_flag=True,
    default=False,
    help="Enable torch.compile before benchmarking",
)
@click.option(
    "--quantize",
    is_flag=True,
    default=False,
    help="Apply dynamic quantization before benchmarking",
)
@click.option(
    "--output",
    "output_csv",
    type=click.Path(),
    default=None,
    help="CSV file to append results to (default: results/bench_<timestamp>.csv)",
)
@handle_cli_errors()
def bench(
    checkpoint: str,
    checkpoint_name: str,
    data_dir: str,
    device: str,
    n_instances: int,
    threads: tuple[int, ...],
    compile_model: bool,
    quantize: bool,
    output_csv: str | None,
) -> None:
    """Benchmark inference throughput/latency on the test dataset."""
    graph_feature_kwargs, resolved_spec = resolve_graph_feature_kwargs(
        "auto",
        None,
        checkpoint_dir=checkpoint,
    )
    test_dataset = load_graph_dataset("test", data_dir, graph_features=graph_feature_kwargs)
    click.echo(f"Graph feature spec: {resolved_spec}")
    if len(test_dataset) == 0:
        raise click.ClickException("Test dataset is empty; cannot benchmark.")

    total_instances = min(n_instances, len(test_dataset))
    graphs = [test_dataset[i] for i in range(total_instances)]

    if not threads:
        cpu_threads = os.cpu_count() or 1
        thread_values = [1, cpu_threads]
    else:
        thread_values = sorted({t for t in threads if t != 0})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_csv) if output_csv else Path("results") / f"bench_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "checkpoint",
        "checkpoint_name",
        "device",
        "threads",
        "compile",
        "quantize",
        "n_instances",
        "mean_time_ms",
        "median_time_ms",
        "p90_time_ms",
        "p95_time_ms",
        "p99_time_ms",
        "std_time_ms",
        "throughput",
    ]

    write_header = not output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for thread_count in thread_values:
            model, _ = load_model_from_checkpoint(
                checkpoint_dir=checkpoint,
                checkpoint_name=checkpoint_name,
                data_dir=data_dir,
                device=device,
            )

            if quantize:
                model = apply_dynamic_quantization(model)
            if compile_model:
                model = maybe_compile_model(model)

            sampler = KnapsackSampler(
                model=model,
                device=device,
                num_threads=thread_count,
                compile_model=False,
                quantize=False,
            )
            stats = _benchmark_sampler(sampler, graphs, device)
            row = {
                "timestamp": timestamp,
                "checkpoint": checkpoint,
                "checkpoint_name": checkpoint_name,
                "device": device,
                "threads": thread_count,
                "compile": int(compile_model),
                "quantize": int(quantize),
                "n_instances": total_instances,
                **stats,
            }
            writer.writerow(row)
            click.echo(
                f"[threads={thread_count}] mean={stats['mean_time_ms']:.3f} ms, "
                f"median={stats['median_time_ms']:.3f} ms, throughput={stats['throughput']:.2f} inst/s"
            )

    click.echo(f"Benchmark results saved to {output_path}")


if __name__ == "__main__":
    main()
