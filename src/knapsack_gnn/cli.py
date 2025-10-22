"""
Unified CLI for Knapsack GNN.

Provides subcommands for training, evaluation, and experiments.
"""

import sys
from pathlib import Path

import click

@click.group()
@click.version_option(version="1.0.0")
def main():
    """
    Knapsack GNN - Learning to Optimize.

    Graph Neural Network for solving the 0-1 Knapsack Problem.

    Examples:
        knapsack-gnn train --config experiments/configs/train_default.yaml
        knapsack-gnn eval --checkpoint checkpoints/run_001 --strategy sampling
        knapsack-gnn pipeline --config experiments/configs/pipeline.yaml
    """
    pass

@main.command()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to training configuration YAML file"
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--device", type=str, default="cpu", help="Device (cpu/cuda)")
@click.option("--epochs", type=int, help="Number of epochs (overrides config)")
@click.option("--batch-size", type=int, help="Batch size (overrides config)")
@click.option("--lr", type=float, help="Learning rate (overrides config)")
def train(config, seed, device, epochs, batch_size, lr):
    """Train a GNN model on knapsack instances."""
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
def eval(checkpoint, strategy, device, test_only):
    """Evaluate a trained model on knapsack instances."""
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
def ood(checkpoint, sizes, strategy, device):
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
def pipeline(config, strategies, skip_train, checkpoint, seed, device):
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
def ablation(mode, config, device):
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
    type=click.Choice(["greedy", "random"], case_sensitive=False),
    multiple=True,
    default=["greedy"],
    help="Baselines to compare",
)
def compare(checkpoint, baseline):
    """Compare GNN with classical baselines."""
    from experiments.analysis.baseline_comparison import main as compare_main

    args = ["--checkpoint-dir", checkpoint]
    for b in baseline:
        args.extend(["--baseline", b])

    old_argv = sys.argv
    try:
        sys.argv = ["compare"] + args
        compare_main()
    finally:
        sys.argv = old_argv

@main.command()
@click.argument("checkpoint", type=click.Path(exists=True))
def demo(checkpoint):
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
    checkpoint,
    output_dir,
    baselines,
    run_cv,
    cv_folds,
    stratify_cv,
    alpha,
    n_bootstrap,
    check_power,
    strategy,
    n_samples,
    latex,
    figures,
    config,
    device,
    seed,
):
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

if __name__ == "__main__":
    main()
