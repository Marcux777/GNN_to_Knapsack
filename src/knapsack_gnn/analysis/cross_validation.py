"""
K-Fold Cross-Validation for GNN Training

Provides robust performance estimation through cross-validation:
- Standard k-fold cross-validation
- Stratified k-fold (by problem size)
- Leave-one-size-out validation
- Nested CV for hyperparameter selection

Usage:
    from knapsack_gnn.analysis.cross_validation import KFoldValidator

    validator = KFoldValidator(n_splits=5)
    cv_results = validator.validate(model_class, dataset, config)
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from knapsack_gnn.analysis.stats import StatisticalAnalyzer
from knapsack_gnn.data.generator import KnapsackDataset


@dataclass
class CVFold:
    """Single fold in cross-validation"""

    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray | None = None


@dataclass
class CVResults:
    """Results from cross-validation"""

    fold_results: list[dict]
    mean_gap: float
    std_gap: float
    ci_95: tuple[float, float]
    mean_train_loss: float
    mean_val_loss: float
    best_fold_id: int
    worst_fold_id: int

    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"Cross-Validation Results ({len(self.fold_results)} folds):\n"
            f"  Mean Gap: {self.mean_gap:.4f}% ± {self.std_gap:.4f}%\n"
            f"  95% CI: [{self.ci_95[0]:.4f}%, {self.ci_95[1]:.4f}%]\n"
            f"  Mean Train Loss: {self.mean_train_loss:.4f}\n"
            f"  Mean Val Loss: {self.mean_val_loss:.4f}\n"
            f"  Best Fold: {self.best_fold_id} ({self.fold_results[self.best_fold_id]['gap']:.4f}%)\n"
            f"  Worst Fold: {self.worst_fold_id} ({self.fold_results[self.worst_fold_id]['gap']:.4f}%)"
        )


class KFoldValidator:
    """
    K-Fold Cross-Validation for GNN models

    Provides robust performance estimates by training on different data splits.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        stratify: bool = False,
        stratify_bins: int = 5,
    ):
        """
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            stratify: Whether to stratify by problem size
            stratify_bins: Number of bins for stratification
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify
        self.stratify_bins = stratify_bins
        self.rng = np.random.RandomState(random_state)

    def create_folds(
        self, dataset: KnapsackDataset, test_size: float | None = None
    ) -> list[CVFold]:
        """
        Create k-fold splits

        Args:
            dataset: KnapsackDataset
            test_size: Optional fraction to hold out as test set

        Returns:
            List of CVFold objects
        """
        n_instances = len(dataset)
        indices = np.arange(n_instances)

        # Shuffle if requested
        if self.shuffle:
            self.rng.shuffle(indices)

        # Hold out test set if requested
        test_indices = None
        if test_size is not None:
            n_test = int(n_instances * test_size)
            test_indices = indices[:n_test]
            indices = indices[n_test:]
            n_instances = len(indices)

        # Stratify by problem size if requested
        if self.stratify:
            sizes = np.array([inst.n_items for inst in dataset.instances])
            sizes = sizes[indices]

            # Create bins
            size_bins = np.percentile(sizes, np.linspace(0, 100, self.stratify_bins + 1))
            bin_assignments = np.digitize(sizes, size_bins[1:-1])

            # Stratified k-fold
            folds = self._stratified_kfold(indices, bin_assignments)
        else:
            # Standard k-fold
            folds = self._standard_kfold(indices)

        # Create CVFold objects
        cv_folds = []
        for fold_id, (train_idx, val_idx) in enumerate(folds):
            cv_folds.append(
                CVFold(
                    fold_id=fold_id,
                    train_indices=train_idx,
                    val_indices=val_idx,
                    test_indices=test_indices,
                )
            )

        return cv_folds

    def _standard_kfold(self, indices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Standard k-fold split"""
        n_instances = len(indices)
        fold_size = n_instances // self.n_splits

        folds = []
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_instances

            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            folds.append((train_indices, val_indices))

        return folds

    def _stratified_kfold(
        self, indices: np.ndarray, bin_assignments: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Stratified k-fold split"""
        folds = [[] for _ in range(self.n_splits)]

        # For each bin, distribute instances across folds
        for bin_id in range(self.stratify_bins):
            bin_mask = bin_assignments == bin_id
            bin_indices = indices[bin_mask]

            # Shuffle bin indices
            self.rng.shuffle(bin_indices)

            # Distribute to folds
            bin_fold_size = len(bin_indices) // self.n_splits
            for i in range(self.n_splits):
                start = i * bin_fold_size
                end = (i + 1) * bin_fold_size if i < self.n_splits - 1 else len(bin_indices)
                folds[i].extend(bin_indices[start:end])

        # Convert to train/val splits
        result_folds = []
        for i in range(self.n_splits):
            val_indices = np.array(folds[i])
            train_indices = np.concatenate(
                [np.array(folds[j]) for j in range(self.n_splits) if j != i]
            )
            result_folds.append((train_indices, val_indices))

        return result_folds

    def validate(
        self,
        train_fn: Callable,
        evaluate_fn: Callable,
        dataset: KnapsackDataset,
        config: dict,
        device: str = "cpu",
        verbose: bool = True,
    ) -> CVResults:
        """
        Run k-fold cross-validation

        Args:
            train_fn: Function(train_dataset, val_dataset, config, device) -> model
            evaluate_fn: Function(model, test_dataset, device) -> Dict with 'mean_gap'
            dataset: Full dataset
            config: Configuration dictionary
            device: Device to use
            verbose: Print progress

        Returns:
            CVResults object
        """
        folds = self.create_folds(dataset)
        fold_results = []

        for fold in folds:
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Fold {fold.fold_id + 1}/{self.n_splits}")
                print(f"{'=' * 70}")

            # Create train/val datasets for this fold
            train_instances = [dataset.instances[i] for i in fold.train_indices]
            val_instances = [dataset.instances[i] for i in fold.val_indices]

            train_dataset = KnapsackDataset(train_instances)
            val_dataset = KnapsackDataset(val_instances)

            # Train model
            if verbose:
                print(f"Training on {len(train_instances)} instances...")

            model, history = train_fn(train_dataset, val_dataset, config, device)

            # Evaluate
            if verbose:
                print(f"Evaluating on {len(val_instances)} instances...")

            eval_results = evaluate_fn(model, val_dataset, device)

            # Store results
            fold_result = {
                "fold_id": fold.fold_id,
                "gap": eval_results["mean_gap"],
                "train_loss": history["train_loss"][-1] if history else None,
                "val_loss": history["val_loss"][-1] if history else None,
                "n_train": len(train_instances),
                "n_val": len(val_instances),
            }
            fold_results.append(fold_result)

            if verbose:
                print(f"Fold {fold.fold_id + 1} Gap: {fold_result['gap']:.4f}%")

        # Aggregate results
        gaps = [r["gap"] for r in fold_results]
        train_losses = [r["train_loss"] for r in fold_results if r["train_loss"] is not None]
        val_losses = [r["val_loss"] for r in fold_results if r["val_loss"] is not None]

        # Compute statistics
        analyzer = StatisticalAnalyzer()
        ci_95 = analyzer.bootstrap_ci(np.array(gaps))

        # Find best and worst folds
        best_fold_id = int(np.argmin(gaps))
        worst_fold_id = int(np.argmax(gaps))

        results = CVResults(
            fold_results=fold_results,
            mean_gap=float(np.mean(gaps)),
            std_gap=float(np.std(gaps, ddof=1)),
            ci_95=ci_95,
            mean_train_loss=float(np.mean(train_losses)) if train_losses else 0.0,
            mean_val_loss=float(np.mean(val_losses)) if val_losses else 0.0,
            best_fold_id=best_fold_id,
            worst_fold_id=worst_fold_id,
        )

        if verbose:
            print(f"\n{'=' * 70}")
            print("CROSS-VALIDATION SUMMARY")
            print(f"{'=' * 70}")
            print(results.summary())
            print(f"{'=' * 70}\n")

        return results


class LeaveOneSizeOutValidator:
    """
    Leave-One-Size-Out Cross-Validation

    Tests generalization by leaving out all instances of a particular size.
    Extreme OOD test.
    """

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: Random seed
        """
        self.random_state = random_state

    def create_folds(self, dataset: KnapsackDataset) -> list[CVFold]:
        """
        Create leave-one-size-out folds

        Args:
            dataset: KnapsackDataset

        Returns:
            List of CVFold objects (one per unique size)
        """
        # Get sizes
        sizes = np.array([inst.n_items for inst in dataset.instances])
        unique_sizes = np.unique(sizes)

        folds = []
        for fold_id, size in enumerate(unique_sizes):
            # Instances with this size are validation
            val_mask = sizes == size
            val_indices = np.where(val_mask)[0]

            # All other instances are training
            train_indices = np.where(~val_mask)[0]

            folds.append(
                CVFold(fold_id=fold_id, train_indices=train_indices, val_indices=val_indices)
            )

        return folds

    def validate(
        self,
        train_fn: Callable,
        evaluate_fn: Callable,
        dataset: KnapsackDataset,
        config: dict,
        device: str = "cpu",
        verbose: bool = True,
    ) -> CVResults:
        """
        Run leave-one-size-out validation

        Similar interface to KFoldValidator.validate()
        """
        folds = self.create_folds(dataset)
        fold_results = []

        sizes = np.array([inst.n_items for inst in dataset.instances])
        unique_sizes = np.unique(sizes)

        for fold, size in zip(folds, unique_sizes, strict=False):
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Leave-out size: {size} items")
                print(f"{'=' * 70}")

            # Create datasets
            train_instances = [dataset.instances[i] for i in fold.train_indices]
            val_instances = [dataset.instances[i] for i in fold.val_indices]

            train_dataset = KnapsackDataset(train_instances)
            val_dataset = KnapsackDataset(val_instances)

            # Train
            if verbose:
                print(f"Training on {len(train_instances)} instances (excluding size {size})...")

            model, history = train_fn(train_dataset, val_dataset, config, device)

            # Evaluate
            if verbose:
                print(f"Evaluating on {len(val_instances)} instances (size {size})...")

            eval_results = evaluate_fn(model, val_dataset, device)

            # Store results
            fold_result = {
                "fold_id": fold.fold_id,
                "size": int(size),
                "gap": eval_results["mean_gap"],
                "train_loss": history["train_loss"][-1] if history else None,
                "val_loss": history["val_loss"][-1] if history else None,
                "n_train": len(train_instances),
                "n_val": len(val_instances),
            }
            fold_results.append(fold_result)

            if verbose:
                print(f"Size {size} Gap: {fold_result['gap']:.4f}%")

        # Aggregate
        gaps = [r["gap"] for r in fold_results]
        train_losses = [r["train_loss"] for r in fold_results if r["train_loss"] is not None]
        val_losses = [r["val_loss"] for r in fold_results if r["val_loss"] is not None]

        analyzer = StatisticalAnalyzer()
        ci_95 = analyzer.bootstrap_ci(np.array(gaps))

        best_fold_id = int(np.argmin(gaps))
        worst_fold_id = int(np.argmax(gaps))

        results = CVResults(
            fold_results=fold_results,
            mean_gap=float(np.mean(gaps)),
            std_gap=float(np.std(gaps, ddof=1)),
            ci_95=ci_95,
            mean_train_loss=float(np.mean(train_losses)) if train_losses else 0.0,
            mean_val_loss=float(np.mean(val_losses)) if val_losses else 0.0,
            best_fold_id=best_fold_id,
            worst_fold_id=worst_fold_id,
        )

        if verbose:
            print(f"\n{'=' * 70}")
            print("LEAVE-ONE-SIZE-OUT SUMMARY")
            print(f"{'=' * 70}")
            print(results.summary())
            print(f"{'=' * 70}\n")

        return results


class NestedCVValidator:
    """
    Nested Cross-Validation for Hyperparameter Selection

    Outer loop: Model evaluation
    Inner loop: Hyperparameter selection

    Provides unbiased performance estimates when doing hyperparameter tuning.
    """

    def __init__(self, n_outer_splits: int = 5, n_inner_splits: int = 3, random_state: int = 42):
        """
        Args:
            n_outer_splits: Number of outer CV folds
            n_inner_splits: Number of inner CV folds
            random_state: Random seed
        """
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.random_state = random_state

    def validate(
        self,
        train_fn: Callable,
        evaluate_fn: Callable,
        dataset: KnapsackDataset,
        hyperparameter_configs: list[dict],
        base_config: dict,
        device: str = "cpu",
        verbose: bool = True,
    ) -> dict:
        """
        Run nested cross-validation

        Args:
            train_fn: Training function
            evaluate_fn: Evaluation function
            dataset: Full dataset
            hyperparameter_configs: List of hyperparameter configurations to try
            base_config: Base configuration (shared across all configs)
            device: Device
            verbose: Verbose output

        Returns:
            Dictionary with nested CV results
        """
        # Outer CV
        outer_cv = KFoldValidator(n_splits=self.n_outer_splits, random_state=self.random_state)
        outer_folds = outer_cv.create_folds(dataset)

        outer_results = []
        best_configs = []

        for outer_fold in outer_folds:
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Outer Fold {outer_fold.fold_id + 1}/{self.n_outer_splits}")
                print(f"{'=' * 70}")

            # Split data for this outer fold
            outer_train_indices = outer_fold.train_indices
            outer_test_indices = outer_fold.val_indices

            outer_train_instances = [dataset.instances[i] for i in outer_train_indices]
            outer_test_instances = [dataset.instances[i] for i in outer_test_indices]

            outer_train_dataset = KnapsackDataset(outer_train_instances)
            outer_test_dataset = KnapsackDataset(outer_test_instances)

            # Inner CV for hyperparameter selection
            inner_cv = KFoldValidator(
                n_splits=self.n_inner_splits, random_state=self.random_state + outer_fold.fold_id
            )

            config_scores = []
            for config_id, hp_config in enumerate(hyperparameter_configs):
                # Merge configs
                config = {**base_config, **hp_config}

                if verbose:
                    print(f"\n  Config {config_id + 1}/{len(hyperparameter_configs)}: {hp_config}")

                # Run inner CV
                inner_results = inner_cv.validate(
                    train_fn=train_fn,
                    evaluate_fn=evaluate_fn,
                    dataset=outer_train_dataset,
                    config=config,
                    device=device,
                    verbose=False,
                )

                config_scores.append(
                    {
                        "config_id": config_id,
                        "config": hp_config,
                        "mean_gap": inner_results.mean_gap,
                        "std_gap": inner_results.std_gap,
                    }
                )

                if verbose:
                    print(
                        f"    Inner CV Gap: {inner_results.mean_gap:.4f}% ± {inner_results.std_gap:.4f}%"
                    )

            # Select best config
            best_config_idx = min(
                range(len(config_scores)), key=lambda i: config_scores[i]["mean_gap"]
            )
            best_config = hyperparameter_configs[best_config_idx]
            best_configs.append(best_config)

            if verbose:
                print(f"\n  Best config: {best_config}")
                print(f"  Inner CV Gap: {config_scores[best_config_idx]['mean_gap']:.4f}%")

            # Train final model with best config on full outer training set
            final_config = {**base_config, **best_config}

            # Use a small val set from outer training for early stopping
            val_size = int(len(outer_train_instances) * 0.1)
            final_train_instances = outer_train_instances[:-val_size]
            final_val_instances = outer_train_instances[-val_size:]

            final_train_dataset = KnapsackDataset(final_train_instances)
            final_val_dataset = KnapsackDataset(final_val_instances)

            if verbose:
                print(f"\n  Training final model on {len(final_train_instances)} instances...")

            model, history = train_fn(final_train_dataset, final_val_dataset, final_config, device)

            # Evaluate on outer test set
            if verbose:
                print(f"  Evaluating on {len(outer_test_instances)} instances...")

            eval_results = evaluate_fn(model, outer_test_dataset, device)

            outer_results.append(
                {
                    "outer_fold_id": outer_fold.fold_id,
                    "best_config": best_config,
                    "gap": eval_results["mean_gap"],
                    "all_config_scores": config_scores,
                }
            )

            if verbose:
                print(f"  Outer Test Gap: {eval_results['mean_gap']:.4f}%")

        # Aggregate outer results
        outer_gaps = [r["gap"] for r in outer_results]

        analyzer = StatisticalAnalyzer()
        ci_95 = analyzer.bootstrap_ci(np.array(outer_gaps))

        results = {
            "mean_gap": float(np.mean(outer_gaps)),
            "std_gap": float(np.std(outer_gaps, ddof=1)),
            "ci_95": ci_95,
            "outer_results": outer_results,
            "best_configs_per_fold": best_configs,
            "most_common_config": self._most_common_config(best_configs),
        }

        if verbose:
            print(f"\n{'=' * 70}")
            print("NESTED CV SUMMARY")
            print(f"{'=' * 70}")
            print(f"Outer CV Gap: {results['mean_gap']:.4f}% ± {results['std_gap']:.4f}%")
            print(f"95% CI: [{ci_95[0]:.4f}%, {ci_95[1]:.4f}%]")
            print(f"Most common best config: {results['most_common_config']}")
            print(f"{'=' * 70}\n")

        return results

    def _most_common_config(self, configs: list[dict]) -> dict:
        """Find most frequently selected configuration"""
        # Convert configs to hashable format
        config_strs = [str(sorted(c.items())) for c in configs]

        # Count occurrences
        from collections import Counter

        counts = Counter(config_strs)
        most_common_str = counts.most_common(1)[0][0]

        # Find original config
        for c in configs:
            if str(sorted(c.items())) == most_common_str:
                return c

        return configs[0]  # Fallback


if __name__ == "__main__":
    print("Cross-Validation Module")
    print("=" * 70)
    print("\nThis module provides k-fold cross-validation for GNN models.")
    print("\nExample usage:")
    print("""
    from knapsack_gnn.analysis.cross_validation import KFoldValidator

    validator = KFoldValidator(n_splits=5, stratify=True)
    cv_results = validator.validate(
        train_fn=my_train_function,
        evaluate_fn=my_eval_function,
        dataset=my_dataset,
        config=my_config
    )

    print(cv_results.summary())
    """)
