"""
Probability Calibration Analysis for GNN Models

Implements calibration metrics and methods:
    - Expected Calibration Error (ECE)
    - Brier Score
    - Reliability plots
    - Temperature scaling (post-hoc calibration)
    - Platt scaling

Goal: Ensure predicted probabilities are well-calibrated for decision-making.
Target: ECE < 0.1 after calibration
"""

import numpy as np
from scipy.optimize import minimize_scalar


class CalibrationMetrics:
    """
    Compute calibration metrics for binary classification probabilities.
    """

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> tuple[float, dict]:
        """
        Compute Expected Calibration Error (ECE).

        ECE measures the difference between predicted probabilities and actual frequencies.
        Lower is better. Target: ECE < 0.1

        Args:
            y_true: True binary labels (0 or 1)
            y_prob: Predicted probabilities [0, 1]
            n_bins: Number of bins for binning probabilities
            strategy: Binning strategy ('uniform' or 'quantile')

        Returns:
            Tuple of (ece_score, diagnostics_dict)

        Example:
            >>> y_true = np.array([1, 0, 1, 1, 0])
            >>> y_prob = np.array([0.9, 0.2, 0.8, 0.6, 0.3])
            >>> ece, diagnostics = CalibrationMetrics.expected_calibration_error(y_true, y_prob)
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have same length")

        # Create bins
        if strategy == "uniform":
            bin_edges = np.linspace(0, 1, n_bins + 1)
        elif strategy == "quantile":
            bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Assign samples to bins
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])

        # Compute ECE
        ece = 0.0
        bin_stats = []

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if not mask.any():
                continue

            bin_probs = y_prob[mask]
            bin_true = y_true[mask]

            # Accuracy (frequency of positives)
            accuracy = float(np.mean(bin_true))
            # Confidence (average predicted probability)
            confidence = float(np.mean(bin_probs))
            # Weight by number of samples
            weight = len(bin_probs) / len(y_true)

            # ECE contribution
            ece += weight * abs(accuracy - confidence)

            bin_stats.append(
                {
                    "bin_idx": bin_idx,
                    "bin_lower": float(bin_edges[bin_idx]),
                    "bin_upper": float(bin_edges[bin_idx + 1]),
                    "count": int(len(bin_probs)),
                    "accuracy": accuracy,
                    "confidence": confidence,
                    "gap": float(accuracy - confidence),
                }
            )

        diagnostics = {
            "ece": float(ece),
            "n_bins": n_bins,
            "strategy": strategy,
            "bin_stats": bin_stats,
        }

        return float(ece), diagnostics

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Compute Brier Score (Mean Squared Error for probabilities).

        Lower is better. Perfect score: 0.0

        Args:
            y_true: True binary labels (0 or 1)
            y_prob: Predicted probabilities [0, 1]

        Returns:
            Brier score

        Example:
            >>> y_true = np.array([1, 0, 1, 1, 0])
            >>> y_prob = np.array([0.9, 0.2, 0.8, 0.6, 0.3])
            >>> brier = CalibrationMetrics.brier_score(y_true, y_prob)
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def compute_reliability_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reliability curve (calibration plot).

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            strategy: Binning strategy

        Returns:
            Tuple of (mean_predicted_probs, fraction_of_positives, bin_counts)
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if strategy == "uniform":
            bin_edges = np.linspace(0, 1, n_bins + 1)
        elif strategy == "quantile":
            bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        bin_indices = np.digitize(y_prob, bin_edges[1:-1])

        mean_predicted = []
        fraction_positive = []
        counts = []

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if not mask.any():
                mean_predicted.append(np.nan)
                fraction_positive.append(np.nan)
                counts.append(0)
                continue

            mean_predicted.append(float(np.mean(y_prob[mask])))
            fraction_positive.append(float(np.mean(y_true[mask])))
            counts.append(int(mask.sum()))

        return np.array(mean_predicted), np.array(fraction_positive), np.array(counts)

    @staticmethod
    def maximum_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        MCE is the maximum gap between confidence and accuracy across bins.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins

        Returns:
            MCE score
        """
        _, diagnostics = CalibrationMetrics.expected_calibration_error(
            y_true, y_prob, n_bins=n_bins
        )

        gaps = [abs(bin_stat["gap"]) for bin_stat in diagnostics["bin_stats"]]

        if not gaps:
            return 0.0

        return float(max(gaps))


class TemperatureScaling:
    """
    Temperature scaling for post-hoc calibration.

    Scales logits by a learned temperature parameter T:
        p_calibrated = softmax(logits / T)

    For binary classification with sigmoid:
        p_calibrated = sigmoid(logits / T)
    """

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        method: str = "ece",
        max_iter: int = 100,
    ) -> float:
        """
        Learn optimal temperature on validation set.

        Args:
            logits: Raw model outputs (before sigmoid)
            y_true: True binary labels
            method: Optimization criterion ('ece' or 'nll')
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature
        """
        logits = np.asarray(logits)
        y_true = np.asarray(y_true)

        def objective(temp: float) -> float:
            if temp <= 0:
                return 1e10

            # Apply temperature scaling
            scaled_probs = self._sigmoid(logits / temp)

            if method == "ece":
                # Minimize ECE
                ece, _ = CalibrationMetrics.expected_calibration_error(y_true, scaled_probs)
                return ece
            elif method == "nll":
                # Minimize negative log-likelihood
                eps = 1e-10
                scaled_probs = np.clip(scaled_probs, eps, 1 - eps)
                nll = -np.mean(
                    y_true * np.log(scaled_probs) + (1 - y_true) * np.log(1 - scaled_probs)
                )
                return float(nll)
            else:
                raise ValueError(f"Unknown method: {method}")

        # Optimize temperature
        result = minimize_scalar(objective, bounds=(0.01, 100.0), method="bounded")
        self.temperature = float(result.x)

        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply learned temperature scaling to logits.

        Args:
            logits: Raw model outputs

        Returns:
            Calibrated probabilities
        """
        logits = np.asarray(logits)
        return self._sigmoid(logits / self.temperature)

    def fit_transform(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        method: str = "ece",
    ) -> np.ndarray:
        """
        Fit temperature and transform in one step.

        Args:
            logits: Raw model outputs
            y_true: True labels
            method: Optimization criterion

        Returns:
            Calibrated probabilities
        """
        self.fit(logits, y_true, method=method)
        return self.transform(logits)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class PlattScaling:
    """
    Platt scaling (logistic regression on outputs).

    Learns parameters A and B such that:
        p_calibrated = sigmoid(A * logits + B)
    """

    def __init__(self) -> None:
        self.A = 1.0
        self.B = 0.0

    def fit(
        self, logits: np.ndarray, y_true: np.ndarray, max_iter: int = 100
    ) -> tuple[float, float]:
        """
        Learn optimal A and B parameters.

        Args:
            logits: Raw model outputs
            y_true: True binary labels
            max_iter: Maximum optimization iterations

        Returns:
            Tuple of (A, B)
        """
        from sklearn.linear_model import LogisticRegression

        logits = np.asarray(logits).reshape(-1, 1)
        y_true = np.asarray(y_true)

        # Fit logistic regression
        lr = LogisticRegression(max_iter=max_iter, solver="lbfgs")
        lr.fit(logits, y_true)

        self.A = float(lr.coef_[0, 0])
        self.B = float(lr.intercept_[0])

        return self.A, self.B

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling.

        Args:
            logits: Raw model outputs

        Returns:
            Calibrated probabilities
        """
        logits = np.asarray(logits)
        scaled = self.A * logits + self.B
        return self._sigmoid(scaled)

    def fit_transform(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(logits, y_true)
        return self.transform(logits)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Comprehensive calibration evaluation.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for ECE

    Returns:
        Dictionary with all calibration metrics

    Example:
        >>> y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        >>> y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.95, 0.15, 0.25, 0.85, 0.75])
        >>> results = evaluate_calibration(y_true, y_prob)
        >>> print(f"ECE: {results['ece']:.4f}")
        >>> print(f"Brier: {results['brier_score']:.4f}")
    """
    metrics = CalibrationMetrics()

    # ECE
    ece, ece_diagnostics = metrics.expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    # MCE
    mce = metrics.maximum_calibration_error(y_true, y_prob, n_bins=n_bins)

    # Brier score
    brier = metrics.brier_score(y_true, y_prob)

    # Reliability curve
    mean_pred, frac_pos, counts = metrics.compute_reliability_curve(y_true, y_prob, n_bins=n_bins)

    # Accuracy
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = float(np.mean(y_pred == y_true))

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "accuracy": accuracy,
        "n_samples": len(y_true),
        "n_bins": n_bins,
        "ece_diagnostics": ece_diagnostics,
        "reliability_curve": {
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
            "counts": counts.tolist(),
        },
    }


if __name__ == "__main__":
    # Test calibration metrics
    print("Testing Calibration Metrics")
    print("=" * 80)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Well-calibrated probabilities
    y_true = np.random.binomial(1, 0.6, n_samples)
    logits_calibrated = np.random.randn(n_samples) * 1.5 + 0.5
    y_prob_calibrated = 1 / (1 + np.exp(-logits_calibrated))

    # Overconfident probabilities
    logits_overconfident = logits_calibrated * 2.0  # Higher temperature
    y_prob_overconfident = 1 / (1 + np.exp(-logits_overconfident))

    print("\n1. Well-Calibrated Probabilities")
    print("-" * 80)
    results_calibrated = evaluate_calibration(y_true, y_prob_calibrated)
    print(f"ECE: {results_calibrated['ece']:.4f}")
    print(f"MCE: {results_calibrated['mce']:.4f}")
    print(f"Brier: {results_calibrated['brier_score']:.4f}")
    print(f"Accuracy: {results_calibrated['accuracy']:.4f}")

    print("\n2. Overconfident Probabilities (Before Calibration)")
    print("-" * 80)
    results_overconfident = evaluate_calibration(y_true, y_prob_overconfident)
    print(f"ECE: {results_overconfident['ece']:.4f}")
    print(f"MCE: {results_overconfident['mce']:.4f}")
    print(f"Brier: {results_overconfident['brier_score']:.4f}")
    print(f"Accuracy: {results_overconfident['accuracy']:.4f}")

    print("\n3. Temperature Scaling")
    print("-" * 80)
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(logits_overconfident, y_true, method="ece")
    y_prob_temp_scaled = temp_scaler.transform(logits_overconfident)

    print(f"Optimal temperature: {optimal_temp:.4f}")
    results_temp_scaled = evaluate_calibration(y_true, y_prob_temp_scaled)
    print(f"ECE after scaling: {results_temp_scaled['ece']:.4f}")
    print(f"MCE after scaling: {results_temp_scaled['mce']:.4f}")
    print(f"Brier after scaling: {results_temp_scaled['brier_score']:.4f}")
    print(f"Accuracy after scaling: {results_temp_scaled['accuracy']:.4f}")

    print("\n4. Platt Scaling")
    print("-" * 80)
    platt_scaler = PlattScaling()
    A, B = platt_scaler.fit(logits_overconfident, y_true)
    y_prob_platt = platt_scaler.transform(logits_overconfident)

    print(f"Platt parameters: A={A:.4f}, B={B:.4f}")
    results_platt = evaluate_calibration(y_true, y_prob_platt)
    print(f"ECE after Platt: {results_platt['ece']:.4f}")
    print(f"MCE after Platt: {results_platt['mce']:.4f}")
    print(f"Brier after Platt: {results_platt['brier_score']:.4f}")
    print(f"Accuracy after Platt: {results_platt['accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("Calibration metrics computed successfully!")
