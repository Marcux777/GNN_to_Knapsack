"""
Structured logging configuration for reproducible experiments.

Provides centralized logging setup with file handlers, console output,
and appropriate formatting for scientific experiments.
"""

import logging
import sys
from pathlib import Path

def setup_logger(
    name: str = "knapsack_gnn",
    log_file: Path | None = None,
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with file and/or console handlers.

    Args:
        name: Logger name (typically module name or "knapsack_gnn")
        log_file: Path to log file (if None, only console logging)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: If True, also log to console (stdout)

    Returns:
        Configured logger instance

    Example:
        >>> from pathlib import Path
        >>> from utils.logger import setup_logger
        >>> logger = setup_logger(
        ...     name="training",
        ...     log_file=Path("checkpoints/run_001/training.log"),
        ...     level=logging.INFO
        ... )
        >>> logger.info("Training started")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (if log_file provided)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger

def get_logger(name: str = "knapsack_gnn") -> logging.Logger:
    """
    Get an existing logger or create a basic one.

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing instance")
    """
    logger = logging.getLogger(name)

    # If no handlers, set up basic console logging
    if not logger.handlers:
        logger = setup_logger(name, log_file=None, console_output=True)

    return logger

def log_experiment_config(
    logger: logging.Logger, config: dict, title: str = "Experiment Configuration"
) -> None:
    """
    Log experiment configuration in a structured format.

    Args:
        logger: Logger instance
        config: Configuration dictionary
        title: Title for the config block

    Example:
        >>> logger = get_logger()
        >>> config = {"seed": 42, "lr": 0.002, "batch_size": 32}
        >>> log_experiment_config(logger, config, "Training Config")
    """
    logger.info("=" * 60)
    logger.info(f"{title:^60}")
    logger.info("=" * 60)

    for key, value in sorted(config.items()):
        logger.info(f"  {key:.<30} {value}")

    logger.info("=" * 60)

def log_metrics(
    logger: logging.Logger, metrics: dict, prefix: str = "", precision: int = 4
) -> None:
    """
    Log metrics in a formatted way.

    Args:
        logger: Logger instance
        metrics: Dictionary of metric name -> value
        prefix: Prefix string (e.g., "Epoch 10 |")
        precision: Number of decimal places for float formatting

    Example:
        >>> logger = get_logger()
        >>> metrics = {"loss": 0.123, "accuracy": 0.956, "gap": 0.0007}
        >>> log_metrics(logger, metrics, prefix="Epoch 10 |", precision=4)
    """
    metric_strs = []
    for name, value in metrics.items():
        if isinstance(value, float):
            metric_strs.append(f"{name}: {value:.{precision}f}")
        else:
            metric_strs.append(f"{name}: {value}")

    message = " | ".join(metric_strs)
    if prefix:
        message = f"{prefix} {message}"

    logger.info(message)
