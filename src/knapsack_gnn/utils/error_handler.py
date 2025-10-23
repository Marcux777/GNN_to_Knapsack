"""
Error handling utilities for Knapsack GNN CLI.

Provides custom exception classes and decorators for handling errors
with informative messages and actionable suggestions.
"""

import functools
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================


class KnapsackGNNError(Exception):
    """Base exception for Knapsack GNN errors."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        """
        Initialize error with message and optional suggestion.

        Args:
            message: Error description
            suggestion: Actionable suggestion for fixing the error
        """
        self.message = message
        self.suggestion = suggestion
        super().__init__(message)

    def format_error(self) -> str:
        """Format error message with suggestion."""
        parts = [f"Error: {self.message}"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return "\n".join(parts)


class ConfigurationError(KnapsackGNNError):
    """Error related to configuration files or parameters."""

    pass


class CheckpointError(KnapsackGNNError):
    """Error related to model checkpoints."""

    pass


class DataError(KnapsackGNNError):
    """Error related to data loading or processing."""

    pass


class ModelError(KnapsackGNNError):
    """Error related to model initialization or inference."""

    pass


class ValidationError(KnapsackGNNError):
    """Error related to input validation."""

    pass


# ============================================================================
# Error Handlers
# ============================================================================


def format_exception_info(exc: Exception, show_traceback: bool = False) -> str:
    """
    Format exception information for display.

    Args:
        exc: The exception to format
        show_traceback: Whether to include full traceback

    Returns:
        Formatted error string
    """
    if isinstance(exc, KnapsackGNNError):
        # Custom errors have nice formatting
        return exc.format_error()
    elif show_traceback:
        # Full traceback for debugging
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    else:
        # Generic error message
        error_type = type(exc).__name__
        return f"Error ({error_type}): {str(exc)}"


def handle_cli_errors(
    debug_flag_name: str = "debug",
) -> Callable[[F], F]:
    """
    Decorator for CLI commands to handle errors gracefully.

    Args:
        debug_flag_name: Name of the debug flag in the command signature

    Returns:
        Decorator function

    Example:
        >>> @click.command()
        >>> @click.option("--debug", is_flag=True)
        >>> @handle_cli_errors()
        >>> def train(debug):
        ...     # command implementation
        ...     pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            debug_mode = kwargs.get(debug_flag_name, False)

            try:
                return func(*args, **kwargs)

            except KnapsackGNNError as e:
                # Custom errors: show formatted message
                click.secho(e.format_error(), fg="red", err=True)
                if debug_mode:
                    click.secho("\nFull traceback:", fg="yellow", err=True)
                    traceback.print_exc()
                sys.exit(1)

            except FileNotFoundError as e:
                # Common file errors with suggestions
                msg = f"File not found: {e.filename}"
                suggestion = "Check that the path exists and is spelled correctly."
                click.secho(f"Error: {msg}", fg="red", err=True)
                click.secho(f"Suggestion: {suggestion}", fg="yellow", err=True)
                if debug_mode:
                    traceback.print_exc()
                sys.exit(1)

            except PermissionError as e:
                msg = f"Permission denied: {e.filename}"
                suggestion = "Check file permissions or run with appropriate privileges."
                click.secho(f"Error: {msg}", fg="red", err=True)
                click.secho(f"Suggestion: {suggestion}", fg="yellow", err=True)
                if debug_mode:
                    traceback.print_exc()
                sys.exit(1)

            except KeyboardInterrupt:
                click.secho("\n\nOperation cancelled by user.", fg="yellow", err=True)
                sys.exit(130)  # Standard exit code for SIGINT

            except Exception as e:
                # Unexpected errors: show full traceback if debug, otherwise generic message
                if debug_mode:
                    click.secho("Unexpected error occurred:", fg="red", err=True)
                    traceback.print_exc()
                else:
                    error_type = type(e).__name__
                    click.secho(f"Unexpected error ({error_type}): {str(e)}", fg="red", err=True)
                    click.secho(
                        "\nTip: Run with --debug flag to see full traceback", fg="yellow", err=True
                    )
                sys.exit(1)

        return wrapper  # type: ignore

    return decorator


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    """
    Validate that checkpoint directory exists and contains required files.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Validated Path object

    Raises:
        CheckpointError: If checkpoint is invalid
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise CheckpointError(
            f"Checkpoint directory not found: {checkpoint_dir}",
            suggestion="Run training first or provide a valid checkpoint path.",
        )

    if not checkpoint_path.is_dir():
        raise CheckpointError(
            f"Checkpoint path is not a directory: {checkpoint_dir}",
            suggestion="Provide the directory containing model.pt, not the file itself.",
        )

    # Check for essential files
    required_files = ["model.pt", "config.yaml"]
    missing_files = [f for f in required_files if not (checkpoint_path / f).exists()]

    if missing_files:
        raise CheckpointError(
            f"Checkpoint directory is missing required files: {', '.join(missing_files)}",
            suggestion="Ensure the checkpoint was created successfully during training.",
        )

    return checkpoint_path


def validate_config_file(config_path: str | Path) -> Path:
    """
    Validate that configuration file exists and is readable.

    Args:
        config_path: Path to config file

    Returns:
        Validated Path object

    Raises:
        ConfigurationError: If config is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            suggestion="Check the path or create a config file (see examples/ for templates).",
        )

    if not config_file.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {config_path}",
            suggestion="Provide a path to a YAML configuration file.",
        )

    if config_file.suffix not in [".yaml", ".yml"]:
        raise ConfigurationError(
            f"Configuration file must be YAML format, got: {config_file.suffix}",
            suggestion="Use .yaml or .yml extension for configuration files.",
        )

    return config_file


def require_positive_int(value: int, name: str) -> int:
    """
    Validate that value is a positive integer.

    Args:
        value: Value to validate
        name: Parameter name for error messages

    Returns:
        Validated value

    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(
            f"{name} must be positive, got: {value}",
            suggestion=f"Provide a positive integer for {name}.",
        )
    return value


def require_probability(value: float, name: str) -> float:
    """
    Validate that value is a valid probability (0.0 to 1.0).

    Args:
        value: Value to validate
        name: Parameter name for error messages

    Returns:
        Validated value

    Raises:
        ValidationError: If value is not in [0, 1]
    """
    if not 0.0 <= value <= 1.0:
        raise ValidationError(
            f"{name} must be between 0.0 and 1.0, got: {value}",
            suggestion=f"Provide a probability value in [0, 1] for {name}.",
        )
    return value
