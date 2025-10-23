"""
Configuration loading and validation utilities.

Provides functions to load YAML configs and validate them against Pydantic schemas.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from knapsack_gnn.config.schemas import ExperimentConfig
from knapsack_gnn.utils.error_handler import ConfigurationError


def load_config(config_path: str | Path) -> ExperimentConfig:
    """
    Load and validate experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated ExperimentConfig object

    Raises:
        ConfigurationError: If file not found, invalid YAML, or validation fails

    Example:
        >>> config = load_config("configs/train_default.yaml")
        >>> print(f"Using seed: {config.seed}")
        >>> print(f"Model: {config.model.type} with {config.model.hidden_dim} hidden dims")
    """
    config_file = Path(config_path)

    # Check file exists
    if not config_file.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            suggestion="Check the path or create a config file (see configs/ for templates).",
        )

    # Load YAML
    try:
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML in config file: {config_path}",
            suggestion=f"Fix YAML syntax error: {e}",
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Failed to read config file: {config_path}",
            suggestion=f"Error: {e}",
        ) from e

    if config_dict is None:
        raise ConfigurationError(
            f"Empty configuration file: {config_path}",
            suggestion="Add configuration parameters to the YAML file.",
        )

    # Validate with Pydantic
    try:
        config = ExperimentConfig(**config_dict)
    except ValidationError as e:
        # Format validation errors nicely
        errors = []
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  - {loc}: {msg}")

        error_msg = "\n".join(errors)
        raise ConfigurationError(
            f"Configuration validation failed for {config_path}:\n{error_msg}",
            suggestion="Fix the configuration errors listed above. "
            "See configs/train_default.yaml for a valid example.",
        ) from e

    return config


def validate_config_file(config_path: str | Path) -> tuple[bool, str]:
    """
    Validate config file without raising exceptions.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (is_valid, message)
        - is_valid: True if config is valid
        - message: Success message or error details

    Example:
        >>> is_valid, msg = validate_config_file("configs/train_default.yaml")
        >>> if is_valid:
        ...     print("Config is valid!")
        ... else:
        ...     print(f"Validation failed: {msg}")
    """
    try:
        load_config(config_path)
        return True, f"✓ Configuration is valid: {config_path}"
    except ConfigurationError as e:
        return False, f"✗ {e.message}"
    except Exception as e:
        return False, f"✗ Unexpected error: {e}"


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """
    Convert ExperimentConfig to dictionary.

    Args:
        config: ExperimentConfig instance

    Returns:
        Dictionary representation of config

    Example:
        >>> config = load_config("configs/train_default.yaml")
        >>> config_dict = config_to_dict(config)
        >>> import yaml
        >>> with open("config_copy.yaml", "w") as f:
        ...     yaml.dump(config_dict, f)
    """
    return config.model_dump()


def save_config(config: ExperimentConfig, output_path: str | Path) -> None:
    """
    Save ExperimentConfig to YAML file.

    Args:
        config: ExperimentConfig instance
        output_path: Path to save YAML file

    Example:
        >>> config = load_config("configs/train_default.yaml")
        >>> save_config(config, "checkpoints/run_001/config.yaml")
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config_to_dict(config)

    with open(output_file, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
