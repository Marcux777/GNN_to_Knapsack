"""
Pydantic schemas for configuration validation.

Defines the structure and validation rules for experiment configuration files.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ReproducibilityConfig(BaseModel):
    """Reproducibility settings for deterministic experiments."""

    deterministic: bool = Field(
        default=True, description="Enable deterministic algorithms (may reduce performance)"
    )
    benchmark: bool = Field(
        default=False,
        description="Enable CuDNN benchmark mode (faster but non-deterministic)",
    )

    @model_validator(mode="after")
    def check_determinism_conflict(self) -> "ReproducibilityConfig":
        """Ensure deterministic and benchmark are not both enabled."""
        if self.deterministic and self.benchmark:
            raise ValueError(
                "Cannot enable both deterministic mode and benchmark mode. "
                "Set benchmark=false for full reproducibility."
            )
        return self


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    enabled: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=10, description="Number of epochs to wait", ge=1)
    min_delta: float = Field(
        default=0.001, description="Minimum change to qualify as improvement", ge=0.0
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    epochs: int = Field(default=50, description="Number of training epochs", ge=1)
    batch_size: int = Field(default=32, description="Batch size for training", ge=1)
    learning_rate: float = Field(default=0.002, description="Learning rate", gt=0.0, le=1.0)
    weight_decay: float = Field(
        default=1e-6, description="Weight decay (L2 regularization)", ge=0.0
    )
    optimizer: Literal["adam", "adamw", "sgd"] = Field(default="adam", description="Optimizer type")
    scheduler: Literal["step", "cosine", "plateau"] | None = Field(
        default=None, description="Learning rate scheduler"
    )
    gradient_clip: float | None = Field(
        default=1.0, description="Gradient clipping value (None to disable)", gt=0.0
    )
    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig, description="Early stopping configuration"
    )


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    type: Literal["pna", "gcn", "gat"] = Field(default="pna", description="Model architecture type")
    hidden_dim: int = Field(default=128, description="Hidden dimension size", ge=8)
    num_layers: int = Field(default=4, description="Number of GNN layers", ge=1, le=10)
    dropout: float = Field(default=0.0, description="Dropout rate", ge=0.0, le=0.9)

    # PNA-specific
    aggregators: list[str] = Field(
        default=["mean", "max", "min", "std"],
        description="Aggregators for PNA (only for type='pna')",
    )
    scalers: list[str] = Field(
        default=["identity", "amplification", "attenuation"],
        description="Scalers for PNA (only for type='pna')",
    )

    @field_validator("aggregators", "scalers")
    @classmethod
    def check_non_empty(cls, v: list[str]) -> list[str]:
        """Ensure aggregators and scalers are non-empty."""
        if not v:
            raise ValueError("Must have at least one aggregator/scaler")
        return v


class DataConfig(BaseModel):
    """Dataset configuration."""

    n_items_min: int = Field(default=10, description="Minimum number of items", ge=1)
    n_items_max: int = Field(default=50, description="Maximum number of items", ge=1)
    train_size: int = Field(default=1000, description="Training set size", ge=1)
    val_size: int = Field(default=200, description="Validation set size", ge=1)
    test_size: int = Field(default=200, description="Test set size", ge=1)
    value_range: tuple[int, int] = Field(default=(1, 2500), description="Range for item values")
    weight_range: tuple[int, int] = Field(default=(1, 2500), description="Range for item weights")
    capacity_ratio: float = Field(
        default=0.5,
        description="Capacity as fraction of total weight",
        gt=0.0,
        le=1.0,
    )

    @field_validator("value_range", "weight_range")
    @classmethod
    def check_valid_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Ensure range is valid (min < max)."""
        if v[0] >= v[1]:
            raise ValueError(f"Invalid range: {v}. Min must be < Max.")
        if v[0] < 1:
            raise ValueError(f"Range minimum must be >= 1, got {v[0]}")
        return v

    @model_validator(mode="after")
    def check_items_range(self) -> "DataConfig":
        """Ensure n_items_min <= n_items_max."""
        if self.n_items_min > self.n_items_max:
            raise ValueError(
                f"n_items_min ({self.n_items_min}) must be <= n_items_max ({self.n_items_max})"
            )
        return self


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_interval: int = Field(default=10, description="Log every N batches", ge=1)
    save_best: bool = Field(default=True, description="Save best model checkpoint")
    save_last: bool = Field(default=True, description="Save last model checkpoint")
    metrics: list[str] = Field(default=["loss", "accuracy", "gap"], description="Metrics to track")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    seed: int = Field(default=42, description="Random seed for reproducibility", ge=0)
    device: str = Field(default="cpu", description="Device to use (cpu/cuda)")

    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration"
    )
    data: DataConfig = Field(default_factory=DataConfig, description="Data configuration")
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    reproducibility: ReproducibilityConfig = Field(
        default_factory=ReproducibilityConfig,
        description="Reproducibility configuration",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device string."""
        v = v.lower()
        if not v.startswith(("cpu", "cuda")):
            raise ValueError(f"Device must be 'cpu' or 'cuda', got '{v}'")
        return v

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int) -> int:
        """Validate seed range."""
        if not (0 <= v < 2**32):
            raise ValueError(f"Seed must be in range [0, {2**32 - 1}], got {v}")
        return v

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Raise error on unknown fields
        validate_assignment = True  # Validate on field assignment
