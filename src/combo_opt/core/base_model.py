"""
Abstract base class for GNN models in combinatorial optimization.

Defines the interface that all GNN models must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data


class AbstractGNNModel(ABC, nn.Module):
    """
    Abstract base class for GNN models solving combinatorial optimization problems.

    All models must inherit from this class and implement the required abstract methods.
    This ensures a consistent interface across different GNN architectures.

    Example:
        >>> from combo_opt.core import AbstractGNNModel
        >>>
        >>> class MyGNNModel(AbstractGNNModel):
        ...     def __init__(self, hidden_dim=64):
        ...         super().__init__()
        ...         self.encoder = nn.Linear(2, hidden_dim)
        ...         self.decoder = nn.Linear(hidden_dim, 1)
        ...
        ...     def forward(self, data):
        ...         x = self.encoder(data.x)
        ...         return torch.sigmoid(self.decoder(x))
        ...
        ...     def get_embeddings(self, data):
        ...         return self.encoder(data.x)
        ...
        ...     @property
        ...     def input_dim(self):
        ...         return 2
    """

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass: graph data -> output probabilities/logits.

        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - (optional) edge_attr, batch, etc.

        Returns:
            Tensor of shape [num_decision_nodes] or [num_decision_nodes, num_classes]
            containing probabilities or logits for each decision variable.

        Example:
            >>> model = MyGNNModel()
            >>> probs = model(data)  # [num_items] for Knapsack
        """
        pass

    @abstractmethod
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """
        Extract node embeddings before final output layer.

        Useful for:
        - Visualization (t-SNE, UMAP)
        - Transfer learning
        - Debugging

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Tensor of shape [num_nodes, hidden_dim] containing node embeddings

        Example:
            >>> embeddings = model.get_embeddings(data)
            >>> print(embeddings.shape)  # [num_items, 64]
        """
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """
        Expected input feature dimension.

        Returns:
            Number of features per node (e.g., 2 for Knapsack: [weight, value])

        Example:
            >>> model = MyGNNModel()
            >>> print(model.input_dim)  # 2
        """
        pass

    def get_num_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            Total number of parameters

        Example:
            >>> model = MyGNNModel()
            >>> print(f"Parameters: {model.get_num_parameters():,}")
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Save model checkpoint with metadata.

        Args:
            path: Path to save checkpoint
            metadata: Optional metadata dict (epoch, metrics, etc.)

        Example:
            >>> model.save_checkpoint(
            ...     "checkpoints/model.pt",
            ...     metadata={"epoch": 50, "val_loss": 0.123}
            ... )
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
            "input_dim": self.input_dim,
        }
        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, **model_kwargs) -> "AbstractGNNModel":
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            **model_kwargs: Arguments to pass to model __init__

        Returns:
            Loaded model instance

        Example:
            >>> model = MyGNNModel.load_checkpoint("checkpoints/model.pt", hidden_dim=64)
        """
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
