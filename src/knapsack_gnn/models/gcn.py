"""
GCN-based GNN Architecture for Knapsack Problem
Simpler alternative to PNA using Graph Convolutional Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class HeterogeneousEncoder(nn.Module):
    """
    Heterogeneous encoder for different node types (items vs. constraint)
    Same as PNA version
    """

    def __init__(self, item_input_dim: int, constraint_input_dim: int, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.item_encoder = nn.Sequential(
            nn.Linear(item_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.constraint_encoder = nn.Sequential(
            nn.Linear(constraint_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        """Encode node features based on their types"""
        item_mask = node_types == 0
        constraint_mask = node_types == 1

        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        if item_mask.any():
            h[item_mask] = self.item_encoder(x[item_mask, :2])

        if constraint_mask.any():
            h[constraint_mask] = self.constraint_encoder(x[constraint_mask, :1])

        return h


class KnapsackGCN(nn.Module):
    """
    GCN-based GNN for Knapsack Problem

    Architecture:
    1. Heterogeneous encoding (items and constraint nodes)
    2. Multiple GCN message passing layers
    3. Decoding to item selection probabilities
    """

    def __init__(
        self,
        item_input_dim: int = 2,
        constraint_input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            item_input_dim: Input dimension for item features (default: 2)
            constraint_input_dim: Input dimension for constraint features (default: 1)
            hidden_dim: Hidden dimension for all layers (default: 64)
            num_layers: Number of GCN message passing layers (default: 3)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Heterogeneous encoder
        self.encoder = HeterogeneousEncoder(item_input_dim, constraint_input_dim, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = GCNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                improved=True,  # Use improved GCN with self-loops
                cached=False,
                normalize=True,
            )
            self.convs.append(conv)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Decoder (only for item nodes)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Item selection probabilities [n_items]
        """
        x, edge_index, node_types = data.x, data.edge_index, data.node_types

        # 1. Heterogeneous encoding
        h = self.encoder(x, node_types)

        # 2. Message passing with GCN
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            # Residual connection (except first layer)
            if i > 0:
                h = h + h_new
            else:
                h = h_new

        # 3. Decode item probabilities (only for item nodes)
        item_mask = node_types == 0
        item_embeddings = h[item_mask]

        # Get probabilities for each item
        probs: torch.Tensor = self.decoder(item_embeddings).squeeze(-1)  # [n_items]

        return probs

    def predict(self, data: Data, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary solution using threshold

        Args:
            data: PyTorch Geometric Data object
            threshold: Threshold for binary decision (default: 0.5)

        Returns:
            Binary solution [n_items]
        """
        probs = self.forward(data)
        return (probs >= threshold).float()


def create_gcn_model(
    hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1
) -> KnapsackGCN:
    """
    Factory function to create KnapsackGCN model

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of GCN layers
        dropout: Dropout rate

    Returns:
        Initialized KnapsackGCN model
    """
    print(f"Creating GCN model (hidden_dim={hidden_dim}, num_layers={num_layers})...")
    model = KnapsackGCN(
        item_input_dim=2,
        constraint_input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model


if __name__ == "__main__":
    # Example usage
    from data.graph_builder import KnapsackGraphBuilder
    from data.knapsack_generator import KnapsackGenerator, KnapsackSolver

    print("Creating sample instance...")
    generator = KnapsackGenerator(seed=42)
    instance = generator.generate_instance(n_items=10)
    instance = KnapsackSolver.solve(instance)

    print("Building graph...")
    builder = KnapsackGraphBuilder()
    graph = builder.build_graph(instance)

    print("\nCreating GCN model...")
    model = create_gcn_model(hidden_dim=64, num_layers=3)

    print("\nForward pass...")
    model.eval()
    with torch.no_grad():
        probs = model(graph)

    print(f"Output probabilities: {probs}")
    print(f"True solution: {graph.y}")
    print(f"Predicted solution (threshold=0.5): {(probs >= 0.5).float()}")

    # Compare model sizes
    print(f"\nGCN parameters: {sum(p.numel() for p in model.parameters())}")
