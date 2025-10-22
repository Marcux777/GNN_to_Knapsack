"""PNA-based GNN architecture for the Knapsack Problem.

This module implements a Principal Neighborhood Aggregation (PNA) graph neural network
tailored for solving the 0-1 Knapsack Problem. The architecture uses heterogeneous node
encoding to handle different node types (items vs. capacity constraint), multiple PNA
message passing layers for expressive feature learning, and specialized decoding for
item selection probabilities.

Key components:
    - HeterogeneousEncoder: Separate encoders for item and constraint nodes
    - PNAConv layers: Multi-scale aggregation with degree-aware scaling
    - ItemDecoder: Converts node embeddings to selection probabilities

Typical usage:
    >>> model = KnapsackPNA(hidden_dim=64, num_layers=3)
    >>> probs = model(graph_data)  # Returns selection probabilities

References:
    Corso et al. "Principal Neighbourhood Aggregation for Graph Nets"
    https://arxiv.org/abs/2004.05718
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, List

class HeterogeneousEncoder(nn.Module):
    """Heterogeneous encoder for different node types in Knapsack graphs.

    Encodes item nodes and constraint nodes separately using dedicated MLPs,
    allowing the model to learn different representations for different node types.
    Item nodes represent items with (weight, value) features, while the constraint
    node represents the knapsack capacity.

    Attributes:
        item_encoder: 2-layer MLP for encoding item nodes.
        constraint_encoder: 2-layer MLP for encoding constraint node.

    Examples:
        >>> encoder = HeterogeneousEncoder(item_input_dim=2, constraint_input_dim=1, hidden_dim=64)
        >>> x = torch.tensor([[10.0, 20.0], [15.0, 30.0], [100.0, 0.0]])  # 2 items + 1 constraint
        >>> node_types = torch.tensor([0, 0, 1])  # 0=item, 1=constraint
        >>> encoded = encoder(x, node_types)
        >>> encoded.shape
        torch.Size([3, 64])
    """

    def __init__(self, item_input_dim: int, constraint_input_dim: int, hidden_dim: int):
        """Initialize the heterogeneous encoder.

        Args:
            item_input_dim: Input feature dimension for item nodes (weight, value), typically 2.
            constraint_input_dim: Input feature dimension for constraint node (capacity), typically 1.
            hidden_dim: Output hidden dimension for encoded node features.
        """
        super().__init__()

        # Separate MLPs for different node types
        self.item_encoder = nn.Sequential(
            nn.Linear(item_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.constraint_encoder = nn.Sequential(
            nn.Linear(constraint_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        """Encode node features based on their types.

        Item nodes use the first 2 features (weight, value) and are encoded via item_encoder.
        Constraint nodes use only the first feature (capacity) and are encoded via constraint_encoder.

        Args:
            x: Node features tensor of shape [num_nodes, feature_dim]. For item nodes,
               expects at least 2 features (weight, value). For constraint node, expects
               at least 1 feature (capacity).
            node_types: Node type indicators tensor of shape [num_nodes] where 0 indicates
                       item node and 1 indicates constraint node.

        Returns:
            Encoded node features tensor of shape [num_nodes, hidden_dim].

        Note:
            This method assumes at most one constraint node per graph. Item nodes can
            number from 1 to n_items.
        """
        # Separate item and constraint nodes
        item_mask = (node_types == 0)
        constraint_mask = (node_types == 1)

        # Initialize output tensor
        h = torch.zeros(x.size(0), self.item_encoder[0].out_features, device=x.device)

        # Encode items (first 2 features: weight, value)
        if item_mask.any():
            h[item_mask] = self.item_encoder(x[item_mask, :2])

        # Encode constraint (last feature: capacity)
        if constraint_mask.any():
            h[constraint_mask] = self.constraint_encoder(x[constraint_mask, :1])

        return h

class KnapsackPNA(nn.Module):
    """PNA-based Graph Neural Network for the 0-1 Knapsack Problem.

    This model uses Principal Neighborhood Aggregation (PNA) to learn item selection
    probabilities for the knapsack problem. The architecture consists of three main stages:

    1. **Heterogeneous Encoding**: Separate encoders for item and constraint nodes
    2. **Message Passing**: Multiple PNA convolution layers with multi-scale aggregation
    3. **Decoding**: MLP decoder that outputs selection probabilities for each item

    The model operates on bipartite graphs where item nodes connect to a single constraint
    node representing the capacity. After message passing, only item node embeddings are
    decoded to produce selection probabilities.

    Attributes:
        encoder: HeterogeneousEncoder for initial node embedding.
        convs: ModuleList of PNA convolution layers.
        dropout: Dropout layer for regularization.
        decoder: MLP decoder for item selection probabilities.

    Examples:
        >>> # Create model
        >>> model = KnapsackPNA(hidden_dim=64, num_layers=3)
        >>> # Forward pass
        >>> probs = model(graph_data)
        >>> # probs is a tensor of shape [n_items] with values in [0, 1]

    References:
        Corso et al. "Principal Neighbourhood Aggregation for Graph Nets" (NeurIPS 2020)
    """

    def __init__(self,
                 item_input_dim: int = 2,
                 constraint_input_dim: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 aggregators: List[str] = ['mean', 'max', 'min', 'std'],
                 scalers: List[str] = ['identity', 'amplification', 'attenuation'],
                 deg: Optional[torch.Tensor] = None):
        """Initialize the KnapsackPNA model.

        Args:
            item_input_dim: Input feature dimension for item nodes (default: 2 for weight/value).
            constraint_input_dim: Input feature dimension for constraint node (default: 1 for capacity).
            hidden_dim: Hidden dimension size for all layers (default: 64).
            num_layers: Number of PNA message passing layers (default: 3). More layers allow
                       deeper feature aggregation but increase computation.
            dropout: Dropout probability for regularization (default: 0.1).
            aggregators: List of aggregation functions for PNA. Default uses mean, max, min, std
                        for multi-scale aggregation.
            scalers: List of scaling functions for PNA. Default uses identity, amplification,
                    attenuation for degree-aware scaling.
            deg: Pre-computed degree histogram for PNA normalization. If None, PNA will compute
                it automatically during the first forward pass.

        Note:
            The default aggregators and scalers provide a good balance between expressiveness
            and computational cost. For smaller problems, reducing aggregators/scalers can
            speed up training.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Heterogeneous encoder
        self.encoder = HeterogeneousEncoder(item_input_dim, constraint_input_dim, hidden_dim)

        # PNA layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=None,  # No edge features for now
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False
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
            nn.Sigmoid()  # Output probabilities in [0, 1]
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

        # 2. Message passing
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
        item_mask = (node_types == 0)
        item_embeddings = h[item_mask]

        # Get probabilities for each item
        probs = self.decoder(item_embeddings).squeeze(-1)  # [n_items]

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

def compute_degree_histogram(dataset, max_degree: int = 100) -> torch.Tensor:
    """
    Compute degree histogram for PNA normalization

    Args:
        dataset: KnapsackGraphDataset
        max_degree: Maximum degree to consider

    Returns:
        Degree histogram tensor
    """
    from torch_geometric.utils import degree

    deg_histogram = torch.zeros(max_degree + 1, dtype=torch.long)

    for data in dataset:
        # Compute in-degrees
        d = degree(data.edge_index[1], num_nodes=data.x.size(0), dtype=torch.long)
        deg_histogram += torch.bincount(d, minlength=deg_histogram.numel())[:max_degree + 1]

    return deg_histogram.float()

class KnapsackPNAWithBatch(KnapsackPNA):
    """
    Extended PNA model that handles batched graphs
    """

    def forward_batch(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass for batched graphs

        Args:
            batch: Batched PyTorch Geometric Data

        Returns:
            Item selection probabilities for all graphs in batch
        """
        x, edge_index, node_types = batch.x, batch.edge_index, batch.node_types

        # 1. Heterogeneous encoding
        h = self.encoder(x, node_types)

        # 2. Message passing
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            # Residual connection
            if i > 0:
                h = h + h_new
            else:
                h = h_new

        # 3. Decode item probabilities (only for item nodes)
        item_mask = (node_types == 0)
        item_embeddings = h[item_mask]

        # Get probabilities for each item
        probs = self.decoder(item_embeddings).squeeze(-1)

        return probs

def create_model(dataset,
                hidden_dim: int = 64,
                num_layers: int = 3,
                dropout: float = 0.1) -> KnapsackPNA:
    """
    Factory function to create KnapsackPNA model with computed degree histogram

    Args:
        dataset: KnapsackGraphDataset
        hidden_dim: Hidden dimension
        num_layers: Number of PNA layers
        dropout: Dropout rate

    Returns:
        Initialized KnapsackPNA model
    """
    print("Computing degree histogram for PNA...")
    deg = compute_degree_histogram(dataset)

    print(f"Creating PNA model (hidden_dim={hidden_dim}, num_layers={num_layers})...")
    model = KnapsackPNA(
        item_input_dim=2,
        constraint_input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        deg=deg
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model

if __name__ == '__main__':
    # Example usage
    from data.knapsack_generator import KnapsackGenerator, KnapsackSolver
    from data.graph_builder import KnapsackGraphBuilder

    print("Creating sample instance...")
    generator = KnapsackGenerator(seed=42)
    instance = generator.generate_instance(n_items=10)
    instance = KnapsackSolver.solve(instance)

    print("Building graph...")
    builder = KnapsackGraphBuilder()
    graph = builder.build_graph(instance)

    print("\nCreating model...")
    # For demo, use simple degree histogram
    deg = torch.tensor([0.0] * 20)
    deg[10] = 10.0  # Most nodes have degree ~10

    model = KnapsackPNA(deg=deg)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    print("\nForward pass...")
    model.eval()
    with torch.no_grad():
        probs = model(graph)

    print(f"Output probabilities: {probs}")
    print(f"True solution: {graph.y}")
    print(f"Predicted solution (threshold=0.5): {(probs >= 0.5).float()}")
