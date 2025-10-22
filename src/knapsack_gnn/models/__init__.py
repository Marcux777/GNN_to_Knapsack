"""GNN model architectures for knapsack optimization."""

from knapsack_gnn.models.pna import KnapsackPNA, KnapsackPNAWithBatch
from knapsack_gnn.models.gcn import KnapsackGCN
from knapsack_gnn.models.gat import KnapsackGAT

# Aliases for backward compatibility
PNAKnapsackGNN = KnapsackPNA
GCNKnapsackGNN = KnapsackGCN
GATKnapsackGNN = KnapsackGAT

__all__ = [
    "KnapsackPNA",
    "KnapsackPNAWithBatch",
    "KnapsackGCN",
    "KnapsackGAT",
    # Aliases for backward compatibility
    "PNAKnapsackGNN",
    "GCNKnapsackGNN",
    "GATKnapsackGNN",
]
