"""
Graph Builder for Knapsack Problem
Converts Knapsack instances into tripartite graphs for GNN processing
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from .generator import KnapsackDataset, KnapsackInstance

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class KnapsackGraphBuilder:
    """Converts Knapsack instances to PyTorch Geometric graph format"""

    def __init__(self, normalize_features: bool = True) -> None:
        """
        Args:
            normalize_features: Whether to normalize node features
        """
        self.normalize_features = normalize_features

    def build_graph(self, instance: KnapsackInstance) -> Data:
        """
        Convert a Knapsack instance to a tripartite graph

        Graph structure:
        - Item nodes: n_items nodes with features [weight, value]
        - Constraint node: 1 node with feature [capacity, 0] to match dimensionality
        - Edges: Each item connects to the constraint node (bipartite structure)

        Args:
            instance: KnapsackInstance to convert

        Returns:
            PyTorch Geometric Data object
        """
        n_items = instance.n_items

        # === Node Features ===
        # Item nodes features: [weight, value]
        item_features = np.stack([instance.weights, instance.values], axis=1).astype(np.float32)

        # Constraint node features: [capacity, 0] to match item feature dimension
        constraint_features = np.array([[instance.capacity, 0.0]], dtype=np.float32)

        # Normalize if requested
        if self.normalize_features:
            # Normalize item features by max values
            max_weight = np.max(instance.weights)
            max_value = np.max(instance.values)
            item_features[:, 0] /= max_weight if max_weight > 0 else 1.0
            item_features[:, 1] /= max_value if max_value > 0 else 1.0

            # Normalize constraint by total weight
            total_weight = np.sum(instance.weights)
            constraint_features /= total_weight if total_weight > 0 else 1.0

        # Concatenate all node features
        # Node indices: [0, n_items-1] are item nodes, n_items is constraint node
        x = np.vstack([item_features, constraint_features])
        x = torch.tensor(x, dtype=torch.float32)

        # === Edge Construction ===
        # Create bipartite edges: each item connects to constraint node
        constraint_node_idx = n_items

        # Edge list: (item_i, constraint) and (constraint, item_i)
        edge_index_list = []
        for i in range(n_items):
            # Bidirectional edges
            edge_index_list.append([i, constraint_node_idx])
            edge_index_list.append([constraint_node_idx, i])

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        # === Node Type Indicators ===
        # 0 = item node, 1 = constraint node
        node_types = torch.zeros(n_items + 1, dtype=torch.long)
        node_types[constraint_node_idx] = 1

        # === Labels ===
        # Binary vector indicating which items are in optimal solution
        # Only item nodes have labels (constraint node doesn't need label)
        if instance.solution is not None:
            y = torch.tensor(instance.solution, dtype=torch.float32)
        else:
            y = torch.zeros(n_items, dtype=torch.float32)

        # === Additional attributes ===
        # Store original instance data for evaluation
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            node_types=node_types,
            n_items=n_items,
            capacity=instance.capacity,
            item_weights=torch.tensor(instance.weights, dtype=torch.float32),
            item_values=torch.tensor(instance.values, dtype=torch.float32),
            optimal_value=instance.optimal_value if instance.optimal_value is not None else 0,
            solve_time=float(instance.solve_time) if instance.solve_time is not None else 0.0,
        )

        return data

    def build_batch(self, instances: list[KnapsackInstance]) -> list[Data]:
        """
        Convert multiple instances to graphs

        Args:
            instances: List of KnapsackInstance objects

        Returns:
            List of PyTorch Geometric Data objects
        """
        return [self.build_graph(inst) for inst in instances]


class KnapsackGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset wrapper for Knapsack graphs
    """

    def __init__(self, knapsack_dataset: KnapsackDataset, normalize_features: bool = True) -> None:
        """
        Args:
            knapsack_dataset: KnapsackDataset containing instances
            normalize_features: Whether to normalize node features
        """
        super().__init__()
        self.knapsack_dataset = knapsack_dataset
        self.graph_builder = KnapsackGraphBuilder(normalize_features=normalize_features)

        # Pre-build all graphs for efficiency
        print(f"Building {len(knapsack_dataset)} graphs...")
        self.graphs = self.graph_builder.build_batch(knapsack_dataset.instances)
        print("Graphs built successfully!")

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


def visualize_graph(data: Data, title: str = "Knapsack Graph") -> "plt":
    """
    Visualize a Knapsack graph using networkx and matplotlib

    Args:
        data: PyTorch Geometric Data object
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # Create networkx graph
    G = nx.Graph()

    n_items = data.n_items
    constraint_idx = n_items

    # Add nodes
    for i in range(n_items):
        G.add_node(i, node_type="item")
    G.add_node(constraint_idx, node_type="constraint")

    # Add edges
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst)

    # Layout
    pos = {}
    # Item nodes in a circle
    angle_step = 2 * np.pi / n_items
    for i in range(n_items):
        angle = i * angle_step
        pos[i] = (np.cos(angle), np.sin(angle))
    # Constraint node at center
    pos[constraint_idx] = (0, 0)

    # Colors based on solution
    node_colors = []
    for i in range(n_items):
        if data.y[i] == 1:
            node_colors.append("lightgreen")  # Selected items
        else:
            node_colors.append("lightblue")  # Not selected
    node_colors.append("red")  # Constraint node

    # Draw
    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=True,
        node_size=500,
        font_size=10,
        font_weight="bold",
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    # Example usage
    from .knapsack_generator import KnapsackGenerator, KnapsackSolver

    print("Creating sample Knapsack instance...")
    generator = KnapsackGenerator(seed=42)
    instance = generator.generate_instance(n_items=10)

    print("Solving instance...")
    instance = KnapsackSolver.solve(instance)

    print(f"\nInstance: {instance}")
    print(f"Optimal value: {instance.optimal_value}")
    print(f"Solution: {instance.solution}")

    print("\nBuilding graph...")
    builder = KnapsackGraphBuilder(normalize_features=True)
    graph = builder.build_graph(instance)

    print("\nGraph properties:")
    print(f"  Number of nodes: {graph.x.shape[0]}")
    print(f"  Number of edges: {graph.edge_index.shape[1]}")
    print(f"  Node features shape: {graph.x.shape}")
    print(f"  Node types: {graph.node_types}")
    print(f"  Labels (solution): {graph.y}")

    # Visualize
    print("\nVisualizing graph...")
    plt = visualize_graph(graph, title=f"Knapsack Graph (Optimal Value: {instance.optimal_value})")
    plt.savefig("knapsack_graph_example.png", dpi=150, bbox_inches="tight")
    print("Graph saved to knapsack_graph_example.png")


# Convenience wrapper function for backward compatibility
def build_bipartite_graph(instance: Any, normalize_features: bool = True, *args: Any) -> Data:
    """
    Build a bipartite graph from a knapsack instance.

    Args:
        instance: KnapsackInstance object or dict with values, weights, capacity keys
        normalize_features: Whether to normalize node features
        *args: If instance is dict, can pass weights and capacity as separate args (for backwards compat)

    Returns:
        PyTorch Geometric Data object
    """
    # Handle backwards compatibility: old API was build_bipartite_graph(values, weights, capacity)
    import numpy as np

    if isinstance(normalize_features, np.ndarray) and len(args) >= 1:
        # Old API: build_bipartite_graph(values, weights, capacity)
        values = instance
        weights = normalize_features
        capacity = args[0]
        instance = KnapsackInstance(weights=weights, values=values, capacity=int(capacity))
        normalize_features = False  # Don't normalize for backwards compatibility

    builder = KnapsackGraphBuilder(normalize_features=normalize_features)
    return builder.build_graph(instance)
