"""
Graph Builder for Knapsack Problem
Converts Knapsack instances into tripartite graphs for GNN processing
"""

import os
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data, Dataset

from knapsack_gnn.utils.logging import get_logger
from knapsack_gnn.utils.feature_flags import parse_graph_feature_spec

from .generator import KnapsackDataset, KnapsackInstance

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)


def _rank_normalize(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return ranks scaled to [0,1]."""
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(arr.size, dtype=np.float32)
    denom = max(arr.size - 1, 1)
    result = cast(NDArray[np.float32], (ranks / denom).astype(np.float32, copy=False))
    return result


def _zscore(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return z-score normalized copy with safe std."""
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    mean = np.mean(arr)
    std = np.std(arr)
    denom = std if std > 1e-6 else 1.0
    result = cast(NDArray[np.float32], ((arr - mean) / denom).astype(np.float32, copy=False))
    return result


class KnapsackGraphBuilder:
    """Converts Knapsack instances to PyTorch Geometric graph format"""

    def __init__(
        self,
        normalize_features: bool = True,
        enable_density: bool = False,
        enable_quadratic_ratio: bool = False,
        enable_bucket_ranks: bool = False,
        buckets: int = 4,
    ) -> None:
        """
        Args:
            normalize_features: Whether to normalize node features
            enable_density: Add capacity density feature (capacity / sum(weights))
            enable_quadratic_ratio: Add value^2 / weight feature
            enable_bucket_ranks: Add bucketized ranks (one-hot) based on item values
            buckets: Number of buckets for bucketized ranks
        """
        self.normalize_features = normalize_features
        self.enable_density = enable_density
        self.enable_quadratic_ratio = enable_quadratic_ratio
        self.enable_bucket_ranks = enable_bucket_ranks
        self.bucket_count = max(buckets, 1)

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
        weights = instance.weights.astype(np.float32)
        values = instance.values.astype(np.float32)

        ratio = values / np.maximum(weights, 1e-6)
        value_rank = _rank_normalize(values)
        weight_rank = _rank_normalize(weights)
        ratio_rank = _rank_normalize(ratio)
        value_z = _zscore(values)
        weight_z = _zscore(weights)

        features: list[np.ndarray] = [
            weights.copy(),
            values.copy(),
            ratio,
            value_rank,
            weight_rank,
            ratio_rank,
            value_z,
            weight_z,
        ]

        if self.enable_quadratic_ratio:
            quad = (values**2) / np.maximum(weights, 1e-6)
            features.append(quad.astype(np.float32))

        if self.enable_bucket_ranks:
            quantiles = np.linspace(0, 1, self.bucket_count + 1)
            bucket_edges = np.quantile(values, quantiles)
            bucket_indices = np.digitize(values, bucket_edges[1:-1], right=True)
            bucket_one_hot = np.eye(self.bucket_count, dtype=np.float32)[bucket_indices]
            features.append(bucket_one_hot.astype(np.float32))

        item_features = np.column_stack(features).astype(np.float32)

        # Constraint node features have matching dimensionality
        constraint_features = np.zeros((1, item_features.shape[1]), dtype=np.float32)
        constraint_features[0, 0] = instance.capacity

        # Normalize if requested
        if self.normalize_features:
            # Normalize item features by max values
            max_weight = float(np.max(weights)) if weights.size else 0.0
            max_value = float(np.max(values)) if values.size else 0.0
            item_features[:, 0] /= max_weight if max_weight > 0 else 1.0
            item_features[:, 1] /= max_value if max_value > 0 else 1.0

            max_ratio = float(np.max(ratio)) if ratio.size else 0.0
            if max_ratio > 0:
                item_features[:, 2] /= max_ratio

            # Normalize constraint by total weight
            total_weight = float(np.sum(weights)) if weights.size else 0.0
            norm = total_weight if total_weight > 0 else 1.0
            constraint_features[:, 0] /= norm

        if self.enable_density:
            total_weight = float(np.sum(weights)) if weights.size else 1.0
            density = instance.capacity / max(total_weight, 1e-6)
            density_feature = np.full((item_features.shape[0], 1), density, dtype=np.float32)
            item_features = np.hstack([item_features, density_feature])
            constraint_density = np.array([[density]], dtype=np.float32)
            constraint_features = np.hstack([constraint_features, constraint_density])

        # Concatenate all node features
        # Node indices: [0, n_items-1] are item nodes, n_items is constraint node
        x_np = np.vstack([item_features, constraint_features])
        if not np.isfinite(x_np).all():
            x_np = np.nan_to_num(x_np, copy=False)
        node_features = torch.tensor(x_np, dtype=torch.float32)

        # === Edge Construction ===
        # Create bipartite edges: each item connects to constraint node
        constraint_node_idx = n_items

        # Edge list: (item_i, constraint) and (constraint, item_i)
        edge_index_list = []
        for i in range(n_items):
            # Bidirectional edges
            edge_index_list.append([i, constraint_node_idx])
            edge_index_list.append([constraint_node_idx, i])

        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

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
            x=node_features,
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

    def __init__(
        self,
        knapsack_dataset: KnapsackDataset,
        normalize_features: bool = True,
        graph_features: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            knapsack_dataset: KnapsackDataset containing instances
            normalize_features: Whether to normalize node features
            graph_features: Optional kwargs forwarded to ``KnapsackGraphBuilder`` to enable
                extra feature columns (density, quadratic ratio, bucket ranks, etc.)
        """
        super().__init__()
        self.knapsack_dataset = knapsack_dataset
        builder_kwargs = self._resolve_feature_flags(graph_features)
        self.graph_builder = KnapsackGraphBuilder(
            normalize_features=normalize_features,
            **builder_kwargs,
        )
        self.graph_feature_flags = builder_kwargs

        # Pre-build all graphs for efficiency
        logger.info("Building %d graphs from dataset", len(knapsack_dataset))
        self.graphs = self.graph_builder.build_batch(knapsack_dataset.instances)
        logger.info("Graphs built successfully")

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]

    @staticmethod
    def _resolve_feature_flags(graph_features: dict[str, Any] | None) -> dict[str, Any]:
        if graph_features is not None:
            return dict(graph_features)

        spec = os.getenv("KNAPSACK_GNN_GRAPH_FEATURES")
        if not spec:
            return {}

        bucket_env = os.getenv("KNAPSACK_GNN_GRAPH_FEATURE_BUCKETS")
        bucket_count: int | None = None
        if bucket_env:
            try:
                bucket_count = int(bucket_env)
            except ValueError:
                logger.warning(
                    "Invalid KNAPSACK_GNN_GRAPH_FEATURE_BUCKETS=%s; falling back to default.",
                    bucket_env,
                )
        try:
            return parse_graph_feature_spec(spec, bucket_count)
        except ValueError as exc:
            logger.error("Failed to parse KNAPSACK_GNN_GRAPH_FEATURES=%s: %s", spec, exc)
            raise


def visualize_graph(data: Data, title: str = "Knapsack Graph") -> "Figure":
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
    return plt.gcf()


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    # Example usage
    from .knapsack_generator import KnapsackGenerator, KnapsackSolver

    logger.info("Creating sample Knapsack instance...")
    generator = KnapsackGenerator(seed=42)
    instance = generator.generate_instance(n_items=10)

    logger.info("Solving instance...")
    instance = KnapsackSolver.solve(instance)

    logger.info("Instance: %s", instance)
    logger.info("Optimal value: %s", instance.optimal_value)
    logger.info("Solution: %s", instance.solution)

    logger.info("Building graph...")
    builder = KnapsackGraphBuilder(normalize_features=True)
    graph = builder.build_graph(instance)

    logger.info("Graph properties:")
    logger.info("  Number of nodes: %s", graph.x.shape[0])
    logger.info("  Number of edges: %s", graph.edge_index.shape[1])
    logger.info("  Node features shape: %s", graph.x.shape)
    logger.info("  Node types: %s", graph.node_types)
    logger.info("  Labels (solution): %s", graph.y)

    # Visualize
    logger.info("Visualizing graph...")
    fig = visualize_graph(graph, title=f"Knapsack Graph (Optimal Value: {instance.optimal_value})")
    fig.savefig("knapsack_graph_example.png", dpi=150, bbox_inches="tight")
    logger.info("Graph saved to knapsack_graph_example.png")


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
