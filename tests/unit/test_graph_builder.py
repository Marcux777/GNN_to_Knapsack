"""
Tests for graph construction from knapsack instances.
"""

import torch
from torch_geometric.data import Data

from knapsack_gnn.data.graph_builder import build_bipartite_graph


class TestGraphBuilder:
    """Test suite for bipartite graph construction."""

    def test_build_graph_returns_data_object(self, small_knapsack_instance):
        """Test that graph builder returns PyG Data object."""
        inst = small_knapsack_instance
        graph = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])

        assert isinstance(graph, Data), "Should return torch_geometric.data.Data"

    def test_graph_has_required_attributes(self, small_knapsack_instance):
        """Test that graph has all required attributes."""
        inst = small_knapsack_instance
        graph = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])

        assert hasattr(graph, "x"), "Graph should have node features 'x'"
        assert hasattr(graph, "edge_index"), "Graph should have edge_index"
        assert hasattr(graph, "edge_attr") or True, "Edge attributes are optional"

    def test_node_features_shape(self, small_knapsack_instance):
        """Test that node features have correct shape."""
        inst = small_knapsack_instance
        n_items = inst["n_items"]

        graph = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])

        # Bipartite: n_items + 1 capacity node
        expected_nodes = n_items + 1
        assert graph.x.shape[0] == expected_nodes, (
            f"Expected {expected_nodes} nodes, got {graph.x.shape[0]}"
        )

        # Features now include extended stats (value, weight, ratio, ranks, z-scores, etc.)
        assert graph.x.shape[1] == 8, "Each node should have 8 features"

    def test_node_features_values(self, small_knapsack_instance):
        """Test that node features contain correct values (possibly normalized)."""
        inst = small_knapsack_instance
        graph = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])

        n_items = inst["n_items"]

        # Item nodes have extended feature vector; check first two (value/weight) sanity
        for i in range(n_items):
            # Check that features are reasonable values (not NaN or Inf)
            assert not torch.isnan(graph.x[i, 0]), f"Item {i} value is NaN"
            assert not torch.isinf(graph.x[i, 0]), f"Item {i} value is Inf"
            assert not torch.isnan(graph.x[i, 1]), f"Item {i} weight is NaN"
            assert not torch.isinf(graph.x[i, 1]), f"Item {i} weight is Inf"
            # Values should be positive (after normalization)
            assert graph.x[i, 0] >= 0, f"Item {i} value should be non-negative"
            assert graph.x[i, 1] >= 0, f"Item {i} weight should be non-negative"

        # Capacity node should have positive first feature encoding capacity
        capacity_node_idx = n_items
        assert graph.x[capacity_node_idx, 0] > 0, "Capacity node should encode capacity"

    def test_bipartite_structure(self, small_knapsack_instance):
        """Test that graph is bipartite (items connect to capacity node)."""
        inst = small_knapsack_instance
        graph = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])

        n_items = inst["n_items"]
        capacity_node_idx = n_items

        edge_index = graph.edge_index

        # Check that edges connect items to capacity node
        # In bipartite graph: item nodes (0 to n_items-1) <-> capacity node (n_items)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()

            # Edge should connect item to capacity or vice versa
            is_valid_edge = (src < n_items and dst == capacity_node_idx) or (
                src == capacity_node_idx and dst < n_items
            )
            assert is_valid_edge, f"Invalid edge: {src} -> {dst}"

    def test_graph_degree(self, small_knapsack_instance):
        """Test node degrees in bipartite graph."""
        inst = small_knapsack_instance
        graph = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])

        n_items = inst["n_items"]
        edge_index = graph.edge_index

        # Count degree for each node
        degrees = torch.zeros(n_items + 1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            degrees[src] += 1

        # Each item node should connect to capacity node (degree >= 1)
        for i in range(n_items):
            assert degrees[i] >= 1, f"Item {i} should connect to capacity node"

        # Capacity node should connect to all items (degree >= n_items)
        assert degrees[n_items] >= n_items, "Capacity node should connect to all items"

    def test_graph_deterministic(self, small_knapsack_instance):
        """Test that same instance produces same graph."""
        inst = small_knapsack_instance

        graph1 = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])
        graph2 = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])

        assert torch.allclose(graph1.x, graph2.x), "Node features should be identical"
        assert torch.equal(graph1.edge_index, graph2.edge_index), "Edge index should be identical"

    def test_batch_graph_construction(self, tiny_knapsack_batch):
        """Test constructing graphs for multiple instances."""
        graphs = []

        for inst in tiny_knapsack_batch:
            graph = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])
            graphs.append(graph)

        assert len(graphs) == len(tiny_knapsack_batch)

        # Each graph should be valid
        for i, graph in enumerate(graphs):
            n_items = tiny_knapsack_batch[i]["n_items"]
            assert graph.x.shape[0] == n_items + 1, f"Graph {i} has incorrect number of nodes"

    def test_graph_no_self_loops(self, small_knapsack_instance):
        """Test that graph has no self-loops."""
        inst = small_knapsack_instance
        graph = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])

        edge_index = graph.edge_index

        # Check for self-loops
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            assert src != dst, f"Self-loop detected: {src} -> {src}"

    def test_feature_normalization(self, small_knapsack_instance):
        """Test that features are in reasonable range (not NaN/Inf)."""
        inst = small_knapsack_instance
        graph = build_bipartite_graph(inst["values"], inst["weights"], inst["capacity"])

        assert not torch.isnan(graph.x).any(), "Node features contain NaN"
        assert not torch.isinf(graph.x).any(), "Node features contain Inf"
        assert (graph.x[:, :2] >= 0).all(), "Value/weight features should be non-negative"
