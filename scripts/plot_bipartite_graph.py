"""Render bipartite (item-capacity) graphs for selected instances."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.path import Path as MplPath

from knapsack_gnn.data.generator import KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphBuilder, visualize_graph
from knapsack_gnn.utils.feature_flags import parse_graph_feature_spec


def _path_deepcopy(self, memo):
    vertices = self.vertices.copy()
    codes = None if self.codes is None else self.codes.copy()
    return MplPath(vertices, codes)


MplPath.__deepcopy__ = _path_deepcopy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot bipartite graphs used for knapsack instances"
    )
    parser.add_argument(
        "--dataset", type=str, default="data/datasets/test.pkl", help="Path to dataset .pkl"
    )
    parser.add_argument(
        "--indices", type=int, nargs="+", default=[0], help="Instance indices to plot"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/bipartite_graphs", help="Directory to save PNGs"
    )
    parser.add_argument("--normalize", action="store_true", help="Use normalized features")
    parser.add_argument(
        "--graph_features",
        type=str,
        default="none",
        help="Optional graph features (density,quadratic,bucket,all,none).",
    )
    parser.add_argument(
        "--graph_feature_buckets",
        type=int,
        default=4,
        help="Bucket count for bucketized ranks (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = KnapsackDataset.load(args.dataset)
    feature_kwargs = parse_graph_feature_spec(args.graph_features, args.graph_feature_buckets)
    builder = KnapsackGraphBuilder(normalize_features=args.normalize, **feature_kwargs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in args.indices:
        instance = dataset.instances[idx]
        graph = builder.build_graph(instance)
        fig = visualize_graph(graph, title=f"Knapsack Graph (idx={idx}, n={instance.n_items})")
        fig.savefig(out_dir / f"bipartite_{idx}.png", dpi=150, bbox_inches="tight")
        print(f"Saved {out_dir / f'bipartite_{idx}.png'}")


if __name__ == "__main__":
    main()
