"""
Knapsack GNN - Complete Demo Script
Learning to Optimize approach using Graph Neural Networks for the Knapsack Problem

This script demonstrates the complete pipeline:
1. Generate Knapsack instances
2. Build graph representations
3. Train PNA-based GNN
4. Evaluate on test instances
5. Compare with optimal solutions

Author: Learning to Optimize Implementation
Based on: "Learning to Solve Combinatorial Optimization with GNNs" approach
"""

import argparse

import numpy as np
import torch

# Import all modules
from knapsack_gnn.data.generator import KnapsackGenerator, KnapsackSolver
from knapsack_gnn.data.graph_builder import KnapsackGraphBuilder, KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import KnapsackSampler
from knapsack_gnn.models.pna import create_model
from knapsack_gnn.training.loop import KnapsackTrainer


def demo_quick_start():
    """Quick demonstration with minimal dataset"""
    print("\n" + "=" * 80)
    print(" " * 20 + "KNAPSACK GNN - QUICK START DEMO")
    print("=" * 80)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # ========== STEP 1: Generate Data ==========
    print("\n[STEP 1/5] Generating Knapsack instances...")
    print("-" * 80)

    generator = KnapsackGenerator(seed=42)
    instances = generator.generate_dataset(
        n_instances=50,
        n_items_range=(10, 20),
        weight_range=(1, 50),
        value_range=(1, 100),
        capacity_ratio=0.5,
    )

    print(f"Generated {len(instances)} instances")
    print(f"Example instance: {instances[0]}")

    # Solve instances
    print("\nSolving instances with exact solver...")
    instances = KnapsackSolver.solve_batch(instances, verbose=False)
    print("All instances solved!")

    # Split data
    train_instances = instances[:30]
    val_instances = instances[30:40]
    test_instances = instances[40:50]

    # ========== STEP 2: Build Graphs ==========
    print("\n[STEP 2/5] Building graph representations...")
    print("-" * 80)

    from knapsack_gnn.data.generator import KnapsackDataset

    train_dataset = KnapsackGraphDataset(KnapsackDataset(train_instances))
    val_dataset = KnapsackGraphDataset(KnapsackDataset(val_instances))
    test_dataset = KnapsackGraphDataset(KnapsackDataset(test_instances))

    print(f"Train: {len(train_dataset)} graphs")
    print(f"Val: {len(val_dataset)} graphs")
    print(f"Test: {len(test_dataset)} graphs")

    sample = train_dataset[0]
    print("\nSample graph:")
    print(f"  Nodes: {sample.x.shape[0]} ({sample.n_items} items + 1 constraint)")
    print(f"  Edges: {sample.edge_index.shape[1]}")
    print(f"  Features: {sample.x.shape}")

    # ========== STEP 3: Create Model ==========
    print("\n[STEP 3/5] Creating PNA-based GNN model...")
    print("-" * 80)

    model = create_model(
        dataset=train_dataset,
        hidden_dim=32,  # Smaller for demo
        num_layers=2,
        dropout=0.1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ========== STEP 4: Train Model ==========
    print("\n[STEP 4/5] Training model...")
    print("-" * 80)

    trainer = KnapsackTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        learning_rate=0.002,
        device=device,
        checkpoint_dir="checkpoints/demo",
    )

    print("Training for 10 epochs (quick demo)...")
    history = trainer.train(num_epochs=10, verbose=True)

    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")

    # ========== STEP 5: Evaluate ==========
    print("\n[STEP 5/5] Evaluating on test set...")
    print("-" * 80)

    sampler = KnapsackSampler(model, device=device)

    # Evaluate all test instances
    gaps = []
    feasible_count = 0

    for i, data in enumerate(test_dataset):
        result = sampler.solve(data, strategy="sampling", n_samples=50)

        if "optimality_gap" in result:
            gaps.append(result["optimality_gap"])

        if result["is_feasible"]:
            feasible_count += 1

        if i < 3:  # Show first 3 examples
            print(f"\nTest instance {i}:")
            print(f"  Optimal value: {result.get('optimal_value', 'N/A')}")
            print(f"  Predicted value: {result['value']}")
            print(f"  Gap: {result.get('optimality_gap', 'N/A'):.2f}%")
            print(f"  Feasible: {result['is_feasible']}")

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Test instances: {len(test_dataset)}")
    print(f"Feasibility rate: {feasible_count / len(test_dataset) * 100:.1f}%")
    if gaps:
        print(f"Mean optimality gap: {np.mean(gaps):.2f}%")
        print(f"Median optimality gap: {np.median(gaps):.2f}%")
        print(f"Best gap: {np.min(gaps):.2f}%")
        print(f"Worst gap: {np.max(gaps):.2f}%")

    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nTo run a full training pipeline, use:")
    print("  python train.py --train_size 1000 --num_epochs 50")
    print("\nTo evaluate a trained model:")
    print("  python evaluate.py --checkpoint_dir checkpoints/run_XXXXXX --visualize")
    print("=" * 80 + "\n")


def demo_single_instance():
    """Detailed demo with a single instance"""
    print("\n" + "=" * 80)
    print(" " * 20 + "SINGLE INSTANCE DETAILED DEMO")
    print("=" * 80)

    # Generate single instance
    print("\n[1] Generating Knapsack instance...")
    generator = KnapsackGenerator(seed=42)
    instance = generator.generate_instance(n_items=15)

    print("\nInstance properties:")
    print(f"  Items: {instance.n_items}")
    print(f"  Capacity: {instance.capacity}")
    print(f"  Weights: {instance.weights[:5]}... (showing first 5)")
    print(f"  Values: {instance.values[:5]}... (showing first 5)")

    # Solve optimally
    print("\n[2] Solving with exact algorithm...")
    instance = KnapsackSolver.solve(instance)

    print("\nOptimal solution found!")
    print(f"  Selected items: {np.where(instance.solution == 1)[0]}")
    print(f"  Total weight: {np.sum(instance.solution * instance.weights)}/{instance.capacity}")
    print(f"  Total value: {instance.optimal_value}")

    # Build graph
    print("\n[3] Building graph representation...")
    builder = KnapsackGraphBuilder()
    graph = builder.build_graph(instance)

    print("\nGraph structure:")
    print(f"  Nodes: {graph.x.shape[0]} (items + constraint)")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Node features shape: {graph.x.shape}")
    print("  Edge connectivity: bipartite (items <-> constraint)")

    print("\n" + "=" * 80)
    print("This demonstrates the core L2O approach:")
    print("  1. Problem -> Graph (tripartite structure)")
    print("  2. GNN learns from optimal solutions")
    print("  3. Inference via probability sampling")
    print("=" * 80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Knapsack GNN Demo - Learning to Optimize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick start demo (recommended first)
  python Knapsack_GNN.py --demo quick

  # Run single instance demo
  python Knapsack_GNN.py --demo single

  # Full training pipeline
  python train.py --train_size 1000 --num_epochs 50

  # Evaluation
  python evaluate.py --checkpoint_dir checkpoints/run_XXXXXX --visualize
        """,
    )

    parser.add_argument(
        "--demo",
        type=str,
        default="quick",
        choices=["quick", "single", "both"],
        help="Demo mode (default: quick)",
    )

    args = parser.parse_args()

    print("\n")
    print("=" * 80)
    print(" " * 15 + "GRAPH NEURAL NETWORKS FOR KNAPSACK PROBLEM")
    print(" " * 20 + "Learning to Optimize Approach")
    print("=" * 80)
    print("\nImplementation of PNA-based GNN for solving 0-1 Knapsack Problem")
    print("Using supervised learning on optimal solutions from exact solver")
    print("=" * 80)

    if args.demo in ["quick", "both"]:
        demo_quick_start()

    if args.demo in ["single", "both"]:
        demo_single_instance()


if __name__ == "__main__":
    main()
