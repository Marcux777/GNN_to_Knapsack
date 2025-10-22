"""
Training Script for Knapsack GNN
Complete pipeline for training PNA-based GNN on Knapsack problem
"""

import argparse
import os
import torch
import numpy as np
from datetime import datetime

# Import project modules
from knapsack_gnn.data.generator import create_datasets, KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.models.pna import create_model
from knapsack_gnn.training.loop import train_model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Knapsack GNN')

    # Data parameters
    parser.add_argument('--train_size', type=int, default=1000,
                       help='Number of training instances (default: 1000)')
    parser.add_argument('--val_size', type=int, default=200,
                       help='Number of validation instances (default: 200)')
    parser.add_argument('--test_size', type=int, default=200,
                       help='Number of test instances (default: 200)')
    parser.add_argument('--n_items_min', type=int, default=10,
                       help='Minimum number of items (default: 10)')
    parser.add_argument('--n_items_max', type=int, default=50,
                       help='Maximum number of items (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--data_dir', type=str, default='data/datasets',
                       help='Directory to save/load datasets')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension (default: 64)')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of PNA layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='Learning rate (default: 0.002)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay (default: 1e-6)')

    # Other parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--generate_data', action='store_true',
                       help='Generate new datasets (otherwise load existing)')

    return parser.parse_args()

def main():
    """Main training pipeline"""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"run_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("="*70)
    print("KNAPSACK GNN TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # ===== STEP 1: Load or Generate Data =====
    print("="*70)
    print("STEP 1: Data Preparation")
    print("="*70)

    if args.generate_data or not os.path.exists(f"{args.data_dir}/train.pkl"):
        print("\nGenerating new datasets...")
        train_dataset, val_dataset, test_dataset = create_datasets(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            n_items_range=(args.n_items_min, args.n_items_max),
            seed=args.seed,
            output_dir=args.data_dir
        )
    else:
        print("\nLoading existing datasets...")
        train_dataset = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
        val_dataset = KnapsackDataset.load(f"{args.data_dir}/val.pkl")
        test_dataset = KnapsackDataset.load(f"{args.data_dir}/test.pkl")

        print("\nDataset statistics:")
        print("Train:", train_dataset.get_statistics())
        print("Val:", val_dataset.get_statistics())
        print("Test:", test_dataset.get_statistics())

    # ===== STEP 2: Build Graphs =====
    print("\n" + "="*70)
    print("STEP 2: Building Graphs")
    print("="*70)

    train_graph_dataset = KnapsackGraphDataset(train_dataset, normalize_features=True)
    val_graph_dataset = KnapsackGraphDataset(val_dataset, normalize_features=True)
    test_graph_dataset = KnapsackGraphDataset(test_dataset, normalize_features=True)

    print(f"\nGraph datasets created:")
    print(f"  Train: {len(train_graph_dataset)} graphs")
    print(f"  Val: {len(val_graph_dataset)} graphs")
    print(f"  Test: {len(test_graph_dataset)} graphs")

    # Sample graph info
    sample_graph = train_graph_dataset[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {sample_graph.x.shape[0]}")
    print(f"  Edges: {sample_graph.edge_index.shape[1]}")
    print(f"  Node features: {sample_graph.x.shape}")

    # ===== STEP 3: Create Model =====
    print("\n" + "="*70)
    print("STEP 3: Creating Model")
    print("="*70)

    model = create_model(
        dataset=train_graph_dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    print(f"\nModel architecture:")
    print(model)

    # ===== STEP 4: Train Model =====
    print("\n" + "="*70)
    print("STEP 4: Training")
    print("="*70)

    model, history = train_model(
        model=model,
        train_dataset=train_graph_dataset,
        val_dataset=val_graph_dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=checkpoint_dir,
        device=args.device
    )

    # ===== STEP 5: Save Final Results =====
    print("\n" + "="*70)
    print("STEP 5: Saving Results")
    print("="*70)

    # Save configuration
    config_path = os.path.join(checkpoint_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"\nConfiguration saved to {config_path}")

    # Print final statistics
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final train accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")

    print("\nBest model saved as: best_model.pt")
    print("Final model saved as: final_model.pt")
    print("\nTo evaluate the model, run:")
    print(f"  python evaluate.py --checkpoint_dir {checkpoint_dir}")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
