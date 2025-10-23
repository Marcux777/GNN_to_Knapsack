"""
Training Pipeline for Knapsack GNN
Implements supervised learning with Binary Cross-Entropy loss
"""

import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm


class KnapsackTrainer:
    """
    Trainer class for Knapsack GNN models
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        batch_size: int = 32,
        learning_rate: float = 0.002,
        weight_decay: float = 1e-6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Args:
            model: KnapsackPNA model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate (default: 0.002 as per paper)
            weight_decay: L2 regularization (default: 1e-6 as per paper)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Data loaders
        self.train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )
        self.val_loader = GeometricDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Loss and optimizer
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy for binary classification
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        self.best_val_loss = float("inf")
        self.epochs_trained = 0

    def train_epoch(self) -> tuple[float, float]:
        """
        Train for one epoch

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            probs = self.model(batch)

            # Compute loss (only on item nodes)
            loss = self.criterion(probs, batch.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * len(batch.y)
            predictions = (probs >= 0.5).float()
            total_correct += (predictions == batch.y).sum().item()
            total_samples += len(batch.y)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def validate(self) -> tuple[float, float]:
        """
        Validate model

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                batch = batch.to(self.device)

                # Forward pass
                probs = self.model(batch)

                # Compute loss
                loss = self.criterion(probs, batch.y)

                # Statistics
                total_loss += loss.item() * len(batch.y)
                predictions = (probs >= 0.5).float()
                total_correct += (predictions == batch.y).sum().item()
                total_samples += len(batch.y)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def train(self, num_epochs: int, verbose: bool = True) -> dict:
        """
        Train model for multiple epochs

        Args:
            num_epochs: Number of epochs to train
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.device}...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            self.epochs_trained += 1

            # Print progress
            if verbose:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"  LR: {current_lr:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                if verbose:
                    print(f"  → Best model saved (val_loss: {val_loss:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        print("\nTraining completed!")
        self.save_checkpoint("final_model.pt")
        self.save_history()

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "epochs_trained": self.epochs_trained,
            "best_val_loss": self.best_val_loss,
        }
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]
        self.epochs_trained = checkpoint["epochs_trained"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Checkpoint loaded from {filepath}")
        print(f"Epochs trained: {self.epochs_trained}")
        print(f"Best val loss: {self.best_val_loss:.4f}")

    def save_history(self):
        """Save training history to JSON"""
        filepath = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {filepath}")

    def plot_training_curves(self, save_path: str | None = None):
        """
        Plot training curves

        Args:
            save_path: Path to save plot (if None, only display)
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        axes[0, 0].plot(self.history["train_loss"], label="Train")
        axes[0, 0].plot(self.history["val_loss"], label="Validation")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.history["train_accuracy"], label="Train")
        axes[0, 1].plot(self.history["val_accuracy"], label="Validation")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Accuracy Curves")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate
        axes[1, 0].plot(self.history["learning_rate"])
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_title("Learning Rate Schedule")
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale("log")

        # Val loss zoom
        axes[1, 1].plot(self.history["val_loss"])
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Validation Loss")
        axes[1, 1].set_title("Validation Loss (Zoom)")
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()

        return fig


def train_model(
    model,
    train_dataset,
    val_dataset,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.002,
    checkpoint_dir: str = "checkpoints",
    **kwargs,
) -> tuple[nn.Module, dict]:
    """
    Convenience function to train a model

    Args:
        model: KnapsackPNA model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_dir: Checkpoint directory
        **kwargs: Additional trainer arguments

    Returns:
        Tuple of (trained_model, history)
    """
    trainer = KnapsackTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )

    history = trainer.train(num_epochs=num_epochs)
    trainer.plot_training_curves(save_path=f"{checkpoint_dir}/training_curves.png")

    return model, history


if __name__ == "__main__":
    # Example usage
    print("This module provides training utilities.")
    print("Use train.py script to train models.")
