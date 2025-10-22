"""Debug repair issue on instance 97"""

import torch
import numpy as np
from knapsack_gnn.data.generator import KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.models.pna import create_model
from knapsack_gnn.decoding.sampling import KnapsackSampler

# Load dataset
dataset = KnapsackDataset.load("data/datasets/test.pkl")
graph_dataset = KnapsackGraphDataset(dataset, normalize_features=True)

# Load model
model = create_model(graph_dataset, hidden_dim=64, num_layers=3, dropout=0.1)
state = torch.load("checkpoints/run_20251020_104533/best_model.pt", map_location="cpu")
model.load_state_dict(state["model_state_dict"])
model.eval()

# Test instance 97
data = graph_dataset[97]
instance = dataset.instances[97]

print("=" * 80)
print(f"Instance 97: {instance.n_items} items, capacity={instance.capacity}")
print(f"Optimal value: {instance.optimal_value}")
print(f"Optimal solution: {instance.solution.sum()} items selected")
print()

# Test sampling without repair
sampler = KnapsackSampler(model, "cpu")
result_sampling = sampler.solve(data, strategy="sampling", max_samples=128)
print(f"Sampling (no repair):")
print(f"  Value: {result_sampling['value']}")
print(f"  Gap: {result_sampling.get('optimality_gap', 0):.2f}%")
print(f"  Feasible: {result_sampling['is_feasible']}")
print(f"  Items selected: {result_sampling['solution'].sum()}")
print()

# Test sampling with repair
result_repair = sampler.solve(data, strategy="sampling_repair", max_samples=128)
print(f"Sampling + Repair:")
print(f"  Value: {result_repair['value']}")
print(f"  Gap: {result_repair.get('optimality_gap', 0):.2f}%")
print(f"  Feasible: {result_repair['is_feasible']}")
print(f"  Items selected: {result_repair['solution'].sum()}")
print(f"  Initial value: {result_repair.get('initial_sampling_value', 0)}")
print(f"  Repair improvement: {result_repair.get('repair_improvement', 0):.2f}")
print()

# Manual debug
weights = instance.weights
values = instance.values
capacity = instance.capacity

print(f"Instance stats:")
print(f"  Total weight: {weights.sum()}")
print(f"  Total value: {values.sum()}")
print(f"  Capacity ratio: {capacity / weights.sum():.2f}")
print(
    f"  Value/weight ratios: min={np.min(values / weights):.2f}, max={np.max(values / weights):.2f}"
)
print()

# Check if repair actually made things worse
initial_sol = result_repair["solution"]
print(f"Checking repair solution:")
sol_weight = np.sum(initial_sol * weights)
sol_value = np.sum(initial_sol * values)
print(f"  Weight: {sol_weight} / {capacity} (feasible: {sol_weight <= capacity})")
print(f"  Value: {sol_value}")
print(
    f"  Gap vs optimal: {(instance.optimal_value - sol_value) / instance.optimal_value * 100:.2f}%"
)
