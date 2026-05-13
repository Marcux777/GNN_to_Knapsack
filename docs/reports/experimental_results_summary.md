# Experimental Results (TL;DR)

- **Run:** `run_20251020_104533` (PNA, 50 epochs, CPU inference)
- **Sampler:** Vectorized 32→64→128 schedule ⇒ **0.068%** mean gap, **14.5 ms** mean latency, **100%** feasibility.
- **Warm-start ILP:** Adds ~1.9 ms to clamp tail gaps ⇒ **0.18%** mean gap, still 100% feasible.
- **Baseline gap:** Greedy ≈0.49%, Random ≈11%; the GNN stays ≈7× more accurate while keeping latency <25 ms.
- **Ablations:** Removing values or weights destroys performance (gap >90%); confirms both features are essential.
- **Architecture check:** PNA > GAT > GCN for variability; all models keep feasibility, but PNA gives the tightest gap distribution.

For full tables/plots, see `results/ablations/**` and `results/bc_ranker_full/**`.

## Feature Toggle Ablation (Quick Run – 2025-11-08)
- Dataset: `train/val/test = 400/80/80` instances sampled from 8–25 items, seed 1337, trained for 6 epochs on CPU via `experiments/pipelines/train_pipeline.py`.
- Decoder: sampling strategy (48 draws, schedule 16→32, temperature 0.9) evaluated with `load_model_from_checkpoint` + `evaluate_model`.
- Artefacts: checkpoints were temporarily stored under `checkpoints/feature_ablation/` (now removed during cleanup); metrics below capture the final run that fed the docs.

| Spec | Extra signals | Mean gap (%) | Median gap (%) | Max gap (%) | Feasibility | Mean latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | base features only | 2.40 | 0.00 | 55.92 | 100% | 1.52 |
| density | + capacity density | 3.07 | 0.00 | 39.80 | 100% | 1.59 |
| density+quadratic | + density & `value²/weight` | **1.83** | 0.00 | 36.85 | 100% | 1.58 |
| all | density + quadratic + bucket ranks | 2.10 | 0.00 | 36.85 | 100% | 1.58 |

**Takeaway:** Injecting density + quadratic ratio tightens the mean optimality gap by ~0.6 pp against the baseline without affecting feasibility or latency, while adding bucket ranks gives diminishing returns at this scale.
