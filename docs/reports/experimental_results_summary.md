# Experimental Results (TL;DR)

- **Run:** `run_20251020_104533` (PNA, 50 epochs, CPU inference)
- **Sampler:** Vectorized 32→64→128 schedule ⇒ **0.068%** mean gap, **14.5 ms** mean latency, **100%** feasibility.
- **Warm-start ILP:** Adds ~1.9 ms to clamp tail gaps ⇒ **0.18%** mean gap, still 100% feasible.
- **Baseline gap:** Greedy ≈0.49%, Random ≈11%; the GNN stays ≈7× more accurate while keeping latency <25 ms.
- **Ablations:** Removing values or weights destroys performance (gap >90%); confirms both features are essential.
- **Architecture check:** PNA > GAT > GCN for variability; all models keep feasibility, but PNA gives the tightest gap distribution.

For full tables/plots, see `results/ablations/**` and `results/bc_ranker_full/**`.
