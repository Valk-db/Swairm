"""
benchmark_real_scale.py
=======================
First real-shape test of the FCS Anchor aggregation pipeline. Everything so
far ran at 128x256 toy scale; this benchmarks aggregate_module at ACTUAL
Qwen2.5-1.5B module shapes to test the unverified claim: "streamed
aggregation is affordable on a cheap CPU-only Anchor."

Qwen2.5-1.5B (verified config constants):
  hidden 1536, 28 layers, GQA (2 KV heads, head_dim 128 -> KV width 256),
  MLP intermediate 8960. Per layer: q/o proj 1536x1536, k/v proj 256x1536,
  gate/up 8960x1536, down 1536x8960.

Measures per module type: dense reconstruction + trimmed mean + randomized
SVD wall-clock, and peak cohort memory. Extrapolates to a full 28-layer,
~196-module round.

Interpretation: your dev machine is likely FASTER than a $150 OptiPlex --
treat results as a lower bound, then apply a 2-4x safety factor for the
real Anchor. If a full round extrapolates to minutes, fine (rounds are
~30min apart). If it extrapolates to >15min, the streaming design needs
rethinking BEFORE main.py gets built.

Run: python benchmark_real_scale.py
"""

import time
import numpy as np
from aggregator import aggregate_module, DEFAULT_RANK_MAP

RNG = np.random.default_rng(42)
N_CLIENTS = 12
HET = 1.0

# module type -> (out_features, in_features, count per layer, rank-map key)
QWEN_MODULES = {
    "q_proj":    (1536, 1536, 1, "attn"),
    "k_proj":    (256, 1536, 1, "kv"),
    "v_proj":    (256, 1536, 1, "kv"),
    "o_proj":    (1536, 1536, 1, "attn"),
    "gate_proj": (8960, 1536, 1, "mlp"),
    "up_proj":   (8960, 1536, 1, "mlp"),
    "down_proj": (1536, 8960, 1, "mlp"),
}
N_LAYERS = 28


def make_cohort(out_dim, in_dim, rank):
    A_shared = RNG.standard_normal((rank, in_dim)) / np.sqrt(in_dim)
    B_shared = RNG.standard_normal((out_dim, rank)) / np.sqrt(rank)
    m_shared = RNG.uniform(0.5, 2.5, out_dim)
    cohort = []
    for _ in range(N_CLIENTS):
        cohort.append({
            "A": A_shared + HET * RNG.standard_normal((rank, in_dim)) / np.sqrt(in_dim),
            "B": B_shared + HET * RNG.standard_normal((out_dim, rank)) / np.sqrt(rank),
            "m": np.clip(m_shared + HET * RNG.normal(0, 0.3, out_dim), 0.1, 3.0),
        })
    return cohort


if __name__ == "__main__":
    print(f"=== Real-shape benchmark: Qwen2.5-1.5B modules, "
          f"{N_CLIENTS} clients ===\n")
    print(f"{'module':>10s} {'shape':>12s} {'rank':>5s} {'cohort_MB':>10s} "
          f"{'time_s':>8s}")

    total_per_layer = 0.0
    for name, (out_dim, in_dim, count, rank_key) in QWEN_MODULES.items():
        rank = DEFAULT_RANK_MAP[rank_key]
        cohort = make_cohort(out_dim, in_dim, rank)
        # dense cohort memory: N_CLIENTS reconstructed float64 matrices
        cohort_mb = N_CLIENTS * out_dim * in_dim * 8 / 1e6

        t0 = time.perf_counter()
        B_new, A_new, m_new, tel = aggregate_module(cohort, np.ones(N_CLIENTS), rank)
        elapsed = time.perf_counter() - t0

        total_per_layer += elapsed * count
        print(f"{name:>10s} {f'{out_dim}x{in_dim}':>12s} {rank:>5d} "
              f"{cohort_mb:>10.1f} {elapsed:>8.3f}")

    n_modules = len(QWEN_MODULES) * N_LAYERS
    full_round = total_per_layer * N_LAYERS
    print(f"\n  per-layer total:      {total_per_layer:.2f}s "
          f"({len(QWEN_MODULES)} modules)")
    print(f"  full round estimate:  {full_round:.1f}s "
          f"({n_modules} modules, {N_LAYERS} layers)")
    print(f"  with 4x Anchor safety factor: {full_round * 4:.1f}s")
    print(f"\n  peak streamed memory = largest single cohort above "
          f"(one module at a time by design)")
    verdict = ("OK -- comfortably inside a 30-min round cadence"
               if full_round * 4 < 900 else
               "WARNING -- rethink streaming/SVD approach before building main.py")
    print(f"  VERDICT: {verdict}")
