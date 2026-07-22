"""
validate_magnitude.py
=====================
Closes the last unverified math claim in the FCS record: DoRA magnitude
aggregation. A prior report claimed "magnitude aggregation gives measurable
benefit" but never captured output. This script tests the ACTUAL production
code path (aggregator.aggregate_module), not a reimplementation.

Three pipelines, identical SVD bottleneck (apples-to-apples):
  frozen    directional aggregate only, magnitudes ignored (m=1 baseline)
  separate  aggregator.py's path: direction via reconstruct->SVD, m via
            coordinate-wise trimmed weighted mean, applied after
  folded    fold m into the dense reconstruction BEFORE aggregation/SVD
            (the alternative flagged as never-validated in the design spec)

Ground truth: fleet mean of full DoRA updates, mean_k[(B_k A_k) * m_k].

Pre-registered decision rules (before running):
  1. separate must beat frozen by >20% at heterogeneity=1.0, else the
     magnitude path adds complexity without benefit -> revisit.
  2. separate vs folded: whichever wins consistently across heterogeneity
     levels becomes the aggregator default; if within noise of each other,
     keep separate (preserves DoRA's m as a first-class adapter component
     for clients, which folded destroys).

Run: python validate_magnitude.py   (fast -- single rounds, toy scale)
"""

import numpy as np
from sklearn.utils.extmath import randomized_svd
from aggregator import aggregate_module, TELEMETRY_MARGIN, DTYPE

RNG = np.random.default_rng(42)
M_DIM, N_DIM, RANK = 128, 256, 4
N_CLIENTS = 12
HET_LEVELS = [0.25, 0.5, 1.0, 2.0]


def rel_frobenius_error(estimate, ground_truth):
    return np.linalg.norm(estimate - ground_truth, "fro") / np.linalg.norm(ground_truth, "fro")


def make_dora_clients(het):
    """Shared basis + non-trivial shared magnitude profile, per-client noise."""
    A_shared = RNG.standard_normal((RANK, N_DIM)) / np.sqrt(N_DIM)
    B_shared = RNG.standard_normal((M_DIM, RANK)) / np.sqrt(RANK)
    m_shared = RNG.uniform(0.5, 2.5, M_DIM)   # non-trivial: m=1 would make
                                               # the frozen baseline unbeatable
    uploads = []
    for _ in range(N_CLIENTS):
        A_k = A_shared + het * RNG.standard_normal((RANK, N_DIM)) / np.sqrt(N_DIM)
        B_k = B_shared + het * RNG.standard_normal((M_DIM, RANK)) / np.sqrt(RANK)
        m_k = np.clip(m_shared + het * RNG.normal(0, 0.3, M_DIM), 0.1, 3.0)
        uploads.append({"A": A_k, "B": B_k, "m": m_k})
    return uploads


def svd_truncate(dense, rank):
    U, S, Vt = randomized_svd(dense, n_components=rank + TELEMETRY_MARGIN,
                              random_state=42)
    return (U[:, :rank] * S[:rank]) @ Vt[:rank]


SCALING = 8.0 / RANK   # lora_alpha=8.0, r=4 -- matches apply_dora_patching()
                       # in torch_client.py; named since it feeds a formula
                       # below, not just a print string.


def dora_effective_weight(W0, B, A, m, scaling=SCALING):
    """Mirrors DoRALinear._compute_dora_weight exactly (row-wise norm).
    torch_client.py: combined = W0 + delta_w; return m * combined/||combined||
    """
    combined = W0 + scaling * (np.asarray(B) @ np.asarray(A))
    row_norm = np.linalg.norm(combined, axis=1, keepdims=True)
    return np.asarray(m, dtype=DTYPE)[:, np.newaxis] * (combined / row_norm)


def run_dora_consistent_experiment():
    """
    Second pass, same three pipelines, DIFFERENT ground truth.

    The experiment above treats magnitude and direction as independent
    linear quantities: gt = mean_k[(B_k@A_k) * m_k] -- no base weight, no
    normalization. Real DoRA (torch_client.py's DoRALinear) computes
        W_k = m_k * (W0 + scaling*(B_k@A_k)) / ||W0 + scaling*(B_k@A_k)||_row
    Normalization does not commute with averaging (mean_k[normalize(v_k)] !=
    normalize(mean_k[v_k]) in general), so a fleet-average m calibrated
    per-client against ||combined_k|| is not guaranteed to relate sensibly
    to ||combined_new||. This block re-runs frozen/separate/folded against
    a ground truth built from the REAL nonlinear formula, to check whether
    D4's conclusion survives contact with actual training semantics.

    This is NEW EVIDENCE for D4, not a re-argument of it -- per this file's
    own header, reopening D4 requires a human decision, so this prints a
    verdict but does not touch aggregator.py or DECISIONS.md.
    """
    print("\n=== DoRA-consistent (nonlinear) ground truth check ===")
    print("(same pipelines as above; gt now includes W0 + row-wise norm)\n")
    print(f"{'het':>5s} {'frozen':>9s} {'separate':>9s} {'folded':>9s} "
          f"{'sep/frozen':>11s}")

    W0 = np.random.default_rng(7).standard_normal((M_DIM, N_DIM)).astype(DTYPE) / np.sqrt(N_DIM)

    results = []
    for het in HET_LEVELS:
        uploads = make_dora_clients(het)
        weights = np.ones(N_CLIENTS)

        # true nonlinear ground truth: mean of each client's REAL effective
        # weight, i.e. what their local DoRALinear actually computes
        gt = np.mean([dora_effective_weight(W0, u["B"], u["A"], u["m"])
                      for u in uploads], axis=0)

        # production path: direction via reconstruct->SVD, m separate --
        # reconstruct the effective weight the same way a client would
        B_new, A_new, m_new, _ = aggregate_module(uploads, weights, RANK)
        est_separate = dora_effective_weight(W0, B_new, A_new, m_new)

        # frozen baseline: same direction estimate, magnitude ignored (m=1) --
        # isolates the value of magnitude tracking, holding direction fixed
        est_frozen = dora_effective_weight(W0, B_new, A_new, np.ones(M_DIM, dtype=DTYPE))

        # folded: m fused into the dense delta before aggregation/SVD, so
        # there's no separate m left to renormalize with -- the SVD output
        # IS the fleet's estimate of the full increment over W0
        dense_folded = np.mean([(np.asarray(u["B"]) @ np.asarray(u["A"]))
                                * np.asarray(u["m"])[:, np.newaxis]
                                for u in uploads], axis=0)
        est_folded = W0 + SCALING * svd_truncate(dense_folded, RANK)

        e_frozen = rel_frobenius_error(est_frozen, gt)
        e_separate = rel_frobenius_error(est_separate, gt)
        e_folded = rel_frobenius_error(est_folded, gt)
        results.append((het, e_frozen, e_separate, e_folded))
        print(f"{het:>5.2f} {e_frozen:>9.4f} {e_separate:>9.4f} "
              f"{e_folded:>9.4f} {e_separate / e_frozen:>11.3f}")

    print("\nSame pre-registered decision rules, checked against the")
    print("nonlinear ground truth instead of the linear proxy:")
    het1 = next(r for r in results if r[0] == 1.0)
    rule1 = het1[2] < 0.8 * het1[1]
    print(f"  Rule 1 (separate beats frozen by >20% at het=1.0): "
          f"{'PASS' if rule1 else 'FAIL -- magnitude path does not clear the bar under real DoRA semantics'}")
    sep_wins = sum(1 for _, _, s, f in results if s < f)
    fold_wins = sum(1 for _, _, s, f in results if f < s)
    print(f"  Rule 2 (separate vs folded): separate wins {sep_wins}/{len(results)}, "
          f"folded wins {fold_wins}/{len(results)}")
    if fold_wins > sep_wins:
        print("    -> folded is more accurate here too, but D4's reasoning for")
        print("       keeping m first-class is architectural (client adapter")
        print("       semantics), not purely accuracy -- still a human call.")
    else:
        print("    -> separate still holds up under the real nonlinear formula.")


if __name__ == "__main__":
    print("=== DoRA magnitude aggregation validation "
          "(production aggregate_module path) ===\n")
    print(f"{'het':>5s} {'frozen':>9s} {'separate':>9s} {'folded':>9s} "
          f"{'sep/frozen':>11s}")

    results = []
    for het in HET_LEVELS:
        uploads = make_dora_clients(het)
        weights = np.ones(N_CLIENTS)

        # ground truth: fleet mean of full DoRA updates
        gt = np.mean([(np.asarray(u["B"]) @ np.asarray(u["A"]))
                      * np.asarray(u["m"])[:, np.newaxis]
                      for u in uploads], axis=0)

        # production path: direction via reconstruct->SVD, m separate
        B_new, A_new, m_new, _ = aggregate_module(uploads, weights, RANK)
        directional = B_new @ A_new
        est_separate = directional * m_new[:, np.newaxis]
        est_frozen = directional                      # identical bottleneck, m=1

        # folded alternative: scale each client's dense update by its m first
        dense_folded = np.mean([(np.asarray(u["B"]) @ np.asarray(u["A"]))
                                * np.asarray(u["m"])[:, np.newaxis]
                                for u in uploads], axis=0)
        est_folded = svd_truncate(dense_folded, RANK)

        e_frozen = rel_frobenius_error(est_frozen, gt)
        e_separate = rel_frobenius_error(est_separate, gt)
        e_folded = rel_frobenius_error(est_folded, gt)
        results.append((het, e_frozen, e_separate, e_folded))
        print(f"{het:>5.2f} {e_frozen:>9.4f} {e_separate:>9.4f} "
              f"{e_folded:>9.4f} {e_separate / e_frozen:>11.3f}")

    print("\nPre-registered decisions:")
    het1 = next(r for r in results if r[0] == 1.0)
    rule1 = het1[2] < 0.8 * het1[1]
    print(f"  Rule 1 (separate beats frozen by >20% at het=1.0): "
          f"{'PASS -- magnitude path earns its keep' if rule1 else 'FAIL -- magnitude path adds complexity without benefit; revisit'}")
    sep_wins = sum(1 for _, _, s, f in results if s < f)
    fold_wins = sum(1 for _, _, s, f in results if f < s)
    print(f"  Rule 2 (separate vs folded): separate wins {sep_wins}/{len(results)}, "
          f"folded wins {fold_wins}/{len(results)}")
    if fold_wins > sep_wins:
        print("    -> folded is more accurate, BUT adopting it destroys m as a "
              "first-class client adapter component -- flag for human decision "
              "rather than auto-switching.")
    else:
        print("    -> keep separate (current aggregator default).")

    run_dora_consistent_experiment()