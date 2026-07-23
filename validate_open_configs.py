"""
validate_open_configs.py -- evidence runs for the two config-surfaced open items
=================================================================================

Closes (or keeps open, per evidence) the two DECISIONS.md items that were
deliberately surfaced as config instead of silently decided:

  1. TRIM_BEFORE_WEIGHTS (aggregator.trimmed_weighted_mean composition order)
     Order only matters on the non-uniform-weight branch, i.e. when
     skew_detected=True activates reciprocal 1/(1+s) staleness weights.
     Question: should per-coordinate extremes be identified on RAW values
     (trim first, weight survivors) or on WEIGHTED values (weight first,
     then trim)?

  2. epoch_transition_weights (hard vs soft curriculum-epoch rejection)
     Default is hard rejection of epoch mismatches. Question: during an
     epoch transition, when only a few clients have re-fetched onto the new
     epoch, does soft-weighting the old-epoch stragglers beat throwing them
     away?

Both experiments run on the PRODUCTION aggregate_round path (same standard
as D4/D5 -- no parallel reimplementation of the math), paired seeds, with a
manual paired t-test (no scipy dependency).

Modeling assumptions, stated openly:
  - Client uploads are noisy low-rank observations of a shared target
    (het=1.0, matching the D4 evidence level).
  - Stale clients observed an OLDER target: shared + drift, drift norm
    proportional to staleness. This is what makes staleness weighting
    matter, and what makes composition order observable.
  - Coordinate corrupters scale their delta by 2.5x -- inside the 3x-median
    norm clip (so the clip cannot catch them) but far enough out that
    per-coordinate trimming should.
  - Old-epoch clients trained toward the PREVIOUS curriculum target
    T_old = T_new - shift. Two shift scales bracket "adjacent, correlated
    epochs" (0.3) and "hard curriculum break" (1.0).

Run: /path/to/python validate_open_configs.py
"""

import numpy as np

import aggregator
from aggregator import aggregate_round

M_DIM, N_DIM, RANK = 128, 256, 4
MODULE = "layers.0.attn.q_proj"
SEEDS = list(range(42, 62))            # 20 paired seeds


# ------------------------------------------------------------------ helpers
def low_rank(rng, rank, scale=1.0):
    B = rng.standard_normal((M_DIM, rank)) / np.sqrt(rank)
    A = rng.standard_normal((rank, N_DIM)) / np.sqrt(N_DIM)
    return scale * B @ A, A, B


def factor(dense, rank):
    """Exact rank-r factorization of a dense target for building uploads."""
    U, S, Vt = np.linalg.svd(dense, full_matrices=False)
    A = np.diag(np.sqrt(S[:rank])) @ Vt[:rank]
    B = U[:, :rank] @ np.diag(np.sqrt(S[:rank]))
    return A, B


def make_upload(rng, device_id, dense_target, fetch_version, epoch,
                het=1.0, coord_corrupt=False):
    A_t, B_t = factor(dense_target, RANK)
    A = A_t + het * rng.standard_normal((RANK, N_DIM)) / np.sqrt(N_DIM)
    B = B_t + het * rng.standard_normal((M_DIM, RANK)) / np.sqrt(RANK)
    m = np.clip(rng.uniform(0.5, 2.5, M_DIM) + rng.normal(0, 0.1, M_DIM),
                0.1, 3.0)
    if coord_corrupt:
        B = B * 2.5           # inside the 3x-median norm clip, outside sanity
    return {"device_id": device_id, "fetch_version": fetch_version,
            "curriculum_epoch": epoch,
            "modules": {MODULE: {"A": A, "B": B, "m": m}}}


def rel_err(result, dense_target):
    mod = result["modules"][MODULE]
    est = np.asarray(mod["B"], dtype=np.float64) @ np.asarray(mod["A"],
                                                              dtype=np.float64)
    return float(np.linalg.norm(est - dense_target, "fro")
                 / np.linalg.norm(dense_target, "fro"))


def paired_t(diffs):
    d = np.asarray(diffs, dtype=np.float64)
    if len(d) < 2 or np.allclose(d.std(ddof=1), 0):
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


# ------------------------------------------------------------------ exp 1
def experiment_trim_order():
    """
    TRIM_BEFORE_WEIGHTS True vs False under active reciprocal weighting.
    Fleet: 8 fresh, 4 stale (staleness 3, drifted target), 2 coordinate
    corrupters. skew_detected=True -> non-uniform weights -> order matters.
    """
    print("=" * 72)
    print("Experiment 1: TRIM_BEFORE_WEIGHTS (order of trim vs weights)")
    print("=" * 72)
    rows = {True: [], False: []}
    for seed in SEEDS:
        for flag in (True, False):
            rng = np.random.default_rng(seed)      # identical world per flag
            target, _, _ = low_rank(rng, RANK)
            drift, _, _ = low_rank(rng, 2, scale=0.5)
            stale_target = target + drift          # what stale clients saw
            uploads = []
            for i in range(8):
                uploads.append(make_upload(rng, f"fresh{i}", target,
                                           fetch_version=10, epoch=3))
            for i in range(4):
                uploads.append(make_upload(rng, f"stale{i}", stale_target,
                                           fetch_version=7, epoch=3))
            for i in range(2):
                uploads.append(make_upload(rng, f"corrupt{i}", target,
                                           fetch_version=10, epoch=3,
                                           coord_corrupt=True))
            aggregator.TRIM_BEFORE_WEIGHTS = flag
            try:
                result = aggregate_round(uploads, current_version=10,
                                         current_epoch=3, skew_detected=True)
            finally:
                aggregator.TRIM_BEFORE_WEIGHTS = True     # restore default
            rows[flag].append(rel_err(result, target))

    tb = np.array(rows[True])
    wb = np.array(rows[False])
    diffs = wb - tb                               # >0 -> trim-first better
    wins = int((diffs > 0).sum())
    print(f"  seeds: {len(SEEDS)} paired")
    print(f"  trim-before-weights : err {tb.mean():.4f} +/- {tb.std():.4f}")
    print(f"  weights-before-trim : err {wb.mean():.4f} +/- {wb.std():.4f}")
    print(f"  trim-first wins {wins}/{len(SEEDS)}, mean gap "
          f"{diffs.mean():+.4f}, paired t={paired_t(diffs):.2f}")
    return tb, wb, diffs


# ------------------------------------------------------------------ exp 2
def experiment_epoch_rejection():
    """
    Hard rejection vs soft transition weights during an epoch changeover.
    Sweep: n_current in {2, 4, 8} of a 12-client cohort (rest old-epoch),
    shift scale in {0.3, 1.0}, soft weight in {0.25, 0.5}.
    Error measured against the NEW epoch target.
    """
    print()
    print("=" * 72)
    print("Experiment 2: hard vs soft curriculum-epoch rejection")
    print("=" * 72)
    N_TOTAL = 12
    summary = []
    for shift_scale in (0.3, 1.0):
        for n_current in (2, 4, 8):
            errs = {"hard": [], "soft25": [], "soft50": []}
            for seed in SEEDS:
                worlds = {}
                for policy, tw in (("hard", None),
                                   ("soft25", {(2, 3): 0.25}),
                                   ("soft50", {(2, 3): 0.5})):
                    rng = np.random.default_rng(seed)   # identical world
                    t_new, _, _ = low_rank(rng, RANK)
                    shift, _, _ = low_rank(rng, 2, scale=shift_scale)
                    t_old = t_new - shift
                    uploads = []
                    for i in range(n_current):
                        uploads.append(make_upload(rng, f"cur{i}", t_new,
                                                   fetch_version=10, epoch=3))
                    for i in range(N_TOTAL - n_current):
                        uploads.append(make_upload(rng, f"old{i}", t_old,
                                                   fetch_version=10, epoch=2))
                    result = aggregate_round(uploads, current_version=10,
                                             current_epoch=3,
                                             epoch_transition_weights=tw)
                    errs[policy].append(rel_err(result, t_new))
            h = np.array(errs["hard"])
            s25 = np.array(errs["soft25"])
            s50 = np.array(errs["soft50"])
            d25, d50 = h - s25, h - s50            # >0 -> soft better
            print(f"  shift={shift_scale:.1f} n_current={n_current:2d}/12 | "
                  f"hard {h.mean():.4f} | soft.25 {s25.mean():.4f} "
                  f"(t={paired_t(d25):+5.2f}) | soft.50 {s50.mean():.4f} "
                  f"(t={paired_t(d50):+5.2f})")
            summary.append((shift_scale, n_current, h.mean(), s25.mean(),
                            s50.mean(), paired_t(d25), paired_t(d50)))
    return summary


if __name__ == "__main__":
    experiment_trim_order()
    experiment_epoch_rejection()
    print()
    print("Interpretation guide:")
    print("  Exp 1: composition order only fires on the non-uniform branch")
    print("    (skew_detected). If one order wins consistently (|t| > ~2.9,")
    print("    the 15/15-style bar used for D2), lock it and drop the flag")
    print("    from the open list. If they tie, keep the default and note")
    print("    that the order is immaterial at this evidence level.")
    print("  Exp 2: hard rejection is safe iff enough current-epoch clients")
    print("    exist. If soft wins only at small n_current and small shift,")
    print("    that argues for keeping hard as default with a documented")
    print("    escape hatch -- not for changing the default.")
