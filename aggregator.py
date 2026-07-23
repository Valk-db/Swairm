"""
aggregator.py -- Anchor-side aggregation core for the Federated Curriculum Swarm
=================================================================================
(v1.2 -- rank-dispatch fix: kv=2 is now reachable)

Pure tensor math, no I/O. The FastAPI/queue layers call aggregate_round() with
already-parsed uploads. Everything here is CPU-only NumPy/scikit-learn.

v1.2 changes:
  - DEFAULT_RANK_MAP's "kv" bucket was unreachable in practice: no module
    name used anywhere (self_attn.k_proj/v_proj, or the toy fleet's
    attn.k_proj/v_proj) contains the literal substring "kv", so k/v
    projections always matched "attn" instead and silently got rank 4, not
    the rank 2 that D6's rank-starvation diagnostic actually validated.
    Fixed by matching k_proj/v_proj on suffix before falling back to the
    attn/mlp substring check. No wire-format or upload-schema change; only
    affects modules with no explicit rank_map entry, which is every caller
    today (main.py doesn't pass one).

v1.1 changes (behavior-identical, benchmark-driven):
  - All dense math in float32 (DTYPE). Adapters are float16 on the wire;
    float64 aggregation was 2x memory and ~2x BLAS time for nothing.
    Real-shape benchmark showed 1.3GB float64 cohorts -- OOM risk on a
    4GB Anchor.
  - trimmed_weighted_mean rewritten: partition-based instead of double
    argsort (which fully sorted every coordinate twice and allocated two
    cohort-sized int64 rank arrays). Uniform-weight case uses a pure
    arithmetic trim (sum minus trimmed extremes) -- no index arrays at all.
    Uniform weights are the locked default for balanced participation, so
    this is the hot path.
  - If this is still too slow on real Anchor hardware, the remaining
    escape hatch is a factored SVD path (aggregate in low-rank form, SVD a
    (K*r x K*r) core, never materialize dense) -- but it is incompatible
    with coordinate-wise trimming, so adopting it is a robustness tradeoff
    requiring a human decision. Not implemented.

POLICY (locked by simulation evidence, 2026-07-20, seeds 42-71):
  - Staleness weighting is CONDITIONAL:
      balanced participation -> uniform weights
      detected participation skew -> reciprocal 1/(1+staleness)
    Evidence: reciprocal won 15/15 fresh seeds under schedule skew
    (paired t=6.07); under balanced participation it discarded ~75% of
    client work for no error benefit.
  - Agreement/relevance weighting is PARKED (unstable cluster asymmetry,
    catastrophic-seed collapse, worst retention). Do not reintroduce
    without new instrumentation.
  - DoRA magnitude m aggregated SEPARATELY (validated on this code path:
    53% error reduction vs frozen baseline at het=1.0; folded alternative
    wins by only 2-5% single-round but destroys m as a first-class client
    adapter component -- kept separate by decision).

PIPELINE (validated in simulate_fedavg.py v2 + async_event_sim v1.1):
  per module, streamed one at a time (bounds peak RAM to one module):
    reconstruct dense dW_k = B_k @ A_k          (avoids bilinear cross-term bias)
    L2-norm clip vs median norm                  (cheap corruption filter)
    coordinate-wise trimmed weighted mean        (robust reduction)
    randomized SVD -> truncate to module rank    (adaptive: kv=2, attn=4, mlp=6)

FORMERLY-OPEN ITEMS, now decided by evidence (validate_open_configs.py,
20 paired seeds on this production path -- see DECISIONS.md D8/D9):
  - TRIM_BEFORE_WEIGHTS = True is LOCKED: trim on raw values, then weight
    survivors. Won 20/20 paired seeds under active reciprocal weighting
    (t=13.4). Weighting first lets down-weighted stale values masquerade
    as extremes, trimming the wrong coordinates. Flag retained only for
    reproducing the experiment.
  - Curriculum epoch handling: hard rejection LOSES to soft transition
    weights in every tested transition regime (t=+21..+100). main.py now
    passes a one-step soft map {(epoch-1, epoch): 0.25}; older epochs are
    still hard-rejected.

Run self-test: python aggregator.py
"""

import numpy as np
from sklearn.utils.extmath import randomized_svd

# ------------------------------------------------------------------ config
DTYPE = np.float32
TELEMETRY_MARGIN = 2         # extra SVD components; feeds rank-starvation ratio
TRIM_FRAC = 0.15             # fraction trimmed per side, per coordinate
MIN_CLIENTS_FOR_TRIM = 5     # below this: weighted mean + norm clip only
NORM_CLIP_MULT = 3.0         # reject deltas with norm > mult * median norm
DEFAULT_RANK_MAP = {"kv": 2, "attn": 4, "mlp": 6}
RANK_STARVATION_THRESHOLD = 0.15   # trailing/top ratio (linear-sum convention;
                                    # re-tune if convention changes)
TRIM_BEFORE_WEIGHTS = True   # LOCKED (D8): 20/20 seeds, t=13.4; do not flip
                             # without new evidence (validate_open_configs.py)


def _default_target_rank(name):
    """Fallback rank for a module with no explicit rank_map entry.

    k_proj/v_proj are matched by suffix first: standard HF-style qualified
    names (self_attn.k_proj, self_attn.v_proj) never contain the literal
    substring "kv", so DEFAULT_RANK_MAP's "kv" entry can only be reached
    this way. Everything else still falls back to the attn/mlp substring
    check against DEFAULT_RANK_MAP, so the three canonical ranks (D6) stay
    defined in one place.
    """
    n = name.lower()
    if n.endswith("k_proj") or n.endswith("v_proj"):
        return DEFAULT_RANK_MAP["kv"]
    for key in ("attn", "mlp"):
        if key in n:
            return DEFAULT_RANK_MAP[key]
    return 4


# ------------------------------------------------------------------ weights
def staleness_weights(uploads, current_version, skew_detected):
    """Conditional policy locked by simulation evidence -- see module docstring."""
    out = []
    for u in uploads:
        s = max(current_version - u["fetch_version"], 0)
        out.append(1.0 / (1.0 + s) if skew_detected else 1.0)
    return np.array(out, dtype=DTYPE)


# ------------------------------------------------------------------ robust math
def trimmed_weighted_mean(stack, weights, trim_frac=TRIM_FRAC):
    """
    Coordinate-wise trimmed, weighted mean.
    stack: (k, ...) array of k client tensors. weights: (k,).
    With k < MIN_CLIENTS_FOR_TRIM or nothing to trim, falls back to a plain
    weighted mean (low-participation mode).

    Fast paths (v1.1):
      uniform weights -> arithmetic trim: (sum - trimmed extremes) / kept
                         via np.partition; no index arrays allocated.
      non-uniform     -> argpartition mask (O(k) per coordinate, no full sort).
    Composition order controlled by TRIM_BEFORE_WEIGHTS (open design item).
    """
    stack = np.asarray(stack, dtype=DTYPE)
    weights = np.asarray(weights, dtype=DTYPE)
    k = stack.shape[0]
    n_trim = int(np.floor(k * trim_frac))
    if k < MIN_CLIENTS_FOR_TRIM or n_trim == 0:
        w = weights / weights.sum()
        return np.tensordot(w, stack, axes=1).astype(DTYPE)

    uniform = np.allclose(weights, weights[0])
    if uniform and TRIM_BEFORE_WEIGHTS:
        # arithmetic trim: total minus the n_trim smallest and largest values
        low = np.partition(stack, n_trim - 1, axis=0)[:n_trim].sum(axis=0)
        high = np.partition(stack, k - n_trim, axis=0)[k - n_trim:].sum(axis=0)
        return ((stack.sum(axis=0) - low - high) / (k - 2 * n_trim)).astype(DTYPE)

    # weighted case: mask out per-coordinate extremes, weight survivors
    if TRIM_BEFORE_WEIGHTS:
        basis = stack
    else:
        basis = stack * weights.reshape((-1,) + (1,) * (stack.ndim - 1))
    keep = np.ones(stack.shape, dtype=bool)
    idx_low = np.argpartition(basis, n_trim - 1, axis=0)[:n_trim]
    idx_high = np.argpartition(basis, k - n_trim, axis=0)[k - n_trim:]
    np.put_along_axis(keep, idx_low, False, axis=0)
    np.put_along_axis(keep, idx_high, False, axis=0)

    w = weights.reshape((-1,) + (1,) * (stack.ndim - 1)) * keep
    denom = w.sum(axis=0)
    denom = np.where(denom > 0, denom, 1.0)
    return ((w * stack).sum(axis=0) / denom).astype(DTYPE)


def norm_clip_mask(dense_list):
    """Reject implausible-magnitude deltas relative to the cohort median."""
    norms = np.array([np.linalg.norm(d) for d in dense_list])
    median = np.median(norms)
    if median <= 0:
        return np.ones(len(dense_list), dtype=bool)
    return norms <= NORM_CLIP_MULT * median


# ------------------------------------------------------------------ per-module
def aggregate_module(module_uploads, weights, target_rank):
    """
    module_uploads: list of dicts {"A": (r, in), "B": (out, r), "m": (out,)}
    Returns (B_new, A_new, m_new, telemetry_dict).
    """
    dense = [np.asarray(u["B"], dtype=DTYPE) @ np.asarray(u["A"], dtype=DTYPE)
             for u in module_uploads]
    mask = norm_clip_mask(dense)
    dense = [d for d, ok in zip(dense, mask) if ok]
    mags = [np.asarray(u["m"], dtype=DTYPE)
            for u, ok in zip(module_uploads, mask) if ok]
    w = np.asarray(weights, dtype=DTYPE)[mask]
    n_rejected = int((~mask).sum())
    if not dense:
        return None, None, None, {"rejected": n_rejected, "aggregated": 0}

    agg_dense = trimmed_weighted_mean(np.stack(dense), w)
    del dense                                          # free cohort before SVD
    m_new = trimmed_weighted_mean(np.stack(mags), w)   # linear path -- no SVD

    n_components = target_rank + TELEMETRY_MARGIN
    U, S, Vt = randomized_svd(agg_dense, n_components=n_components,
                              random_state=42)
    A_new = (np.diag(np.sqrt(S[:target_rank])) @ Vt[:target_rank]).astype(DTYPE)
    B_new = (U[:, :target_rank] @ np.diag(np.sqrt(S[:target_rank]))).astype(DTYPE)

    top = S[:target_rank].sum()
    trailing = S[target_rank:n_components].sum()
    ratio = float(trailing / top) if top > 0 else 0.0

    telemetry = {
        "rejected": n_rejected,
        "aggregated": len(mags),
        "trailing_ratio": ratio,
        "rank_starved": ratio > RANK_STARVATION_THRESHOLD,
        "update_norm": float(np.linalg.norm(agg_dense)),
    }
    return B_new, A_new, m_new, telemetry


# ------------------------------------------------------------------ round
def aggregate_round(uploads, current_version, current_epoch, rank_map=None,
                    skew_detected=False, epoch_transition_weights=None):
    """
    uploads: list of dicts per adapter_schema.json:
      {device_id, fetch_version, curriculum_epoch,
       modules: {name: {"A", "B", "m"}}}
    rank_map: module_name -> target rank. Falls back to matching k_proj/
      v_proj by suffix (DEFAULT_RANK_MAP['kv']), then 'attn'/'mlp' by
      substring, else rank 4. See _default_target_rank.
    epoch_transition_weights: optional {(from_epoch, to_epoch): weight} soft
      map. Default None = hard rejection of epoch mismatches (conservative).

    Returns {"version", "modules": {name: {"B","A","m"}}, "telemetry"}.
    Streams one module at a time -- peak memory bounded to one module's cohort.
    """
    kept, epoch_w = [], []
    for u in uploads:
        if u["curriculum_epoch"] == current_epoch:
            kept.append(u)
            epoch_w.append(1.0)
        elif epoch_transition_weights:
            ew = epoch_transition_weights.get(
                (u["curriculum_epoch"], current_epoch), 0.0)
            if ew > 0:
                kept.append(u)
                epoch_w.append(ew)
    n_epoch_rejected = len(uploads) - len(kept)
    if not kept:
        return {"version": current_version, "modules": {},
                "telemetry": {"epoch_rejected": n_epoch_rejected,
                              "note": "no valid uploads; version unchanged"}}

    weights = staleness_weights(kept, current_version, skew_detected)
    weights = weights * np.array(epoch_w, dtype=DTYPE)

    module_names = list(kept[0]["modules"].keys())
    new_modules, telemetry = {}, {"epoch_rejected": n_epoch_rejected,
                                  "skew_detected": skew_detected,
                                  "modules": {}}
    for name in module_names:                      # streamed: one at a time
        target_rank = (rank_map or {}).get(name)
        if target_rank is None:
            target_rank = _default_target_rank(name)
        cohort = [u["modules"][name] for u in kept]
        B_new, A_new, m_new, mod_tel = aggregate_module(cohort, weights,
                                                        target_rank)
        if B_new is not None:
            new_modules[name] = {"B": B_new, "A": A_new, "m": m_new}
        telemetry["modules"][name] = mod_tel

    return {"version": current_version + 1,        # monotonic, even post-rollback
            "modules": new_modules, "telemetry": telemetry}


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def make_upload(device_id, fetch_version, epoch, m_dim, n_dim, rank,
                    shared, het, corrupt=False):
        A = shared["A"] + het * rng.standard_normal((rank, n_dim)) / np.sqrt(n_dim)
        B = shared["B"] + het * rng.standard_normal((m_dim, rank)) / np.sqrt(rank)
        m = np.clip(shared["m"] + het * rng.normal(0, 0.1, m_dim), 0.1, 3.0)
        if corrupt:
            B = B * 50.0
        return {"device_id": device_id, "fetch_version": fetch_version,
                "curriculum_epoch": epoch,
                "modules": {"layers.0.attn.q_proj": {"A": A, "B": B, "m": m}}}

    M_DIM, N_DIM, RANK = 128, 256, 4
    shared = {"A": rng.standard_normal((RANK, N_DIM)) / np.sqrt(N_DIM),
              "B": rng.standard_normal((M_DIM, RANK)) / np.sqrt(RANK),
              "m": rng.uniform(0.5, 2.5, M_DIM)}

    uploads = [make_upload(f"dev{i}", fetch_version=10, epoch=3,
                           m_dim=M_DIM, n_dim=N_DIM, rank=RANK,
                           shared=shared, het=1.0) for i in range(10)]
    uploads.append(make_upload("dev-corrupt", 10, 3, M_DIM, N_DIM, RANK,
                               shared, het=1.0, corrupt=True))
    uploads.append(make_upload("dev-stale", 4, 3, M_DIM, N_DIM, RANK,
                               shared, het=1.0))
    uploads.append(make_upload("dev-wrong-epoch", 10, 2, M_DIM, N_DIM, RANK,
                               shared, het=1.0))

    print("=== aggregator.py v1.1 self-test ===")
    for skew in (False, True):
        result = aggregate_round(uploads, current_version=10, current_epoch=3,
                                 skew_detected=skew)
        tel = result["telemetry"]
        mod = tel["modules"]["layers.0.attn.q_proj"]
        print(f"\n  skew_detected={skew}:")
        print(f"    new version:        {result['version']}")
        print(f"    epoch-rejected:     {tel['epoch_rejected']} (expect 1)")
        print(f"    norm-clip rejected: {mod['rejected']} (expect 1)")
        print(f"    aggregated:         {mod['aggregated']} (expect 11)")
        print(f"    trailing_ratio:     {mod['trailing_ratio']:.4f} "
              f"(rank_starved={mod['rank_starved']})")
        out = result["modules"]["layers.0.attn.q_proj"]
        print(f"    shapes: B{out['B'].shape} A{out['A'].shape} m{out['m'].shape}")
        print(f"    m mean: {out['m'].mean():.3f} "
              f"(shared m mean was {shared['m'].mean():.3f})")
    print("\nSelf-test complete. Expect values matching v1.0 to ~float32 "
          "precision (trailing_ratio ~0.064 / ~0.067, m mean ~1.559).")

    print("\n=== rank dispatch sanity check (v1.2 fix) ===")
    for name, expected in [
        ("model.layers.5.self_attn.k_proj", 2),   # torch_client.py naming
        ("model.layers.5.self_attn.v_proj", 2),
        ("model.layers.5.self_attn.q_proj", 4),
        ("model.layers.5.self_attn.o_proj", 4),
        ("layers.0.attn.k_proj", 2),               # toy-fleet-style naming
        ("layers.0.attn.v_proj", 2),
        ("layers.0.attn.q_proj", 4),               # what swairm-client uses today
        ("model.layers.5.mlp.down_proj", 6),
        ("model.layers.5.mlp.gate_proj", 6),
    ]:
        got = _default_target_rank(name)
        status = "OK" if got == expected else "FAIL"
        print(f"    {status}: {name:38s} -> {got} (expected {expected})")
