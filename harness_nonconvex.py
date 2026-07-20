"""
harness_nonconvex.py -- Non-convex validation harness (v0.1)
=============================================================
Re-tests three policies decided (or left open) on LINEAR-proxy evidence,
under real gradient descent on a non-convex model:

  E1  D2 conditional staleness weighting  (uniform vs reciprocal 1/(1+s))
  E2  TRIM_BEFORE_WEIGHTS ordering        (True vs False, skew + sneaky corruption)
  E3  Curriculum epoch handling           (hard rejection vs soft transition weights)

Model: frozen 2-layer tanh MLP; clients train a true DoRA adapter
(A, B, m) on layer 1 with manual-gradient SGD. Aggregation goes through
the PRODUCTION aggregate_round() code path (imported, not reimplemented,
per D4's lesson). Non-convexity sources: tanh, DoRA row-normalization,
and staleness realized as genuinely training from an old global snapshot.

Pre-registered decision rules:
  CONFIRMED  -> same policy wins (paired t > 2.75, no catastrophic seeds)
  REOPENED   -> flips with |t| > 2, or any catastrophic seed
                (final_err > 4x the per-seed best arm)

Run:  python harness_nonconvex.py --selftest   (single-client convergence check)
      python harness_nonconvex.py --quick      (5 seeds, smoke test)
      python harness_nonconvex.py              (full: seeds 42-71)
"""

import sys

import numpy as np

import aggregator
from aggregator import aggregate_round

# ------------------------------------------------------------------ config
IN_DIM, HID_DIM, OUT_DIM, RANK = 64, 48, 8, 4
N_CLIENTS = 12
N_SLOW = 4                  # demographically distinct, stale-fetching group
BATCH, LOCAL_STEPS = 64, 15
LR = 0.3                    # tune if --selftest fails (diverges: lower it)
ROUNDS = 30
FETCH_EVERY_SLOW = 6        # slow group refetches the global every N rounds
CLIENT_HET = 0.5            # within-group client heterogeneity
GROUP_HET = 1.2             # slow-group teacher offset (demographic skew)
LABEL_NOISE = 0.05
EPOCH_ROUND = 15            # E3: curriculum transition round
N_LAGGARDS = 3              # E3: clients uploading the old epoch late
LAG_ROUNDS = 5
CORRUPT_SCALE = 2.5         # E2: below NORM_CLIP_MULT=3.0 -> survives the
                            # clip, must be caught by the trim. Sneaky.
SEEDS = list(range(42, 72))
EVAL_N = 512
CATASTROPHIC_MULT = 4.0


# ------------------------------------------------------------------ DoRA math
def dora_weight(W0, A, B, m):
    V = W0 + B @ A
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return m[:, None] * V / norms, V, norms


def forward(X, W0, W2, adapter):
    Wp, _, _ = dora_weight(W0, adapter["A"], adapter["B"], adapter["m"])
    return np.tanh(X @ Wp.T) @ W2.T


def local_train(W0, W2, adapter, X, T, steps=LOCAL_STEPS, lr=LR):
    """Manual-gradient SGD through the DoRA parameterization. Returns a
    NEW adapter dict (never mutates the fetched global snapshot)."""
    A, B, m = adapter["A"].copy(), adapter["B"].copy(), adapter["m"].copy()
    for _ in range(steps):
        Wp, V, norms = dora_weight(W0, A, B, m)
        Z = X @ Wp.T
        H = np.tanh(Z)
        Y = H @ W2.T
        dY = 2.0 * (Y - T) / T.size
        dZ = (dY @ W2) * (1.0 - H * H)
        dWp = dZ.T @ X                                   # (out, in)
        Vhat = V / norms
        dm = np.sum(dWp * Vhat, axis=1)
        proj = np.sum(dWp * Vhat, axis=1, keepdims=True) * Vhat
        dV = (m[:, None] / norms) * (dWp - proj)
        A -= lr * (B.T @ dV)
        B -= lr * (dV @ A.T)
        m = np.clip(m - lr * dm, 0.05, 5.0)
    return {"A": A, "B": B, "m": m}


def mse(X, T, W0, W2, adapter):
    Y = forward(X, W0, W2, adapter)
    return float(np.mean((Y - T) ** 2))


# ------------------------------------------------------------------ world setup
def make_teacher(rng, W0):
    A = rng.standard_normal((RANK, IN_DIM)) / np.sqrt(IN_DIM)
    B = rng.standard_normal((HID_DIM, RANK)) / np.sqrt(RANK)
    V = W0 + B @ A
    m = np.linalg.norm(V, axis=1) * rng.uniform(0.7, 1.4, HID_DIM)
    return {"A": A, "B": B, "m": m}


def perturb(rng, t, het):
    return {"A": t["A"] + het * rng.standard_normal(t["A"].shape) / np.sqrt(IN_DIM),
            "B": t["B"] + het * rng.standard_normal(t["B"].shape) / np.sqrt(RANK),
            "m": np.clip(t["m"] + het * rng.normal(0, 0.1, HID_DIM), 0.05, 5.0)}


def init_global(W0, rng):
    """LoRA-standard init (A random, B zero) so gradients flow; DoRA init
    m = row-norm(W0) so the effective weight starts exactly at W0."""
    return {"A": rng.standard_normal((RANK, IN_DIM)) * 0.01,
            "B": np.zeros((HID_DIM, RANK)),
            "m": np.linalg.norm(W0, axis=1)}


# ------------------------------------------------------------------ fleet loop
def run_fleet(seed, scenario, policy, trim_before=True, epoch_weights=None,
              corrupt=False, rounds=ROUNDS):
    """
    scenario: 'balanced' (one population, everyone fetches fresh) or
              'skewed'   (slow group: distinct teacher + stale fetches)
    policy:   'uniform' or 'reciprocal' -> passed as skew_detected to the
              production aggregator (we test the weighting, not the detector)
    Returns {'errs': per-round fleet eval, 'final': ..., 'retention': ...}
    """
    aggregator.TRIM_BEFORE_WEIGHTS = trim_before
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((HID_DIM, IN_DIM)) / np.sqrt(IN_DIM)
    W2 = rng.standard_normal((OUT_DIM, HID_DIM)) / np.sqrt(HID_DIM)

    teachers = {1: make_teacher(rng, W0)}
    teachers["slow1"] = (perturb(rng, teachers[1], GROUP_HET)
                         if scenario == "skewed" else teachers[1])
    if epoch_weights is not None or scenario == "epoch":
        teachers[2] = perturb(rng, teachers[1], 1.0)      # curriculum shift
        teachers["slow2"] = teachers[2]

    clients = []
    for i in range(N_CLIENTS):
        slow = i >= N_CLIENTS - N_SLOW    # stale fetchers exist in BOTH scenarios
        clients.append({"id": f"dev{i}", "slow": slow,
                        "off": {1: perturb(rng, {"A": np.zeros_like(teachers[1]["A"]),
                                                 "B": np.zeros_like(teachers[1]["B"]),
                                                 "m": np.zeros(HID_DIM)}, CLIENT_HET)},
                        "fetch_v": 0})

    def client_teacher(c, ep):
        base = teachers[f"slow{ep}" if c["slow"] else ep]
        o = c["off"][1]
        return {"A": base["A"] + o["A"], "B": base["B"] + o["B"],
                "m": np.clip(base["m"] + o["m"], 0.05, 5.0)}

    X_eval = rng.standard_normal((EVAL_N, IN_DIM))

    def fleet_err(adapter, ep):
        errs, ns = [], []
        for key, n in ((ep, N_CLIENTS - (N_SLOW if scenario == "skewed" else 0)),
                       (f"slow{ep}", N_SLOW if scenario == "skewed" else 0)):
            if n == 0:
                continue
            T = forward(X_eval, W0, W2, teachers[key])
            errs.append(mse(X_eval, T, W0, W2, adapter))
            ns.append(n)
        return float(np.average(errs, weights=ns))

    history = [init_global(W0, rng)]
    version, epoch = 0, 1
    errs, kept_weight = [], []

    for r in range(rounds):
        epoch_now = 2 if (2 in teachers and r >= EPOCH_ROUND) else 1
        epoch = epoch_now
        corrupt_id = rng.integers(0, N_CLIENTS - N_SLOW) if corrupt else -1
        uploads = []
        for i, c in enumerate(clients):
            if not c["slow"] or r % FETCH_EVERY_SLOW == 0:
                c["fetch_v"] = version                    # refetch
            start = history[c["fetch_v"]]
            lagging = (2 in teachers and i < N_LAGGARDS
                       and EPOCH_ROUND <= r < EPOCH_ROUND + LAG_ROUNDS)
            train_ep = 1 if lagging else epoch_now
            t = client_teacher(c, train_ep)
            X = rng.standard_normal((BATCH, IN_DIM))
            T = forward(X, W0, W2, t) + LABEL_NOISE * rng.standard_normal((BATCH, OUT_DIM))
            trained = local_train(W0, W2, start, X, T)
            if i == corrupt_id:
                trained = {"A": trained["A"], "m": trained["m"],
                           "B": trained["B"] * CORRUPT_SCALE}
            uploads.append({"device_id": c["id"], "fetch_version": c["fetch_v"],
                            "curriculum_epoch": train_ep,
                            "modules": {"layer1": trained}})

        result = aggregate_round(uploads, current_version=version,
                                 current_epoch=epoch_now,
                                 rank_map={"layer1": RANK},
                                 skew_detected=(policy == "uniform"),  # inverted post-D8,
                                 epoch_transition_weights=epoch_weights)
        if result["modules"]:
            g = result["modules"]["layer1"]
            adapter = {k: np.asarray(g[k], dtype=np.float64) for k in ("A", "B", "m")}
            version = result["version"]
            history.append(adapter)
        w = [(1.0 / (1.0 + max(version - 1 - u["fetch_version"], 0))
              if policy == "reciprocal" else 1.0) for u in uploads]
        kept_weight.append(float(np.mean(w)))
        errs.append(fleet_err(history[-1], epoch_now))

    return {"errs": errs, "final": errs[-1],
            "retention": float(np.mean(kept_weight))}


# ------------------------------------------------------------------ stats
def paired_t(diffs):
    d = np.asarray(diffs, dtype=np.float64)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))


def report(name, arms, seeds):
    """arms: {label: [final_err per seed]}. Prints decision-log-style summary."""
    labels = list(arms)
    print(f"\n=== {name} ===")
    for lab in labels:
        v = np.array(arms[lab])
        print(f"  {lab:>28s}: mean final_err {v.mean():.4f}  (std {v.std():.4f})")
    best_per_seed = np.min([arms[l] for l in labels], axis=0)
    for lab in labels:
        cat = [s for s, (e, b) in zip(seeds, zip(arms[lab], best_per_seed))
               if e > CATASTROPHIC_MULT * max(b, 1e-9)]
        if cat:
            print(f"  !! CATASTROPHIC seeds for {lab}: {cat}")
    if len(labels) == 2:
        a, b = labels
        d = np.array(arms[a]) - np.array(arms[b])
        wins = int((d > 0).sum())
        t = paired_t(d)
        print(f"  {b} beats {a}: {wins}/{len(seeds)} seeds, "
              f"paired t={t:.2f}, mean gap {d.mean():.4f}")
        verdict = ("CONFIRMED" if t > 2.75 and wins >= 0.8 * len(seeds)
                   else "REOPEN" if t < -2 else "INCONCLUSIVE")
        print(f"  verdict vs linear-proxy expectation: {verdict}")

def sweep(seeds):
    """E1 sensitivity: is the D2 flip robust across regimes?"""
    global FETCH_EVERY_SLOW, GROUP_HET
    print("=== E1 sensitivity sweep (positive gap = reciprocal better) ===")
    for fe in (3, 6, 9):
        FETCH_EVERY_SLOW = fe
        for gh, scens in ((0.6, ("balanced", "skewed")), (1.2, ("skewed",)),
                          (2.0, ("skewed",))):
            GROUP_HET = gh
            for scen in scens:
                u = [run_fleet(s, scen, "uniform")["final"] for s in seeds]
                r = [run_fleet(s, scen, "reciprocal")["final"] for s in seeds]
                d = np.array(u) - np.array(r)
                print(f"  fetch_every={fe} group_het={gh:.1f} {scen:>8s}: "
                      f"reciprocal wins {int((d > 0).sum())}/{len(seeds)}, "
                      f"t={paired_t(d):+.2f}, gap {d.mean():+.4f}")

# ------------------------------------------------------------------ experiments
def main(seeds):
    # E1 -- D2 revalidation: staleness policy under both participation regimes
    for scen in ("balanced", "skewed"):
        arms = {"uniform": [], "reciprocal 1/(1+s)": []}
        ret = {}
        for pol, lab in (("uniform", "uniform"), ("reciprocal", "reciprocal 1/(1+s)")):
            runs = [run_fleet(s, scen, pol) for s in seeds]
            arms[lab] = [r["final"] for r in runs]
            ret[lab] = np.mean([r["retention"] for r in runs])
        report(f"E1 staleness policy ({scen})", arms, seeds)
        print(f"  retention: " + ", ".join(f"{k}={v:.3f}" for k, v in ret.items()))

    # E2 -- trim ordering: needs non-uniform weights (skew) + sneaky corruption
    arms = {}
    for tb in (True, False):
        runs = [run_fleet(s, "skewed", "reciprocal", trim_before=tb, corrupt=True)
                for s in seeds]
        arms[f"TRIM_BEFORE_WEIGHTS={tb}"] = [r["final"] for r in runs]
    report("E2 trim/weight composition order (skew + corruption)", arms, seeds)

    # E3 -- epoch transition: hard rejection vs soft weight maps
    arms, dips = {}, {}
    for lab, ew in (("hard rejection", None),
                    ("soft w=0.3", {(1, 2): 0.3}),
                    ("soft w=0.5", {(1, 2): 0.5})):
        runs = [run_fleet(s, "epoch", "uniform", epoch_weights=ew) for s in seeds]
        arms[lab] = [r["final"] for r in runs]
        dips[lab] = np.mean([max(r["errs"][EPOCH_ROUND:EPOCH_ROUND + LAG_ROUNDS])
                             for r in runs])
    report("E3 curriculum-epoch handling", arms, seeds)
    print("  transition dip (mean max err, rounds "
          f"{EPOCH_ROUND}-{EPOCH_ROUND + LAG_ROUNDS - 1}): "
          + ", ".join(f"{k}={v:.4f}" for k, v in dips.items()))


def selftest():
    print("=== harness self-test: single-client convergence ===")
    rng = np.random.default_rng(0)
    W0 = rng.standard_normal((HID_DIM, IN_DIM)) / np.sqrt(IN_DIM)
    W2 = rng.standard_normal((OUT_DIM, HID_DIM)) / np.sqrt(HID_DIM)
    teacher = make_teacher(rng, W0)
    X = rng.standard_normal((256, IN_DIM))
    T = forward(X, W0, W2, teacher)
    a = init_global(W0, rng)
    e0 = mse(X, T, W0, W2, a)
    a = local_train(W0, W2, a, X, T, steps=200)
    e1 = mse(X, T, W0, W2, a)
    print(f"  loss {e0:.4f} -> {e1:.4f}")
    assert np.isfinite(e1) and e1 < 0.5 * e0, \
        f"training failed to converge (adjust LR={LR})"
    print("Self-test PASSED.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--sweep" in sys.argv:
        sweep(SEEDS[:5] if "--quick" in sys.argv else SEEDS)
    else:
        main(SEEDS[:5] if "--quick" in sys.argv else SEEDS)
