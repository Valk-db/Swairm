"""
async_event_sim.py  (v1.1)
==========================

Event-driven (minute-by-minute) system-dynamics simulator for the Federated
Curriculum Swarm (FCS). Companion to simulate_fedavg.py (v2), which validates
single-round aggregation *mechanics*; this file validates *asynchronous
dynamics*: version skew, staleness weighting, circadian participation,
thermal interruptions, and per-cluster retention.

Primary question answered (flagged open in prior review):
    Does staleness down-weighting (e.g. 1/(1+s)) actually improve
    convergence, or does it just discard useful client work?

v1.1 changes (per review):
  1. Cluster offsets are now exact mirrors (+off / -off from one draw), so
     any w_cl0 / w_cl1 asymmetry is purely dynamics-driven, not offset-norm
     luck.
  2. Seed sweep (N_SEEDS runs per policy); results reported mean +/- std.
     One seed is an anecdote; do not change the roadmap on an anecdote.
  3. Second scenario, "schedule_skew": cluster 0 charges at night, cluster 1
     during the day. Directly tests the cold-start EMA-crystallization
     hypothesis (agreement baseline forming around whichever demographic
     uploads first, then starving the other).

Policies compared (identical world per seed across policies):
    uniform         w = 1
    reciprocal      w = 1/(1+s)
    bounded_exp     w = exp(-lambda * s)
    velocity_aware  w = exp(-lambda * s * v)   v = normalized EMA(||applied||)
    relevance_full  velocity_aware * agreement(cosine vs EMA of global delta),
                    with cold-start blending (disabled -> blended -> full)

Assumptions (best-guess where spec describes behavior, not formulas):
  - Client "training" is a linear, convex proxy for SGD fine-tuning:
    delta_k = CLIENT_LR * (T_k - G_at_fetch) + noise, truncated to adapter
    rank. Standard FL-simulator practice; fine for skew/weighting mechanics,
    but real non-convex landscapes may behave less smoothly. Policy rankings
    here are necessary evidence, not sufficient.
  - Staleness s = global versions elapsed since the client fetched.
  - No lease loss / device death / upload corruption modeled yet (v2 work).
  - Single module; DoRA magnitude vector m not modeled here.

Pre-registered null outcome: with 30-min flushes and ~45-90-min jobs, mean
staleness may be only 1-3 versions. If all policies tie, that is a finding
("at family scale, staleness weighting barely matters"), not a failure --
and it argues for keeping the simplest policy.

Run: python3 async_event_sim.py
Outputs: summary tables to stdout; async_event_sim.png if matplotlib is
available (script runs fine without it).
"""

import numpy as np
from sklearn.utils.extmath import randomized_svd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ---------------------------------------------------------------- constants
M, N, RANK = 128, 256, 4
N_CLIENTS = 20
SIM_DAYS = 7
SIM_MINUTES = SIM_DAYS * 24 * 60
AGG_INTERVAL_MIN = 30          # anchor flushes its pending buffer this often
CLIENT_LR = 0.6
SERVER_LR = 0.5
TRAIN_MINUTES_MEAN = 45        # active training minutes per job
THERMAL_PAUSE_PROB = 0.02      # per active minute
THERMAL_PAUSE_RANGE = (10, 30)
START_PROB_WHEN_CHARGING = 0.10  # per idle minute on charger
LAMBDA_DECAY = 0.5
AGREEMENT_COLD_START_N = 8     # flushes before agreement is fully enabled
BASE_SEED = 42
N_SEEDS = 5
POLICIES = ["uniform", "reciprocal", "bounded_exp",
            "velocity_aware", "relevance_full"]
SCENARIOS = ["balanced", "schedule_skew"]


def rel_frobenius_error(estimate, ground_truth):
    return np.linalg.norm(estimate - ground_truth, "fro") / np.linalg.norm(ground_truth, "fro")


def truncate_rank(dense, rank):
    U, S, Vt = randomized_svd(dense, n_components=rank + 2, random_state=42)
    return (U[:, :rank] * S[:rank]) @ Vt[:rank]


def low_rank_matrix(rng, rank, scale=1.0):
    B = rng.standard_normal((M, rank)) / np.sqrt(rank)
    A = rng.standard_normal((rank, N)) / np.sqrt(N)
    return scale * (B @ A)


def p_charging(hour, cluster, scenario):
    """
    Diurnal charging probability.
      balanced:       both clusters share one profile (regular overnight,
                      erratic daytime) -- any w_cl asymmetry is small-sample
                      luck in upload order, not schedule demographics.
      schedule_skew:  cluster 0 = night chargers, cluster 1 = day chargers --
                      the EMA-crystallization trap: the agreement baseline
                      can form around night-cluster uploads before the day
                      cluster ever participates.
    """
    night = (hour >= 22 or hour < 7)
    day = (9 <= hour < 17)
    if scenario == "balanced":
        return 0.90 if night else 0.25
    if scenario == "schedule_skew":
        if cluster == 0:
            return 0.90 if night else 0.10
        return 0.90 if day else 0.10
    raise ValueError(scenario)


# ---------------------------------------------------------------- client
class Client:
    IDLE, TRAINING, PAUSED = 0, 1, 2

    def __init__(self, cid, cluster, t_local, rng):
        self.cid = cid
        self.cluster = cluster
        self.t_local = t_local            # this client's non-IID view of target
        self.rng = rng
        self.state = Client.IDLE
        self.remaining = 0.0              # active minutes left in current job
        self.pause_left = 0
        self.fetch_version = None
        self.g_snapshot = None
        self.minutes_spent = 0.0

    def step(self, hour, anchor, scenario):
        charging = self.rng.random() < p_charging(hour, self.cluster, scenario)
        if self.state == Client.IDLE:
            if charging and self.rng.random() < START_PROB_WHEN_CHARGING:
                self.fetch_version = anchor.version
                self.g_snapshot = anchor.G.copy()
                self.remaining = max(10, self.rng.normal(TRAIN_MINUTES_MEAN, 15))
                self.minutes_spent = 0.0
                self.state = Client.TRAINING
        elif self.state == Client.PAUSED:
            self.pause_left -= 1
            if self.pause_left <= 0 and charging:
                self.state = Client.TRAINING     # resume from checkpoint
        elif self.state == Client.TRAINING:
            if not charging:
                self.state = Client.PAUSED       # checkpoint, wait for charger
                self.pause_left = 1
                return
            if self.rng.random() < THERMAL_PAUSE_PROB:
                self.state = Client.PAUSED       # thermal .serious -> sleep
                self.pause_left = int(self.rng.integers(*THERMAL_PAUSE_RANGE))
                return
            self.remaining -= 1
            self.minutes_spent += 1
            if self.remaining <= 0:
                delta = CLIENT_LR * (self.t_local - self.g_snapshot)
                delta += 0.05 * self.rng.standard_normal((M, N)) / np.sqrt(N)
                delta = truncate_rank(delta, RANK)   # adapter-rank constraint
                anchor.receive(dict(cid=self.cid, cluster=self.cluster,
                                    delta=delta,
                                    fetch_version=self.fetch_version,
                                    minutes=self.minutes_spent))
                self.state = Client.IDLE


# ---------------------------------------------------------------- anchor
class Anchor:
    def __init__(self, policy_name):
        self.G = np.zeros((M, N))
        self.version = 0
        self.buffer = []
        self.policy_name = policy_name
        self.velocity_ema = None      # EMA of ||applied delta||
        self.velocity_ref = None
        self.delta_ema = None         # EMA of applied delta direction
        self.n_flushes = 0
        # telemetry
        self.errors, self.error_minutes = [], []
        self.weights_log = []         # (cluster, weight, staleness, minutes)

    def receive(self, upload):
        self.buffer.append(upload)

    def _velocity(self):
        if self.velocity_ema is None:
            return 1.0
        return self.velocity_ema / (self.velocity_ref + 1e-12)

    def _weight(self, upload):
        s = self.version - upload["fetch_version"]
        p = self.policy_name
        if p == "uniform":
            w = 1.0
        elif p == "reciprocal":
            w = 1.0 / (1.0 + s)
        elif p == "bounded_exp":
            w = float(np.exp(-LAMBDA_DECAY * s))
        elif p in ("velocity_aware", "relevance_full"):
            w = float(np.exp(-LAMBDA_DECAY * s * self._velocity()))
        else:
            raise ValueError(p)
        if p == "relevance_full" and self.delta_ema is not None:
            cos = float(np.sum(upload["delta"] * self.delta_ema) /
                        (np.linalg.norm(upload["delta"]) *
                         np.linalg.norm(self.delta_ema) + 1e-12))
            agree = float(np.clip(0.5 + 0.5 * cos, 0.0, 1.0))
            # cold-start blending: disabled -> blended -> fully enabled
            blend = float(np.clip(self.n_flushes / AGREEMENT_COLD_START_N, 0.0, 1.0))
            w *= (1.0 - blend) + blend * agree
        return w, s

    def flush(self, minute, target):
        if not self.buffer:
            return
        deltas, weights = [], []
        for u in self.buffer:
            w, s = self._weight(u)
            self.weights_log.append((u["cluster"], w, s, u["minutes"]))
            deltas.append(u["delta"])
            weights.append(w)
        self.buffer = []
        wsum = float(np.sum(weights))
        if wsum > 1e-9:
            agg = np.tensordot(np.array(weights) / wsum, np.array(deltas), axes=1)
            applied = SERVER_LR * agg
            self.G = truncate_rank(self.G + applied, RANK)
            norm = np.linalg.norm(applied)
            if self.velocity_ema is None:
                self.velocity_ema = norm
                self.velocity_ref = norm
                self.delta_ema = applied.copy()
            else:
                self.velocity_ema = 0.25 * norm + 0.75 * self.velocity_ema
                self.velocity_ref = 0.02 * norm + 0.98 * self.velocity_ref
                self.delta_ema = 0.25 * applied + 0.75 * self.delta_ema
            self.version += 1
        self.n_flushes += 1
        self.errors.append(rel_frobenius_error(self.G, target))
        self.error_minutes.append(minute)


# ---------------------------------------------------------------- one run
def run_policy(policy_name, seed, scenario):
    rng = np.random.default_rng(seed)          # identical world per policy
    target_shared = low_rank_matrix(rng, RANK)
    off = low_rank_matrix(rng, 1, scale=0.6)
    offsets = [off, -off]                      # exact antisymmetric clusters
    clients = []
    for cid in range(N_CLIENTS):
        cluster = cid % 2
        t_local = (target_shared + offsets[cluster]
                   + 0.15 * low_rank_matrix(np.random.default_rng(seed + 100 + cid), 2))
        clients.append(Client(cid, cluster, t_local,
                              np.random.default_rng(seed + 1000 + cid)))
    # ground truth = fleet mean of local targets (what ideal aggregation seeks)
    target = np.mean([c.t_local for c in clients], axis=0)
    anchor = Anchor(policy_name)
    for minute in range(SIM_MINUTES):
        hour = (minute // 60) % 24
        for c in clients:
            c.step(hour, anchor, scenario)
        if minute % AGG_INTERVAL_MIN == 0:
            anchor.flush(minute, target)
    return anchor


def summarize(anchor):
    wl = anchor.weights_log
    tot_min = sum(x[3] for x in wl)
    kept_min = sum(x[1] * x[3] for x in wl)
    stal = [x[2] for x in wl]
    per_cluster = {}
    for cl in (0, 1):
        ws = [x[1] for x in wl if x[0] == cl]
        per_cluster[cl] = float(np.mean(ws)) if ws else float("nan")
    return dict(final_err=anchor.errors[-1] if anchor.errors else float("nan"),
                best_err=min(anchor.errors) if anchor.errors else float("nan"),
                retention=kept_min / max(tot_min, 1e-9),
                mean_staleness=float(np.mean(stal)) if stal else float("nan"),
                max_staleness=max(stal) if stal else 0,
                cluster0_w=per_cluster[0], cluster1_w=per_cluster[1],
                n_uploads=len(wl))


if __name__ == "__main__":
    seeds = [BASE_SEED + i for i in range(N_SEEDS)]
    plot_runs = {}   # (scenario, policy) -> anchor from first seed, for plotting

    for scenario in SCENARIOS:
        print(f"\n=== scenario: {scenario} | {SIM_DAYS}d, {N_CLIENTS} clients, "
              f"flush every {AGG_INTERVAL_MIN}min, {N_SEEDS} seeds ===\n")
        print(f"{'policy':16s} {'final_err':>15s} {'retention':>15s} "
              f"{'mean_s':>7s} {'max_s':>6s} {'w_cl0':>7s} {'w_cl1':>7s} {'uploads':>8s}")
        for p in POLICIES:
            runs = []
            for seed in seeds:
                a = run_policy(p, seed, scenario)
                if seed == seeds[0]:
                    plot_runs[(scenario, p)] = a
                runs.append(summarize(a))
            def agg(key):
                vals = np.array([r[key] for r in runs], dtype=float)
                return vals.mean(), vals.std()
            fe_m, fe_s = agg("final_err")
            rt_m, rt_s = agg("retention")
            ms_m, _ = agg("mean_staleness")
            mx = max(r["max_staleness"] for r in runs)
            c0_m, _ = agg("cluster0_w")
            c1_m, _ = agg("cluster1_w")
            nu_m, _ = agg("n_uploads")
            print(f"{p:16s} {fe_m:7.4f}+/-{fe_s:5.4f} {rt_m:7.3f}+/-{rt_s:5.3f} "
                  f"{ms_m:7.2f} {mx:6d} {c0_m:7.3f} {c1_m:7.3f} {nu_m:8.0f}")

    print("\nInterpretation guide:")
    print("  - uniform vs reciprocal closes the open question: if reciprocal's")
    print("    final_err is not better (beyond the +/- spread), staleness decay")
    print("    is just discarding work -- check retention for the cost.")
    print("  - w_cl0 vs w_cl1 under relevance_full, balanced scenario: clusters")
    print("    are exact mirrors, so any persistent asymmetry is the agreement")
    print("    metric suppressing valid diversity (feedback loop), not the data.")
    print("  - schedule_skew scenario: if relevance_full's w_cl1 (day cluster)")
    print("    is depressed relative to balanced, the cold-start EMA crystallized")
    print("    around the night cluster -- the crystallization hypothesis holds.")
    print("  - If everything ties within spread: staleness weighting doesn't")
    print("    matter at this scale. That's a finding, not a failure -- keep the")
    print("    simplest policy.")

    if HAVE_MPL:
        fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(13, 4.8), sharey=True)
        for ax, scenario in zip(np.atleast_1d(axes), SCENARIOS):
            for p in POLICIES:
                a = plot_runs[(scenario, p)]
                ax.plot(np.array(a.error_minutes) / 60.0, a.errors,
                        label=p, linewidth=1)
            ax.set_xlabel("simulated hours")
            ax.set_title(f"scenario: {scenario}  (seed {seeds[0]})")
            ax.legend(fontsize=8)
        np.atleast_1d(axes)[0].set_ylabel("rel. Frobenius error vs fleet-mean target")
        plt.suptitle("Convergence under asynchronous participation, by staleness policy")
        plt.tight_layout()
        plt.savefig("async_event_sim.png", dpi=140)
        print("\nPlot written: async_event_sim.png")
    else:
        print("\n(matplotlib not installed -- skipped plot; tables above are complete."
              " `pip install matplotlib` to get async_event_sim.png)")