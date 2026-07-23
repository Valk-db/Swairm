# FCS Decision Log

Every entry: what was decided, and the evidence. Reopening a decision
requires new evidence, not re-argument.

## D1. Aggregation: reconstruct dense -> robust mean -> randomized SVD
Naive factor averaging (mean(B) @ mean(A)) has bilinear cross-term bias.
Verified: additive-noise clients, naive 0.8810 vs svd 0.8512 at het=4.0;
rotated-subspace clients, naive stuck at 0.7071 regardless of target rank
(4/8/16 identical) while reconstruct+SVD reaches 0.0000 at rank 8.
(simulate_fedavg.py v2)

## D2. Staleness weighting: CONDITIONAL
- Balanced participation -> uniform weights. Evidence: reciprocal 1/(1+s)
  gave no error benefit (0.1716 vs 0.1632) while discarding 75% of client
  work (retention 0.249). (async_events_system.py, 5 seeds)
- Detected participation skew -> reciprocal 1/(1+s). Evidence: won 15/15
  fresh seeds (57-71), paired t=6.07, mean gap 0.145 (~32% error reduction).
  (followup confirmation run)
- Skew detection wired in main.py v1.2: device-composition Jaccard between
  night and day windows (volume alone can't distinguish balanced-but-diurnal
  from demographic skew). Thresholds are heuristics pending real fleet data.

## D3. Agreement/relevance weighting: PARKED
Three reasons, all measured: (1) cluster-weight asymmetry with direction
flipping per seed on exact-mirror clusters (mean |gap| 0.041 vs ~0.015 for
staleness-only policies); (2) catastrophic-seed collapse (final_err 0.7357
vs ~0.15 baseline, seed 45); (3) worst retention (0.170). Do not
reintroduce without cluster-tagged instrumentation in the real fleet.

## D4. DoRA magnitude m: aggregated SEPARATELY, kept as first-class
Rule 1 verified on the production aggregate_module path: separate-m cuts
error 53% vs frozen m=1 at het=1.0 (0.2572 vs 0.5517). Folded alternative
won 4/4 levels but only by 2-5% relative -- rejected because it destroys m
as a client adapter component, breaking DoRA training semantics fleet-wide.
(validate_magnitude.py)

## D5. Performance: float32 + partition-based trim
Real-shape benchmark (Qwen2.5-1.5B modules, 196-module round): float64 +
double-argsort trim = 448s local / 1792s with 4x Anchor safety factor
(FAILED 15-min budget, 1.3GB cohorts). v1.1 (float32, partition trim) =
201s / 803s (PASS), behavior-identical to float32 precision. Escape hatch
if real Anchor is still too slow: factored SVD path -- but it sacrifices
coordinate-wise trimming; human decision required. (benchmark_real_scale.py)

## D6. Adaptive rank: kv=2, attn=4, mlp=6
Rank-starvation diagnostic (trailing singular-value ratio) separates
true-rank<=4 from >=6 cleanly (0.064 vs 0.408); threshold 0.15 uses the
LINEAR-sum convention -- re-tune if convention changes to squared.

## D7. Upload semantics: full-adapter FedAvg (replace, not delta-accumulate)
Clients upload FULL post-training adapter state; each round's aggregate
REPLACES the global. Pinned by swarm_client.py; consistent with all prior
validation. Verified end-to-end over HTTP: 12-client fleet, 10 rounds,
dir err 1.0 -> 0.0839 (noise floor), magnitude err 0.0064, monotonic
convergence, no oscillation. Revisit only if delta-accumulation semantics
are needed for real MLX training.

## D8. Trim/weight composition order: TRIM_BEFORE_WEIGHTS = True, LOCKED
Trim on raw values first, then weight survivors. Won 20/20 paired seeds
under active reciprocal weighting (paired t=13.4). Weighting first lets
down-weighted stale values masquerade as extremes, trimming the wrong
coordinates. Flag retained in aggregator.py only for reproducing the
experiment. (validate_open_configs.py, exp 1)

## D9. Curriculum-epoch handling: soft one-step transition weight 0.25
Hard rejection LOSES to soft transition weights in every tested regime
(shift 0.3/1.0 x n_current 2/4/8, t=+21..+100). main.py passes a one-step
soft map {(epoch-1, epoch): EPOCH_TRANSITION_WEIGHT=0.25}; epochs older
than one step remain hard-rejected. Env override:
FCS_EPOCH_TRANSITION_WEIGHT. (validate_open_configs.py, exp 2)

## Open items (deliberately not decided)
- detect_skew() thresholds (SKEW_* in main.py) -- heuristics untuned
  until real fleet participation data exists
- state.json -> SQLite upgrade trigger (multi-writer or >dozen devices)
- Convex-proxy caveat: all simulation evidence assumes a linear training
  proxy; real MLX fine-tuning may punish staleness differently. Only real
  on-device training closes this.
