# FCS Decision Log

Every entry: what was decided, and the evidence. Reopening a decision
requires new evidence, not re-argument.

## D1. Aggregation: reconstruct dense -> robust mean -> randomized SVD
Naive factor averaging (mean(B) @ mean(A)) has bilinear cross-term bias.
Verified: additive-noise clients, naive 0.8810 vs svd 0.8512 at het=4.0;
rotated-subspace clients, naive stuck at 0.7071 regardless of target rank
(4/8/16 identical) while reconstruct+SVD reaches 0.0000 at rank 8.
(simulate_fedavg.py v2)

## D2. Staleness weighting: CONDITIONAL -- INVERTED by non-convex evidence
Original (linear proxy): balanced -> uniform, skew -> reciprocal.
Under real gradient training with replace semantics (D7), the sign flips
both ways (harness_nonconvex.py sweep, seeds 42-71):
- No demographic skew -> reciprocal 1/(1+s). Stale uploads are
  less-converged full adapters that drag the replace-average backward;
  reciprocal wins 26/30, 29/30, 30/30 seeds at fetch_every=3/6/9
  (t=+2.99/+7.85/+11.57). Zero-staleness fleets unaffected (weights
  collapse to uniform).
- Detected demographic skew -> UNIFORM. Reciprocal persistently
  under-weights the stale population, under-fitting its share of the
  fleet objective (uniform wins at group_het>=1.2, fetch_every<=6,
  t up to -7.62). CAVEAT: sign crosses over when staleness is deep
  relative to divergence (fetch_every=9, het=0.6: reciprocal t=+9.44);
  all skewed-cell gaps are <~5% relative. Uniform chosen under skew:
  symmetric MSE risk, full work retention, no systematic demographic
  under-representation.
- detect_skew() (main.py v1.2, device-set Jaccard night/day) unchanged;
  only the weighting it triggers is inverted. Thresholds still UNTUNED.

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

## D8. Non-convex validation: proxy caveat narrowed
harness_nonconvex.py: frozen 2-layer tanh MLP; clients train true DoRA
adapters (A, B, m) with manual-gradient SGD; staleness = genuinely
training from an old global; aggregation through the PRODUCTION
aggregate_round() path. Findings (seeds 42-71, paired t):
- D2 conditional inverted (see revised D2).
- TRIM_BEFORE_WEIGHTS: no robust difference. Sign flips with staleness
  depth (False wins 27/30 t=+5.48 at fetch_every=4; True wins 26/30
  t=-5.13 at fetch_every=6), effect <1% relative either way. Default
  stays True. CLOSED.
- Epoch handling: hard rejection beats soft (w=0.3/0.5) on final err
  AND transition-dip depth in every run. Hard rejection LOCKED.
Remaining: still a proxy (small MLP, synthetic teachers); real MLX
on-device training is the final arbiter.

## Open items (deliberately not decided)
- Skew-detector threshold tuning (SKEW_JACCARD_THRESHOLD et al.; needs
  real fleet participation data)
- D2 crossover: deep staleness + mild skew favors reciprocal; revisit
  the inverted conditional if real fleets live in that regime
- state.json -> SQLite upgrade trigger (multi-writer or >dozen devices)
- Proxy caveat, narrowed by D8: only real MLX on-device training
  closes it

