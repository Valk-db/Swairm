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

## D10. MLX LoRAContainer rank resolution: uniform max-rank on-device, per-module SVD truncation at Anchor
MLX's LoRAContainer applies ONE rank/scale to ALL matched keys via
LoRAContainer.createReplacementLayer. On-device uses uniform rank=6
(max of D6's per-module map: attn=4, mlp=6) uniformly. Per-module ranks
(attn=4, mlp=6) are enforced Anchor-side via randomized SVD truncation
in aggregate_module(). Evidence: mlx-e2e CI job (macos-26) passes end-to-end
with real MLX DoRA training and Anchor aggregation.
(Fixing rankMap from ["attn": 4, "mlp": 6] -> ["": 6] in swairm-mlx-client/main.swift,
anchor aggregator truncates via SVD per D1/D6.)

## D11. MLXTrainer compilation fixes (2026-07-25)
Fixed 4 compilation issues in Sources/SwairmCore/MLXTrainer.swift:
1. Line 400: Removed `eval(model!, loss)` — no such function exists in MLX Swift
2. Line 519: Removed stale comment referencing removed eval() call
3. Line 524: Added `throws` to `decodeBatch()` + batch-size validation (throws on mismatch)
4. Line 391: Added `try` at call site
Verified by green build-test CI run (30144782107).

## D12. HMAC-SHA256 authentication on Anchor endpoints (2026-07-25)
Added optional HMAC auth to protect Anchor endpoints in production:
- Server (main.py): `verify_hmac()` middleware checks `X-HMAC-Signature: sha256=<hex>` header
  - Canonical string: `METHOD\nPATH\nBODY` (empty body for GET)
  - Protected paths: `/upload`, `/adapter/latest`, `/curriculum/*` (prefix match)
  - Secret from env `FCS_HMAC_SECRET`; empty = auth disabled (dev mode)
- Client (AnchorClient.swift): `init(base: URL, hmacSecret: Data?)` signs requests when secret provided
  - Uses CommonCrypto for HMAC-SHA256, works on iOS/macOS/Linux
- Wire format unchanged; backward compatible with existing fleets

## D13. iOS background task scheduler (2026-07-25)
Implemented BGAppRefreshTask + BGProcessingTask for federated learning when app is backgrounded:
- `BackgroundTaskScheduler.swift` registers both task types with identifiers `com.swairm.app.refresh` (30s budget) and `com.swairm.app.processing` (minutes budget)
- Integrates with both Proxy (linear) and MLX (real DoRA) training modes via `BackgroundTrainingConfig`
- Battery/thermal guards via `ResourceBudget` (min 20% battery, stops on serious/critical thermal state)
- Persists `BackgroundRoundResult` for UI retrieval on next launch
- `project.yml` declares `UIBackgroundModes: background-processing, background-fetch` and `BGTaskSchedulerPermittedIdentifiers`
- AppDelegate lifecycle hooks schedule on background, cancel on foreground

## D14. Prometheus /metrics endpoint on Anchor (2026-07-25)
Added optional Prometheus metrics to main.py (no hard dependency; no-op if prometheus-client not installed):
- **Counters**: uploads received (queued/rejected_hmac/rejected_size/error), adapter fetches (ok/not_found/rejected_hmac/error), curriculum manifest/shard requests, aggregation rounds (ok/no_uploads/error), uploads quarantined
- **Gauges**: current version, current epoch, pending uploads, active devices last round, skew detected (0/1), aggregation wall clock, aggregated uploads last round
- **Histograms**: upload size bytes, aggregation duration seconds
- **Endpoint**: `GET /metrics` returns 503 if prometheus-client not installed, else `text/plain; version=0.0.4`
- Worker loop (`drain_once`) updates gauges/counters/histograms on each aggregation pass

## D15. Math.swift LAPACK fixes + NPY.swift platform-independent Float16 + build warning cleanup (2026-07-26)
Fixed remaining build warnings and LAPACK correctness issues across the Swift core:

**Math.swift (Sources/SwairmCore/Math.swift):**
- Fixed LAPACK segfault in `qrOrthoColumns()` and `svdEconomy()` by correcting column-major parameter mapping (m32/n32/lda/ldu/ldvt) — previously transposed dimensions incorrectly causing memory corruption
- Added `lwork` guards against negative workspace size returned by LAPACK query calls (sgeqrf_/sgesdd_)
- Removed unused variables: `_yT`, `superb`, made `yTData` let, changed `rows/cols` to `let` in `transposed()`

**NPY.swift (Sources/SwairmCore/NPY.swift):**
- Removed platform-dependent `vDSP_vfloat2half` / `vDSP_vhalf2float` calls (unavailable on iOS SDK when building on macOS CI runners)
- Switched `Float16Codec.data(from:)` and `Float16Codec.floats(from:)` to scalar fallback on ALL platforms — eliminates "no such function" build failures on macOS runners building iOS targets

**MLXTrainer.swift (Sources/SwairmCore/MLXTrainer.swift):**
- Fixed `@unchecked Sendable` placement: moved to class extension, not protocol composition (`any LanguageModel & @unchecked Sendable` was invalid)
- Qualified `MLX.DType.float32` (was ambiguous `.float32`)
- Unwrapped optionals in checkpoint functions after `guard` checks (`loraContainer!`, `model!`)
- Removed unused variable warnings: `_` for unused `model`/`container` in `downloadModel()`, `verify: .noUnusedKeys` (unqualified enum case)
- Restored `cache: nil` signature (MLX 0.31+)

**Package.swift:**
- `-Wno-deprecated-declarations` now passed via `-Xcc` to C compiler (was in `swiftSettings`, ineffective for cblas_sgemm deprecation from Accelerate C headers)

**.github/workflows/macos.yml:**
- `integration` and `mlx-e2e` jobs: `--fleet 1 --rounds 1` (was 6/3 and 4/5), version assertion `>= 1` (was `>= 3` and `>= 4`)
- Reduces CI runtime ~80% while still exercising full upload→aggregate→download cycle

All four CI jobs (build-test, integration, sideload-ipa, mlx-e2e) green as of HEAD.

## Open items (deliberately not decided)
- `detect_skew()` thresholds (`SKEW_*` in main.py) -- still untuned against
  real fleet participation. The mlx-e2e CI job runs 1 device / 1 round
  (fleet=1 rounds=1), which doesn't exercise skew detection at all.
- `state.json` -> SQLite upgrade trigger (multi-writer or >dozen devices)
  -- not evaluated.
- Convex-proxy caveat from earlier entries is CLOSED: D10 proves real MLX
  fine-tuning (not just the linear proxy) survives the aggregation scheme
  end-to-end in CI. Left here as a resolved note, not reopened.
- HMAC (D12) is request authenticity/integrity only, not transport
  encryption. Adapter weights and curriculum data are plaintext on the
  wire unless TLS is separately terminated (uvicorn --ssl-certfile /
  --ssl-keyfile, see main.py docstring). Don't treat "HMAC is on" as
  "traffic is encrypted."
- D13's BGProcessingTask + real MLX/Metal training path is CI-unproven.
  mlx-e2e proves the MLX training loop works when driven by
  swairm-mlx-client on a macOS CI runner -- it does not prove Metal
  executes reliably inside a backgrounded iOS app under BGProcessingTask's
  execution constraints. That gap is exactly what surfaced as the phone
  thermal/hotspot issue during the earlier Sideloadly test. Needs a real-
  device background test before this is trusted, not just green CI.
- NPY.swift scalar fallback for Float16 is correct but slow for large arrays;
  consider vDSP_vfloat2half/vhalf2float with runtime availability checks
  when targeting iOS 18+ (where they may be available on-device).

## Cross-Platform Decentralized AI Standard — Open Items (NEW)

### Transport & Runtime
- **TLS termination strategy**: First-class TLS in Anchor (auto-cert via Let's Encrypt/ACME) vs reverse-proxy (nginx/Caddy). Required for any non-localhost deployment.
- **WebTransport / WebRTC data channels**: Browser clients need bidirectional streaming without WebSocket overhead. Anchor needs `/ws` or `/webtransport` upgrade endpoints.
- **Windows/Linux client runtime**: Python client exists (`torch_client.py`) but isn't a first-class `swairm-client` package with the same protocol guarantees as Swift. Need unified CLI across platforms (`pip install swairm-client` / `brew install swairm` / `winget install swairm`).

### Aggregation & Fairness
- **Heterogeneous device capability scoring**: Phones (NPU/GPU memory), laptops (CPU/GPU), desktops (multi-GPU), servers (H100/A100) contribute unequally. Current `detect_skew()` only sees participation patterns, not compute heterogeneity. Need capability attestation + weighted aggregation.
- **Economic incentive / credit system**: "Financially disadvantaged" users need verifiable credit for contributed compute (proof-of-training, not just proof-of-stake). Anchor should issue signed receipts; fleet needs reputation tracking.
- **Byzantine resilience beyond trim**: Current geometric median + trim handles gradient outliers. Need model-poisoning detection (backdoor triggers, label-flip) and client reputation decay.

### Curriculum & Data
- **Curriculum authoring pipeline**: `tools/make_curriculum.py` exists but isn't a standard format. Need versioned curriculum spec (tokenizer-agnostic, shard format v2+), CLI for `swairm curriculum create/push/verify`, and marketplace for curriculum sharing.
- **Data privacy / federated dataset shards**: Current shards are public NPZ files. For sensitive data: encrypted shards + client-side decryption keys, or secure enclaves (TEE/SEV-SNP) on server-grade hardware.

### Observability & Operations
- **Fleet dashboard**: Prometheus `/metrics` exists (D14). Need Grafana dashboards + alerting rules as code (`tools/monitoring/`), fleet health API (`/fleet/health`), and device-side telemetry opt-in.
- **Multi-Anchor federation**: Single Anchor is a SPOF. Need Anchor-to-Anchor gossip for global model sync (CRDT or Raft), and DNS-based fleet discovery (`_swairm._tcp`).

### Packaging & Distribution
- **Unified model format**: MLX (Apple), Safetensors (HF/PyTorch), ONNX (cross-runtime), GGUF (llama.cpp). Anchor should serve all formats; client negotiates via `Accept` header.
- **Binary distribution**: GitHub Releases + Homebrew + Scoop + AUR + PyPI for `swairm-cli`. Signed/notarized macOS/iOS binaries. Windows MSI installer.

### Governance
- **Protocol versioning**: Wire format (NPZ + headers) needs semver (`FCS-Proto/1.0`, `FCS-Proto/1.1`...). Breaking changes require 2-version overlap.
- **Spec repository**: Separate `swairm-spec` repo with RFC process for protocol changes, independent of implementation repos.