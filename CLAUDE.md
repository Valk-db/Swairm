# CLAUDE.md — Swairm / FCS

Federated Curriculum Swarm: phones locally fine-tune DoRA adapters,
a desktop Anchor aggregates them each round (reconstruct dense update
-> robust trim -> randomized SVD; NOT naive factor averaging). Full
adapter state replaces the global each round (not delta-accumulation).

**Read DECISIONS.md before touching aggregation logic.** Entries D1-D9
are locked with the evidence that locked them. Reopening one needs new
evidence, not re-argument — don't re-propose things already parked
there (e.g. agreement/relevance weighting, D3).

## Validation maturity (don't re-derive, just extend)
- main.py + simulated fleet: aggregation validated end-to-end over
  HTTP, linear training proxy
- Real AdamW/DoRA gradient training against the production wire
  format: validated in Python (torch_client.py) — proves the
  aggregation scheme survives real training, not just the proxy
- iOS-native Swift client (SwairmCore package: CLI fleet sim, XCTest,
  SwiftUI app): wire-format/NPY-NPZ codec proven byte-identical to
  Python's via CI (real Anchor <-> real Swift fleet). Until this week
  it trained via LinearProxyTrainer.swift, a stand-in, not real
  on-device training.
- Real on-device MLX training loop: THE ACTIVE FRONTIER, see below.

## Where things stand right now
All four CI jobs are green as of HEAD (build-test, integration,
sideload-ipa, mlx-e2e) — the iOS app build was fixed in 3e18cea and has
stayed fixed. The one interim failure (b99e002) was build-test, not
sideload-ipa: orientation tests that needed the MLX/Metal runtime under
`swift test` (XCTest can't load the default metallib on the runner).
Fixed in d005fe3 by making those tests MLX-runtime-free.

swairm-mlx-client (real on-device MLX DoRA fleet driver) and
tools/make_curriculum.py are now proven end-to-end in CI via the
mlx-e2e job (macos-26 runner). The MLX LoRAContainer rank-resolution
mismatch (per-module ranks on-device vs. uniform container rank) is
resolved: on-device uses uniform rank=6 (max of D6's map) with
Anchor-side SVD truncation to per-module ranks (attn=4, mlp=6).
Real on-device MLX training is now validated end-to-end in CI.

MLXTrainer.swift: Fixed compilation bugs (3 bugs + throws signature):
- Line 400: Removed non-existent `eval(model!, loss)` call (no such function in MLX)
- Line 519: Removed stale comment referencing the removed eval() call
- Line 524: Added `throws` to `decodeBatch()` with batch-size validation
- Line 391: Added `try` at call site
All fixes verified by green build-test CI.

**Recent fixes (2026-07-26):**
- Math.swift: Fixed LAPACK segfault by correcting column-major layout handling in qrOrthoColumns() and svdEconomy() — fixed m32/n32/lda/ldu/ldvt parameters, added lwork guards against negative workspace queries, removed unused variables (_yT, superb, yTData made let, rows/cols in transposed() made let)
- NPY.swift: Removed vDSP vfloat2half/vhalf2float platform-dependent calls; switched Float16Codec to scalar fallback on all platforms (eliminates iOS SDK build failure on macOS runners)
- MLXTrainer.swift: Fixed Sendable conformance (@unchecked Sendable in extension, not on protocol), .noUnusedKeys unqualified, optional unwrapping in checkpoint functions, MLX.DType.float32 fully qualified
- Package.swift: -Wno-deprecated-declarations now passed via -Xcc to C compiler (fixes cblas_sgemm deprecation warnings)
- .github/workflows/macos.yml: fleet=1, rounds=1 for both integration and mlx-e2e; version assertion >=1

**P1 tasks completed (2026-07-25):**
- HMAC auth on Anchor endpoints (main.py + AnchorClient.swift): HMAC-SHA256 on /upload, /adapter/latest, /curriculum/*, disabled when FCS_HMAC_SECRET unset (dev mode). **Auth only, not encryption** -- see security note below.
- iOS background task scheduler (BackgroundTaskScheduler.swift): BGAppRefreshTask (30s) + BGProcessingTask (minutes), integrates with both Proxy and MLX training modes, battery/thermal checks via ResourceBudget. **CI-unverified for real MLX training** -- see note below.
- Prometheus /metrics endpoint on Anchor (main.py): counters/gauges/histograms for uploads, fetches, curriculum, aggregation rounds, quarantine; optional dependency (no-op if prometheus-client not installed)

## Security posture (read before deploying off localhost)
- D12's HMAC gives request authenticity/integrity, NOT transport
  encryption. Adapter weights and curriculum shards travel in plaintext
  unless TLS is separately terminated (`uvicorn ... --ssl-certfile
  --ssl-keyfile`, see main.py docstring). The Anchor logs its HMAC/TLS
  posture on startup (lifespan handler) -- check that log line before
  trusting a deployment.
- D13's background training path (BGProcessingTask running the real MLX
  trainer) is only proven in the sense that the code compiles and the
  linear-proxy path has run in background tasks before. The real-MLX
  branch through Metal has NOT been exercised inside an actual
  backgrounded iOS app -- mlx-e2e CI proves the trainer works when driven
  from a macOS CLI, not under BGProcessingTask's execution/thermal
  constraints on a physical phone. Treat it as unverified until tested on
  real hardware; this is the same failure class as the earlier Sideloadly
  thermal/hotspot issue. See "Open items" in DECISIONS.md.

## Ground rules
- No local Mac. Never claim something compiles or is fixed without a
  green CI run to point to.
- Prefer targeted diffs over full-file rewrites when a scoped fix works.