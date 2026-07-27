# Swairm / FCS — TODO

Generated 2026-07-27 from a full pass over github.com/Valk-db/Swairm
(Python Anchor + Swift/MLX client). Ordered by priority, not by size —
some of these are five-minute fixes, some aren't.

## Critical — before anything else

- [x] ~~Prove real on-device MLX training survives a backgrounded,
  thermally-realistic phone.~~ **Superseded 2026-07-27**: background
  execution is being dropped entirely. Devices are now dedicated —
  foreground only, screen on, no other apps — for maximum sustained
  compute. Replaces the BGProcessingTask question with the item below.

- [ ] **Characterize sustained foreground max-compute thermal/power
  behavior.** Screen-on + Metal at full tilt continuously is a real,
  sustained power draw — realistically this means devices stay on wall
  power indefinitely, which is close to the exact charging+heat
  conditions that caused the original Sideloadly overheating issue, just
  deliberate now instead of incidental. Run a real device screen-on,
  plugged in, no other apps, for a few hours and watch
  `ProcessInfo.thermalState`. If it climbs into `.serious`, iOS throttles
  the chip on its own regardless of what the app does — so thermal-aware
  pacing (below) isn't optional even in foreground mode, it's *how* you
  sustain peak throughput instead of getting silently clocked down.
- [ ] **Remove the BGTaskScheduler plumbing.**
  `BackgroundTaskScheduler.swift`'s BGAppRefreshTask/BGProcessingTask
  registration, the `UIBackgroundModes` (`background-processing`,
  `background-fetch`) + `BGTaskSchedulerPermittedIdentifiers` entries in
  `App/project.yml`, and the AppDelegate background/foreground scheduling
  hooks are all dead weight now.
- [ ] **Add `UIApplication.shared.isIdleTimerDisabled = true`** while
  training is active so the screen doesn't auto-lock; reset to `false`
  when training stops.
- [ ] **Rework the thermal/battery guard from "abort task" to "pace the
  loop."** `ResourceBudget.stopOnSeriousThermalState` (used in
  `LinearProxyTrainer`/`MLXTrainer`) was written for "should I even
  attempt this background task." For a continuous foreground loop,
  reframe it as "pause briefly / drop batch size approaching `.serious`,
  resume when cooler" so an unattended long run degrades gracefully
  instead of just stopping.
- [ ] **Decide on Guided Access for "no other apps."** iOS's Guided
  Access (triple-click side button) physically locks the device to one
  app without any code changes or MDM enrollment — worth using if you
  want that constraint enforced by the OS rather than by whoever's in
  the room.
- [ ] **Decide auto-start vs. manual start.** If these are dedicated,
  unattended devices, does the app need to start training on launch
  (nobody's there to tap Start), or is manual start still fine?

- [x] **Fix or explicitly shelve the Let's Encrypt path in `cert_manager.py`.**
  Nothing in the repo serves `/.well-known/acme-challenge/<token>`.
  `request_letsencrypt_cert()` calls `acme_client.answer_challenge()`
  telling Let's Encrypt "come verify me," but there's nothing to verify
  against — it will fail at `acme_client.poll(authz)`.
  Pick one: (a) add a route in `main.py` serving the key authorization
  for the live challenge, or (b) put the Anchor behind Caddy/nginx and
  let *it* handle ACME, and update the module docstring + DECISIONS.md
  to say standalone Let's Encrypt mode isn't functional yet. Self-signed
  mode is unaffected either way.

## Security hardening

- [ ] **Make HMAC protection structural, not manual.** `HMAC_PROTECTED_PATHS`
  (top of `main.py`) is defined but never read anywhere — enforcement is
  six separate `if not verify_hmac(...)` calls pasted into individual
  handlers. Convert to a FastAPI dependency (`Depends(require_hmac)`) or
  middleware actually driven by a path list, so a new endpoint is
  protected by default instead of by remembering to paste the check in.
- [ ] **Cover `/ws/{client_id}` with the same auth.** Currently accepts any
  connection unauthenticated and broadcasts `NEW_VERSION:{version}` to
  whoever's listening. Low value leak today, but it's the one gap in an
  otherwise-consistent scheme.
- [ ] **Add replay protection to the HMAC scheme.** Canonical string is
  `METHOD\nPATH\nBODY` — no timestamp, no nonce. A captured valid request
  can be replayed indefinitely. Add a timestamp header + short validity
  window (e.g. reject if `abs(now - ts) > 300s`) and fold the timestamp
  into the signed string, on both the Python and Swift sides.
- [ ] **Fix `/upload`'s size-check ordering.** `raw = await request.body()`
  fully buffers the payload before `len(raw) > MAX_UPLOAD_BYTES` is ever
  checked — the memory cost already happened by the time you reject.
  Check `Content-Length` up front, or stream with a hard cutoff.
- [ ] **Add the same path-traversal guard to `model_name`** in
  `/models/base/{model_name}/manifest` and `/models/base/{model_name}/{file_name}`
  that `file_name` / `shard_name` already get elsewhere in `main.py`.
- [ ] **Move the ACME calls off the event loop.** `request_letsencrypt_cert()`
  and the 12-hour renewal loop make synchronous `acme`-library calls
  inside `async def` functions — a real renewal will stall the *entire*
  Anchor (every phone's upload/fetch/status) for the duration of the
  handshake. Wrap in `asyncio.to_thread(...)`.

## CI & build reliability

- [ ] **Commit `Package.resolved`.** `mlx-swift` currently floats on
  `from: "0.10.0"`, which — per your own CI comment — let the resolver
  silently jump to 0.31.6 with a completely different vendored kernel
  set. Pinning gets you reproducible builds and removes a whole class of
  "nothing changed but it broke" bugs.
- [ ] **Stop unconditionally clearing the SwiftPM cache** in every job
  (`build-test`, `integration`, `mlx-e2e`, `sideload-ipa` all run
  `rm -rf .build ~/Library/Caches/org.swift.swiftpm`). Once
  `Package.resolved` is pinned, try a run without the clear step — the
  clearing was likely a defensive workaround for the version-drift
  problem above, which pinning solves more directly and for free.
- [ ] **Add a slower "real fleet" CI job**, separate from the fast
  per-push smoke test — e.g. `fleet=12 rounds=3` on a nightly/weekly
  schedule instead of every push. Today's `fleet=1 rounds=1` means skew
  detection and trimming never get exercised over the *live wire path*,
  only inside `main.py --selftest`'s pure-Python synthetic version.

## Testing rigor

- [ ] **Convert `aggregator.py`'s `__main__` self-test from prints to
  asserts.** It already has the right scenario (13 synthetic uploads:
  outlier, stale, wrong-epoch) and the right expected values right there
  in the docstring ("trailing_ratio ~0.064/~0.067, m mean ~1.559") — just
  needs `assert abs(x - expected) < tol` instead of `print(...)` so a
  regression fails CI instead of requiring a human to notice a drifted
  number.
- [ ] **Consider direct, shape-sweeping unit tests for `qrOrthoColumns` /
  `svdEconomy`** in isolation — checking `Q^T Q ≈ I` and
  `U·diag(S)·Vt ≈ A` across a few tall/wide/square shapes — separate from
  the existing `testTruncatedSVDReconstructsLowRankMatrix`. Coverage at
  the `truncatedSVD` level is already good; this would pinpoint which
  internal step broke if a future refactor reintroduces a
  transposed-dimension bug, given this exact code has already had
  multiple segfault-class fixes.

## Code quality / cleanup

- [ ] Extract the repeated path-traversal check (`".." in x or "/" in x or
  "\\" in x`) into one shared helper instead of copy-pasting it across
  `shard_name` / `file_name` / (new) `model_name` checks.
- [ ] Extract the duplicated manifest-generation logic (sha256 + size per
  expected file) shared between `_prepare_base_model()` and the
  `/models/base/{model_name}/manifest` endpoint.
- [ ] Raise the Prometheus `AGGREGATION_DURATION_SECONDS` histogram's top
  bucket (currently 300s) to cover D5's own worst-case benchmark number
  (803s with the 4x Anchor safety factor) — right now a slow-but-real
  round just falls into `+Inf` with no granularity on how far over
  budget it actually was.

## Scope / product judgment

- [ ] **Hold off on the new "Cross-Platform Decentralized AI Standard"
  wishlist in DECISIONS.md** (WebTransport, economic incentives,
  multi-Anchor gossip, five package managers, an RFC process). It's a
  legitimate brainstorm but it's a multi-year roadmap, not a next step —
  and it landed in the same file that otherwise enforces "decide with
  evidence, don't re-litigate." Revisit once the Critical items above
  are actually resolved.