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
MLXTrainer.swift was just refactored to use MLX's built-in
LoRAContainer/DoRA infra (was hand-rolled before). Current CI
(.github/workflows/macos.yml — no local Mac, this is the only build
oracle):
- `build-test` (swift build + swift test, SwairmCore package): passing
- `integration` (real Anchor + real Swift fleet over HTTP): passing
- `sideload-ipa` ("Build unsigned iOS app" step, xcodebuild for the
  actual iOS target): FAILING on every commit since the refactor.
  Last touched: App/SwairmApp/ContentView.swift and
  MLXDeviceLoopController.swift. The package itself compiles clean —
  this break is specific to the iOS app target build.

## Ground rules
- No local Mac. Never claim something compiles or is fixed without a
  green CI run to point to.
- Prefer targeted diffs over full-file rewrites when a scoped fix works.