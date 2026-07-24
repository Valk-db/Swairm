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
All three CI jobs are green as of HEAD (build-test, integration,
sideload-ipa) — the iOS app build was fixed in 3e18cea and has stayed
fixed. The one interim failure (b99e002) was build-test, not
sideload-ipa: orientation tests that needed the MLX/Metal runtime
under `swift test` (XCTest can't load the default metallib on the
runner). Fixed in d005fe3 by making those tests MLX-runtime-free.

swairm-mlx-client (real on-device MLX DoRA fleet driver) and
tools/make_curriculum.py now exist but no CI job runs them — real
on-device MLX training is implemented but not yet proven end-to-end
in CI. That's the actual active frontier now, see next section.

## Ground rules
- No local Mac. Never claim something compiles or is fixed without a
  green CI run to point to.
- Prefer targeted diffs over full-file rewrites when a scoped fix works.