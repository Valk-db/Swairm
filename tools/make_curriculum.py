"""Generate synthetic curriculum shards for the CI real-MLX-round job.

Writes shard_*.npz files in the format CurriculumLoader expects:
  token_ids: [num_sequences, seq_len]  uint32
  labels:    [num_sequences, seq_len]  uint32

The content is a fixed, learnable pattern (not pure noise) so the DoRA
trainer has a real gradient signal: each sequence is a contiguous window
into one shared random "document", and labels are the same window shifted
by one (next-token prediction), the standard LM training setup. Token ids
are kept well under the model's vocab size so any small MLX model accepts
them.

Usage:
  python tools/make_curriculum.py --out curriculum \
      --sequences 64 --seq-len 64 --vocab 32000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="curriculum")
    ap.add_argument("--sequences", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=32000,
                    help="token ids are drawn < vocab; keep under model vocab")
    ap.add_argument("--shard-size", type=int, default=32,
                    help="sequences per shard file")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # One shared random "document" per run; sequences are windows into it so
    # the pattern is learnable and consistent across devices.
    doc = rng.integers(0, args.vocab, size=args.sequences * args.seq_len + 1,
                       dtype=np.int64)

    n_shards = (args.sequences + args.shard_size - 1) // args.shard_size
    for shard in range(n_shards):
        lo = shard * args.shard_size
        hi = min(lo + args.shard_size, args.sequences)
        count = hi - lo

        token_ids = np.empty((count, args.seq_len), dtype=np.uint32)
        labels = np.empty((count, args.seq_len), dtype=np.uint32)
        for i in range(count):
            start = (lo + i) * args.seq_len
            window = doc[start:start + args.seq_len + 1]
            token_ids[i] = window[:-1]   # inputs
            labels[i] = window[1:]       # next-token targets

        path = out / f"shard_{shard:05d}.npz"
        np.savez_compressed(path, token_ids=token_ids, labels=labels)
        print(f"wrote {path} ({count} sequences)")

    print(f"done: {n_shards} shard(s), {args.sequences} sequences, "
          f"seq_len={args.seq_len}, vocab<{args.vocab}")


if __name__ == "__main__":
    main()
