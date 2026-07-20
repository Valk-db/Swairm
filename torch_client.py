"""
torch_client.py -- pure-PyTorch reference client (v0.1): REAL training, no MLX
================================================================================
Closes the proxy caveat (D8 remainder) down to "MLX runtime only": clients
fine-tune true DoRA adapters (A, B, m) on a FROZEN char-level transformer
with real AdamW gradient descent on real text, speaking the production wire
format to the Anchor.

Modes:
  offline (default): in-process fleet, aggregation via the production
      aggregate_round(). No server needed. Fast iteration.
  --online: real HTTP against a running Anchor (uvicorn main:app),
      exercising upload -> queue -> worker -> snapshot -> re-download.

Semantics pinned (D7): clients upload FULL post-training adapter state; the
aggregate REPLACES the global. Module names carry 'attn'/'mlp' substrings so
the Anchor's DEFAULT_RANK_MAP assigns exactly the ranks the client trains
(attn=4, mlp=6, head fallback=4) -- required, or A/B shapes break on reload.
Base model is rebuilt identically everywhere from BASE_SEED ("the phone
ships with the base model"). Wire format: float16 npz, same as main.py.

Data: byte-level tinyshakespeare, auto-downloaded to data/ on first run
(or pass --data your_file.txt). Each device trains on a distinct contiguous
shard -> natural heterogeneity. Held-out tail is the shared fleet eval.
"""

import argparse
import io
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ config
BASE_SEED = 1337
VOCAB = 256                  # byte-level
D_MODEL, N_HEADS, N_BLOCKS, D_FF = 128, 4, 2, 256
CTX_MAX = 256
RANK_ATTN, RANK_MLP, RANK_HEAD = 4, 6, 4     # must mirror aggregator ranks
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")
DATA_PATH = Path("data") / "tinyshakespeare.txt"
EVAL_FRAC = 0.1
MODULE_SUFFIXES = ("A", "B", "m")


# ------------------------------------------------------------------ model
class DoRALinear(nn.Module):
    """Frozen W0 + trainable DoRA adapter. W' = m * (W0 + BA)/||W0 + BA||_row."""

    def __init__(self, in_f, out_f, rank):
        super().__init__()
        self.register_buffer("W0", torch.randn(out_f, in_f) / math.sqrt(in_f))
        self.A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        self.m = nn.Parameter(self.W0.norm(dim=1).clone())

    def forward(self, x):
        V = self.W0 + self.B @ self.A
        Wp = self.m.unsqueeze(1) * V / V.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return x @ Wp.T


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = DoRALinear(D_MODEL, D_MODEL, RANK_ATTN)
        self.k_proj = DoRALinear(D_MODEL, D_MODEL, RANK_ATTN)
        self.v_proj = DoRALinear(D_MODEL, D_MODEL, RANK_ATTN)
        self.o_proj = DoRALinear(D_MODEL, D_MODEL, RANK_ATTN)

    def forward(self, x):
        b, t, c = x.shape
        hd = c // N_HEADS
        q = self.q_proj(x).view(b, t, N_HEADS, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, N_HEADS, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, t, N_HEADS, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, c))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = DoRALinear(D_MODEL, D_FF, RANK_MLP)
        self.fc2 = DoRALinear(D_FF, D_MODEL, RANK_MLP)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn, self.mlp = Attn(), MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Parameter(torch.randn(CTX_MAX, D_MODEL) * 0.01)
        self.blocks = nn.ModuleList(Block() for _ in range(N_BLOCKS))
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = DoRALinear(D_MODEL, VOCAB, RANK_HEAD)  # 'head' -> fallback rank 4
        for p in self.parameters():                        # freeze EVERYTHING...
            p.requires_grad_(False)
        for mod in self.modules():                         # ...except adapters
            if isinstance(mod, DoRALinear):
                for p in (mod.A, mod.B, mod.m):
                    p.requires_grad_(True)

    def forward(self, idx):
        x = self.tok(idx) + self.pos[: idx.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def build_model():
    torch.manual_seed(BASE_SEED)          # identical frozen base everywhere
    return TinyTransformer()


# ------------------------------------------------------------------ adapters
def export_adapter(model):
    """Full adapter state (D7), float16 like the wire format."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, DoRALinear):
            out[name] = {k: getattr(mod, k).detach().numpy().astype(np.float16)
                         for k in MODULE_SUFFIXES}
    return out


def load_adapter(model, mods):
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, DoRALinear) and name in mods:
                for k in MODULE_SUFFIXES:
                    getattr(mod, k).copy_(torch.as_tensor(
                        np.asarray(mods[name][k], dtype=np.float32)))


# ------------------------------------------------------------------ data
def load_corpus(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading corpus -> {path} ...")
        try:
            urllib.request.urlretrieve(DATA_URL, path)
        except Exception as exc:
            raise SystemExit(f"download failed ({exc}); pass --data <file.txt>")
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    data = torch.from_numpy(raw.copy()).long()
    n_eval = int(len(data) * EVAL_FRAC)
    return data[:-n_eval], data[-n_eval:]


def get_batch(data, rng, batch, ctx):
    ix = rng.integers(0, len(data) - ctx - 1, size=batch)
    x = torch.stack([data[i:i + ctx] for i in ix])
    y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
    return x, y


# ------------------------------------------------------------------ train/eval
def train_client(model, shard, rng, steps, lr, batch, ctx):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    loss = None
    for _ in range(steps):
        x, y = get_batch(shard, rng, batch, ctx)
        loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


@torch.no_grad()
def eval_loss(model, data, ctx, iters=25, batch=16):
    model.eval()
    rng = np.random.default_rng(0)        # fixed eval batches
    losses = [float(F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1)))
              for x, y in (get_batch(data, rng, batch, ctx)
                           for _ in range(iters))]
    return float(np.mean(losses))


# ------------------------------------------------------------------ wire/http
def pack_upload(device_id, fetch_version, curriculum_epoch, modules):
    arrays = {"__meta__": np.frombuffer(json.dumps({
        "device_id": device_id, "fetch_version": fetch_version,
        "curriculum_epoch": curriculum_epoch}).encode(), dtype=np.uint8)}
    for name, mod in modules.items():
        for k in MODULE_SUFFIXES:
            arrays[f"{name}::{k}"] = np.asarray(mod[k], dtype=np.float16)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def fetch_global(base):
    """Returns (version, mods) or None if the Anchor has no adapter yet."""
    try:
        with urllib.request.urlopen(f"{base}/adapter/latest", timeout=10) as r:
            version = int(r.headers.get("X-Adapter-Version", "0"))
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    mods = {}
    with np.load(io.BytesIO(data)) as z:
        for key in z.files:
            name, part = key.rsplit("::", 1)
            mods.setdefault(name, {})[part] = z[key]
    return version, mods


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())


def post_upload(base, raw):
    req = urllib.request.Request(f"{base}/upload", data=raw, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


# ------------------------------------------------------------------ fleets
def shards_for(train, n):
    per = len(train) // n
    return [train[i * per:(i + 1) * per] for i in range(n)]


def run_offline(args):
    from aggregator import aggregate_round      # production path, no server
    train, held = load_corpus(args.data)
    shards = shards_for(train, args.devices)
    model = build_model()
    init = export_adapter(model)
    global_mods, version = init, 0
    base_err = eval_loss(model, held, args.ctx)
    print(f"frozen base (B=0) held-out loss: {base_err:.4f} "
          f"(uniform-bytes ceiling ~{math.log(VOCAB):.2f})")

    dev0_last = None
    for r in range(args.rounds):
        t0 = time.time()
        uploads = []
        for i in range(args.devices):
            load_adapter(model, global_mods)
            rng = np.random.default_rng(10_000 * (r + 1) + i)
            last = train_client(model, shards[i], rng, args.steps,
                                args.lr, args.batch, args.ctx)
            mods = export_adapter(model)
            if i == 0:
                dev0_last = mods
            uploads.append({"device_id": f"dev{i}", "fetch_version": version,
                            "curriculum_epoch": 1, "modules": mods})
        result = aggregate_round(uploads, current_version=version,
                                 current_epoch=1)
        global_mods, version = result["modules"], result["version"]
        load_adapter(model, global_mods)
        ev = eval_loss(model, held, args.ctx)
        print(f"[round {r}] global v{version} held-out loss {ev:.4f} "
              f"(last local train loss {last:.4f}, {time.time() - t0:.0f}s)")

    load_adapter(model, dev0_last)
    solo = eval_loss(model, held, args.ctx)
    load_adapter(model, global_mods)
    fed = eval_loss(model, held, args.ctx)
    print(f"\nfinal held-out: federated global {fed:.4f} vs "
          f"dev0 solo adapter {solo:.4f} "
          f"(federated should generalize better across shards)")


def run_online(args):
    train, held = load_corpus(args.data)
    shards = shards_for(train, args.devices)
    model = build_model()
    init = export_adapter(model)
    for r in range(args.rounds):
        status = get_json(f"{args.anchor}/status")
        got = fetch_global(args.anchor)
        version, global_mods = got if got else (0, init)
        load_adapter(model, global_mods)
        ev = eval_loss(model, held, args.ctx)
        print(f"[round {r}] anchor v{status['version']} "
              f"(epoch {status['curriculum_epoch']}, "
              f"skew={status['skew_detected']}) | held-out loss {ev:.4f}")
        for i in range(args.devices):
            load_adapter(model, global_mods)
            rng = np.random.default_rng(10_000 * (r + 1) + i)
            train_client(model, shards[i], rng, args.steps,
                         args.lr, args.batch, args.ctx)
            post_upload(args.anchor, pack_upload(
                f"dev{i}", version, status["curriculum_epoch"],
                export_adapter(model)))
        print(f"          uploaded {args.devices} adapters; "
              f"waiting {args.interval:.0f}s for the worker...")
        time.sleep(args.interval)
    got = fetch_global(args.anchor)
    if got:
        load_adapter(model, got[1])
        print(f"\nfinal downloaded global held-out loss: "
              f"{eval_loss(model, held, args.ctx):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--anchor", default="http://127.0.0.1:8000")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--devices", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--interval", type=float, default=25.0)
    args = ap.parse_args()
    assert args.ctx <= CTX_MAX
    torch.set_num_threads(max(torch.get_num_threads(), 4))
    (run_online if args.online else run_offline)(args)
