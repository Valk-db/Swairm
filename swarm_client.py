"""
swarm_client.py -- fake phone client for the FCS Anchor (linear-proxy trainer)
===============================================================================
Simulates a fleet of phones over real HTTP: polls the Anchor for the latest
global adapter, "trains" toward a per-device synthetic target using the same
linear proxy validated in the simulators, and POSTs full adapter uploads.

Purpose: exercise the ENTIRE Anchor loop (HTTP -> queue -> worker ->
aggregate -> snapshot -> re-download) end-to-end on one machine, before any
Swift/MLX work. It does NOT validate real training -- the convex-proxy
caveat from DECISIONS.md applies in full.

Semantics pinned here (see DECISIONS.md): clients upload FULL post-training
adapter state; the Anchor's per-round aggregate REPLACES the global
(FedAvg-on-weights). All prior validation used these semantics.

Stdlib HTTP (urllib) -- no extra deps beyond numpy/scikit-learn.
pack_upload() is duplicated from main.py deliberately: importing main.py
would create the server's queue directories as a side effect, and a real
client won't share the server's code anyway.

Usage (two terminals):
  1)  $env:FCS_AGG_INTERVAL_S="20"
      uvicorn main:app --port 8000
  2)  python swarm_client.py --fleet 12 --rounds 10 --interval 25
"""

import argparse
import io
import json
import time
import urllib.error
import urllib.request

import numpy as np
from sklearn.utils.extmath import randomized_svd

M_DIM, N_DIM, RANK = 128, 256, 4
MODULE = "layers.0.attn.q_proj"
CLIENT_LR = 0.5
NOISE = 0.05


# ------------------------------------------------------------------ payload
def pack_upload(device_id, fetch_version, curriculum_epoch, modules) -> bytes:
    arrays = {"__meta__": np.frombuffer(json.dumps({
        "device_id": device_id, "fetch_version": fetch_version,
        "curriculum_epoch": curriculum_epoch}).encode(), dtype=np.uint8)}
    for name, mod in modules.items():
        arrays[f"{name}::A"] = np.asarray(mod["A"], dtype=np.float16)
        arrays[f"{name}::B"] = np.asarray(mod["B"], dtype=np.float16)
        arrays[f"{name}::m"] = np.asarray(mod["m"], dtype=np.float16)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


# ------------------------------------------------------------------ http
def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())


def get_adapter(base):
    """Returns (version, dir_matrix, magnitude) or None if no adapter yet."""
    try:
        with urllib.request.urlopen(f"{base}/adapter/latest", timeout=10) as r:
            version = int(r.headers.get("X-Adapter-Version", "0"))
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    with np.load(io.BytesIO(data)) as z:
        A = z[f"{MODULE}::A"].astype(np.float32)
        B = z[f"{MODULE}::B"].astype(np.float32)
        m = z[f"{MODULE}::m"].astype(np.float32)
    return version, B @ A, m


def post_upload(base, raw):
    req = urllib.request.Request(
        f"{base}/upload", data=raw, method="POST",
        headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


# ------------------------------------------------------------------ devices
def make_targets(n_devices):
    """Shared low-rank basis + per-device perturbation, non-trivial m --
    same construction the validation suite used."""
    shared = np.random.default_rng(42)
    D_shared = ((shared.standard_normal((M_DIM, RANK)) / np.sqrt(RANK))
                @ (shared.standard_normal((RANK, N_DIM)) / np.sqrt(N_DIM)))
    m_shared = shared.uniform(0.5, 2.5, M_DIM)
    targets = {}
    for i in range(n_devices):
        rng = np.random.default_rng(1000 + i)
        D_k = D_shared + 0.3 * ((rng.standard_normal((M_DIM, RANK)) / np.sqrt(RANK))
                                @ (rng.standard_normal((RANK, N_DIM)) / np.sqrt(N_DIM)))
        m_k = np.clip(m_shared + rng.normal(0, 0.2, M_DIM), 0.1, 3.0)
        targets[f"dev{i}"] = (D_k, m_k, rng)
    fleet_dir = np.mean([t[0] for t in targets.values()], axis=0)
    fleet_m = np.mean([t[1] for t in targets.values()], axis=0)
    return targets, fleet_dir, fleet_m


def train_step(g_dir, g_m, d_k, m_k, rng):
    """Linear proxy: move local copy toward local target, refactor to rank."""
    new_dir = (g_dir + CLIENT_LR * (d_k - g_dir)
               + NOISE * rng.standard_normal((M_DIM, N_DIM)) / np.sqrt(N_DIM))
    U, S, Vt = randomized_svd(new_dir.astype(np.float32),
                              n_components=RANK, random_state=42)
    A = np.diag(np.sqrt(S)) @ Vt
    B = U @ np.diag(np.sqrt(S))
    m_new = g_m + CLIENT_LR * (m_k - g_m)
    return A, B, m_new


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="http://127.0.0.1:8000")
    ap.add_argument("--fleet", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--interval", type=float, default=25.0,
                    help="seconds between fleet upload waves; set slightly "
                         "longer than the Anchor's FCS_AGG_INTERVAL_S")
    args = ap.parse_args()

    targets, fleet_dir, fleet_m = make_targets(args.fleet)
    fleet_norm = np.linalg.norm(fleet_dir)

    for rnd in range(args.rounds):
        status = get_json(f"{args.anchor}/status")
        adapter = get_adapter(args.anchor)
        if adapter is None:
            version, g_dir, g_m = 0, np.zeros((M_DIM, N_DIM), np.float32), \
                np.ones(M_DIM, np.float32)
        else:
            version, g_dir, g_m = adapter

        err = np.linalg.norm(g_dir - fleet_dir) / fleet_norm
        print(f"[round {rnd}] anchor v{status['version']} "
              f"(epoch {status['curriculum_epoch']}, "
              f"skew={status['skew_detected']}) | "
              f"global-vs-fleet-target err={err:.4f}")

        for device_id, (d_k, m_k, rng) in targets.items():
            A, B, m_new = train_step(g_dir, g_m, d_k, m_k, rng)
            raw = pack_upload(device_id, fetch_version=version,
                              curriculum_epoch=status["curriculum_epoch"],
                              modules={MODULE: {"A": A, "B": B, "m": m_new}})
            post_upload(args.anchor, raw)
        print(f"          uploaded {args.fleet} adapters; waiting "
              f"{args.interval:.0f}s for the worker...")
        time.sleep(args.interval)

    final = get_adapter(args.anchor)
    if final:
        _, g_dir, g_m = final
        err = np.linalg.norm(g_dir - fleet_dir) / fleet_norm
        m_err = np.linalg.norm(g_m - fleet_m) / np.linalg.norm(fleet_m)
        print(f"\nFinal: dir err={err:.4f}, magnitude err={m_err:.4f} "
              f"(both should shrink toward a noise floor across rounds)")
