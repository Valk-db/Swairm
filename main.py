"""
main.py -- FCS Anchor service (v1.1)
====================================
FastAPI shell + directory-as-queue + single background aggregation worker,
wired to aggregator.py (the validated math core).

v1.1: replaced deprecated @app.on_event("startup") with the lifespan
context manager (current FastAPI pattern). No behavior change.

Design (per locked spec):
  - Upload handler does NO parsing/aggregation: it writes the raw payload to
    queue/temp/ and atomically os.rename()s it into queue/pending/
    (Maildir pattern -- crash-safe, no partial files visible to the worker).
  - ONE background worker drains pending/ on an interval, validates,
    aggregates via aggregate_round(), writes a versioned snapshot to
    models/, moves inputs to processed/ (or quarantine/ if malformed).
    Single-writer: only the worker mutates state.
  - Version numbers are monotonic, including after any future rollback.

Deviations from spec, made openly (upgrade later if needed):
  - State is an atomic-rename JSON file, not SQLite/WAL. At family scale
    with a single writer, SQLite adds nothing yet; the queue durability
    lives in the filesystem, exactly as designed.
  - detect_skew() is a stub returning False (uniform weights). Wiring the
    Circadian participation baseline into it is the next telemetry task --
    the conditional reciprocal policy activates only once that lands.

Payload format (one .npz per upload):
  "__meta__": JSON string {device_id, fetch_version, curriculum_epoch}
  "<module>::A", "<module>::B", "<module>::m" per module

Run server:    pip install fastapi uvicorn
               uvicorn main:app --host 0.0.0.0 --port 8000
Self-test:     python main.py --selftest     (no HTTP, no fastapi needed)
"""

import io
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from aggregator import aggregate_round

# ------------------------------------------------------------------ config
BASE_DIR = Path(__file__).resolve().parent
QUEUE_TEMP = BASE_DIR / "queue" / "temp"
QUEUE_PENDING = BASE_DIR / "queue" / "pending"
QUEUE_PROCESSED = BASE_DIR / "queue" / "processed"
QUEUE_QUARANTINE = BASE_DIR / "queue" / "quarantine"
MODELS_DIR = BASE_DIR / "models"
STATE_PATH = BASE_DIR / "state.json"
AGG_INTERVAL_S = 30 * 60        # worker drain interval (matches sim cadence)
META_KEYS = {"device_id", "fetch_version", "curriculum_epoch"}

for d in (QUEUE_TEMP, QUEUE_PENDING, QUEUE_PROCESSED, QUEUE_QUARANTINE,
          MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ state
def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"version": 0, "curriculum_epoch": 1, "rounds": 0,
            "participation_log": []}


def save_state(state):
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE_PATH)          # atomic on the same filesystem


# ------------------------------------------------------------------ payloads
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


def unpack_upload(path: Path) -> dict:
    with np.load(path) as z:
        meta = json.loads(bytes(z["__meta__"]).decode())
        if not META_KEYS.issubset(meta):
            raise ValueError(f"missing meta keys: {META_KEYS - set(meta)}")
        modules = {}
        for key in z.files:
            if key == "__meta__":
                continue
            name, part = key.rsplit("::", 1)
            modules.setdefault(name, {})[part] = z[key]
        for name, mod in modules.items():
            if set(mod) != {"A", "B", "m"}:
                raise ValueError(f"module {name} incomplete: {set(mod)}")
    return {"device_id": meta["device_id"],
            "fetch_version": int(meta["fetch_version"]),
            "curriculum_epoch": int(meta["curriculum_epoch"]),
            "modules": modules}


def enqueue(raw: bytes) -> str:
    """Atomic write path used by both the HTTP handler and the self-test."""
    name = f"{int(time.time())}_{uuid.uuid4().hex}.npz"
    tmp = QUEUE_TEMP / name
    tmp.write_bytes(raw)
    os.rename(tmp, QUEUE_PENDING / name)   # atomic: worker never sees partials
    return name


def save_snapshot(result) -> Path:
    arrays = {}
    for name, mod in result["modules"].items():
        arrays[f"{name}::A"] = mod["A"]
        arrays[f"{name}::B"] = mod["B"]
        arrays[f"{name}::m"] = mod["m"]
    path = MODELS_DIR / f"v_{result['version']:05d}.npz"
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------ skew hook
def detect_skew(state) -> bool:
    """
    STUB. Returns False -> uniform weights (the locked balanced-mode policy).
    TODO: wire the Circadian per-hour participation baseline here; when
    demographic skew is detected, returning True activates reciprocal
    1/(1+s) weighting (locked skew-mode policy, 15/15 seeds, t=6.07).
    """
    return False


# ------------------------------------------------------------------ worker
def drain_once(state, verbose=True):
    """One worker pass: validate pending uploads, aggregate, snapshot."""
    files = sorted(QUEUE_PENDING.glob("*.npz"))
    if not files:
        return None
    uploads, sources = [], []
    for f in files:
        try:
            uploads.append(unpack_upload(f))
            sources.append(f)
        except Exception as exc:
            if verbose:
                print(f"[worker] quarantined {f.name}: {exc}")
            os.rename(f, QUEUE_QUARANTINE / f.name)
    if not uploads:
        return None

    result = aggregate_round(uploads,
                             current_version=state["version"],
                             current_epoch=state["curriculum_epoch"],
                             skew_detected=detect_skew(state))
    if result["modules"]:
        snap = save_snapshot(result)
        state["version"] = result["version"]
        state["rounds"] += 1
        state["participation_log"].append(
            {"t": time.time(), "n_uploads": len(uploads)})
        state["participation_log"] = state["participation_log"][-2000:]
        save_state(state)
        if verbose:
            print(f"[worker] round {state['rounds']}: aggregated "
                  f"{len(uploads)} uploads -> version {state['version']} "
                  f"({snap.name})")
    for f in sources:
        os.rename(f, QUEUE_PROCESSED / f.name)
    return result


def worker_loop():
    state = load_state()
    while True:
        try:
            drain_once(state)
        except Exception as exc:
            print(f"[worker] round failed, queue preserved: {exc}")
        time.sleep(AGG_INTERVAL_S)


# ------------------------------------------------------------------ HTTP layer
try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request, Response

    @asynccontextmanager
    async def lifespan(app):
        threading.Thread(target=worker_loop, daemon=True).start()
        yield

    app = FastAPI(title="FCS Anchor", lifespan=lifespan)

    @app.get("/status")
    def status():
        state = load_state()
        return {"version": state["version"],
                "curriculum_epoch": state["curriculum_epoch"],
                "rounds": state["rounds"],
                "pending": len(list(QUEUE_PENDING.glob("*.npz")))}

    @app.post("/upload")
    async def upload(request: Request):
        raw = await request.body()
        name = enqueue(raw)                 # no parsing here, by design
        return {"queued": name}

    @app.get("/adapter/latest")
    def adapter_latest():
        state = load_state()
        if state["version"] == 0:
            return Response(status_code=404,
                            content="no global adapter yet")
        path = MODELS_DIR / f"v_{state['version']:05d}.npz"
        return Response(content=path.read_bytes(),
                        media_type="application/octet-stream",
                        headers={"X-Adapter-Version": str(state["version"]),
                                 "X-Curriculum-Epoch":
                                     str(state["curriculum_epoch"])})
except ImportError:
    app = None      # fastapi not installed; --selftest still works


# ------------------------------------------------------------------ self-test
def selftest():
    print("=== main.py self-test (no HTTP) ===")
    rng = np.random.default_rng(42)
    M_DIM, N_DIM, RANK = 128, 256, 4
    shared_A = rng.standard_normal((RANK, N_DIM)) / np.sqrt(N_DIM)
    shared_B = rng.standard_normal((M_DIM, RANK)) / np.sqrt(RANK)
    shared_m = rng.uniform(0.5, 2.5, M_DIM)

    state = load_state()
    v0 = state["version"]
    for i in range(12):
        mod = {"layers.0.attn.q_proj": {
            "A": shared_A + rng.standard_normal((RANK, N_DIM)) / np.sqrt(N_DIM),
            "B": shared_B + rng.standard_normal((M_DIM, RANK)) / np.sqrt(RANK),
            "m": np.clip(shared_m + rng.normal(0, 0.1, M_DIM), 0.1, 3.0)}}
        enqueue(pack_upload(f"dev{i}", fetch_version=v0,
                            curriculum_epoch=state["curriculum_epoch"],
                            modules=mod))
    (QUEUE_PENDING / "garbage.npz").write_bytes(b"not an npz file")

    print(f"  queued: 12 valid uploads + 1 garbage file")
    result = drain_once(state)
    assert result is not None, "worker produced nothing"
    assert state["version"] == v0 + 1, "version did not advance"
    snap = MODELS_DIR / f"v_{state['version']:05d}.npz"
    assert snap.exists(), "snapshot missing"
    with np.load(snap) as z:
        shapes = {k: z[k].shape for k in z.files}
    print(f"  version: {v0} -> {state['version']}")
    print(f"  snapshot: {snap.name}, arrays: {shapes}")
    print(f"  quarantined: {len(list(QUEUE_QUARANTINE.glob('*.npz')))} "
          f"(expect >= 1)")
    print(f"  processed:   {len(list(QUEUE_PROCESSED.glob('*.npz')))}")
    tel = result["telemetry"]["modules"]["layers.0.attn.q_proj"]
    print(f"  telemetry: aggregated={tel['aggregated']}, "
          f"trailing_ratio={tel['trailing_ratio']:.4f}")
    print("Self-test PASSED.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("Run the server with: uvicorn main:app --host 0.0.0.0 --port 8000")
        print("Or validate the pipeline with: python main.py --selftest")
