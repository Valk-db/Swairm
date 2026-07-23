"""
main.py -- FCS Anchor service (v1.2 with WebSockets)
====================================================
FastAPI shell + directory-as-queue + single background aggregation worker,
wired to aggregator.py (the validated math core).

v1.2: detect_skew() implemented (was a stub). Completes decision D2's
conditional staleness policy.
  Detection principle: demographic skew is NOT detectable from hourly
  volume (balanced fleets are also diurnal -- everyone charges at night).
  The distinguishing signal is DEVICE COMPOSITION: if the device set active
  in the night window and the device set active in the day window are both
  substantial but nearly disjoint (low Jaccard similarity), participation
  is demographically skewed -> reciprocal 1/(1+s) weighting activates.
  Thresholds are heuristics pending real fleet data -- marked in config.
v1.1: lifespan handler replacing deprecated @app.on_event.

Design (per locked spec):
  - Upload handler does NO parsing/aggregation: raw payload -> queue/temp/
    -> atomic os.replace() into queue/pending/ (Maildir pattern).
  - ONE background worker drains pending/, validates, aggregates via
    aggregate_round(), snapshots to models/. Single-writer.
  - Version numbers are monotonic, including after any future rollback.

Deviations from spec, made openly:
  - State is an atomic-rename JSON file, not SQLite/WAL (single writer,
    family scale; upgrade trigger documented in DECISIONS.md).

Note: participation_log entries gained "hour" and "devices" fields in
v1.2. Old state.json files are read compatibly (missing fields skipped);
deleting state.json (gitignored) also resets cleanly.

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
import asyncio
from pathlib import Path

import numpy as np

from aggregator import aggregate_round

# =================================================================-- WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()
main_loop = None

# ------------------------------------------------------------------ config
BASE_DIR = Path(__file__).resolve().parent
QUEUE_TEMP = BASE_DIR / "queue" / "temp"
QUEUE_PENDING = BASE_DIR / "queue" / "pending"
QUEUE_PROCESSED = BASE_DIR / "queue" / "processed"
QUEUE_QUARANTINE = BASE_DIR / "queue" / "quarantine"
MODELS_DIR = BASE_DIR / "models"
STATE_PATH = BASE_DIR / "state.json"
AGG_INTERVAL_S = int(os.environ.get("FCS_AGG_INTERVAL_S", str(30 * 60)))        # worker drain interval (matches sim cadence)
META_KEYS = {"device_id", "fetch_version", "curriculum_epoch"}

# --- skew-detector heuristics (UNTUNED -- revisit with real fleet data) ---
SKEW_WINDOW_ROUNDS = 336        # trailing rounds examined (~7d at 30min)
SKEW_MIN_HISTORY = 48           # don't judge before ~1 day of rounds
SKEW_MIN_FRACTION = 0.20        # both windows need >=20% of upload volume
SKEW_JACCARD_THRESHOLD = 0.30   # device-set overlap below this = skew


def _is_night(hour):
    return hour >= 22 or hour < 7


def _is_day(hour):
    return 9 <= hour < 17


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
    os.replace(tmp, STATE_PATH)           # atomic on the same filesystem


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
    with np.load(path, allow_pickle=False) as z:
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
    os.replace(tmp, QUEUE_PENDING / name)   # atomic: worker never sees partials
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


# ------------------------------------------------------------------ skew
def detect_skew(state) -> bool:
    """
    Demographic-skew detector (decision D2).

    Compares WHICH devices are active at night vs during the day over the
    trailing window. Volume alone cannot distinguish "balanced but diurnal"
    (everyone charges at night) from true demographic skew (different
    populations at different times) -- device-set overlap can.

    Returns True (-> reciprocal 1/(1+s) weighting) only when:
      - enough history exists (SKEW_MIN_HISTORY rounds), and
      - BOTH night and day windows carry >= SKEW_MIN_FRACTION of upload
        volume (one quiet window = ordinary diurnal pattern, not skew), and
      - Jaccard(night_devices, day_devices) < SKEW_JACCARD_THRESHOLD.
    """
    log = state["participation_log"][-SKEW_WINDOW_ROUNDS:]
    if len(log) < SKEW_MIN_HISTORY:
        return False
    night_dev, day_dev = set(), set()
    night_n = day_n = total_n = 0
    for rec in log:
        hour = rec.get("hour")
        devices = rec.get("devices", [])
        total_n += len(devices)
        if hour is None:
            continue
        if _is_night(hour):
            night_dev.update(devices)
            night_n += len(devices)
        elif _is_day(hour):
            day_dev.update(devices)
            day_n += len(devices)
    if total_n == 0:
        return False
    if (night_n < SKEW_MIN_FRACTION * total_n
            or day_n < SKEW_MIN_FRACTION * total_n):
        return False                      # ordinary diurnal concentration
    union = night_dev | day_dev
    if not union:
        return False
    jaccard = len(night_dev & day_dev) / len(union)
    return jaccard < SKEW_JACCARD_THRESHOLD


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
            os.replace(f, QUEUE_QUARANTINE / f.name)
    if not uploads:
        return None

    skew = detect_skew(state)
    epoch = state["curriculum_epoch"]
    result = aggregate_round(uploads,
                             current_version=state["version"],
                             current_epoch=epoch,
                             skew_detected=skew,
                             epoch_transition_weights={
                                 (epoch - 1, epoch): EPOCH_TRANSITION_WEIGHT})
    if result["modules"]:
        snap = save_snapshot(result)
        state["version"] = result["version"]
        state["rounds"] += 1
        state["participation_log"].append(
            {"t": time.time(),
             "hour": time.localtime().tm_hour,
             "devices": sorted({u["device_id"] for u in uploads}),
             "n_uploads": len(uploads)})
        state["participation_log"] = state["participation_log"][-2000:]
        save_state(state)
        
        # Broadcast the new version instantly via WebSocket
        if main_loop and main_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(f"NEW_VERSION:{state['version']}"), main_loop
            )

        if verbose:
            print(f"[worker] round {state['rounds']}: aggregated "
                  f"{len(uploads)} uploads -> version {state['version']} "
                  f"({snap.name}, skew_detected={skew})")
    for f in sources:
        os.replace(f, QUEUE_PROCESSED / f.name)
    return result


def worker_loop():
    state = load_state()
    while True:
        try:
            drain_once(state)
        except Exception as exc:
            print(f"[worker] round failed, queue preserved: {exc}")
        time.sleep(AGG_INTERVAL_S)


# ------------------------------------------------------------------ HTTP & WS layer
try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect

    @asynccontextmanager
    async def lifespan(app):
        global main_loop
        main_loop = asyncio.get_running_loop()
        threading.Thread(target=worker_loop, daemon=True).start()
        yield

    app = FastAPI(title="FCS Anchor", lifespan=lifespan)

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        await manager.connect(websocket)
        try:
            while True:
                _ = await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @app.get("/status")
    def status():
        state = load_state()
        return {"version": state["version"],
                "curriculum_epoch": state["curriculum_epoch"],
                "rounds": state["rounds"],
                "skew_detected": detect_skew(state),
                "pending": len(list(QUEUE_PENDING.glob("*.npz")))}

    @app.post("/upload")
    async def upload(request: Request):
        raw = await request.body()
        name = enqueue(raw)                # no parsing here, by design
        return {"queued": name}

    @app.get("/adapter/latest")
    def adapter_latest():
        state = load_state()
        if state["version"] == 0:
            return Response(status_code=404,
                            content="no global adapter yet")
        path = MODELS_DIR / f"v_{state['version']:05d}.npz"
        if not path.exists():
            return Response(status_code=404,
                            content="adapter file missing on disk")
        return Response(content=path.read_bytes(),
                        media_type="application/octet-stream",
                        headers={"X-Adapter-Version": str(state["version"]),
                                 "X-Curriculum-Epoch":
                                     str(state["curriculum_epoch"])})
except ImportError:
    app = None      # fastapi not installed; --selftest still works


# ------------------------------------------------------------------ self-test
def selftest():
    print("=== main.py v1.2 self-test (no HTTP) ===")
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

    print("  queued: 12 valid uploads + 1 garbage file")
    result = drain_once(state)
    assert result is not None, "worker produced nothing"
    assert state["version"] == v0 + 1, "version did not advance"
    snap = MODELS_DIR / f"v_{state['version']:05d}.npz"
    assert snap.exists(), "snapshot missing"
    print(f"  version: {v0} -> {state['version']}, snapshot: {snap.name}")
    print(f"  quarantined: {len(list(QUEUE_QUARANTINE.glob('*.npz')))} "
          f"(expect >= 1)")

    # --- detect_skew unit checks on synthetic participation logs ---------
    def synth_log(disjoint):
        log = []
        for r in range(120):                     # 60 night + 60 day rounds
            if r % 2 == 0:
                hour, devs = 2, [f"night{d}" for d in range(5)]
            else:
                hour = 14
                devs = ([f"day{d}" for d in range(5)] if disjoint
                        else [f"night{d}" for d in range(5)])
            log.append({"t": 0, "hour": hour, "devices": devs,
                        "n_uploads": len(devs)})
        return log

    balanced = {"participation_log": synth_log(disjoint=False)}
    skewed = {"participation_log": synth_log(disjoint=True)}
    sparse = {"participation_log": synth_log(disjoint=True)[:10]}
    assert detect_skew(balanced) is False, "balanced flagged as skew"
    assert detect_skew(skewed) is True, "skew not detected"
    assert detect_skew(sparse) is False, "judged on insufficient history"
    print("  detect_skew: balanced=False, disjoint=True, "
          "sparse-history=False -- all correct")
    print("Self-test PASSED.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("Run the server with: uvicorn main:app --host 0.0.0.0 --port 8000")
        print("Or validate the pipeline with: python main.py --selftest")
