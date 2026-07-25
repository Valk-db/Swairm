"""
main.py -- FCS Anchor service (v1.3 with curriculum download)
====================================================
FastAPI shell + directory-as-queue + single background aggregation worker,
wired to aggregator.py (the validated math core).

v1.3: Added curriculum download endpoints (GET /curriculum/<epoch>/manifest.json,
GET /curriculum/<epoch>/shard_<N>.npz). Clients can now fetch curriculum shards
for real on-device MLX training.
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

Curriculum shard format (.npz):
  token_ids: [num_sequences, seq_len]  uint32
  labels:    [num_sequences, seq_len]  uint32

Run server:    pip install fastapi uvicorn
               uvicorn main:app --host 0.0.0.0 --port 8000
               # With TLS: uvicorn main:app --host 0.0.0.0 --port 8000 --ssl-certfile=cert.pem --ssl-keyfile=key.pem
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
import hmac
import hashlib
from pathlib import Path

import numpy as np

from aggregator import aggregate_round

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = None
    generate_latest = CONTENT_TYPE_LATEST = None

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
# Upload size limit (DoS protection) - 50MB default
MAX_UPLOAD_MB = int(os.environ.get("FCS_MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# Soft weight for uploads exactly one curriculum epoch behind (decision D9,
# validate_open_configs.py: soft 0.25 beats hard rejection in every tested
# transition regime, t=+21..+100). Older epochs remain hard-rejected.
EPOCH_TRANSITION_WEIGHT = float(os.environ.get("FCS_EPOCH_TRANSITION_WEIGHT", "0.25"))
META_KEYS = {"device_id", "fetch_version", "curriculum_epoch"}

# --- skew-detector heuristics (UNTUNED -- revisit with real fleet data) ---
SKEW_WINDOW_ROUNDS = 336        # trailing rounds examined (~7d at 30min)
SKEW_MIN_HISTORY = 48           # don't judge before ~1 day of rounds
SKEW_MIN_FRACTION = 0.20        # both windows need >=20% of upload volume
SKEW_JACCARD_THRESHOLD = 0.30   # device-set overlap below this = skew


# =============================================================-- Prometheus metrics
if PROMETHEUS_AVAILABLE:
    # Counters
    UPLOADS_RECEIVED = Counter(
        "fcs_uploads_received_total",
        "Total number of adapter uploads received",
        ["result"]  # "queued", "rejected_hmac", "rejected_size", "error"
    )
    ADAPTER_FETCHES = Counter(
        "fcs_adapter_fetches_total",
        "Total number of adapter fetches from /adapter/latest",
        ["result"]  # "ok", "not_found", "rejected_hmac", "error"
    )
    CURRICULUM_MANIFEST_REQUESTS = Counter(
        "fcs_curriculum_manifest_requests_total",
        "Total number of curriculum manifest requests",
        ["result", "epoch"]
    )
    CURRICULUM_SHARD_REQUESTS = Counter(
        "fcs_curriculum_shard_requests_total",
        "Total number of curriculum shard requests",
        ["result", "epoch"]
    )
    AGGREGATION_ROUNDS = Counter(
        "fcs_aggregation_rounds_total",
        "Total number of aggregation rounds completed",
        ["result"]  # "ok", "no_uploads", "error"
    )
    UPLOADS_QUARANTINED = Counter(
        "fcs_uploads_quarantined_total",
        "Total number of uploads quarantined during aggregation"
    )

    # Gauges
    CURRENT_VERSION = Gauge(
        "fcs_current_version",
        "Current global adapter version number"
    )
    CURRENT_EPOCH = Gauge(
        "fcs_current_epoch",
        "Current curriculum epoch number"
    )
    PENDING_UPLOADS = Gauge(
        "fcs_pending_uploads",
        "Number of uploads in queue/pending"
    )
    ACTIVE_DEVICES = Gauge(
        "fcs_active_devices",
        "Number of distinct devices that uploaded in the last round"
    )
    SKEW_DETECTED = Gauge(
        "fcs_skew_detected",
        "Whether demographic skew was detected in the last round (1=yes, 0=no)"
    )
    AGGREGATION_WALL_CLOCK = Gauge(
        "fcs_aggregation_wall_clock_seconds",
        "Wall-clock time of the last aggregation round in seconds"
    )
    AGGREGATED_UPLOADS = Gauge(
        "fcs_aggregated_uploads_last_round",
        "Number of uploads aggregated in the last round"
    )

    # Histograms
    UPLOAD_SIZE_BYTES = Histogram(
        "fcs_upload_size_bytes",
        "Size of upload payloads in bytes",
        buckets=[1024, 10240, 102400, 1048576, 10485760, 52428800]
    )
    AGGREGATION_DURATION_SECONDS = Histogram(
        "fcs_aggregation_duration_seconds",
        "Time spent in aggregation round (drain_once)",
        buckets=[1, 5, 10, 30, 60, 120, 300]
    )
else:
    UPLOADS_RECEIVED = ADAPTER_FETCHES = CURRICULUM_MANIFEST_REQUESTS = None
    CURRICULUM_SHARD_REQUESTS = AGGREGATION_ROUNDS = UPLOADS_QUARANTINED = None
    CURRENT_VERSION = CURRENT_EPOCH = PENDING_UPLOADS = ACTIVE_DEVICES = None
    SKEW_DETECTED = AGGREGATION_WALL_CLOCK = AGGREGATED_UPLOADS = None
    UPLOAD_SIZE_BYTES = AGGREGATION_DURATION_SECONDS = None


def _is_night(hour):
    return hour >= 22 or hour < 7


def _is_day(hour):
    return 9 <= hour < 17


# ================================================================-- TLS / HMAC auth
# HMAC secret for request authentication (disabled when empty for dev)
HMAC_SECRET = os.environ.get("FCS_HMAC_SECRET", "").encode()
# Endpoints that require HMAC auth (exact paths and prefixes)
HMAC_PROTECTED_PATHS = {
    "/upload",
    "/adapter/latest",
    "/curriculum/",   # prefix match for all curriculum endpoints
}

def verify_hmac(request: Request, body: bytes) -> bool:
    """Verify HMAC-SHA256 signature on request. Returns True if valid or auth disabled."""
    if not HMAC_SECRET:
        return True  # auth disabled (dev mode)
    sig_header = request.headers.get("X-HMAC-Signature")
    if not sig_header:
        return False
    # Signature format: "sha256=<hex>"
    if not sig_header.startswith("sha256="):
        return False
    provided = sig_header[7:]  # strip "sha256="
    # Canonical string: METHOD\nPATH\nBODY
    canonical = f"{request.method}\n{request.url.path}\n".encode() + body
    expected = hmac.new(HMAC_SECRET, canonical, hashlib.sha256).hexdigest()
    return hmac.compare_digest(provided, expected)


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
    import time as time_module
    start_time = time_module.time()

    files = sorted(QUEUE_PENDING.glob("*.npz"))
    if not files:
        if PROMETHEUS_AVAILABLE:
            AGGREGATION_ROUNDS.labels(result="no_uploads").inc()
        return None
    uploads, sources = [], []
    quarantined_count = 0
    for f in files:
        try:
            uploads.append(unpack_upload(f))
            sources.append(f)
        except Exception as exc:
            if verbose:
                print(f"[worker] quarantined {f.name}: {exc}")
            os.replace(f, QUEUE_QUARANTINE / f.name)
            quarantined_count += 1
    if not uploads:
        if PROMETHEUS_AVAILABLE:
            AGGREGATION_ROUNDS.labels(result="no_uploads").inc()
            if quarantined_count > 0:
                UPLOADS_QUARANTINED.inc(quarantined_count)
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

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            CURRENT_VERSION.set(state["version"])
            CURRENT_EPOCH.set(state["curriculum_epoch"])
            ACTIVE_DEVICES.set(len({u["device_id"] for u in uploads}))
            SKEW_DETECTED.set(1 if skew else 0)
            AGGREGATION_WALL_CLOCK.set(time_module.time())
            AGGREGATED_UPLOADS.set(len(uploads))
            AGGREGATION_ROUNDS.labels(result="ok").inc()
            AGGREGATION_DURATION_SECONDS.observe(time_module.time() - start_time)
            if quarantined_count > 0:
                UPLOADS_QUARANTINED.inc(quarantined_count)
    else:
        if PROMETHEUS_AVAILABLE:
            AGGREGATION_ROUNDS.labels(result="error").inc()
            AGGREGATION_DURATION_SECONDS.observe(time_module.time() - start_time)
            if quarantined_count > 0:
                UPLOADS_QUARANTINED.inc(quarantined_count)
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
            if PROMETHEUS_AVAILABLE:
                AGGREGATION_ROUNDS.labels(result="error").inc()
        time.sleep(AGG_INTERVAL_S)


# ------------------------------------------------------------------ HTTP & WS layer
try:
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect

    @asynccontextmanager
    async def lifespan(app):
        global main_loop
        main_loop = asyncio.get_running_loop()
        # Security posture banner -- printed once at startup so it's always
        # visible in server logs, regardless of how uvicorn was invoked.
        # HMAC (D12) authenticates requests; it does NOT encrypt them. TLS
        # is a separate flag on the uvicorn invocation (see module docstring).
        if HMAC_SECRET:
            print("[startup] HMAC auth: ENABLED (X-HMAC-Signature required on protected paths)")
        else:
            print("[startup] HMAC auth: DISABLED (FCS_HMAC_SECRET unset -- dev mode, requests unauthenticated)")
        print("[startup] Transport encryption: NOT managed by this app -- "
              "traffic is plaintext unless uvicorn was started with "
              "--ssl-certfile/--ssl-keyfile. HMAC alone does not encrypt "
              "adapter or curriculum payloads.")
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

    @app.get("/metrics")
    def metrics():
        """Prometheus metrics endpoint."""
        if not PROMETHEUS_AVAILABLE:
            return Response(status_code=503, content="prometheus_client not installed")
        # Update gauges with current state
        state = load_state()
        if CURRENT_VERSION:
            CURRENT_VERSION.set(state["version"])
        if CURRENT_EPOCH:
            CURRENT_EPOCH.set(state["curriculum_epoch"])
        if PENDING_UPLOADS:
            PENDING_UPLOADS.set(len(list(QUEUE_PENDING.glob("*.npz"))))
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/upload")
    async def upload(request: Request):
        raw = await request.body()
        if len(raw) > MAX_UPLOAD_BYTES:
            if PROMETHEUS_AVAILABLE:
                UPLOADS_RECEIVED.labels(result="rejected_size").inc()
                UPLOAD_SIZE_BYTES.observe(len(raw))
            return Response(status_code=413,
                            content=f"Payload {len(raw)} bytes exceeds {MAX_UPLOAD_MB}MB limit")
        # HMAC verification
        if not verify_hmac(request, raw):
            if PROMETHEUS_AVAILABLE:
                UPLOADS_RECEIVED.labels(result="rejected_hmac").inc()
                UPLOAD_SIZE_BYTES.observe(len(raw))
            return Response(status_code=401, content="Invalid HMAC signature")
        name = enqueue(raw)                # no parsing here, by design
        if PROMETHEUS_AVAILABLE:
            UPLOADS_RECEIVED.labels(result="queued").inc()
            UPLOAD_SIZE_BYTES.observe(len(raw))
        return {"queued": name}

    @app.get("/adapter/latest")
    def adapter_latest(request: Request):
        if not verify_hmac(request, b""):
            if PROMETHEUS_AVAILABLE:
                ADAPTER_FETCHES.labels(result="rejected_hmac").inc()
            return Response(status_code=401, content="Invalid HMAC signature")
        state = load_state()
        if state["version"] == 0:
            if PROMETHEUS_AVAILABLE:
                ADAPTER_FETCHES.labels(result="not_found").inc()
            return Response(status_code=404,
                            content="no global adapter yet")
        path = MODELS_DIR / f"v_{state['version']:05d}.npz"
        if not path.exists():
            if PROMETHEUS_AVAILABLE:
                ADAPTER_FETCHES.labels(result="error").inc()
            return Response(status_code=404,
                            content="adapter file missing on disk")
        if PROMETHEUS_AVAILABLE:
            ADAPTER_FETCHES.labels(result="ok").inc()
        return Response(content=path.read_bytes(),
                        media_type="application/octet-stream",
                        headers={"X-Adapter-Version": str(state["version"]),
                                 "X-Curriculum-Epoch":
                                     str(state["curriculum_epoch"])})


    @app.get("/curriculum/{epoch}/manifest")
    def curriculum_manifest(epoch: int, request: Request):
        """Return manifest of shard files for a curriculum epoch."""
        if not verify_hmac(request, b""):
            if PROMETHEUS_AVAILABLE:
                CURRICULUM_MANIFEST_REQUESTS.labels(result="rejected_hmac", epoch=str(epoch)).inc()
            return Response(status_code=401, content="Invalid HMAC signature")
        curriculum_dir = BASE_DIR / "curriculum" / f"epoch_{epoch}"
        if not curriculum_dir.exists():
            if PROMETHEUS_AVAILABLE:
                CURRICULUM_MANIFEST_REQUESTS.labels(result="not_found", epoch=str(epoch)).inc()
            return Response(status_code=404,
                            content=f"curriculum epoch {epoch} not found")
        shards = sorted([f.name for f in curriculum_dir.glob("shard_*.npz")])
        if not shards:
            if PROMETHEUS_AVAILABLE:
                CURRICULUM_MANIFEST_REQUESTS.labels(result="not_found", epoch=str(epoch)).inc()
            return Response(status_code=404,
                            content=f"no shards in epoch {epoch}")
        # Compute SHA256 of each shard for integrity verification
        import hashlib
        shard_info = []
        for shard_name in shards:
            shard_path = curriculum_dir / shard_name
            sha256 = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            # Get shape info from NPZ
            try:
                with np.load(shard_path, allow_pickle=False) as z:
                    token_shape = list(z["token_ids"].shape)
                    label_shape = list(z["labels"].shape)
            except Exception:
                token_shape = [0, 0]
                label_shape = [0, 0]
            shard_info.append({
                "name": shard_name,
                "sha256": sha256,
                "token_shape": token_shape,
                "label_shape": label_shape
            })
        manifest = {
            "epoch": epoch,
            "total_shards": len(shards),
            "total_sequences": sum(s["token_shape"][0] for s in shard_info),
            "sequence_length": shard_info[0]["token_shape"][1] if shard_info else 0,
            "shards": shard_info
        }
        if PROMETHEUS_AVAILABLE:
            CURRICULUM_MANIFEST_REQUESTS.labels(result="ok", epoch=str(epoch)).inc()
        return manifest


    @app.get("/curriculum/{epoch}/{shard_name}")
    def curriculum_shard(epoch: int, shard_name: str, request: Request):
        """Stream a single curriculum shard."""
        if not verify_hmac(request, b""):
            if PROMETHEUS_AVAILABLE:
                CURRICULUM_SHARD_REQUESTS.labels(result="rejected_hmac", epoch=str(epoch)).inc()
            return Response(status_code=401, content="Invalid HMAC signature")
        # Validate shard name to prevent path traversal
        if ".." in shard_name or "/" in shard_name or "\\" in shard_name:
            if PROMETHEUS_AVAILABLE:
                CURRICULUM_SHARD_REQUESTS.labels(result="invalid_name", epoch=str(epoch)).inc()
            return Response(status_code=400, content="invalid shard name")
        shard_path = BASE_DIR / "curriculum" / f"epoch_{epoch}" / shard_name
        if not shard_path.exists():
            if PROMETHEUS_AVAILABLE:
                CURRICULUM_SHARD_REQUESTS.labels(result="not_found", epoch=str(epoch)).inc()
            return Response(status_code=404, content="shard not found")
        if PROMETHEUS_AVAILABLE:
            CURRICULUM_SHARD_REQUESTS.labels(result="ok", epoch=str(epoch)).inc()
        return Response(
            content=shard_path.read_bytes(),
            media_type="application/octet-stream",
            headers={"X-Shard-Name": shard_name}
        )
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
