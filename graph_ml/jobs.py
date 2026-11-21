"""
Hintergrund-Jobs, Geräte-Erkennung und RunPod-Client für /graph_ml
===================================================================
Gleiches Muster wie /forecasting (Aurora) und /computer_vision_quality:
Rechenintensives läuft nie im Request (gunicorn-Timeout 120 s), sondern als
Thread-Job mit Fortschritt und Log; die UI fragt den Status per Polling ab.

Geräte-Reihenfolge:  lokale CUDA-GPU  →  RunPod Serverless  →  CPU

Umgebungsvariablen (nur im Render-Dashboard setzen, nie im Code):
  RUNPOD_API_KEY               – RunPod API-Schlüssel (gemeinsam mit Aurora/CV)
  RUNPOD_GRAPHML_ENDPOINT_ID   – eigener Graph-ML-Worker (runpod_graphml/)
  RUNPOD_ENDPOINT_ID           – bestehender Aurora/CV-Worker (nur Status-Anzeige)
  RUNPOD_TIMEOUT               – max. Wartezeit in s (Standard 1800)
  KUMO_API_KEY                 – optional, KumoRFM (nur „gesetzt ja/nein“ wird angezeigt)
"""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import requests

log = logging.getLogger("graph_ml.jobs")

RUNPOD_BASE = "https://api.runpod.ai/v2"
POLL_INTERVAL = 3
HEALTH_TTL = 60
MAX_JOBS = 50


# ━━ Job-Store ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Job:
    id: str
    kind: str
    title: str
    status: str = "queued"          # queued · running · completed · failed
    progress: int = 0
    message: str = ""
    device: str | None = None
    log: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        end = self.finished_at or time.time()
        d["elapsed_s"] = round(end - (self.started_at or self.created_at), 1)
        return d


class JobContext:
    """Wird an die Job-Funktion übergeben: Fortschritt melden und loggen."""

    def __init__(self, job: Job, store: "JobStore"):
        self.job, self._store = job, store

    def progress(self, pct: int, msg: str = "") -> None:
        with self._store.lock:
            self.job.progress = max(0, min(100, int(pct)))
            if msg:
                self.job.message = msg
                self.job.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    def log(self, msg: str) -> None:
        with self._store.lock:
            self.job.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    def set_device(self, device: str) -> None:
        with self._store.lock:
            self.job.device = device


class JobStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def submit(self, kind: str, title: str, fn: Callable[..., Any], **kwargs) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, title=title)
        with self.lock:
            self._jobs[job.id] = job
            if len(self._jobs) > MAX_JOBS:
                for old in sorted(self._jobs.values(), key=lambda j: j.created_at)[: len(self._jobs) - MAX_JOBS]:
                    if old.status in ("completed", "failed"):
                        self._jobs.pop(old.id, None)
        ctx = JobContext(job, self)

        def runner():
            with self.lock:
                job.status, job.started_at = "running", time.time()
            try:
                result = fn(ctx, **kwargs)
                with self.lock:
                    job.result, job.status, job.progress = result, "completed", 100
            except Exception as exc:          # Job-Fehler gehören in den Job, nicht in den Server-Log allein
                log.error("Job %s (%s) fehlgeschlagen: %s\n%s", job.id, kind, exc, traceback.format_exc())
                with self.lock:
                    job.status, job.error = "failed", str(exc)
                    job.log.append(f"{time.strftime('%H:%M:%S')}  FEHLER: {exc}")
            finally:
                with self.lock:
                    job.finished_at = time.time()

        threading.Thread(target=runner, daemon=True, name=f"graphml-{kind}-{job.id}").start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 10) -> list[dict]:
        with self.lock:
            jobs = sorted(self._jobs.values(), key=lambda j: -j.created_at)[:limit]
            return [j.to_dict() for j in jobs]


store = JobStore()


# ━━ Geräte-Erkennung ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def _cached(key: str, ttl: float, fn: Callable[[], Any]) -> Any:
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = fn()
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def nvidia_gpus() -> list[dict]:
    """GPUs laut Treiber (nvidia-smi) – unabhängig davon, ob torch installiert ist."""
    def probe():
        exe = shutil.which("nvidia-smi")
        if not exe:
            return []
        try:
            out = subprocess.run(
                [exe, "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({"index": int(parts[0]), "name": parts[1], "memory_total_mb": int(float(parts[2])),
                             "memory_used_mb": int(float(parts[3])), "utilization_pct": int(float(parts[4])),
                             "driver": parts[5]})
        return gpus
    return _cached("nvidia", 30, probe)


def torch_cuda() -> dict:
    """torch + CUDA prüfen. torch wird nur importiert, wenn ein NVIDIA-Treiber da ist –
    auf Render (CPU) würde der Import sonst dauerhaft ~200 MB RAM belegen."""
    def probe():
        if importlib.util.find_spec("torch") is None:
            return {"installed": False, "cuda": False, "reason": "torch nicht installiert"}
        version = _version("torch")
        if not nvidia_gpus():
            return {"installed": True, "version": version, "cuda": False, "reason": "kein NVIDIA-Treiber"}
        try:
            import torch
            if torch.cuda.is_available():
                return {"installed": True, "version": version, "cuda": True,
                        "cuda_version": torch.version.cuda,
                        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
            return {"installed": True, "version": version, "cuda": False, "reason": "CPU-Build von torch"}
        except Exception as exc:
            return {"installed": True, "version": version, "cuda": False, "reason": str(exc)}
    return _cached("torch", 300, probe)


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


ENGINE_PACKAGES = [
    ("lightgbm", "LightGBM", "Baum-Modell"),
    ("scikit-learn", "scikit-learn", "Baselines & Metriken"),
    ("networkx", "NetworkX", "Graph-Features"),
    ("scipy", "SciPy", "Sparse-Graphen"),
    ("torch", "PyTorch", "GNN-Basis"),
    ("torch_geometric", "PyTorch Geometric", "Relational Deep Learning"),
    ("tabpfn", "TabPFN", "Tabular Foundation Model"),
    ("kumoai", "Kumo SDK", "Relationales Foundation Model"),
    ("pandas", "pandas", "Daten"),
    ("numpy", "NumPy", "Daten"),
]


def _memory() -> dict:
    def kb(path: str, key: str) -> int | None:
        try:
            with open(path) as fh:
                for line in fh:
                    if line.startswith(key):
                        return int(line.split()[1])
        except OSError:
            return None
        return None
    rss_kb = kb("/proc/self/status", "VmRSS:")
    total_kb = kb("/proc/meminfo", "MemTotal:")
    limit = None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
            if raw.isdigit() and int(raw) < (1 << 60):
                limit = int(raw) // (1024 * 1024)
                break
        except OSError:
            continue
    return {"process_rss_mb": round(rss_kb / 1024) if rss_kb else None,
            "system_total_mb": round(total_kb / 1024) if total_kb else None,
            "container_limit_mb": limit}


# ━━ RunPod ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def runpod_env() -> dict:
    return {
        "api_key_set": bool(os.environ.get("RUNPOD_API_KEY")),
        "graphml_endpoint_id": os.environ.get("RUNPOD_GRAPHML_ENDPOINT_ID") or None,
        "aurora_endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID") or None,
    }


def runpod_health(endpoint_id: str | None) -> dict | None:
    key = os.environ.get("RUNPOD_API_KEY")
    if not (key and endpoint_id):
        return None

    def probe():
        try:
            r = requests.get(f"{RUNPOD_BASE}/{endpoint_id}/health",
                             headers={"Authorization": f"Bearer {key}"}, timeout=4)
            if r.status_code != 200:
                return {"ok": False, "error": f"HTTP {r.status_code}"}
            data = r.json()
            w = data.get("workers", {})
            return {"ok": True, "workers_ready": w.get("ready", 0) + w.get("idle", 0),
                    "workers_running": w.get("running", 0), "jobs_queued": data.get("jobs", {}).get("inQueue", 0)}
        except (requests.RequestException, ValueError) as exc:
            return {"ok": False, "error": str(exc)[:120]}
    return _cached(f"rp:{endpoint_id}", HEALTH_TTL, probe)


class RunPodClient:
    """Minimaler Serverless-Client: /run senden, /status pollen (wie aurora_inference.py)."""

    def __init__(self, endpoint_id: str | None = None):
        self.key = os.environ.get("RUNPOD_API_KEY", "")
        self.endpoint = endpoint_id or os.environ.get("RUNPOD_GRAPHML_ENDPOINT_ID", "")
        self.timeout = int(os.environ.get("RUNPOD_TIMEOUT", "1800"))

    @property
    def configured(self) -> bool:
        return bool(self.key and self.endpoint)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    def submit(self, payload: dict) -> str:
        r = requests.post(f"{RUNPOD_BASE}/{self.endpoint}/run", json={"input": payload},
                          headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()["id"]

    def wait(self, job_id: str, ctx: JobContext | None = None) -> dict:
        t0 = time.time()
        last = None
        while time.time() - t0 < self.timeout:
            r = requests.get(f"{RUNPOD_BASE}/{self.endpoint}/status/{job_id}", headers=self._headers(), timeout=15)
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            if status != last and ctx:
                ctx.log(f"RunPod-Status: {status}")
                last = status
            if status == "COMPLETED":
                out = data.get("output") or {}
                if isinstance(out, dict) and out.get("error"):
                    raise RuntimeError(f"RunPod-Worker: {out['error']}")
                return out
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod-Job {status}: {data.get('error', '')}")
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(f"RunPod-Job nach {self.timeout} s nicht fertig")


# ━━ System-Status (für Header-Chips und Detail-Panel) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def host_label() -> str:
    if os.environ.get("RENDER"):
        return f"Render · {os.environ.get('RENDER_SERVICE_NAME', 'Web Service')}"
    return f"Lokal · {platform.node()}"


def pick_device() -> str:
    if torch_cuda().get("cuda"):
        return "local_gpu"
    if RunPodClient().configured:
        return "runpod"
    return "cpu"


def system_info(with_health: bool = True) -> dict:
    env = runpod_env()
    gpus = nvidia_gpus()
    tc = torch_cuda()
    packages = [{"package": p, "name": n, "purpose": purpose, "version": _version(p)}
                for p, n, purpose in ENGINE_PACKAGES]
    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        load = None
    gpu_engine_ready = bool(tc.get("cuda")) or bool(env["api_key_set"] and env["graphml_endpoint_id"])
    info = {
        "host": host_label(),
        "on_render": bool(os.environ.get("RENDER")),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "cpu_count": os.cpu_count(),
        "load_avg": load,
        "memory": _memory(),
        "gpu": {"nvidia": gpus, "torch": tc},
        "runpod": {
            "api_key_set": env["api_key_set"],
            "graphml": {"endpoint_set": bool(env["graphml_endpoint_id"]),
                        "endpoint_id": _mask(env["graphml_endpoint_id"]),
                        "health": runpod_health(env["graphml_endpoint_id"]) if with_health else None},
            "aurora": {"endpoint_set": bool(env["aurora_endpoint_id"]),
                       "endpoint_id": _mask(env["aurora_endpoint_id"]),
                       "health": runpod_health(env["aurora_endpoint_id"]) if with_health else None},
        },
        "kumo": {"api_key_set": bool(os.environ.get("KUMO_API_KEY")), "sdk": _version("kumoai")},
        "packages": packages,
        "preferred_device": pick_device(),
    }
    info["chips"] = _chips(info, gpu_engine_ready)
    return info


def _mask(v: str | None) -> str | None:
    if not v:
        return None
    return v[:4] + "…" + v[-3:] if len(v) > 8 else "…"


def _chips(info: dict, gpu_ready: bool) -> list[dict]:
    mem = info["memory"]
    gpus = info["gpu"]["nvidia"]
    tc = info["gpu"]["torch"]
    rp = info["runpod"]
    if tc.get("cuda"):
        gpu_chip = {"state": "ok", "value": tc["devices"][0], "hint": "Lokale CUDA-GPU über PyTorch nutzbar"}
    elif rp["api_key_set"] and rp["graphml"]["endpoint_set"]:
        h = rp["graphml"]["health"] or {}
        gpu_chip = {"state": "ok" if h.get("ok", True) else "warn", "value": "RunPod Serverless",
                    "hint": "GPU-Jobs laufen auf dem Graph-ML-Endpoint"}
    elif gpus:
        best = max(gpus, key=lambda g: g["memory_total_mb"])
        gpu_chip = {"state": "warn", "value": f"{best['name'].replace('NVIDIA GeForce ', '')} (ohne torch)",
                    "hint": "GPU vorhanden, PyTorch mit CUDA fehlt – venv für die Vorberechnung anlegen"}
    else:
        gpu_chip = {"state": "off", "value": "nicht konfiguriert",
                    "hint": "RUNPOD_GRAPHML_ENDPOINT_ID setzen, um GNN/TabPFN auf GPU zu rechnen"}
    tabpfn = next(p for p in info["packages"] if p["package"] == "tabpfn")
    return [
        {"id": "cpu", "label": "CPU", "state": "ok",
         "value": f"{info['cpu_count']} Kerne · {mem['process_rss_mb'] or '–'} MB",
         "hint": f"{info['host']} – LightGBM, Regeln und Graph-Features laufen hier"},
        {"id": "gpu", "label": "GPU", **gpu_chip},
        {"id": "kumo", "label": "KumoRFM", "state": "ok" if info["kumo"]["api_key_set"] else "off",
         "value": "Key gesetzt" if info["kumo"]["api_key_set"] else "kein Key",
         "hint": "Relationales Foundation Model – optional, Nutzungsbedingungen vorab klären"},
        {"id": "tabpfn", "label": "TabPFN",
         "state": "ok" if tabpfn["version"] or gpu_ready else "off",
         "value": tabpfn["version"] or ("über GPU-Worker" if gpu_ready else "nicht verfügbar"),
         "hint": "Tabular Foundation Model – Lizenz vor kommerzieller Nutzung prüfen"},
    ]


# ━━ Selbsttest: zeigt, auf welchem Gerät gerechnet würde ━━━━━━━━━━━━━━━━━━━━━━━━━

def _bench_numpy(n: int = 1024, reps: int = 3) -> float:
    a = np.random.default_rng(0).standard_normal((n, n), dtype=np.float32)
    a @ a
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ a
    dt = (time.perf_counter() - t0) / reps
    return 2 * n ** 3 / dt / 1e9


def _bench_torch_cuda(n: int = 4096, reps: int = 10) -> dict:
    import torch
    dev = torch.device("cuda")
    a = torch.randn(n, n, device=dev)
    torch.cuda.synchronize()
    a @ a
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ a
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    return {"gflops": round(2 * n ** 3 / dt / 1e9, 1), "device": torch.cuda.get_device_name(0)}


def run_selftest(ctx: JobContext) -> dict:
    results = []
    ctx.progress(5, "Prüfe Laufzeitumgebung …")
    info = system_info(with_health=False)
    ctx.log(f"Host: {info['host']} · Python {info['python']} · {info['cpu_count']} CPU-Kerne")

    ctx.progress(20, "CPU-Benchmark (NumPy, 1024×1024 Matrixmultiplikation) …")
    ctx.set_device("CPU")
    gf = _bench_numpy()
    results.append({"device": "CPU", "where": info["host"], "gflops": round(gf, 1), "ok": True})
    ctx.log(f"CPU: {gf:.1f} GFLOP/s")

    tc = torch_cuda()
    if tc.get("cuda"):
        ctx.progress(50, "Lokale GPU-Benchmark (PyTorch CUDA) …")
        ctx.set_device("Lokale GPU")
        try:
            r = _bench_torch_cuda()
            results.append({"device": "Lokale GPU", "where": r["device"], "gflops": r["gflops"], "ok": True})
            ctx.log(f"GPU {r['device']}: {r['gflops']} GFLOP/s")
        except Exception as exc:
            results.append({"device": "Lokale GPU", "where": "–", "ok": False, "error": str(exc)})
    else:
        gpus = nvidia_gpus()
        note = (f"{len(gpus)} NVIDIA-GPU(s) erkannt ({', '.join(g['name'] for g in gpus)}), "
                f"aber {tc.get('reason')}") if gpus else tc.get("reason", "keine lokale GPU")
        results.append({"device": "Lokale GPU", "where": "–", "ok": False, "skipped": True, "error": note})
        ctx.log(f"Lokale GPU übersprungen: {note}")

    client = RunPodClient()
    if client.configured:
        ctx.progress(65, "RunPod Graph-ML-Endpoint: Selbsttest-Job senden …")
        ctx.set_device("RunPod")
        try:
            rp_id = client.submit({"job_type": "graphml_selftest"})
            ctx.log(f"RunPod-Job {rp_id} angenommen")
            out = client.wait(rp_id, ctx)
            results.append({"device": "RunPod GPU", "where": out.get("gpu", "RunPod"), "gflops": out.get("gflops"),
                            "ok": True})
        except Exception as exc:
            results.append({"device": "RunPod GPU", "where": "–", "ok": False, "error": str(exc)[:200]})
            ctx.log(f"RunPod: {exc}")
    else:
        results.append({"device": "RunPod GPU", "where": "–", "ok": False, "skipped": True,
                        "error": "RUNPOD_GRAPHML_ENDPOINT_ID nicht gesetzt (Worker folgt mit Prompt 3)"})
        ctx.log("RunPod übersprungen: Graph-ML-Endpoint nicht konfiguriert")

    ctx.progress(95, "Fertig – wähle bevorzugtes Gerät")
    best = max((r for r in results if r.get("ok") and r.get("gflops")), key=lambda r: r["gflops"])
    ctx.set_device(best["device"])
    return {"results": results, "preferred": best["device"], "preferred_device": pick_device()}
