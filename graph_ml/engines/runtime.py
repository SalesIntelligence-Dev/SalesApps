"""
Wo rechnen GPU-Engines?  lokale CUDA-GPU (z. B. .venv auf der RTX 3060)  →  RunPod  →  nicht verfügbar

Im Render-Webprozess wird torch nie importiert (≈ 200 MB RAM). Dort laufen
GPU-Engines ausschließlich über den eigenen Graph-ML-Worker auf RunPod, der
dieselbe Engine-Logik und die Demo-Daten im Image mitbringt.
"""
from __future__ import annotations

import importlib.util
import os

from .base import on_render

_LOCAL: tuple[bool, str] | None = None


def local_gpu() -> tuple[bool, str]:
    """(verfügbar, Gerätename) – torch nur importieren, wenn ein NVIDIA-Treiber da ist und nicht auf Render."""
    global _LOCAL
    if _LOCAL is not None:
        return _LOCAL
    if on_render() or os.environ.get("GRAPHML_DISABLE_LOCAL_GPU"):
        _LOCAL = (False, "")
        return _LOCAL
    from ..jobs import nvidia_gpus
    if not nvidia_gpus() or importlib.util.find_spec("torch") is None:
        _LOCAL = (False, "")
        return _LOCAL
    import torch
    if not torch.cuda.is_available():
        _LOCAL = (False, "")
        return _LOCAL
    # leistungsstärkste GPU wählen (z. B. RTX 3060 statt GTX 1650)
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    torch.cuda.set_device(best)
    _LOCAL = (True, torch.cuda.get_device_name(best))
    return _LOCAL


def torch_device():
    import torch
    ok, _ = local_gpu()
    if ok:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def runpod_ready() -> bool:
    from ..jobs import RunPodClient
    return RunPodClient().configured


def gpu_backend() -> str | None:
    if local_gpu()[0]:
        return "local"
    if runpod_ready():
        return "runpod"
    return None


def run_remote(use_case: str, engine_id: str, ctx=None) -> dict:
    """Engine auf dem Graph-ML-Worker rechnen lassen; liefert das fertige Engine-Ergebnis."""
    from ..jobs import RunPodClient
    client = RunPodClient()
    job_id = client.submit({"job_type": "graphml_engine", "use_case": use_case, "engine": engine_id})
    if ctx:
        ctx.log(f"RunPod-Job {job_id}: {engine_id} für {use_case}")
    out = client.wait(job_id, ctx)
    if "engine_result" not in out:
        raise RuntimeError(f"Unerwartete Worker-Antwort: {list(out)[:5]}")
    res = out["engine_result"]
    res.setdefault("meta", {})["device"] = f"RunPod · {out.get('gpu', 'GPU')}"
    # Hilfsfelder (Scores je Fall usw.) für Top-Listen und gepaartes Bootstrap zurück in numpy wandeln
    import numpy as np
    priv = res.pop("private", {}) or {}
    for k, v in priv.items():
        if k.endswith("_shape") or k in ("_recs", "_hit_stats"):
            res[k] = v
            continue
        arr = np.asarray(v, dtype=float)
        if f"{k}_shape" in priv:
            arr = arr.reshape(priv[f"{k}_shape"])
        res[k] = arr
    return res
