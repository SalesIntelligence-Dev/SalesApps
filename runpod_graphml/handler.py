"""
RunPod Serverless Worker – GraphReady (/graph_ml)
=================================================
Eigener Endpoint (RUNPOD_GRAPHML_ENDPOINT_ID), getrennt vom Aurora/CV-Worker
in runpod/. Das Image enthält das Paket graph_ml inklusive der Demo-Daten
(Online Retail II als npz) – die Anfrage bleibt dadurch winzig.

job_type
  graphml_selftest   GPU-Benchmark                         → {"gpu", "gflops"}
  graphml_engine     eine Engine für einen Use Case rechnen → {"engine_result", "gpu"}
  graphml_backtest   kompletten Rückblick eines Use Case    → {"result", "gpu"}
  graphml_pipeline   Playbooks anwenden + Checks neu (+ Rückblick) → {"result", "gpu"}
  graphml_gnn_train / graphml_lightgcn / graphml_tabpfn – Kurzformen für graphml_engine
"""
from __future__ import annotations

import logging
import sys
import time
import traceback

sys.path.insert(0, "/app")
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("graphml-worker")

ALIASES = {"graphml_gnn_train": "gnn", "graphml_lightgcn": "lightgcn", "graphml_tabpfn": "tabpfn"}


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        return "unbekannt"


def selftest() -> dict:
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 4096
    a = torch.randn(n, n, device=dev)
    a @ a
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        a @ a
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    return {"gpu": _gpu_name(), "gflops": round(2 * n ** 3 / dt / 1e9, 1)}


def _private(r: dict) -> dict:
    """numpy-Hilfsfelder (_scores, _ap, …) als Listen übertragen; runtime.run_remote baut sie zurück."""
    out = {}
    for k in ("_scores", "_ap", "_series_err", "_series_tot", "_yhat", "_ytrue"):
        if k in r:
            out[k] = [round(float(x), 5) for x in r[k].ravel()] if hasattr(r[k], "ravel") else r[k]
            if k in ("_yhat", "_ytrue"):
                out[k + "_shape"] = list(r[k].shape)
    if "_recs" in r:
        out["_recs"] = [[int(p) for p in rec] for rec in r["_recs"]]
    if "_hit_stats" in r:
        out["_hit_stats"] = r["_hit_stats"]
    return out


def run_engine(use_case: str, engine_id: str) -> dict:
    from graph_ml import backtest as bt
    from graph_ml import evaluation as ev
    from graph_ml.engines import make_engines
    task = bt.get_task(use_case)
    engines = {e.id: e for e in make_engines(task)}
    if engine_id not in engines:
        raise ValueError(f"Engine {engine_id} unterstützt {use_case} nicht")
    rule_scores = None
    if task.kind == "binary" and engine_id != "rule":
        rule_scores = ev.run_engine(task, engines["rule"])["_scores"]
    r = ev.run_engine(task, engines[engine_id], rule_scores=rule_scores)
    priv = _private(r)
    clean = ev._clean(r)
    clean["private"] = priv
    return clean


def handler(job):
    inp = job.get("input") or {}
    jt = inp.get("job_type", "")
    log.info("Job %s: %s", job.get("id"), jt)
    try:
        if jt == "graphml_selftest":
            return selftest()
        if jt == "graphml_engine" or jt in ALIASES:
            engine_id = inp.get("engine") or ALIASES[jt]
            return {"engine_result": run_engine(inp["use_case"], engine_id), "gpu": _gpu_name()}
        if jt == "graphml_pipeline":
            from graph_ml import playbooks as pb
            res = pb.run_variant(inp.get("steps") or [], fx=float(inp.get("fx") or pb.DEFAULT_FX),
                                 governance=inp.get("governance") or {}, backtest=bool(inp.get("backtest")))
            return {"result": res, "gpu": _gpu_name()}
        if jt == "graphml_backtest":
            from graph_ml import evaluation as ev
            return {"result": ev.run_use_case(inp["use_case"]), "gpu": _gpu_name()}
        return {"error": f"Unbekannter job_type '{jt}'. Unterstützt: graphml_selftest, graphml_engine, "
                         "graphml_backtest, graphml_pipeline, graphml_gnn_train, graphml_lightgcn, graphml_tabpfn"}
    except Exception as exc:
        log.error("%s\n%s", exc, traceback.format_exc())
        return {"error": str(exc)[:500]}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
