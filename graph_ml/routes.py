"""
Flask-Blueprint für /graph_ml (GraphReady · Graph-ML-Readiness-Plattform).
Seite unter /graph_ml, API unter /graph_ml/api/… – vollständig isoliert
von den übrigen SalesApps-Routen.
"""
from __future__ import annotations

import logging
from pathlib import Path

import json

from flask import Blueprint, Response, jsonify, render_template, request

from .data_loader import get_registry
from .jobs import run_selftest, store, system_info
from .usecases import list_use_cases

log = logging.getLogger("graph_ml.routes")

graph_bp = Blueprint("graph_ml", __name__, url_prefix="", template_folder="../templates")

UPLOAD_EXT = {".csv", ".txt", ".xlsx", ".xls", ".parquet", ".json"}


def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# ── Seite ─────────────────────────────────────────────────────────────────────

@graph_bp.route("/graph_ml")
@graph_bp.route("/graph_ml/")
def graph_ml_page():
    return render_template("graph_ml.html")


# ── System & Jobs ─────────────────────────────────────────────────────────────

@graph_bp.route("/graph_ml/api/system")
def api_system():
    info = system_info(with_health=request.args.get("health", "1") != "0")
    info["data_memory_mb"] = get_registry().memory_mb()
    return jsonify(info)


@graph_bp.route("/graph_ml/api/jobs/selftest", methods=["POST"])
def api_selftest():
    job = store.submit("selftest", "System-Selbsttest (CPU · GPU · RunPod)", run_selftest)
    return jsonify(job.to_dict()), 202


@graph_bp.route("/graph_ml/api/jobs")
def api_jobs():
    return jsonify(store.list())


@graph_bp.route("/graph_ml/api/jobs/<job_id>")
def api_job(job_id: str):
    job = store.get(job_id)
    if not job:
        return _err("Job nicht gefunden", 404)
    return jsonify(job.to_dict())


# ── Use Cases ─────────────────────────────────────────────────────────────────

@graph_bp.route("/graph_ml/api/usecases")
def api_usecases():
    return jsonify(list_use_cases())


# ── Konnektoren (Modul 1) ─────────────────────────────────────────────────────

@graph_bp.route("/graph_ml/api/datasets")
def api_datasets():
    return jsonify(get_registry().list())


@graph_bp.route("/graph_ml/api/datasets/<ds_id>")
def api_dataset(ds_id: str):
    reg = get_registry()
    if not reg.has(ds_id):
        return _err("Datensatz nicht gefunden", 404)
    try:
        return jsonify(reg.overview(ds_id))
    except Exception as exc:
        log.exception("Übersicht %s", ds_id)
        return _err(f"Übersicht fehlgeschlagen: {exc}", 500)


@graph_bp.route("/graph_ml/api/datasets/<ds_id>", methods=["DELETE"])
def api_dataset_delete(ds_id: str):
    if not get_registry().delete(ds_id):
        return _err("Nur eigene Uploads können entfernt werden", 400)
    return jsonify({"ok": True})


@graph_bp.route("/graph_ml/api/datasets/<ds_id>/preview")
def api_preview(ds_id: str):
    reg = get_registry()
    if not reg.has(ds_id):
        return _err("Datensatz nicht gefunden", 404)
    try:
        return jsonify(reg.preview(ds_id, request.args.get("table"), request.args.get("n", 50, type=int)))
    except Exception as exc:
        log.exception("Vorschau %s", ds_id)
        return _err(f"Vorschau fehlgeschlagen: {exc}", 500)


@graph_bp.route("/graph_ml/api/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("files")
    if not files:
        return _err("Keine Datei übermittelt")
    payload = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in UPLOAD_EXT:
            return _err(f"Dateityp {ext or '?'} nicht unterstützt (CSV, XLSX, Parquet, JSON)")
        payload.append((f.filename, f.read()))
    try:
        ds = get_registry().add_upload(payload)
    except ImportError:
        return _err("Parquet-Upload benötigt pyarrow – bitte CSV oder XLSX verwenden")
    except ValueError as exc:
        return _err(str(exc))
    except Exception as exc:
        log.exception("Upload")
        return _err(f"Datei konnte nicht gelesen werden: {exc}")
    return jsonify(ds.card()), 201


# ── Graph-Mapping (Modul 2) ───────────────────────────────────────────────────

@graph_bp.route("/graph_ml/api/datasets/<ds_id>/mapping")
def api_mapping(ds_id: str):
    reg = get_registry()
    if not reg.has(ds_id):
        return _err("Datensatz nicht gefunden", 404)
    try:
        # Eingebaute Datensätze kommen aus dem Artefakt; neu rechnen nur für Uploads
        refresh = request.args.get("refresh") == "1" and not reg.get(ds_id).builtin
        return jsonify(reg.mapping(ds_id, refresh=refresh))
    except Exception as exc:
        log.exception("Mapping %s", ds_id)
        return _err(f"Graph-Mapping fehlgeschlagen: {exc}", 500)


@graph_bp.route("/graph_ml/api/datasets/<ds_id>/customers")
def api_customers(ds_id: str):
    if ds_id != "online_retail_ii":
        return jsonify([])
    return jsonify(get_registry().sample_customers())


@graph_bp.route("/graph_ml/api/datasets/<ds_id>/subgraph")
def api_subgraph(ds_id: str):
    if ds_id != "online_retail_ii":
        return _err("Beispiel-Subgraph nur für Online Retail II verfügbar")
    reg = get_registry()
    cid = request.args.get("customer_id", type=int)
    if request.args.get("random") == "1" or cid is None:
        cid = reg.random_customer(request.args.get("seed", type=int))
    try:
        return jsonify(reg.customer_subgraph(cid, max_nodes=min(request.args.get("max_nodes", 150, type=int), 150)))
    except KeyError as exc:
        return _err(str(exc).strip("'"), 404)


# ── Readiness-Checks (Modul 3) ────────────────────────────────────────────────

@graph_bp.route("/graph_ml/api/checks", methods=["POST"])
def api_checks():
    body = request.get_json(silent=True) or {}
    ds_id = body.get("dataset", "online_retail_ii")
    reg = get_registry()
    if not reg.has(ds_id):
        return _err("Datensatz nicht gefunden", 404)
    try:
        stage = int(body.get("stage", 2))
    except (TypeError, ValueError):
        return _err("Stufe muss 1, 2 oder 3 sein")
    use_cases = body.get("use_cases") or None
    try:
        return jsonify(reg.checks(ds_id, stage, use_cases))
    except Exception as exc:
        log.exception("Checks %s", ds_id)
        return _err(f"Readiness-Checks fehlgeschlagen: {exc}", 500)


@graph_bp.route("/graph_ml/api/datasets/<ds_id>/profile")
def api_profile(ds_id: str):
    """Kennzahlen-Profil (Stufe 2) als JSON-Download – zeigt, was beim Profilcheck das Unternehmen verlässt."""
    reg = get_registry()
    if not reg.has(ds_id):
        return _err("Datensatz nicht gefunden", 404)
    prof = reg.profile(ds_id)
    if prof is None:
        return _err("Für ein reines Schema gibt es kein Profil – Stufe 2 braucht Daten oder einen Profil-Export")
    prof = {k: v for k, v in prof.items() if k not in ("source_version", "artifact_version")}
    name = f"graphready_profil_{ds_id}.json"
    return Response(json.dumps(prof, ensure_ascii=False, indent=1), mimetype="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ── Rückblick & Baseline-Vergleich (Modul 4) ──────────────────────────────────

def _live_mode() -> dict:
    """Wo „Live neu berechnen“ rechnet: lokal im Prozess oder (auf Render) nur über den GPU-Worker."""
    from .engines.base import on_render
    from .engines.runtime import runpod_ready
    if on_render():
        return {"mode": "runpod" if runpod_ready() else "off",
                "note": ("Live-Rechnung über den Graph-ML-Worker auf RunPod" if runpod_ready() else
                         "Auf Render nur über den GPU-Worker (RUNPOD_GRAPHML_ENDPOINT_ID) – angezeigt wird die "
                         "Vorberechnung")}
    return {"mode": "local", "note": "Live-Rechnung lokal; GPU-Engines nur mit CUDA-Umgebung (.venv)"}


@graph_bp.route("/graph_ml/api/backtest")
def api_backtest_summary():
    from . import evaluation as ev
    from .engines import engine_catalog
    return jsonify({"summary": ev.summary(), "engines": engine_catalog(), "live": _live_mode(),
                    "use_cases": ev.UC_ORDER, "not_measurable": ev.NOT_MEASURABLE})


@graph_bp.route("/graph_ml/api/backtest/<uc>")
def api_backtest(uc: str):
    from . import evaluation as ev
    if uc not in ev.UC_ORDER:
        return _err("Für diesen Use Case gibt es keinen Rückblick", 404)
    res = ev.load(uc)
    if not res:
        return _err("Noch nicht vorberechnet – python -m graph_ml.precompute ausführen", 404)
    return jsonify(res)


def _run_backtest_job(ctx, uc: str):
    from . import evaluation as ev
    mode = _live_mode()["mode"]
    if mode == "runpod":
        from .jobs import RunPodClient
        client = RunPodClient()
        ctx.set_device("RunPod")
        job_id = client.submit({"job_type": "graphml_backtest", "use_case": uc})
        ctx.log(f"RunPod-Job {job_id} gestartet")
        out = client.wait(job_id, ctx)
        res = out["result"]
        res["host"] = f"RunPod · {out.get('gpu', 'GPU')}"
    else:
        ctx.set_device("Lokal")
        res = ev.run_use_case(uc, ctx=ctx, previous=ev.load(uc))
    res["live"] = True
    ev.save(res)
    get_registry()._checks.clear()
    return {"use_case": uc, "best": res.get("best"), "vorhersagekraft": res.get("vorhersagekraft")}


@graph_bp.route("/graph_ml/api/backtest/<uc>/run", methods=["POST"])
def api_backtest_run(uc: str):
    from . import evaluation as ev
    if uc not in ev.UC_ORDER:
        return _err("Für diesen Use Case gibt es keinen Rückblick", 404)
    live = _live_mode()
    if live["mode"] == "off":
        return _err(live["note"], 409)
    job = store.submit("backtest", f"Rückblick {uc}", _run_backtest_job, uc=uc)
    return jsonify(job.to_dict()), 202


# ── Fix-Playbooks & Business Case (Modul 5) ───────────────────────────────────

def _recommended_vid() -> str | None:
    from . import playbooks as pb
    p = pb.ARTIFACTS / "pipeline_recommended.json"
    return json.loads(p.read_text())["variant"] if p.exists() else None


@graph_bp.route("/graph_ml/api/playbooks")
def api_playbooks():
    from . import playbooks as pb
    rec = _recommended_vid()
    recres = pb.load_variant(rec) if rec else None
    checks = get_registry().checks("online_retail_ii", 2)
    req = pb.data_request(checks)
    return jsonify({
        "catalog": pb.catalog(), "order": pb.ORDER, "effects": pb.single_effects() or {},
        "recommended": {"variant": rec, "steps": pb.recommended_steps(),
                        "comparison": recres["comparison"] if recres else None,
                        "effects": recres["effects"] if recres else None,
                        "backtest": recres.get("backtest") if recres else None},
        "baseline": pb.baseline_checks(), "data_request": {"sections": req["sections"]},
        "live": _live_mode(), "governance_items": pb.GOV_ITEMS,
    })


def _apply_job(ctx, steps, fx, governance, backtest):
    from . import playbooks as pb
    if _live_mode()["mode"] == "runpod":
        from .jobs import RunPodClient
        client = RunPodClient()
        jid = client.submit({"job_type": "graphml_pipeline", "steps": steps, "fx": fx, "governance": governance,
                             "backtest": backtest})
        res = client.wait(jid, ctx)["result"]
        pb.save_variant(res)
    else:
        res = pb.run_variant(steps, fx=fx, governance=governance, backtest=backtest, ctx=ctx)
    return {"variant": res["variant"]}


@graph_bp.route("/graph_ml/api/playbooks/apply", methods=["POST"])
def api_playbooks_apply():
    """Playbooks anwenden → Checks neu (Stufe 2), optional Rückblick (Stufe 3). Vorberechnete Versionen sofort."""
    from . import playbooks as pb
    body = request.get_json(silent=True) or {}
    steps = pb.normalize_steps(body.get("steps") or [])
    fx = float(body.get("fx") or pb.DEFAULT_FX)
    governance = {k: bool(v) for k, v in (body.get("governance") or {}).items() if k in pb.GOV_ITEMS}
    backtest = bool(body.get("backtest"))
    vid = pb.variant_id(steps, fx)
    cached = pb.load_variant(vid)
    if cached and (not backtest or cached.get("backtest")):
        before = pb.baseline_checks(governance)
        after = pb.checks_with_governance(cached["after"]["profile"], governance, vid) if governance else \
            cached["after"]["checks"]
        return jsonify({"variant": vid, "cached": True, "steps": steps, "effects": cached["effects"],
                        "comparison": pb.compare(before, after), "backtest": cached.get("backtest"),
                        "after_counts": after["counts"], "before_counts": before["counts"]})
    if not steps and governance:
        before = pb.baseline_checks()
        after = pb.baseline_checks(governance)
        return jsonify({"variant": "v0", "cached": True, "steps": [], "effects": [],
                        "comparison": pb.compare(before, after), "after_counts": after["counts"],
                        "before_counts": before["counts"]})
    live = _live_mode()
    if live["mode"] == "off":
        return _err("Diese Kombination ist nicht vorberechnet. " + live["note"], 409)
    job = store.submit("playbooks", f"Playbooks {vid}", _apply_job, steps=steps, fx=fx, governance=governance,
                       backtest=backtest)
    return jsonify({"variant": vid, "cached": False, "job": job.to_dict()}), 202


@graph_bp.route("/graph_ml/api/playbooks/result/<vid>")
def api_playbooks_result(vid: str):
    from . import playbooks as pb
    res = pb.load_variant(vid)
    if not res:
        return _err("Pipeline-Version nicht gefunden", 404)
    gov = res.get("governance") or {}
    return jsonify({"variant": vid, "cached": True, "steps": res["steps"], "effects": res["effects"],
                    "comparison": res["comparison"], "backtest": res.get("backtest"),
                    "after_counts": res["after"]["checks"]["counts"], "before_counts": res["before"]["counts"],
                    "governance": gov})


@graph_bp.route("/graph_ml/api/playbooks/datarequest.<fmt>")
def api_data_request(fmt: str):
    from . import playbooks as pb
    req = pb.data_request(get_registry().checks("online_retail_ii", 2))
    if fmt == "csv":
        return Response(req["csv"], mimetype="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="graphready_datenanforderung.csv"'})
    return Response(req["markdown"], mimetype="text/markdown",
                    headers={"Content-Disposition": 'attachment; filename="graphready_datenanforderung.md"'})


@graph_bp.route("/graph_ml/api/business_case")
def api_business_case():
    from . import business_case as bc
    variant = request.args.get("variant", "v0")
    if variant == "recommended":
        variant = _recommended_vid() or "v0"
    out = bc.overview(variant)
    out["available_variants"] = ["v0"] + ([_recommended_vid()] if _recommended_vid() else [])
    return jsonify(out)

