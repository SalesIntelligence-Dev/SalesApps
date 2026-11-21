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

