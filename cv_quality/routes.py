"""
Flask Blueprint for CV Quality app.
All routes are under /computer_vision_quality and /cv/api/
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from flask import (
    Blueprint, jsonify, render_template, request,
    send_file, abort, Response,
)

from .store import get_store, new_id, InferenceResult
from .job_runner import (
    IMAGES_DIR, MODELS_DIR, OUTPUTS_DIR,
    run_patchcore_training, run_demo_training,
    run_foundad_training, submit_job,
)

logger = logging.getLogger("cv_quality.routes")

cv_bp = Blueprint("cv_quality", __name__,
                  url_prefix="",
                  template_folder="../templates")

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

for d in [IMAGES_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Main entry point ──────────────────────────────────────────────────────────
@cv_bp.route("/computer_vision_quality")
@cv_bp.route("/computer_vision_quality/")
def cv_main():
    return render_template("computer_vision_quality.html")


# ── Health ────────────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/health")
def cv_health():
    try:
        import torch
        torch_ok = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch_ok = False
        device = "cpu (no torch)"
    return jsonify({
        "status": "ok",
        "torch_available": torch_ok,
        "device": device,
        "runpod_enabled": os.environ.get("RUNPOD_ENABLED", "false"),
    })


# ── Projects ──────────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/projects", methods=["GET"])
def list_projects():
    store = get_store()
    projects = store.list_projects()
    return jsonify([_project_dict(p) for p in projects])


@cv_bp.route("/cv/api/projects", methods=["POST"])
def create_project():
    data = request.get_json(force=True) or {}
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify({"error": "Project name required"}), 400
    if len(name) > 100:
        return jsonify({"error": "Name too long (max 100 chars)"}), 400
    description = str(data.get("description", "")).strip()[:500]
    store = get_store()
    project = store.create_project(name, description)
    return jsonify(_project_dict(project)), 201


@cv_bp.route("/cv/api/projects/<pid>", methods=["GET"])
def get_project(pid: str):
    store = get_store()
    p = store.get_project(pid)
    if not p:
        return jsonify({"error": "Project not found"}), 404
    return jsonify(_project_dict(p))


@cv_bp.route("/cv/api/projects/<pid>/images", methods=["GET"])
def list_images(pid: str):
    store = get_store()
    p = store.get_project(pid)
    if not p:
        return jsonify({"error": "Project not found"}), 404
    normal_dir = IMAGES_DIR / pid / "normal"
    defect_dir = IMAGES_DIR / pid / "defect"
    images = []
    for label, d in [("normal", normal_dir), ("defect", defect_dir)]:
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in ALLOWED_IMAGE_EXT:
                    images.append({"name": f.name, "label": label, "path": str(f)})
    return jsonify({"images": images, "total": len(images)})


# ── Image upload ──────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/projects/<pid>/upload", methods=["POST"])
def upload_images(pid: str):
    store = get_store()
    p = store.get_project(pid)
    if not p:
        return jsonify({"error": "Project not found"}), 404

    label = request.form.get("label", "normal")
    if label not in ("normal", "defect"):
        return jsonify({"error": "label must be 'normal' or 'defect'"}), 400

    dest_dir = IMAGES_DIR / pid / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    errors = []

    files = request.files.getlist("images")
    for f in files:
        if not f.filename:
            continue
        filename = _safe_filename(f.filename)
        ext = Path(filename).suffix.lower()

        # Handle ZIP
        if ext == ".zip":
            try:
                zf = zipfile.ZipFile(io.BytesIO(f.read()))
                for name in zf.namelist():
                    if Path(name).suffix.lower() in ALLOWED_IMAGE_EXT:
                        safe = _safe_filename(Path(name).name)
                        dest = dest_dir / safe
                        dest.write_bytes(zf.read(name))
                        saved.append(safe)
            except Exception as e:
                errors.append(f"ZIP error: {e}")
        elif ext in ALLOWED_IMAGE_EXT:
            dest = dest_dir / filename
            f.save(str(dest))
            saved.append(filename)
        else:
            errors.append(f"Unsupported format: {filename}")

    # Update counts
    normal_count = len(list((IMAGES_DIR / pid / "normal").glob("*"))) if (IMAGES_DIR / pid / "normal").exists() else 0
    defect_count = len(list((IMAGES_DIR / pid / "defect").glob("*"))) if (IMAGES_DIR / pid / "defect").exists() else 0
    store.update_project_counts(pid, normal_count, defect_count)

    return jsonify({"saved": len(saved), "files": saved, "errors": errors})


# ── Serve uploaded images ─────────────────────────────────────────────────────
@cv_bp.route("/cv/api/images/<path:img_path>")
def serve_image(img_path: str):
    # Security: only serve from /tmp/cv_quality/
    full = Path("/tmp/cv_quality") / img_path
    full = full.resolve()
    if not str(full).startswith("/tmp/cv_quality"):
        abort(403)
    if not full.exists():
        abort(404)
    return send_file(str(full))


# ── Training ──────────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/projects/<pid>/train", methods=["POST"])
def start_training(pid: str):
    store = get_store()
    p = store.get_project(pid)
    if not p:
        return jsonify({"error": "Project not found"}), 404

    data = request.get_json(force=True) or {}
    algorithm = data.get("algorithm", "patchcore")
    backbone = data.get("backbone", "resnet18")
    coreset_ratio = float(data.get("coreset_ratio", 0.1))
    k_neighbors = int(data.get("k_neighbors", 9))
    input_size = int(data.get("input_size", 224))
    demo_mode = bool(data.get("demo_mode", False))
    # FoundAD params
    n_shot = int(data.get("n_shot", 4))
    encoder = data.get("encoder", "dinov2_vitb14")
    steps = int(data.get("steps", 200))

    job = store.create_job(
        project_id=pid,
        model_name=data.get("model_name", f"{algorithm}-{backbone}"),
        algorithm=algorithm,
        backbone=backbone if algorithm == "patchcore" else encoder,
    )

    if demo_mode:
        submit_job(run_demo_training, job_id=job.id, project_id=pid)
    elif algorithm == "patchcore":
        normal_dir = IMAGES_DIR / pid / "normal"
        if not normal_dir.exists() or not any(normal_dir.iterdir()):
            store.get_job(job.id)  # ensure job exists
            return jsonify({"error": "No normal images found. Upload images first."}), 400
        image_paths = [str(f) for f in normal_dir.iterdir()
                       if f.suffix.lower() in ALLOWED_IMAGE_EXT]
        submit_job(
            run_patchcore_training,
            job_id=job.id, project_id=pid,
            image_paths=image_paths,
            backbone=backbone,
            coreset_ratio=coreset_ratio,
            k_neighbors=k_neighbors,
            input_size=input_size,
        )
    elif algorithm == "foundad":
        normal_dir = IMAGES_DIR / pid / "normal"
        image_paths = []
        if normal_dir.exists():
            image_paths = [str(f) for f in normal_dir.iterdir()
                           if f.suffix.lower() in ALLOWED_IMAGE_EXT]
        submit_job(
            run_foundad_training,
            job_id=job.id, project_id=pid,
            normal_paths=image_paths,
            n_shot=n_shot,
            encoder=encoder,
            steps=steps,
        )
    else:
        return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400

    return jsonify(_job_dict(job)), 202


# ── Job status ────────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    store = get_store()
    job = store.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(_job_dict(job))


@cv_bp.route("/cv/api/jobs", methods=["GET"])
def list_jobs():
    store = get_store()
    project_id = request.args.get("project_id")
    jobs = list(store._jobs.values())
    if project_id:
        jobs = [j for j in jobs if j.project_id == project_id]
    return jsonify([_job_dict(j) for j in sorted(jobs, key=lambda x: x.created_at, reverse=True)])


# ── Inference ─────────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/inference", methods=["POST"])
def run_inference():
    store = get_store()
    model_id = request.form.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400

    model = store.get_model(model_id)
    if not model:
        return jsonify({"error": "Model not found"}), 404

    if not model.checkpoint_path or not Path(model.checkpoint_path).exists():
        # Return simulated result for demo models
        if model_id in ("model1", "model2"):
            return _simulated_inference(model)
        return jsonify({"error": "Model checkpoint not found. Please retrain the model."}), 404

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "image file required"}), 400

    filename = _safe_filename(file.filename or "test.jpg")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": f"Unsupported image format: {ext}"}), 400

    tmp_path = OUTPUTS_DIR / f"test_{uuid.uuid4().hex[:8]}{ext}"
    file.save(str(tmp_path))

    try:
        if model.algorithm == "PatchCore":
            from .patchcore import PatchCoreDetector
            detector = PatchCoreDetector()
            detector.load(model.checkpoint_path)
            result = detector.predict(tmp_path, OUTPUTS_DIR / "heatmaps")

        elif model.algorithm == "FoundAD":
            from .foundad_adapter import FoundADAdapter
            adapter = FoundADAdapter()
            adapter.load(model.checkpoint_path)
            result = adapter.predict(tmp_path, OUTPUTS_DIR / "heatmaps")

        else:
            return jsonify({"error": f"Unknown algorithm: {model.algorithm}"}), 500

    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": f"Inference failed: {e}"}), 500

    # Build response with base64-encoded images
    resp = dict(result)
    resp["model_id"] = model_id
    resp["model_name"] = model.name
    resp["algorithm"] = model.algorithm

    if result.get("heatmap_path"):
        resp["heatmap_b64"] = _img_to_b64(result["heatmap_path"])
    if result.get("overlay_path"):
        resp["overlay_b64"] = _img_to_b64(result["overlay_path"])

    # Store result
    ir = InferenceResult(
        id=new_id(),
        model_id=model_id,
        project_id=model.project_id,
        image_path=str(tmp_path),
        anomaly_score=result["anomaly_score"],
        decision=result["decision"],
        threshold=result["threshold"],
        inference_ms=result["inference_ms"],
        heatmap_path=result.get("heatmap_path"),
        overlay_path=result.get("overlay_path"),
        created_at=time.time(),
        device=result.get("device", "CPU"),
    )
    store.add_result(ir)

    # Alert check
    if result["decision"] == "anomalous":
        nok_rate = store.monitoring_stats(model.project_id).get("nok_rate", 0)
        if nok_rate > 20:
            store.add_alert("warning", f"NOK rate {nok_rate:.1f}% exceeds threshold in project {model.project_id}",
                            project_id=model.project_id)

    tmp_path.unlink(missing_ok=True)
    return jsonify(resp)


def _simulated_inference(model) -> Response:
    """Return realistic simulated result for demo models."""
    import random
    rng = random.Random(time.time())
    score = round(rng.uniform(0.3, 0.9), 4)
    decision = "anomalous" if score > model.threshold else "normal"
    return jsonify({
        "model_id": model.id,
        "model_name": model.name,
        "algorithm": model.algorithm,
        "anomaly_score": score,
        "normalized_score": round(score, 4),
        "decision": decision,
        "threshold": round(model.threshold, 4),
        "inference_ms": round(rng.uniform(80, 250), 1),
        "device": model.device,
        "simulated": True,
        "note": "Demo model – upload real images and train to get actual predictions",
    })


# ── Models ────────────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/models", methods=["GET"])
def list_models():
    store = get_store()
    project_id = request.args.get("project_id")
    models = store.list_models(project_id)
    return jsonify([_model_dict(m) for m in models])


@cv_bp.route("/cv/api/models/<mid>", methods=["GET"])
def get_model(mid: str):
    store = get_store()
    m = store.get_model(mid)
    if not m:
        return jsonify({"error": "Model not found"}), 404
    return jsonify(_model_dict(m))


@cv_bp.route("/cv/api/models/<mid>/deploy", methods=["POST"])
def deploy_model(mid: str):
    store = get_store()
    m = store.get_model(mid)
    if not m:
        return jsonify({"error": "Model not found"}), 404
    m.status = "deployed"
    return jsonify({"status": "deployed", "model_id": mid})


# ── Monitoring ────────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/monitoring/stats", methods=["GET"])
def monitoring_stats():
    store = get_store()
    project_id = request.args.get("project_id")
    stats = store.monitoring_stats(project_id)
    return jsonify(stats)


@cv_bp.route("/cv/api/monitoring/results", methods=["GET"])
def monitoring_results():
    store = get_store()
    project_id = request.args.get("project_id")
    limit = min(int(request.args.get("limit", 20)), 100)
    results = store.get_results(project_id, limit)
    return jsonify([{
        "id": r.id,
        "anomaly_score": r.anomaly_score,
        "decision": r.decision,
        "threshold": r.threshold,
        "inference_ms": r.inference_ms,
        "device": r.device,
        "created_at": r.created_at,
        "model_id": r.model_id,
    } for r in results])


# ── Alerts ────────────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/alerts", methods=["GET"])
def get_alerts():
    store = get_store()
    return jsonify(store.get_alerts())


# ── Reports ───────────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/reports/export", methods=["GET"])
def export_report():
    store = get_store()
    project_id = request.args.get("project_id")
    results = store.get_results(project_id, limit=500)
    import csv, io as sio
    buf = sio.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "timestamp", "model_id", "anomaly_score", "decision",
                     "threshold", "inference_ms", "device"])
    for r in results:
        writer.writerow([r.id, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.created_at)),
                         r.model_id, r.anomaly_score, r.decision,
                         r.threshold, r.inference_ms, r.device])
    output = buf.getvalue().encode()
    return Response(output, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=cv_report.csv"})


# ── Demo endpoint ─────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/demo/start", methods=["POST"])
def start_demo():
    """Create a demo project and start training."""
    store = get_store()
    project = store.create_project(
        name="Demo – Synthetic Surface Inspection",
        description="Auto-generated demo project with synthetic normal/defect images",
    )
    job = store.create_job(
        project_id=project.id,
        model_name="Demo-PatchCore-v1",
        algorithm="PatchCore",
        backbone="resnet18",
    )
    submit_job(run_demo_training, job_id=job.id, project_id=project.id)
    return jsonify({
        "project_id": project.id,
        "job_id": job.id,
        "message": "Demo started. Poll /cv/api/jobs/<job_id> for status.",
    }), 202


# ── FoundAD info ──────────────────────────────────────────────────────────────
@cv_bp.route("/cv/api/foundad/info", methods=["GET"])
def foundad_info():
    from .foundad_adapter import FoundADAdapter, RUNPOD_ENABLED
    adapter = FoundADAdapter()
    return jsonify({
        "runpod_enabled": RUNPOD_ENABLED,
        "runpod_configured": adapter.runpod_available,
        "gpu_available": adapter.gpu_available,
        "device_recommendation": adapter.get_device_recommendation(),
        "supported_encoders": ["dinov2_vitb14"],
        "note": "DINOv2 ViT-B/14 from facebookresearch/dinov2 via torch.hub",
    })


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_filename(name: str) -> str:
    import re
    name = Path(name).name  # strip path components
    name = re.sub(r"[^\w\.\-]", "_", name)
    return name[:200] or "upload"


def _img_to_b64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def _project_dict(p) -> dict:
    return {
        "id": p.id, "name": p.name, "description": p.description,
        "created_at": p.created_at, "image_count": p.image_count,
        "normal_count": p.normal_count, "defect_count": p.defect_count,
        "status": p.status,
    }


def _job_dict(j) -> dict:
    return {
        "id": j.id, "project_id": j.project_id, "model_name": j.model_name,
        "algorithm": j.algorithm, "backbone": j.backbone, "status": j.status,
        "created_at": j.created_at, "started_at": j.started_at,
        "completed_at": j.completed_at, "progress": j.progress,
        "log": j.log[-20:], "error": j.error, "model_id": j.model_id,
        "metrics": j.metrics,
    }


def _model_dict(m) -> dict:
    return {
        "id": m.id, "project_id": m.project_id, "name": m.name,
        "algorithm": m.algorithm, "backbone": m.backbone, "version": m.version,
        "created_at": m.created_at, "status": m.status,
        "metrics": m.metrics, "threshold": m.threshold, "device": m.device,
    }
