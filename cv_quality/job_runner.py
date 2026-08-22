"""
Background job runner for training jobs.
Uses threading (no Celery/Redis dependency for MVP).
"""
from __future__ import annotations

import threading
import time
import logging
import traceback
from pathlib import Path
from typing import List

from .store import CVStore, get_store, new_id
from .patchcore import PatchCoreDetector, generate_demo_images

logger = logging.getLogger("cv_quality.jobs")

MODELS_DIR = Path("/tmp/cv_quality/models")
IMAGES_DIR = Path("/tmp/cv_quality/images")
OUTPUTS_DIR = Path("/tmp/cv_quality/outputs")


def run_patchcore_training(job_id: str, project_id: str, image_paths: List[str],
                           backbone: str = "resnet18", coreset_ratio: float = 0.1,
                           k_neighbors: int = 9, input_size: int = 224):
    """Execute PatchCore training in background thread."""
    store = get_store()

    def progress(pct: int, msg: str):
        store.update_job(job_id, progress=pct)
        store.append_log(job_id, msg)

    try:
        store.update_job(job_id, status="preparing", started_at=time.time())
        progress(5, f"Starting PatchCore training with {len(image_paths)} images")
        progress(10, f"Backbone: {backbone} | Coreset: {coreset_ratio*100:.0f}% | k={k_neighbors}")

        detector = PatchCoreDetector(
            backbone=backbone,
            input_size=input_size,
            coreset_ratio=coreset_ratio,
            k_neighbors=k_neighbors,
        )
        store.update_job(job_id, status="running")
        train_result = detector.train(image_paths, progress_callback=progress)

        progress(96, "Saving model checkpoint...")
        ckpt_dir = MODELS_DIR / f"{project_id}_{job_id}"
        ckpt_path = ckpt_dir / "patchcore.pkl"
        detector.save(ckpt_path)
        progress(98, f"Model saved to {ckpt_path}")

        # Register model in store
        model = store.create_model(
            project_id=project_id,
            name=f"PatchCore-{backbone}-v{int(time.time()) % 10000}",
            algorithm="PatchCore",
            backbone=backbone,
            checkpoint_path=str(ckpt_path),
            metrics={"train_images": len(image_paths), **train_result},
            threshold=train_result.get("threshold", 0.5),
            device=detector.device.upper(),
        )

        store.update_job(
            job_id,
            status="completed",
            completed_at=time.time(),
            progress=100,
            model_id=model.id,
            metrics=train_result,
        )
        progress(100, f"Training complete. Model ID: {model.id}")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Training job {job_id} failed: {e}\n{tb}")
        store.update_job(job_id, status="failed", error=str(e), completed_at=time.time())
        store.append_log(job_id, f"ERROR: {e}")


def run_demo_training(job_id: str, project_id: str):
    """Run demo training with synthetic images."""
    store = get_store()

    def progress(pct: int, msg: str):
        store.update_job(job_id, progress=pct)
        store.append_log(job_id, msg)

    try:
        store.update_job(job_id, status="preparing", started_at=time.time())
        progress(5, "Generating synthetic demo dataset (MVTec-like)...")

        demo_dir = IMAGES_DIR / f"demo_{project_id}"
        demo_images = generate_demo_images(demo_dir, n_normal=15, n_defect=5)
        normal_paths = demo_images["normal"]
        progress(15, f"Generated {len(normal_paths)} normal images")

        store.update_project_counts(project_id, len(normal_paths), len(demo_images["defect"]))
        run_patchcore_training(
            job_id=job_id,
            project_id=project_id,
            image_paths=normal_paths,
            backbone="resnet18",
            coreset_ratio=0.1,
            k_neighbors=9,
            input_size=224,
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Demo job {job_id} failed: {e}\n{tb}")
        store.update_job(job_id, status="failed", error=str(e), completed_at=time.time())


# run_foundad_training is defined later in this file (after _poll_runpod_and_save)


def _poll_runpod_and_save(job_id: str, project_id: str, runpod_job_id: str,
                          algorithm: str, output_dir: Path,
                          encoder: str = "dinov2_vitb14", n_shot: int = 4):
    """
    Poll the shared RunPod endpoint (same as Aurora) until the CV training job
    completes, then save the returned model state and register it in the store.
    """
    import requests, base64, pickle, os, time as t_mod
    store = get_store()
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    if not api_key or not endpoint_id:
        store.update_job(job_id, status="failed", error="RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not set")
        return

    headers = {"Authorization": f"Bearer {api_key}"}
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{runpod_job_id}"
    timeout = int(os.environ.get("RUNPOD_TIMEOUT", "1800"))
    t_start = t_mod.time()

    store.append_log(job_id, f"Polling RunPod job {runpod_job_id}…")
    store.update_job(job_id, status="running")

    while t_mod.time() - t_start < timeout:
        try:
            resp = requests.get(status_url, headers=headers, timeout=15).json()
            rp_status = resp.get("status", "")
            store.append_log(job_id, f"RunPod status: {rp_status}")

            if rp_status == "COMPLETED":
                output = resp.get("output", {})
                if output.get("error"):
                    store.update_job(job_id, status="failed", error=output["error"])
                    return

                # Save model state
                output_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = output_dir / f"{algorithm}_checkpoint.pkl"
                state_b64 = output.get("model_state_b64", "")
                if state_b64:
                    with open(ckpt_path, "wb") as f:
                        f.write(base64.b64decode(state_b64))
                    store.append_log(job_id, f"Model state saved to {ckpt_path}")
                else:
                    store.update_job(job_id, status="failed", error="No model_state_b64 in RunPod response")
                    return

                # Register model
                metrics = output.get("metrics", {})
                metrics["device"] = "GPU (RunPod)"
                model = store.create_model(
                    project_id=project_id,
                    name=f"{algorithm.upper()}-GPU-{encoder[:8]}-{n_shot}shot" if algorithm == "foundad"
                         else f"PatchCore-GPU",
                    algorithm=algorithm.capitalize() if algorithm == "foundad" else "PatchCore",
                    backbone=encoder,
                    checkpoint_path=str(ckpt_path),
                    metrics=metrics,
                    threshold=float(output.get("threshold", 0.5)),
                    device="GPU (RunPod)",
                )
                store.update_job(job_id, status="completed", completed_at=t_mod.time(),
                                 progress=100, model_id=model.id, metrics=metrics)
                store.append_log(job_id, f"GPU training complete. Model: {model.id}")
                return

            elif rp_status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                err = resp.get("output", {}).get("error", rp_status)
                store.update_job(job_id, status="failed", error=f"RunPod: {err}")
                return

        except Exception as e:
            store.append_log(job_id, f"Polling error: {e}")

        t_mod.sleep(8)

    store.update_job(job_id, status="failed", error="RunPod job timed out")


def run_foundad_training(job_id: str, project_id: str, normal_paths: List[str],
                         n_shot: int = 4, encoder: str = "dinov2_vitb14", steps: int = 200):
    """Execute FoundAD training – locally (CPU) or via RunPod GPU."""
    from .foundad_adapter import FoundADAdapter
    store = get_store()

    def progress(pct: int, msg: str):
        store.update_job(job_id, progress=pct)
        store.append_log(job_id, msg)

    try:
        store.update_job(job_id, status="preparing", started_at=time.time())
        adapter = FoundADAdapter(n_shot=n_shot, encoder=encoder, steps=steps)
        rec = adapter.get_device_recommendation()

        if not rec["can_run"]:
            store.update_job(job_id, status="failed", error=rec["warning"], completed_at=time.time())
            store.append_log(job_id, f"GPU required: {rec['warning']}")
            return

        progress(5, f"FoundAD: {encoder} | {n_shot}-shot | device: {rec['device']}")
        store.update_job(job_id, status="running")

        ckpt_dir = MODELS_DIR / f"{project_id}_{job_id}"
        result = adapter.train(
            normal_paths=[Path(p) for p in normal_paths[:n_shot]],
            output_dir=ckpt_dir,
            progress_callback=progress,
        )

        # RunPod async path: poll in background
        if result.get("runpod_job_id"):
            rp_id = result["runpod_job_id"]
            store.update_job(job_id, status="running",
                             metrics={"runpod_job_id": rp_id, "message": result["message"]})
            progress(20, f"RunPod GPU job submitted: {rp_id}. Polling for completion…")
            _poll_runpod_and_save(
                job_id=job_id,
                project_id=project_id,
                runpod_job_id=rp_id,
                algorithm="foundad",
                output_dir=ckpt_dir,
                encoder=encoder,
                n_shot=n_shot,
            )
            return

        # Local training completed
        model = store.create_model(
            project_id=project_id,
            name=f"FoundAD-{encoder[:8]}-{n_shot}shot",
            algorithm="FoundAD",
            backbone=encoder,
            checkpoint_path=result.get("checkpoint", ""),
            metrics=result,
            threshold=result.get("threshold", 0.5),
            device=result.get("device", "CPU"),
        )
        store.update_job(job_id, status="completed", completed_at=time.time(),
                         progress=100, model_id=model.id, metrics=result)
        progress(100, f"FoundAD training complete. Model: {model.id}")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"FoundAD job {job_id} failed: {e}\n{tb}")
        store.update_job(job_id, status="failed", error=str(e), completed_at=time.time())
        store.append_log(job_id, f"ERROR: {e}")


def submit_job(target_fn, **kwargs) -> str:
    """Submit a job to run in a background thread."""
    t = threading.Thread(target=target_fn, kwargs=kwargs, daemon=True)
    t.start()
    return kwargs.get("job_id", "unknown")
