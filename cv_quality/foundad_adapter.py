"""
FoundAD Adapter – Few-Shot Anomaly Detection with Foundation Visual Encoders.

Based on the local reimplementation in foundad_reimplementation_Ver3.ipynb.
This adapter wraps the notebook's core logic into a service-compatible interface.

GPU Note: DINOv2 ViT-B/14 at 518x518 resolution requires significant memory.
On CPU: feasible but very slow (~30-60 min for 1000 training steps).
Recommended: GPU via RunPod for full training.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger("cv_quality.foundad")

RUNPOD_ENABLED = os.environ.get("RUNPOD_ENABLED", "false").lower() == "true"
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")


class FoundADAdapter:
    """
    Wraps FoundAD few-shot anomaly detection.

    Training mode:
    - CPU: Light config (small image size, fewer steps) – slow but works
    - GPU (RunPod): Full config from notebook – recommended

    Prediction:
    - Loads trained checkpoint, runs inference
    - Returns same format as PatchCoreDetector
    """

    def __init__(self, n_shot: int = 4, encoder: str = "dinov2_vitb14",
                 device: str = "auto", steps: int = 200):
        self.n_shot = n_shot
        self.encoder = encoder
        self.steps = steps
        self.trained = False
        self.threshold = 0.5
        self.checkpoint_path: Optional[str] = None

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    @property
    def requires_gpu(self) -> bool:
        return self.encoder in ("dinov2_vitb14", "dinov3_vitb16") and self.steps >= 500

    @property
    def gpu_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def runpod_available(self) -> bool:
        return RUNPOD_ENABLED and bool(RUNPOD_API_KEY) and bool(RUNPOD_ENDPOINT_ID)

    def get_device_recommendation(self) -> dict:
        if self.gpu_available:
            return {"device": "GPU (local)", "warning": None, "can_run": True}
        if self.runpod_available:
            return {"device": "GPU (RunPod)", "warning": None, "can_run": True}
        if self.steps <= 200:
            return {
                "device": "CPU",
                "warning": f"Running {self.steps} steps on CPU. Estimated time: 15-30 min.",
                "can_run": True,
            }
        return {
            "device": None,
            "warning": (
                f"FoundAD with {self.steps} steps and encoder '{self.encoder}' "
                "requires GPU acceleration. Configure RunPod (RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID) "
                "to enable GPU training."
            ),
            "can_run": False,
        }

    def train(
        self,
        normal_paths: List[Path],
        defect_paths: Optional[List[Path]] = None,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> dict:
        """
        Train FoundAD using the notebook's core algorithm.
        For CPU: uses reduced config (small image, fewer steps).
        For GPU: full paper config.
        """
        rec = self.get_device_recommendation()
        if not rec["can_run"]:
            raise RuntimeError(rec["warning"])

        if self.runpod_available and not self.gpu_available:
            return self._submit_runpod_job(normal_paths, defect_paths, output_dir, progress_callback)

        return self._train_local(normal_paths, defect_paths, output_dir, progress_callback)

    def _train_local(self, normal_paths, defect_paths, output_dir, progress_callback):
        """Run training locally (CPU or local GPU) using notebook logic."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("torch is required for FoundAD training")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        if progress_callback:
            progress_callback(5, "Initializing FoundAD encoder...")

        # CPU-safe config: smaller image, fewer steps
        cpu_mode = self.device == "cpu"
        image_size = 224 if cpu_mode else 518
        actual_steps = min(self.steps, 100) if cpu_mode else self.steps
        n_shot = min(self.n_shot, 4)

        if progress_callback:
            progress_callback(10, f"Loading {self.encoder} encoder ({self.device.upper()})...")

        try:
            encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            encoder.eval()
            encoder.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2 encoder: {e}. Check internet connection.")

        if progress_callback:
            progress_callback(25, f"Extracting features from {n_shot} reference images...")

        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        from PIL import Image
        import numpy as np

        # Few-shot: select n_shot normal images as reference
        ref_paths = normal_paths[:n_shot]
        ref_features = []
        with torch.no_grad():
            for p in ref_paths:
                img = Image.open(p).convert("RGB")
                t = transform(img).unsqueeze(0).to(self.device)
                feat = encoder.forward_features(t)
                if "x_norm_patchtokens" in feat:
                    f = feat["x_norm_patchtokens"]
                else:
                    f = feat.get("x_norm_clstoken", feat.get("last_hidden_state", None))
                    if f is None:
                        raise RuntimeError("Unexpected encoder output format")
                ref_features.append(f.cpu())

        if progress_callback:
            progress_callback(40, "Building few-shot reference representation...")

        ref_stack = torch.cat(ref_features, dim=1)  # (1, n_shot*patches, D)

        # Simple manifold projection: linear layer trained to distinguish
        # reference patch distribution
        D = ref_stack.shape[-1]
        projector = torch.nn.Sequential(
            torch.nn.Linear(D, D // 2),
            torch.nn.GELU(),
            torch.nn.Linear(D // 2, D),
        ).to(self.device)
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3, weight_decay=1e-4)

        if progress_callback:
            progress_callback(45, f"Training manifold projector ({actual_steps} steps)...")

        ref_flat = ref_stack.squeeze(0).to(self.device)
        projector.train()
        for step in range(actual_steps):
            idx = torch.randperm(ref_flat.shape[0])[:32]
            batch = ref_flat[idx]
            proj = projector(batch)
            loss = torch.nn.functional.mse_loss(proj, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 20 == 0 and progress_callback:
                pct = 45 + int(step / actual_steps * 40)
                progress_callback(pct, f"Training step {step}/{actual_steps}, loss={loss.item():.4f}")

        projector.eval()

        if progress_callback:
            progress_callback(87, "Computing threshold from reference scores...")

        # Compute reference reconstruction errors for threshold
        with torch.no_grad():
            proj = projector(ref_flat)
            errors = torch.mean((proj - ref_flat) ** 2, dim=-1).cpu().numpy()
        self.threshold = float(np.percentile(errors, 95))

        # Save checkpoint
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            ckpt = output_dir / "foundad_checkpoint.pt"
            torch.save({
                "encoder_name": self.encoder,
                "projector_state": projector.state_dict(),
                "ref_features": ref_stack,
                "threshold": self.threshold,
                "n_shot": n_shot,
                "D": D,
                "image_size": image_size,
                "device": self.device,
            }, ckpt)
            self.checkpoint_path = str(ckpt)

        self.trained = True
        if progress_callback:
            progress_callback(100, "FoundAD training complete")

        return {
            "encoder": self.encoder,
            "n_shot": n_shot,
            "steps": actual_steps,
            "threshold": round(self.threshold, 6),
            "device": self.device.upper(),
            "checkpoint": self.checkpoint_path,
        }

    def _submit_runpod_job(self, normal_paths, defect_paths, output_dir, progress_callback):
        """
        Submit FoundAD training job to the shared RunPod endpoint (same as Aurora).
        Images are sent as base64 – no shared filesystem needed.
        """
        import base64, requests
        if progress_callback:
            progress_callback(5, "Encoding images for GPU job...")

        images_b64 = []
        for p in list(normal_paths)[:self.n_shot]:
            with open(p, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode())

        if progress_callback:
            progress_callback(10, f"Submitting FoundAD GPU job ({len(images_b64)} reference images)...")

        payload = {
            "input": {
                "job_type": "cv_foundad_train",
                "normal_images_b64": images_b64,
                "n_shot": self.n_shot,
                "encoder": self.encoder,
                "steps": self.steps,
                "image_size": 518,
            }
        }
        headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
        url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        job_data = resp.json()
        runpod_id = job_data.get("id", "")

        if progress_callback:
            progress_callback(15, f"RunPod job submitted: {runpod_id}")

        # Store model_state_b64 in output_dir when job completes (polling done by job_runner)
        return {
            "runpod_job_id": runpod_id,
            "status": "submitted",
            "device": "GPU (RunPod)",
            "output_dir": str(output_dir) if output_dir else None,
            "message": f"FoundAD GPU training submitted to RunPod (job {runpod_id}). Polling for completion…",
        }

    def predict(self, img_path: Path, output_dir: Path) -> dict:
        """Run FoundAD inference on a single image."""
        if not self.trained and not self.checkpoint_path:
            raise RuntimeError("FoundAD model not trained")
        try:
            import torch
        except ImportError:
            raise RuntimeError("torch required for FoundAD inference")

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        D = ckpt["D"]
        image_size = ckpt.get("image_size", 224)
        projector = torch.nn.Sequential(
            torch.nn.Linear(D, D // 2),
            torch.nn.GELU(),
            torch.nn.Linear(D // 2, D),
        )
        projector.load_state_dict(ckpt["projector_state"])
        projector.eval()

        encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        encoder.eval()

        import torchvision.transforms as T
        from PIL import Image
        import numpy as np

        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        t0 = time.time()
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = encoder.forward_features(tensor)
            if "x_norm_patchtokens" in feat:
                patches = feat["x_norm_patchtokens"].squeeze(0)
            else:
                patches = list(feat.values())[0].squeeze(0)
            proj = projector(patches)
            errors = torch.mean((proj - patches) ** 2, dim=-1).cpu().numpy()
        elapsed_ms = (time.time() - t0) * 1000

        # Patch-level scores → heatmap
        score = float(errors.max())
        threshold = ckpt.get("threshold", self.threshold)
        decision = "anomalous" if score > threshold else "normal"

        # Simple heatmap from errors
        n_patches = len(errors)
        side = int(math.sqrt(n_patches))
        patch_map = errors[:side*side].reshape(side, side)
        from cv_quality.patchcore import _make_heatmap, _overlay_heatmap
        hmap = _make_heatmap(patch_map.flatten(), (side, side), image_size)
        output_dir.mkdir(parents=True, exist_ok=True)
        import uuid
        stem = str(uuid.uuid4())[:8]
        heatmap_p = output_dir / f"foundad_hmap_{stem}.png"
        overlay_p = output_dir / f"foundad_overlay_{stem}.png"
        Image.fromarray(hmap[:, :, :3]).save(heatmap_p)
        overlay_arr = _overlay_heatmap(img_path, hmap, image_size)
        Image.fromarray(overlay_arr).save(overlay_p)

        return {
            "anomaly_score": round(float(score), 6),
            "normalized_score": round(min(float(score) / (threshold + 1e-8), 1.0), 4),
            "decision": decision,
            "threshold": round(float(threshold), 6),
            "inference_ms": round(elapsed_ms, 1),
            "heatmap_path": str(heatmap_p),
            "overlay_path": str(overlay_p),
            "device": self.device.upper(),
            "model": "FoundAD",
            "encoder": self.encoder,
            "n_shot": ckpt.get("n_shot", self.n_shot),
        }

    def load(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.trained = True
        try:
            import torch
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.threshold = ckpt.get("threshold", 0.5)
            self.encoder = ckpt.get("encoder_name", self.encoder)
            self.n_shot = ckpt.get("n_shot", self.n_shot)
        except Exception as e:
            logger.warning(f"Could not read FoundAD checkpoint metadata: {e}")
