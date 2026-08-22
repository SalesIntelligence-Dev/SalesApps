"""
PatchCore Anomaly Detection – CPU-first implementation.
Based on: "Towards Total Recall in Industrial Anomaly Detection" (Roth et al., 2022)

Uses pretrained ResNet feature extraction + KNN-based anomaly scoring.
Falls back to HOG features if torch is unavailable.
"""
from __future__ import annotations

import math
import pickle
import time
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("cv_quality.patchcore")

# ── torch availability ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available – using HOG fallback for PatchCore")

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── Image preprocessing ───────────────────────────────────────────────────────
_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD  = (0.229, 0.224, 0.225)


def _load_image(path: str | Path, size: int = 224) -> "np.ndarray":
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def _normalize(arr: np.ndarray) -> np.ndarray:
    mean = np.array(_IMG_MEAN, dtype=np.float32)
    std  = np.array(_IMG_STD,  dtype=np.float32)
    return (arr - mean) / std


# ── HOG fallback (no torch) ───────────────────────────────────────────────────
def _hog_features(img_arr: np.ndarray, patch_size: int = 32) -> np.ndarray:
    """Simple HOG-like gradient features, 8x8 patch grid."""
    try:
        from skimage.feature import hog
        features, _ = hog(img_arr, orientations=8, pixels_per_cell=(patch_size, patch_size),
                           cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        return features.flatten()
    except ImportError:
        # Ultra-fallback: raw pixel patches
        h, w, _ = img_arr.shape
        grid = h // patch_size
        patches = []
        for i in range(grid):
            for j in range(grid):
                patch = img_arr[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].flatten()
                patches.append(patch)
        return np.array(patches).flatten()


# ── Torch feature extractor ───────────────────────────────────────────────────
class _ResNetExtractor:
    """Extracts patch-level features from ResNet18 layer2 + layer3."""

    def __init__(self, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not available")
        self.device = device
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        model.eval()
        # layer2: 28x28x128, layer3: 14x14x256
        self.layer2 = torch.nn.Sequential(*list(model.children())[:6])
        self.layer3 = torch.nn.Sequential(*list(model.children())[:7])
        self.layer2.to(device)
        self.layer3.to(device)
        for p in list(self.layer2.parameters()) + list(self.layer3.parameters()):
            p.requires_grad_(False)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        with torch.no_grad():
            f2 = self.layer2(x)
            f3 = self.layer3(x)
            f3_up = torch.nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([f2, f3_up], dim=1)

    def extract_patches(self, img_path: str | Path, size: int = 224) -> np.ndarray:
        """Returns (N_patches, C) array of patch features for one image."""
        arr = _load_image(img_path, size)
        arr = _normalize(arr)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        features = self.forward(tensor)  # (1, C, H, W)
        _B, C, H, W = features.shape
        patches = features[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
        return patches, (H, W)


# ── Greedy coreset subsampling ────────────────────────────────────────────────
def _coreset_subsample(features: np.ndarray, ratio: float) -> np.ndarray:
    """Greedy farthest-first traversal coreset. O(n*k) time."""
    if ratio >= 1.0:
        return features
    n = len(features)
    target = max(1, int(n * ratio))
    if target >= n:
        return features
    selected = [0]
    min_dists = np.full(n, np.inf)
    for _ in range(target - 1):
        last = features[selected[-1]]
        dists = np.sum((features - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected.append(int(np.argmax(min_dists)))
    return features[selected]


# ── Anomaly map generation ────────────────────────────────────────────────────
def _make_heatmap(patch_scores: np.ndarray, spatial: Tuple[int, int],
                  orig_size: int = 224) -> np.ndarray:
    """Upsample patch scores to image size, apply Gaussian blur, return RGBA uint8."""
    import scipy.ndimage as ndi
    H, W = spatial
    score_map = patch_scores.reshape(H, W)
    # Upsample
    zoom_h = orig_size / H
    zoom_w = orig_size / W
    score_map = ndi.zoom(score_map, (zoom_h, zoom_w), order=1)
    # Blur
    score_map = ndi.gaussian_filter(score_map, sigma=4)
    # Normalize
    mn, mx = score_map.min(), score_map.max()
    if mx > mn:
        score_map = (score_map - mn) / (mx - mn)
    # Apply jet colormap manually
    cmap = _jet_colormap(score_map)
    return cmap


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    """Jet colormap: blue→cyan→green→yellow→red, returns RGBA uint8."""
    H, W = x.shape
    r = np.clip(1.5 - np.abs(x * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(x * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(x * 4 - 1), 0, 1)
    a = np.ones_like(x) * 0.65
    return (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)


def _overlay_heatmap(orig_path: str | Path, heatmap_rgba: np.ndarray, orig_size: int = 224) -> np.ndarray:
    """Blend heatmap onto original image."""
    orig = np.array(Image.open(orig_path).convert("RGB").resize((orig_size, orig_size))).astype(np.float32)
    alpha = heatmap_rgba[:, :, 3:4] / 255.0
    heat_rgb = heatmap_rgba[:, :, :3].astype(np.float32)
    blended = orig * (1 - alpha) + heat_rgb * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── PatchCore Detector ────────────────────────────────────────────────────────
class PatchCoreDetector:
    """
    Industrial anomaly detector using PatchCore algorithm.
    - Trains on normal images only (one-class learning)
    - CPU-first: uses ResNet18 if torch available, else HOG fallback
    - Greedy coreset subsampling for memory efficiency
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        input_size: int = 224,
        coreset_ratio: float = 0.1,
        k_neighbors: int = 9,
        device: str = "auto",
    ):
        self.backbone = backbone
        self.input_size = input_size
        self.coreset_ratio = coreset_ratio
        self.k = k_neighbors
        self.memory_bank: Optional[np.ndarray] = None
        self.spatial_shape: Optional[Tuple[int, int]] = None
        self._knn: Optional[NearestNeighbors] = None
        self.threshold: float = 0.5
        self.trained: bool = False
        self.train_scores: List[float] = []

        if device == "auto":
            self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        self._extractor = None
        self._use_torch = TORCH_AVAILABLE

    def _get_extractor(self) -> _ResNetExtractor:
        if self._extractor is None and self._use_torch:
            self._extractor = _ResNetExtractor(device=self.device)
        return self._extractor

    def _extract(self, img_path: str | Path) -> Tuple[np.ndarray, Tuple[int, int]]:
        if self._use_torch:
            ext = self._get_extractor()
            return ext.extract_patches(img_path, self.input_size)
        else:
            arr = _load_image(img_path, self.input_size)
            feat = _hog_features(arr, patch_size=32)
            # Fake spatial shape for HOG
            grid = self.input_size // 32
            feat = feat.reshape(grid * grid, -1) if feat.ndim == 1 else feat
            return feat, (grid, grid)

    def train(
        self,
        image_paths: List[str | Path],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> dict:
        """Build memory bank from normal images."""
        if not image_paths:
            raise ValueError("No training images provided")

        t0 = time.time()
        all_patches = []
        spatial = None

        for i, path in enumerate(image_paths):
            try:
                patches, spatial = self._extract(path)
                all_patches.append(patches)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
            pct = int((i + 1) / len(image_paths) * 60)
            if progress_callback:
                progress_callback(pct, f"Extracting features: {i+1}/{len(image_paths)}")

        if not all_patches:
            raise RuntimeError("All images failed to process")

        if progress_callback:
            progress_callback(65, "Building memory bank...")

        memory = np.vstack(all_patches).astype(np.float32)
        self.spatial_shape = spatial

        if progress_callback:
            progress_callback(70, f"Coreset subsampling ({int(self.coreset_ratio*100)}%)...")
        self.memory_bank = _coreset_subsample(memory, self.coreset_ratio)

        if progress_callback:
            progress_callback(80, "Fitting nearest-neighbor index...")
        self._knn = NearestNeighbors(n_neighbors=self.k, algorithm="ball_tree", n_jobs=-1)
        self._knn.fit(self.memory_bank)

        if progress_callback:
            progress_callback(85, "Computing validation scores for threshold...")
        # Compute per-image scores on training set for threshold selection
        self.train_scores = []
        for path in image_paths[:min(len(image_paths), 20)]:
            try:
                score, _ = self._score_image(path)
                self.train_scores.append(score)
            except Exception:
                pass

        # Threshold: 95th percentile of training scores (generous for normal images)
        if self.train_scores:
            self.threshold = float(np.percentile(self.train_scores, 95))
        else:
            self.threshold = 0.5

        self.trained = True
        elapsed = time.time() - t0

        if progress_callback:
            progress_callback(95, "Training complete")

        return {
            "memory_bank_size": len(self.memory_bank),
            "original_patches": len(memory),
            "coreset_ratio": self.coreset_ratio,
            "threshold": round(self.threshold, 4),
            "train_images": len(image_paths),
            "elapsed_s": round(elapsed, 1),
            "device": self.device,
        }

    def _score_image(self, img_path: str | Path) -> Tuple[float, np.ndarray]:
        """Compute anomaly score and per-patch scores for one image."""
        patches, spatial = self._extract(img_path)
        dists, _ = self._knn.kneighbors(patches.astype(np.float32))
        patch_scores = dists[:, 0]  # min distance to nearest neighbor
        image_score = float(patch_scores.max())
        return image_score, patch_scores

    def predict(self, img_path: str | Path, output_dir: Path) -> dict:
        """Run inference on a single image. Returns prediction dict."""
        if not self.trained:
            raise RuntimeError("Model not trained")
        t0 = time.time()
        score, patch_scores = self._score_image(img_path)
        elapsed_ms = (time.time() - t0) * 1000

        # Normalize score relative to training max
        if self.train_scores:
            train_max = max(self.train_scores) + 1e-8
            norm_score = min(score / train_max, 1.0)
        else:
            norm_score = min(score, 1.0)

        decision = "anomalous" if norm_score > self.threshold / (max(self.train_scores) + 1e-8) else "normal"
        # Actually use raw threshold comparison
        decision = "anomalous" if score > self.threshold else "normal"

        # Generate heatmap
        heatmap_path = None
        overlay_path = None
        try:
            hmap = _make_heatmap(patch_scores, self.spatial_shape or (7, 7), self.input_size)
            output_dir.mkdir(parents=True, exist_ok=True)
            import uuid
            stem = str(uuid.uuid4())[:8]
            heatmap_p = output_dir / f"hmap_{stem}.png"
            overlay_p = output_dir / f"overlay_{stem}.png"
            Image.fromarray(hmap[:, :, :3]).save(heatmap_p)
            overlay_arr = _overlay_heatmap(img_path, hmap, self.input_size)
            Image.fromarray(overlay_arr).save(overlay_p)
            heatmap_path = str(heatmap_p)
            overlay_path = str(overlay_p)
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")

        return {
            "anomaly_score": round(float(score), 4),
            "normalized_score": round(float(norm_score), 4),
            "decision": decision,
            "threshold": round(self.threshold, 4),
            "inference_ms": round(elapsed_ms, 1),
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path,
            "device": self.device.upper(),
            "model": "PatchCore",
            "backbone": self.backbone,
        }

    def evaluate(self, normal_paths: List[Path], defect_paths: List[Path]) -> dict:
        """Evaluate on labeled test set. Returns AUROC, AP, F1, etc."""
        if not self.trained:
            raise RuntimeError("Model not trained")
        scores = []
        labels = []
        for p in normal_paths:
            try:
                s, _ = self._score_image(p)
                scores.append(s)
                labels.append(0)
            except Exception:
                pass
        for p in defect_paths:
            try:
                s, _ = self._score_image(p)
                scores.append(s)
                labels.append(1)
            except Exception:
                pass
        if not scores:
            return {}
        scores = np.array(scores)
        labels = np.array(labels)
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
        preds = (scores > self.threshold).astype(int)
        metrics = {
            "image_auroc": round(float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else 0.0, 4),
            "ap": round(float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else 0.0, 4),
            "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
            "threshold": round(float(self.threshold), 4),
            "n_normal": int((labels == 0).sum()),
            "n_defect": int((labels == 1).sum()),
        }
        return metrics

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "backbone": self.backbone,
            "input_size": self.input_size,
            "coreset_ratio": self.coreset_ratio,
            "k": self.k,
            "memory_bank": self.memory_bank,
            "spatial_shape": self.spatial_shape,
            "threshold": self.threshold,
            "train_scores": self.train_scores,
            "device": self.device,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str | Path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.backbone = state["backbone"]
        self.input_size = state["input_size"]
        self.coreset_ratio = state["coreset_ratio"]
        self.k = state["k"]
        self.memory_bank = state["memory_bank"]
        self.spatial_shape = state["spatial_shape"]
        self.threshold = state["threshold"]
        self.train_scores = state.get("train_scores", [])
        self.device = state.get("device", "cpu")
        self._knn = NearestNeighbors(n_neighbors=self.k, algorithm="ball_tree", n_jobs=-1)
        self._knn.fit(self.memory_bank)
        self.trained = True


# ── Demo data generator ───────────────────────────────────────────────────────
def generate_demo_images(output_dir: Path, n_normal: int = 20, n_defect: int = 5, seed: int = 42) -> dict:
    """
    Generates synthetic MVTec-like images for demo purposes.
    Normal: uniform texture. Defect: texture with scratch/spot anomaly.
    """
    rng = np.random.RandomState(seed)
    normal_dir = output_dir / "normal"
    defect_dir = output_dir / "defect"
    normal_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)
    size = 256

    def _base_texture(seed_val):
        rng2 = np.random.RandomState(seed_val)
        base = rng2.randint(100, 180, size=(size, size, 3), dtype=np.uint8)
        # Add subtle grain
        grain = rng2.randint(-15, 15, size=(size, size, 3)).astype(np.int16)
        return np.clip(base.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    normal_paths = []
    for i in range(n_normal):
        arr = _base_texture(seed + i)
        p = normal_dir / f"normal_{i:03d}.png"
        Image.fromarray(arr).save(p)
        normal_paths.append(str(p))

    defect_paths = []
    for i in range(n_defect):
        arr = _base_texture(seed + 1000 + i)
        # Add scratch anomaly
        img = Image.fromarray(arr)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        x0, y0 = rng.randint(30, 100), rng.randint(30, 100)
        x1, y1 = x0 + rng.randint(60, 150), y0 + rng.randint(5, 30)
        draw.line([(x0, y0), (x1, y1)], fill=(220, 50, 50), width=3)
        if rng.rand() > 0.5:
            cx, cy = rng.randint(100, 180), rng.randint(100, 180)
            r = rng.randint(8, 20)
            draw.ellipse([(cx-r, cy-r), (cx+r, cy+r)], fill=(240, 80, 40))
        p = defect_dir / f"defect_{i:03d}.png"
        img.save(p)
        defect_paths.append(str(p))

    return {"normal": normal_paths, "defect": defect_paths}
