"""
RunPod Serverless Handler – Aurora Inference + CV Quality (Anomaly Detection)
=============================================================================
Dispatches by job["input"]["job_type"]:
  - "aurora_forecast"     (default, original behavior)
  - "cv_patchcore_train"  (PatchCore feature extraction on GPU)
  - "cv_foundad_train"    (FoundAD DINOv2 training on GPU)
  - "cv_inference"        (PatchCore or FoundAD inference, returns heatmap)
"""

import base64
import io
import logging
import os
import pickle
import re
import sys
import traceback
from pathlib import Path

import runpod

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("handler")

CHECKPOINT_PATH = Path(os.getenv("AURORA_MODEL_PATH", "/app/aurora_checkpoint.pt"))
EPS = 1e-8

# ── Aurora model (loaded once at worker start) ────────────────────────────────
MODEL_ERROR = None
aurora_model = None
device = "cpu"

try:
    import torch
    import numpy as np
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device.upper()}")

    from aurora import load_model

    def _remap_keys(state_dict: dict) -> dict:
        new_sd = {}
        for k, v in state_dict.items():
            k = re.sub(r'\.encoder\.layer\.(\d+)\.', r'.layers.\1.', k)
            k = k.replace('.attention.attention.query.', '.attention.q_proj.')
            k = k.replace('.attention.attention.key.',   '.attention.k_proj.')
            k = k.replace('.attention.attention.value.', '.attention.v_proj.')
            k = k.replace('.attention.output.dense.',    '.attention.o_proj.')
            k = k.replace('.intermediate.dense.',        '.mlp.fc1.')
            k = k.replace('.output.dense.',              '.mlp.fc2.')
            new_sd[k] = v
        return new_sd

    aurora_model = load_model()
    if CHECKPOINT_PATH.exists():
        size_mb = CHECKPOINT_PATH.stat().st_size / 1024**2
        log.info(f"Loading Aurora checkpoint ({size_mb:.0f} MB)…")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        sd = _remap_keys(ckpt["model_state_dict"])
        result = aurora_model.load_state_dict(sd, strict=False)
        log.info(f"Aurora loaded | missing keys: {len(result.missing_keys)}")
    else:
        log.warning(f"No checkpoint at {CHECKPOINT_PATH} – loading from HuggingFace…")
    aurora_model = aurora_model.to(device)
    aurora_model.eval()
    log.info(f"Aurora ready | {sum(p.numel() for p in aurora_model.parameters()):,} params")
except Exception as e:
    MODEL_ERROR = traceback.format_exc()
    log.error(f"Aurora load error:\n{MODEL_ERROR}")


# ══════════════════════════════════════════════════════════════════════════════
# CV Quality helpers (inline – no dependency on cv_quality module)
# ══════════════════════════════════════════════════════════════════════════════

_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD  = (0.229, 0.224, 0.225)


def _b64_to_image(b64str: str):
    """Decode base64 string to PIL Image."""
    from PIL import Image
    data = base64.b64decode(b64str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _image_to_b64(img, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _preprocess(img, size: int = 224):
    """Resize + normalize to (1, 3, H, W) tensor."""
    import torch
    import numpy as np
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
    ])
    return transform(img).unsqueeze(0)


# ── PatchCore feature extractor (GPU) ────────────────────────────────────────
_patchcore_extractor = None


def _get_patchcore_extractor(dev: str):
    global _patchcore_extractor
    if _patchcore_extractor is None:
        import torch
        import torchvision.models as tv
        import torch.nn as nn
        model = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT)
        model.eval()

        class _Extractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer2 = nn.Sequential(*list(model.children())[:6])
                self.layer3 = nn.Sequential(*list(model.children())[:7])

            @torch.no_grad()
            def forward(self, x):
                f2 = self.layer2(x)
                f3 = self.layer3(x)
                f3u = torch.nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
                return torch.cat([f2, f3u], dim=1)

        _patchcore_extractor = _Extractor().to(dev)
        for p in _patchcore_extractor.parameters():
            p.requires_grad_(False)
        log.info("PatchCore ResNet-18 extractor loaded")
    return _patchcore_extractor


def _extract_patches(img, extractor, dev: str, size: int = 224):
    import torch
    tensor = _preprocess(img, size).to(dev)
    feat = extractor(tensor)  # (1, C, H, W)
    _, C, H, W = feat.shape
    patches = feat[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    return patches, (H, W)


def _coreset_subsample(features, ratio: float):
    import numpy as np
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


def _make_heatmap(patch_scores, spatial, orig_size: int = 224):
    import numpy as np
    import scipy.ndimage as ndi
    H, W = spatial
    score_map = patch_scores.reshape(H, W)
    zh, zw = orig_size / H, orig_size / W
    score_map = ndi.zoom(score_map, (zh, zw), order=1)
    score_map = ndi.gaussian_filter(score_map, sigma=4)
    mn, mx = score_map.min(), score_map.max()
    if mx > mn:
        score_map = (score_map - mn) / (mx - mn)
    # Jet colormap
    r = np.clip(1.5 - np.abs(score_map * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(score_map * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(score_map * 4 - 1), 0, 1)
    hmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return hmap


def _overlay(orig_img, hmap_arr, orig_size: int = 224):
    import numpy as np
    from PIL import Image
    orig = np.array(orig_img.resize((orig_size, orig_size))).astype(np.float32)
    heat = hmap_arr.astype(np.float32)
    blended = orig * 0.45 + heat * 0.55
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# Job handlers
# ══════════════════════════════════════════════════════════════════════════════

def _handle_aurora(inp: dict) -> dict:
    """Original Aurora time-series forecast handler."""
    if MODEL_ERROR:
        return {"error": f"Aurora model failed to load: {MODEL_ERROR}"}
    try:
        import torch
        import numpy as np

        input_series = inp["input_series"]
        mean_val     = float(inp["mean"])
        std_val      = float(inp["std"]) + EPS
        horizon      = min(int(inp.get("horizon", 168)), 720)
        num_samples  = min(int(inp.get("num_samples", 50)), 200)
        text_prompt  = inp.get("text_prompt")

        tensor = torch.tensor(input_series, dtype=torch.float32).unsqueeze(0).to(device)
        kwargs = dict(inputs=tensor, max_output_length=horizon,
                      num_samples=num_samples, inference_token_len=48)

        if text_prompt and hasattr(aurora_model, "tokenizer"):
            tok = aurora_model.tokenizer
            enc = tok(text_prompt, padding="max_length", truncation=True,
                      max_length=200, return_tensors="pt")
            kwargs["text_input_ids"]      = enc["input_ids"].to(device)
            kwargs["text_attention_mask"] = enc["attention_mask"].to(device)

        log.info(f"Aurora forecast: horizon={horizon}, samples={num_samples}")
        with torch.no_grad():
            out = aurora_model.generate(**kwargs)

        samples_norm = out.cpu().float().numpy()[0]
        samples_orig = np.clip(samples_norm * std_val + mean_val, 0, None)
        return {
            "samples": samples_orig.tolist(),
            "mean":    samples_orig.mean(axis=0).tolist(),
            "median":  np.median(samples_orig, axis=0).tolist(),
            "q10":     np.quantile(samples_orig, 0.10, axis=0).tolist(),
            "q90":     np.quantile(samples_orig, 0.90, axis=0).tolist(),
        }
    except Exception:
        return {"error": traceback.format_exc()}


def _handle_patchcore_train(inp: dict) -> dict:
    """
    PatchCore training on GPU.
    Input:  normal_images_b64 (list of base64 PNG/JPG strings)
            backbone, coreset_ratio, k_neighbors, input_size
    Output: memory_bank_b64 (pickled numpy array, base64),
            threshold, spatial_shape, metrics
    """
    try:
        import torch
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        images_b64    = inp["normal_images_b64"]
        coreset_ratio = float(inp.get("coreset_ratio", 0.1))
        k             = int(inp.get("k_neighbors", 9))
        size          = int(inp.get("input_size", 224))

        log.info(f"PatchCore train: {len(images_b64)} images, coreset={coreset_ratio}, GPU={device}")
        extractor = _get_patchcore_extractor(device)

        all_patches, spatial = [], None
        for b64 in images_b64:
            img = _b64_to_image(b64)
            patches, spatial = _extract_patches(img, extractor, device, size)
            all_patches.append(patches)

        if not all_patches:
            return {"error": "No images could be processed"}

        memory = np.vstack(all_patches).astype(np.float32)
        log.info(f"Memory bank: {memory.shape} → coreset subsampling…")
        core = _coreset_subsample(memory, coreset_ratio)
        log.info(f"Coreset: {core.shape}")

        # Compute training-set scores for threshold
        knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
        knn.fit(core)
        train_scores = []
        for b64 in images_b64[:min(len(images_b64), 20)]:
            img = _b64_to_image(b64)
            patches, _ = _extract_patches(img, extractor, device, size)
            dists, _ = knn.kneighbors(patches.astype(np.float32))
            train_scores.append(float(dists[:, 0].max()))
        threshold = float(np.percentile(train_scores, 95)) if train_scores else 0.5

        state = {
            "memory_bank": core,
            "spatial_shape": spatial,
            "threshold": threshold,
            "train_scores": train_scores,
            "input_size": size,
            "k": k,
        }
        state_b64 = base64.b64encode(pickle.dumps(state)).decode()

        return {
            "model_state_b64": state_b64,
            "threshold": threshold,
            "memory_bank_size": len(core),
            "original_patches": len(memory),
            "device": device.upper(),
            "metrics": {"train_images": len(images_b64), "threshold": threshold},
        }
    except Exception:
        return {"error": traceback.format_exc()}


def _handle_foundad_train(inp: dict) -> dict:
    """
    FoundAD training on GPU using DINOv2 ViT-B/14.
    Input:  normal_images_b64 (list, n_shot images),
            n_shot, encoder ("dinov2_vitb14"), steps
    Output: projector_state_b64, ref_features_b64, threshold, metrics
    """
    try:
        import torch
        import numpy as np

        images_b64  = inp["normal_images_b64"]
        n_shot      = min(int(inp.get("n_shot", 4)), len(images_b64))
        steps       = int(inp.get("steps", 1000))
        encoder_name = inp.get("encoder", "dinov2_vitb14")
        size        = int(inp.get("image_size", 518))

        log.info(f"FoundAD train: {n_shot}-shot, {steps} steps, {encoder_name}, GPU={device}")

        # Load DINOv2
        log.info("Loading DINOv2 encoder…")
        encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        encoder.eval().to(device)

        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
        ])

        # Extract reference features
        ref_features = []
        for b64 in images_b64[:n_shot]:
            img = _b64_to_image(b64)
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = encoder.forward_features(tensor)
                if "x_norm_patchtokens" in feat:
                    f = feat["x_norm_patchtokens"]
                else:
                    f = list(feat.values())[0]
            ref_features.append(f.cpu())
        ref_stack = torch.cat(ref_features, dim=1)

        # Train manifold projector
        D = ref_stack.shape[-1]
        projector = torch.nn.Sequential(
            torch.nn.Linear(D, D // 2),
            torch.nn.GELU(),
            torch.nn.Linear(D // 2, D),
        ).to(device)
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3, weight_decay=1e-4)
        ref_flat = ref_stack.squeeze(0).to(device)

        projector.train()
        for step in range(steps):
            idx = torch.randperm(ref_flat.shape[0])[:32]
            batch = ref_flat[idx]
            loss = torch.nn.functional.mse_loss(projector(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 200 == 0:
                log.info(f"  Step {step}/{steps} loss={loss.item():.5f}")

        projector.eval()
        with torch.no_grad():
            errors = torch.mean((projector(ref_flat) - ref_flat) ** 2, dim=-1).cpu().numpy()
        threshold = float(np.percentile(errors, 95))

        state = {
            "projector_state": {k: v.cpu() for k, v in projector.state_dict().items()},
            "ref_features": ref_stack.cpu(),
            "threshold": threshold,
            "D": D,
            "image_size": size,
            "n_shot": n_shot,
            "encoder_name": encoder_name,
        }
        state_b64 = base64.b64encode(pickle.dumps(state)).decode()

        return {
            "model_state_b64": state_b64,
            "threshold": threshold,
            "device": device.upper(),
            "metrics": {"n_shot": n_shot, "steps": steps, "threshold": threshold},
        }
    except Exception:
        return {"error": traceback.format_exc()}


def _handle_cv_inference(inp: dict) -> dict:
    """
    Run PatchCore or FoundAD inference on a single image.
    Input:  algorithm ("patchcore" or "foundad"),
            model_state_b64 (pickled state),
            image_b64, threshold (optional override)
    Output: anomaly_score, decision, threshold, heatmap_b64, overlay_b64, inference_ms
    """
    try:
        import torch
        import numpy as np
        import time
        from PIL import Image
        from sklearn.neighbors import NearestNeighbors

        algorithm = inp.get("algorithm", "patchcore")
        img = _b64_to_image(inp["image_b64"])
        state = pickle.loads(base64.b64decode(inp["model_state_b64"]))
        threshold_override = inp.get("threshold")

        t0 = time.time()

        if algorithm == "patchcore":
            size = state.get("input_size", 224)
            extractor = _get_patchcore_extractor(device)
            patches, spatial = _extract_patches(img, extractor, device, size)
            k = state.get("k", 9)
            knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
            knn.fit(state["memory_bank"])
            dists, _ = knn.kneighbors(patches.astype(np.float32))
            patch_scores = dists[:, 0]
            score = float(patch_scores.max())
            threshold = threshold_override if threshold_override is not None else state["threshold"]
            # Normalize
            train_max = max(state.get("train_scores", [score]), default=score) + EPS
            norm_score = min(score / train_max, 1.0)

        elif algorithm == "foundad":
            size = state.get("image_size", 518)
            D = state["D"]
            projector = torch.nn.Sequential(
                torch.nn.Linear(D, D // 2),
                torch.nn.GELU(),
                torch.nn.Linear(D // 2, D),
            )
            projector.load_state_dict(state["projector_state"])
            projector.eval().to(device)

            encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            encoder.eval().to(device)

            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((size, size)), T.ToTensor(),
                T.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
            ])
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = encoder.forward_features(tensor)
                patches = feat.get("x_norm_patchtokens", list(feat.values())[0]).squeeze(0)
                patch_errors = torch.mean((projector(patches) - patches) ** 2, dim=-1).cpu().numpy()
            patch_scores = patch_errors
            score = float(patch_scores.max())
            n = len(patch_scores)
            side = int(n ** 0.5)
            spatial = (side, side)
            threshold = threshold_override if threshold_override is not None else state["threshold"]
            norm_score = min(score / (threshold + EPS), 1.0)
        else:
            return {"error": f"Unknown algorithm: {algorithm}"}

        elapsed_ms = (time.time() - t0) * 1000
        decision = "anomalous" if score > threshold else "normal"

        # Generate heatmap
        hmap_arr = _make_heatmap(patch_scores, spatial, 224)
        hmap_img = Image.fromarray(hmap_arr)
        overlay_img = _overlay(img, hmap_arr, 224)

        return {
            "anomaly_score": round(float(score), 6),
            "normalized_score": round(float(norm_score), 4),
            "decision": decision,
            "threshold": round(float(threshold), 6),
            "inference_ms": round(elapsed_ms, 1),
            "heatmap_b64": _image_to_b64(hmap_img),
            "overlay_b64": _image_to_b64(overlay_img),
            "device": device.upper(),
        }
    except Exception:
        return {"error": traceback.format_exc()}


# ══════════════════════════════════════════════════════════════════════════════
# Main dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def handler(job: dict) -> dict:
    inp = job.get("input", {})
    job_type = inp.get("job_type", "aurora_forecast")
    log.info(f"Job received: job_type={job_type}")

    if job_type == "aurora_forecast":
        return _handle_aurora(inp)
    elif job_type == "cv_patchcore_train":
        return _handle_patchcore_train(inp)
    elif job_type == "cv_foundad_train":
        return _handle_foundad_train(inp)
    elif job_type == "cv_inference":
        return _handle_cv_inference(inp)
    else:
        return {"error": f"Unknown job_type '{job_type}'. Supported: aurora_forecast, cv_patchcore_train, cv_foundad_train, cv_inference"}


runpod.serverless.start({"handler": handler})
