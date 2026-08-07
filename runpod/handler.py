"""
RunPod Serverless Handler – Aurora Inference
=============================================
Dieses Script läuft auf der RunPod-GPU.
Es empfängt Zeitreihendaten, führt Aurora-Inference durch
und gibt Forecast-Ergebnisse als JSON zurück.

Erwartet input:
  input_series  – Liste von Floats (normalisierte Zeitreihe, lookback=528)
  mean          – Originalmittelwert (für De-Normalisierung)
  std           – Originalstandardabweichung
  freq          – "h", "D" oder "W"
  horizon       – Anzahl Vorhersage-Schritte
  num_samples   – Anzahl Szenarien (10–200)
  text_prompt   – Optionaler Domänen-Kontext (kann None sein)

Gibt zurück:
  samples  – (S, H) Liste von Listen
  mean     – (H,) Liste
  median   – (H,) Liste
  q10      – (H,) Liste
  q90      – (H,) Liste
"""

import logging
import os
import re
from pathlib import Path

import numpy as np
import torch
import runpod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aurora-handler")

# ── Modell einmalig beim Worker-Start laden ───────────────────────────────
CHECKPOINT_PATH = Path(os.getenv("AURORA_MODEL_PATH", "/app/aurora_checkpoint.pt"))
EPS = 1e-8

log.info("Lade Aurora Modell …")

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

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Gerät: {device.upper()}")

model = load_model()

if CHECKPOINT_PATH.exists():
    log.info(f"Lade Checkpoint: {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size / 1024**2:.0f} MB)")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    sd = _remap_keys(ckpt["model_state_dict"])
    result = model.load_state_dict(sd, strict=False)
    log.info(f"Checkpoint geladen | fehlende Keys: {len(result.missing_keys)}")
else:
    log.warning(f"Kein Checkpoint unter {CHECKPOINT_PATH} – lade von HuggingFace …")

model = model.to(device)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
log.info(f"Aurora bereit | {n_params:,} Parameter")


# ── RunPod Handler ────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    inp = job["input"]

    input_series = inp["input_series"]   # Liste von Floats
    mean_val     = float(inp["mean"])
    std_val      = float(inp["std"]) + EPS
    horizon      = int(inp.get("horizon", 168))
    num_samples  = int(inp.get("num_samples", 50))
    text_prompt  = inp.get("text_prompt")

    horizon     = min(horizon, 720)
    num_samples = min(num_samples, 200)

    tensor = torch.tensor(input_series, dtype=torch.float32).unsqueeze(0).to(device)

    kwargs = dict(
        inputs              = tensor,
        max_output_length   = horizon,
        num_samples         = num_samples,
        inference_token_len = 48,
    )

    if text_prompt and hasattr(model, "tokenizer"):
        tok = model.tokenizer
        enc = tok(
            text_prompt,
            padding        = "max_length",
            truncation     = True,
            max_length     = 200,
            return_tensors = "pt",
        )
        kwargs["text_input_ids"]      = enc["input_ids"].to(device)
        kwargs["text_attention_mask"] = enc["attention_mask"].to(device)

    log.info(f"Starte Inference: horizon={horizon}, samples={num_samples}")

    with torch.no_grad():
        out = model.generate(**kwargs)   # (1, S, H)

    samples_norm = out.cpu().float().numpy()[0]   # (S, H)
    samples_orig = np.clip(samples_norm * std_val + mean_val, 0, None)

    mean_f   = samples_orig.mean(axis=0)
    median_f = np.median(samples_orig, axis=0)
    q10      = np.quantile(samples_orig, 0.10, axis=0)
    q90      = np.quantile(samples_orig, 0.90, axis=0)

    log.info("Inference abgeschlossen.")

    return {
        "samples": samples_orig.tolist(),
        "mean":    mean_f.tolist(),
        "median":  median_f.tolist(),
        "q10":     q10.tolist(),
        "q90":     q90.tolist(),
    }


runpod.serverless.start({"handler": handler})
