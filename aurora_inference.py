"""
Aurora Inference Engine – Azure Web App
========================================
Lädt aurora_checkpoint.pt (lokal) oder fällt auf HuggingFace zurück.

Checkpoint-Pfad:
  Standard : aurora_checkpoint.pt  (neben app.py)
  Override  : Umgebungsvariable AURORA_MODEL_PATH

Checkpoint-Format (gespeichert in aurora_v4.ipynb Zelle 4b):
  {
    'model_state_dict': model.state_dict(),
    'aurora_version':   '0.2.0',
    'model_class':      'AuroraModel',
    'config': {
        'lookback':        528,
        'token_len':       48,
        'default_horizon': 96,
        'default_samples': 50,
        'n_parameters':    210848460,
    },
  }
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

log = logging.getLogger("aurora_inference")

EPS = 1e-8


def _remap_vision_encoder_keys(state_dict: dict) -> dict:
    """
    Mappt alte HuggingFace BeitModel-Keys (transformers < 4.50)
    auf neue SiglipVisionModel-Keys (transformers >= 4.50).

    Alte Struktur: encoder.layer.X.attention.attention.query.*
    Neue Struktur: layers.X.attention.q_proj.*
    """
    import re
    new_sd = {}
    for k, v in state_dict.items():
        # 1. encoder.layer.X → layers.X
        k = re.sub(r'\.encoder\.layer\.(\d+)\.', r'.layers.\1.', k)
        # 2. Attention-Projektionen
        k = k.replace('.attention.attention.query.', '.attention.q_proj.')
        k = k.replace('.attention.attention.key.',   '.attention.k_proj.')
        k = k.replace('.attention.attention.value.', '.attention.v_proj.')
        # 3. Output-Projektion (muss vor .output.dense kommen)
        k = k.replace('.attention.output.dense.', '.attention.o_proj.')
        # 4. MLP-Layer
        k = k.replace('.intermediate.dense.', '.mlp.fc1.')
        k = k.replace('.output.dense.',       '.mlp.fc2.')
        new_sd[k] = v
    return new_sd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Konfiguration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AuroraConfig:
    """
    Alle Inference-Parameter. Werte können per Umgebungsvariable
    oder aus dem Checkpoint überschrieben werden.
    """
    model_path:      str   = "aurora_checkpoint.pt"
    lookback:        int   = 528
    token_len:       int   = 48
    eps:             float = 1e-8
    device:          str   = "auto"
    text_max_len:    int   = 200
    max_horizon:     int   = 720   # max 30 Tage á 24 h
    max_samples:     int   = 200
    default_horizon: int   = 96
    default_samples: int   = 50

    @classmethod
    def from_env(cls) -> "AuroraConfig":
        """Erzeugt Config aus Defaults + Umgebungsvariablen."""
        cfg = cls()
        cfg.model_path = os.getenv("AURORA_MODEL_PATH", cfg.model_path)
        cfg.device     = os.getenv("AURORA_DEVICE",     cfg.device)
        return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Daten-Strukturen
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AuroraInput:
    tensor:              torch.Tensor              # (1, lookback)
    means:               np.ndarray               # (1, 1)
    stds:                np.ndarray               # (1, 1)
    columns:             list
    mode:                str                      # immer 'unimodal'
    history_values:      np.ndarray               # (lookback,)
    history_timestamps:  Optional[pd.DatetimeIndex] = None
    freq:                str = "h"


@dataclass
class ForecastResult:
    samples_orig:        np.ndarray               # (S, H)
    mean:                np.ndarray               # (H,)
    median:              np.ndarray               # (H,)
    q_lo:                np.ndarray               # (H,)
    q_hi:                np.ndarray               # (H,)
    horizon:             int
    num_samples:         int
    target_column:       str
    text_used:           Optional[str]
    history_values:      np.ndarray
    history_timestamps:  Optional[pd.DatetimeIndex] = None
    forecast_timestamps: Optional[pd.DatetimeIndex] = None
    daily_mean:          Optional[np.ndarray] = None
    daily_q10:           Optional[np.ndarray] = None
    daily_q90:           Optional[np.ndarray] = None
    daily_dates:         Optional[list] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Inference Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AuroraInference:
    """
    Singleton-kompatibler Inference-Wrapper für die Aurora Web App.

    Ladereihenfolge:
      1. Lokaler Checkpoint (AURORA_MODEL_PATH / aurora_checkpoint.pt)
         → Architektur via aurora.load_model(), Gewichte aus .pt
      2. Fallback: aurora.load_model() lädt direkt von HuggingFace
         (benötigt Internet-Zugang + ~1.5 GB Download)
    """

    def __init__(self, config: Optional[AuroraConfig] = None):
        self.config  = config or AuroraConfig.from_env()
        self.device  = self._resolve_device()
        self.model   = None
        self._loaded = False

    # ── Interne Hilfsmethoden ─────────────────────────────────────────

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    # ── Modell laden ──────────────────────────────────────────────────

    def load(self) -> None:
        """
        Lädt Aurora einmalig. Methode ist idempotent.

        Mit lokalem Checkpoint:
          - aurora.load_model() gibt die Modell-Architektur (leer oder pretrained)
          - load_state_dict() überschreibt die Gewichte mit dem Checkpoint
          - Kein HuggingFace-Download der Gewichte nötig

        Ohne Checkpoint: vollständiger HuggingFace-Download (~1.5 GB).
        """
        if self._loaded:
            return

        from aurora import load_model

        pt_path = Path(self.config.model_path)

        if pt_path.exists():
            size_mb = pt_path.stat().st_size / 1024 ** 2
            log.info(f"Lade Checkpoint: {pt_path}  ({size_mb:.0f} MB)")

            ckpt = torch.load(
                pt_path,
                map_location=self.device,
                weights_only=False,   # benötigt für Aurora-Architektur-Objekte
            )

            self.model = load_model()

            sd = _remap_vision_encoder_keys(ckpt["model_state_dict"])
            result = self.model.load_state_dict(sd, strict=False)
            if result.missing_keys:
                log.warning(f"Fehlende Keys nach Remap: {len(result.missing_keys)} "
                            f"(erste 3: {result.missing_keys[:3]})")
            if result.unexpected_keys:
                log.warning(f"Unerwartete Keys nach Remap: {len(result.unexpected_keys)} "
                            f"(erste 3: {result.unexpected_keys[:3]})")
            if not result.missing_keys and not result.unexpected_keys:
                log.info("Checkpoint vollständig geladen — alle Keys gemappt.")

            # Konfiguration aus Checkpoint-Metadaten übernehmen
            if "config" in ckpt:
                c = ckpt["config"]
                self.config.lookback  = c.get("lookback",   self.config.lookback)
                self.config.token_len = c.get("token_len",  self.config.token_len)
                log.info(
                    f"Checkpoint v{ckpt.get('aurora_version','?')} | "
                    f"{c.get('n_parameters', 0):,} Parameter"
                )
        else:
            log.warning(
                f"Kein Checkpoint unter '{pt_path}' — "
                "lade Modell von HuggingFace (benötigt Internet)"
            )
            self.model = load_model()

        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True

        n = sum(p.numel() for p in self.model.parameters())
        log.info(f"Aurora bereit | {n:,} Parameter | Gerät: {self.device.upper()}")

    @property
    def ready(self) -> bool:
        return self._loaded

    # ── Daten-Vorbereitung ────────────────────────────────────────────

    def prepare(
        self,
        df:         pd.DataFrame,
        target_col: str,
        time_col:   Optional[str] = None,
        freq:       str = "h",
    ) -> AuroraInput:
        """
        DataFrame → AuroraInput (normalisierter Tensor + Metadaten).

        Die letzten `lookback` Zeilen werden als Eingabe-Fenster verwendet.
        """
        df = df.copy()
        timestamps = None

        if time_col and time_col in df.columns:
            timestamps = pd.to_datetime(df[time_col])
            df = df.drop(columns=[time_col])

        df = df.select_dtypes(include=[np.number])

        if target_col not in df.columns:
            target_col = df.columns[0]

        arr = df[[target_col]].values.astype(np.float32)
        arr = pd.DataFrame(arr).ffill().bfill().values.astype(np.float32)

        T = arr.shape[0]
        if T < self.config.lookback:
            raise ValueError(
                f"Zu wenig Daten: {T} Zeilen übergeben, "
                f"mindestens {self.config.lookback} benötigt."
            )

        window = arr[-self.config.lookback:, 0]
        mean   = float(window.mean())
        std    = float(window.std()) + self.config.eps
        norm_w = (window - mean) / std

        tensor = (
            torch.tensor(norm_w, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        hist_ts = None
        if timestamps is not None:
            hist_ts = pd.DatetimeIndex(
                timestamps.iloc[-self.config.lookback:].values
            )

        return AuroraInput(
            tensor             = tensor,
            means              = np.array([[mean]]),
            stds               = np.array([[std]]),
            columns            = [target_col],
            mode               = "unimodal",
            history_values     = window,
            history_timestamps = hist_ts,
            freq               = freq,
        )

    # ── Inference ─────────────────────────────────────────────────────

    def forecast(
        self,
        aurora_input: AuroraInput,
        horizon:      int = 168,
        num_samples:  int = 50,
        text_prompt:  Optional[str] = None,
    ) -> ForecastResult:
        """
        Probabilistischer Forecast aus einem AuroraInput.

        Parameters
        ----------
        aurora_input : Ausgabe von prepare()
        horizon      : Vorhersage-Schritte (max: config.max_horizon)
        num_samples  : Anzahl Szenarien (max: config.max_samples)
        text_prompt  : Optionaler englischer Domänen-Kontext

        Returns
        -------
        ForecastResult mit samples_orig (S, H), mean, median, q_lo, q_hi
        sowie optionaler Tages-Aggregation für stündliche Daten.
        """
        if not self._loaded:
            self.load()

        horizon     = min(int(horizon),     self.config.max_horizon)
        num_samples = min(int(num_samples), self.config.max_samples)

        kwargs = dict(
            inputs              = aurora_input.tensor,
            max_output_length   = horizon,
            num_samples         = num_samples,
            inference_token_len = self.config.token_len,
        )

        # Text-Prompt tokenisieren (nur wenn Modell einen Tokenizer hat)
        if text_prompt and hasattr(self.model, "tokenizer"):
            tok = self.model.tokenizer
            enc = tok(
                text_prompt,
                padding    = "max_length",
                truncation = True,
                max_length = self.config.text_max_len,
                return_tensors = "pt",
            )
            kwargs["text_input_ids"]      = enc["input_ids"].to(self.device)
            kwargs["text_attention_mask"] = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.model.generate(**kwargs)   # (1, S, H)

        out_np = out.cpu().float().numpy()
        samples_norm = out_np[0]                  # (S, H)

        # De-Normalisierung + Clip auf ≥ 0
        std    = aurora_input.stds[0, 0]
        mean   = aurora_input.means[0, 0]
        samples_orig = np.clip(samples_norm * (std + self.config.eps) + mean, 0, None)

        mean_f   = samples_orig.mean(axis=0)
        median_f = np.median(samples_orig, axis=0)
        q_lo     = np.quantile(samples_orig, 0.10, axis=0)
        q_hi     = np.quantile(samples_orig, 0.90, axis=0)

        # Forecast-Zeitstempel
        fore_ts = None
        if aurora_input.history_timestamps is not None:
            last_ts = aurora_input.history_timestamps[-1]
            fore_ts = pd.date_range(
                start  = last_ts + pd.tseries.frequencies.to_offset(aurora_input.freq),
                periods= horizon,
                freq   = aurora_input.freq,
            )

        # Tages-Aggregation (nur für stündliche Daten)
        daily_mean = daily_q10 = daily_q90 = daily_dates = None
        if fore_ts is not None and aurora_input.freq == "h":
            df_s = pd.DataFrame(samples_orig.T, index=fore_ts).clip(lower=0)
            daily = df_s.resample("D").sum()
            daily_mean  = daily.mean(axis=1).values
            daily_q10   = daily.quantile(0.10, axis=1).values
            daily_q90   = daily.quantile(0.90, axis=1).values
            daily_dates = [d.strftime("%a %d.%m.") for d in daily.index]

        return ForecastResult(
            samples_orig       = samples_orig,
            mean               = mean_f,
            median             = median_f,
            q_lo               = q_lo,
            q_hi               = q_hi,
            horizon            = horizon,
            num_samples        = num_samples,
            target_column      = aurora_input.columns[0],
            text_used          = text_prompt,
            history_values     = aurora_input.history_values,
            history_timestamps = aurora_input.history_timestamps,
            forecast_timestamps= fore_ts,
            daily_mean         = daily_mean,
            daily_q10          = daily_q10,
            daily_q90          = daily_q90,
            daily_dates        = daily_dates,
        )
