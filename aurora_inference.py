"""
Aurora Inference – RunPod Serverless Client
===========================================
Sendet Anfragen an den RunPod-Endpoint statt Aurora lokal zu laden.
Konfiguration über Umgebungsvariablen:
  RUNPOD_API_KEY    – RunPod API-Schlüssel
  RUNPOD_ENDPOINT_ID – ID des Serverless-Endpoints
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("aurora_inference")

RUNPOD_BASE = "https://api.runpod.ai/v2"
POLL_INTERVAL = 3   # Sekunden zwischen Status-Abfragen
MAX_WAIT = 600       # max. 10 Minuten warten


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Konfiguration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AuroraConfig:
    model_path:      str   = ""
    lookback:        int   = 528
    token_len:       int   = 48
    eps:             float = 1e-8
    device:          str   = "runpod"
    text_max_len:    int   = 200
    max_horizon:     int   = 720
    max_samples:     int   = 200
    default_horizon: int   = 96
    default_samples: int   = 50

    @classmethod
    def from_env(cls) -> "AuroraConfig":
        return cls()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Daten-Strukturen (identisch zu vorher, damit app.py unverändert bleibt)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AuroraInput:
    tensor:              object
    means:               np.ndarray
    stds:                np.ndarray
    columns:             list
    mode:                str
    history_values:      np.ndarray
    history_timestamps:  Optional[pd.DatetimeIndex] = None
    freq:                str = "h"


@dataclass
class ForecastResult:
    samples_orig:        np.ndarray
    mean:                np.ndarray
    median:              np.ndarray
    q_lo:                np.ndarray
    q_hi:                np.ndarray
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RunPod Client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AuroraInference:
    """
    Schickt Forecast-Anfragen an den RunPod Serverless Endpoint.
    Aurora läuft auf RunPod-GPU – nicht lokal im Container.
    """

    def __init__(self, config: Optional[AuroraConfig] = None):
        self.config      = config or AuroraConfig.from_env()
        self.device      = "runpod"
        self._api_key    = os.getenv("RUNPOD_API_KEY", "")
        self._endpoint   = os.getenv("RUNPOD_ENDPOINT_ID", "")
        self._loaded     = bool(self._api_key and self._endpoint)

        if not self._api_key:
            log.warning("RUNPOD_API_KEY nicht gesetzt – Inference nicht verfügbar.")
        if not self._endpoint:
            log.warning("RUNPOD_ENDPOINT_ID nicht gesetzt – Inference nicht verfügbar.")

    def load(self) -> None:
        # Kein lokales Laden nötig – RunPod startet den Worker bei Bedarf
        self._loaded = bool(self._api_key and self._endpoint)

    @property
    def ready(self) -> bool:
        return bool(self._api_key and self._endpoint)

    # ── Daten-Vorbereitung ─────────────────────────────────────────

    def prepare(
        self,
        df:         pd.DataFrame,
        target_col: str,
        time_col:   Optional[str] = None,
        freq:       str = "h",
    ) -> AuroraInput:
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
        lookback = self.config.lookback
        if T < lookback:
            raise ValueError(
                f"Zu wenig Daten: {T} Zeilen, mindestens {lookback} benötigt."
            )

        window = arr[-lookback:, 0]
        mean   = float(window.mean())
        std    = float(window.std()) + self.config.eps
        norm_w = (window - mean) / std

        hist_ts = None
        if timestamps is not None:
            hist_ts = pd.DatetimeIndex(
                timestamps.iloc[-lookback:].values
            )

        return AuroraInput(
            tensor             = norm_w.tolist(),   # JSON-serialisierbar für RunPod
            means              = np.array([[mean]]),
            stds               = np.array([[std]]),
            columns            = [target_col],
            mode               = "unimodal",
            history_values     = window,
            history_timestamps = hist_ts,
            freq               = freq,
        )

    # ── RunPod API ─────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _submit_job(self, payload: dict) -> str:
        url = f"{RUNPOD_BASE}/{self._endpoint}/run"
        r = requests.post(url, json={"input": payload}, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()["id"]

    def _poll_job(self, job_id: str) -> dict:
        url = f"{RUNPOD_BASE}/{self._endpoint}/status/{job_id}"
        waited = 0
        while waited < MAX_WAIT:
            r = requests.get(url, headers=self._headers(), timeout=30)
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            if status == "COMPLETED":
                return data["output"]
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod Job fehlgeschlagen: {status} – {data.get('error', '')}")
            time.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
        raise TimeoutError(f"RunPod Job {job_id} nach {MAX_WAIT}s noch nicht fertig.")

    # ── Inference ──────────────────────────────────────────────────

    def forecast(
        self,
        aurora_input: AuroraInput,
        horizon:      int = 168,
        num_samples:  int = 50,
        text_prompt:  Optional[str] = None,
    ) -> ForecastResult:
        if not self.ready:
            raise RuntimeError(
                "RunPod nicht konfiguriert. "
                "Bitte RUNPOD_API_KEY und RUNPOD_ENDPOINT_ID als Umgebungsvariablen setzen."
            )

        horizon     = min(int(horizon),     self.config.max_horizon)
        num_samples = min(int(num_samples), self.config.max_samples)

        payload = {
            "input_series":  aurora_input.tensor,
            "mean":          float(aurora_input.means[0, 0]),
            "std":           float(aurora_input.stds[0, 0]),
            "freq":          aurora_input.freq,
            "horizon":       horizon,
            "num_samples":   num_samples,
            "text_prompt":   text_prompt,
        }

        log.info("Sende Job an RunPod Endpoint %s …", self._endpoint)
        job_id = self._submit_job(payload)
        log.info("Job ID: %s – warte auf Ergebnis …", job_id)
        output = self._poll_job(job_id)

        samples_orig = np.array(output["samples"])      # (S, H)
        mean_f       = np.array(output["mean"])
        median_f     = np.array(output["median"])
        q_lo         = np.array(output["q10"])
        q_hi         = np.array(output["q90"])

        # Forecast-Zeitstempel
        fore_ts = None
        if aurora_input.history_timestamps is not None:
            last_ts = aurora_input.history_timestamps[-1]
            fore_ts = pd.date_range(
                start   = last_ts + pd.tseries.frequencies.to_offset(aurora_input.freq),
                periods = horizon,
                freq    = aurora_input.freq,
            )

        # Tages-Aggregation
        daily_mean = daily_q10 = daily_q90 = daily_dates = None
        if fore_ts is not None and aurora_input.freq == "h":
            df_s = pd.DataFrame(samples_orig.T, index=fore_ts).clip(lower=0)
            daily = df_s.resample("D").sum()
            daily_mean  = daily.mean(axis=1).values
            daily_q10   = daily.quantile(0.10, axis=1).values
            daily_q90   = daily.quantile(0.90, axis=1).values
            daily_dates = [d.strftime("%a %d.%m.") for d in daily.index]

        return ForecastResult(
            samples_orig        = samples_orig,
            mean                = mean_f,
            median              = median_f,
            q_lo                = q_lo,
            q_hi                = q_hi,
            horizon             = horizon,
            num_samples         = num_samples,
            target_column       = aurora_input.columns[0],
            text_used           = text_prompt,
            history_values      = aurora_input.history_values,
            history_timestamps  = aurora_input.history_timestamps,
            forecast_timestamps = fore_ts,
            daily_mean          = daily_mean,
            daily_q10           = daily_q10,
            daily_q90           = daily_q90,
            daily_dates         = daily_dates,
        )
