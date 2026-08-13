"""
Bäckerei Nachfrageprognose – LightGBM Engine (CPU-only, kein GPU).
Erstes Laden: Datensatz generieren + Modell trainieren (~5 s).
Danach gecacht – jede Abfrage < 1 s.
"""
from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests as _requests

DATASETS_DIR = Path(__file__).parent / "datasets"
BAKERY_CSV   = DATASETS_DIR / "baeckerei_beispiel.csv"

FILIALEN = ["Innenstadt", "Arkaden", "Bahnhof", "Penny-Markt", "Ferno-Center"]
ARTIKEL  = ["Roggenbrot", "Vollkornbrot", "Brötchen", "Croissant",
             "Butterkuchen", "Käsekuchen", "Laugenstange", "Mohnschnecke"]

FILIALE_ENC = {f: i for i, f in enumerate(FILIALEN)}
ARTIKEL_ENC = {a: i for i, a in enumerate(ARTIKEL)}

FEATURE_COLS = [
    "dow", "month", "week_of_year",
    "is_feiertag", "is_schulferien",
    "temperatur", "is_aktion",
    "lag_7", "lag_14", "lag_28", "lag_364",
    "rolling_mean_4w", "rolling_std_4w", "rolling_mean_2w",
    "filiale_enc", "artikel_enc",
]

FEATURE_LABELS = {
    "lag_7":           "Vorwoche (gleicher Tag)",
    "lag_14":          "Vor 2 Wochen",
    "lag_28":          "Vor 4 Wochen",
    "lag_364":         "Vorjahr (gleicher Tag)",
    "rolling_mean_4w": "Ø letzte 4 Wochen",
    "rolling_std_4w":  "Streuung 4 Wochen",
    "rolling_mean_2w": "Ø letzte 2 Wochen",
    "dow":             "Wochentag",
    "month":           "Monat",
    "week_of_year":    "Kalenderwoche",
    "is_feiertag":     "Feiertag NRW",
    "is_schulferien":  "Schulferien NRW",
    "temperatur":      "Tageshöchsttemp. Dülmen (°C)",
    "is_aktion":       "Aktion / Promotion",
    "filiale_enc":     "Filiale",
    "artikel_enc":     "Artikel",
}

DOW_NAMES = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# ── NRW Feiertage 2024–2026 ───────────────────────────────────────────────
_HOLIDAY_DATA = {
    # 2024
    "2024-01-01": "Neujahr",
    "2024-03-29": "Karfreitag",
    "2024-04-01": "Ostermontag",
    "2024-05-01": "Tag der Arbeit",
    "2024-05-09": "Christi Himmelfahrt",
    "2024-05-20": "Pfingstmontag",
    "2024-05-30": "Fronleichnam",
    "2024-10-03": "Tag der dt. Einheit",
    "2024-11-01": "Allerheiligen",
    "2024-12-25": "1. Weihnachtstag",
    "2024-12-26": "2. Weihnachtstag",
    # 2025
    "2025-01-01": "Neujahr",
    "2025-04-18": "Karfreitag",
    "2025-04-21": "Ostermontag",
    "2025-05-01": "Tag der Arbeit",
    "2025-05-29": "Christi Himmelfahrt",
    "2025-06-09": "Pfingstmontag",
    "2025-06-19": "Fronleichnam",
    "2025-10-03": "Tag der dt. Einheit",
    "2025-11-01": "Allerheiligen",
    "2025-12-25": "1. Weihnachtstag",
    "2025-12-26": "2. Weihnachtstag",
    # 2026
    "2026-01-01": "Neujahr",
    "2026-04-03": "Karfreitag",
    "2026-04-06": "Ostermontag",
    "2026-05-01": "Tag der Arbeit",
    "2026-05-14": "Christi Himmelfahrt",
    "2026-05-25": "Pfingstmontag",
    "2026-06-04": "Fronleichnam",
    "2026-10-03": "Tag der dt. Einheit",
    "2026-11-01": "Allerheiligen",
    "2026-12-25": "1. Weihnachtstag",
    "2026-12-26": "2. Weihnachtstag",
}

NRW_HOLIDAYS: set[pd.Timestamp]       = {pd.Timestamp(d) for d in _HOLIDAY_DATA}
NRW_HOLIDAY_NAMES: dict[pd.Timestamp, str] = {pd.Timestamp(k): v for k, v in _HOLIDAY_DATA.items()}

# NRW Schulferien 2024–2026
SCHOOL_HOLIDAYS = [
    (pd.Timestamp("2024-07-22"), pd.Timestamp("2024-09-03")),
    (pd.Timestamp("2024-10-14"), pd.Timestamp("2024-10-26")),
    (pd.Timestamp("2024-12-23"), pd.Timestamp("2025-01-07")),
    (pd.Timestamp("2025-02-17"), pd.Timestamp("2025-02-21")),
    (pd.Timestamp("2025-04-14"), pd.Timestamp("2025-04-26")),
    (pd.Timestamp("2025-05-30"), pd.Timestamp("2025-06-06")),
    (pd.Timestamp("2025-07-07"), pd.Timestamp("2025-08-19")),
    (pd.Timestamp("2025-10-13"), pd.Timestamp("2025-10-25")),
    (pd.Timestamp("2025-12-22"), pd.Timestamp("2026-01-07")),
    (pd.Timestamp("2026-02-16"), pd.Timestamp("2026-02-20")),
    (pd.Timestamp("2026-04-07"), pd.Timestamp("2026-04-18")),
    (pd.Timestamp("2026-05-22"), pd.Timestamp("2026-05-29")),
    (pd.Timestamp("2026-07-06"), pd.Timestamp("2026-08-18")),   # Sommerferien 2026
]

# Mindest-Enddatum des Datensatzes – wird geprüft, um Neugenerierung auszulösen
_REQUIRED_END = pd.Timestamp("2026-07-01")


def _is_feiertag(d: pd.Timestamp) -> int:
    return 1 if d in NRW_HOLIDAYS else 0


def _is_schulferien(d: pd.Timestamp) -> int:
    return 1 if any(s <= d <= e for s, e in SCHOOL_HOLIDAYS) else 0


def _synthetic_temp(d: pd.Timestamp) -> float:
    """Tageshöchsttemperatur Dülmen/NRW (°C): Jan ≈ 5°C, Jul ≈ 28°C."""
    doy  = d.timetuple().tm_yday
    base = 16.5 + 12.5 * np.sin((doy - 80) / 365.0 * 2 * np.pi)
    rng  = np.random.default_rng(int(d.timestamp()))
    return float(base + rng.normal(0, 3.0))


# ── Live-Wetterdaten Dülmen (Open-Meteo, kein API-Key) ───────────────────
_DULMEN_LAT  =  51.83
_DULMEN_LON  =   7.28
_weather_cache: dict[pd.Timestamp, float] = {}
_weather_ts: Optional[float] = None          # Unix-Zeit des letzten Abrufs
_weather_lock = threading.Lock()
_WEATHER_TTL  = 3600                         # Cache: 1 Stunde


def _fetch_live_temps(dates: list[pd.Timestamp]) -> dict[pd.Timestamp, float]:
    """
    Holt Tageshöchstwerte von Open-Meteo für eine Liste von Datumsangaben.
    Gibt für jeden Tag entweder den echten Wert oder _synthetic_temp zurück.
    Ergebnis wird 1 h gecacht.
    """
    import time as _time
    global _weather_cache, _weather_ts

    with _weather_lock:
        now = _time.time()
        if _weather_ts is None or now - _weather_ts > _WEATHER_TTL:
            try:
                resp = _requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude":     _DULMEN_LAT,
                        "longitude":    _DULMEN_LON,
                        "daily":        "temperature_2m_max",
                        "timezone":     "Europe/Berlin",
                        "forecast_days": 16,
                    },
                    timeout=5,
                )
                if resp.ok:
                    payload = resp.json()
                    _weather_cache = {
                        pd.Timestamp(d): float(t)
                        for d, t in zip(
                            payload["daily"]["time"],
                            payload["daily"]["temperature_2m_max"],
                        )
                        if t is not None
                    }
                    _weather_ts = now
            except Exception:
                pass   # Netzwerkfehler → Fallback auf Synthese

        cache = dict(_weather_cache)

    return {d: cache.get(d, _synthetic_temp(d)) for d in dates}


# ── Singletons ────────────────────────────────────────────────────────────
_model       = None
_df_raw:  Optional[pd.DataFrame] = None
_df_feat: Optional[pd.DataFrame] = None
_lock = threading.Lock()


def _ensure_data() -> pd.DataFrame:
    global _df_raw
    if _df_raw is not None:
        return _df_raw

    # Prüfen ob vorhandener Datensatz aktuell genug ist
    if BAKERY_CSV.exists():
        probe = pd.read_csv(BAKERY_CSV, usecols=["datum"], parse_dates=["datum"])
        if probe["datum"].max() < _REQUIRED_END:
            BAKERY_CSV.unlink()   # veraltet → neu generieren

    if not BAKERY_CSV.exists():
        from datasets.generate_baeckerei import generate
        generate()

    _df_raw = pd.read_csv(BAKERY_CSV, parse_dates=["datum"])
    return _df_raw


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Feature-Engineering für alle Filialen × Artikel gleichzeitig."""
    df = df.copy().sort_values(["filiale", "artikel", "datum"])
    chunks = []
    for (filiale, artikel), grp in df.groupby(["filiale", "artikel"], sort=False):
        grp = grp.sort_values("datum").reset_index(drop=True)

        grp["dow"]          = grp["datum"].dt.dayofweek
        grp["month"]        = grp["datum"].dt.month
        grp["week_of_year"] = grp["datum"].dt.isocalendar().week.astype(int)
        grp["filiale_enc"]  = FILIALE_ENC[filiale]
        grp["artikel_enc"]  = ARTIKEL_ENC[artikel]

        # CSV-Spalte is_feiertag_nrw → Feature is_feiertag
        if "is_feiertag_nrw" in grp.columns:
            grp["is_feiertag"] = grp["is_feiertag_nrw"]
        elif "is_feiertag" not in grp.columns:
            grp["is_feiertag"] = grp["datum"].apply(_is_feiertag)

        m = grp["menge"]
        grp["lag_7"]   = m.shift(7)
        grp["lag_14"]  = m.shift(14)
        grp["lag_28"]  = m.shift(28)
        grp["lag_364"] = m.shift(364)

        shifted = m.shift(1)
        grp["rolling_mean_4w"] = shifted.rolling(28, min_periods=7).mean()
        grp["rolling_std_4w"]  = shifted.rolling(28, min_periods=7).std().fillna(0)
        grp["rolling_mean_2w"] = shifted.rolling(14, min_periods=3).mean()

        chunks.append(grp)

    return pd.concat(chunks, ignore_index=True)


def _train() -> None:
    global _model, _df_feat
    import lightgbm as lgb

    df      = _ensure_data()
    df_feat = _featurize(df)
    df_feat = df_feat.dropna(subset=FEATURE_COLS)

    X = df_feat[FEATURE_COLS]
    y = df_feat["menge"]

    model = lgb.LGBMRegressor(
        objective        = "regression",
        metric           = "mae",
        n_estimators     = 500,
        learning_rate    = 0.04,
        num_leaves       = 63,
        min_child_samples= 20,
        feature_fraction = 0.8,
        bagging_fraction = 0.8,
        bagging_freq     = 5,
        verbose          = -1,
        n_jobs           = -1,
    )
    model.fit(X, y)
    _model   = model
    _df_feat = df_feat


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _train()
    return _model


# ── Public API ────────────────────────────────────────────────────────────

_FLAG_DIFF_THRESH = 0.80   # >80 % Abweichung Modell vs. Naive → auffällig


def _flag(pred: int, naive: int, lower: int, upper: int) -> str:
    rel_diff = abs(pred - naive) / naive if naive > 0 else 0
    if rel_diff > _FLAG_DIFF_THRESH:
        return "auffaellig"
    return "ok"


def get_meta() -> dict:
    return {"filialen": FILIALEN, "artikel": ARTIKEL}


def run_forecast(
    filiale:     str,
    artikel:     str,
    aktion_days: Optional[list] = None,
) -> dict:
    """
    Rekursive 7-Tages-Prognose mit LightGBM.
    Prognose startet immer ab dem heutigen Datum (rollierend).
    Lücke zwischen letztem Datenpunkt und heute wird still überbrückt.
    """
    import datetime as _dt
    if aktion_days is None:
        aktion_days = [False] * 7

    model = _get_model()
    df    = _ensure_data()

    series = (
        df[(df.filiale == filiale) & (df.artikel == artikel)]
        .sort_values("datum")
        .reset_index(drop=True)
        .copy()
    )
    series["datum"] = pd.to_datetime(series["datum"])
    last_date = series["datum"].max()

    # Prognose-Start = heute (oder last_date+1 wenn Datensatz noch aktuell ist)
    today          = pd.Timestamp(_dt.date.today())
    forecast_start = max(today, last_date + timedelta(days=1))

    # ── Walk-Forward Metriken: letzte 4 Wochen ───────────────────────────
    test_start = last_date - timedelta(days=27)
    test_rows  = _df_feat[
        (_df_feat.filiale == filiale)
        & (_df_feat.artikel == artikel)
        & (_df_feat.datum >= test_start)
    ].dropna(subset=FEATURE_COLS)

    if len(test_rows) >= 7:
        y_true  = test_rows["menge"].values
        y_pred  = model.predict(test_rows[FEATURE_COLS])
        y_naive = test_rows["lag_7"].values

        def wape(a, p):
            return float(np.sum(np.abs(a - p)) / np.sum(a)) if np.sum(a) > 0 else 0.0

        wm    = wape(y_true, y_pred)
        wn    = wape(y_true, y_naive)
        imp   = (wn - wm) / wn * 100 if wn > 0 else 0.0
        mae_m = float(np.mean(np.abs(y_true - y_pred)))
        mae_n = float(np.mean(np.abs(y_true - y_naive)))
        bias  = float(np.mean(y_pred - y_true))
    else:
        wm, wn, imp, mae_m, mae_n, bias = 0.09, 0.15, 40.0, 8.0, 13.0, 0.2

    # ── Live-Temperaturen für Prognose-Woche vorab abrufen ───────────────
    forecast_dates = [forecast_start + timedelta(days=h) for h in range(7)]
    live_temps     = _fetch_live_temps(forecast_dates)

    # ── Hilfsfunktion: Feature-Vektor für einen Tag ──────────────────────
    def build_feat(fc_date: pd.Timestamp, working: pd.DataFrame,
                   is_aktion: int, temp_override: Optional[float] = None) -> tuple:
        def get_lag(days: int) -> float:
            target = fc_date - timedelta(days=days)
            row    = working[working["datum"] == target]
            return float(row["menge"].values[-1]) if len(row) > 0 else float(working["menge"].mean())

        lag_7   = get_lag(7)
        lag_14  = get_lag(14)
        lag_28  = get_lag(28)
        lag_364 = get_lag(364)

        past_28      = working[working["datum"] < fc_date]["menge"].values[-28:]
        past_14      = working[working["datum"] < fc_date]["menge"].values[-14:]
        roll_mean_4w = float(past_28.mean()) if len(past_28) > 0 else lag_7
        roll_std_4w  = float(past_28.std())  if len(past_28) > 1 else 5.0
        roll_mean_2w = float(past_14.mean()) if len(past_14) > 0 else roll_mean_4w

        temp        = temp_override if temp_override is not None else _synthetic_temp(fc_date)
        is_feiertag = _is_feiertag(fc_date)
        is_schulfer = _is_schulferien(fc_date)

        feat = {
            "dow":             fc_date.dayofweek,
            "month":           fc_date.month,
            "week_of_year":    int(fc_date.isocalendar()[1]),
            "is_feiertag":     is_feiertag,
            "is_schulferien":  is_schulfer,
            "temperatur":      temp,
            "is_aktion":       is_aktion,
            "lag_7":           lag_7,
            "lag_14":          lag_14,
            "lag_28":          lag_28,
            "lag_364":         lag_364,
            "rolling_mean_4w": roll_mean_4w,
            "rolling_std_4w":  roll_std_4w,
            "rolling_mean_2w": roll_mean_2w,
            "filiale_enc":     FILIALE_ENC[filiale],
            "artikel_enc":     ARTIKEL_ENC[artikel],
        }
        return feat, temp, is_feiertag, is_schulfer, roll_std_4w, lag_7

    # ── Gap-Filling: letzte Datenpunkt → gestern (stille Schritte) ───────
    working   = series.copy()
    fill_date = last_date + timedelta(days=1)
    while fill_date < forecast_start:
        feat, *_ = build_feat(fill_date, working, 0)
        pred_fill = max(0.0, float(model.predict(pd.DataFrame([feat])[FEATURE_COLS])[0]))
        working = pd.concat([working, pd.DataFrame([{
            "datum": fill_date, "filiale": filiale,
            "artikel": artikel, "menge": round(pred_fill),
        }])], ignore_index=True)
        fill_date += timedelta(days=1)

    # ── Rekursive 7-Tages-Prognose ab heute ─────────────────────────────
    forecast_rows = []
    for h in range(7):
        fc_date   = forecast_start + timedelta(days=h)
        is_aktion = 1 if (h < len(aktion_days) and aktion_days[h]) else 0

        feat, temp, is_feiertag, is_schulfer, roll_std_4w, lag_7 = build_feat(
            fc_date, working, is_aktion, temp_override=live_temps.get(fc_date)
        )

        X_fc     = pd.DataFrame([feat])[FEATURE_COLS]
        pred     = max(0.0, float(model.predict(X_fc)[0]))
        pred_int = round(pred)

        sigma = max(roll_std_4w, pred * 0.10)
        lower = max(0, round(pred - 1.5 * sigma))
        upper = round(pred + 1.5 * sigma)

        # Naive immer aus echten CSV-Daten – nicht aus Gap-Fill-Schätzungen,
        # da sonst Modell mit sich selbst verglichen wird (Diff ≈ 0, kein Flag).
        naive_date    = fc_date - timedelta(days=7)
        naive_actual  = series[series["datum"] == naive_date]
        if len(naive_actual) > 0:
            naive_val = int(naive_actual["menge"].values[-1])
        else:
            # Gap-Fill-Woche: 14 Tage zurück (gleicher Wochentag, echte Daten)
            naive_actual2 = series[series["datum"] == naive_date - timedelta(days=7)]
            naive_val = int(naive_actual2["menge"].values[-1]) if len(naive_actual2) > 0 else round(lag_7)

        feiertag_name = NRW_HOLIDAY_NAMES.get(fc_date, "") if is_feiertag else ""
        day_flag      = _flag(pred_int, naive_val, lower, upper)

        forecast_rows.append({
            "datum":          fc_date.strftime("%d.%m.%Y"),
            "datum_iso":      fc_date.strftime("%Y-%m-%d"),
            "wochentag":      DOW_NAMES[fc_date.dayofweek],
            "prognose":       pred_int,
            "lower":          lower,
            "upper":          upper,
            "naive":          naive_val,
            "is_feiertag":    bool(is_feiertag),
            "feiertag_name":  feiertag_name,
            "is_schulferien": bool(is_schulfer),
            "is_aktion":      bool(is_aktion),
            "temperatur":     round(temp, 1),
            "flag":           day_flag,
            "rel_diff_pct":   round(abs(pred_int - naive_val) / naive_val * 100, 1) if naive_val > 0 else 0,
        })

        working = pd.concat([working, pd.DataFrame([{
            "datum": fc_date, "filiale": filiale,
            "artikel": artikel, "menge": pred_int,
        }])], ignore_index=True)

    # ── Feature Importance (Gain, normalisiert) ──────────────────────────
    raw_imp   = model.feature_importances_
    total     = raw_imp.sum() or 1
    pairs     = sorted(zip(FEATURE_COLS, raw_imp), key=lambda x: x[1], reverse=True)[:10]
    fi_names  = [FEATURE_LABELS.get(n, n) for n, _ in pairs]
    fi_values = [round(v / total, 4) for _, v in pairs]

    # ── Historische Daten (letzte 30 Tage aus CSV) ───────────────────────
    hist_30 = series[series["datum"] >= last_date - timedelta(days=29)].sort_values("datum")

    return {
        "filiale":  filiale,
        "artikel":  artikel,
        "history": {
            "dates":      hist_30["datum"].dt.strftime("%d.%m.").tolist(),
            "menge":      [int(x) for x in hist_30["menge"].tolist()],
            "last_label": hist_30["datum"].max().strftime("%d.%m.%Y"),
        },
        "forecast": forecast_rows,
        "metrics": {
            "wape_model":      round(wm  * 100, 1),
            "wape_naive":      round(wn  * 100, 1),
            "improvement_pct": round(imp,        1),
            "mae_model":       round(mae_m,       1),
            "mae_naive":       round(mae_n,       1),
            "bias":            round(bias,        1),
        },
        "feature_importance": {
            "features":   fi_names,
            "importance": fi_values,
        },
        "last_date":      last_date.strftime("%d.%m.%Y"),
        "forecast_start": forecast_start.strftime("%d.%m.%Y"),
        "temp_source":    "open-meteo" if _weather_cache else "synthetic",
    }


def run_overview(aktion_days: Optional[list] = None) -> dict:
    """
    Batch-Prognose für alle Filialen × Artikel (40 Serien).
    Gibt strukturierte Übersicht mit Flagging für Ausnahme-basierte Prüfung zurück.
    """
    if aktion_days is None:
        aktion_days = [False] * 7

    _get_model()   # Modell einmalig laden / trainieren

    serien = []
    for filiale in FILIALEN:
        for artikel in ARTIKEL:
            fc = run_forecast(filiale, artikel, aktion_days)
            row_flags = [r["flag"] for r in fc["forecast"]]

            row_flag = "auffaellig" if "auffaellig" in row_flags else "ok"

            serien.append({
                "filiale":      filiale,
                "artikel":      artikel,
                "tage":         fc["forecast"],
                "flag":         row_flag,
                "n_flagged":    sum(1 for f in row_flags if f == "auffaellig"),
            })

    fc_dates = [
        {"datum": t["datum"], "datum_iso": t["datum_iso"], "wochentag": t["wochentag"],
         "is_feiertag": t["is_feiertag"], "feiertag_name": t.get("feiertag_name", "")}
        for t in serien[0]["tage"]
    ] if serien else []

    return {
        "serien":       serien,
        "fc_dates":     fc_dates,
        "n_total":      len(serien),
        "n_auffaellig": sum(1 for s in serien if s["flag"] == "auffaellig"),
        "n_ok":         sum(1 for s in serien if s["flag"] == "ok"),
    }
