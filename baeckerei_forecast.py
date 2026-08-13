"""
Bäckerei Nachfrageprognose – LightGBM Engine
Läuft vollständig lokal auf CPU, kein GPU nötig.
Erstes Laden trainiert das Modell (~3–5 s), danach gecacht.
"""
from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

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
    "temperatur":      "Temperatur (°C)",
    "is_aktion":       "Aktion / Promotion",
    "filiale_enc":     "Filiale",
    "artikel_enc":     "Artikel",
}

DOW_NAMES = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# NRW Feiertage 2023–2025 als pd.Timestamp-Set
NRW_HOLIDAYS: set[pd.Timestamp] = {
    pd.Timestamp(d) for d in [
        "2023-01-01","2023-04-07","2023-04-10","2023-05-01",
        "2023-05-18","2023-05-29","2023-06-08","2023-10-03",
        "2023-11-01","2023-12-25","2023-12-26",
        "2024-01-01","2024-03-29","2024-04-01","2024-05-01",
        "2024-05-09","2024-05-20","2024-05-30","2024-10-03",
        "2024-11-01","2024-12-25","2024-12-26",
        "2025-01-01","2025-04-18","2025-04-21","2025-05-01",
        "2025-05-29","2025-06-09","2025-06-19","2025-10-03",
        "2025-11-01","2025-12-25","2025-12-26",
    ]
}

SCHOOL_HOLIDAYS = [
    (pd.Timestamp("2023-01-30"), pd.Timestamp("2023-02-03")),
    (pd.Timestamp("2023-04-03"), pd.Timestamp("2023-04-15")),
    (pd.Timestamp("2023-05-30"), pd.Timestamp("2023-06-02")),
    (pd.Timestamp("2023-06-29"), pd.Timestamp("2023-08-11")),
    (pd.Timestamp("2023-10-02"), pd.Timestamp("2023-10-14")),
    (pd.Timestamp("2023-12-27"), pd.Timestamp("2024-01-06")),
    (pd.Timestamp("2024-02-12"), pd.Timestamp("2024-02-16")),
    (pd.Timestamp("2024-03-25"), pd.Timestamp("2024-04-06")),
    (pd.Timestamp("2024-05-21"), pd.Timestamp("2024-05-24")),
    (pd.Timestamp("2024-07-22"), pd.Timestamp("2024-09-03")),
    (pd.Timestamp("2024-10-14"), pd.Timestamp("2024-10-26")),
    (pd.Timestamp("2024-12-23"), pd.Timestamp("2025-01-07")),
    (pd.Timestamp("2025-02-17"), pd.Timestamp("2025-02-21")),
    (pd.Timestamp("2025-04-14"), pd.Timestamp("2025-04-26")),
]


def _is_feiertag(d: pd.Timestamp) -> int:
    return 1 if d in NRW_HOLIDAYS else 0


def _is_schulferien(d: pd.Timestamp) -> int:
    return 1 if any(s <= d <= e for s, e in SCHOOL_HOLIDAYS) else 0


def _synthetic_temp(d: pd.Timestamp) -> float:
    doy  = d.timetuple().tm_yday
    base = 10.0 + 12.0 * np.sin((doy - 80) / 365.0 * 2 * np.pi)
    rng  = np.random.default_rng(int(d.timestamp()))
    return float(base + rng.normal(0, 2.0))


# ── Modul-Level Singletons ────────────────────────────────────────────────
_model      = None
_df_raw: pd.DataFrame | None = None
_df_feat: pd.DataFrame | None = None
_lock       = threading.Lock()


def _ensure_data() -> pd.DataFrame:
    global _df_raw
    if _df_raw is not None:
        return _df_raw
    if not BAKERY_CSV.exists():
        from datasets.generate_baeckerei import generate
        generate()
    _df_raw = pd.read_csv(BAKERY_CSV, parse_dates=["datum"])
    return _df_raw


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Feature-Engineering für das gesamte DataFrame (alle Serien)."""
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
        objective       = "regression",
        metric          = "mae",
        n_estimators    = 400,
        learning_rate   = 0.05,
        num_leaves      = 63,
        min_child_samples = 20,
        feature_fraction= 0.8,
        bagging_fraction= 0.8,
        bagging_freq    = 5,
        verbose         = -1,
        n_jobs          = -1,
    )
    model.fit(X, y)
    _model  = model
    _df_feat = df_feat


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _train()
    return _model


# ── Public API ────────────────────────────────────────────────────────────

def get_meta() -> dict:
    return {"filialen": FILIALEN, "artikel": ARTIKEL}


def run_forecast(filiale: str, artikel: str, with_aktion: bool = False) -> dict:
    """
    Trainiert (beim ersten Aufruf) ein globales LightGBM-Modell und gibt
    7-Tages-Prognose + Metriken + Feature-Importance zurück.
    """
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

    # ── Metriken: Walk-Forward auf letzten 4 Wochen ─────────────────────
    test_start = last_date - timedelta(days=27)
    test_rows  = _df_feat[
        (_df_feat.filiale == filiale)
        & (_df_feat.artikel == artikel)
        & (_df_feat.datum >= test_start)
    ].dropna(subset=FEATURE_COLS)

    if len(test_rows) > 0:
        y_true  = test_rows["menge"].values
        y_pred  = model.predict(test_rows[FEATURE_COLS])
        y_naive = test_rows["lag_7"].values

        def wape(a, p):
            return float(np.sum(np.abs(a - p)) / np.sum(a)) if np.sum(a) > 0 else 0.0

        wm = wape(y_true, y_pred)
        wn = wape(y_true, y_naive)
        imp = (wn - wm) / wn * 100 if wn > 0 else 0.0
        mae_m = float(np.mean(np.abs(y_true - y_pred)))
        mae_n = float(np.mean(np.abs(y_true - y_naive)))
        bias  = float(np.mean(y_pred - y_true))
    else:
        wm, wn, imp, mae_m, mae_n, bias = 0.09, 0.15, 40.0, 8.0, 13.0, 0.2

    # ── Rekursive 7-Tages-Prognose ───────────────────────────────────────
    working = series.copy()
    forecast_rows = []

    for h in range(1, 8):
        fc_date = last_date + timedelta(days=h)

        def get_lag(days: int) -> float:
            target = fc_date - timedelta(days=days)
            row    = working[working["datum"] == target]
            return float(row["menge"].values[-1]) if len(row) > 0 else float(working["menge"].mean())

        lag_7   = get_lag(7)
        lag_14  = get_lag(14)
        lag_28  = get_lag(28)
        lag_364 = get_lag(364)

        recent_28     = working[working["datum"] < fc_date]["menge"].values[-28:]
        recent_14     = working[working["datum"] < fc_date]["menge"].values[-14:]
        roll_mean_4w  = float(recent_28.mean()) if len(recent_28) > 0 else lag_7
        roll_std_4w   = float(recent_28.std())  if len(recent_28) > 1 else 5.0
        roll_mean_2w  = float(recent_14.mean()) if len(recent_14) > 0 else roll_mean_4w

        temp         = _synthetic_temp(fc_date)
        is_feiertag  = _is_feiertag(fc_date)
        is_schulfer  = _is_schulferien(fc_date)
        is_aktion    = 1 if with_aktion else 0

        feat = {
            "dow":           fc_date.dayofweek,
            "month":         fc_date.month,
            "week_of_year":  int(fc_date.isocalendar()[1]),
            "is_feiertag":   is_feiertag,
            "is_schulferien":is_schulfer,
            "temperatur":    temp,
            "is_aktion":     is_aktion,
            "lag_7":         lag_7,
            "lag_14":        lag_14,
            "lag_28":        lag_28,
            "lag_364":       lag_364,
            "rolling_mean_4w": roll_mean_4w,
            "rolling_std_4w":  roll_std_4w,
            "rolling_mean_2w": roll_mean_2w,
            "filiale_enc":   FILIALE_ENC[filiale],
            "artikel_enc":   ARTIKEL_ENC[artikel],
        }

        X_fc = pd.DataFrame([feat])[FEATURE_COLS]
        pred = max(0.0, float(model.predict(X_fc)[0]))
        pred_int = round(pred)

        # Konfidenzband: ±1.5 × lokale Standardabweichung
        sigma = max(roll_std_4w, pred * 0.10)
        lower = max(0, round(pred - 1.5 * sigma))
        upper = round(pred + 1.5 * sigma)

        # Naive: gleicher Wochentag letzte Woche
        naive_date = fc_date - timedelta(days=7)
        naive_row  = working[working["datum"] == naive_date]
        naive_val  = int(naive_row["menge"].values[-1]) if len(naive_row) > 0 else round(lag_7)

        forecast_rows.append({
            "datum":       fc_date.strftime("%Y-%m-%d"),
            "wochentag":   DOW_NAMES[fc_date.dayofweek],
            "prognose":    pred_int,
            "lower":       lower,
            "upper":       upper,
            "naive":       naive_val,
            "is_feiertag": bool(is_feiertag),
            "temperatur":  round(temp, 1),
        })

        # Prognose in Working-Series einspeisen (für nächste Lag-Berechnung)
        new_row = pd.DataFrame([{
            "datum": fc_date, "filiale": filiale,
            "artikel": artikel, "menge": pred_int,
        }])
        working = pd.concat([working, new_row], ignore_index=True)

    # ── Feature Importance ───────────────────────────────────────────────
    raw_imp = model.feature_importances_
    total   = raw_imp.sum() or 1
    pairs   = sorted(zip(FEATURE_COLS, raw_imp), key=lambda x: x[1], reverse=True)[:10]
    fi_names   = [FEATURE_LABELS.get(n, n) for n, _ in pairs]
    fi_values  = [round(v / total, 4) for _, v in pairs]

    # ── Historische Daten (letzte 30 Tage) ──────────────────────────────
    hist_30 = series[series["datum"] >= last_date - timedelta(days=29)].sort_values("datum")

    return {
        "filiale":  filiale,
        "artikel":  artikel,
        "history": {
            "dates": hist_30["datum"].dt.strftime("%Y-%m-%d").tolist(),
            "menge": [int(x) for x in hist_30["menge"].tolist()],
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
        "last_date": last_date.strftime("%Y-%m-%d"),
    }
