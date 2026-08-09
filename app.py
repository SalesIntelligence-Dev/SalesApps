"""
Aurora Web App – Flask Backend
Azure-ready: Umgebungsvariablen steuern Konfiguration.
"""
import os
import uuid
import threading
import logging
import io
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    send_file, abort,
)

from aurora_inference import AuroraInference, AuroraConfig
from report_generator import generate_report
from chart_generator import make_forecast_chart, make_daily_bar_chart

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("aurora-app")

# ── App ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB Upload-Limit

DATASETS_DIR = Path(__file__).parent / "datasets"
ALLOWED_EXT  = {".csv", ".xlsx", ".xls", ".txt"}

# ── Aurora Singleton ─────────────────────────────────────────────────────
# AuroraConfig.from_env() liest AURORA_MODEL_PATH (Standard: aurora_checkpoint.pt)
_aurora      = AuroraInference(AuroraConfig.from_env())
_aurora_lock = threading.Lock()


def get_aurora() -> AuroraInference:
    if not _aurora.ready:
        with _aurora_lock:
            if not _aurora.ready:
                _aurora.load()
    return _aurora


# ── Job-Store (in-memory, Einzel-Instanz) ────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

FREQ_MAP = {
    "hourly":  "h",
    "daily":   "D",
    "weekly":  "W",
}

HORIZON_MAP = {
    "1w":  {"h": 168,  "D": 7,   "W": 1},
    "2w":  {"h": 336,  "D": 14,  "W": 2},
    "4w":  {"h": 672,  "D": 28,  "W": 4},
    "3m":  {"h": 2160, "D": 90,  "W": 13},
}


# ── Hilfsfunktionen ───────────────────────────────────────────────────────

def _load_dataframe(file_obj, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_obj)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(file_obj)
    if ext == ".txt":
        # Versuche Tab, Semikolon, Komma
        content = file_obj.read().decode("utf-8", errors="replace")
        for sep in ["\t", ";", ","]:
            try:
                df = pd.read_csv(io.StringIO(content), sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
        return pd.read_csv(io.StringIO(content))
    raise ValueError(f"Nicht unterstütztes Dateiformat: {ext}")


def _detect_time_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if any(x in col.lower() for x in ["time", "date", "datum", "ts", "timestamp"]):
            try:
                pd.to_datetime(df[col].iloc[:5])
                return col
            except Exception:
                pass
    return None


def _detect_freq(df: pd.DataFrame, time_col: str | None) -> str:
    if time_col:
        try:
            ts = pd.to_datetime(df[time_col]).sort_values()
            delta = (ts.iloc[1] - ts.iloc[0]).total_seconds()
            if delta <= 3600:
                return "h"
            if delta <= 86400:
                return "D"
            return "W"
        except Exception:
            pass
    return "h"


def _run_forecast_job(job_id: str, df: pd.DataFrame, params: dict):
    """Läuft in eigenem Thread."""
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"] = "Modell wird vorbereitet …"

        aurora = get_aurora()
        freq   = params.get("freq", "h")

        with _jobs_lock:
            _jobs[job_id]["progress"] = "Daten werden normalisiert …"

        aurora_input = aurora.prepare(
            df         = df,
            target_col = params["target_col"],
            time_col   = params.get("time_col"),
            freq       = freq,
        )

        with _jobs_lock:
            _jobs[job_id]["progress"] = "Inference läuft … (kann 1–5 Min dauern)"

        result = aurora.forecast(
            aurora_input = aurora_input,
            horizon      = params["horizon"],
            num_samples  = params.get("num_samples", 50),
            text_prompt  = params.get("text_prompt"),
        )

        report = generate_report(
            result            = result,
            scenario_question = params.get("scenario_question", ""),
            scenario_type     = params.get("scenario_type", "allgemein"),
            freq              = freq,
            safety_buffer     = params.get("safety_buffer", 0.15),
        )

        chart1 = make_forecast_chart(result)
        chart2 = make_daily_bar_chart(result)

        with _jobs_lock:
            _jobs[job_id].update({
                "status":   "done",
                "progress": "Fertig",
                "report":   report,
                "chart1":   chart1,
                "chart2":   chart2,
                "meta": {
                    "target_col": result.target_column,
                    "horizon":    result.horizon,
                    "samples":    result.num_samples,
                    "freq":       freq,
                },
            })

    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc)
        logger.error(traceback.format_exc())
        with _jobs_lock:
            _jobs[job_id].update({
                "status":  "error",
                "error":   str(exc),
            })


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    import os
    api_key    = os.environ.get("RUNPOD_API_KEY", "")
    endpoint   = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    return jsonify({
        "status":              "ok",
        "runpod_ready":        _aurora.ready,
        "runpod_api_key_set":  bool(api_key),
        "runpod_endpoint_set": bool(endpoint),
        "runpod_endpoint_id":  endpoint or "nicht gesetzt",
        "device":              _aurora.device,
    })


@app.route("/api/datasets")
def list_datasets():
    result = []
    for csv_path in sorted(DATASETS_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, nrows=3)
            cols = list(df.columns)
        except Exception:
            cols = []
        meta_path = csv_path.with_suffix(".meta.txt")
        description = meta_path.read_text(encoding="utf-8").strip() if meta_path.exists() else csv_path.stem
        result.append({
            "id":          csv_path.stem,
            "name":        csv_path.name,
            "description": description,
            "columns":     cols,
        })
    return jsonify(result)


@app.route("/api/datasets/<dataset_id>/preview")
def preview_dataset(dataset_id: str):
    csv_path = DATASETS_DIR / f"{dataset_id}.csv"
    if not csv_path.exists():
        abort(404)
    df = pd.read_csv(csv_path, nrows=10)
    return jsonify({
        "columns": list(df.columns),
        "rows":    df.to_dict(orient="records"),
        "total_rows": sum(1 for _ in open(csv_path)) - 1,
    })


@app.route("/api/forecast", methods=["POST"])
def start_forecast():
    """
    Erwartet multipart/form-data:
      file (optional)       – hochgeladene CSV/XLSX/TXT
      dataset_id (optional) – ID eines Default-Datensatzes
      target_col            – Spalte für Zeitreihe
      time_col (optional)   – Zeitstempel-Spalte
      freq                  – hourly / daily / weekly
      horizon               – 1w / 2w / 4w / 3m
      num_samples           – 20–100
      text_prompt (optional)– Aurora Domänen-Kontext (Englisch)
      scenario_question     – Geschäftliche Frage (Deutsch)
      scenario_type         – allgemein / einkauf / personal / …
      safety_buffer         – 0.0–0.5
    """
    # ── Daten laden ──────────────────────────────────────────────────────
    df = None

    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        if Path(f.filename).suffix.lower() not in ALLOWED_EXT:
            return jsonify({"error": "Nur CSV, XLSX, XLS und TXT werden unterstützt."}), 400
        df = _load_dataframe(f.stream, f.filename)

    elif request.form.get("dataset_id"):
        dataset_id = request.form["dataset_id"]
        csv_path   = DATASETS_DIR / f"{dataset_id}.csv"
        if not csv_path.exists():
            return jsonify({"error": f"Datensatz nicht gefunden: {dataset_id}"}), 404
        df = pd.read_csv(csv_path)

    else:
        return jsonify({"error": "Bitte eine Datei hochladen oder einen Default-Datensatz wählen."}), 400

    # ── Parameter ─────────────────────────────────────────────────────────
    freq_key    = request.form.get("freq", "hourly")
    freq        = FREQ_MAP.get(freq_key, "h")
    horizon_key = request.form.get("horizon", "1w")
    horizon     = HORIZON_MAP.get(horizon_key, {}).get(freq, 168)
    num_samples = int(request.form.get("num_samples", 50))
    num_samples = max(10, min(100, num_samples))

    time_col   = request.form.get("time_col") or _detect_time_col(df)
    target_col = request.form.get("target_col")
    if not target_col:
        num_cols = [c for c in df.columns if c != time_col]
        num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            return jsonify({"error": "Keine numerische Spalte gefunden."}), 400
        target_col = num_cols[0]

    params = {
        "target_col":       target_col,
        "time_col":         time_col,
        "freq":             freq,
        "horizon":          horizon,
        "num_samples":      num_samples,
        "text_prompt":      request.form.get("text_prompt", "").strip() or None,
        "scenario_question": request.form.get("scenario_question", "Prognose").strip(),
        "scenario_type":    request.form.get("scenario_type", "allgemein"),
        "safety_buffer":    float(request.form.get("safety_buffer", 0.15)),
    }

    # ── Job starten ───────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "progress": "Warteschlange …"}

    t = threading.Thread(
        target=_run_forecast_job,
        args=(job_id, df, params),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/forecast/<job_id>")
def get_forecast(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        abort(404)
    return jsonify(job)


@app.route("/api/forecast/<job_id>/download")
def download_forecast(job_id: str):
    """Gibt die tägliche Prognose als Excel-Datei zurück."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job.get("status") != "done":
        abort(404)

    daily = job["report"].get("daily_table", [])
    if not daily:
        abort(404)

    df_out = pd.DataFrame(daily)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Aurora Prognose", index=False)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"aurora_prognose_{job_id}.xlsx",
    )


# ── Datasets generieren beim Start ───────────────────────────────────────

def _ensure_datasets():
    DATASETS_DIR.mkdir(exist_ok=True)
    gen_script = Path(__file__).parent / "datasets" / "generate_defaults.py"
    if gen_script.exists():
        existing = list(DATASETS_DIR.glob("*.csv"))
        if len(existing) < 3:
            logger.info("Generiere Default-Datensätze …")
            import subprocess, sys
            subprocess.run([sys.executable, str(gen_script)], check=False)


# ── Entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _ensure_datasets()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    # Aurora vorladen im Hintergrund (optional)
    if os.environ.get("PRELOAD_MODEL", "false").lower() == "true":
        threading.Thread(target=get_aurora, daemon=True).start()

    app.run(host="0.0.0.0", port=port, debug=debug)
