"""
Online Retail II – Download und Aufbereitung (einmalig, offline).

  python -m graph_ml.prepare_data            # Download (falls nötig) + Parquet + Artefakte
  python -m graph_ml.prepare_data --skip-download

Ausgabe: data/online_retail_ii.npz (kompaktes NumPy-Format, lädt ohne pyarrow)
und data/online_retail_ii.meta.json. Die Rohdaten bleiben vollständig erhalten (alle Zeilen inkl. fehlender
Kundennummern, Stornos, Dubletten). Bereinigung ist Aufgabe der Fix-Playbooks.

Quelle: UCI Machine Learning Repository, „Online Retail II“, CC BY 4.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR      = Path(__file__).parent / "data"
RAW_DIR       = DATA_DIR / "raw"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

SOURCE_URL   = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_PATH     = RAW_DIR / "online_retail_ii.zip"
XLSX_NAME    = "online_retail_II.xlsx"
NPZ_PATH     = DATA_DIR / "online_retail_ii.npz"
META_PATH    = DATA_DIR / "online_retail_ii.meta.json"

COLUMN_MAP = {
    "Invoice":     "invoice",
    "StockCode":   "stock_code",
    "Description": "description",
    "Quantity":    "quantity",
    "InvoiceDate": "invoice_date",
    "Price":       "price_gbp",
    "Customer ID": "customer_id",
    "Country":     "country",
}


def _log(msg: str) -> None:
    print(f"[prepare_data] {msg}", flush=True)


def download(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        _log(f"Zip vorhanden ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
        return ZIP_PATH
    _log(f"Lade {SOURCE_URL} …")
    with requests.get(SOURCE_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(ZIP_PATH, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    _log(f"Download fertig ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
    return ZIP_PATH


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_excel(zip_path: Path) -> pd.DataFrame:
    xlsx_path = RAW_DIR / XLSX_NAME
    if not xlsx_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(XLSX_NAME, RAW_DIR)
    try:
        import python_calamine  # noqa: F401  – deutlich schneller, optional
        engine = "calamine"
    except ImportError:
        engine = "openpyxl"
    _log(f"Lese beide Blätter mit {engine} (openpyxl braucht einige Minuten) …")
    t0 = time.time()
    sheets = pd.read_excel(
        xlsx_path, sheet_name=None, engine=engine,
        dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
    )
    _log(f"Blätter: {list(sheets)} – {time.time() - t0:.0f} s")
    return pd.concat(sheets.values(), ignore_index=True)


def to_compact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    for col in ("invoice", "stock_code", "description", "country"):
        s = df[col].astype("string").str.strip()
        df[col] = s.astype("category")
    df["quantity"]     = df["quantity"].astype("int32")
    df["price_gbp"]    = df["price_gbp"].astype("float32")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"]  = df["customer_id"].round().astype("Int32")
    return df


NON_ARTICLE_CODES = {
    "POST", "DOT", "M", "D", "C2", "C3", "BANK CHARGES", "AMAZONFEE", "CRUK",
    "S", "B", "PADS", "ADJUST", "ADJUST2", "TEST001", "TEST002", "SP1002",
}


def is_non_article(codes: pd.Series) -> pd.Series:
    s = codes.astype("string").str.upper()
    return s.isin(NON_ARTICLE_CODES) | s.str.startswith("GIFT_") | s.str.startswith("TEST")


def raw_facts(df: pd.DataFrame) -> dict:
    """Kennzahlen der unveränderten Rohdaten (für Konnektor-Karte und Checks)."""
    inv = df["invoice"].astype("string")
    storno_mask = inv.str.startswith("C")
    adjust_mask = inv.str.startswith("A")
    no_cust     = df["customer_id"].isna()
    return {
        "rows":                 int(len(df)),
        "columns":              int(df.shape[1]),
        "date_min":             df["invoice_date"].min().isoformat(),
        "date_max":             df["invoice_date"].max().isoformat(),
        "months":               int(round((df["invoice_date"].max() - df["invoice_date"].min()).days / 30.44)),
        "customers":            int(df["customer_id"].nunique()),
        "stock_codes":          int(df["stock_code"].nunique()),
        "descriptions":         int(df["description"].nunique()),
        "countries":            int(df["country"].nunique()),
        "invoices":             int(df["invoice"].nunique()),
        "storno_invoices":      int(df.loc[storno_mask, "invoice"].nunique()),
        "storno_rows":          int(storno_mask.sum()),
        "adjustment_rows":      int(adjust_mask.sum()),
        "rows_without_customer": int(no_cust.sum()),
        "share_without_customer": float(no_cust.mean()),
        "revenue_gbp_without_customer": float((df["quantity"] * df["price_gbp"])[no_cust & ~storno_mask].sum()),
        "revenue_gbp_total":    float((df["quantity"] * df["price_gbp"])[~storno_mask].sum()),
        "duplicate_rows":       int(df.duplicated().sum()),
        "price_zero_rows":      int((df["price_gbp"] == 0).sum()),
        "negative_qty_rows":    int((df["quantity"] < 0).sum()),
        "negative_qty_without_storno": int(((df["quantity"] < 0) & ~storno_mask).sum()),
        "non_article_rows":     int(is_non_article(df["stock_code"]).sum()),
        "missing_description_rows": int(df["description"].isna().sum()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = ZIP_PATH if args.skip_download else download(force=args.force_download)
    raw = read_excel(zip_path)
    df  = to_compact(raw)
    del raw

    from graph_ml.data_loader import save_online_retail
    save_online_retail(df, NPZ_PATH)
    _log(f"NPZ: {NPZ_PATH.name} – {NPZ_PATH.stat().st_size / 1e6:.1f} MB, {len(df):,} Zeilen")

    meta = {
        "id":          "online_retail_ii",
        "name":        "Online Retail II",
        "source":      "UCI Machine Learning Repository – Online Retail II",
        "source_url":  SOURCE_URL,
        "license":     "CC BY 4.0",
        "currency":    "GBP",
        "zip_sha256":  _sha256(zip_path),
        "prepared_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "facts":       raw_facts(df),
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Meta: {json.dumps(meta['facts'], ensure_ascii=False)}")

    # Graph-Mapping als Artefakt vorberechnen → Render antwortet sofort
    from graph_ml.data_loader import get_registry
    reg = get_registry()
    reg.reload_builtin()
    for ds_id in ("online_retail_ii", "adventureworks"):
        if reg.has(ds_id):
            reg.mapping(ds_id, refresh=True)
            if reg.get(ds_id).kind == "data":
                reg.overview(ds_id)
            _log(f"Artefakte geschrieben: {ds_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
