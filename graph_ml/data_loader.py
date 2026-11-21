"""
Konnektoren (Modul 1)
=====================
Datensatz-Registry für /graph_ml:
  - online_retail_ii : echte Transaktionen (UCI, CC BY 4.0) → relationales Schema
                       customers · invoices · invoice_lines · products · countries
  - adventureworks   : nur Schema (Microsoft-DDL) – Demo für Stufe 1
  - upload_<id>      : eigene CSV/XLSX/Parquet-Tabellen oder Schema-Import

Speicher (Render ≈ 512 MB): Daten werden erst bei Bedarf geladen und dann
gecacht. Karten und Graph-Mapping kommen aus Meta-Datei bzw. Artefakt.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .graph_mapping import ARTIFACT_VERSION, build_mapping, build_schema_mapping

PROFILE_KIND = "profile"

log = logging.getLogger("graph_ml.data")

DATA_DIR      = Path(__file__).parent / "data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ORII_NPZ      = DATA_DIR / "online_retail_ii.npz"
ORII_META     = DATA_DIR / "online_retail_ii.meta.json"
AW_SCHEMA     = DATA_DIR / "adventureworks_schema.json"

RAW_TABLE        = "rohdaten"
MAX_UPLOADS      = 5
MAX_UPLOAD_ROWS  = 2_000_000
TABLE_LABELS_ORII = {
    "customers": "Kunden", "invoices": "Rechnungen", "invoice_lines": "Positionen",
    "products": "Artikel", "countries": "Länder",
}


@dataclass
class Dataset:
    id: str
    name: str
    kind: str                       # "data" | "schema"
    source: str
    license: str
    description: str
    builtin: bool
    version: str
    facts: dict = field(default_factory=dict)
    schema: dict | None = None      # kind == "schema"
    profile: dict | None = None     # kind == "profile" (hochgeladenes Kennzahlen-Profil, Stufe 2)
    tables: dict | None = None      # kind == "data" (lazy für builtin)
    labels: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    files: list = field(default_factory=list)

    def card(self) -> dict:
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "source": self.source,
            "license": self.license, "description": self.description, "builtin": self.builtin,
            "facts": self.facts, "files": self.files,
        }


# ── Online Retail II → relationales Schema ────────────────────────────────────

def cat_startswith(s: pd.Series, prefix: str) -> np.ndarray:
    """str.startswith auf Kategorien statt auf 1 Mio. Zeilen-Strings (spart Speicher)."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        flags = pd.Series(s.cat.categories).astype(str).str.startswith(prefix).to_numpy()
        codes = s.cat.codes.to_numpy()
        return np.where(codes >= 0, flags[np.clip(codes, 0, None)], False)
    return s.astype("string").str.startswith(prefix).fillna(False).to_numpy(dtype=bool)


CAT_COLS = ("invoice", "stock_code", "description", "country")


def save_online_retail(df: pd.DataFrame, path: Path) -> None:
    """Kompaktes NumPy-Format: Kategorien als Codes + Wörterbuch, Datum in Sekunden.
    Lädt ohne pyarrow und mit etwa halbem Speicherbedarf gegenüber Parquet."""
    arrs = {}
    for c in df.columns:
        s = df[c]
        if c in CAT_COLS:
            arrs[f"{c}__codes"] = s.cat.codes.to_numpy()
            arrs[f"{c}__cats"] = np.asarray(s.cat.categories.astype(str), dtype="U")
        elif c == "customer_id":
            arrs[c] = s.fillna(-1).astype("int32").to_numpy()
        elif c == "invoice_date":
            arrs[c] = s.to_numpy().astype("datetime64[s]")
        else:
            arrs[c] = s.to_numpy()
    np.savez_compressed(path, **arrs)


def load_online_retail(path: Path = None) -> pd.DataFrame:
    z = np.load(path or ORII_NPZ)
    cols = {}
    for c in ("invoice", "stock_code", "description", "quantity", "invoice_date", "price_gbp",
              "customer_id", "country"):
        if c in CAT_COLS:
            cols[c] = pd.Categorical.from_codes(z[f"{c}__codes"], categories=z[f"{c}__cats"].astype(object))
        elif c == "customer_id":
            raw = z[c]
            cols[c] = pd.arrays.IntegerArray(np.where(raw < 0, 0, raw).astype("int32"), raw < 0)
        elif c == "invoice_date":
            cols[c] = z[c].astype("datetime64[ns]")
        else:
            cols[c] = z[c]
    return pd.DataFrame(cols)


def _codes(s: pd.Series) -> np.ndarray:
    return s.cat.codes.to_numpy().astype(np.int64)


def _mode_codes(key: np.ndarray, val: np.ndarray, n_val: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Häufigster Wert je Schlüssel über Integer-Codes (val < 0 = leer).
    Gibt (Schlüssel, Modus-Wert, Anzahl verschiedener Werte) zurück."""
    m = val >= 0
    pair = key[m] * (n_val + 1) + val[m]
    uniq, counts = np.unique(pair, return_counts=True)
    k, v = uniq // (n_val + 1), uniq % (n_val + 1)
    order = np.lexsort((-counts, k))
    k, v = k[order], v[order]
    first = np.r_[True, k[1:] != k[:-1]]
    n_distinct = np.bincount(k, minlength=int(key.max()) + 1)[k[first]]
    return k[first], v[first], n_distinct


def build_online_retail_tables(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Flacher Export → normalisierte Tabellen wie in einem ERP. Keine Bereinigung.
    Arbeitet auf Kategorien-Codes (speicherschonend für Render)."""
    inv_codes = _codes(df["invoice"])
    cust = df["customer_id"].to_numpy(dtype="float64", na_value=np.nan)
    _, first_idx = np.unique(inv_codes, return_index=True)
    beleg = np.select([cat_startswith(df["invoice"], "C"), cat_startswith(df["invoice"], "A")],
                      ["Storno", "Anpassung"], "Rechnung")
    sub = df.iloc[first_idx]
    invoices = pd.DataFrame({
        "invoice":      sub["invoice"].to_numpy(),
        "customer_id":  sub["customer_id"].array,
        "invoice_date": sub["invoice_date"].to_numpy(),
        "country":      sub["country"].to_numpy(),
        "beleg_typ":    pd.Categorical(beleg[first_idx]),
    })
    invoices["invoice"] = invoices["invoice"].astype(df["invoice"].dtype)
    invoices["country"] = invoices["country"].astype(df["country"].dtype)
    if "orig_invoice" in df.columns:          # PB-02: Storno → Originalrechnung (heuristisch zugeordnet)
        oi = df.loc[df["orig_invoice"].notna(), ["invoice", "orig_invoice"]].drop_duplicates("invoice")
        invoices["original_invoice"] = invoices["invoice"].astype(str).map(
            oi.set_index(oi["invoice"].astype(str))["orig_invoice"]).astype("category")

    # Konsistenz: Rechnungen mit mehreren Kunden / Kunden mit mehreren Ländern
    ok = ~np.isnan(cust)
    ic = np.unique(np.c_[inv_codes[ok], cust[ok].astype(np.int64)], axis=0)
    multi_cust = int((np.bincount(ic[:, 0].astype(np.int64)) > 1).sum())
    cc = np.unique(np.c_[cust[ok].astype(np.int64), _codes(df["country"])[ok]], axis=0)
    multi_country = int((np.unique(cc[:, 0], return_counts=True)[1] > 1).sum())

    price_col = "price_eur" if "price_eur" in df.columns else "price_gbp"     # PB-08: Währung
    lines = pd.DataFrame({
        "line_id":    np.arange(1, len(df) + 1, dtype=np.int32),
        "invoice":    df["invoice"].array,
        "stock_code": df["stock_code"].array,
        "quantity":   df["quantity"].to_numpy(),
        price_col:    df[price_col].to_numpy(),
    })

    stock_cats = df["stock_code"].cat.categories
    desc_cats  = df["description"].cat.categories
    k, v, n_desc = _mode_codes(_codes(df["stock_code"]), _codes(df["description"]), len(desc_cats))
    desc_codes = np.full(len(stock_cats), -1, dtype=np.int64)
    desc_codes[k] = v
    n_desc_all = np.zeros(len(stock_cats), dtype=np.int16)
    n_desc_all[k] = n_desc
    used = np.zeros(len(stock_cats), dtype=bool)
    used[np.unique(_codes(df["stock_code"]))] = True
    products = pd.DataFrame({
        "stock_code":       pd.Categorical.from_codes(np.flatnonzero(used), categories=stock_cats),
        "description":      pd.Categorical.from_codes(desc_codes[used], categories=desc_cats),
        "n_beschreibungen": n_desc_all[used],
    })
    if "pb05_group" in df.columns:            # PB-05: Artikelgruppe aus Freitext
        gcat = df["pb05_group"].astype("category")
        gk, gv, _ = _mode_codes(_codes(df["stock_code"]), _codes(gcat), len(gcat.cat.categories))
        gc = np.full(len(stock_cats), -1, dtype=np.int64)
        gc[gk] = gv
        products["product_group"] = pd.Categorical.from_codes(gc[used], categories=gcat.cat.categories)

    cust_ids = np.unique(cust[ok]).astype(np.int64)
    ck, cv, _ = _mode_codes(np.searchsorted(cust_ids, cust[ok].astype(np.int64)),
                            _codes(df["country"])[ok], len(df["country"].cat.categories))
    country_codes = np.full(len(cust_ids), -1, dtype=np.int64)
    country_codes[ck] = cv
    customers = pd.DataFrame({
        "customer_id": pd.array(cust_ids, dtype="Int32"),
        "country":     pd.Categorical.from_codes(country_codes, categories=df["country"].cat.categories),
    })
    countries = pd.DataFrame({"country": pd.Categorical(df["country"].cat.categories)})

    tables = {"customers": customers, "invoices": invoices, "invoice_lines": lines,
              "products": products, "countries": countries}
    quality = {
        "invoices_multi_customer": multi_cust,
        "products_multi_description": int((products["n_beschreibungen"] > 1).sum()),
        "customers_multi_country": multi_country,
    }
    return tables, quality


def product_descriptions(df: pd.DataFrame) -> pd.Series:
    """Artikelnummer → häufigste Beschreibung (ohne die Tabellen aufzubauen)."""
    stock_cats, desc_cats = df["stock_code"].cat.categories, df["description"].cat.categories
    k, v, _ = _mode_codes(_codes(df["stock_code"]), _codes(df["description"]), len(desc_cats))
    return pd.Series(np.asarray(desc_cats)[v], index=np.asarray(stock_cats)[k])


def _monthly(df: pd.DataFrame) -> list[dict]:
    m = df["invoice_date"].dt.to_period("M")
    storno = cat_startswith(df["invoice"], "C")
    no_cust = df["customer_id"].isna().to_numpy()
    rev = (df["quantity"] * df["price_gbp"]).to_numpy(dtype=float)
    frame = pd.DataFrame({"m": m, "storno": storno, "no_cust": no_cust & ~storno,
                          "with_cust": ~no_cust & ~storno, "rev": np.where(storno, 0, rev)})
    agg = frame.groupby("m").agg(storno=("storno", "sum"), no_cust=("no_cust", "sum"),
                                 with_cust=("with_cust", "sum"), revenue_gbp=("rev", "sum"))
    return [{"month": str(k), **{c: (round(float(v), 2) if c == "revenue_gbp" else int(v))
                                 for c, v in row.items()}} for k, row in agg.iterrows()]


# ── Upload-Parser ─────────────────────────────────────────────────────────────

def _safe_name(name: str) -> str:
    stem = Path(name).stem.strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return out or "tabelle"


def _read_csv(raw: bytes) -> pd.DataFrame:
    head = raw[:20_000].decode("utf-8", errors="replace")
    try:
        sep = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("CSV konnte nicht gelesen werden")


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Datumsspalten erkennen, Strings mit wenigen Ausprägungen als Kategorie."""
    for col in df.columns:
        s = df[col]
        if s.dtype == object:
            sample = s.dropna().astype(str).head(500)
            name_hint = any(h in col.lower() for h in ("date", "datum", "zeit", "time"))
            if len(sample) and (name_hint or sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9):
                parsed = pd.to_datetime(s, errors="coerce", dayfirst=not sample.str.match(r"^\d{4}-").all())
                if parsed.notna().mean() >= 0.9 * s.notna().mean():
                    df[col] = parsed
                    continue
            if s.nunique(dropna=True) < 0.5 * len(s):
                df[col] = s.astype("category")
    return df


def _schema_from_csv(df: pd.DataFrame) -> dict | None:
    cols = {c.lower(): c for c in df.columns}
    t = cols.get("tabelle") or cols.get("table") or cols.get("table_name")
    c = cols.get("spalte") or cols.get("column") or cols.get("column_name")
    if not (t and c):
        return None
    typ = cols.get("typ") or cols.get("type") or cols.get("data_type")
    tables: dict[str, list] = {}
    for _, row in df.iterrows():
        tables.setdefault(str(row[t]), []).append({
            "name": str(row[c]), "sql_type": str(row[typ]) if typ else "",
            "type": _semantic(str(row[typ]) if typ else ""), "nullable": True,
        })
    return {"tables": [{"name": k, "columns": v, "primary_key": []} for k, v in tables.items()],
            "foreign_keys": []}


def _semantic(sql_type: str) -> str:
    t = sql_type.lower()
    if any(k in t for k in ("date", "time")):
        return "datetime"
    if any(k in t for k in ("int",)):
        return "integer"
    if any(k in t for k in ("dec", "num", "float", "double", "money", "real")):
        return "number"
    if "bool" in t or t == "bit":
        return "boolean"
    if "text" in t or "clob" in t:
        return "text"
    return "string"


def _schema_from_json(obj) -> dict:
    """Akzeptiert {"tables":[{name, columns:[{name,type}]}], "foreign_keys":[…]} oder {"tabelle":["spalte",…]}."""
    if isinstance(obj, dict) and isinstance(obj.get("tables"), list):
        for t in obj["tables"]:
            t["columns"] = [c if isinstance(c, dict) else {"name": str(c)} for c in t.get("columns", [])]
            for c in t["columns"]:
                c.setdefault("type", _semantic(c.get("sql_type", "")))
            t.setdefault("primary_key", [])
        obj.setdefault("foreign_keys", [])
        return obj
    if isinstance(obj, dict):
        return {"tables": [{"name": k, "primary_key": [],
                            "columns": [{"name": str(c), "type": "string"} for c in v]} for k, v in obj.items()],
                "foreign_keys": []}
    raise ValueError("Unbekanntes Schema-Format")


# ── Registry ──────────────────────────────────────────────────────────────────

class Registry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._load_lock = threading.RLock()   # tables() ruft raw_online_retail() unter derselben Sperre
        self._datasets: dict[str, Dataset] = {}
        self._mappings: dict[str, dict] = {}
        self._overviews: dict[str, dict] = {}
        self._raw: pd.DataFrame | None = None        # Online Retail II flach (für Subgraph)
        self._orii_quality: dict = {}
        self._customer_stats: pd.DataFrame | None = None
        self._arrays: dict | None = None
        self._product_desc: pd.Series | None = None
        self._profiles: dict[str, dict] = {}
        self._checks: dict[tuple, dict] = {}
        self.reload_builtin()

    # builtin -----------------------------------------------------------------
    def reload_builtin(self) -> None:
        with self._lock:
            if ORII_META.exists():
                meta = json.loads(ORII_META.read_text(encoding="utf-8"))
                version = meta.get("zip_sha256", "")[:12] + f"-{ORII_NPZ.stat().st_size if ORII_NPZ.exists() else 0}"
                self._datasets["online_retail_ii"] = Dataset(
                    id="online_retail_ii", name="Online Retail II", kind="data",
                    source=meta["source"], license=meta["license"],
                    description=("Echte Transaktionen eines britischen Online-Großhändlers (12/2009–12/2011) – "
                                 "mit echten Lücken: fehlende Kundennummern, Stornos, Dubletten."),
                    builtin=True, version=version, facts=meta["facts"], labels=TABLE_LABELS_ORII,
                    files=[ORII_NPZ.name],
                )
            if AW_SCHEMA.exists():
                schema = json.loads(AW_SCHEMA.read_text(encoding="utf-8"))
                version = hashlib.sha256(AW_SCHEMA.read_bytes()).hexdigest()[:12]
                self._datasets["adventureworks"] = Dataset(
                    id="adventureworks", name="AdventureWorks", kind="schema",
                    source=schema["source"], license=schema["license"], description=schema["description"],
                    builtin=True, version=version, schema=schema,
                    facts={"tables": len(schema["tables"]),
                           "columns": sum(len(t["columns"]) for t in schema["tables"]),
                           "foreign_keys": len(schema["foreign_keys"]),
                           "schemas": sorted({t["schema"] for t in schema["tables"]})},
                    files=[AW_SCHEMA.name],
                )
            from . import connectors as cn
            for ds_id in cn.available():                          # Datensätze v2 (graph_ml/connectors.py)
                meta = cn.load_meta(ds_id)
                if not meta.get("render", True) and os.environ.get("RENDER"):
                    continue                                       # nicht-kommerzielle Daten nicht auf Render
                npz = DATA_DIR / f"{ds_id}.npz"
                rows = meta.get("rows", {})
                self._datasets[ds_id] = Dataset(
                    id=ds_id, name=meta["name"], kind="data", source=meta["source"], license=meta["license"],
                    description=meta["description"], builtin=True,
                    version=hashlib.sha256((meta.get("prepared_at", "") + str(npz.stat().st_size)).encode()).hexdigest()[:12],
                    facts={"tables": len(meta["tables"]), "rows": sum(rows.values()), "rows_by_table": rows,
                           "foreign_keys": len(meta.get("foreign_keys", [])), "synthetic": meta.get("synthetic"),
                           "source_url": meta.get("source_url")},
                    labels=meta.get("labels") or {}, files=[npz.name])
            self._mappings.clear()
            self._overviews.clear()
            self._profiles.clear()
            self._checks.clear()

    def max_stage(self, ds_id: str) -> int:
        kind = self.get(ds_id).kind
        return 1 if kind == "schema" else 2 if kind == PROFILE_KIND else 3

    def has(self, ds_id: str) -> bool:
        return ds_id in self._datasets

    def get(self, ds_id: str) -> Dataset:
        if ds_id not in self._datasets:
            raise KeyError(ds_id)
        return self._datasets[ds_id]

    def list(self) -> list[dict]:
        order = {"online_retail_ii": 0, "adventureworks_data": 1, "wwi": 2, "scania": 3, "cmapss": 4, "adventureworks": 5}
        ds = sorted(self._datasets.values(), key=lambda d: (order.get(d.id, 9), -d.created_at))
        return [d.card() for d in ds]

    # Daten laden -------------------------------------------------------------
    def raw_online_retail(self) -> pd.DataFrame:
        if self._raw is None:
            with self._load_lock:
                if self._raw is None:
                    t0 = time.time()
                    self._raw = load_online_retail(ORII_NPZ)
                    log.info("Online Retail II geladen: %s Zeilen in %.1f s", f"{len(self._raw):,}", time.time() - t0)
        return self._raw

    def tables(self, ds_id: str) -> dict[str, pd.DataFrame]:
        ds = self.get(ds_id)
        if ds.kind != "data":
            return {}
        if ds.tables is None and ds_id == "online_retail_ii":
            with self._load_lock:
                if ds.tables is None:
                    t0 = time.time()
                    ds.tables, self._orii_quality = build_online_retail_tables(self.raw_online_retail())
                    log.info("Relationales Schema abgeleitet in %.1f s", time.time() - t0)
        elif ds.tables is None and ds.builtin:
            from . import connectors as cn
            with self._load_lock:
                if ds.tables is None:
                    # Speicher (Render ≈ 512 MB): nur ein Datensatz v2 gleichzeitig im Speicher
                    for other in self._datasets.values():
                        if other.id != ds_id and other.id in cn.SPECS and other.tables is not None:
                            other.tables = None
                    import gc
                    gc.collect()
                    t0 = time.time()
                    ds.tables = cn.load_tables(ds_id)
                    log.info("%s geladen: %d Tabellen in %.1f s", ds_id, len(ds.tables), time.time() - t0)
        return ds.tables or {}

    def _connector_extra(self, ds_id: str) -> dict:
        """Deklarierte Schlüssel, Schema-Gruppen und Beschriftungen aus der meta.json eines Datensatzes v2."""
        from . import connectors as cn
        meta = cn.load_meta(ds_id) or {}
        pks = {t: v[0] for t, v in (meta.get("primary_keys") or {}).items() if v and len(v) == 1}
        return {"declared_fks": meta.get("foreign_keys") or [], "schemas": meta.get("schemas") or {},
                "labels": meta.get("labels") or {}, "declared_pks": pks}

    # Mapping -----------------------------------------------------------------
    def _artifact_path(self, kind: str, ds_id: str) -> Path:
        return ARTIFACTS_DIR / f"{kind}_{ds_id}.json"

    def _load_artifact(self, kind: str, ds: Dataset) -> dict | None:
        if not ds.builtin:
            return None
        p = self._artifact_path(kind, ds.id)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("source_version") != ds.version or data.get("artifact_version") != ARTIFACT_VERSION:
            return None
        return data

    def _save_artifact(self, kind: str, ds: Dataset, data: dict) -> None:
        if not ds.builtin:
            return
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        data = {**data, "source_version": ds.version, "artifact_version": ARTIFACT_VERSION}
        try:
            self._artifact_path(kind, ds.id).write_text(
                json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
        except OSError as exc:          # z. B. schreibgeschützt im Container
            log.warning("Artefakt nicht gespeichert: %s", exc)

    def mapping(self, ds_id: str, refresh: bool = False) -> dict:
        ds = self.get(ds_id)
        if not refresh:
            if ds_id in self._mappings:
                return self._mappings[ds_id]
            art = self._load_artifact("mapping", ds)
            if art:
                self._mappings[ds_id] = art
                return art
        if ds.kind == PROFILE_KIND:
            from .profiler import profile_to_mapping
            m = profile_to_mapping(ds.profile)
            m["dataset_id"] = ds_id
        elif ds.kind == "schema":
            m = build_schema_mapping(ds.schema, dataset_id=ds_id)
        else:
            tables = self.tables(ds_id)
            from . import connectors as cn
            extra = self._online_retail_extra() if ds_id == "online_retail_ii" else \
                self._connector_extra(ds_id) if ds_id in cn.SPECS else {}
            extra.setdefault("labels", ds.labels)
            m = build_mapping(tables, dataset_id=ds_id, extra=extra,
                              sample_note=None if ds.builtin else "Stichprobe")
        m["source_version"] = ds.version
        self._mappings[ds_id] = m
        self._save_artifact("mapping", ds, m)
        return m

    def _online_retail_extra(self, facts: dict | None = None, quality: dict | None = None,
                             flags: set | None = None) -> dict:
        """Befunde und Hinweise für Online Retail II – aus den Kennzahlen des jeweiligen Datenstands."""
        f = facts or self.get("online_retail_ii").facts
        q = quality if quality is not None else self._orii_quality
        flags = flags or set()
        extra = self._orii_extra_raw(f, q)
        drop = {"PB-02"} if "PB-02" in flags else set()
        for key, cnt in (("PB-06", f["duplicate_rows"]), ("PB-03", f["non_article_rows"]),
                         ("PB-04", q.get("products_multi_description", 0)),
                         ("PB-07", f["price_zero_rows"] + f["negative_qty_without_storno"])):
            if not cnt:
                drop.add(key)
        extra["findings"] = [x for x in extra["findings"] if x.get("playbook") not in drop]
        if "PB-02" in flags:
            extra["missing_relations"] = []
        if "PB-01" in flags or not f.get("rows_without_customer"):
            extra["insight_hints"] = []
        if "PB-04" in flags:
            extra["derived_columns"] = [d for d in extra["derived_columns"] if d["column"] != "products.n_beschreibungen"]
        return extra

    def _orii_extra_raw(self, f: dict, q: dict) -> dict:
        share_storno = f["storno_invoices"] / max(f["invoices"], 1)
        return {
            "labels": TABLE_LABELS_ORII,
            "declared_pks": {"invoice_lines": "line_id"},
            "derived_columns": [
                {"column": "customers.country", "note": "häufigstes Land über den Gesamtzeitraum abgeleitet"},
                {"column": "products.description", "note": "häufigste Beschreibung über den Gesamtzeitraum"},
                {"column": "products.n_beschreibungen", "note": "über den Gesamtzeitraum gezählt"},
            ],
            "missing_relations": [{
                "id": "invoices.storno->invoices.original",
                "from_table": "invoices", "from_column": "(keine Referenzspalte)",
                "to_table": "invoices", "to_column": "invoice",
                "label": "Storno → Originalrechnung",
                "rows": f["storno_invoices"], "linked_rows": 0, "null_rows": f["storno_invoices"],
                "orphan_rows": 0,
                "note": (f"{_n(f['storno_invoices'])} Storno-Belege ({_de(share_storno * 100, 1)} % aller Belege) "
                         "verweisen nicht auf die Originalrechnung."),
            }],
            "findings": [
                {"severity": "bad", "check": "Struktur & Schlüssel", "playbook": "PB-02",
                 "title": "Stornos ohne Bezug zur Originalrechnung",
                 "text": (f"{_n(f['storno_invoices'])} Storno-Belege mit {_n(f['storno_rows'])} Positionen – "
                          "für Retouren-Root-Cause muss die Verknüpfung heuristisch hergestellt werden.")},
                {"severity": "warn", "check": "Spaltenqualität", "playbook": "PB-06",
                 "title": f"{_n(f['duplicate_rows'])} exakte Dubletten",
                 "text": "Identische Positionen (gleiche Rechnung, Artikel, Menge, Zeit) – vor dem Training entfernen."},
                {"severity": "warn", "check": "Spaltenqualität", "playbook": "PB-04",
                 "title": f"{_n(q.get('products_multi_description', 0))} Artikel mit mehreren Beschreibungen",
                 "text": "Uneinheitlicher Artikelstamm: gleiche Artikelnummer, unterschiedliche Texte."},
                {"severity": "warn", "check": "Spaltenqualität", "playbook": "PB-03",
                 "title": f"{_n(f['non_article_rows'])} Positionen mit Nicht-Artikel-Codes",
                 "text": "POST, Porto, Gebühren, manuelle Buchungen (M, D, BANK CHARGES, AMAZONFEE …)."},
                {"severity": "info", "check": "Spaltenqualität", "playbook": "PB-07",
                 "title": f"{_n(f['price_zero_rows'])} × Preis 0 · {_n(f['negative_qty_without_storno'])} × negative Menge ohne Storno",
                 "text": "Bestandskorrekturen und Muster ohne Umsatz – markieren statt löschen."},
                {"severity": "info", "check": "Struktur & Schlüssel", "playbook": None,
                 "title": f"{q.get('customers_multi_country', 0)} Kunden in mehreren Ländern",
                 "text": "Mögliche Konzernstrukturen oder Dubletten – Land als Kante, nicht als Attribut."},
            ],
            "insight_hints": [{
                "icon": "money",
                "title": "Umsatz ohne Kundenbezug",
                "text": (f"£ {_de(f['revenue_gbp_without_customer'] / 1e6)} Mio. von £ {_de(f['revenue_gbp_total'] / 1e6)} "
                         "Mio. Umsatz hängen an keinem Kunden – für Warenkorb und Empfehlung nutzbar, für Churn "
                         "und Lead-Scoring nicht (Playbook PB-01)."),
            }],
        }

    # Übersicht für Schritt 1 -------------------------------------------------
    def overview(self, ds_id: str) -> dict:
        ds = self.get(ds_id)
        if ds_id in self._overviews:
            return self._overviews[ds_id]
        art = self._load_artifact("overview", ds)
        if art:
            self._overviews[ds_id] = art
            return art
        if ds.kind == PROFILE_KIND:
            ov = {"dataset": ds.card(), "tables": [
                {"name": t["name"], "label": t.get("label", t["name"]), "rows": t.get("rows"),
                 "n_columns": t.get("n_columns") or len(t["columns"]), "columns": [c["name"] for c in t["columns"]]}
                for t in ds.profile["tables"]]}
            self._overviews[ds_id] = ov
            return ov
        if ds.kind == "schema":
            ov = {"dataset": ds.card(), "tables": [
                {"name": t["name"], "schema": t.get("schema"), "rows": None, "n_columns": len(t["columns"]),
                 "columns": [c["name"] for c in t["columns"]]} for t in ds.schema["tables"]]}
        else:
            tables = self.tables(ds_id)
            ov = {"dataset": ds.card(), "tables": [
                {"name": name, "label": ds.labels.get(name, name), "rows": int(len(df)),
                 "n_columns": int(df.shape[1]), "columns": list(df.columns)} for name, df in tables.items()]}
            if ds_id == "online_retail_ii":
                raw = self.raw_online_retail()
                ov["tables"].insert(0, {"name": RAW_TABLE, "label": "Original-Export (flach)", "rows": int(len(raw)),
                                        "n_columns": int(raw.shape[1]), "columns": list(raw.columns), "raw": True})
                ov["monthly"] = _monthly(raw)
                ov["quality"] = self._orii_quality
                ov["sample_customers"] = self.sample_customers()
                # Vorschauen als Artefakt → Render muss für Schritt 1 nichts laden
                ov["previews"] = {name: self._preview_live(ds_id, name, 50)
                                  for name in [RAW_TABLE, *tables.keys()]}
        self._overviews[ds_id] = ov
        self._save_artifact("overview", ds, ov)
        return ov

    def preview(self, ds_id: str, table: str | None, n: int = 50) -> dict:
        ds = self.get(ds_id)
        if ds.builtin and ds.kind == "data":
            prev = self.overview(ds_id).get("previews", {})
            key = table or RAW_TABLE
            if key in prev and int(n) <= len(prev[key]["rows"]):
                return {**prev[key], "rows": prev[key]["rows"][: int(n)]}
        return self._preview_live(ds_id, table, n)

    def _preview_live(self, ds_id: str, table: str | None, n: int = 50) -> dict:
        ds = self.get(ds_id)
        n = max(5, min(int(n), 200))
        if ds.kind == PROFILE_KIND:
            tables = {t["name"]: t for t in ds.profile["tables"]}
            table = table if table in tables else next(iter(tables))
            cols = ["Spalte", "Typ", "Rollen", "Leerwerte", "Ausprägungen"]
            rows = [[c["name"], c.get("kind"), ", ".join(c.get("roles", [])),
                     None if c.get("null_rate") is None else f"{c['null_rate'] * 100:.1f} %".replace(".", ","),
                     c.get("n_unique")] for c in tables[table]["columns"]]
            return {"table": table, "columns": cols, "rows": rows, "total_rows": tables[table].get("rows"),
                    "schema_only": False, "profile_only": True}
        if ds.kind == "schema":
            tables = {t["name"]: t for t in ds.schema["tables"]}
            table = table if table in tables else next(iter(tables))
            t = tables[table]
            cols = ["Spalte", "SQL-Typ", "Typ", "Pflicht", "PK"]
            rows = [[c["name"], c.get("sql_type", ""), c.get("type", ""), "nein" if c.get("nullable") else "ja",
                     "✓" if c["name"] in (t.get("primary_key") or []) else ""] for c in t["columns"]]
            return {"table": table, "columns": cols, "rows": rows, "total_rows": None, "schema_only": True}
        if ds_id == "online_retail_ii" and table in (None, RAW_TABLE):
            # Flacher Original-Export: Hälfte mit, Hälfte ohne Kundennummer – Lücken sofort sichtbar
            raw = self.raw_online_retail()
            no_cust = np.flatnonzero(raw["customer_id"].isna().to_numpy())
            idx = np.r_[np.arange(n // 2), no_cust[: n - n // 2]]
            df_show = raw.iloc[idx]
            rows = [[_cell(v) for v in r] for r in df_show.itertuples(index=False, name=None)]
            return {"table": RAW_TABLE, "columns": list(df_show.columns), "rows": rows,
                    "total_rows": int(len(raw)), "schema_only": False}
        tables = self.tables(ds_id)
        if table not in tables:
            table = next(iter(tables))
        df = tables[table]
        df_show = df.head(n)
        rows = [[_cell(v) for v in r] for r in df_show.itertuples(index=False, name=None)]
        return {"table": table, "columns": list(df_show.columns), "rows": rows,
                "total_rows": int(len(df)), "schema_only": False}

    # Upload ------------------------------------------------------------------
    def add_upload(self, files: list[tuple[str, bytes]]) -> Dataset:
        tables: dict[str, pd.DataFrame] = {}
        schema: dict | None = None
        names = []
        for fname, raw in files:
            ext = Path(fname).suffix.lower()
            names.append(fname)
            if ext == ".json":
                obj = json.loads(raw.decode("utf-8-sig"))
                if isinstance(obj, dict) and "profile_version" in obj:
                    return self._add_profile(fname, obj)
                schema = _schema_from_json(obj)
                continue
            if ext in (".csv", ".txt"):
                df = _read_csv(raw)
                as_schema = _schema_from_csv(df)
                if as_schema:
                    schema = as_schema
                    continue
                tables[_safe_name(fname)] = df
            elif ext in (".xlsx", ".xls"):
                sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
                for sheet, df in sheets.items():
                    key = _safe_name(fname) if len(sheets) == 1 else f"{_safe_name(fname)}_{_safe_name(sheet)}"
                    tables[key] = df
            elif ext == ".parquet":
                tables[_safe_name(fname)] = pd.read_parquet(io.BytesIO(raw))
            else:
                raise ValueError(f"Dateityp {ext} wird nicht unterstützt")
        total_rows = sum(len(df) for df in tables.values())
        if total_rows > MAX_UPLOAD_ROWS:
            raise ValueError(f"Zu groß für die Demo: {total_rows:,} Zeilen (max. {MAX_UPLOAD_ROWS:,}).")
        ds_id = "upload_" + uuid.uuid4().hex[:8]
        if tables:
            tables = {k: _coerce(v) for k, v in tables.items()}
            facts = {"tables": len(tables), "rows": int(total_rows),
                     "columns": int(sum(df.shape[1] for df in tables.values()))}
            ds = Dataset(id=ds_id, name=names[0] if len(names) == 1 else f"{len(names)} Dateien", kind="data",
                         source="Eigener Upload", license="–", description="Hochgeladene Tabellen",
                         builtin=False, version=ds_id, facts=facts, tables=tables, files=names)
        elif schema:
            facts = {"tables": len(schema["tables"]),
                     "columns": sum(len(t["columns"]) for t in schema["tables"]),
                     "foreign_keys": len(schema.get("foreign_keys", []))}
            ds = Dataset(id=ds_id, name=names[0], kind="schema", source="Schema-Import", license="–",
                         description="Tabellen- und Feldliste (Stufe 1)", builtin=False, version=ds_id,
                         facts=facts, schema=schema, files=names)
        else:
            raise ValueError("Keine verwertbaren Tabellen gefunden")
        with self._lock:
            uploads = sorted((d for d in self._datasets.values() if not d.builtin), key=lambda d: d.created_at)
            for old in uploads[: max(0, len(uploads) - MAX_UPLOADS + 1)]:
                self._datasets.pop(old.id, None)
                self._mappings.pop(old.id, None)
                self._overviews.pop(old.id, None)
            self._datasets[ds_id] = ds
        return ds

    def _add_profile(self, fname: str, profile: dict) -> Dataset:
        from .profiler import assert_privacy
        assert_privacy(profile)
        ds_id = "upload_" + uuid.uuid4().hex[:8]
        ev = profile.get("events") or {}
        facts = {"tables": len(profile.get("tables", [])),
                 "rows": int(sum(t.get("rows") or 0 for t in profile.get("tables", []))),
                 "months": (ev.get("time") or {}).get("months"), "generated_at": profile.get("generated_at")}
        ds = Dataset(id=ds_id, name=profile.get("dataset_name") or fname, kind=PROFILE_KIND,
                     source="Profil-Export (nur Kennzahlen)", license="–",
                     description="Kennzahlen-Profil aus dem Profiler – keine Datenzeilen, keine personenbezogenen Daten.",
                     builtin=False, version=ds_id, facts=facts, profile=profile, files=[fname])
        with self._lock:
            self._datasets[ds_id] = ds
        return ds

    # Profil (Stufe 2) ----------------------------------------------------------
    def profile(self, ds_id: str) -> dict | None:
        """Kennzahlen-Profil. Für Online Retail II aus dem Artefakt (wie ein lokaler Kunden-Export)."""
        ds = self.get(ds_id)
        if ds.kind == PROFILE_KIND:
            return ds.profile
        if ds.kind == "schema":
            return None
        if ds_id in self._profiles:
            return self._profiles[ds_id]
        art = self._load_artifact("profile", ds)
        if art:
            self._profiles[ds_id] = art
            return art
        from .profiler import assemble_events, build_profile, events_from_online_retail
        from .semantics import bind_fields, schema_view
        mapping = self.mapping(ds_id)
        tables = self.tables(ds_id)
        if ds_id == "online_retail_ii":
            events = events_from_online_retail(self.raw_online_retail())
        else:
            events = assemble_events(tables, mapping, bind_fields(schema_view(mapping)))
        prof = build_profile(dataset_name=ds.name, mapping=mapping, tables=tables, events=events,
                             source=f"GraphReady Profiler · {ds.source}")
        self._profiles[ds_id] = prof
        self._save_artifact("profile", ds, prof)
        return prof

    def schema_view(self, ds_id: str) -> dict:
        from .semantics import schema_view
        return schema_view(self.mapping(ds_id))

    def checks(self, ds_id: str, stage: int, use_cases: list[str] | None = None) -> dict:
        from .checks import run_checks
        ds = self.get(ds_id)
        stage = max(1, min(int(stage), 3))
        key = (ds_id, ds.version, stage, tuple(sorted(use_cases or [])))
        if key in self._checks:
            return self._checks[key]
        if stage == 1 or ds.kind == "schema":
            res = run_checks(stage=stage, view=self.schema_view(ds_id), use_case_ids=use_cases, dataset_id=ds_id)
        else:
            res = run_checks(stage=stage, view={}, profile=self.profile(ds_id), use_case_ids=use_cases,
                             dataset_id=ds_id)
        res["dataset_kind"] = ds.kind
        res["max_stage_dataset"] = self.max_stage(ds_id)
        self._checks[key] = res
        if stage == 3 and ds_id in ("online_retail_ii", "wwi"):
            res = _merge_backtest(res, ds_id)
        return res

    def delete(self, ds_id: str) -> bool:
        with self._lock:
            ds = self._datasets.get(ds_id)
            if not ds or ds.builtin:
                return False
            self._datasets.pop(ds_id)
            self._mappings.pop(ds_id, None)
            self._overviews.pop(ds_id, None)
            self._profiles.pop(ds_id, None)
            self._checks = {k: v for k, v in self._checks.items() if k[0] != ds_id}
            return True

    # Beispiel-Subgraph (Online Retail II) ------------------------------------
    def _orii_arrays(self) -> dict:
        if self._arrays is None:
            raw = self.raw_online_retail()
            self._arrays = {
                "cust":  raw["customer_id"].astype("float64").to_numpy(na_value=np.nan),
                "stock": raw["stock_code"].cat.codes.to_numpy(),
            }
        return self._arrays

    def customer_stats(self) -> pd.DataFrame:
        if self._customer_stats is None:
            raw = self.raw_online_retail()
            d = raw.loc[raw["customer_id"].notna(), ["customer_id", "invoice", "invoice_date", "country",
                                                      "quantity", "price_gbp"]]
            storno = cat_startswith(d["invoice"], "C")
            d = d.assign(rev=np.where(storno, 0.0, (d["quantity"] * d["price_gbp"]).astype(float)), storno=storno)
            g = d.groupby("customer_id")
            st = pd.DataFrame({
                "revenue": g["rev"].sum(),
                "invoices": g["invoice"].nunique(),
                "first": g["invoice_date"].min(),
                "last": g["invoice_date"].max(),
                "stornos": d[d["storno"]].groupby("customer_id")["invoice"].nunique(),
                "country": g["country"].first(),
            }).fillna({"stornos": 0})
            self._customer_stats = st
        return self._customer_stats

    def sample_customers(self) -> list[dict]:
        ov = self._overviews.get("online_retail_ii") or self._load_artifact("overview", self.get("online_retail_ii"))
        if ov and ov.get("sample_customers"):
            return ov["sample_customers"]
        st = self.customer_stats()
        end = st["last"].max()
        picks: list[tuple[str, int]] = []
        top = st.sort_values("revenue", ascending=False)
        picks += [("Top-Kunde (Umsatz)", int(i)) for i in top.index[:3]]
        mid = st[(st["invoices"] >= 5) & (st["invoices"] <= 12)].sort_values("revenue")
        if len(mid):
            picks.append(("Typischer Stammkunde", int(mid.index[len(mid) // 2])))
        churned = st[(st["last"] < end - pd.Timedelta(days=270)) & (st["invoices"] >= 6)].sort_values(
            "revenue", ascending=False)
        if len(churned):
            picks.append(("Abgewandert (> 9 Monate inaktiv)", int(churned.index[0])))
        cold = st[st["invoices"] == 1].sort_values("revenue", ascending=False)
        if len(cold):
            picks.append(("Cold Start (1 Rechnung)", int(cold.index[len(cold) // 3])))
        ret = st[st["stornos"] >= 5].sort_values("stornos", ascending=False)
        if len(ret):
            picks.append(("Viele Stornos (After Sales)", int(ret.index[min(2, len(ret) - 1)])))
        out, seen = [], set()
        for label, cid in picks:
            if cid in seen:
                continue
            seen.add(cid)
            r = st.loc[cid]
            out.append({"customer_id": cid, "label": label, "country": str(r["country"]),
                        "invoices": int(r["invoices"]), "revenue_gbp": round(float(r["revenue"]), 2),
                        "last": r["last"].date().isoformat()})
        return out

    def random_customer(self, seed: int | None = None) -> int:
        st = self.customer_stats()
        rng = np.random.default_rng(seed)
        return int(st.index[rng.integers(0, len(st))])

    def customer_subgraph(self, customer_id: int, max_nodes: int = 150) -> dict:
        raw = self.raw_online_retail()
        arr = self._orii_arrays()
        st = self.customer_stats()
        if customer_id not in st.index:
            raise KeyError(f"Kunde {customer_id} nicht gefunden")
        rows = raw.iloc[np.flatnonzero(arr["cust"] == customer_id)]
        storno = cat_startswith(rows["invoice"], "C")
        rows = rows.assign(rev=(rows["quantity"] * rows["price_gbp"]).astype(float), storno=storno)

        inv = (rows.groupby("invoice", observed=True)
               .agg(date=("invoice_date", "min"), n=("stock_code", "size"), rev=("rev", "sum"),
                    storno=("storno", "first"))
               .sort_values("date", ascending=False))
        n_inv_total = len(inv)
        inv = inv.head(12)
        keep_rows = rows[rows["invoice"].isin(inv.index)]
        prod = (keep_rows.groupby("stock_code", observed=True)
                .agg(rev=("rev", "sum"), qty=("quantity", "sum"), n=("invoice", "nunique"))
                .assign(absrev=lambda x: x["rev"].abs()).sort_values("absrev", ascending=False))
        n_prod_total = int(rows["stock_code"].nunique())
        budget = max_nodes - 2 - len(inv)
        prod = prod.head(max(10, min(32, int(budget * 0.4))))
        prod_codes = set(prod.index.astype(str))

        # Nachbarkunden: kaufen dieselben Artikel (Co-Purchase, 2 Schritte entfernt)
        code_idx = raw["stock_code"].cat.categories.get_indexer(list(prod.index.astype(str)))
        mask = np.isin(arr["stock"], code_idx) & ~np.isnan(arr["cust"]) & (arr["cust"] != customer_id)
        nb = raw.iloc[np.flatnonzero(mask)][["customer_id", "stock_code"]].drop_duplicates()
        nb_counts = nb.groupby("customer_id").size().sort_values(ascending=False)
        n_nb_total = int(len(nb_counts))
        nb_keep = nb_counts.head(max(5, min(40, max_nodes - 2 - len(inv) - len(prod))))
        # je Nachbar höchstens 3 Kanten (die meistgekauften gemeinsamen Artikel) – lesbarer Graph
        rank = prod["absrev"].rank(ascending=False)
        nb_edges = nb[nb["customer_id"].isin(nb_keep.index)].assign(
            r=lambda x: x["stock_code"].astype(str).map(rank.set_axis(rank.index.astype(str))))
        nb_edges = nb_edges.sort_values("r").groupby("customer_id").head(3)[["customer_id", "stock_code"]]

        if self._product_desc is None:
            self._product_desc = product_descriptions(raw)
        products_desc = self._product_desc
        country = str(st.loc[customer_id, "country"])
        nodes, edges = [], []
        cid = f"c:{customer_id}"
        nodes.append({"data": {"id": cid, "label": f"Kunde {customer_id}", "type": "customer_focus",
                               "info": f"{n_inv_total} Rechnungen · £ {st.loc[customer_id, 'revenue']:,.0f}"}})
        nodes.append({"data": {"id": f"l:{country}", "label": country, "type": "country"}})
        edges.append({"data": {"source": cid, "target": f"l:{country}", "type": "country"}})
        for code, r in inv.iterrows():
            iid = f"i:{code}"
            nodes.append({"data": {"id": iid, "label": str(code), "type": "storno" if r["storno"] else "invoice",
                                   "info": f"{r['date']:%d.%m.%Y} · {int(r['n'])} Pos. · £ {r['rev']:,.0f}"}})
            edges.append({"data": {"source": cid, "target": iid, "type": "places"}})
        for code, r in prod.iterrows():
            desc = products_desc.get(code)
            nodes.append({"data": {"id": f"p:{code}", "label": _short(str(desc) if pd.notna(desc) else str(code)),
                                   "type": "product", "info": f"{code} · {int(r['qty'])} Stück"}})
        pairs = keep_rows[keep_rows["stock_code"].astype(str).isin(prod_codes)][["invoice", "stock_code"]]
        for inv_code, code in pairs.drop_duplicates().itertuples(index=False, name=None):
            edges.append({"data": {"source": f"i:{inv_code}", "target": f"p:{code}", "type": "contains"}})
        for other, shared in nb_keep.items():
            nodes.append({"data": {"id": f"n:{int(other)}", "label": f"{int(other)}", "type": "neighbor",
                                   "info": f"{int(shared)} gemeinsame Artikel"}})
        for other, code in nb_edges.itertuples(index=False, name=None):
            edges.append({"data": {"source": f"n:{int(other)}", "target": f"p:{code}", "type": "co_purchase"}})

        return {
            "customer_id": int(customer_id), "nodes": nodes, "edges": edges,
            "stats": {
                "invoices": n_inv_total, "invoices_shown": int(len(inv)),
                "stornos": int(inv["storno"].sum()) if len(inv) else 0,
                "stornos_total": int(rows.loc[rows["storno"], "invoice"].nunique()),
                "products": n_prod_total, "products_shown": int(len(prod)),
                "neighbors": n_nb_total, "neighbors_shown": int(len(nb_keep)),
                "revenue_gbp": round(float(st.loc[customer_id, "revenue"]), 2),
                "first": st.loc[customer_id, "first"].date().isoformat(),
                "last": st.loc[customer_id, "last"].date().isoformat(),
                "country": country,
            },
        }

    def memory_mb(self) -> dict:
        def df_mb(df):
            return df.memory_usage(deep=False).sum() / 1e6
        out = {"raw": round(df_mb(self._raw), 1) if self._raw is not None else 0.0}
        ds = self._datasets.get("online_retail_ii")
        out["tables"] = round(sum(df_mb(t) for t in ds.tables.values()), 1) if ds and ds.tables else 0.0
        return out


def _merge_backtest(res: dict, ds_id: str = "online_retail_ii") -> dict:
    """Stufe 3: Ampel-Dimension „Vorhersagekraft“ aus dem Rückblick; Gesamt = schwächste belastbare Dimension.
    Nur Rückblicke, die auf genau diesem Datensatz gerechnet wurden (Account-Scoring → WideWorldImporters)."""
    import copy

    from .backtest import DATASET_OF
    from .evaluation import summary
    res = copy.deepcopy(res)
    summ = summary()
    rank = {"ok": 0, "warn": 1, "bad": 2}
    for u in res["use_cases"]:
        home = {"wwi": "wwi"}.get(DATASET_OF.get(u["id"], ""), "online_retail_ii")
        if home != ds_id:
            continue
        s = summ.get(u["id"]) or {}
        vk = s.get("vorhersagekraft")
        if s.get("status") == "pending" or not vk:
            continue
        u["ampel"]["vorhersagekraft"] = {**vk, "best": s.get("best_name"), "metric": s.get("main_metric_label"),
                                         "value": s.get("best_value")}
        u["ampel"]["gesamt"] = {"status": "pending"}
    # Euro-Potenzial (Was-wäre-wenn, realistisch) → Gesamt = schwächste der drei Dimensionen
    from .business_case import merge_ampel, overview
    cases = overview()["cases"]
    for u in res["use_cases"]:
        if u["ampel"]["vorhersagekraft"].get("status") not in ("pending", None):
            merge_ampel(u, cases.get(u["id"]))
    res["backtest_merged"] = True
    return res


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _short(s: str, n: int = 26) -> str:
    s = s.strip().title()
    return s if len(s) <= n else s[: n - 1] + "…"


def _cell(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 2)
    return v if isinstance(v, (int, str, bool)) else str(v)


_registry: Registry | None = None
_registry_lock = threading.Lock()


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = Registry()
    return _registry
