"""
Konnektoren v2 – weitere Datensätze als kompakte, relationale Tabellen
======================================================================
Jeder Datensatz wird einmal lokal aufbereitet und als `graph_ml/data/<id>.npz` (Tabellen spaltenweise,
Texte als Kategorien) plus `<id>.meta.json` (Quelle, Lizenz, Gruppen, deklarierte Schlüssel) gespeichert.
Render lädt nur diese Dateien – nie die Rohdownloads.

  python -m graph_ml.connectors --only adventureworks_data,cmapss,scania,wwi

| id                  | Quelle                                   | Lizenz        | auf Render |
|---------------------|------------------------------------------|---------------|------------|
| adventureworks_data | Microsoft sql-server-samples (CSV)       | MIT           | ja         |
| wwi                 | WideWorldImporters-Full.bak (SQL Server) | MIT           | ja         |
| cmapss              | NASA Prognostics Data Repository         | Public Domain | ja         |
| scania              | SCANIA Component X (SND, DOI 10.5878/jvb5-d390) | CC BY 4.0 | ja (verdichtet) |

Nicht-kommerzielle Datensätze (Olist, SAP SALT) sind als `render: false` vorgesehen und werden nur lokal geladen.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
LOCAL_DIR = DATA_DIR / "local"            # große Zwischenstände, nicht in Git
AW_SCHEMA = DATA_DIR / "adventureworks_schema.json"

SPECS: dict[str, dict] = {
    "adventureworks_data": {
        "name": "AdventureWorks (mit Daten)",
        "source": "Microsoft sql-server-samples · AdventureWorks OLTP (CSV)",
        "source_url": "https://github.com/microsoft/sql-server-samples/releases/tag/adventureworks",
        "license": "MIT (künstliche Beispieldaten)", "render": True, "synthetic": True,
        "description": ("Fahrradhersteller mit Aufträgen und Rabatten, Sonderangeboten, Gebieten, Vertrieblern mit Quoten, "
                        "Stücklisten, Fertigungsaufträgen mit Ausschuss und Lieferanten – ERP-typisch über 60 Tabellen."),
    },
    "wwi": {
        "name": "WideWorldImporters",
        "source": "Microsoft sql-server-samples · WideWorldImporters-Full.bak",
        "source_url": "https://github.com/microsoft/sql-server-samples/releases/tag/wide-world-importers-v1.0",
        "license": "MIT (künstliche Beispieldaten)", "render": True, "synthetic": True,
        "description": ("Großhändler: Bestellung → Rechnung, Einkaufsgemeinschaften, Sonderkonditionen, Lieferanten, "
                        "Lagerbewegungen – Angebots-, Account- und Lager-Use-Cases."),
    },
    "cmapss": {
        "name": "NASA C-MAPSS (Turbinen)",
        "source": "NASA Prognostics Center of Excellence · Turbofan Engine Degradation Simulation",
        "source_url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
        "license": "Public Domain (NASA)", "render": True, "synthetic": True,
        "description": ("Simulierte Triebwerke bis zum Ausfall: 4 Flotten, je Zyklus 3 Betriebsparameter und 21 Sensoren – "
                        "Benchmark für Restlebensdauer und Predictive Maintenance."),
    },
    "scania": {
        "name": "SCANIA Component X (After Sales)",
        "source": "Scania CV AB · Swedish National Data Service (DOI 10.5878/jvb5-d390)",
        "source_url": "https://doi.org/10.5878/jvb5-d390",
        "license": "CC BY 4.0 · Kharazian et al., Scientific Data 12 (2025) 493", "render": True, "synthetic": False,
        "description": ("Echte Lkw-Flotte (anonymisiert): Fahrzeugkonfiguration, Betriebsdaten-Auslesungen in der Werkstatt "
                        "und Reparaturen der Komponente X – offener Benchmark für Ausfall und Reparatur."),
    },
}


# ── Speicherformat ────────────────────────────────────────────────────────────

def save_tables(ds_id: str, tables: dict[str, pd.DataFrame], meta: dict) -> dict:
    """Tabellen spaltenweise speichern: Zahlen kompakt, Texte als Kategorien-Codes, Zeiten als int64."""
    arrays, cols = {}, {}
    for t, df in tables.items():
        cols[t] = []
        for c in df.columns:
            s = df[c]
            key = f"{t}|{c}"
            if pd.api.types.is_datetime64_any_dtype(s):
                arrays[key] = s.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                kind = "datetime"
            elif pd.api.types.is_bool_dtype(s):
                arrays[key] = s.to_numpy(dtype=bool)
                kind = "bool"
            elif pd.api.types.is_numeric_dtype(s):
                v = s.to_numpy(dtype="float64", na_value=np.nan)
                if np.isfinite(v).all() and np.allclose(v, np.round(v)) and np.abs(v).max(initial=0) < 2**31:
                    arrays[key] = v.astype(np.int32)
                    kind = "int"
                else:
                    arrays[key] = v.astype(np.float32) if (np.abs(v[np.isfinite(v)]).max(initial=0) < 1e7) else v
                    kind = "float"
            else:
                cat = s.astype("string").astype("category")
                lens = cat.cat.categories.str.len()
                if len(cat.cat.categories) > 0.5 * max(len(s), 1) and len(lens) and lens.to_series().mean() > 120:
                    continue                                   # Text-Blobs (JSON, Rohsensorik) – kein Merkmal, nur Speicher
                arrays[key] = cat.cat.codes.to_numpy().astype(np.int32)
                arrays[key + "|cats"] = np.asarray(cat.cat.categories.astype(str), dtype=object).astype("U")
                kind = "string"
            cols[t].append({"name": c, "kind": kind})
    np.savez_compressed(DATA_DIR / f"{ds_id}.npz", **arrays)
    meta = {**SPECS.get(ds_id, {}), **meta, "id": ds_id, "tables": cols,
            "rows": {t: int(len(df)) for t, df in tables.items()},
            "prepared_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    (DATA_DIR / f"{ds_id}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
    return meta


def load_meta(ds_id: str) -> dict | None:
    p = DATA_DIR / f"{ds_id}.meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_tables(ds_id: str) -> dict[str, pd.DataFrame]:
    meta = load_meta(ds_id)
    z = np.load(DATA_DIR / f"{ds_id}.npz", allow_pickle=False)
    out = {}
    for t, cols in meta["tables"].items():
        data = {}
        for c in cols:
            a = z[f"{t}|{c['name']}"]
            if c["kind"] == "datetime":
                v = a.astype("datetime64[ns]")
                v[a == np.iinfo(np.int64).min] = np.datetime64("NaT")
                data[c["name"]] = v
            elif c["kind"] == "string":
                cats = z[f"{t}|{c['name']}|cats"]
                data[c["name"]] = pd.Categorical.from_codes(a, categories=pd.Index(cats.astype(str)), validate=False) \
                    if len(cats) else pd.Categorical([None] * len(a))
            else:
                data[c["name"]] = a
        out[t] = pd.DataFrame(data)
    return out


def available() -> list[str]:
    return [k for k in SPECS if (DATA_DIR / f"{k}.npz").exists() and (DATA_DIR / f"{k}.meta.json").exists()]


# ── AdventureWorks mit Daten ──────────────────────────────────────────────────

AW_SKIP_TABLES = {"Production.ProductPhoto", "Production.Document", "Production.Illustration",
                  "HumanResources.JobCandidate", "Person.Password", "dbo.AWBuildVersion", "dbo.DatabaseLog",
                  "dbo.ErrorLog", "Production.ProductModelIllustration", "Production.ProductDocument"}
AW_DROP_TYPES = ("varbinary", "xml", "uniqueidentifier", "hierarchyid", "geography")


def _aw_formats(sql: str) -> dict[str, tuple[str, str]]:
    out = {}
    for sch, t, f, opts in re.findall(r"BULK INSERT \[(\w+)\]\.\[(\w+)\] FROM '\$\(SqlSamplesSourceDataPath\)(\w+\.csv)'"
                                      r"\s*WITH\s*\((.*?)\)", sql, re.S):
        fld = re.search(r"FIELDTERMINATOR\s*=\s*'([^']*)'", opts).group(1)
        row = re.search(r"ROWTERMINATOR\s*=\s*'([^']*)'", opts).group(1)
        out[f"{sch}.{t}"] = (f, "\t" if fld == r"\t" else fld, "&|\n" if row.startswith("&|") else "\n")
    return out


def _aw_cast(s: pd.Series, col: dict) -> pd.Series:
    t = col["sql_type"].lower()
    if any(x in t for x in ("int", "bit")):
        v = pd.to_numeric(s, errors="coerce")
        return v.astype("float64") if v.isna().any() else v.astype("int64")
    if any(x in t for x in ("money", "decimal", "numeric", "float", "real")):
        return pd.to_numeric(s, errors="coerce").astype("float64")
    if "date" in t or "time" in t:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    return s.replace("", None)


def prepare_adventureworks_data(log=print) -> dict:
    raw = RAW_DIR / "adventureworks" / "oltp.zip"
    schema = json.loads(AW_SCHEMA.read_text(encoding="utf-8"))
    zf = zipfile.ZipFile(raw)
    names = {Path(n).name: n for n in zf.namelist()}
    sql = zf.read(names["instawdb.sql"]).decode("utf-8", errors="replace")
    fmts = _aw_formats(sql)
    tables, schemas, labels = {}, {}, {}
    for t in schema["tables"]:
        name = t["name"]
        if name in AW_SKIP_TABLES or name not in fmts:
            continue
        fname, fsep, rsep = fmts[name]
        txt = zf.read(names[fname]).decode("utf-8-sig", errors="replace")
        if txt.startswith("﻿"):
            txt = txt[1:]
        rows = [r for r in txt.split(rsep) if r.strip("\r\n")]
        recs = [r.rstrip("\r").split(fsep) for r in rows]
        # manche CSV enthalten berechnete Spalten (z. B. TotalDue), andere nicht – nach Feldzahl wählen
        width = max((len(r) for r in recs[:50]), default=0)
        cols = t["columns"] if width == len(t["columns"]) else [c for c in t["columns"] if not c.get("computed")]
        if width != len(cols):
            log(f"[aw] Warnung {name}: {width} Felder, erwartet {len(cols)}")
        recs = [r +[""] * (len(cols) - len(r)) if len(r) < len(cols) else r[:len(cols)] for r in recs]
        df = pd.DataFrame(recs, columns=[c["name"] for c in cols])
        keep = [c for c in cols if not any(x in c["sql_type"].lower() for x in AW_DROP_TYPES)
                and c["name"] not in ("rowguid", "ModifiedDate")]
        df = pd.DataFrame({c["name"]: _aw_cast(df[c["name"]], c) for c in keep})
        short = name.split(".", 1)[1]
        tables[short] = df
        schemas[short] = t["schema"]
        log(f"[aw] {name}: {len(df):,} Zeilen, {df.shape[1]} Spalten")
    fks = [{"from_table": f["from_table"].split(".", 1)[1], "from_columns": f["from_columns"],
            "to_table": f["to_table"].split(".", 1)[1], "to_columns": f["to_columns"]}
           for f in schema["foreign_keys"]
           if f["from_table"].split(".", 1)[1] in tables and f["to_table"].split(".", 1)[1] in tables]
    pks = {t["name"].split(".", 1)[1]: t.get("primary_key") for t in schema["tables"]
           if t["name"].split(".", 1)[1] in tables}
    return save_tables("adventureworks_data", tables, {"schemas": schemas, "foreign_keys": fks, "primary_keys": pks,
                                                       "labels": labels})


# ── NASA C-MAPSS ──────────────────────────────────────────────────────────────

CMAPSS_FLEETS = {"FD001": ("1 Betriebsbedingung (Meeresniveau)", "HPC-Degradation"),
                 "FD002": ("6 Betriebsbedingungen", "HPC-Degradation"),
                 "FD003": ("1 Betriebsbedingung (Meeresniveau)", "HPC- und Fan-Degradation"),
                 "FD004": ("6 Betriebsbedingungen", "HPC- und Fan-Degradation")}
CMAPSS_COLS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]


def _inner_zip(path: Path, pattern: str) -> zipfile.ZipFile:
    z = zipfile.ZipFile(path)
    for n in z.namelist():
        if re.search(pattern, n, re.I) and n.lower().endswith(".zip"):
            return zipfile.ZipFile(io.BytesIO(z.read(n)))
    return z


def prepare_cmapss(log=print) -> dict:
    z = _inner_zip(RAW_DIR / "cmapss" / "cmapss.zip", r"CMAPSSData")
    names = {Path(n).name: n for n in z.namelist()}
    engines, cycles = [], []
    for fd, (cond, fault) in CMAPSS_FLEETS.items():
        for split in ("train", "test"):
            df = pd.read_csv(io.BytesIO(z.read(names[f"{split}_{fd}.txt"])), sep=r"\s+", header=None,
                             names=CMAPSS_COLS, engine="python")
            df["engine_id"] = fd + "-" + split[:2] + "-" + df["unit"].astype(str)
            last = df.groupby("engine_id")["cycle"].max()
            e = pd.DataFrame({"engine_id": last.index, "fleet_id": fd, "split": "Training" if split == "train" else "Test",
                              "cycles_observed": last.to_numpy()})
            if split == "train":
                e["failed"] = True
                e["rul_at_end"] = 0
            else:
                rul = pd.read_csv(io.BytesIO(z.read(names[f"RUL_{fd}.txt"])), header=None, names=["rul"])["rul"]
                order = df.drop_duplicates("engine_id")["engine_id"].to_numpy()
                e = e.set_index("engine_id").loc[order].reset_index()
                e["failed"] = False
                e["rul_at_end"] = rul.to_numpy()[: len(e)]
            engines.append(e)
            cycles.append(df.drop(columns=["unit"]))
            log(f"[cmapss] {split}_{fd}: {len(df):,} Zyklen, {len(e)} Triebwerke")
    fleets = pd.DataFrame([{"fleet_id": k, "conditions": v[0], "fault_modes": v[1]} for k, v in CMAPSS_FLEETS.items()])
    cyc = pd.concat(cycles, ignore_index=True)
    cyc.insert(0, "cycle_id", np.arange(len(cyc)))
    tables = {"fleets": fleets, "engines": pd.concat(engines, ignore_index=True), "cycles": cyc}
    return save_tables("cmapss", tables, {
        "schemas": {"fleets": "Stammdaten", "engines": "Anlagen", "cycles": "Betrieb"},
        "foreign_keys": [{"from_table": "engines", "from_columns": ["fleet_id"], "to_table": "fleets", "to_columns": ["fleet_id"]},
                         {"from_table": "cycles", "from_columns": ["engine_id"], "to_table": "engines", "to_columns": ["engine_id"]}],
        "primary_keys": {"fleets": ["fleet_id"], "engines": ["engine_id"], "cycles": ["cycle_id"]},
        "labels": {"fleets": "Flotten", "engines": "Triebwerke", "cycles": "Betriebszyklen"}})


# ── SCANIA Component X ────────────────────────────────────────────────────────

def prepare_scania(log=print) -> dict:
    """Konfiguration (8 Merkmale) als eigene Tabellen → Fahrzeuge teilen Konfigurationsknoten (Graph-Signal).

    Die 1,1 Mio. Werkstatt-Auslesungen bleiben vollständig lokal (data/local/scania_readouts.npz); ins Repository
    kommen je Fahrzeug die letzte Auslesung plus Verlaufskennzahlen.
    """
    d = RAW_DIR / "scania"
    spec = pd.read_csv(d / "train_specifications.csv")
    tte = pd.read_csv(d / "train_tte.csv")
    log(f"[scania] Spezifikation {len(spec):,} Fahrzeuge · Reparaturen {int(tte['in_study_repair'].sum()):,}")
    t0 = time.time()
    ro = pd.read_csv(d / "train_operational_readouts.csv", dtype=np.float32)
    ro["vehicle_id"] = ro["vehicle_id"].astype(np.int32)
    log(f"[scania] Auslesungen {len(ro):,} × {ro.shape[1]} in {time.time() - t0:.0f} s")
    LOCAL_DIR.mkdir(exist_ok=True)
    np.savez_compressed(LOCAL_DIR / "scania_readouts.npz", columns=np.asarray(ro.columns, dtype="U"),
                        values=ro.to_numpy(dtype=np.float32))
    feats = [c for c in ro.columns if c not in ("vehicle_id", "time_step")]
    bases = pd.Series([c.split("_")[0] for c in feats]).value_counts()
    counters = [c for c in feats if bases[c.split("_")[0]] == 1]                 # Zähler; Histogramme haben _0…_n
    ro = ro.sort_values(["vehicle_id", "time_step"])
    g = ro.groupby("vehicle_id")
    last = g.tail(1).set_index("vehicle_id")
    summary = pd.DataFrame({"vehicle_id": last.index.astype(np.int32),
                            "n_readouts": g.size().reindex(last.index).to_numpy(),
                            "first_time_step": g["time_step"].min().reindex(last.index).to_numpy(),
                            "last_time_step": last["time_step"].to_numpy()})
    for c in counters:
        summary[f"last_{c}"] = last[c].to_numpy()
    readouts_sample = ro[ro["vehicle_id"].isin(spec["vehicle_id"].sample(400, random_state=7))][
        ["vehicle_id", "time_step"] + counters].reset_index(drop=True)
    readouts_sample.insert(0, "readout_id", np.arange(len(readouts_sample)))
    vehicles = spec.merge(tte, on="vehicle_id", how="left").rename(columns={
        "length_of_study_time_step": "study_length", "in_study_repair": "repaired"})
    spec_cols = [c for c in spec.columns if c.lower().startswith("spec")]
    tables = {"vehicles": vehicles.rename(columns={c: c.lower() for c in spec_cols})}
    fks, schemas, labels = [], {"vehicles": "Fahrzeuge"}, {"vehicles": "Fahrzeuge"}
    for c in spec_cols:
        vals = pd.DataFrame({c.lower(): sorted(spec[c].dropna().unique())})
        vals["n_vehicles"] = vals[c.lower()].map(spec[c].value_counts()).astype(int)
        name = f"config_{c.lower().split('_')[-1]}"
        tables[name] = vals
        fks.append({"from_table": "vehicles", "from_columns": [c.lower()], "to_table": name, "to_columns": [c.lower()]})
        schemas[name], labels[name] = "Konfiguration", f"Konfiguration {c.split('_')[-1]}"
    repairs = vehicles.loc[vehicles["repaired"] == 1, ["vehicle_id", "study_length"]].rename(
        columns={"study_length": "repair_time_step"}).reset_index(drop=True)
    repairs.insert(0, "repair_id", np.arange(len(repairs)))
    tables["repairs"] = repairs
    tables["readout_summary"] = summary
    tables["readouts_sample"] = readouts_sample
    for t, sch, lab in (("repairs", "Service", "Reparaturen Komponente X"),
                        ("readout_summary", "Betrieb", "Auslesungen je Fahrzeug"),
                        ("readouts_sample", "Betrieb", "Auslesungen (Stichprobe 400 Fahrzeuge)")):
        schemas[t], labels[t] = sch, lab
        fks.append({"from_table": t, "from_columns": ["vehicle_id"], "to_table": "vehicles", "to_columns": ["vehicle_id"]})
    return save_tables("scania", tables, {
        "schemas": schemas, "labels": labels, "foreign_keys": fks,
        "primary_keys": {"vehicles": ["vehicle_id"], "repairs": ["repair_id"], "readout_summary": ["vehicle_id"],
                         "readouts_sample": ["readout_id"], **{t: [t.replace("config_", "spec_")] for t in tables
                                                                 if t.startswith("config_")}},
        "local_files": ["data/local/scania_readouts.npz"],
        "counters": counters})


# ── WideWorldImporters (SQL-Server-Container) ─────────────────────────────────

WWI_CONTAINER = "graphready-mssql"
WWI_PW = "GraphReady!Local1"          # nur lokaler Wegwerf-Container ohne Netzfreigabe nach außen


def _sql(q: str, db: str = "master", fmt: bool = False) -> str:
    cmd = ["docker", "exec", WWI_CONTAINER, "/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", WWI_PW,
           "-C", "-d", db, "-b", "-Q", q] + (["-h", "-1", "-W", "-s", "\t"] if fmt else [])
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def prepare_wwi(log=print) -> dict:
    bak = (RAW_DIR / "wwi" / "WideWorldImporters-Full.bak").resolve()
    subprocess.run(["docker", "rm", "-f", WWI_CONTAINER], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", WWI_CONTAINER, "-e", "ACCEPT_EULA=Y", "-e", f"MSSQL_SA_PASSWORD={WWI_PW}",
                    "-e", "MSSQL_PID=Developer", "-v", f"{bak.parent}:/backup", "mcr.microsoft.com/mssql/server:2022-latest"],
                   check=True, capture_output=True)
    try:
        for _ in range(60):
            try:
                _sql("SELECT 1")
                break
            except subprocess.CalledProcessError:
                time.sleep(3)
        files = _sql("SET NOCOUNT ON; RESTORE FILELISTONLY FROM DISK='/backup/WideWorldImporters-Full.bak'", fmt=True)
        moves = []
        for line in files.strip().splitlines():
            logical = line.split("\t")[0].strip()
            if logical:
                moves.append(f"MOVE '{logical}' TO '/var/opt/mssql/data/{logical}'")
        _sql("RESTORE DATABASE WWI FROM DISK='/backup/WideWorldImporters-Full.bak' WITH " + ", ".join(moves) + ", REPLACE")
        log("[wwi] Datenbank wiederhergestellt")
        tl = _sql("SET NOCOUNT ON; SELECT s.name+'.'+t.name FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id "
                  "WHERE t.temporal_type <> 1 AND s.name <> 'Integration' ORDER BY 1", db="WWI", fmt=True)
        tables, schemas = {}, {}
        for full in [x.strip() for x in tl.splitlines() if x.strip()]:
            sch, name = full.split(".")
            cols = _sql(f"SET NOCOUNT ON; SELECT c.name, ty.name FROM sys.columns c JOIN sys.types ty ON ty.user_type_id=c.user_type_id "
                        f"WHERE c.object_id=OBJECT_ID('{full}') AND c.is_computed=0 AND ty.name NOT IN "
                        f"('varbinary','geography','image','xml','sysname','timestamp') "
                        f"AND c.name NOT IN ('LastEditedBy','LastEditedWhen','ValidFrom','ValidTo') "
                        f"AND c.name NOT LIKE '%JSON%' AND c.name NOT LIKE 'Photo' "
                        f"AND c.name NOT IN ('ReturnedDeliveryData','FullSensorData') ORDER BY c.column_id", db="WWI", fmt=True)
            cl = [ln.split("\t") for ln in cols.strip().splitlines() if ln.strip()]
            sel = ", ".join(f"REPLACE(REPLACE(CONVERT(nvarchar(4000), [{c}], 126), CHAR(9), ' '), CHAR(10), ' ')" for c, _ in cl)
            out = subprocess.run(["docker", "exec", WWI_CONTAINER, "/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa",
                                  "-P", WWI_PW, "-C", "-d", "WWI", "-b", "-h", "-1", "-W", "-s", "\t", "-Q",
                                  f"SET NOCOUNT ON; SELECT {sel} FROM {full}"], capture_output=True, text=True)
            if out.returncode:
                raise RuntimeError(f"Export {full}: {out.stdout[-300:]} {out.stderr[-300:]}")
            out = out.stdout
            recs = [ln.split("\t") for ln in out.splitlines() if ln != ""]
            df = pd.DataFrame(recs, columns=[c for c, _ in cl]).replace({"NULL": None, "": None})
            for c, ty in cl:
                if ty in ("int", "bigint", "smallint", "tinyint", "bit", "decimal", "numeric", "money", "float", "real"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                elif ty in ("date", "datetime", "datetime2", "smalldatetime"):
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            tables[name] = df
            schemas[name] = sch
            log(f"[wwi] {full}: {len(df):,} Zeilen, {df.shape[1]} Spalten")
        fk = _sql("SET NOCOUNT ON; SELECT OBJECT_NAME(f.parent_object_id), cp.name, OBJECT_NAME(f.referenced_object_id), cr.name "
                  "FROM sys.foreign_key_columns f JOIN sys.columns cp ON cp.object_id=f.parent_object_id AND cp.column_id=f.parent_column_id "
                  "JOIN sys.columns cr ON cr.object_id=f.referenced_object_id AND cr.column_id=f.referenced_column_id",
                  db="WWI", fmt=True)
        fks = []
        for ln in fk.strip().splitlines():
            a, ca, b, cb = [x.strip() for x in ln.split("\t")]
            if a in tables and b in tables and ca in tables[a] and cb in tables[b] and ca not in ("LastEditedBy",):
                fks.append({"from_table": a, "from_columns": [ca], "to_table": b, "to_columns": [cb]})
        pk = _sql("SET NOCOUNT ON; SELECT t.name, c.name FROM sys.indexes i JOIN sys.index_columns ic ON ic.object_id=i.object_id "
                  "AND ic.index_id=i.index_id JOIN sys.columns c ON c.object_id=ic.object_id AND c.column_id=ic.column_id "
                  "JOIN sys.tables t ON t.object_id=i.object_id WHERE i.is_primary_key=1", db="WWI", fmt=True)
        pks: dict[str, list] = {}
        for ln in pk.strip().splitlines():
            t, c = [x.strip() for x in ln.split("\t")]
            if t in tables:
                pks.setdefault(t, []).append(c)
        return save_tables("wwi", tables, {"schemas": schemas, "foreign_keys": fks, "primary_keys": pks})
    finally:
        subprocess.run(["docker", "rm", "-f", WWI_CONTAINER], capture_output=True)


PREPARE = {"adventureworks_data": prepare_adventureworks_data, "cmapss": prepare_cmapss, "scania": prepare_scania,
           "wwi": prepare_wwi}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=",".join(PREPARE))
    args = ap.parse_args(argv)
    for ds in args.only.split(","):
        t0 = time.time()
        meta = PREPARE[ds](log=lambda m: print(m, flush=True))
        size = (DATA_DIR / f"{ds}.npz").stat().st_size / 1e6
        print(f"[connectors] {ds}: {len(meta['tables'])} Tabellen, {sum(meta['rows'].values()):,} Zeilen, "
              f"{size:.1f} MB in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ── Adapter: WideWorldImporters im Positionsformat von Online Retail II ───────
# So laufen Merkmale, Leakage-Guard, Graph-Merkmale und alle Engines aus backtest.py unverändert auf WWI.

WWI_USD_EUR = 0.92


def wwi_as_retail() -> pd.DataFrame:
    t = load_tables("wwi")
    inv, ln, cu = t["Invoices"], t["InvoiceLines"], t["Customers"]
    city = t["Cities"].set_index("CityID")["StateProvinceID"]
    state = t["StateProvinces"].set_index("StateProvinceID")["StateProvinceName"].astype(str)
    bg = t["BuyingGroups"].set_index("BuyingGroupID")["BuyingGroupName"].astype(str)
    cat = t["CustomerCategories"].set_index("CustomerCategoryID")["CustomerCategoryName"].astype(str)
    sg = t["StockItemStockGroups"].sort_values("StockGroupID").drop_duplicates("StockItemID").set_index("StockItemID")["StockGroupID"]
    sgn = t["StockGroups"].set_index("StockGroupID")["StockGroupName"].astype(str)
    c = cu.set_index("CustomerID")
    df = ln.merge(inv[["InvoiceID", "CustomerID", "InvoiceDate"]], on="InvoiceID", how="left")
    cid = df["CustomerID"].astype(int)
    out = pd.DataFrame({
        "invoice": df["InvoiceID"].astype(int).astype(str).astype("category"),
        "stock_code": df["StockItemID"].astype(int).astype(str).astype("category"),
        "description": df["Description"].astype(str).astype("category"),
        "quantity": df["Quantity"].astype(np.int32),
        "invoice_date": pd.to_datetime(df["InvoiceDate"]),
        "price_eur": (df["UnitPrice"].astype(float) * WWI_USD_EUR).astype(np.float32),
        "customer_id": cid.astype("Int32"),
        "country": cid.map(c["DeliveryCityID"]).map(city).map(state).fillna("unbekannt").astype("category"),
        "pb05_group": df["StockItemID"].map(sg).map(sgn).fillna("Sonstiges").astype("category"),
        "buying_group": cid.map(c["BuyingGroupID"]).map(bg).fillna("unabhängig").astype("category"),
        "customer_category": cid.map(c["CustomerCategoryID"]).map(cat).fillna("unbekannt").astype("category"),
        "credit_limit": cid.map(c["CreditLimit"]).astype(float).astype(np.float32),
    })
    out["price_gbp"] = out["price_eur"]                  # Spaltenname der Rohdaten; Währung steht in price_eur (EUR)
    return out.sort_values("invoice_date", kind="stable").reset_index(drop=True)


# ── Adapter: AdventureWorks im Positionsformat (Vertriebsabdeckung & Gebiete) ─

AW_USD_EUR = 0.92


def aw_as_retail() -> pd.DataFrame:
    """Auftragspositionen mit Gebiet (country), Produktkategorie (pb05_group), Vertriebler-Portfolio (buying_group)
    und Kundentyp Händler/Endkunde (customer_category) – Händler-Vertriebler kommt aus Store.SalesPersonID."""
    t = load_tables("adventureworks_data")
    h, d, cu, st = t["SalesOrderHeader"], t["SalesOrderDetail"], t["Customer"], t["Store"]
    ter = t["SalesTerritory"].set_index("TerritoryID")["Name"].astype(str)
    prod = t["Product"].set_index("ProductID")
    sub = t["ProductSubcategory"].set_index("ProductSubcategoryID")
    cat = t["ProductCategory"].set_index("ProductCategoryID")["Name"].astype(str)
    pcat = prod["ProductSubcategoryID"].map(sub["ProductCategoryID"]).map(cat)
    c = cu.set_index("CustomerID")
    store_sp = c["StoreID"].map(st.set_index("BusinessEntityID")["SalesPersonID"])
    df = d.merge(h[["SalesOrderID", "CustomerID", "OrderDate", "TerritoryID"]], on="SalesOrderID", how="left")
    cid = df["CustomerID"].astype(int)
    is_store = cid.map(c["StoreID"]).notna()
    sp = cid.map(store_sp)
    out = pd.DataFrame({
        "invoice": df["SalesOrderID"].astype(int).astype(str).astype("category"),
        "stock_code": df["ProductID"].astype(int).astype(str).astype("category"),
        "description": df["ProductID"].map(prod["Name"]).astype(str).astype("category"),
        "quantity": df["OrderQty"].astype(np.int32),
        "invoice_date": pd.to_datetime(df["OrderDate"]),
        "price_eur": (df["UnitPrice"].astype(float) * (1 - df["UnitPriceDiscount"].astype(float)) * AW_USD_EUR).astype(np.float32),
        "customer_id": cid.astype("Int32"),
        "country": df["TerritoryID"].map(ter).fillna("unbekannt").astype("category"),
        "pb05_group": df["ProductID"].map(pcat).fillna("Sonstiges").astype("category"),
        "buying_group": np.where(sp.notna(), "Vertriebler " + sp.fillna(0).astype(int).astype(str), "ohne Vertriebler"),
        "customer_category": np.where(is_store, "Händler", "Endkunde"),
    })
    out["buying_group"] = out["buying_group"].astype("category")
    out["customer_category"] = out["customer_category"].astype("category")
    out["price_gbp"] = out["price_eur"]
    return out.sort_values("invoice_date", kind="stable").reset_index(drop=True)
