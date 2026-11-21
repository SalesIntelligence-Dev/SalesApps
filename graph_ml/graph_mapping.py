"""
Graph-Mapping (Modul 2)
=======================
Liest eine relationale Datenbank als zeitlichen, heterogenen Graphen
(Relational Deep Learning, Fey et al.): jede Tabelle ein Knotentyp, jede
Zeile ein Knoten, jede Primär-/Fremdschlüssel-Beziehung ein Kantentyp.

Erkennt automatisch
  - Primärschlüssel-Kandidaten (eindeutig, nicht leer)
  - Fremdschlüssel per Namensähnlichkeit + Werte-Inklusion
  - Zeitspalten, Kategorien mit sehr vielen Ausprägungen, Freitextspalten
und misst je Beziehung Verknüpfungsquote, verwaiste und leere Verweise sowie
die Abdeckung über mehrere Schritte (z. B. Position → Rechnung → Kunde).

Zwei Modi:
  build_mapping(tables)         – mit Daten (Stufe 2/3)
  build_schema_mapping(schema)  – nur Tabellen-/Feldliste (Stufe 1)
"""
from __future__ import annotations

import re
import time
from collections import deque

import numpy as np
import pandas as pd

ARTIFACT_VERSION = 6

ID_SUFFIXES = ("id", "nr", "no", "nummer", "number", "code", "key", "schluessel")
TIME_HINTS  = ("date", "datum", "time", "zeit", "timestamp", "created", "modified")

HIGH_CARD      = 1000      # ab so vielen Ausprägungen: „sehr viele Kategorien“
LINK_OK, LINK_WARN = 0.95, 0.70
MAX_FK_SAMPLE  = 500_000   # Upload-Tabellen darüber: Inklusion auf Stichprobe


# ── Namens-Heuristik ──────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _strip_id(n: str) -> str:
    for suf in ID_SUFFIXES:
        if n.endswith(suf) and len(n) > len(suf) + 1:
            return n[: -len(suf)]
    return n


def _singular(n: str) -> str:
    if n.endswith("ies"):
        return n[:-3] + "y"
    if n.endswith(("sses", "xes", "ches", "shes")):
        return n[:-2]
    if n.endswith("s") and not n.endswith("ss"):
        return n[:-1]
    return n


def _table_stem(table: str) -> str:
    return _singular(_norm(table.split(".")[-1]))


def name_match(fk_col: str, pk_col: str, target_table: str) -> float:
    """0–1: Wie plausibel verweist fk_col auf target_table.pk_col?"""
    a, b, t = _norm(fk_col), _norm(pk_col), _table_stem(target_table)
    if a == b:
        return 1.0
    if a == t or _strip_id(a) == t:
        return 0.9
    if _strip_id(a) and _strip_id(a) == _strip_id(b):
        return 0.85
    if len(t) >= 4 and t in a and _strip_id(b) in a:
        return 0.7
    return 0.0


def _id_like(col: str) -> bool:
    n = _norm(col)
    return any(n.endswith(s) for s in ID_SUFFIXES) or n in ("invoice", "beleg", "kunde", "artikel")


# ── Spaltenprofil ─────────────────────────────────────────────────────────────

def _kind(s: pd.Series) -> str:
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return _kind(pd.Series(dt.categories))
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime"
    if pd.api.types.is_integer_dtype(dt):
        return "int"
    if pd.api.types.is_float_dtype(dt):
        v = s.dropna()
        v = v.iloc[:50_000] if len(v) > 50_000 else v
        return "int" if len(v) and bool((v % 1 == 0).all()) else "float"
    return "string"


def _string_stats(s: pd.Series) -> tuple[int, float, float, list[str]]:
    """n_unique, Ø-Länge, Anteil mit Leerzeichen, Beispiele – gewichtet nach Häufigkeit."""
    vc = s.value_counts(dropna=True)
    vc = vc[vc > 0]
    if vc.empty:
        return 0, 0.0, 0.0, []
    vals = pd.Series(vc.index.astype(str))
    w = vc.to_numpy(dtype=float)
    lens = vals.str.len().to_numpy(dtype=float)
    spaces = vals.str.contains(" ", regex=False).to_numpy(dtype=float)
    return (int(len(vc)), float((lens * w).sum() / w.sum()),
            float((spaces * w).sum() / w.sum()), [str(v) for v in vals[:3]])


def profile_column(s: pd.Series) -> dict:
    n = len(s)
    nulls = int(s.isna().sum())
    kind = _kind(s)
    p = {"kind": kind, "rows": n, "null_rows": nulls,
         "null_rate": round(nulls / n, 4) if n else 0.0}
    if kind == "string":
        n_unique, avg_len, space_share, examples = _string_stats(s)
        p.update(n_unique=n_unique, avg_len=round(avg_len, 1),
                 space_share=round(space_share, 3), examples=examples)
    else:
        v = s.dropna()
        p["n_unique"] = int(v.nunique())
        if kind == "datetime" and len(v):
            p["min"], p["max"] = v.min().isoformat(), v.max().isoformat()
            p["examples"] = [v.iloc[0].isoformat()]
        elif kind in ("int", "float") and len(v):
            p["min"], p["max"] = float(v.min()), float(v.max())
            p["examples"] = [_scalar(x) for x in v.iloc[:3]]
        else:
            p["examples"] = [_scalar(x) for x in v.iloc[:3]]
    nonnull = n - nulls
    p["unique_ratio"] = round(p["n_unique"] / nonnull, 4) if nonnull else 0.0
    return p


def _scalar(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x if isinstance(x, (int, float, str, bool)) or x is None else str(x)


def _assign_roles(col: str, p: dict, pk: str | None, fk_cols: set[str]) -> list[str]:
    roles = []
    if col == pk:
        roles.append("PK")
    if col in fk_cols:
        roles.append("FK")
    k = p["kind"]
    if k == "datetime":
        roles.append("Zeit")
    elif k == "string":
        if p.get("avg_len", 0) >= 15 and p.get("space_share", 0) >= 0.5 and p.get("n_unique", 0) >= 50:
            roles.append("Freitext")
        elif p.get("n_unique", 0) > HIGH_CARD and "PK" not in roles:
            roles.append("Hohe Kardinalität")
        elif not roles:
            roles.append("Kategorie")
    elif k == "bool":
        roles.append("Flag")
    elif k in ("int", "float"):
        if not roles:
            roles.append("Schlüssel-Kandidat" if _id_like(col) and p.get("unique_ratio", 0) > 0.5 else "Kennzahl")
        elif p.get("n_unique", 0) > HIGH_CARD and "PK" not in roles:
            roles.append("Hohe Kardinalität")
    return roles


# ── Primärschlüssel ───────────────────────────────────────────────────────────

def detect_pk(table: str, df: pd.DataFrame, profiles: dict) -> str | None:
    stem = _table_stem(table)
    best, best_score = None, 0.0
    for i, col in enumerate(df.columns):
        p = profiles[col]
        if p["null_rows"] or p["kind"] in ("float", "datetime", "bool") or p["rows"] == 0:
            continue
        if p["n_unique"] != p["rows"]:
            continue
        score = 1.0
        n = _norm(col)
        if _strip_id(n) == stem or n == stem:
            score += 2
        if _id_like(col):
            score += 1
        score += 0.3 if i == 0 else 0
        if score > best_score:
            best, best_score = col, score
    return best


# ── Hilfen für Verknüpfung über Kategorien/Ints ───────────────────────────────

def _key_values(s: pd.Series) -> pd.Series:
    """Vergleichbare Schlüsselwerte: Ganzzahlen als float64 (NaN = leer), sonst str/None.
    Kategorien werden über ihre Codes aufgelöst – keine Kopie von 1 Mio. Strings."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = _key_values(pd.Series(s.cat.categories)).to_numpy()
        codes = s.cat.codes.to_numpy()
        fill = np.nan if cats.dtype.kind == "f" else None
        out = np.where(codes >= 0, cats[np.clip(codes, 0, None)], fill)
        return pd.Series(out, index=s.index, dtype=cats.dtype)
    if _kind(s) == "int":
        return pd.Series(pd.to_numeric(s, errors="coerce").astype("float64").to_numpy(na_value=np.nan),
                         index=s.index)
    if s.dtype == object and len(s) and all(isinstance(v, (int, float, np.floating, np.integer))
                                            for v in s.dropna().iloc[:20]):
        return pd.Series(pd.to_numeric(s, errors="coerce").astype("float64"), index=s.index)
    return s.astype(object).where(s.notna(), None).map(lambda v: v if v is None else str(v))


def _unique_keys(s: pd.Series) -> pd.Index:
    if isinstance(s.dtype, pd.CategoricalDtype):
        used = s.cat.remove_unused_categories().cat.categories
        return pd.Index(_key_values(pd.Series(used)).dropna().unique())
    return pd.Index(_key_values(s.dropna().drop_duplicates()).dropna().unique())


# ── Mapping mit Daten ─────────────────────────────────────────────────────────

def build_mapping(tables: dict[str, pd.DataFrame], *, dataset_id: str, extra: dict | None = None,
                  sample_note: str | None = None) -> dict:
    t0 = time.time()
    extra = extra or {}
    profiles = {t: {c: profile_column(df[c]) for c in df.columns} for t, df in tables.items()}
    pks = {t: detect_pk(t, df, profiles[t]) for t, df in tables.items()}
    for t, pk in (extra.get("declared_pks") or {}).items():
        pks[t] = pk

    key_cache: dict[tuple[str, str], pd.Index] = {}

    def keys(t: str, c: str) -> pd.Index:
        if (t, c) not in key_cache:
            key_cache[(t, c)] = _unique_keys(tables[t][c])
        return key_cache[(t, c)]

    # Fremdschlüssel: Namensähnlichkeit als Vorfilter, Werte-Inklusion als Beweis
    relations = []
    for a, dfa in tables.items():
        for col in dfa.columns:
            pa = profiles[a][col]
            if pa["kind"] in ("float", "datetime", "bool") or (col == pks[a] and _is_surrogate(a, col)):
                continue
            best = None
            for b, pkb in pks.items():
                if not pkb or (b == a and col == pkb):
                    continue
                score = name_match(col, pkb, b)
                if score < 0.6:
                    continue
                if (pa["kind"] == "string") != (profiles[b][pkb]["kind"] == "string"):
                    continue
                va = keys(a, col)
                if not len(va):
                    continue
                incl = float(va.isin(keys(b, pkb)).mean())
                if incl < 0.5 or (score < 0.85 and incl < 0.9):
                    continue
                if best is None or (score, incl) > (best[2], best[3]):
                    best = (b, pkb, score, incl)
            if best:
                b, pkb, score, incl = best
                relations.append(_measure_relation(a, col, b, pkb, tables, score, incl, sample_note))

    for rel in extra.get("missing_relations", []):
        relations.append({**rel, "kind": "missing", "status": "bad", "linkage_rate": 0.0})

    fk_cols = {(r["from_table"], r["from_column"]) for r in relations if r["kind"] == "fk"}
    table_out = []
    for t, df in tables.items():
        cols = []
        for c in df.columns:
            p = profiles[t][c]
            cols.append({"name": c, **p, "roles": _assign_roles(c, p, pks[t], {fc for ft, fc in fk_cols if ft == t})})
        time_cols = [c["name"] for c in cols if "Zeit" in c["roles"]]
        table_out.append({
            "name": t, "label": extra.get("labels", {}).get(t, t), "rows": int(len(df)),
            "n_columns": int(df.shape[1]), "primary_key": pks[t], "time_columns": time_cols,
            "columns": cols,
        })

    paths = _path_coverage(tables, relations, pks)
    mapping = {
        "artifact_version": ARTIFACT_VERSION,
        "dataset_id": dataset_id,
        "mode": "data",
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "compute_seconds": None,
        "tables": table_out,
        "relations": relations,
        "paths": paths,
    }
    mapping["derived_columns"] = list(extra.get("derived_columns", []))
    mapping["graph"] = _graph_summary(mapping)
    mapping["findings"] = _findings(mapping) + list(extra.get("findings", []))
    mapping["insights"] = _insights(mapping, extra.get("insight_hints", []))
    mapping["summary"] = _summary(mapping)
    mapping["compute_seconds"] = round(time.time() - t0, 2)
    return mapping


def _is_surrogate(table: str, col: str) -> bool:
    return _strip_id(_norm(col)) in ("line", "row", "zeile", "position", "pos")


def _measure_relation(a, col, b, pkb, tables, score, incl, sample_note) -> dict:
    s = tables[a][col]
    n = len(s)
    if n > MAX_FK_SAMPLE:
        s = s.sample(MAX_FK_SAMPLE, random_state=0) if sample_note else s
    vals = _key_values(s)
    null_rows = int(vals.isna().sum())
    found = vals.isin(set(_unique_keys(tables[b][pkb])))
    linked = int(found.sum())
    m = len(vals)
    rate = linked / m if m else 0.0
    note = None
    if a == b:
        # Selbstbezug (z. B. Storno → Originalrechnung): nur ein Teil der Zeilen trägt ihn – Quote über befüllte Zeilen
        filled = m - null_rows
        note = f"nur {_pct(filled / m if m else 0)} der Zeilen tragen den Bezug (z. B. Storno-Belege)"
        rate = linked / filled if filled else 0.0
    return {
        "id": f"{a}.{col}->{b}.{pkb}",
        "kind": "fk",
        "from_table": a, "from_column": col, "to_table": b, "to_column": pkb,
        "name_score": round(score, 2), "inclusion": round(incl, 4),
        "rows": n, "linked_rows": int(round(linked * n / m)) if m else 0,
        "null_rows": int(round(null_rows * n / m)) if m else 0,
        "orphan_rows": int(round((m - linked - null_rows) * n / m)) if m else 0,
        "linkage_rate": round(rate, 4),
        "status": "ok" if rate >= LINK_OK else "warn" if rate >= LINK_WARN else "bad",
        "sampled": n > MAX_FK_SAMPLE and bool(sample_note),
        **({"note": note, "self": True} if note else {}),
    }


def _map_through(src: pd.Series, target: pd.DataFrame, key: str, value: str) -> pd.Series:
    """src (Schlüssel auf target.key) → Werte von target.value, NaN wenn nicht gefunden."""
    lut = pd.Series(_key_values(target[value]).to_numpy(), index=_key_values(target[key]).to_numpy())
    lut = lut[~lut.index.duplicated()]
    if isinstance(src.dtype, pd.CategoricalDtype):
        cats = _key_values(pd.Series(src.cat.categories)).to_numpy()
        mapped = lut.reindex(cats).to_numpy(dtype=object)
        codes = src.cat.codes.to_numpy()
        return pd.Series(np.where(codes >= 0, mapped[np.clip(codes, 0, None)], None), dtype=object)
    return pd.Series(_key_values(src).map(lut).to_numpy(), dtype=object)


def _path_coverage(tables, relations, pks, max_hops: int = 3) -> list[dict]:
    """Anteil Zeilen der größten Ereignistabellen, die über FK-Ketten jede Entität erreichen
    (z. B. Position → Rechnung → Kunde)."""
    out_edges: dict[str, list[dict]] = {}
    for r in relations:
        if r["kind"] == "fk":
            out_edges.setdefault(r["from_table"], []).append(r)
    sources = sorted((t for t in tables if out_edges.get(t)), key=lambda t: -len(tables[t]))[:2]
    results = []
    for src in sources:
        # frontier: (Tabelle, Schlüsselspalte, Schlüsselwerte je Quellzeile | None für die Quelle selbst)
        frontier = deque([(src, None, None, [src])])
        seen = {src}
        while frontier:
            table, key_col, key_vals, via = frontier.popleft()
            if len(via) > max_hops:
                continue
            for r in out_edges.get(table, []):
                nxt = r["to_table"]
                if nxt in seen:
                    continue
                if key_vals is None:
                    fk_vals = _key_values(tables[table][r["from_column"]]).reset_index(drop=True)
                else:
                    fk_vals = _map_through(key_vals, tables[table], key_col, r["from_column"])
                ok = fk_vals.notna() & fk_vals.isin(set(_unique_keys(tables[nxt][r["to_column"]])))
                cov = float(ok.mean()) if len(ok) else 0.0
                results.append({"from_table": src, "to_table": nxt, "via": via + [nxt],
                                "hops": len(via), "coverage": round(cov, 4),
                                "status": "ok" if cov >= LINK_OK else "warn" if cov >= LINK_WARN else "bad"})
                seen.add(nxt)
                frontier.append((nxt, r["to_column"], fk_vals.where(ok, None), via + [nxt]))
    return results


def _graph_summary(m: dict) -> dict:
    fks = [r for r in m["relations"] if r["kind"] in ("fk", "declared")]
    n_nodes = sum(t["rows"] for t in m["tables"])
    n_edges = sum(r.get("linked_rows", 0) for r in fks)
    return {
        "node_types": [{"name": t["name"], "label": t.get("label", t["name"]), "rows": t["rows"],
                        "has_time": bool(t["time_columns"])} for t in m["tables"]],
        "edge_types": [{"id": r["id"], "from": r["from_table"], "to": r["to_table"],
                        "column": r["from_column"], "rows": r.get("linked_rows")} for r in fks],
        "n_nodes": n_nodes, "n_edges": n_edges,
    }


def _summary(m: dict) -> dict:
    fks = [r for r in m["relations"] if r["kind"] == "fk"]
    rates = [r["linkage_rate"] for r in fks]
    return {
        "tables": len(m["tables"]),
        "columns": sum(t["n_columns"] for t in m["tables"]),
        "relations": len(fks) + len([r for r in m["relations"] if r["kind"] == "declared"]),
        "missing_relations": len([r for r in m["relations"] if r["kind"] == "missing"]),
        "avg_linkage": round(float(np.mean(rates)), 4) if rates else None,
        "min_linkage": round(float(np.min(rates)), 4) if rates else None,
        "time_tables": len([t for t in m["tables"] if t["time_columns"]]),
        "free_text_columns": sum(1 for t in m["tables"] for c in t["columns"] if "Freitext" in c["roles"]),
        "high_card_columns": sum(1 for t in m["tables"] for c in t["columns"] if "Hohe Kardinalität" in c["roles"]),
        "findings": {s: len([f for f in m["findings"] if f["severity"] == s]) for s in ("bad", "warn", "info")},
    }


def _pct(x: float) -> str:
    return f"{x * 100:.1f} %".replace(".", ",")


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _findings(m: dict) -> list[dict]:
    out = []
    labels = {t["name"]: t.get("label", t["name"]) for t in m["tables"]}
    for r in m["relations"]:
        if r["kind"] != "fk" or r["status"] == "ok" or r.get("self"):
            continue
        miss = 1 - r["linkage_rate"]
        out.append({
            "severity": r["status"], "check": "Struktur & Schlüssel", "playbook": None,
            "title": f"{_pct(miss)} der {labels[r['from_table']]} ohne {labels[r['to_table']]}",
            "text": f"Spalte {r['from_column']}: {_n(r['null_rows'])} leere und {_n(r['orphan_rows'])} verwaiste Verweise.",
        })
    for p in m["paths"]:
        if p["status"] != "ok" and p["hops"] > 1:
            out.append({
                "severity": p["status"], "check": "Struktur & Schlüssel", "playbook": None,
                "title": f"Pfad {' → '.join(labels[t] for t in p['via'])}: {_pct(p['coverage'])} Abdeckung",
                "text": "Zeilen, die die Ziel-Entität nicht erreichen, fehlen allen Use Cases auf dieser Entität.",
            })
    for t in m["tables"]:
        if not t["primary_key"]:
            out.append({"severity": "warn", "check": "Struktur & Schlüssel", "playbook": None,
                        "title": f"{t['label']}: kein eindeutiger Schlüssel",
                        "text": "Ohne Primärschlüssel lassen sich Zeilen nicht als Knoten adressieren."})
        for c in t["columns"]:
            if "Freitext" in c["roles"]:
                out.append({"severity": "info", "check": "Freitext", "playbook": "PB-05",
                            "title": f"Freitext in {t['label']}.{c['name']}",
                            "text": (f"Ø {c.get('avg_len', 0):.0f} Zeichen, {_n(c['n_unique'])} Varianten – relationale "
                                     "Modelle lesen Freitext oft nicht; Überführung in Kategorien empfohlen.")})
            elif "Hohe Kardinalität" in c["roles"] and "FK" not in c["roles"]:
                out.append({"severity": "info", "check": "Spaltenqualität", "playbook": None,
                            "title": f"Sehr viele Ausprägungen: {t['label']}.{c['name']}",
                            "text": f"{_n(c['n_unique'])} Werte – besser als eigener Knotentyp im Graphen statt One-Hot-Spalte."})
    linked = {r["from_table"] for r in m["relations"]} | {r["to_table"] for r in m["relations"]}
    for t in m["tables"]:
        if t["name"] not in linked and len(m["tables"]) > 1:
            out.append({"severity": "warn", "check": "Struktur & Schlüssel", "playbook": None,
                        "title": f"{t['label']} ist mit keiner Tabelle verknüpft",
                        "text": "Isolierte Tabellen tragen kein Signal über mehrere Schritte."})
    return out


def _insights(m: dict, hints: list[dict]) -> list[dict]:
    g = m["graph"]
    labels = {t["name"]: t.get("label", t["name"]) for t in m["tables"]}
    out = [{
        "icon": "graph",
        "title": "Ihre Datenbank als Graph",
        "text": (f"{len(g['node_types'])} Knotentypen und {len(g['edge_types'])} Kantentypen – "
                 f"{_num(g['n_nodes'])} Knoten und {_num(g['n_edges'])} Kanten. Jede Zeile wird ein Knoten, "
                 "jede Schlüsselbeziehung eine Kante."),
    }]
    chain = _bridge_chain(m["paths"])
    if chain:
        out.append({
            "icon": "hops",
            "title": "Signal über mehrere Schritte",
            "text": (f"{' — '.join(labels[t] for t in chain)}: Über diese Kette erreicht man vom Kunden "
                     "seine Artikel und von dort ähnliche Kunden (Co-Purchase). Graph ML nutzt solche "
                     "Mehrsprung-Signale, ein flaches Tabellenmodell sieht sie nicht."),
        })
    high = [(t["label"], c["name"], c["n_unique"]) for t in m["tables"] for c in t["columns"]
            if "Hohe Kardinalität" in c["roles"] and "FK" in c["roles"]]
    if high:
        out.append({
            "icon": "nodes",
            "title": "Viele Ausprägungen werden Knoten",
            "text": ("; ".join(f"{t}.{c} ({_n(n)})" for t, c, n in high[:3]) + " – für Baum-Modelle müssten "
                     "diese Schlüssel aufwendig kodiert werden, im Graphen sind sie natürliche Knoten."),
        })
    gaps = [p for p in m["paths"] if p["status"] != "ok"]
    if gaps:
        p = min(gaps, key=lambda p: p["coverage"])
        out.append({
            "icon": "gap",
            "title": "Lücken im Graphen",
            "text": (f"Nur {_pct(p['coverage'])} der {labels[p['from_table']]} erreichen "
                     f"{labels[p['to_table']]}. Für Use Cases auf dieser Entität fehlen die übrigen Zeilen."),
        })
    time_tables = [labels[t["name"]] for t in m["tables"] if t["time_columns"]]
    if time_tables:
        out.append({
            "icon": "time",
            "title": "Zeitstempel vorhanden",
            "text": (f"Ereigniszeit in {', '.join(time_tables)} – zeitlicher Split und Labels wie "
                     "„kein Auftrag in 90 Tagen“ sind berechenbar (Schutz vor Leakage)."),
        })
    else:
        out.append({"icon": "time", "title": "Keine Zeitstempel",
                    "text": "Ohne Ereigniszeit ist kein zeitlicher Split möglich – Rückblick (Stufe 3) entfällt."})
    return out + hints


def _bridge_chain(paths: list[dict]) -> list[str] | None:
    """Längste Kette über eine Brückentabelle: Ziel A ← … ← Quelle → … → Ziel B."""
    by_src: dict[str, list[dict]] = {}
    for p in paths:
        by_src.setdefault(p["from_table"], []).append(p)
    best = None
    for src, ps in by_src.items():
        ps = sorted(ps, key=lambda p: -p["hops"])
        for a in ps:
            for b in ps:
                if a is b or a["via"][1] == b["via"][1]:
                    continue
                chain = list(reversed(a["via"])) + b["via"][1:]
                if best is None or len(chain) > len(best):
                    best = chain
    return best if best and len(best) >= 4 else None


def _num(n: int | float | None) -> str:
    if n is None:
        return "–"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}".replace(".", ",") + " Mio."
    return f"{int(n):,}".replace(",", ".")


# ── Mapping nur aus Schema (Stufe 1) ──────────────────────────────────────────

def build_schema_mapping(schema: dict, *, dataset_id: str) -> dict:
    t0 = time.time()
    tables = schema["tables"]
    pks = {t["name"]: (t.get("primary_key") or [None])[0] for t in tables}
    for t in tables:
        if not pks[t["name"]]:
            stem = _table_stem(t["name"])
            for c in t["columns"]:
                if _strip_id(_norm(c["name"])) == stem and _id_like(c["name"]):
                    pks[t["name"]] = c["name"]
                    break
        if not pks[t["name"]] and t["columns"]:
            # ERP-Konvention (z. B. SAP VBAK.VBELN, KNA1.KUNNR): Schlüsselfeld steht vorn
            pks[t["name"]] = t["columns"][0]["name"]

    relations = []
    for fk in schema.get("foreign_keys", []):
        relations.append({
            "id": f"{fk['from_table']}.{','.join(fk['from_columns'])}->{fk['to_table']}",
            "kind": "declared", "from_table": fk["from_table"], "from_column": ",".join(fk["from_columns"]),
            "to_table": fk["to_table"], "to_column": ",".join(fk["to_columns"]),
            "status": "unknown", "linkage_rate": None, "name": fk.get("name"),
        })
    if not schema.get("foreign_keys"):
        # Kein FK deklariert → per Namensähnlichkeit vorschlagen
        for t in tables:
            for c in t["columns"]:
                if c["name"] == pks[t["name"]]:
                    continue
                for b, pkb in pks.items():
                    if b == t["name"] or not pkb:
                        continue
                    if name_match(c["name"], pkb, b) >= 0.85:
                        relations.append({
                            "id": f"{t['name']}.{c['name']}->{b}", "kind": "declared",
                            "from_table": t["name"], "from_column": c["name"], "to_table": b,
                            "to_column": pkb, "status": "unknown", "linkage_rate": None,
                            "name": "per Namensähnlichkeit erkannt",
                        })
                        break

    fk_cols = {(r["from_table"], c) for r in relations for c in r["from_column"].split(",")}
    table_out = []
    for t in tables:
        cols = []
        for c in t["columns"]:
            typ = c.get("type", "string")
            roles = []
            if c["name"] == pks[t["name"]] or c["name"] in (t.get("primary_key") or []):
                roles.append("PK")
            if (t["name"], c["name"]) in fk_cols:
                roles.append("FK")
            if typ == "datetime":
                roles.append("Zeit")
            elif typ in ("text",) or (typ == "string" and re.search(r"(comment|note|descr|text|notiz|bemerk)",
                                                                   c["name"], re.I)):
                roles.append("Freitext")
            elif typ == "boolean":
                roles.append("Flag")
            elif typ in ("number",):
                roles.append("Kennzahl")
            elif not roles:
                roles.append("Kategorie" if typ == "string" else "Attribut")
            cols.append({"name": c["name"], "kind": typ, "sql_type": c.get("sql_type"),
                         "nullable": c.get("nullable"), "roles": roles})
        time_cols = [c["name"] for c in cols if "Zeit" in c["roles"]]
        table_out.append({"name": t["name"], "label": t["name"], "schema": t.get("schema"),
                          "rows": None, "n_columns": len(cols), "primary_key": pks[t["name"]],
                          "time_columns": time_cols,
                          "only_modified_date": bool(time_cols) and all(
                              re.search(r"modif|geaendert|changed|updated", c, re.I) for c in time_cols),
                          "columns": cols})

    m = {
        "artifact_version": ARTIFACT_VERSION, "dataset_id": dataset_id, "mode": "schema",
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "tables": table_out, "relations": relations, "paths": [],
    }
    m["graph"] = {
        "node_types": [{"name": t["name"], "label": t["name"], "rows": None, "has_time": bool(t["time_columns"]),
                        "schema": t.get("schema")} for t in table_out],
        "edge_types": [{"id": r["id"], "from": r["from_table"], "to": r["to_table"], "column": r["from_column"],
                        "rows": None} for r in relations],
        "n_nodes": None, "n_edges": None,
    }
    only_mod = [t["name"] for t in table_out if t["only_modified_date"]]
    linked = {r["from_table"] for r in relations} | {r["to_table"] for r in relations}
    isolated = [t["name"] for t in table_out if t["name"] not in linked]
    m["findings"] = [
        {"severity": "info", "check": "Struktur & Schlüssel", "playbook": None,
         "title": "Nur Struktur geprüft – Verknüpfungsquoten unbekannt",
         "text": "Stufe 1 sieht nur Tabellen und Felder. Falsches Grün möglich: Felder können existieren, aber leer sein."},
    ]
    if only_mod:
        m["findings"].append({
            "severity": "warn", "check": "Zeit & Leakage", "playbook": None,
            "title": f"{len(only_mod)} Tabellen nur mit Änderungsdatum",
            "text": ("ModifiedDate wird bei jeder Änderung überschrieben – kein Ereignis-Zeitstempel, "
                     "Leakage-Risiko im Rückblick. Beispiele: " + ", ".join(only_mod[:4])),
        })
    if isolated:
        m["findings"].append({
            "severity": "warn", "check": "Struktur & Schlüssel", "playbook": None,
            "title": f"{len(isolated)} Tabellen ohne Beziehung",
            "text": "Ohne Kanten kein Signal über mehrere Schritte: " + ", ".join(isolated[:5]),
        })
    hubs = sorted(((t, sum(1 for r in relations if t in (r["from_table"], r["to_table"])))
                   for t in pks), key=lambda x: -x[1])[:3]
    event_tables = [t["name"] for t in table_out if t["time_columns"] and not t["only_modified_date"]]
    m["insights"] = [
        {"icon": "graph", "title": "Schema als Graph",
         "text": f"{len(table_out)} Knotentypen und {len(relations)} Kantentypen – typisch für ein ERP "
                 "(10 bis 50+ verknüpfte Tabellen)."},
        {"icon": "hops", "title": "Zentrale Entitäten",
         "text": "Die meisten Beziehungen laufen über " + ", ".join(f"{h[0]} ({h[1]})" for h in hubs) + "."},
        {"icon": "time", "title": "Ereignistabellen",
         "text": f"{len(event_tables)} Tabellen mit echter Ereigniszeit (z. B. "
                 + ", ".join(event_tables[:3]) + ") – Basis für Labels und zeitlichen Split."},
        {"icon": "gap", "title": "Was Stufe 1 nicht weiß",
         "text": "Ob die Verknüpfungen gefüllt sind, zeigt erst der Profilcheck (Stufe 2) – ohne Rohdaten."},
    ]
    m["summary"] = {
        "tables": len(table_out), "columns": sum(t["n_columns"] for t in table_out),
        "relations": len(relations), "missing_relations": 0, "avg_linkage": None, "min_linkage": None,
        "time_tables": len([t for t in table_out if t["time_columns"]]),
        "event_tables": len(event_tables),
        "free_text_columns": sum(1 for t in table_out for c in t["columns"] if "Freitext" in c["roles"]),
        "high_card_columns": 0,
        "schemas": sorted({t.get("schema") for t in table_out if t.get("schema")}),
        "findings": {s: len([f for f in m["findings"] if f["severity"] == s]) for s in ("bad", "warn", "info")},
    }
    m["compute_seconds"] = round(time.time() - t0, 2)
    return m
