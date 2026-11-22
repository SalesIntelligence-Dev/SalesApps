"""
Data-Classifier-Analyse (Schritt 3) – welche manuell gepflegten Felder kann ML vorschlagen?
==========================================================================================
  Durchsucht jede Datenbank nach Spalten, die Mitarbeiter von Hand pflegen (Kennzeichen, Klassifikationen,
  Dispositionsparameter) oder die lückenhaft sind, und prüft je Feld, ob es sich aus den übrigen Daten vorhersagen lässt.
  Leitbeispiel: Ist ein Stücklisten-Bauteil ein Ersatzteil (einzeln verkaufsfähig) oder nicht?

  Vorgehen je Datenbank (Muster: Kaskade aus „Ersatzteil-Klassifikation in Kfz-Stücklisten“)
    1  Tabellen einordnen: Stammdaten · Bewegungsdaten · Wertelisten · Personen/Geo (ausgeschlossen)
    2  Kandidaten finden: Kennzeichen (2 Werte), Klassifikation (≤ 60 Klassen), Parameter (numerisch);
       Wahrscheinlichkeit „manuell gepflegt“ aus Tabellenrolle, Spaltenname, Füllgrad, Kardinalität
    3  Kontext bauen: eigene Felder · über Schlüssel verbundene Stammdaten (Graph-Kontext) · Strukturtabellen
       (z. B. Stückliste: Vater, Anzahl Verwendungen) · Text (Benennung, Benennung des Vaters) · sprechende Nummern
    4  Stufe 0: Einzelfeld-Regeln. Trennt ein einzelnes Feld fast perfekt, ist es eine Regel – oder eine Folge der
       Entscheidung (Leakage). Solche Felder werden ausgewiesen und aus dem ML-Vergleich genommen.
    5  Stufe 1: Regel-Baseline · LightGBM eigene Felder · LightGBM + Verknüpfungen · Text (Zeichen-n-Gramme) ·
       TabPFN · Ensemble; 5-fach kreuzvalidiert, Text- und Nachbarschaftsmerkmale verschachtelt out-of-fold
    6  Automatisierungsquote: Anteil der Datensätze, die bei ≥ 98 % Trefferquote ohne Mensch entschieden werden;
       bei Kennzeichen zusätzlich der recall-sichere Auto-Negativ-Anteil und die nachweisbare Fehlergrenze
       α = 1 − 0,05^(1/n) (95 %, Ordnungsstatistik)
    7  Euro: eingesparter Pflegeaufwand (Annahmen editierbar) und der Wert, an dem das Feld hängt

  .venv/bin/python -m graph_ml.classifier_scan
"""
from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import connectors as cn

warnings.filterwarnings("ignore")
SEED = 7
FOLDS = 5
ART = Path(__file__).parent / "artifacts"
MAX_ROWS = 20000
MIN_LABELED = 40
MIN_CLASS = 5
LEAK_BA = 0.97                  # Einzelfeld trennt (fast) perfekt → Regel oder Folgefeld
PRECISION = (0.98, 0.99)
FX = {"adventureworks_data": cn.AW_USD_EUR, "wwi": cn.WWI_USD_EUR}

MANUAL_RE = re.compile(r"(flag|class|type|kind|categor|group|gruppe|status|level|point|days|limit|style|color|colour|farbe|"
                       r"size|groesse|line$|rating|preferred|active|chiller|brand|marke|shelf|bin|lagerplatz|reorder|safety|"
                       r"target|leadtime|lead_time|kennzeichen|klasse|typ|method|terms|package|outer|percent|weight|gewicht|"
                       r"supplier|vendor|lieferant|territory|tax)", re.I)
SYSTEM_RE = re.compile(r"(modified|rowguid|created|updated|timestamp|total|amount|balance|sum$|count|^n_|number$|nr$|"
                       r"population|sequence|revision|^split$|^failed$|rul_|cycles?_|study_|repaired|readout|time_step|"
                       r"onorder|received|rejected|stocked|scrapped|quantity$|qty$)", re.I)
HARD_SYSTEM_RE = re.compile(r"(modified|rowguid|transaction|^split$|^failed$|rul_|readout|time_step|population|sequence|revision|"
                            r"photo|culture|^is(system|external)|logon)", re.I)
MONEY_RE = re.compile(r"(price|cost|preis|kosten|amount|betrag|salary|rate$|discount)", re.I)
PII_TABLE_RE = re.compile(r"(person|people|employee|contact|email|phone|creditcard|salesperson|login|password|user)", re.I)
GEO_TABLE_RE = re.compile(r"(country|countries|state|province|cit(y|ies)|region|currency|culture|unitmeasure|address|"
                          r"deliverymethod|paymentmethod|transactiontype|systemparameter|colors|packagetypes)", re.I)
PERSONAL_COL_RE = re.compile(r"(gender|marital|birth|title$|suffix|middlename|firstname|lastname|nationalid|salary|vacation|"
                             r"sickleave|phone|fax|email|website|url|addressline|postal|password)", re.I)
CODE_RE = re.compile(r"(number|nummer|code|sku|matnr|sachnr|nr$)", re.I)

# Leitbeispiel (aus dem Konzept „Ersatzteil-Klassifikation“): je Datensatz gezielt nachgebildet
SPECIAL = {
    "adventureworks_data": [{
        "id": "Product.FinishedGoodsFlag@BOM", "table": "Product", "column": "FinishedGoodsFlag", "positive": "1",
        "subset": ("BillOfMaterials", "ComponentID"), "kind": "flag", "minutes": 4.0,
        "name": "Ersatzteil-Kennzeichen je Stücklisten-Bauteil",
        "label": "Bauteil ist einzeln verkaufsfähig (Ersatzteil) – FinishedGoodsFlag der Bauteile in der Stückliste",
        "decision_time_exclude": {
            "ProductSubcategoryID": "Katalog-Unterkategorie wird erst mit der ET-Freigabe vergeben",
            "ProductModelID": "Katalogmodell entsteht mit der Freigabe", "ListPrice": "ET-Preis ist Folge der Entscheidung",
            "SellStartDate": "Verkaufsstart ist Folge der Entscheidung", "SellEndDate": "Verkaufsende ist Folge der Entscheidung",
            "SafetyStockLevel": "Sicherheitsbestand folgt der Bevorratungsentscheidung",
            "ReorderPoint": "Meldebestand folgt der Bevorratungsentscheidung",
            "DiscontinuedDate": "Folge der Entscheidung"},
        "value": "stock", "note": "AdventureWorks hat kein eigenes ET-Kennzeichen; das Verkaufsfähig-Kennzeichen der Stücklisten-"
                                  "Bauteile entspricht fachlich der ET-Entscheidung (einzeln lieferbar ja/nein)."}],
}
SUPERSEDES = {"adventureworks_data": {"Product.FinishedGoodsFlag"}}


# ── 1 · Tabellen einordnen ────────────────────────────────────────────────────

def table_roles(mapping: dict) -> dict[str, dict]:
    rel = [r for r in mapping.get("relations", []) if r.get("kind") == "fk"]
    inbound = {}
    for r in rel:
        inbound[r["to_table"]] = inbound.get(r["to_table"], 0) + 1
    out = {}
    for t in mapping["tables"]:
        cols = t["columns"]
        has_dt = any(c["kind"] == "datetime" for c in cols)
        has_text = any("Freitext" in c.get("roles", []) for c in cols)
        personal = sum(1 for c in cols if PERSONAL_COL_RE.search(c["name"])) / max(len(cols), 1)
        transactional = (t["rows"] >= 20000 or (has_dt and t["rows"] >= 5000)
                         or bool(re.search(r"transaction|orderline|orderdetail|invoiceline|journal|history", t["name"], re.I)))
        person_cols = any(re.search(r"firstname|lastname|fullname|gender|birth|marital", c["name"], re.I) for c in cols)
        if PII_TABLE_RE.search(t["name"]) or (personal >= 0.25 and person_cols):
            role = "personen"
        elif GEO_TABLE_RE.search(t["name"]):
            role = "referenz"
        elif t["rows"] <= 60 and len(cols) <= 4:
            role = "werteliste"
        elif transactional:
            role = "bewegung"
        else:
            role = "stamm"
        out[t["name"]] = {"role": role, "rows": t["rows"], "has_text": has_text, "inbound": inbound.get(t["name"], 0),
                          "label": t.get("label", t["name"]), "pk": t.get("primary_key")}
    return out


ROLE_TEXT = {"stamm": "Stammdaten", "bewegung": "Bewegungsdaten", "werteliste": "Werteliste", "personen": "Personen (ausgeschlossen)",
             "referenz": "Referenz-/Geodaten (ausgeschlossen)"}


def _fk_map(mapping: dict, tables: dict) -> list[dict]:
    return [r for r in mapping.get("relations", []) if r.get("kind") == "fk" and r["from_table"] in tables and r["to_table"] in tables
            and (r.get("linkage_rate") or 0) >= 0.5]


# ── 2 · Kandidaten ───────────────────────────────────────────────────────────

def candidates(mapping: dict, roles: dict, tables: dict, ds_id: str) -> tuple[list[dict], list[dict]]:
    fks = _fk_map(mapping, tables)
    fk_of = {(r["from_table"], r["from_column"]): r for r in fks}
    cands, gaps = [], []
    for t in mapping["tables"]:
        ro = roles[t["name"]]
        for c in t["columns"]:
            name, nu, nr = c["name"], c.get("n_unique") or 0, c.get("null_rate") or 0.0
            r = fk_of.get((t["name"], name))
            reason = None
            if ro["role"] in ("personen", "referenz", "werteliste"):
                reason = ROLE_TEXT[ro["role"]]
            elif "PK" in c.get("roles", []) and not r:
                reason = "Primärschlüssel"
            elif c["kind"] == "datetime":
                reason = "Zeitstempel"
            elif "Freitext" in c.get("roles", []):
                reason = "Freitext – Lookup oder Sprachmodell statt Klassifikator"
            elif nu < 2:
                reason = "nur ein Wert"
            elif HARD_SYSTEM_RE.search(name):
                reason = "systemgeneriert"
            elif PERSONAL_COL_RE.search(name):
                reason = "personenbezogen – bewusst nicht vorhergesagt"
            elif r and (roles.get(r["to_table"], {}).get("rows", 0) > 300 or roles.get(r["to_table"], {}).get("role") == "personen"):
                reason = "Verweis auf große Tabelle/Person – Schlüssel, keine Klassifikation"
            elif "Hohe Kardinalität" in c.get("roles", []) or ((c.get("unique_ratio") or 0) > 0.5 and c["kind"] == "string"):
                reason = "Schlüssel-/Kennungsspalte"
            elif SYSTEM_RE.search(name) and not MANUAL_RE.search(name):
                reason = "System-/Bewegungsgröße"
            elif MONEY_RE.search(name) and not re.search(r"limit|taxrate", name, re.I) or re.search(r"taxamt|taxamount", name, re.I):
                reason = "Preis/Kosten – eigene Use Cases (Pricing)"
            elif ro["role"] == "bewegung" and not (r and roles.get(r["to_table"], {}).get("rows", 9e9) <= 60):
                reason = "Bewegungsdaten – nur Erfassungscodes werden geprüft"
            if reason:
                if nr >= 0.05 and ro["role"] not in ("personen",) and c["kind"] != "datetime" and nr < 1.0:
                    gaps.append({"table": t["name"], "column": name, "null_rate": round(nr, 4), "reason": reason})
                continue
            numeric = c["kind"] in ("int", "float", "bool")
            kind = "flag" if nu == 2 else ("class" if (not numeric or nu <= 25 or r) and nu <= 60 else ("param" if numeric else None))
            if kind is None:
                continue
            score = 0.15 if ro["role"] == "stamm" else 0.05
            score += 0.15 if ro["has_text"] else 0.0
            score += 0.25 if MANUAL_RE.search(name) else 0.0
            score += 0.15 if 0.02 <= nr <= 0.95 else 0.0
            score += 0.10 if kind in ("flag", "class") else 0.05
            score += 0.10 if r else 0.0                                         # Zuordnung zu einer Werteliste
            if ro["role"] == "bewegung":
                score += 0.15 if re.search(r"reason|grund|method|offer|type|art", name, re.I) else 0.0
            cands.append({"id": f"{t['name']}.{name}", "table": t["name"], "column": name, "kind": kind, "n_unique": int(nu),
                          "null_rate": round(nr, 4), "rows": int(t["rows"]), "table_role": ro["role"],
                          "manual_score": round(min(score, 1.0), 2), "fk_to": r["to_table"] if r else None})
    sup = SUPERSEDES.get(ds_id, set())
    cands = [c for c in cands if c["id"] not in sup]
    for s in SPECIAL.get(ds_id, []):
        if s["table"] in tables:
            cands.insert(0, {"id": s["id"], "table": s["table"], "column": s["column"], "kind": s["kind"], "n_unique": 2,
                             "null_rate": 0.0, "rows": int(roles[s["table"]]["rows"]), "table_role": "stamm", "manual_score": 0.95,
                             "fk_to": None, "special": True})
    return cands, gaps


# ── 3 · Kontext-Merkmale ─────────────────────────────────────────────────────

def _num_or_cat(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    x = pd.to_numeric(s.astype(str), errors="coerce") if not isinstance(s.dtype, pd.CategoricalDtype) else None
    if x is not None and x.notna().mean() > 0.95 and s.notna().mean() > 0:
        return x
    return s.astype(str).where(s.notna(), None).astype("category")


def _usable(col: dict) -> bool:
    if col["kind"] == "datetime" or "Freitext" in col.get("roles", []):
        return False
    if PERSONAL_COL_RE.search(col["name"]) or re.search(r"rowguid|modified", col["name"], re.I):
        return False
    if col["kind"] == "string" and (col.get("unique_ratio") or 0) > 0.5:
        return False
    return (col.get("n_unique") or 0) >= 2


def _text_cols(t: dict) -> list[str]:
    return [c["name"] for c in t["columns"] if "Freitext" in c.get("roles", [])][:2]


def _code_cols(t: dict) -> list[str]:
    return [c["name"] for c in t["columns"] if c["kind"] == "string" and CODE_RE.search(c["name"])
            and (c.get("unique_ratio") or 0) >= 0.5 and (c.get("avg_len") or 99) <= 25][:1]


def build_context(mapping: dict, tables: dict, roles: dict, cand: dict, special: dict | None) -> dict:
    """Merkmalstabelle je Datensatz der Kandidatentabelle: eigene Felder, verbundene Stammdaten, Struktur, Text, Gruppen."""
    tm = {t["name"]: t for t in mapping["tables"]}
    T, col = cand["table"], cand["column"]
    tdef = tm[T]
    df = tables[T].reset_index(drop=True)
    pk = tdef.get("primary_key")
    pk = pk if isinstance(pk, str) else None
    if special and special.get("subset"):
        st, sc = special["subset"]
        keep = set(pd.to_numeric(tables[st][sc], errors="coerce").dropna().astype(int))
        df = df[pd.to_numeric(df[pk], errors="coerce").isin(keep)].reset_index(drop=True)
    cap = MAX_ROWS if roles[T]["role"] == "stamm" else 8000
    if len(df) > cap:
        df = df.sample(cap, random_state=SEED).reset_index(drop=True)
    excl = dict((special or {}).get("decision_time_exclude", {}))
    fks = _fk_map(mapping, tables)
    own, rel, text = {}, {}, []
    sources = {}
    # eigene Felder
    for c in tdef["columns"]:
        n = c["name"]
        if n == col or n == pk or n in excl or not _usable(c):
            continue
        r = next((x for x in fks if x["from_table"] == T and x["from_column"] == n), None)
        if r and roles.get(r["to_table"], {}).get("rows", 0) > 300 and roles.get(r["to_table"], {}).get("role") != "stamm":
            continue
        own[n] = _num_or_cat(df[n])
        sources[n] = "eigen"
    for n in _code_cols(tdef):                                    # sprechende Nummer: erstes Segment
        if n in excl:
            continue
        pre = df[n].astype(str).str.split(r"[-_ ./]", regex=True).str[0].str[:6]
        top = pre.value_counts().index[:50]
        own[f"{n}·Präfix"] = pre.where(pre.isin(top), "sonst").astype("category")
        sources[f"{n}·Präfix"] = "eigen"
    for n in _text_cols(tdef):
        text.append(df[n].astype(str).fillna(""))
    # über Schlüssel verbundene Stammdaten (Eltern): nicht über die Zielspalte selbst
    for r in [x for x in fks if x["from_table"] == T and x["from_column"] != col and x["from_column"] not in excl]:
        P = r["to_table"]
        if roles.get(P, {}).get("role") in ("personen",) or P == T or roles.get(P, {}).get("rows", 0) > 50000:
            continue
        pdf = tables[P].drop_duplicates(r["to_column"]).set_index(r["to_column"])
        key = df[r["from_column"]]
        try:
            key = key.astype(pdf.index.dtype)
        except (TypeError, ValueError):
            pass
        for c in tm[P]["columns"]:
            if c["name"] == r["to_column"] or not _usable(c) or c["name"] in excl:
                continue
            fname = f"{P}.{c['name']}"
            rel[fname] = _num_or_cat(key.map(pdf[c["name"]]))
            sources[fname] = f"über {r['from_column']} → {P}"
        for n in _text_cols(tm[P])[:1]:
            text.append(key.map(pdf[n]).astype(str).fillna(""))
    # Struktur- und Brückentabellen, die auf diese Tabelle zeigen (z. B. Stückliste, Lieferant je Artikel, Warengruppen)
    groups = []
    if pk:
        keyv = df[pk]
        for r in [x for x in fks if x["to_table"] == T and roles.get(x["from_table"], {}).get("role") == "stamm" and x["from_table"] != T]:
            B = r["from_table"]
            bdf = tables[B]
            a = r["from_column"]
            g = bdf.groupby(a)
            cnt = g.size()
            fname = f"{B}·Anzahl ({a})"
            rel[fname] = keyv.map(cnt).fillna(0).astype(float)
            sources[fname] = f"Struktur {B}"
            for c in tm[B]["columns"]:
                if c["kind"] in ("int", "float") and _usable(c) and c["name"] != a and not any(
                        x["from_table"] == B and x["from_column"] == c["name"] for x in fks):
                    fname = f"{B}·Ø {c['name']} ({a})"
                    rel[fname] = keyv.map(pd.to_numeric(bdf[c["name"]], errors="coerce").groupby(bdf[a]).mean()).astype(float)
                    sources[fname] = f"Struktur {B}"
            for o in [x for x in fks if x["from_table"] == B and x["from_column"] != a]:
                e = bdf[[a, o["from_column"]]].dropna().drop_duplicates().rename(columns={a: "e", o["from_column"]: "g"})
                if e.empty or e["g"].nunique() < 2:
                    continue
                groups.append({"name": f"{B}.{a}→{o['from_column']}", "via": f"{T} → {B} ({a}) → {o['to_table']}", "edges": e})
                O = o["to_table"]                                   # Text der verbundenen Entität (z. B. Benennung des Vaters)
                if O in tm and _text_cols(tm[O]):
                    on = _text_cols(tm[O])[0]
                    names = tables[O].drop_duplicates(o["to_column"]).set_index(o["to_column"])[on].astype(str)
                    nm = e.assign(n=e["g"].map(names)).groupby("e")["n"].apply(lambda s: " ".join(s.dropna().astype(str)[:5]))
                    text.append(keyv.map(nm).fillna("").astype(str))
    y_raw = df[col]
    txt = None
    if text:
        txt = text[0].astype(str)
        for s in text[1:]:
            txt = txt + " | " + s.astype(str)
        txt = txt.str.lower()
    return {"df": df, "y": y_raw, "own": pd.DataFrame(own, index=df.index), "rel": pd.DataFrame(rel, index=df.index),
            "text": txt, "groups": groups, "sources": sources, "pk": pk, "excluded": excl}


# ── 4/5 · Stufe 0 und Stufe 1 ────────────────────────────────────────────────

def _folds(y: np.ndarray, k: int = FOLDS) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold
    f = np.zeros(len(y), int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)):
        f[te] = i
    return f


def _binned(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
        return s.astype(str).fillna("∅")
    x = s.astype(float)
    if x.nunique() <= 30:
        return x.round(6).astype(str).where(x.notna(), "∅")
    try:
        b = pd.qcut(x, 10, duplicates="drop").astype(str)
    except ValueError:
        b = x.round(2).astype(str)
    return b.where(x.notna(), "∅")


def _ba(y, p) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, p))


def single_field_rules(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray) -> list[dict]:
    """Stufe 0: jede Spalte allein als Lookup-Regel (häufigster Wert je Ausprägung), out-of-fold."""
    out = []
    for c in X.columns:
        b = _binned(X[c]).to_numpy()
        pred = np.empty(len(y), dtype=object)
        for k in range(folds.max() + 1):
            tr, te = folds != k, folds == k
            m = pd.Series(y[tr]).groupby(b[tr]).agg(lambda s: s.value_counts().index[0])
            glob = pd.Series(y[tr]).value_counts().index[0]
            pred[te] = pd.Series(b[te]).map(m).fillna(glob).to_numpy()
        out.append({"feature": c, "ba": _ba(y, pred.astype(str)), "pred": pred.astype(str)})
    return sorted(out, key=lambda r: -r["ba"])


def _lgbm_oof(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, classes: list[str]):
    import lightgbm as lgb
    yi = np.array([classes.index(v) for v in y])
    P = np.zeros((len(y), len(classes)))
    imp = np.zeros(X.shape[1])
    params = dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=max(3, min(20, len(y) // 40)), feature_fraction=0.85,
                  seed=SEED, verbose=-1, num_threads=4, class_weight="balanced")
    for k in range(folds.max() + 1):
        tr, te = folds != k, folds == k
        if len(classes) == 2:
            m = lgb.LGBMClassifier(n_estimators=300, objective="binary", **params)
        else:
            m = lgb.LGBMClassifier(n_estimators=300, objective="multiclass", **params)
        m.fit(X[tr], yi[tr])
        pr = m.predict_proba(X[te])
        P[np.ix_(te, list(m.classes_))] = pr
        imp += m.booster_.feature_importance("gain")
    return P, imp / max(folds.max() + 1, 1)


def _text_fp(txt: pd.Series, y: np.ndarray, tr: np.ndarray, te: np.ndarray, classes: list[str]) -> np.ndarray:
    """M2: Zeichen-n-Gramm-TF-IDF + logistische Regression (robust gegen Abkürzungen wie „HL“, „ZSB“)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, max_features=30000)
    m = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")
    m.fit(v.fit_transform(txt.iloc[tr]), y[tr])
    P = np.zeros((len(te), len(classes)))
    P[:, [classes.index(c) for c in m.classes_]] = m.predict_proba(v.transform(txt.iloc[te]))
    return P


def _knn_fp(txt: pd.Series, y: np.ndarray, tr: np.ndarray, te: np.ndarray, classes: list[str], k_nb: int = 7) -> np.ndarray:
    """M3: Klassenanteil unter den k ähnlichsten Datensätzen (Benennung) – liefert zugleich die Evidenz für den Prüfer."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=30000)
    A = v.fit_transform(txt.iloc[tr])
    nn = NearestNeighbors(n_neighbors=min(k_nb, len(tr)), metric="cosine").fit(A)
    _, ix = nn.kneighbors(v.transform(txt.iloc[te]))
    lab = y[tr][ix]
    return np.stack([(lab == c).mean(1) for c in classes], axis=1)


def _oof(fp, txt, y, folds, classes) -> np.ndarray:
    P = np.zeros((len(y), len(classes)))
    for k in range(folds.max() + 1):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        P[te] = fp(txt, y, tr, te, classes)
    return P


def _group_share(groups: list[dict], keys: pd.Series, y: np.ndarray, train: np.ndarray, classes: list[str]) -> np.ndarray:
    """Graph-Nachbarschaft: Klassenanteil der Datensätze, die dieselbe Gruppe teilen (z. B. denselben Vater in der
    Stückliste) – nur aus Trainingszeilen, ohne die Zeile selbst."""
    out = []
    lab = pd.Series(np.where(train, y, None), index=keys.to_numpy())
    for g in groups:
        e = g["edges"]
        e = e[e["e"].isin(keys)]
        ey = e.assign(y=e["e"].map(lab.groupby(level=0).first()))
        res = np.full((len(keys), len(classes)), np.nan)
        for j, c in enumerate(classes):
            ey["hit"] = (ey["y"] == c).astype(float).where(ey["y"].notna())
            gs = ey.groupby("g")["hit"].agg(["sum", "count"])
            m = ey.merge(gs, left_on="g", right_index=True)
            own = m["hit"].fillna(0)
            m["s"], m["n"] = m["sum"] - own, m["count"] - m["hit"].notna().astype(float)
            per = m.groupby("e")[["s", "n"]].sum()
            share = (per["s"] / per["n"].replace(0, np.nan))
            res[:, j] = keys.map(share).to_numpy(dtype=float)
        out.append(res)
    return np.concatenate(out, axis=1) if out else np.zeros((len(keys), 0))


def _auto_share(P: np.ndarray, y: np.ndarray, classes: list[str], prec: float) -> tuple[float, float]:
    """Größter Anteil der Datensätze, der – nach Sicherheit sortiert – mit Trefferquote ≥ prec automatisch entschieden wird."""
    conf = P.max(1)
    pred = np.array(classes)[P.argmax(1)]
    o = np.argsort(-conf, kind="stable")
    hit = np.cumsum(pred[o] == y[o]) / np.arange(1, len(y) + 1)
    ok = np.flatnonzero(hit >= prec)
    if not len(ok):
        return 0.0, 1.0
    n = ok.max() + 1
    return float(n / len(y)), float(conf[o][n - 1])


def evaluate(mapping: dict, tables: dict, roles: dict, cand: dict, log=print) -> dict:
    t0 = time.time()
    special = next((s for s in SPECIAL.get(cand.get("_ds"), []) if s["id"] == cand["id"]), None)
    ctx = build_context(mapping, tables, roles, cand, special)
    y_all = ctx["y"]
    labeled = y_all.notna().to_numpy()
    y_str = y_all.astype(str).str.strip().where(y_all.notna() & (y_all.astype(str).str.strip() != ""), None)
    if pd.api.types.is_numeric_dtype(y_all) and cand["kind"] != "param":
        y_str = y_all.map(lambda v: None if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
    res = {"id": cand["id"], "table": cand["table"], "column": cand["column"], "kind": cand["kind"],
           "rows": int(len(y_all)), "labeled": int(labeled.sum()), "missing": int((~labeled).sum()),
           "excluded": [{"feature": k, "reason": v} for k, v in ctx["excluded"].items()]}
    if cand["kind"] == "param":
        return {**res, **_evaluate_param(ctx, labeled, cand, t0)}
    vc = y_str[labeled].value_counts()
    keep = [c for c, n in vc.items() if n >= MIN_CLASS]
    if labeled.sum() < MIN_LABELED or len(keep) < 2:
        return {**res, "status": "na", "verdict": "nicht prüfbar",
                "reason": f"zu wenige Beispiele ({int(labeled.sum())} gefüllt, {len(keep)} Klassen mit ≥ {MIN_CLASS})",
                "seconds": round(time.time() - t0, 1)}
    m = labeled & y_str.isin(keep).to_numpy()
    y = y_str[m].to_numpy().astype(str)
    classes = sorted(keep, key=lambda c: -vc[c])
    pos = special["positive"] if special else (("1" if "1" in classes else ("True" if "True" in classes else classes[-1]))
                                               if len(classes) == 2 else None)
    folds = _folds(y)
    own, rel = ctx["own"][m].reset_index(drop=True), ctx["rel"][m].reset_index(drop=True)
    txt = ctx["text"][m].reset_index(drop=True) if ctx["text"] is not None else None
    keys = ctx["df"][ctx["pk"]][m].reset_index(drop=True) if ctx["pk"] else None
    # Graph-Nachbarschaft zuerst (verschachtelt out-of-fold), damit auch sie durch die Folgefeld-Prüfung läuft
    G = pd.DataFrame(index=own.index)
    if ctx["groups"] and keys is not None and len(classes) <= 10:
        parts = [_group_share(ctx["groups"], keys, y, folds != k, classes) for k in range(FOLDS)]
        Garr = np.zeros_like(parts[0])
        for k in range(FOLDS):
            Garr[folds == k] = parts[k][folds == k]
        G = pd.DataFrame(Garr, columns=[f"Graph·{g['name']}·{c}" for g in ctx["groups"] for c in classes])
        for g in ctx["groups"]:
            for c in classes:
                ctx["sources"][f"Graph·{g['name']}·{c}"] = f"Nachbarschaft {g['name'].split('.')[0]}"
    allX = pd.concat([own, rel, G], axis=1)
    # Stufe 0 – Einzelfeld-Regeln; (fast) perfekte Trenner = Regel oder Folgefeld → aus dem ML-Vergleich nehmen.
    # Ist ein Merkmal einer verbundenen Tabelle ein Folgefeld, ist es die ganze Tabelle (z. B. Sonderangebote nur für
    # verkaufsfähige Teile) – dann fallen alle ihre Merkmale heraus.
    rules = single_field_rules(allX, y, folds) if allX.shape[1] else []
    leaks = [r for r in rules if r["ba"] >= LEAK_BA]
    def _src_table(f):
        src = ctx["sources"].get(f, "eigen")
        return src.split(" ", 1)[1].split(" ")[0] if src.startswith(("Struktur ", "Nachbarschaft ")) else None
    leak_tables = {t for t in (_src_table(r["feature"]) for r in leaks) if t}
    leak_names = {r["feature"] for r in leaks} | {f for f in allX.columns if _src_table(f) in leak_tables}
    clean = [r for r in rules if r["feature"] not in leak_names]
    own_c = own.drop(columns=[c for c in own.columns if c in leak_names])
    rel_c = rel.drop(columns=[c for c in rel.columns if c in leak_names])
    G_c = G.drop(columns=[c for c in G.columns if c in leak_names])
    engines, P_of = [], {}
    if clean:
        best_rule = clean[0]
        pr = np.zeros((len(y), len(classes)))
        pr[np.arange(len(y)), [classes.index(v) if v in classes else 0 for v in best_rule["pred"]]] = 1.0
        P_of["rule"] = pr
        engines.append(("rule", f"Regel: häufigster Wert je „{best_rule['feature']}“", "Regel"))
    else:
        maj = np.zeros((len(y), len(classes)))
        maj[:, 0] = 1.0
        P_of["rule"] = maj
        engines.append(("rule", "Regel: häufigster Wert", "Regel"))
    imp_full = None
    if own_c.shape[1]:
        P_of["lgbm"], _ = _lgbm_oof(own_c, y, folds, classes)
        engines.append(("lgbm", "LightGBM (eigene Felder)", "Baum"))
    graph_X = pd.concat([own_c, rel_c, G_c], axis=1)
    if graph_X.shape[1] > own_c.shape[1]:
        P_of["lgbm_graph"], imp_full = _lgbm_oof(graph_X, y, folds, classes)
        engines.append(("lgbm_graph", "LightGBM + Verknüpfungen (Graph-Kontext)", "Graph-ML light"))
    if txt is not None:
        P_of["text"] = _oof(_text_fp, txt, y, folds, classes)
        engines.append(("text", "Text-Modell (Zeichen-n-Gramme der Benennung)", "Text"))
    tp = _tabpfn_oof(graph_X, y, folds, classes) if len(y) <= 5000 else None
    if tp is not None:
        P_of["tabpfn"] = tp
        engines.append(("tabpfn", "TabPFN v2", "TFM"))
    # Ensemble: Merkmale + Text- und kNN-Wahrscheinlichkeiten aus verschachtelten Folds
    if txt is not None and graph_X.shape[1]:
        P_ens = np.zeros((len(y), len(classes)))
        for k in range(FOLDS):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            inner = _folds(y[tr], 4)
            t_tr = txt.iloc[tr].reset_index(drop=True)
            S = np.zeros((len(y), 2 * len(classes)))
            S[tr] = np.hstack([_oof(_text_fp, t_tr, y[tr], inner, classes), _oof(_knn_fp, t_tr, y[tr], inner, classes)])
            S[te] = np.hstack([_text_fp(txt, y, tr, te, classes), _knn_fp(txt, y, tr, te, classes)])
            Xs = pd.concat([graph_X.reset_index(drop=True), pd.DataFrame(S, columns=[f"S{i}" for i in range(S.shape[1])])], axis=1)
            P_ens[te], _ = _lgbm_fit_predict(Xs, y, tr, te, classes)
        P_of["ensemble"] = P_ens
        engines.append(("ensemble", "Ensemble: LightGBM + Text + ähnliche Datensätze (Stacking)", "Ensemble"))
    # Kennzahlen
    out_eng = []
    for eid, name, klass in engines:
        P = P_of[eid]
        pred = np.array(classes)[P.argmax(1)]
        a98, t98 = _auto_share(P, y, classes, PRECISION[0])
        a99, _ = _auto_share(P, y, classes, PRECISION[1])
        mt = {"balanced_accuracy": round(_ba(y, pred), 4), "accuracy": round(float((pred == y).mean()), 4),
              "auto_98": round(a98, 4), "auto_99": round(a99, 4)}
        if pos:
            from sklearn.metrics import average_precision_score, roc_auc_score
            yp = (y == pos).astype(int)
            sc = P[:, classes.index(pos)]
            if eid != "rule":
                mt["auroc"], mt["auprc"] = round(float(roc_auc_score(yp, sc)), 4), round(float(average_precision_score(yp, sc)), 4)
            o = np.argsort(-sc, kind="stable")              # Auto-ET: großzügig, solange ≥ 98 % der Gesetzten echte ET sind
            pr_pos = np.cumsum(yp[o]) / np.arange(1, len(yp) + 1)
            okp = np.flatnonzero(pr_pos >= PRECISION[0])
            n_ap = int(okp.max() + 1) if len(okp) else 0
            if n_ap:                                       # Zone endet beim letzten echten ET, nicht in nachfolgenden Fehlalarmen
                n_ap = int(np.flatnonzero(yp[o][:n_ap] == 1).max() + 1)
            mt["auto_positive"] = round(n_ap / len(yp), 4)
            mt["auto_positive_fp"] = int((yp[o][:n_ap] == 0).sum())   # gemessene Fehlalarme in der Auto-ET-Zone
            t_low = sc[yp == 1].min()                      # keine übersehene Positive im Rückblick
            mt["auto_negative"] = round(float(((sc < t_low) & (yp == 0)).sum() / len(y)), 4)
            mt["negatives_caught"] = round(float(((sc < t_low) & (yp == 0)).sum() / max((yp == 0).sum(), 1)), 4)
        out_eng.append({"id": eid, "name": name, "klass": klass, "metrics": mt})
    best = max((e for e in out_eng if e["id"] != "rule"), key=lambda e: (e["metrics"]["auto_98"], e["metrics"]["balanced_accuracy"]),
               default=out_eng[0])
    rule_e = out_eng[0]
    Pb = P_of[best["id"]]
    a98, t98 = _auto_share(Pb, y, classes, PRECISION[0])
    # Graubereich-Beispiele und Vorschläge für fehlende Werte
    disp = _display_names(ctx, m)
    conf, predb = Pb.max(1), np.array(classes)[Pb.argmax(1)]
    grey = np.argsort(np.abs(conf - t98))[:6] if a98 < 1 else np.argsort(conf)[:6]
    examples = [{"name": disp[i], "actual": y[i], "predicted": predb[i], "confidence": round(float(conf[i]), 3)} for i in grey]
    missing_fill = _predict_missing(ctx, m, labeled, graph_X, y, classes, best["id"], t98, leak_names)
    n_pos = int((y == pos).sum()) if pos else None
    alpha = (1 - 0.05 ** (1 / n_pos)) if n_pos else None
    lift = (best["metrics"]["balanced_accuracy"] - rule_e["metrics"]["balanced_accuracy"]) / max(rule_e["metrics"]["balanced_accuracy"], 1e-9)
    rule_cover = None
    if leaks:
        lr = leaks[0]
        rule_cover = {"feature": lr["feature"], "ba": round(lr["ba"], 4), "source": ctx["sources"].get(lr["feature"], "eigen")}
    if a98 >= 0.6:
        verdict, status = "ML vorhersagbar", "ok"
    elif a98 >= 0.2:
        verdict, status = "teilweise – Graubereich zum Menschen", "warn"
    else:
        verdict, status = "kaum vorhersagbar", "bad"
    if rule_cover and status != "ok":
        verdict, status = "Regel/Lookup möglich (oder Folgefeld)", "warn"
    if rule_e["metrics"]["balanced_accuracy"] >= 0.95 and rule_e["metrics"]["auto_98"] >= 0.9:
        verdict, status = "Regel genügt (Stufe 0)", "ok"
    imp = []
    if imp_full is not None:
        imp = sorted([{"feature": c, "source": ctx["sources"].get(c, "eigen"),
                       "importance": round(float(v), 1)} for c, v in zip(graph_X.columns, imp_full)], key=lambda r: -r["importance"])[:8]
    log(f"[classifier] {cand['id']:42s} {len(y):6d} · beste {best['id']:10s} BA {best['metrics']['balanced_accuracy']:.3f} · "
        f"Auto@98 % {a98:.2f} · Regel BA {rule_e['metrics']['balanced_accuracy']:.3f} · Folgefelder {len(leaks)}")
    return {**res, "status": status, "verdict": verdict, "classes": [{"value": c, "n": int(vc[c])} for c in classes][:12],
            "positive": pos, "n_positive": n_pos, "alpha_95": round(alpha, 5) if alpha else None,
            "engines": out_eng, "best": best["id"], "lift_vs_rule": round(lift, 4), "auto_98": round(a98, 4),
            "threshold_98": round(t98, 4), "rule_or_leak": [{"feature": r["feature"], "ba": round(r["ba"], 4),
                                                              "source": ctx["sources"].get(r["feature"], "eigen")} for r in leaks[:6]],
            "rule_cover": rule_cover, "leak_tables": sorted(leak_tables), "importance": imp, "examples": examples, "missing_fill": missing_fill,
            "features": {"own": int(own_c.shape[1]), "linked": int(rel_c.shape[1]), "groups": [g["via"] for g in ctx["groups"]],
                         "text": txt is not None},
            "seconds": round(time.time() - t0, 1)}


def _lgbm_fit_predict(X, y, tr, te, classes):
    import lightgbm as lgb
    yi = np.array([classes.index(v) for v in y])
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_data_in_leaf=max(3, min(20, len(y) // 40)),
                           feature_fraction=0.85, seed=SEED, verbose=-1, num_threads=4, class_weight="balanced",
                           objective="binary" if len(classes) == 2 else "multiclass")
    m.fit(X.iloc[tr], yi[tr])
    P = np.zeros((len(te), len(classes)))
    P[:, list(m.classes_)] = m.predict_proba(X.iloc[te])
    return P, m


def _tabpfn_oof(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, classes: list[str]):
    if X.shape[1] == 0 or len(classes) > 10:
        return None
    try:
        import torch
        from tabpfn import TabPFNClassifier
    except Exception:
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xn = X.copy()
    for c in Xn.columns:
        if isinstance(Xn[c].dtype, pd.CategoricalDtype):
            Xn[c] = Xn[c].cat.codes.astype(float).replace(-1, np.nan)
    Xn = Xn.astype(float).to_numpy()[:, :400]
    P = np.zeros((len(y), len(classes)))
    try:
        for k in range(folds.max() + 1):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            if len(tr) > 3000:
                tr = np.random.default_rng(SEED).choice(tr, 3000, replace=False)
            try:
                from tabpfn.constants import ModelVersion
                m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=dev)
            except Exception:
                m = TabPFNClassifier(device=dev)
            m.fit(Xn[tr], y[tr])
            P[np.ix_(te, [classes.index(c) for c in m.classes_])] = m.predict_proba(Xn[te])
        return P
    except Exception:
        return None


def _display_names(ctx: dict, m: np.ndarray) -> list[str]:
    df = ctx["df"][m].reset_index(drop=True)
    tc = [c for c in df.columns if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype)]
    name = next((c for c in df.columns if re.search(r"name|benennung|description", c, re.I)), tc[0] if tc else None)
    if name is None:
        return [str(i) for i in range(len(df))]
    return df[name].astype(str).str[:60].tolist()


def _predict_missing(ctx, m, labeled, graph_X, y, classes, best, t98, leak_names) -> dict | None:
    """Vorschläge für die Lücken: bestes Merkmalsmodell auf allen gefüllten Zeilen, angewandt auf die leeren."""
    miss = ~labeled
    if miss.sum() == 0:
        return None
    try:
        own, rel = ctx["own"], ctx["rel"]
        X_all = pd.concat([own.drop(columns=[c for c in own.columns if c in leak_names]),
                           rel.drop(columns=[c for c in rel.columns if c in leak_names])], axis=1)
        X_all = X_all[[c for c in graph_X.columns if c in X_all.columns]]
        tr = np.flatnonzero(m)
        P, _ = _lgbm_fit_predict(X_all.reset_index(drop=True), _y_full(y, m, classes), tr, np.flatnonzero(miss), classes)
        conf = P.max(1)
        pred = np.array(classes)[P.argmax(1)]
        sure = conf >= t98
        vc = pd.Series(pred[sure]).value_counts()
        return {"missing": int(miss.sum()), "sure": int(sure.sum()), "share_sure": round(float(sure.mean()), 4),
                "proposals": [{"value": k, "n": int(v)} for k, v in vc.head(6).items()], "model": "LightGBM + Verknüpfungen"}
    except Exception as exc:  # Vorschläge sind ein Zusatz – nie die Analyse abbrechen
        return {"missing": int(miss.sum()), "sure": 0, "share_sure": 0.0, "proposals": [], "error": type(exc).__name__}


def _y_full(y, m, classes):
    full = np.array([classes[0]] * len(m), dtype=object)
    full[np.flatnonzero(m)] = y
    return full.astype(str)


def _evaluate_param(ctx, labeled, cand, t0) -> dict:
    """Numerische Dispositions-/Stammparameter: Anteil der Vorschläge innerhalb ±10 % gegen Regel (Median je Gruppe)."""
    import lightgbm as lgb
    y = pd.to_numeric(ctx["y"], errors="coerce")
    mk = labeled & y.notna().to_numpy()
    if mk.sum() < MIN_LABELED:
        return {"status": "na", "verdict": "nicht prüfbar", "reason": "zu wenige gefüllte Werte", "seconds": round(time.time() - t0, 1)}
    yv = y[mk].to_numpy(float)
    X = pd.concat([ctx["own"], ctx["rel"]], axis=1)[mk].reset_index(drop=True)
    from sklearn.model_selection import KFold
    folds = np.zeros(len(yv), int)
    for i, (_, te) in enumerate(KFold(FOLDS, shuffle=True, random_state=SEED).split(yv)):
        folds[te] = i
    rules = []
    for c in X.columns:
        b = _binned(X[c]).to_numpy()
        pr = np.zeros(len(yv))
        for k in range(FOLDS):
            tr, te = folds != k, folds == k
            med = pd.Series(yv[tr]).groupby(b[tr]).median()
            pr[te] = pd.Series(b[te]).map(med).fillna(np.median(yv[tr])).to_numpy()
        rules.append((float(np.mean(np.abs(pr - yv) <= 0.1 * np.abs(yv) + 1e-9)), c, pr))
    rules.sort(key=lambda r: -r[0])
    leaks = [r for r in rules if r[0] >= 0.98]
    Xc = X.drop(columns=[r[1] for r in leaks])
    P = np.zeros(len(yv))
    for k in range(FOLDS):
        tr, te = folds != k, folds == k
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15, min_data_in_leaf=5, objective="l1", verbose=-1,
                              seed=SEED, num_threads=4)
        m.fit(Xc[tr], yv[tr])
        P[te] = m.predict(Xc[te])
    within = float(np.mean(np.abs(P - yv) <= 0.1 * np.abs(yv) + 1e-9))
    rule_w = next((r[0] for r in rules if r[1] not in {x[1] for x in leaks}), 0.0)
    status = "ok" if within >= 0.6 else "warn" if within >= 0.3 else "bad"
    return {"status": status, "verdict": {"ok": "ML vorhersagbar", "warn": "teilweise – Graubereich zum Menschen",
                                          "bad": "kaum vorhersagbar"}[status],
            "engines": [{"id": "rule", "name": "Regel: Median je Gruppe", "klass": "Regel", "metrics": {"within_10": round(rule_w, 4)}},
                        {"id": "lgbm_graph", "name": "LightGBM + Verknüpfungen", "klass": "Graph-ML light", "metrics": {"within_10": round(within, 4)}}],
            "best": "lgbm_graph", "auto_98": round(within, 4), "lift_vs_rule": round((within - rule_w) / max(rule_w, 1e-9), 4),
            "rule_or_leak": [{"feature": r[1], "ba": round(r[0], 4), "source": ctx["sources"].get(r[1], "eigen")} for r in leaks[:4]],
            "metric_note": "Anteil der Vorschläge innerhalb ±10 % des gepflegten Werts", "seconds": round(time.time() - t0, 1)}


# ── 7 · Euro-Bezug ───────────────────────────────────────────────────────────

MINUTES = {"flag": 1.5, "class": 2.0, "param": 3.0}
EURO_DEFAULTS = {"rate_eur_h": 60.0, "holding_rate": 0.25, "fn_cost_eur": 5000.0}


def _years(tables: dict, mapping: dict) -> float:
    lo, hi = None, None
    for t in mapping["tables"]:
        for c in t["columns"]:
            if c["kind"] == "datetime" and t["rows"] >= 1000 and not re.search(r"modified|birth|hire|valid", c["name"], re.I):
                s = pd.to_datetime(tables[t["name"]][c["name"]], errors="coerce").dropna()
                if len(s):
                    lo = s.min() if lo is None else min(lo, s.min())
                    hi = s.max() if hi is None else max(hi, s.max())
    return max(((hi - lo).days / 365.25) if lo is not None else 1.0, 0.5)


def value_basis(ds_id: str, tables: dict, mapping: dict, table: str, cand_id: str, ctx_rows: int, special: dict | None,
                role: str = "stamm") -> dict:
    """Wert, an dem die Datensätze der Tabelle hängen: Umsatz pro Jahr aus Positionen (1 Schritt über Schlüssel) oder
    Bestandswert; beim Ersatzteil-Beispiel die Bevorratungskosten."""
    fx = FX.get(ds_id, 1.0)
    years = _years(tables, mapping)
    fks = _fk_map(mapping, tables)
    tm = {t["name"]: t for t in mapping["tables"]}
    pk = tm[table].get("primary_key")
    if special and special.get("value") == "stock":
        p = tables["Product"]
        comp = set(pd.to_numeric(tables["BillOfMaterials"]["ComponentID"], errors="coerce").dropna().astype(int))
        c = p[p["ProductID"].isin(comp)]
        et = c[c["FinishedGoodsFlag"].astype(int) == 1]
        stock = (pd.to_numeric(et["SafetyStockLevel"], errors="coerce") * pd.to_numeric(et["StandardCost"], errors="coerce")).fillna(0) * fx
        new_year = _new_per_year(tables["BillOfMaterials"], "ComponentID", "StartDate")
        return {"kind": "stock", "label": "Bevorratungswert der Ersatzteile (Sicherheitsbestand × Standardkosten)",
                "total_eur": round(float(stock.sum()), 0), "per_positive_eur": round(float(stock.mean()), 0),
                "entries_per_year": new_year, "years": round(years, 2)}
    rev = None
    if isinstance(pk, str):
        for r in [x for x in fks if x["to_table"] == table]:
            B = tables[r["from_table"]]
            cols = {c.lower(): c for c in B.columns}
            money = next((cols[c] for c in ("extendedprice", "linetotal", "subtotal", "transactionamount") if c in cols), None)
            s = None
            if money:
                s = pd.to_numeric(B[money].astype(str), errors="coerce")
            elif "unitprice" in cols and any(q in cols for q in ("orderqty", "quantity")):
                q = cols.get("orderqty") or cols.get("quantity")
                s = pd.to_numeric(B[cols["unitprice"]], errors="coerce") * pd.to_numeric(B[q], errors="coerce")
            if s is not None:
                v = s.groupby(B[r["from_column"]]).sum()
                rev = (v, r["from_table"])
                break
    entries = int(round(len(tables[table]) / years)) if role == "bewegung" else _new_per_year_table(tables[table], tm[table], years)
    if rev is not None:
        v, src = rev
        return {"kind": "revenue", "label": f"Umsatz pro Jahr der Datensätze (aus {src})",
                "total_eur": round(float(v.sum() * fx / years), 0), "entries_per_year": entries, "years": round(years, 2)}
    return {"kind": "none", "label": "kein Umsatz-/Bestandsbezug ermittelbar", "total_eur": None, "entries_per_year": entries,
            "years": round(years, 2)}


def _new_per_year(df: pd.DataFrame, key: str, date: str) -> int:
    d = pd.to_datetime(df[date], errors="coerce")
    first = d.groupby(df[key]).min().dropna()
    if first.empty:
        return 0
    span = max((first.max() - first.min()).days / 365.25, 1.0)
    return int(round(len(first) / span))


def _new_per_year_table(df: pd.DataFrame, tdef: dict, years: float) -> int:
    dc = [c["name"] for c in tdef["columns"] if c["kind"] == "datetime" and re.search(r"start|open|created|since|valid", c["name"], re.I)]
    if dc:
        d = pd.to_datetime(df[dc[0]], errors="coerce").dropna()
        if len(d) and d.nunique() > 1:
            span = max((d.max() - d.min()).days / 365.25, 1.0)
            return int(round(len(d) / span))
    return int(round(len(df) * 0.15))                     # Annahme: 15 % Neuanlagen bzw. Änderungen pro Jahr


def euro(r: dict, params: dict | None = None) -> dict:
    """Identisch zur JS-Funktion dcEuro: eingesparter Pflegeaufwand + (Ersatzteil) Bevorratungs- und Fehlerkosten."""
    p = {**EURO_DEFAULTS, **(params or {})}
    v = r.get("value") or {}
    minutes = p.get("minutes", r.get("minutes", MINUTES.get(r["kind"], 2.0)))
    auto = 0.0 if r.get("inherited") else (r.get("auto_98") or 0.0)
    entries = v.get("entries_per_year") or 0
    labor = entries * minutes / 60 * p["rate_eur_h"] * auto
    fill = r.get("missing_fill") or {}
    gap_once = (fill.get("sure") or 0) * minutes / 60 * p["rate_eur_h"]
    out = {"labor_year": round(labor, 0), "gap_once": round(gap_once, 0), "entries_per_year": entries, "minutes": minutes,
           "exposure_eur": v.get("total_eur")}
    if v.get("kind") == "stock" and r.get("n_positive"):
        # Kaskade aus dem ET-Konzept: „ET“ automatisch bei ≥ 98 % Trefferquote, „kein ET“ nur unter der recall-sicheren
        # Schwelle; der Rest geht zum Menschen. Kosten: Fehlalarme werden bevorratet, übersehene ET bis zur Grenze α.
        best = next((e for e in r["engines"] if e["id"] == r["best"]), {})
        m = best.get("metrics", {})
        auto_pos, auto_neg = m.get("auto_positive", 0.0), m.get("auto_negative", 0.0)
        labor = entries * minutes / 60 * p["rate_eur_h"] * (auto_pos + auto_neg)
        stock_et = (v.get("per_positive_eur") or 0) * p["holding_rate"]
        fp_rate = (m.get("auto_positive_fp") or 0) / max(r["labeled"], 1)   # gemessen, nicht angenommen
        fp_stock = entries * fp_rate * stock_et
        # Obergrenze übersehener ET (95 %) – Prüfgrenze aus dem Kalibrierungsumfang, kein erwarteter Schaden
        fn_bound = entries * (r["n_positive"] / max(r["labeled"], 1)) * (r.get("alpha_95") or 0) * p["fn_cost_eur"]
        out.update({"labor_year": round(labor, 0), "fp_stock_year": round(fp_stock, 0), "fn_risk_bound_year": round(fn_bound, 0),
                    "stock_per_et_year": round(stock_et, 0), "auto_positive": auto_pos, "auto_negative": auto_neg})
        out["net_year"] = round(labor - fp_stock, 0)
    else:
        out["net_year"] = round(labor, 0)
    return out


# ── Gesamtlauf ───────────────────────────────────────────────────────────────

def scan(ds_id: str, log=lambda m: print(m, flush=True)) -> dict:
    from .data_loader import get_registry
    t0 = time.time()
    reg = get_registry()
    mapping = reg.mapping(ds_id)
    tables = reg.tables(ds_id)
    roles = table_roles(mapping)
    cands, gaps = candidates(mapping, roles, tables, ds_id)
    likely = [c for c in cands if c["manual_score"] >= 0.4]
    log(f"[classifier] {ds_id}: {len(mapping['tables'])} Tabellen · {len(cands)} Kandidaten · {len(likely)} wahrscheinlich manuell")
    results = []
    for c in likely:
        c["_ds"] = ds_id
        try:
            r = evaluate(mapping, tables, roles, c, log)
        except Exception as exc:
            r = {"id": c["id"], "table": c["table"], "column": c["column"], "kind": c["kind"], "status": "na",
                 "verdict": "nicht prüfbar", "reason": f"{type(exc).__name__}: {str(exc)[:120]}"}
            log(f"[classifier] {c['id']}: {r['reason']}")
        special = next((s for s in SPECIAL.get(ds_id, []) if s["id"] == c["id"]), None)
        r.update({"manual_score": c["manual_score"], "null_rate": c["null_rate"], "table_role": c["table_role"],
                  "table_label": roles[c["table"]]["label"], "fk_to": c.get("fk_to"), "special": bool(special)})
        if special:
            r.update({"name": special["name"], "label": special["label"], "note": special["note"], "minutes": special["minutes"]})
        if c["table_role"] == "bewegung" and any(str(l.get("source", "")).startswith("über") for l in r.get("rule_or_leak", [])):
            # Erfassungsfeld, dessen Wert sich aus verbundenen Stammdaten ergibt: das System übernimmt ihn – kein Pflegeaufwand
            src = next(l for l in r["rule_or_leak"] if str(l.get("source", "")).startswith("über"))
            r.update({"status": "ok", "verdict": "Regel genügt – aus Stammdaten übernommen", "inherited": True,
                      "inherited_from": f"{src['feature']} ({src['source']})"})
        r["value"] = (value_basis(ds_id, tables, mapping, c["table"], c["id"], r.get("rows", 0), special, c["table_role"])
                      if r.get("status") != "na" else None)
        if r.get("status") != "na":
            r["euro"] = euro(r)
        results.append(r)
    results.sort(key=lambda r: (not r.get("special"), {"ok": 0, "warn": 1, "bad": 2, "na": 3}.get(r.get("status"), 4),
                                -((r.get("euro") or {}).get("net_year") or 0)))
    counts = {k: sum(1 for r in results if r.get("status") == k) for k in ("ok", "warn", "bad", "na")}
    return {"dataset": ds_id, "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1),
            "tables": [{"name": k, "role": v["role"], "role_text": ROLE_TEXT[v["role"]], "rows": v["rows"]} for k, v in roles.items()],
            "n_candidates": len(cands), "n_likely": len(likely),
            "other_candidates": [{"id": c["id"], "manual_score": c["manual_score"], "kind": c["kind"]} for c in cands if c["manual_score"] < 0.4][:40],
            "results": results, "counts": counts, "gaps": sorted(gaps, key=lambda g: -g["null_rate"])[:25],
            "euro_defaults": {**EURO_DEFAULTS, "minutes": MINUTES},
            "method": ["Stufe 0 · Regeln & Lookups (Einzelfeld-Trenner; Folgefelder ausgewiesen)",
                       "Stufe 1 · LightGBM, Graph-Kontext, Text-Modell, TabPFN, Ensemble (5-fach, verschachtelt out-of-fold)",
                       "Stufe 2 · LLM-Prüfer für Graubereich und neue Datensätze (Konzept, nicht gerechnet)",
                       "Stufe 3 · Mensch entscheidet den Graubereich und misst die Automatik blind (Konzept)"]}


def artifact_path(ds_id: str) -> Path:
    return ART / f"classifier_{ds_id}.json"


def load(ds_id: str) -> dict | None:
    p = artifact_path(ds_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(datasets: tuple[str, ...] = ("adventureworks_data", "wwi", "online_retail_ii", "scania", "cmapss")) -> int:
    log = lambda m: print(m, flush=True)  # noqa: E731
    for ds in datasets:
        res = scan(ds, log)
        artifact_path(ds).write_text(json.dumps(_clean(res), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {ds}: {res['counts']} · {res['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
