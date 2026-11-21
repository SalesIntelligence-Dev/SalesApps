"""
Fix-Playbooks (Modul 5)
=======================
Jedes Playbook ist ein reproduzierbarer Bereinigungsschritt auf einer KOPIE
der Rohdaten. Eine Pipeline-Version ist die geordnete Liste angewendeter
Playbooks mit ihren Optionen (v0 = Rohdaten, nie überschrieben).

Kreislauf aus dem PDF: Playbooks anwenden → Checks neu (Stufe 2) → auf Wunsch
Rückblick neu (Stufe 3) → Vorher/Nachher der Ampel, bis ein Use Case trägt.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"
DEFAULT_FX = 1.15
# Reihenfolge: erst vereinheitlichen (PB-04), dann Dubletten – so bleibt die Pipeline idempotent
ORDER = ["PB-04", "PB-06", "PB-03", "PB-07", "PB-02", "PB-01", "PB-05", "PB-08"]   # Daten-Playbooks
NON_ARTICLE = {"POST", "DOT", "M", "D", "C2", "C3", "BANK CHARGES", "AMAZONFEE", "CRUK", "S", "B", "PADS",
               "ADJUST", "ADJUST2"}
GOV_ITEMS = ["Pseudonymisierung der Kundennummern", "Speicherort Deutschland / EU", "Betriebsrat informiert",
             "Schreibrechte für den Rückspielweg ins CRM/ERP", "Lösch- und Aufbewahrungskonzept"]

CATALOG: list[dict] = [
    {"id": "PB-01", "title": "Fehlende Kundennummern", "kind": "data",
     "problem": "22,8 % der Positionen haben keine Kundennummer – für kundenbezogene Use Cases unbrauchbar.",
     "checks": ["structure", "columns"], "use_cases": ["churn", "cross_selling", "recommendation", "lead_scoring"],
     "options": [
         {"id": "A", "label": "Positionen ohne Kunde ausschließen", "pro": "saubere Kundenhistorie",
          "con": "Warenkorb-Analysen verlieren 22,8 % der Positionen"},
         {"id": "B", "label": "Gast-Kunde je Rechnung anlegen", "pro": "alle Positionen bleiben für Warenkorb/Empfehlung",
          "con": "nur für Warenkorb und Empfehlung zulässig, nicht für Churn/Lead"}],
     "default": "A",
     "note": "Option A kostet Absatzprognose und Warenkorb-Analysen 22,8 % der Positionen – Option B wählen, "
             "wenn diese Use Cases im Fokus stehen."},
    {"id": "PB-02", "title": "Stornos ohne Originalbezug", "kind": "data",
     "problem": "Storno-Belege verweisen nicht auf die Originalrechnung – Retouren-Analysen hängen in der Luft.",
     "checks": ["structure"], "use_cases": ["returns_root_cause"],
     "options": [{"id": "A", "label": "Heuristisch zuordnen (Kunde, Artikel, Menge, ≤ 90 Tage)",
                  "pro": "Beziehung Storno → Original wird messbar", "con": "≈ 20 % bleiben ohne Treffer"}],
     "default": "A"},
    {"id": "PB-03", "title": "Nicht-Artikel-Codes", "kind": "data",
     "problem": "Porto, Gebühren und manuelle Buchungen (POST, M, D, BANK CHARGES …) verzerren Artikel-Statistiken.",
     "checks": ["columns"], "use_cases": ["recommendation", "cross_selling", "returns_root_cause", "demand_forecast"],
     "options": [{"id": "A", "label": "Ausfiltern", "pro": "nur echte Artikel im Graphen", "con": "Gebühren fehlen in Umsatzsummen"}],
     "default": "A"},
    {"id": "PB-04", "title": "Artikelstamm konsolidieren", "kind": "data",
     "problem": "Artikelnummern in Groß-/Kleinschreibung und bis zu mehrere Beschreibungen je Artikel.",
     "checks": ["columns", "text"], "use_cases": ["recommendation", "cross_selling", "demand_forecast"],
     "options": [{"id": "A", "label": "Nummern vereinheitlichen, häufigste Beschreibung je Artikel",
                  "pro": "ein Knoten je Artikel", "con": "seltene Varianten verlieren ihre Eigenbezeichnung"}],
     "default": "A"},
    {"id": "PB-05", "title": "Freitext → Artikelgruppen", "kind": "data",
     "problem": "Artikel haben nur eine Freitext-Beschreibung, keine Artikelgruppe – Cross-Selling nur als Proxy.",
     "checks": ["structure", "label", "text"], "use_cases": ["cross_selling", "demand_forecast", "recommendation"],
     "options": [{"id": "A", "label": "TF-IDF + Clustering (24 Gruppen, sprechende Namen)",
                  "pro": "lokal, reproduzierbar, ohne externen Dienst", "con": "Gruppen statistisch, nicht fachlich gepflegt"}],
     "default": "A", "note": "Optional per Sprachmodell verfeinerbar – hier bewusst lokal ohne externen Dienst."},
    {"id": "PB-06", "title": "Exakte Dubletten", "kind": "data",
     "problem": "34.335 identische Positionen (gleiche Rechnung, Artikel, Menge, Zeit).",
     "checks": ["columns"], "use_cases": ["churn", "recommendation", "returns_root_cause", "demand_forecast"],
     "options": [{"id": "A", "label": "Entfernen", "pro": "keine doppelt gezählten Mengen", "con": "nur exakt gleiche Zeilen – fast gleiche bleiben"}],
     "default": "A"},
    {"id": "PB-07", "title": "Preis 0, negative Mengen, Ausreißer", "kind": "data",
     "problem": "Preis 0, negative Mengen ohne Storno (Bestandskorrekturen) und extreme Einzelmengen.",
     "checks": ["columns"], "use_cases": ["demand_forecast", "margin", "returns_root_cause"],
     "options": [{"id": "B", "label": "Ausschließen", "pro": "saubere Absatz- und Preisstatistik", "con": "Korrekturen nicht mehr sichtbar"},
                 {"id": "A", "label": "Nur markieren", "pro": "nichts geht verloren", "con": "Modelle sehen die Ausreißer weiter"}],
     "default": "B"},
    {"id": "PB-08", "title": "Währung GBP → EUR", "kind": "data",
     "problem": "Preise in GBP – Business Case und Vergleich mit deutschen Kunden in EUR.",
     "checks": ["columns"], "use_cases": ["churn", "margin", "demand_forecast"],
     "options": [{"id": "A", "label": f"Fester Kurs {DEFAULT_FX:.2f} €/£".replace(".", ","), "pro": "einfach, nachvollziehbar",
                  "con": "keine Tageskurse"}],
     "default": "A", "param": {"fx": DEFAULT_FX}},
    {"id": "PB-09", "title": "Governance-Checkliste", "kind": "checklist",
     "problem": "Personenbezug, Speicherort und Rückspielweg sind vor einem Pilot zu klären.",
     "checks": ["governance"], "use_cases": ["churn", "lead_scoring", "cross_selling", "recommendation", "next_best_action"],
     "items": GOV_ITEMS},
    {"id": "PB-10", "title": "Fehlende Tabellen anfordern", "kind": "request",
     "problem": "Für Margen, Warranty und Next Best Action fehlen Tabellen – es entsteht eine Datenanforderung.",
     "checks": ["structure", "label"], "use_cases": ["margin", "warranty_root_cause", "next_best_action"]},
]


# Aufwand in Personentagen (von–bis) für den Reparaturplan im Report – Schätzung für einen Readiness-Sprint
EFFORT_DAYS = {"PB-01": (0.5, 1), "PB-02": (1, 2), "PB-03": (0.5, 0.5), "PB-04": (0.5, 1), "PB-05": (1.5, 3),
               "PB-06": (0.25, 0.5), "PB-07": (0.5, 1), "PB-08": (0.25, 0.25), "PB-09": (1, 3), "PB-10": (5, 15)}
EFFORT_NOTE = {"PB-05": "inkl. fachlicher Abnahme der Gruppen", "PB-09": "Abstimmung mit Datenschutz, Betriebsrat, IT",
               "PB-10": "Datenbeschaffung beim Kunden (ERP-Export, Garantiefälle, Kampagnen)"}


def effort_text(pid: str) -> str:
    lo, hi = EFFORT_DAYS.get(pid, (0, 0))
    f = lambda x: (f"{x:.2f}".rstrip("0").rstrip(".")).replace(".", ",")  # noqa: E731
    return f"{f(lo)} PT" if lo == hi else f"{f(lo)}–{f(hi)} PT"


def catalog() -> list[dict]:
    return [{**c, "effort": effort_text(c["id"]), "effort_days": list(EFFORT_DAYS.get(c["id"], (0, 0))),
             "effort_note": EFFORT_NOTE.get(c["id"])} for c in CATALOG]


# ── einzelne Playbooks (immer auf Kopien, idempotent) ─────────────────────────

def _cat_flags(s: pd.Series, fn) -> np.ndarray:
    cats = pd.Series(s.cat.categories.astype(str))
    flags = cats.map(fn).to_numpy(dtype=bool)
    codes = s.cat.codes.to_numpy()
    return np.where(codes >= 0, flags[np.clip(codes, 0, None)], False)


def _is_storno(df):
    return _cat_flags(df["invoice"], lambda x: x.startswith("C"))


def _non_article(code: str) -> bool:
    u = code.upper()
    return u in NON_ARTICLE or not re.search(r"\d", u) or u.startswith(("TEST", "GIFT_"))


def pb06(df, opt, ctx):
    return df.drop_duplicates(subset=[c for c in df.columns if c in
                                      ("invoice", "stock_code", "description", "quantity", "invoice_date",
                                       "price_gbp", "customer_id", "country")]).reset_index(drop=True)


def pb03(df, opt, ctx):
    return df[~_cat_flags(df["stock_code"], _non_article)].reset_index(drop=True)


def pb07(df, opt, ctx):
    st = _is_storno(df)
    bad_price = (df["price_gbp"].to_numpy() == 0) & ~st
    bad_qty = (df["quantity"].to_numpy() < 0) & ~st
    huge = np.abs(df["quantity"].to_numpy()) >= 10_000
    flag = bad_price | bad_qty | huge
    if opt == "A":
        out = df.copy()
        out["pb07_flag"] = flag
        return out
    return df[~flag].reset_index(drop=True)


def pb04(df, opt, ctx):
    from .data_loader import _mode_codes
    out = df.copy()
    norm = out["stock_code"].astype(str).str.strip().str.upper()
    out["stock_code"] = norm.astype("category")
    desc = out["description"]
    k, v, _ = _mode_codes(out["stock_code"].cat.codes.to_numpy().astype(np.int64),
                          desc.cat.codes.to_numpy().astype(np.int64), len(desc.cat.categories))
    best = np.full(len(out["stock_code"].cat.categories), -1, dtype=np.int64)
    best[k] = v
    codes = best[out["stock_code"].cat.codes.to_numpy()]
    out["description"] = pd.Categorical.from_codes(codes, categories=desc.cat.categories).remove_unused_categories()
    return out


def pb02(df, opt, ctx):
    """Storno → Originalrechnung: letzter Kauf desselben Kunden und Artikels mit Menge ≥ Storno-Menge, ≤ 90 Tage davor."""
    out = df.copy()
    st = _is_storno(out)
    cust = out["customer_id"].to_numpy(dtype="float64", na_value=np.nan)
    has = ~np.isnan(cust)
    S = pd.DataFrame({"i": np.flatnonzero(st & has), })
    S["cust"], S["prod"] = cust[S["i"]], out["stock_code"].cat.codes.to_numpy()[S["i"]]
    S["time"], S["q"] = out["invoice_date"].to_numpy()[S["i"]], np.abs(out["quantity"].to_numpy()[S["i"]])
    B = pd.DataFrame({"j": np.flatnonzero(~st & has)})
    B["cust"], B["prod"] = cust[B["j"]], out["stock_code"].cat.codes.to_numpy()[B["j"]]
    B["time"], B["bq"] = out["invoice_date"].to_numpy()[B["j"]], out["quantity"].to_numpy()[B["j"]]
    B["inv"] = out["invoice"].astype(str).to_numpy()[B["j"]]
    m = pd.merge_asof(S.sort_values("time"), B.sort_values("time"), on="time", by=["cust", "prod"],
                      direction="backward", tolerance=pd.Timedelta(days=90), allow_exact_matches=True)
    m = m[m["inv"].notna() & (m["bq"] >= m["q"] * 0.5)]
    orig = np.full(len(out), None, dtype=object)
    orig[m["i"].to_numpy()] = m["inv"].to_numpy()
    out["orig_invoice"] = pd.Categorical(orig)
    ctx["pb02_match_rate"] = float(len(m) / max(len(S), 1))
    return out


def pb01(df, opt, ctx):
    if opt == "B":
        out = df.copy()
        miss = out["customer_id"].isna().to_numpy()
        pseudo = -(out["invoice"].cat.codes.to_numpy().astype(np.int64) + 1)
        cid = out["customer_id"].to_numpy(dtype="float64", na_value=np.nan)
        cid = np.where(miss, pseudo, cid)
        out["customer_id"] = pd.array(cid.astype(np.int64), dtype="Int32")
        out["is_guest"] = miss
        return out
    return df[df["customer_id"].notna()].reset_index(drop=True)


_DE = {"CHRISTMAS": "Weihnachten", "LIGHT": "Licht", "LIGHTS": "Licht", "CANDLE": "Kerzen", "HOLDER": "Halter",
       "BAG": "Taschen", "MUG": "Tassen", "CUP": "Tassen", "PLATE": "Teller", "BOWL": "Schalen", "CAKE": "Kuchen",
       "TIN": "Dosen", "BOX": "Boxen", "CARD": "Karten", "PAPER": "Papier", "WRAP": "Geschenkpapier",
       "HEART": "Herzen", "HANGING": "Hänger", "DECORATION": "Deko", "SIGN": "Schilder", "FRAME": "Rahmen",
       "CUSHION": "Kissen", "NECKLACE": "Ketten", "BRACELET": "Armbänder", "EARRINGS": "Ohrringe",
       "GARDEN": "Garten", "PARTY": "Party", "BOTTLE": "Flaschen", "WATER": "Wärmflaschen", "CLOCK": "Uhren",
       "KEY": "Schlüssel", "RING": "Ringe", "TEA": "Tee", "SET": "Sets", "VINTAGE": "Vintage", "STAR": "Sterne",
       "GLASS": "Glas", "LUNCH": "Lunchboxen", "JUMBO": "Jumbo-Taschen", "RETROSPOT": "Retrospot",
       "SPACEBOY": "Spaceboy", "BIRD": "Vögel", "FLOWER": "Blumen", "ROSE": "Rosen", "BABY": "Baby",
       "CHILDRENS": "Kinder", "PINK": "Rosa", "WHITE": "Weiß", "BLUE": "Blau", "RED": "Rot", "GREEN": "Grün",
       "WOODEN": "Holz", "METAL": "Metall", "ZINC": "Zink", "STICKER": "Sticker", "NOTEBOOK": "Notizbücher",
       "PEN": "Stifte", "PENCILS": "Stifte", "UMBRELLA": "Schirme", "SPOT": "Punkte", "DOORMAT": "Fußmatten",
       "HOT": "Wärmflaschen", "PURSE": "Geldbörsen", "BUNTING": "Wimpel", "ORNAMENT": "Ornamente",
       "LANTERN": "Laternen", "TRAY": "Tabletts", "JAR": "Gläser", "BASKET": "Körbe", "APRON": "Schürzen",
       "TOWEL": "Handtücher", "DOLLY": "Puppen", "GIFT": "Geschenke", "STRAWBERRY": "Erdbeere"}
STOP = {"OF", "AND", "THE", "WITH", "IN", "SET", "SMALL", "LARGE", "MINI", "ASSORTED", "DESIGN", "COLOUR", "COLOURS",
        "PACK", "PCS", "PIECE", "S", "A", "TO", "FOR", "ON", "IS"}


def _cluster_name(terms: list[str], used: set) -> str:
    de = []
    for t in terms:
        w = _DE.get(t)
        if w and w not in de:
            de.append(w)
    name = " & ".join(de[:2]) if de else " / ".join(t.title() for t in terms[:2])
    base, i = name, 2
    while name in used:
        name, i = f"{base} {i}", i + 1
    used.add(name)
    return name


def pb05(df, opt, ctx, k: int = 24):
    """Freitext → Artikelgruppen: TF-IDF über Beschreibungen, KMeans, Namen aus den stärksten Begriffen."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from .data_loader import _mode_codes
    out = df.copy()
    codes = out["stock_code"].cat.codes.to_numpy().astype(np.int64)
    desc = out["description"]
    kk, vv, _ = _mode_codes(codes, desc.cat.codes.to_numpy().astype(np.int64), len(desc.cat.categories))
    stock = out["stock_code"].cat.categories
    texts = pd.Series("", index=range(len(stock)), dtype=object)
    texts.iloc[kk] = np.asarray(desc.cat.categories.astype(str))[vv]
    non_art = np.array([_non_article(str(c)) for c in stock])
    use = (texts.str.len() > 0).to_numpy() & ~non_art
    vec = TfidfVectorizer(token_pattern=r"(?u)\b[A-Za-z]{3,}\b", stop_words=[w.lower() for w in STOP],
                          lowercase=True, min_df=3, sublinear_tf=True)
    X = vec.fit_transform(texts[use].to_numpy())
    km = KMeans(n_clusters=k, n_init=4, random_state=7).fit(X)
    terms = np.asarray(vec.get_feature_names_out())
    used: set = set()
    names = []
    for c in range(k):
        top = terms[np.argsort(-km.cluster_centers_[c])[:4]]
        names.append(_cluster_name([t.upper() for t in top], used))
    lab = np.full(len(stock), "Sonstiges", dtype=object)
    lab[np.flatnonzero(use)] = np.asarray(names)[km.labels_]
    lab[non_art] = "Gebühren & Porto"
    out["pb05_group"] = pd.Categorical(lab[codes])
    sizes = pd.Series(lab[np.flatnonzero(use)]).value_counts()
    ctx["pb05_groups"] = [{"name": n, "products": int(sizes.get(n, 0)),
                           "terms": [t for t in terms[np.argsort(-km.cluster_centers_[i])[:5]]]}
                          for i, n in enumerate(names)]
    return out


def pb08(df, opt, ctx):
    fx = float(ctx.get("fx", DEFAULT_FX))
    out = df.copy()
    out["price_eur"] = (out["price_gbp"].to_numpy() * fx).astype(np.float32)
    return out


APPLY = {"PB-01": pb01, "PB-02": pb02, "PB-03": pb03, "PB-04": pb04, "PB-05": pb05, "PB-06": pb06,
         "PB-07": pb07, "PB-08": pb08}


# ── Wirkung messen ────────────────────────────────────────────────────────────

def _stats(df: pd.DataFrame) -> dict:
    st = _is_storno(df)
    rev = (df["quantity"].to_numpy() * df["price_gbp"].to_numpy())
    return {"rows": int(len(df)), "customers": int(df["customer_id"].nunique()),
            "revenue_gbp": float(rev[~st].sum()), "products": int(df["stock_code"].nunique())}


def _effect(before: dict, after: dict, df_after: pd.DataFrame, pid: str, ctx: dict) -> dict:
    e = {"rows_before": before["rows"], "rows_after": after["rows"], "rows_removed": before["rows"] - after["rows"],
         "customers_before": before["customers"], "customers_after": after["customers"],
         "revenue_change_gbp": round(after["revenue_gbp"] - before["revenue_gbp"], 2),
         "products_before": before["products"], "products_after": after["products"]}
    if pid == "PB-02":
        e["match_rate"] = round(ctx.get("pb02_match_rate", 0), 4)
        e["matched_storno_rows"] = int(df_after["orig_invoice"].notna().sum())
    if pid == "PB-05":
        e["groups"] = ctx.get("pb05_groups", [])
    if pid == "PB-07" and "pb07_flag" in df_after:
        e["flagged_rows"] = int(df_after["pb07_flag"].sum())
    if pid == "PB-01" and "is_guest" in df_after:
        e["guest_rows"] = int(df_after["is_guest"].sum())
    if pid == "PB-08":
        e["fx"] = ctx.get("fx", DEFAULT_FX)
    return e


def normalize_steps(steps: list[dict]) -> list[dict]:
    """In feste Reihenfolge bringen, Standardoptionen ergänzen, nur Daten-Playbooks."""
    by = {s["id"]: s for s in steps if s.get("id") in APPLY}
    out = []
    for pid in ORDER:
        if pid in by:
            spec = next(c for c in CATALOG if c["id"] == pid)
            out.append({"id": pid, "option": by[pid].get("option") or spec.get("default", "A")})
    return out


def variant_id(steps: list[dict], fx: float = DEFAULT_FX) -> str:
    key = json.dumps({"steps": normalize_steps(steps), "fx": fx}, sort_keys=True)
    return "p" + hashlib.sha1(key.encode()).hexdigest()[:8]


def recommended_steps() -> list[dict]:
    return [{"id": pid, "option": next(c for c in CATALOG if c["id"] == pid)["default"]} for pid in ORDER]


def run_pipeline(raw: pd.DataFrame, steps: list[dict], fx: float = DEFAULT_FX, log=None) -> tuple[pd.DataFrame, list]:
    """Wendet die Playbooks auf eine Kopie an. Rohdaten bleiben unverändert."""
    df = raw
    effects = []
    ctx = {"fx": fx}
    for st in normalize_steps(steps):
        t0 = time.time()
        before = _stats(df)
        df = APPLY[st["id"]](df, st["option"], ctx)
        after = _stats(df)
        eff = _effect(before, after, df, st["id"], ctx)
        eff.update(id=st["id"], option=st["option"], seconds=round(time.time() - t0, 1))
        effects.append(eff)
        if log:
            log(f"{st['id']} ({st['option']}): {before['rows'] - after['rows']:+,} Zeilen".replace(",", "."))
    return df, effects


# ── Neu prüfen ────────────────────────────────────────────────────────────────

def recheck(df: pd.DataFrame, steps: list[dict], vid: str, governance: dict | None = None) -> dict:
    """Bereinigter Datenstand → Tabellen → Graph-Mapping → Profil → Checks (Stufe 2)."""
    from .checks import run_checks
    from .data_loader import TABLE_LABELS_ORII, build_online_retail_tables, get_registry
    from .graph_mapping import build_mapping
    from .prepare_data import raw_facts
    from .profiler import build_profile, events_from_online_retail
    reg = get_registry()
    flags = {s["id"] for s in normalize_steps(steps)}
    facts = raw_facts(df)
    tables, quality = build_online_retail_tables(df)
    extra = reg._online_retail_extra(facts=facts, quality=quality, flags=flags)
    extra["labels"] = TABLE_LABELS_ORII
    mapping = build_mapping(tables, dataset_id=f"online_retail_ii~{vid}", extra=extra)
    profile = build_profile(dataset_name=f"Online Retail II · {vid}", mapping=mapping, tables=tables,
                            events=events_from_online_retail(df), source=f"GraphReady Pipeline {vid}")
    profile["governance"] = dict(governance or {})
    res = run_checks(stage=2, view={}, profile=profile, dataset_id=f"online_retail_ii~{vid}")
    return {"facts": facts, "mapping_summary": mapping["summary"], "checks": _compact(res), "profile": profile}


def checks_with_governance(profile: dict, governance: dict, vid: str) -> dict:
    """PB-09: Checkliste ändert nur die Governance-Prüfung – direkt am gespeicherten Profil neu rechnen (< 1 s)."""
    from .checks import run_checks
    prof = dict(profile, governance=dict(governance or {}))
    return _compact(run_checks(stage=2, view={}, profile=prof, dataset_id=f"online_retail_ii~{vid}"))


def _compact(res: dict) -> dict:
    return {"counts": res["counts"], "avg_readiness": res["avg_readiness"],
            "use_cases": [{"id": u["id"], "name": u["name"], "readiness": u["readiness"], "max_stage": u["max_stage"],
                           "checks": [{"dim": c["dim"], "status": c["status"], "score": c["score"], "reason": c["reason"]}
                                      for c in u["checks"]]} for u in res["use_cases"]]}


def baseline_checks(governance: dict | None = None) -> dict:
    from .data_loader import get_registry
    if governance and any(governance.values()):
        return checks_with_governance(get_registry().profile("online_retail_ii"), governance, "v0")
    return _compact(get_registry().checks("online_retail_ii", 2))


def compare(before: dict, after: dict) -> list[dict]:
    """Vorher/Nachher je Use Case inkl. geänderter Prüfungen."""
    out = []
    bmap = {u["id"]: u for u in before["use_cases"]}
    for a in after["use_cases"]:
        b = bmap[a["id"]]
        dims = []
        for cb, ca in zip(b["checks"], a["checks"]):
            if cb["score"] != ca["score"] or cb["status"] != ca["status"]:
                dims.append({"dim": ca["dim"], "before": cb["score"], "after": ca["score"],
                             "status_before": cb["status"], "status_after": ca["status"], "reason": ca["reason"]})
        out.append({"id": a["id"], "name": a["name"],
                    "before": b["readiness"], "after": a["readiness"],
                    "delta": a["readiness"]["score"] - b["readiness"]["score"], "changed": dims})
    return out


def run_variant(steps: list[dict], fx: float = DEFAULT_FX, governance: dict | None = None,
                backtest: bool = False, ctx=None, reuse_backtest: bool = False) -> dict:
    """Komplette Neu-Prüfung einer Pipeline-Version (optional mit Rückblick)."""
    from .data_loader import get_registry
    t0 = time.time()
    vid = variant_id(steps, fx)
    raw = get_registry().raw_online_retail()
    log = ctx.log if ctx else None
    if ctx:
        ctx.progress(5, "Playbooks anwenden …")
    df, effects = run_pipeline(raw, steps, fx, log=log)
    if ctx:
        ctx.progress(40, "Checks neu (Stufe 2) …")
    rc = recheck(df, steps, vid, governance)
    result = {"variant": vid, "steps": normalize_steps(steps), "fx": fx, "governance": governance or {},
              "effects": effects, "after": rc, "before": baseline_checks(),
              "generated_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    result["comparison"] = compare(result["before"], rc["checks"])
    if backtest:
        from . import backtest as bt
        from . import evaluation as ev
        bt.register_variant(vid, df)
        result["backtest"] = {}
        for i, uc in enumerate(ev.UC_ORDER):
            if ctx:
                ctx.progress(50 + int(45 * i / len(ev.UC_ORDER)), f"Rückblick {uc} auf {vid} …")
            r = ev.load(uc, vid) if reuse_backtest else None
            if r is None:
                r = ev.run_use_case(uc, variant=vid)
                ev.save(r)
            result["backtest"][uc] = {"best": r["best"], "vorhersagekraft": r["vorhersagekraft"]}
    result["seconds"] = round(time.time() - t0, 1)
    save_variant(result)
    return result


def save_variant(res: dict) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"pipeline_{res['variant']}.json").write_text(json.dumps(res, ensure_ascii=False, default=str),
                                                                encoding="utf-8")


def load_variant(vid: str) -> dict | None:
    p = ARTIFACTS / f"pipeline_{vid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def single_effects() -> dict | None:
    p = ARTIFACTS / "playbook_effects.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def compute_single_effects(log=print) -> dict:
    """Wirkung jedes Daten-Playbooks einzeln (Standardoption) auf Rohdaten – für die Karten in Schritt 3."""
    from .data_loader import get_registry
    raw = get_registry().raw_online_retail()
    base = baseline_checks()
    out = {}
    for pid in ORDER:
        t0 = time.time()
        steps = [{"id": pid}]
        df, eff = run_pipeline(raw, steps)
        rc = recheck(df, steps, variant_id(steps))
        out[pid] = {"effect": eff[0], "comparison": compare(base, rc["checks"])}
        log(f"[playbooks] {pid} einzeln: {time.time() - t0:.0f} s")
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "playbook_effects.json").write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    return out


# ── PB-10: Datenanforderung ───────────────────────────────────────────────────

def data_request(checks_res: dict) -> dict:
    """Welche Tabellen/Felder für blockierte Use Cases aus dem ERP nötig sind – inkl. Mapping-Vorlage."""
    from .usecases import FIELDS
    rows, sections = [], []
    for u in checks_res["use_cases"]:
        need = [f for f in u["fields"] if f["state"] == "missing" and f["importance"] in ("pflicht", "messung")]
        if not need:
            continue
        items = []
        for f in need:
            spec = FIELDS[f["field"]]
            ex = ", ".join(spec.get("syn", [])[:5])
            items.append({"field": f["field"], "label": f["label"], "importance": f["importance"], "examples": ex})
            rows.append([u["name"], f["label"], f["importance"], ex, "", ""])
        sections.append({"use_case": u["id"], "name": u["name"], "fields": items, "label": u["label"]})
    md = ["# Datenanforderung – GraphReady", "",
          "Für die folgenden Use Cases fehlen Tabellen oder Felder im Export. Bitte die Quelle im ERP/CRM benennen.", ""]
    for sct in sections:
        md += [f"## {sct['name']}", f"Vorhersageziel: {sct['label']}", "", "| Feld | Pflicht | typische Spaltennamen |",
               "|---|---|---|"]
        md += [f"| {i['label']} | {i['importance']} | {i['examples']} |" for i in sct["fields"]]
        md.append("")
    md += ["Zusätzlich: Verknüpfungen über Schlüssel (z. B. Seriennummer → Charge → Lieferant) und Zeitstempel je Ereignis."]
    csv = ["use_case;feld;wichtigkeit;typische_spaltennamen;quelle_tabelle;quelle_spalte"] + \
          [";".join(r) for r in rows]
    return {"sections": sections, "markdown": "\n".join(md), "csv": "\n".join(csv)}
