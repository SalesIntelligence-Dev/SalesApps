"""
Readiness-Checks (Modul 3)
==========================
Neun Prüfdimensionen je Use Case (PDF S. 4–5). Die ersten fünf sind die
use-case-spezifische Kern-IP und zählen zusammen 70 %.

Stufe 1 (Strukturcheck): bekommt NUR die Schema-Sicht (Tabellen, Spalten,
  Rollen, Beziehungen) – keine Zahlen, keine Datenzeilen.
Stufe 2 (Profilcheck):   bekommt NUR das Profil (aggregierte Kennzahlen).
Stufe 3 (Rückblick):     Datenreife wie Stufe 2; Vorhersagekraft und
  Euro-Potenzial folgen mit dem Backtest (Prompt 3/4).
"""
from __future__ import annotations

import re

from .semantics import bind_fields, connected, personal_columns
from .usecases import FIELDS, USE_CASES

DIMENSIONS = [
    {"id": "structure", "name": "Struktur & Schlüssel", "short": "Struktur", "core": True, "weight": 14},
    {"id": "time", "name": "Zeit & Leakage", "short": "Zeit", "core": True, "weight": 14},
    {"id": "label", "name": "Label & Historie", "short": "Label", "core": True, "weight": 14},
    {"id": "model", "name": "Datenlage & Modellwahl", "short": "Modellwahl", "core": True, "weight": 14},
    {"id": "coverage", "name": "Abdeckung ähnlicher Fälle", "short": "Cold Start", "core": True, "weight": 14},
    {"id": "columns", "name": "Spaltenqualität", "short": "Spalten", "core": False, "weight": 7.5},
    {"id": "text", "name": "Freitext", "short": "Freitext", "core": False, "weight": 7.5},
    {"id": "size", "name": "Größe & Limits", "short": "Größe", "core": False, "weight": 7.5},
    {"id": "governance", "name": "Governance & Rückspielweg", "short": "Governance", "core": False, "weight": 7.5},
]

TABPFN_MAX_ROWS = 10_000
KUMO_MAX_MB = 30
LEAK_RE = re.compile(r"(status|storno|cancel|return|retour|^last|letzte|total|summe|score|flag|churn|kuendig|"
                     r"closed|resolved|deleted|beschreibungen)", re.I)
STAGE_NOTES = {
    1: "Strukturcheck: nur Tabellen- und Feldliste. Falsches Grün möglich – Felder können existieren, aber leer sein.",
    2: "Profilcheck: nur aggregierte Kennzahlen. Es verlassen keine personenbezogenen Daten das Unternehmen.",
    3: "Rückblick: zeitlich strikt getrennter Backtest – Vorhersagekraft je Use Case aus Schritt 5. Datenreife wie im Profilcheck.",
}


# ── Formatierung ──────────────────────────────────────────────────────────────

def _pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{x * 100:.{digits}f} %".replace(".", ",")


def _n(x) -> str:
    return "–" if x is None else f"{int(round(x)):,}".replace(",", ".")


def _status(score: float) -> str:
    return "ok" if score >= 75 else "warn" if score >= 50 else "bad"


def _res(dim: str, status: str, score: float | None, reason: str, metrics=None, actions=None, **kw) -> dict:
    return {"dim": dim, "status": status, "score": None if score is None else int(round(max(0, min(100, score)))),
            "reason": reason, "metrics": metrics or [], "actions": actions or [], **kw}


def _m(label: str, value) -> dict:
    return {"label": label, "value": value}


def _a(text: str, playbook: str | None = None) -> dict:
    return {"text": text, "playbook": playbook}


# ── Kontext ───────────────────────────────────────────────────────────────────

class Ctx:
    def __init__(self, stage: int, view: dict, profile: dict | None, bindings: dict):
        self.stage, self.view, self.profile, self.b = stage, view, profile, bindings
        self.ev = (profile or {}).get("events") or {}
        self.q = self.ev.get("quality", {})
        self.us = self.ev.get("usecase_stats", {})
        self.cold = self.ev.get("cold_start", {})
        self.tables = {t["name"]: t for t in (profile["tables"] if profile else view["tables"])}
        self.derived = {d["column"]: d["note"] for d in (view.get("derived_columns") or [])}

    @property
    def provisional(self) -> bool:
        return self.stage == 1

    def col(self, binding: dict | None) -> dict | None:
        if not binding or binding.get("via") != "column":
            return None
        t = self.tables.get(binding["table"])
        return next((c for c in (t or {}).get("columns", []) if c["name"] == binding["column"]), None)

    def bound_tables(self, uc: dict) -> set[str]:
        """Tabellen der gebundenen Felder plus deren Stammdaten (Ziele der Schlüssel-FKs)."""
        bound = {self.b[f]["table"] for f, *_ in uc["requires"] if self.b.get(f)}
        keys = {(self.b[f]["table"], self.b[f]["column"]) for f in ("customer_key", "product_key")
                if self.b.get(f) and any(x[0] == f for x in uc["requires"])}
        rels = (self.profile or {}).get("relations") or self.view.get("relations", [])
        for r in rels:
            if (r["from_table"], r["from_column"]) in keys and r.get("kind") != "missing":
                bound.add(r["to_table"])
        return bound


def _requirements(uc: dict, ctx: Ctx) -> dict:
    r = {"found": [], "missing": [], "substituted": [], "optional_missing": [], "measure_missing": []}
    for f, imp, alt, pb in uc["requires"]:
        if ctx.b.get(f):
            r["found"].append(f)
        elif imp == "pflicht" and alt and ctx.b.get(alt):
            r["substituted"].append((f, alt, pb))
        elif imp == "pflicht":
            r["missing"].append((f, pb))
        elif imp == "messung":
            r["measure_missing"].append((f, pb))
        else:
            r["optional_missing"].append(f)
    return r


def _label(f: str) -> str:
    return FIELDS[f]["label"]


# ── 1 Struktur & Schlüssel ────────────────────────────────────────────────────

def check_structure(uc: dict, ctx: Ctx) -> dict:
    req = _requirements(uc, ctx)
    n_req = sum(1 for _, imp, *_ in uc["requires"] if imp == "pflicht")
    n_found_req = n_req - len(req["missing"]) - len(req["substituted"])
    unlinked = []
    for a, b in uc["links"]:
        if ctx.b.get(a) and ctx.b.get(b):
            if not connected(ctx.view, ctx.b[a]["table"], ctx.b[b]["table"]):
                unlinked.append((a, b))
    score = (100 - 45 * len(req["missing"]) - 12 * len(req["substituted"]) - 3 * len(req["optional_missing"])
             - 8 * len(req["measure_missing"]) - 30 * len(unlinked))
    metrics = [_m("Pflichtfelder", f"{n_found_req} von {n_req}"),
               _m("Optionale Felder", f"{sum(1 for _, i, *_ in uc['requires'] if i == 'optional') - len(req['optional_missing'])} "
                                       f"von {sum(1 for _, i, *_ in uc['requires'] if i == 'optional')}")]
    actions, reasons = [], []
    for f, pb in req["missing"]:
        actions.append(_a(f"{_label(f)} aus ERP/CRM anbinden (Datenanforderung)", pb or "PB-10"))
    for f, alt, pb in req["substituted"]:
        reasons.append(f"{_label(f)} fehlt, ersetzbar über {_label(alt)}")
        actions.append(_a(f"{_label(f)} aus {_label(alt)} ableiten", pb))
    for f, pb in req["measure_missing"]:
        actions.append(_a(f"{_label(f)} erfassen, damit ein Label lernbar wird", pb))
    for a, b in unlinked:
        reasons.append(f"keine Schlüsselverbindung {_label(a)} ↔ {_label(b)}")
        actions.append(_a(f"Verknüpfung {_label(a)} ↔ {_label(b)} herstellen", "PB-10"))

    if ctx.stage >= 2 and "customer_key" in [f for f, imp, *_ in uc["requires"] if imp == "pflicht"]:
        null_share = ctx.q.get("null_share_customer")
        if null_share is not None:
            cov = 1 - null_share
            metrics.append(_m("Positionen mit Kundennummer", _pct(cov)))
            if cov < 0.95:
                score = min(score, 100 - (0.95 - cov) * 150)
                reasons.append(f"nur {_pct(cov)} der Positionen sind einem Kunden zugeordnet")
                actions.append(_a("Positionen ohne Kundennummer ausschließen oder als Gast-Kunden führen", "PB-01"))
    if ctx.stage >= 2 and uc["id"] == "returns_root_cause":
        rr = ctx.us.get("returns_root_cause") or {}
        if rr:
            metrics.append(_m("Stornos heuristisch zuordenbar (≤ 90 Tage)", _pct(rr.get("match_rate_90d"))))
    if ctx.stage >= 2:
        missing_rel = [r for r in (ctx.profile or {}).get("relations", []) if r.get("kind") == "missing"]
        if uc["id"] == "returns_root_cause" and missing_rel:
            metrics.append(_m("Fehlende Beziehung", missing_rel[0].get("label") or "Storno → Original"))

    if req["missing"]:
        status = "bad"
        reason = (f"Pflichtfeld fehlt: {', '.join(_label(f) for f, _ in req['missing'])} – "
                  "der Use Case ist so nicht abbildbar.")
    elif unlinked:
        status, reason = "bad", "Felder vorhanden, aber " + "; ".join(reasons) + "."
    else:
        status = _status(score)
        reason = ("Alle Pflichtfelder vorhanden und über Schlüssel verbunden." if not reasons
                  else reasons[0][0].upper() + "; ".join(reasons)[1:] + ".")
        if ctx.provisional and not reasons:
            reason = "Grundsätzlich abbildbar: alle Pflichtfelder vorhanden und verbunden."
    return _res("structure", status, score, reason, metrics, actions, provisional=ctx.provisional,
                requirements=req)


# ── 2 Zeit & Leakage ──────────────────────────────────────────────────────────

def _leakage_candidates(uc: dict, ctx: Ctx) -> list[dict]:
    out = []
    dt = ctx.b.get("doc_type")
    if dt and uc["id"] not in ("returns_root_cause",):
        out.append({"column": f"{dt['table']}.{dt['column']}",
                    "note": "Storno-/Retourenbelege entstehen nach dem Kauf – als Merkmal nur Belege vor dem Stichtag zählen"})
    tables = ctx.bound_tables(uc)
    for tname in tables:
        t = ctx.tables.get(tname) or {}
        for c in t.get("columns", []):
            full = f"{tname}.{c['name']}"
            roles = set(c.get("roles", []))
            if any(x["column"] == full for x in out):
                continue
            if full in ctx.derived:
                out.append({"column": full, "note": ctx.derived[full]})
            elif LEAK_RE.search(c["name"]) and not roles & {"PK", "FK", "Zeit"}:
                note = ("Status ohne Historie – wird nachträglich überschrieben"
                        if re.search("status", c["name"], re.I) else
                        "entsteht nach dem Ereignis – nur mit Zeitstempel vor dem Stichtag nutzen")
                out.append({"column": full, "note": note})
    return out


def check_time(uc: dict, ctx: Ctx) -> dict:
    tb = ctx.b.get("event_time")
    needs_time = any(f == "event_time" and imp == "pflicht" for f, imp, *_ in uc["requires"])
    if not tb:
        if needs_time:
            return _res("time", "bad", 10, "Kein Ereignis-Zeitstempel – kein zeitlicher Split und kein Label möglich.",
                        actions=[_a("Belegdatum / Ereigniszeit exportieren", "PB-10")], provisional=ctx.provisional)
        return _res("time", "warn", 60, "Kein Zeitstempel – Auswertung nur als Momentaufnahme.",
                    provisional=ctx.provisional)
    t = ctx.tables.get(tb["table"]) or {}
    leaks = _leakage_candidates(uc, ctx)
    score = 92 - min(24, 8 * len(leaks))
    metrics = [_m("Ereigniszeit", f"{tb['table']}.{tb['column']}"), _m("Leakage-Kandidaten", str(len(leaks)))]
    actions = []
    reason = "Ereigniszeit vorhanden – zeitlicher Split möglich."
    if t.get("only_modified_date"):
        score -= 30
        reason = "Nur Änderungsdatum vorhanden – wird überschrieben, Leakage-Risiko im Rückblick."
    if ctx.stage >= 2 and ctx.ev.get("time"):
        tm = ctx.ev["time"]
        n_cut = len(ctx.ev.get("cutoffs", []))
        metrics.insert(1, _m("Zeitraum", f"{tm['months']:.0f} Monate".replace(".", ",")))
        metrics.insert(2, _m("Stichtage (≥ 12 Monate Historie, 90 Tage Label)", str(n_cut)))
        if tm.get("months_without_events"):
            score -= 10
            metrics.append(_m("Monate ohne Ereignisse", str(tm["months_without_events"])))
        if n_cut < 3:
            score = min(score, 40)
            reason = f"Nur {n_cut} zeitliche Splits möglich – zu wenig Historie für einen belastbaren Rückblick."
        elif n_cut < 6:
            score = min(score, 65)
            reason = f"{n_cut} zeitliche Splits möglich – knapp für einen stabilen Rückblick."
        else:
            reason = f"Ereigniszeit vorhanden, {n_cut} zeitliche Splits mit 12 Monaten Historie möglich."
    if leaks:
        reason += f" {len(leaks)} Felder als Leakage-Kandidat markiert."
        actions.append(_a("Leakage-Kandidaten nur mit Zeitstempel vor dem Stichtag bzw. zum Stichtag neu berechnet nutzen"))
    return _res("time", _status(score), score, reason, metrics, actions, provisional=ctx.provisional,
                leakage=leaks)


# ── 3 Label & Historie ────────────────────────────────────────────────────────

def _windows(stats: dict) -> tuple[dict | None, int]:
    w = stats.get("windows") or []
    if not w:
        return None, 0
    return w[-1], sum(x["eligible"] for x in w[:-1])


def check_label(uc: dict, ctx: Ctx) -> dict:
    req = _requirements(uc, ctx)
    if req["missing"]:
        return _res("label", "bad", 10, "Label nicht berechenbar: " + ", ".join(_label(f) for f, _ in req["missing"])
                    + " fehlt.", actions=[_a("Fehlende Daten anfordern", "PB-10")], provisional=ctx.provisional)
    if not uc.get("learnable", True) and req["measure_missing"]:
        nl = uc.get("nolabel", {})
        return _res("label", "warn", 45,
                    nl.get("label", "Kein lernbares Label: Kampagnen- und Reaktionsdaten fehlen – nur ein Regelwerk über andere Scores."),
                    [_m("Fehlt", ", ".join(_label(f) for f, _ in req["measure_missing"]))],
                    [_a(nl.get("action", "Aktionen und Reaktionen im CRM protokollieren"), "PB-10")], provisional=ctx.provisional,
                    forced=True)
    if ctx.stage == 1:
        return _res("label", "ok", 80, "Label aus den Feldern berechenbar – Fallzahl erst im Profilcheck prüfbar.",
                    [_m("Label", uc["label"])], provisional=True)

    stats = ctx.us.get(uc["id"])
    min_pos = uc.get("min_positives", 200)
    months = (ctx.ev.get("time") or {}).get("months", 0)
    metrics, actions = [], []
    if not stats:
        return _res("label", "warn", 50, "Fallzahlen im Profil nicht enthalten – Profil mit Ereignisdaten erzeugen.",
                    provisional=False)
    if uc["id"] == "cross_selling" and stats.get("positives") is None:
        metrics += [_m("Aktive Kunden im Testfenster", _n(stats["eligible"])),
                    _m("Proxy: Kunden mit neuem Artikel", _n(stats["proxy_positives"]))]
        return _res("label", "warn", 58,
                    "Ohne Artikelgruppe nur ein Proxy-Label (neuer Artikel statt neuer Gruppe) – erst nach PB-05 exakt.",
                    metrics, [_a("Artikelgruppen aus Beschreibungen bilden", "PB-05")], provisional=False)
    if "windows" in stats:
        test, train = _windows(stats)
        p, e = test["positives"], test["eligible"]
        metrics += [_m("Positivfälle im Testfenster", _n(p)), _m("Grundgesamtheit", _n(e)),
                    _m("Positivrate", _pct(p / e if e else None)), _m("Trainingsfälle (frühere Fenster)", _n(train))]
        rate = p / e if e else 0
    elif uc["id"] == "demand_forecast":
        p = stats["series_ge52_weeks"]
        metrics += [_m("Reihen mit ≥ 52 aktiven Wochen", _n(p)), _m("Historie", f"{stats['history_weeks']} Wochen"),
                    _m("Nullwochen (Top 100)", _pct(stats.get("zero_share_top100")))]
        rate = None
    else:
        p = stats.get("positives") or 0
        metrics += [_m("Positivfälle im Testfenster", _n(p)), _m("Kunden im Testfenster", _n(stats.get("eligible")))]
        if "avg_products_per_customer" in stats:
            metrics.append(_m("Ø Artikel je Kunde", str(stats["avg_products_per_customer"]).replace(".", ",")))
        rate = None
    metrics.append(_m("Historie", f"{months:.0f} Monate"))
    score = 95 if p >= 2.5 * min_pos else 80 if p >= min_pos else 60 if p >= 0.5 * min_pos else 35
    if months < 12:
        score -= 30
    elif months < 18:
        score -= 10
    reason = f"{_n(p)} Positivfälle im letzten Fenster (Mindestens {_n(min_pos)}), {months:.0f} Monate Historie."
    if uc["entity"] != "customer":
        rate = None
    if rate is not None and rate > 0.4:
        reason += f" Positivrate {_pct(rate)} ist hoch – Definition schärfen (z. B. 120 Tage oder Umsatzrückgang > 50 %)."
        actions.append(_a("Label-Definition mit dem Vertrieb abstimmen"))
    elif rate is not None and rate < 0.02:
        reason += f" Positivrate {_pct(rate)} ist sehr niedrig – Fenster verlängern."
    return _res("label", _status(score), score, reason, metrics, actions, provisional=False)


# ── 4 Datenlage & Modellwahl („Beyond IID“) ───────────────────────────────────

def _train_rows(uc: dict, ctx: Ctx) -> int | None:
    s = ctx.us.get(uc["id"]) or {}
    if "windows" in s:
        return int(sum(x["eligible"] for x in s["windows"][:-1]))
    if uc["id"] == "recommendation":
        return int(s.get("positives") or 0) * max(1, len(ctx.ev.get("cutoffs", [])) - 1)
    if uc["id"] == "cross_selling":
        return int(s.get("eligible") or 0) * max(1, len(ctx.ev.get("cutoffs", [])) - 1)
    if uc["id"] == "demand_forecast":
        return int(s.get("series_ge26_weeks") or 0) * int(s.get("history_weeks") or 0)
    if uc["id"] == "margin":
        return int(ctx.ev.get("rows") or 0)
    return None


def check_model(uc: dict, ctx: Ctx) -> dict:
    req = _requirements(uc, ctx)
    if req["missing"]:
        return _res("model", "na", None, "Ohne Pflichtfelder keine Modellwahl.", provisional=ctx.provisional)
    if not uc.get("learnable", True) and req["measure_missing"] and uc.get("nolabel", {}).get("engine"):
        nl = uc["nolabel"]
        return _res("model", "na", None, nl["model"], provisional=ctx.provisional, engine=nl["engine"])
    if not uc.get("learnable", True) and req["measure_missing"]:
        return _res("model", "na", None, "Kein lernbares Label – statt eines Modells ein Regelwerk über die Scores "
                    "der anderen Use Cases.", provisional=ctx.provisional,
                    engine={"primary": "Regelwerk über Churn-, Cross-Selling-, Empfehlungs- und Neukunden-Scores",
                            "challengers": ["Uplift-Modell, sobald Reaktionsdaten vorliegen"],
                            "baseline": "Regelwerk", "reason": "Ohne Reaktionsdaten keine Trainingsbasis"})
    tables = ctx.bound_tables(uc)
    rel_tables = len(tables)
    engine = {"baseline": "Baum-Modell (LightGBM)", "challengers": []}
    facts = []
    if ctx.stage == 1:
        multi = rel_tables >= 3
        engine["primary"] = "Baum-Modell (LightGBM)"
        if uc["entity"] == "series":
            engine["challengers"].append("Aurora (Zeitreihen-Foundation-Model)")
        if multi:
            engine["challengers"].append("Relational (GNN / KumoRFM)")
        engine["challengers"].append("TFM (TabPFN) – Größe erst in Stufe 2 bekannt")
        reason = (f"{rel_tables} verknüpfte Tabellen im Pfad – relationales Signal möglich; Größe und Verschiebung "
                  "erst im Profilcheck klassifizierbar.")
        return _res("model", "ok", 75, reason, [_m("Tabellen im Pfad", str(rel_tables))], provisional=True,
                    engine=engine)

    rows = _train_rows(uc, ctx) or 0
    shift = (ctx.ev.get("shift") or {}).get("max")
    avg_orders = ctx.cold.get("avg_orders_per_customer")
    n_products = ctx.q.get("product_keys")
    grouped = bool(avg_orders and avg_orders > 1.5)
    high_card = bool(n_products and n_products > 1000)
    text_only = any(f == "product_group" for f, *_ in req["substituted"])
    facts = [_m("Trainingszeilen (geschätzt)", _n(rows)),
             _m("Zeitliche Verschiebung (PSI)", "–" if shift is None else f"{shift:.3f}".replace(".", ",")),
             _m("Gruppiert (Ø Belege je Kunde)", "–" if avg_orders is None else str(avg_orders).replace(".", ",")),
             _m("Kardinalität Artikel", _n(n_products)), _m("Freitext statt Kategorie", "ja" if text_only else "nein")]
    why = []
    if uc["entity"] == "series":
        engine["primary"] = "Baum-Modell (LightGBM)"
        engine["challengers"] = ["Aurora (Zeitreihen-Foundation-Model)", "saisonal-naiv als Untergrenze"]
        why.append("Zeitreihen mit Sondereffekten: Bäume mit Lag-Features gewinnen oft, Aurora als Zero-Shot-Vergleich")
    elif uc["id"] == "recommendation":
        engine["primary"] = "Relational (LightGCN / GNN)"
        engine["challengers"] = ["KumoRFM (In-Context)", "Baum + Graph-Features"]
        why.append("Rangliste über Kunde–Artikel-Graph: Mehrsprung-Signal (ähnliche Käufer) ist der Kern")
    else:
        tfm_fit = rows <= TABPFN_MAX_ROWS and (shift is None or shift < 0.1) and not grouped
        engine["primary"] = "TFM (TabPFN)" if tfm_fit else "Baum-Modell (LightGBM)"
        if rows > TABPFN_MAX_ROWS:
            why.append(f"{_n(rows)} Trainingszeilen > {_n(TABPFN_MAX_ROWS)} – TFM nur mit Stichprobe, Bäume skalieren")
        if grouped:
            why.append(f"wiederkehrende Kunden (Ø {str(avg_orders).replace('.', ',')} Belege) – gruppierte Daten, "
                       "laut „Beyond IID“ Vorteil Bäume")
        if shift is not None:
            why.append("geringe zeitliche Verschiebung" if shift < 0.1 else
                       f"deutliche Verschiebung (PSI {shift:.2f}) – Bäume robuster".replace(".", ","))
        if high_card and grouped:
            engine["challengers"].append("Relational (GNN / KumoRFM)")
            why.append(f"{_n(n_products)} Artikel im Co-Kauf-Graph – relationales Signal als Herausforderer")
        if not tfm_fit:
            engine["challengers"].append("TFM (TabPFN, Stichprobe ≤ 10.000)")
    engine["reason"] = "; ".join(why)
    score = 88 if rows >= 1000 else 62 if rows >= 200 else 30
    reason = f"Empfehlung: {engine['primary']}" + (f" – {why[0]}." if why else ".")
    return _res("model", _status(score), score, reason, facts, provisional=False, engine=engine)


# ── 5 Abdeckung ähnlicher Fälle (Cold Start) ──────────────────────────────────

def _cov_score(share: float) -> float:
    return 90 if share <= 0.25 else 72 if share <= 0.45 else 55 if share <= 0.65 else 35


def check_coverage(uc: dict, ctx: Ctx) -> dict:
    if ctx.stage == 1:
        hint = ("Artikelmerkmale vorhanden – helfen bei neuen Artikeln."
                if ctx.b.get("product_group") or ctx.b.get("product_text") else "")
        return _res("coverage", "na", None, ("Erst mit Kennzahlen prüfbar (Anteil Kunden/Artikel mit wenig Kontext). "
                                             + hint).strip(), provisional=True)
    c, metrics, actions = ctx.cold, [], []
    if not c:
        return _res("coverage", "na", None, "Keine Ereignis-Kennzahlen im Profil.")
    if uc["entity"] == "customer":
        share = c.get("share_lt3_orders", 0)
        metrics += [_m("Kunden mit < 3 Belegen", _pct(share)), _m("Nur ein Beleg", _pct(c.get("share_single_order"))),
                    _m("Neukunden letzte 90 Tage (Anteil an aktiven)", _pct(c.get("share_new_of_active")))]
        if uc["id"] == "lead_scoring":
            return _res("coverage", "warn", 60,
                        "Per Definition Cold Start: nur der erste Warenkorb als Kontext – Artikel- und Ländermerkmale "
                        "aus dem Graphen nutzen.", metrics,
                        [_a("Ähnliche Kunden über gemeinsame Artikel finden (Graph)")], provisional=False)
        if uc["id"] in ("recommendation", "cross_selling"):
            metrics += [_m("Neue Artikel im Testfenster", _pct(c.get("share_new_products_test"))),
                        _m("Artikel mit < 5 Verkäufen", _pct(c.get("share_products_lt5_sales")))]
            share = max(share, c.get("share_new_products_test") or 0)
        score = _cov_score(share)
        reason = f"{_pct(c.get('share_lt3_orders'))} der Kunden haben weniger als 3 Belege – wenig eigener Kontext."
        if score < 75:
            actions.append(_a("Kontext über ähnliche Kunden (Land, Artikel) statt nur eigene Historie"))
        return _res("coverage", _status(score), score, reason, metrics, actions, provisional=False)
    if uc["entity"] == "series":
        s = ctx.us.get("demand_forecast") or {}
        short = 1 - (s.get("series_ge52_weeks", 0) / max(s.get("series", 1), 1))
        vol_short = 1 - (s.get("volume_share_ge52") if s.get("volume_share_ge52") is not None else 1 - short)
        metrics += [_m("Reihen mit < 52 aktiven Wochen", _pct(short)), _m("Absatzmenge in diesen Reihen", _pct(vol_short)),
                    _m("Reihen gesamt", _n(s.get("series")))]
        score = _cov_score(vol_short)
        return _res("coverage", _status(score), score,
                    f"{_pct(short)} der Artikel haben weniger als ein Jahr Verkaufshistorie, tragen aber nur "
                    f"{_pct(vol_short)} der Menge – kurze Reihen über Artikelgruppen bündeln.",
                    metrics, [_a("Kurze Reihen über Artikelgruppen prognostizieren", "PB-05")], provisional=False)
    share = c.get("share_products_lt5_sales", 0)
    metrics += [_m("Artikel mit < 5 Verkäufen", _pct(share)), _m("Artikel gesamt", _n(c.get("products")))]
    score = _cov_score(share)
    return _res("coverage", _status(score), score,
                f"{_pct(share)} der Artikel haben weniger als 5 Verkäufe – Kohorten über Artikelgruppen bilden.",
                metrics, provisional=False)


# ── 6 Spaltenqualität ─────────────────────────────────────────────────────────

def check_columns(uc: dict, ctx: Ctx) -> dict:
    used = {f for f, *_ in uc["requires"] if ctx.b.get(f)}
    if ctx.stage == 1:
        bad = []
        for f in used:
            col, kind = ctx.col(ctx.b[f]), FIELDS[f].get("kinds")
            if col and kind == "numeric" and col.get("kind") not in ("int", "float", "integer", "number"):
                bad.append(f)
        if bad:
            return _res("columns", "warn", 60, "Unpassende Typen: " + ", ".join(_label(f) for f in bad) + ".",
                        provisional=True)
        return _res("columns", "ok", 82, "Spaltentypen plausibel – Füllgrad und Werte erst im Profilcheck prüfbar.",
                    provisional=True)
    q, rows = ctx.q, max(ctx.ev.get("rows") or 1, 1)
    score, metrics, actions = 100.0, [], []

    def pen(cond, share, factor, cap, label, text, pb):
        nonlocal score
        if cond and share > 0:
            score -= min(cap, share * factor)
            metrics.append(_m(label, _pct(share, 2)))
            actions.append(_a(text, pb))

    pen(True, q.get("duplicate_share", 0), 300, 12, "Exakte Dubletten", "Exakte Dubletten entfernen", "PB-06")
    pen("product_key" in used, q.get("non_article_rows", 0) / rows, 400, 8, "Nicht-Artikel-Positionen",
        "Porto, Gebühren und manuelle Buchungen ausfiltern", "PB-03")
    pen("price" in used, q.get("price_zero_rows", 0) / rows, 400, 8, "Preis 0", "Positionen mit Preis 0 markieren",
        "PB-07")
    pen("quantity" in used, q.get("negative_qty_without_storno", 0) / rows, 400, 8, "Negative Menge ohne Storno",
        "Bestandskorrekturen markieren", "PB-07")
    pk = max(q.get("product_keys") or 1, 1)
    pen("product_key" in used, q.get("product_case_variants", 0) / pk, 60, 6, "Artikelnummern-Varianten (Groß/klein)",
        "Artikelstamm konsolidieren", "PB-04")
    if q.get("products_with_text"):
        pen(bool({"product_text", "product_group"} & used) or uc["id"] in ("cross_selling", "recommendation"),
            q.get("products_multi_text", 0) / q["products_with_text"], 30, 8, "Artikel mit mehreren Beschreibungen",
            "Beschreibungen je Artikel vereinheitlichen", "PB-04")
    for f in used - {"customer_key"}:
        col = ctx.col(ctx.b[f])
        if col and (col.get("null_rate") or 0) > 0.05:
            score -= 15 if col["null_rate"] > 0.2 else 8
            metrics.append(_m(f"Leer: {_label(f)}", _pct(col["null_rate"])))
    price_b = ctx.b.get("price")
    if "price" in used and price_b and re.search(r"(gbp|usd|chf|eur)", price_b["column"], re.I):
        cur = re.search(r"(gbp|usd|chf|eur)", price_b["column"], re.I).group(1).upper()
        if cur != "EUR":
            metrics.append(_m("Währung", cur))
            actions.append(_a(f"Preise von {cur} in EUR umrechnen", "PB-08"))
    reason = ("Spalten weitgehend sauber." if score >= 90 else
              f"{len([a for a in actions if a['playbook']])} Spaltenprobleme mit Reparaturvorschlag.")
    return _res("columns", _status(score), score, reason, metrics, actions, provisional=False)


# ── 7 Freitext ────────────────────────────────────────────────────────────────

def check_text(uc: dict, ctx: Ctx) -> dict:
    req = _requirements(uc, ctx)
    text_sub = [x for x in req["substituted"] if x[1] == "product_text"]
    tcol = ctx.col(ctx.b.get("product_text"))
    metrics = []
    if tcol and ctx.stage >= 2:
        metrics = [_m("Ø Länge", f"{tcol.get('avg_len', 0):.0f} Zeichen"), _m("Varianten", _n(tcol.get("n_unique")))]
    if text_sub:
        return _res("text", "warn", 55,
                    "Keine Artikelgruppen – die Beschreibung (Freitext) ist die einzige Artikelinfo. Relationale "
                    "Modelle lesen Freitext oft nicht.", metrics,
                    [_a("Beschreibungen per Clustering oder Sprachmodell in Kategorien überführen", "PB-05")],
                    provisional=ctx.provisional)
    if uc["id"] == "warranty_root_cause" and not ctx.b.get("defect_code"):
        return _res("text", "warn", 60, "Fehlercodes fehlen – oft nur als Freitext im Servicebericht vorhanden.",
                    actions=[_a("Fehlerbeschreibungen kategorisieren", "PB-05")], provisional=ctx.provisional)
    uses_text = any(f in ("product_text",) for f, *_ in uc["requires"]) and tcol
    if uses_text and not ctx.b.get("product_group"):
        return _res("text", "ok", 80, "Beschreibung nur optional genutzt – Kategorien würden das Modell stärken.",
                    metrics, [_a("Artikelgruppen ableiten", "PB-05")], provisional=ctx.provisional)
    return _res("text", "ok", 100, "Kein Freitext im Pfad des Use Cases.", metrics, provisional=ctx.provisional)


# ── 8 Größe & Limits ──────────────────────────────────────────────────────────

def check_size(uc: dict, ctx: Ctx) -> dict:
    if ctx.stage == 1:
        return _res("size", "na", None, "Zeilenzahlen sind im Strukturcheck unbekannt.", provisional=True)
    rows = _train_rows(uc, ctx)
    tables = ctx.bound_tables(uc)
    mb = sum((ctx.tables[t].get("rows") or 0) * (ctx.tables[t].get("n_columns") or len(ctx.tables[t]["columns"])) * 8
             for t in tables if t in ctx.tables) / 1e6
    score, metrics, actions, notes = 100, [], [], []
    metrics += [_m("Trainingszeilen (geschätzt)", _n(rows)), _m("Relationale Daten (geschätzt)", f"{mb:.0f} MB")]
    if rows is not None and rows > TABPFN_MAX_ROWS:
        score -= 12
        notes.append(f"TabPFN: Kontext max. {_n(TABPFN_MAX_ROWS)} Zeilen → Stichprobe")
    if mb > KUMO_MAX_MB:
        score -= 12
        notes.append(f"KumoRFM: {KUMO_MAX_MB} MB je Anfrage → Zeitfenster oder Stichprobe")
    if rows is not None and rows < 500:
        score = 50
        notes.append("sehr wenige Trainingsfälle")
    for n in notes:
        actions.append(_a(n))
    reason = ("Größe passt zu allen Engines." if not notes else "; ".join(notes) + ".")
    return _res("size", _status(score), score, reason, metrics, actions, provisional=False)


# ── 9 Governance & Rückspielweg ───────────────────────────────────────────────

CHECKLIST = ["Pseudonymisierung der Kundennummern", "Speicherort Deutschland / EU", "Betriebsrat informiert",
             "Schreibrechte für den Rückspielweg ins CRM/ERP", "Lösch- und Aufbewahrungskonzept"]


def check_governance(uc: dict, ctx: Ctx) -> dict:
    personal, sensitive = personal_columns(ctx.view)
    uses_customer = bool(ctx.b.get("customer_key")) and any(f == "customer_key" for f, *_ in uc["requires"])
    metrics = [_m("Personenbezogene Felder im Schema", str(len(personal) + len(sensitive)))]
    reasons = []
    score = 75
    if uses_customer:
        score = 62
        reasons.append("Kundennummer ist pseudonym, aber personenbezogen (DSGVO)")
    if sensitive:
        score = min(score, 52)
        reasons.append(f"sensible Felder im Schema ({', '.join(s.split('.')[-1] for s in sensitive[:3])}) – "
                       "nicht in den Export übernehmen")
    if personal:
        metrics.append(_m("Beispiele", ", ".join(p.split(".")[-1] for p in personal[:4])))
    gov = (ctx.profile or {}).get("governance") or {}
    done = [i for i in CHECKLIST if gov.get(i)]
    if done:
        score = min(90, score + 6 * len(done))
        metrics.append(_m("Checkliste (PB-09)", f"{len(done)} von {len(CHECKLIST)} erledigt"))
    if gov.get("Schreibrechte für den Rückspielweg ins CRM/ERP"):
        reasons.append("Rückspielweg ins CRM geklärt")
        metrics.append(_m("Rückspielweg", "geklärt"))
    else:
        reasons.append("Rückspielweg ins CRM noch nicht eingerichtet")
        metrics.append(_m("Rückspielweg", "nicht vorhanden"))
    text = "; ".join(reasons)
    return _res("governance", _status(score), score, text[:1].upper() + text[1:] + ".", metrics,
                [_a("Freigabe-Checkliste abarbeiten", "PB-09")], provisional=ctx.provisional, checklist=CHECKLIST)


CHECKS = [check_structure, check_time, check_label, check_model, check_coverage, check_columns, check_text,
          check_size, check_governance]


# ── Aggregation ───────────────────────────────────────────────────────────────

def readiness(uc: dict, results: list[dict], stage: int) -> dict:
    dims = {d["id"]: d for d in DIMENSIONS}
    num = den = 0.0
    for r in results:
        if r["status"] != "na" and r["score"] is not None:
            w = dims[r["dim"]]["weight"]
            num += w * r["score"]
            den += w
    score = num / den if den else 0
    by = {r["dim"]: r for r in results}
    core_bad = [d for d in ("structure", "time", "label", "model", "coverage") if by[d]["status"] == "bad"]
    other_bad = [d for d in ("columns", "text", "size", "governance") if by[d]["status"] == "bad"]
    if score < 50 or by["structure"]["status"] == "bad" or by["label"]["status"] == "bad":
        status = "bad"
    elif score < 75 or core_bad or other_bad:
        status = "warn"
    else:
        status = "ok"
    if uc.get("cap") == "warn" and status == "ok":
        status = "warn"
    if status == "bad":
        why = ("Pflichtdaten fehlen" if by["structure"]["status"] == "bad" else
               "Label nicht berechenbar" if by["label"]["status"] == "bad" else "zu viele Lücken")
    elif status == "warn":
        weak = [dims[r["dim"]]["short"] for r in results if r["status"] in ("warn", "bad")]
        why = "Schwachstellen: " + ", ".join(weak[:4])
    else:
        why = "Datenlage trägt den Use Case"
    return {"score": int(round(score)), "status": status, "provisional": stage == 1, "reason": why}


def max_stage(results: list[dict], uc: dict) -> int:
    by = {r["dim"]: r for r in results}
    if by["structure"]["status"] == "bad":
        return 1
    if not uc.get("learnable", True) or by["label"]["status"] == "bad":
        return 2
    return 3


def column_ampel(ctx: Ctx) -> list[dict]:
    """Spalten-Ampel je Tabelle mit konkreten Reparaturen."""
    by_col = {(b["table"], b["column"]): f for f, b in ctx.b.items() if b and b.get("via") == "column"}
    q, rows = ctx.q, max(ctx.ev.get("rows") or 1, 1)
    out = []
    for tname, t in ctx.tables.items():
        cols = []
        for c in t["columns"]:
            field = by_col.get((tname, c["name"]))
            issues, status = [], "ok"
            nr = c.get("null_rate")
            roles = c.get("roles", [])
            if ctx.stage >= 2 and nr is not None and nr > 0.05:
                status = "bad" if nr > 0.2 else "warn"
                issues.append({"text": f"{_pct(nr)} leer", "playbook": "PB-01" if field == "customer_key" else None})
            if "Freitext" in roles:
                status = "warn" if status == "ok" else status
                issues.append({"text": "Freitext – in Kategorien überführen", "playbook": "PB-05"})
            if "Hohe Kardinalität" in roles and "FK" not in roles and "PK" not in roles:
                issues.append({"text": "sehr viele Ausprägungen – als Knotentyp führen", "playbook": None})
            full = f"{tname}.{c['name']}"
            if full in ctx.derived:
                status = "warn" if status == "ok" else status
                issues.append({"text": ctx.derived[full] + " (Leakage-Kandidat)", "playbook": None})
            if ctx.stage >= 2 and field == "product_key":
                if q.get("product_case_variants"):
                    status = "warn" if status == "ok" else status
                    issues.append({"text": f"{_n(q['product_case_variants'])} Varianten nur in Groß-/Kleinschreibung",
                                   "playbook": "PB-04"})
                if q.get("non_article_rows"):
                    issues.append({"text": f"{_n(q['non_article_rows'])} Positionen mit Nicht-Artikel-Codes",
                                   "playbook": "PB-03"})
            if ctx.stage >= 2 and field == "price":
                if q.get("price_zero_rows"):
                    status = "warn" if status == "ok" else status
                    issues.append({"text": f"{_n(q['price_zero_rows'])} × Preis 0", "playbook": "PB-07"})
                if re.search(r"gbp|usd|chf", c["name"], re.I):
                    issues.append({"text": "Fremdwährung – in EUR umrechnen", "playbook": "PB-08"})
            if ctx.stage >= 2 and field == "quantity" and q.get("negative_qty_without_storno"):
                status = "warn" if status == "ok" else status
                issues.append({"text": f"{_n(q['negative_qty_without_storno'])} negative Mengen ohne Storno",
                               "playbook": "PB-07"})
            if ctx.stage >= 2 and field == "product_text" and q.get("products_multi_text"):
                status = "warn" if status == "ok" else status
                issues.append({"text": f"{_n(q['products_multi_text'])} Artikel mit mehreren Beschreibungen",
                               "playbook": "PB-04"})
            if ctx.stage == 1 and not issues:
                status = "na"
            cols.append({"name": c["name"], "kind": c.get("kind"), "roles": roles, "field": field,
                         "field_label": FIELDS[field]["label"] if field else None, "null_rate": nr,
                         "status": status, "issues": issues})
        out.append({"table": tname, "label": t.get("label", tname), "rows": t.get("rows"), "columns": cols})
    return out


def run_checks(*, stage: int, view: dict, profile: dict | None = None, use_case_ids: list[str] | None = None,
               dataset_id: str | None = None) -> dict:
    """Stufe 1: nur `view`. Stufe ≥ 2: nur `profile` (die Schema-Sicht stammt dann aus dem Profil)."""
    eff_stage = min(stage, 2) if profile is not None else 1
    if eff_stage >= 2:
        from .profiler import profile_to_mapping
        from .semantics import schema_view
        view = schema_view(profile_to_mapping(profile))
        view["derived_columns"] = profile.get("derived_columns", [])
        bindings = profile.get("bindings") or bind_fields(view)
    else:
        bindings = bind_fields(view)
    ctx = Ctx(eff_stage, view, profile if eff_stage >= 2 else None, bindings)
    ucs = [u for u in USE_CASES if not use_case_ids or u["id"] in use_case_ids]
    results = []
    for uc in ucs:
        rs = [fn(uc, ctx) for fn in CHECKS]
        ready = readiness(uc, rs, eff_stage)
        engine = next((r.get("engine") for r in rs if r.get("engine")), None)
        req = rs[0].pop("requirements")
        results.append({
            "id": uc["id"], "name": uc["name"], "area": uc["area"], "lighthouse": uc["lighthouse"],
            "label": uc["label"], "label_detail": uc["label_detail"], "baseline": uc["baseline"],
            "lever": uc["lever"], "blockers": uc["blockers"], "app_route": uc["app_route"],
            "app_name": uc["app_name"], "note": uc.get("note"),
            "fields": [{"field": f, "label": _label(f), "importance": imp, "playbook": pb,
                        "binding": bindings.get(f), "alt": alt, "alt_label": _label(alt) if alt else None,
                        "state": ("found" if f in req["found"] else
                                  "substituted" if any(x[0] == f for x in req["substituted"]) else "missing")}
                       for f, imp, alt, pb in uc["requires"]],
            "checks": rs, "readiness": ready, "engine": engine, "max_stage": max_stage(rs, uc),
            "ampel": {
                "datenreife": {"status": ready["status"], "score": ready["score"], "provisional": ready["provisional"]},
                "vorhersagekraft": {"status": "pending", "note": "Rückblick (Stufe 3) – folgt mit dem Backtest"},
                "euro": {"status": "pending", "note": "Was-wäre-wenn – folgt mit den Fix-Playbooks"},
                "gesamt": {"status": ready["status"], "provisional": True,
                           "note": "Vorläufig: bisher ist nur die Datenreife belastbar"},
            },
        })
    counts = {s: sum(1 for r in results if r["readiness"]["status"] == s) for s in ("ok", "warn", "bad")}
    downgraded = stage > eff_stage and not (stage == 3 and eff_stage == 2)
    if stage == 3 and eff_stage == 2:
        note = STAGE_NOTES[3]
    elif downgraded:
        note = (f"Stufe {stage} braucht Daten oder ein Profil – angezeigt wird Stufe {eff_stage}. "
                + STAGE_NOTES[eff_stage])
    else:
        note = STAGE_NOTES[eff_stage]
    return {
        "dataset": dataset_id, "stage": stage, "stage_used": eff_stage, "stage_note": note,
        "downgraded": downgraded,
        "dimensions": DIMENSIONS, "use_cases": results, "counts": counts,
        "avg_readiness": int(round(sum(r["readiness"]["score"] for r in results) / max(len(results), 1))),
        "bindings": bindings, "columns": column_ampel(ctx),
        "privacy": (profile or {}).get("privacy") if eff_stage >= 2 else None,
        "profile_meta": ({"generated_at": profile.get("generated_at"), "source": profile.get("source"),
                          "dataset_name": profile.get("dataset_name")} if eff_stage >= 2 else None),
    }
