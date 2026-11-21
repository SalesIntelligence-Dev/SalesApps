"""
Aktivierung (Modul 6): Aktionslisten, Next Best Action, Kontrollgruppe, Export, Monitoring
=========================================================================================
Stichtag der Aktivierung ist der Tag nach dem letzten Beleg (10.12.2011). Trainiert wird auf allen
Stichtagen, deren 90-Tage-Label vollständig beobachtet ist (bis 01.09.2011); bewertet wird der
Stichtag selbst – ohne Label. Die Wahrscheinlichkeiten werden isotonisch auf dem Rückblick-Testfenster
kalibriert (gleiche Engine, ohne diesen Stichtag trainiert, Labels bekannt).

  Lehre aus dem Kumo-Praxistest (PDF): zu viele falsche „sichere“ Vorhersagen.
  Deshalb: kalibrierte Schwellen aus Kosten und Nutzen, markierte Unsicherheit, Mensch in der Schleife.

Der Rückspielweg ins CRM ist nur eine Vorschau – an externe Systeme wird nichts gesendet.

  .venv/bin/python -m graph_ml.precompute --only-activation      # mit GPU-Engines (GNN, TabPFN)
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import backtest as bt
from . import business_case as bc

log = logging.getLogger("graph_ml.activation")

ARTIFACTS = Path(__file__).parent / "artifacts"
T_NOW = pd.Timestamp("2011-12-10")                       # Tag nach dem letzten Beleg (09.12.2011)
ANCHOR = {"churn": T_NOW, "lead_scoring": T_NOW, "cross_selling": T_NOW,
          "returns_root_cause": pd.Timestamp("2011-11-01")}  # Retouren: Positionen ab 01.11.2011
LIST_UCS = ["churn", "lead_scoring", "cross_selling", "returns_root_cause"]
CONTROL_SHARE = 0.20
SEED = 7
SURE = 0.80                                              # ab hier gilt eine Vorhersage als „sicher“
GRENZ_ABS, GRENZ_REL = 0.03, 0.25                        # Grenzfall: |p − Schwelle| ≤ max(3 Pp., 25 % der Schwelle)
MAX_ROWS = {"returns_root_cause": 2000}
FALLBACK = ["lgbm_graph", "lgbm", "rule"]

UC_META = {
    "churn": {"name": "Churn & Retention", "entity": "Kunde", "event": "Abwanderung",
              "action": "Retention-Anruf (Außendienst)", "channel": "Telefon", "effect_sign": -1,
              "effect_text": "weniger Abwanderer"},
    "lead_scoring": {"name": "Neukunden-Scoring", "entity": "Neukunde", "event": "Wiederkauf in 90 Tagen",
                     "action": "Zweitkauf-Impuls (Willkommens-Anruf)", "channel": "Telefon", "effect_sign": 1,
                     "effect_text": "mehr Wiederkäufer"},
    "cross_selling": {"name": "Graph Cross-Selling", "entity": "Kunde × Gruppe", "event": "Kauf in neuer Gruppe",
                      "action": "Cross-Selling-Angebot", "channel": "E-Mail/Webshop", "effect_sign": 1,
                      "effect_text": "mehr Käufe in neuen Gruppen"},
    "returns_root_cause": {"name": "Retouren- & Storno-Risiko", "entity": "Position", "event": "Storno binnen 30 Tagen",
                           "action": "Position vor Versand prüfen", "channel": "Innendienst", "effect_sign": -1,
                           "effect_text": "weniger Stornos"},
}
UNC_TEXT = {"grenzfall": "Grenzfall: Wahrscheinlichkeit nahe der Schwelle",
            "uneinig": "Engines uneinig (Rang weicht vom Baum-Modell ab)",
            "wenig_historie": "wenig Historie – ähnliche Fälle selten"}

_ANCHOR_LOCK = threading.RLock()
_CACHE: dict = {}


# ── Stichtag verschieben ──────────────────────────────────────────────────────

@contextmanager
def anchored(T: pd.Timestamp):
    """Aufgaben zum Aktivierungs-Stichtag bauen: Backtest-Code sieht T als „Test“-Stichtag.

    Nur für die Vorberechnung und lokale Jobs – der Lock verhindert parallele Umstellungen.
    """
    with _ANCHOR_LOCK:
        old = bt.T_TEST
        bt.T_TEST = T
        try:
            yield
        finally:
            bt.T_TEST = old


def _variant() -> tuple[str, str]:
    """Datenstand der Aktivierung: empfohlenes Playbook-Paket (v1), falls vorberechnet, sonst Rohdaten v0."""
    p = ARTIFACTS / "pipeline_recommended.json"
    if p.exists():
        vid = json.loads(p.read_text())["variant"]
        if (ARTIFACTS / f"backtest_churn~{vid}.json").exists():
            return vid, "v1 nach Fix-Playbooks"
    return "v0", "v0 Rohdaten"


def _ensure_variant(vid: str) -> None:
    if vid == "v0" or vid in bt._VARIANTS:
        return
    from . import playbooks as pb
    from .data_loader import get_registry
    meta = pb.load_variant(vid)
    df, _ = pb.run_pipeline(get_registry().raw_online_retail(), meta["steps"], meta.get("fx", pb.DEFAULT_FX))
    bt.register_variant(vid, df)


def _task_now(uc: str, vid: str) -> bt.Task:
    b = bt.get_base(vid)
    with anchored(ANCHOR[uc]):
        task = bt.BUILDERS[uc](b)
    task.extra.update(base_key=vid, currency=b.currency, anchor=ANCHOR[uc].date().isoformat())
    return task


# ── Engines, Kalibrierung, Gründe ─────────────────────────────────────────────

def _make(task, eid: str):
    from .engines import make_engines
    engines = {e.id: e for e in make_engines(task)}
    notes = []
    for cand in [eid] + [f for f in FALLBACK if f != eid]:
        e = engines.get(cand)
        if e is None:
            continue
        ok, why = e.available()
        if ok:
            if cand != eid:
                notes.append(f"{eid} hier nicht verfügbar ({why}) – Fallback {e.name}")
            return e, notes
        notes.append(f"{cand}: {why}")
    raise RuntimeError("keine Engine verfügbar")


def _fit_predict(task, eid: str, T: pd.Timestamp | None = None):
    e, notes = _make(task, eid)
    if T is None:
        e.timed_fit(task)
        s = np.asarray(e.predict(task), dtype=float)
    else:
        with anchored(T):                       # GNN & Co. lesen den Stichtag beim Rechnen
            e.timed_fit(task)
            s = np.asarray(e.predict(task), dtype=float)
    return e, s, notes


def _iso_fit(s: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
    return iso


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def calibrate(s: np.ndarray, y: np.ndarray, proba: bool) -> dict:
    """Isotonische Kalibrierung auf dem Rückblick-Testfenster; Güte per 2-fach-Kreuzanpassung (kein In-Sample-Lob)."""
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, 2, len(s))
    p_cf = np.zeros(len(s))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        p_cf[te] = _iso_fit(s[tr], y[tr]).predict(s[te])
    iso = _iso_fit(s, y)

    def sure(p):
        m = p >= SURE
        return {"n": int(m.sum()), "share": round(float(m.mean()), 4),
                "observed": round(float(y[m].mean()), 4) if m.any() else None}

    raw = np.clip(s, 0, 1) if proba else None
    return {"x": [round(float(v), 6) for v in iso.X_thresholds_], "y": [round(float(v), 5) for v in iso.y_thresholds_],
            "n": int(len(s)), "base_rate": round(float(y.mean()), 4),
            "ece_before": round(_ece(raw, y), 4) if raw is not None else None, "ece_after": round(_ece(p_cf, y), 4),
            "sure_before": sure(raw) if raw is not None else None, "sure_after": sure(p_cf),
            "method": "isotonisch, kreuzangepasst (2 Hälften)"}


P_MIN, P_MAX = 0.005, 0.995                              # nie „100 % sicher“ – auch die oberste Stufe kann irren


def apply_calibration(cal: dict, s: np.ndarray) -> np.ndarray:
    return np.clip(np.interp(s, np.asarray(cal["x"]), np.asarray(cal["y"])), P_MIN, P_MAX)


def _contrib(engine, task) -> np.ndarray:
    from .engines.cpu import _frame
    X = _frame(task.test, engine.cols, task.cat_features)
    return engine.model.predict(X, num_iteration=engine.best_iter, pred_contrib=True)[:, :-1]


def _pct_rank(s: np.ndarray) -> np.ndarray:
    return pd.Series(s).rank(pct=True, method="average").to_numpy()


# ── Klartext-Gründe ───────────────────────────────────────────────────────────

def _eur(v, fx=1.0) -> str:
    return f"{float(v) * fx:,.0f} €".replace(",", ".")


def _pct(v) -> str:
    return f"{float(v) * 100:.0f} %"


def reason_text(feat: str, v, ctx: dict) -> str:
    """Ein Merkmal mit Wert als verständlicher Grund (Deutsch)."""
    fx = ctx.get("fx", 1.0)
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = float("nan")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{bt.FEATURE_LABELS.get(feat, feat)}: unbekannt"
    names = ctx.get("country_names")
    groups = ctx.get("group_names")
    T = {
        "recency": lambda: "letzter Kauf gestern" if f <= 1 else f"seit {f:.0f} Tagen kein Kauf",
        "freq_30": lambda: (f"1 Bestellung in 30 Tagen" if f == 1 else f"{f:.0f} Bestellungen in 30 Tagen"),
        "freq_90": lambda: (f"1 Bestellung in 90 Tagen" if f == 1 else f"{f:.0f} Bestellungen in 90 Tagen"),
        "freq_365": lambda: (f"1 Bestellung in 12 Monaten" if f == 1 else f"{f:.0f} Bestellungen in 12 Monaten"),
        "monetary_90": lambda: f"Umsatz 90 Tage {_eur(f, fx)}",
        "monetary_365": lambda: f"Umsatz 12 Monate {_eur(f, fx)}",
        "avg_basket": lambda: f"Ø Warenkorb {_eur(f, fx)}",
        "tenure": lambda: f"Kunde seit {f:.0f} Tagen",
        "n_products": lambda: f"nur 1 Artikel in 12 Monaten" if f == 1 else f"{f:.0f} verschiedene Artikel in 12 Monaten",
        "storno_cnt": lambda: f"{f:.0f} Stornos in 12 Monaten" if f else "keine Stornos",
        "storno_share": lambda: f"Stornoanteil {_pct(f)}",
        "trend_orders": lambda: ("Bestellungen neu aufgebaut (Vorquartal ohne Bestellung)" if f > 10 else
                                 f"Bestellungen rückläufig ({_pct(f)} des Vorquartals)" if f < 1
                                 else f"Bestellungen steigend ({f:.1f}× Vorquartal)".replace(".", ",")),
        "trend_revenue": lambda: ("Umsatz neu aufgebaut (Vorquartal ohne Umsatz)" if f > 10 else
                                  f"Umsatz rückläufig ({_pct(f)} des Vorquartals)" if f < 1
                                  else f"Umsatz steigend ({f:.1f}× Vorquartal)".replace(".", ",")),
        "gap_mean": lambda: f"Ø {f:.0f} Tage zwischen Bestellungen",
        "gap_std": lambda: f"unregelmäßiger Rhythmus (± {f:.0f} Tage)",
        "overdue": lambda: f"{f:.1f}× über dem üblichen Bestellabstand".replace(".", ","),
        "country": lambda: f"Land {names[int(f)]}" if names is not None and 0 <= f < len(names) else "Land unbekannt",
        "g_nb_inactive": lambda: f"{_pct(f)} der Kunden mit gleichen Artikeln inaktiv (Graph)",
        "g_nb_size": lambda: "große Kauf-Nachbarschaft (Graph)" if f > 6 else "kleine Kauf-Nachbarschaft (Graph)",
        "g_prod_pop": lambda: "kauft eher Topseller (Graph)" if f > ctx.get("pop_mid", 5) else "kauft eher Nischenartikel (Graph)",
        "g_pagerank": lambda: "zentral im Kauf-Graphen" if f > ctx.get("pr_mid", 0) else "Randlage im Kauf-Graphen",
        "first_value": lambda: f"erster Warenkorb {_eur(f, fx)}",
        "first_lines": lambda: f"{f:.0f} Positionen im Erstkauf",
        "first_products": lambda: f"{f:.0f} Artikel im Erstkauf",
        "first_avg_price": lambda: f"Ø Stückpreis {f * fx:.2f} €".replace(".", ","),
        "first_storno": lambda: "Storno nach dem Erstkauf" if f else "kein Storno nach dem Erstkauf",
        "days_since_first": lambda: f"Erstkauf vor {f:.0f} Tagen",
        "g_repeat_prop": lambda: f"Artikel mit {_pct(f)} Wiederkaufquote (Graph)",
        "g_country_repeat": lambda: f"Wiederkaufquote im Land {_pct(f)}",
        "group": lambda: f"Gruppe {groups[int(f)]}" if groups and 0 <= f < len(groups) else "Gruppe",
        "group_pop": lambda: f"Gruppe beliebt ({f:.0f} Käufer in 90 Tagen)",
        "group_growth": lambda: f"Gruppe wächst ({f:.1f}×)".replace(".", ",") if f >= 1 else f"Gruppe schrumpft ({f:.1f}×)".replace(".", ","),
        "n_groups": lambda: f"kauft bereits {f:.0f} Gruppen",
        "g_affinity_max": lambda: f"{_pct(f)} Co-Kauf-Affinität zu gekaufter Gruppe (Graph)",
        "g_affinity_mean": lambda: f"Ø Co-Kauf-Affinität {_pct(f)} (Graph)",
        "qty": lambda: f"Menge {f:.0f}",
        "price": lambda: f"Stückpreis {f * fx:.2f} €".replace(".", ","),
        "value": lambda: f"Warenwert {_eur(f, fx)}",
        "price_rel": lambda: f"Preis {f:.2f}× des Üblichen".replace(".", ","),
        "basket_lines": lambda: f"{f:.0f} Positionen im Beleg",
        "new_for_customer": lambda: "Artikel neu für den Kunden" if f else "Artikel kennt der Kunde",
        "new_product": lambda: "neuer Artikel ohne Historie" if f else "Artikel mit Historie",
        "prod_sales": lambda: "Artikel selten verkauft" if f < 4 else "Artikel häufig verkauft",
        "cust_orders": lambda: f"{f:.0f} frühere Belege des Kunden",
        "month": lambda: f"Monat {f:.0f}",
        "g_prod_return": lambda: f"Retourenquote des Artikels {_pct(f)}",
        "g_cust_return": lambda: f"Retourenquote des Kunden {_pct(f)}",
        "g_nb_return": lambda: f"Retourenquote ähnlicher Käufe {_pct(f)} (Graph)",
    }
    if feat.startswith("g_svd_"):
        lab = (ctx.get("svd_labels") or [])
        i = int(feat[6:])
        return (f"Kaufmuster wie Käufer von {lab[i]} (Graph-Einbettung)" if i < len(lab)
                else f"Kaufmuster-Segment {i + 1} (Graph-Einbettung)")
    fn = T.get(feat)
    try:
        return fn() if fn else f"{bt.FEATURE_LABELS.get(feat, feat)}: {f:.2f}".replace(".", ",")
    except Exception:
        return f"{bt.FEATURE_LABELS.get(feat, feat)}"


# ── Aktionslisten je Use Case ─────────────────────────────────────────────────

def _defaults(uc: str) -> dict:
    return bc.defaults(uc) if hasattr(bc, "defaults") else {p["id"]: p["default"] for p in bc.PARAMS[uc]}


def _values(uc: str, task_bt, te_now: pd.DataFrame, fx: float, b) -> tuple[np.ndarray, str]:
    """Wert je Fall in EUR – gleiche Definition wie im Was-wäre-wenn (evaluation._whatif)."""
    if uc == "churn":
        return te_now["monetary_365"].to_numpy() / 4 * fx, "Ø Quartalsumsatz, der bei Abwanderung ausbleibt"
    te = task_bt.test
    y = te[task_bt.label].to_numpy().astype(bool)
    T0 = bt.T_TEST
    if uc == "lead_scoring":
        fut = b.inv[(b.inv["time"] >= T0) & (b.inv["time"] < T0 + pd.Timedelta(days=bt.HORIZON))]
        v = te["cust"].map(fut.groupby("cust")["value"].sum()).fillna(0).to_numpy()
        ratio = np.median(v[y] / np.maximum(te["first_value"].to_numpy()[y], 1)) if y.any() else 1.0
        return (te_now["first_value"].to_numpy() * ratio * fx,
                f"erwarteter Folgeumsatz in 90 Tagen (≈ {ratio:.2f}× erster Warenkorb, aus dem Rückblick)".replace(".", ","))
    if uc == "cross_selling":
        fut = b.lines[(b.lines["time"] >= T0) & (b.lines["time"] < T0 + pd.Timedelta(days=bt.HORIZON))]
        gv = fut.groupby(["cust", "group"])["value"].sum()
        v = np.array([gv.get((c, g), 0.0) for c, g in zip(te["cust"], te["group"])])
        pos = pd.DataFrame({"g": te["group"].to_numpy()[y], "v": v[y]})
        by_g = pos.groupby("g")["v"].median()
        glob = float(pos["v"].median()) if len(pos) else 0.0
        return (te_now["group"].map(by_g).fillna(glob).to_numpy() * fx,
                "erwarteter Umsatz in der neuen Gruppe (Median der Erstkäufe je Gruppe im Rückblick)")
    return te_now["value"].to_numpy() * fx, "Warenwert der Position"


def _ev(uc: str, p: np.ndarray, value: np.ndarray, prm: dict) -> tuple[np.ndarray, np.ndarray]:
    """Erwarteter Wert je Fall und kalibrierte Entscheidungsschwelle p* (Aktion lohnt ab p ≥ p*)."""
    rate, cost = prm["rate"] / 100, float(prm["cost"])
    if uc == "returns_root_cause":
        per = float(prm["cost_return"]) + value * float(prm["value_loss"]) / 100
        gain = per * rate
    else:
        gain = value * rate * float(prm.get("margin", 100)) / 100
    ev = p * gain - cost
    thr = np.where(gain > 0, cost / np.maximum(gain, 1e-9), np.inf)
    return ev, thr


def _history_flag(uc: str, te: pd.DataFrame) -> np.ndarray:
    if uc == "churn":
        return te["freq_365"].to_numpy() <= 1
    if uc == "lead_scoring":
        return te["first_products"].to_numpy() <= 2
    if uc == "cross_selling":
        return te["n_groups"].to_numpy() <= 1
    return te["cust_orders"].to_numpy() <= 1


def build_list(uc: str, vid: str, log=print) -> dict:
    t0 = time.time()
    meta = UC_META[uc]
    b = bt.get_base(vid)
    fx = 1.0 if b.currency == "EUR" else bc.FX_DEFAULT
    from . import evaluation as ev
    res = ev.load(uc, vid) or ev.load(uc)
    eid = (res.get("recommendation") or {}).get("engine") or res.get("best") or "lgbm_graph"
    log(f"[aktivierung] {uc}: Engine {eid} · Kalibrierung auf dem Rückblick-Testfenster …")
    task_bt = bt.get_task(uc, vid)
    eng_a, s_a, notes = _fit_predict(task_bt, eid)
    y_a = task_bt.test[task_bt.label].to_numpy().astype(int)
    cal = calibrate(s_a, y_a, eng_a.proba)
    log(f"[aktivierung] {uc}: ECE {cal['ece_before']} → {cal['ece_after']} · Stichtag {ANCHOR[uc].date()} …")
    task_now = _task_now(uc, vid)
    eng_b, s_b, notes_b = _fit_predict(task_now, eng_a.id, ANCHOR[uc])
    notes += [n for n in notes_b if n not in notes]
    if eng_b.id in ("lgbm_graph", "lgbm"):
        sur, s_sur = eng_b, s_b
    else:
        sur, s_sur, _ = _fit_predict(task_now, "lgbm_graph", ANCHOR[uc])
    contrib = _contrib(sur, task_now)
    te = task_now.test.reset_index(drop=True)
    p = apply_calibration(cal, s_b)
    prm = _defaults(uc)
    value, value_text = _values(uc, task_bt, te, fx, b)
    evv, thr = _ev(uc, p, value, prm)
    # Unsicherheit: Grenzfall · Engines uneinig · wenig Historie
    unc_g = np.isfinite(thr) & (np.abs(p - thr) <= np.maximum(GRENZ_ABS, GRENZ_REL * thr))
    unc_u = (np.abs(_pct_rank(s_b) - _pct_rank(s_sur)) > 0.35) if sur is not eng_b else np.zeros(len(te), bool)
    unc_h = _history_flag(uc, te)
    # Kreuzselling: je Kunde nur die beste Gruppe (eine Botschaft je Kunde)
    keep = np.arange(len(te))
    if uc == "cross_selling":
        order = np.lexsort((-p, -evv, te["cust"].to_numpy()))
        first = ~pd.Series(te["cust"].to_numpy()[order]).duplicated().to_numpy()
        keep = order[first]
    population = len(keep)
    share = prm.get("list_share", 10) / 100
    capacity = max(1, int(round(population * share)))
    if MAX_ROWS.get(uc):
        capacity = min(capacity, MAX_ROWS[uc])     # Prüfkapazität des Innendienstes je Zeitraum
    rank_order = keep[np.argsort(-evv[keep], kind="stable")]
    in_list = np.zeros(len(te), bool)
    pos_ev = rank_order[evv[rank_order] > 0][:capacity]
    in_list[pos_ev] = True
    cut = MAX_ROWS.get(uc)
    rows_idx = rank_order[:cut] if cut else rank_order
    feats = list(sur.cols)
    top3 = np.argsort(-contrib[rows_idx], axis=1)[:, :3]
    cols = {"cust": [], "id": [], "score": [], "p": [], "value": [], "ev": [], "thr": [], "in_list": [],
            "unc": [], "reasons": [], "extra": []}
    inv_names = None
    if uc == "returns_root_cause":
        src = bt._VARIANTS.get(vid)
        if src is None:
            from .data_loader import get_registry
            src = get_registry().raw_online_retail()
        inv_names = np.asarray(src["invoice"].cat.categories)
    for k, i in enumerate(rows_idx):
        r = te.iloc[i]
        cust = int(r["cust"])
        flags = [c for c, m in (("grenzfall", unc_g[i]), ("uneinig", unc_u[i]), ("wenig_historie", unc_h[i])) if m]
        rs = []
        for j in top3[k]:
            if contrib[i, j] <= 0:
                continue
            fval = r[feats[j]]
            rs.append([int(j), None if pd.isna(fval) else round(float(fval), 4), round(float(contrib[i, j]), 4)])
        if uc == "cross_selling":
            gid = int(r["group"])
            ex, rid = {"group": gid, "group_name": b.group_names[gid]}, f"{cust}:{gid}"
        elif uc == "returns_root_cause":
            pr = int(r["prod"])
            ex = {"product": str(b.prod_names[pr])[:48], "qty": int(r["qty"]),
                  "invoice": str(inv_names[int(r["inv"])]) if inv_names is not None else "",
                  "date": pd.Timestamp(r["time"]).date().isoformat()}
            rid = f"{ex['invoice']}:{pr}:{int(r['line'])}"
        else:
            ex = {"recency": int(r["recency"]), "monetary_365": round(float(r["monetary_365"]) * fx, 2)}
            if uc == "lead_scoring":
                ex["first_value"] = round(float(r["first_value"]) * fx, 2)
            rid = str(cust)
        ex["country"] = int(r["country"]) if "country" in r and not pd.isna(r["country"]) else -1
        cols["cust"].append(cust)
        cols["id"].append(rid)
        cols["score"].append(round(float(s_b[i]), 5))
        cols["p"].append(round(float(p[i]), 4))
        cols["value"].append(round(float(value[i]), 2))
        cols["ev"].append(round(float(evv[i]), 2))
        cols["thr"].append(round(float(min(thr[i], 9.99)), 4))
        cols["in_list"].append(int(in_list[i]))
        cols["unc"].append(flags)
        cols["reasons"].append(rs)
        cols["extra"].append(ex)
    lst = np.flatnonzero(in_list)
    unc_any = unc_g | unc_u | unc_h
    out = {
        "uc": uc, **meta, "engine": {"id": eng_b.id, "name": eng_b.name, "klass": eng_b.klass,
                                     "device": getattr(eng_b, "device", "CPU"),
                                     "recommended": eid, "reasons_from": sur.name},
        "notes": notes, "calibration": cal, "anchor": ANCHOR[uc].date().isoformat(),
        "train_until": task_now.val_cutoff.date().isoformat() if task_now.val_cutoff is not None else None,
        "params": prm, "value_text": value_text, "population": int(population), "capacity": int(capacity),
        "features": feats, "feature_labels": [bt.FEATURE_LABELS.get(f, f) for f in feats],
        "threshold_text": ("Aktion, wenn erwarteter Wert > 0: kalibrierte Wahrscheinlichkeit × Wert × "
                           f"Quote {prm['rate']} % − Kosten {prm['cost']} € je Kontakt – also ab p ≥ Kosten ÷ "
                           "(Wert × Quote). Dazu die Kapazität: "
                           + (f"{capacity} Prüfungen je Zeitraum (Innendienst)." if MAX_ROWS.get(uc) else
                              f"oberste {prm.get('list_share', 10)} % ({capacity} Fälle).")),
        "summary": {"in_list": int(len(lst)), "ev_list": round(float(evv[lst].sum()), 2),
                    "p_list": round(float(p[lst].mean()), 4) if len(lst) else None,
                    "uncertain_list": int(unc_any[lst].sum()), "sure_list": int(((p >= SURE) & ~unc_any)[lst].sum()),
                    "positive_ev": int((evv[keep] > 0).sum())},
        "rows": cols, "seconds": round(time.time() - t0, 1),
        "_drift": {"ref": task_now.train, "cur": task_now.test,
                   "features": [f for f in feats if f not in task_now.cat_features and f != "month"],
                   "p_ref": apply_calibration(cal, s_a), "p_cur": p},
    }
    log(f"[aktivierung] {uc}: {len(lst)} in der Liste · {out['seconds']} s")
    return out


def build_reco(vid: str, log=print) -> dict:
    """Top-3-Artikel je aktivem Kunden (LightGBM-Ranker auf Graph-Kandidaten) zum Stichtag."""
    from .engines.cpu import LGBMRankEngine, RANK_FEATS, _rank_pairs
    t0 = time.time()
    b = bt.get_base(vid)
    fx = 1.0 if b.currency == "EUR" else bc.FX_DEFAULT
    task = bt.get_task("recommendation", vid)
    eng = LGBMRankEngine().timed_fit(task)
    snap = bt.reco_snapshot(b, T_NOW)
    te = _rank_pairs(snap, False, b)
    te["s"] = eng.model.predict(te[RANK_FEATS], num_iteration=eng.model.best_iteration)
    top = te.sort_values(["u", "s"], ascending=[True, False]).groupby("u").head(3)
    hist = b.lines[b.lines["time"] < T_NOW]
    line_val = hist.groupby("prod")["value"].median()
    prm = _defaults("recommendation")
    rate, cost = prm["rate"] / 100, float(prm["cost"])
    rows = {"cust": [], "items": [], "names": [], "p": [], "ev": []}
    for u, g in top.groupby("u"):
        prods = g["p"].to_numpy()
        ps = g["s"].to_numpy()
        vals = line_val.reindex(prods).fillna(0).to_numpy() * fx
        rows["cust"].append(int(snap["users"][u]))
        rows["items"].append([int(x) for x in prods])
        rows["names"].append([str(b.prod_names[x])[:36] for x in prods])
        rows["p"].append([round(float(x), 4) for x in ps])
        rows["ev"].append(round(float((ps * vals).sum() * rate - cost), 2))
    log(f"[aktivierung] recommendation: {len(rows['cust'])} Kunden · {time.time() - t0:.0f} s")
    return {"uc": "recommendation", "name": "Produkt-Empfehlung", "engine": {"id": eng.id, "name": eng.name},
            "anchor": T_NOW.date().isoformat(), "params": prm, "rows": rows,
            "value_text": "Σ Wahrscheinlichkeit × typischer Positionswert der Top-3-Artikel × Quote − Kosten"}


# ── Kontrollgruppe, Fallzahl, Next Best Action ────────────────────────────────

def assign_control(custs, strata, share: float = CONTROL_SHARE, seed: int = SEED) -> set:
    """Zufällige, stratifizierte Holdout-Gruppe mit festem Seed (je Schicht exakt gerundet share)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"c": np.asarray(custs), "s": np.asarray(strata)}).drop_duplicates("c").sort_values(["s", "c"])
    ctrl = set()
    for _, g in df.groupby("s", sort=True):
        ids = g["c"].to_numpy().copy()
        rng.shuffle(ids)
        ctrl |= set(int(x) for x in ids[: int(round(len(ids) * share))])
    return ctrl


def pilot_size(p0: float, effect_rel: float, n_t: int, n_c: int, alpha: float = 0.05, power: float = 0.8) -> dict:
    """Fallzahl für den Pilot (Zwei-Stichproben-Test auf Anteile, zweiseitig, Normal-Approximation)."""
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    p0 = float(min(max(p0, 1e-4), 1 - 1e-4))
    p1 = float(min(max(p0 * (1 + effect_rel), 1e-4), 1 - 1e-4))
    d = abs(p1 - p0)
    mde = z * math.sqrt(p0 * (1 - p0) * (1 / max(n_t, 1) + 1 / max(n_c, 1)))
    n = n_t + n_c
    c = n_c / n if n else CONTROL_SHARE
    req = z ** 2 * (p0 * (1 - p0) / c + p1 * (1 - p1) / (1 - c)) / d ** 2 if d > 0 else float("inf")
    return {"p0": round(p0, 4), "p1": round(p1, 4), "effect_abs": round(p1 - p0, 4), "effect_rel": effect_rel,
            "n_t": int(n_t), "n_c": int(n_c), "mde_abs": round(mde, 4), "mde_rel": round(mde / p0, 4),
            "detectable": bool(d >= mde), "n_required": int(math.ceil(req)) if math.isfinite(req) else None,
            "periods_needed": int(math.ceil(req / n)) if n and math.isfinite(req) else None,
            "alpha": alpha, "power": power}


NBA_ACTIONS = {
    "control": ("Kontrollgruppe – bewusst keine Aktion", "–"),
    "retention": ("Retention-Anruf (Außendienst)", "Telefon"),
    "onboarding": ("Zweitkauf-Impuls mit Empfehlungen", "Telefon/E-Mail"),
    "cross": ("Cross-Selling-Angebot", "E-Mail/Webshop"),
    "reco": ("Persönliche Empfehlung (Top-3)", "Newsletter"),
    "none": ("Keine Aktion – kein positiver erwarteter Wert", "–"),
}
NBA_RULES = [
    "1 · Kontrollgruppe bleibt unberührt (Messung im Pilot).",
    "2 · Hohes Abwanderungsrisiko (kalibriert ≥ 50 %, in der Anruf-Kapazität, Wert > 0) → Retention zuerst – "
    "kein Verkaufsangebot an Kunden, die gerade gehen.",
    "3 · Neukunden mit positivem Wert → Zweitkauf-Impuls (mit den Top-Empfehlungen).",
    "4 · Sonst die Aktion mit dem höchsten erwarteten Wert (Cross-Selling, Empfehlung, Retention).",
    "5 · Höchstens eine Aktion je Kunde und Quartal; verdrängte Aktionen werden als Konflikt ausgewiesen.",
]


def build_nba(lists: dict, reco: dict, control: set) -> dict:
    def by_cust(uc):
        L = lists.get(uc)
        if not L:
            return {}
        r = L["rows"]
        return {c: i for i, c in enumerate(r["cust"])}

    ch, ld, cr = by_cust("churn"), by_cust("lead_scoring"), by_cust("cross_selling")
    rc = {c: i for i, c in enumerate(reco["rows"]["cust"])} if reco else {}
    custs = sorted(set(ch) | set(ld) | set(cr) | set(rc))
    R = {k: lists[k]["rows"] for k in lists}
    out = {"cust": [], "action": [], "text": [], "channel": [], "ev": [], "why": [], "conflicts": []}
    counts: dict[str, int] = {}
    ev_sum: dict[str, float] = {}
    for c in custs:
        cand = {}
        if c in ch:
            i = ch[c]
            if R["churn"]["in_list"][i] and R["churn"]["ev"][i] > 0:
                cand["retention"] = (R["churn"]["ev"][i], R["churn"]["p"][i])
        if c in ld:
            i = ld[c]
            if R["lead_scoring"]["ev"][i] > 0:
                cand["onboarding"] = (R["lead_scoring"]["ev"][i], R["lead_scoring"]["p"][i])
        if c in cr:
            i = cr[c]
            if R["cross_selling"]["ev"][i] > 0:
                cand["cross"] = (R["cross_selling"]["ev"][i], R["cross_selling"]["p"][i])
        if c in rc:
            i = rc[c]
            if reco["rows"]["ev"][i] > 0:
                cand["reco"] = (reco["rows"]["ev"][i], max(reco["rows"]["p"][i]))
        conflicts, why = [], ""
        if c in control:
            act, why = "control", "zufällig der Kontrollgruppe zugeordnet (stratifiziert, Seed 7)"
            conflicts = [f"{NBA_ACTIONS[k][0]} ausgesetzt" for k in cand]
        elif "retention" in cand and cand["retention"][1] >= 0.5:
            act = "retention"
            why = f"Abwanderungsrisiko {cand['retention'][1] * 100:.0f} % – Bindung vor Verkauf"
            conflicts = [f"{NBA_ACTIONS[k][0]} zurückgestellt (Regel 2)" for k in cand if k != act]
        elif "onboarding" in cand:
            act = "onboarding"
            why = f"Neukunde, Wiederkauf-Chance {cand['onboarding'][1] * 100:.0f} %"
            conflicts = [f"{NBA_ACTIONS[k][0]} zurückgestellt (Regel 3)" for k in cand if k not in (act, "reco")]
        elif cand:
            act = max(cand, key=lambda k: cand[k][0])
            why = f"höchster erwarteter Wert ({_eur(cand[act][0])})"
            conflicts = [f"{NBA_ACTIONS[k][0]} ({_eur(cand[k][0])}) zurückgestellt – 1 Aktion je Quartal"
                         for k in sorted(cand, key=lambda k: -cand[k][0]) if k != act]
        else:
            act, why = "none", "kein Kandidat mit positivem erwartetem Wert"
        text, channel = NBA_ACTIONS[act]
        if act == "cross":
            text += f": {R['cross_selling']['extra'][cr[c]]['group_name']}"
        if act in ("reco", "onboarding") and c in rc:
            text += " – " + ", ".join(n.title() for n in reco["rows"]["names"][rc[c]][:2])
        e = cand[act][0] if act in cand else 0.0
        out["cust"].append(int(c))
        out["action"].append(act)
        out["text"].append(text)
        out["channel"].append(channel)
        out["ev"].append(round(float(e), 2))
        out["why"].append(why)
        out["conflicts"].append(conflicts)
        counts[act] = counts.get(act, 0) + 1
        ev_sum[act] = ev_sum.get(act, 0.0) + float(e)
    return {"rows": out, "rules": NBA_RULES, "counts": counts,
            "ev": {k: round(v, 2) for k, v in ev_sum.items()},
            "conflicts": int(sum(1 for x in out["conflicts"] if x))}


def _control_from(lists: dict) -> tuple[set, dict]:
    """Strata: Abwanderungsrisiko-Dezil × Umsatz-Terzil (Churn-Liste); übrige Kunden je Liste."""
    ch = lists["churn"]["rows"]
    p = np.asarray(ch["p"])
    v = np.asarray([e["monetary_365"] for e in ch["extra"]])
    dec = pd.Series(p).rank(pct=True, method="first").mul(10).clip(upper=9.999).astype(int).to_numpy()
    ter = pd.Series(v).rank(pct=True, method="first").mul(3).clip(upper=2.999).astype(int).to_numpy()
    custs, strata = list(ch["cust"]), [f"R{d}·U{t}" for d, t in zip(dec, ter)]
    seen = set(custs)
    for uc in ("lead_scoring", "cross_selling", "returns_root_cause"):
        for c in lists[uc]["rows"]["cust"]:
            if c not in seen:
                seen.add(c)
                custs.append(c)
                strata.append(f"ohne Churn-Score·{uc}")
    ctrl = assign_control(custs, strata)
    df = pd.DataFrame({"c": custs, "s": strata})
    df["ctrl"] = df["c"].isin(ctrl)
    st = df.groupby("s")["ctrl"].agg(["size", "mean"]).reset_index()
    return ctrl, {"share": CONTROL_SHARE, "seed": SEED, "customers": len(custs), "control": len(ctrl),
                  "strata": [{"stratum": r["s"], "n": int(r["size"]), "control_share": round(float(r["mean"]), 3)}
                             for _, r in st.iterrows()],
                  "method": "Zufall je Schicht (Abwanderungsrisiko-Dezil × Umsatz-Terzil), fester Seed 7"}


# ── Monitoring ────────────────────────────────────────────────────────────────

PSI_WARN, PSI_ALARM = 0.10, 0.25
SEASON_NOTE = ("Einordnung: Der Stichtag liegt im Weihnachtsgeschäft – ein Teil der Drift ist Saison, kein Datenfehler. "
               "„Kundentreue“ driftet, weil die Historie erst im Dezember 2009 beginnt und mit jedem Monat wächst. "
               "Im Betrieb wird gegen denselben Monat des Vorjahres verglichen und monatlich neu trainiert.")
DQ_RULES = {"cust_fill": ("Füllgrad Kundennummer", "min", 0.70, 0.60),
            "desc_fill": ("Beschreibung gefüllt", "min", 0.985, 0.97),
            "storno_linked": ("Stornos mit Originalbeleg (PB-02)", "min", 0.75, 0.60),
            "price0": ("Positionen mit Preis 0", "max", 0.015, 0.03),
            "non_article": ("Nicht-Artikel-Codes", "max", 0.02, 0.04),
            "dup": ("exakte Dubletten", "max", 0.04, 0.08)}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float | None:
    """Population Stability Index mit Quantil-Grenzen der Referenz."""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 20 or len(cur) < 20:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return round(float(((c - r) * np.log(c / r)).sum()), 4)


def psi_status(v: float | None) -> str:
    if v is None:
        return "na"
    return "ok" if v < PSI_WARN else "warn" if v < PSI_ALARM else "bad"


def _row_hash(raw: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Kombinierter 64-Bit-Hash je Zeile (speichersparend statt DataFrame.duplicated über viele Spalten)."""
    h = np.zeros(len(raw), dtype=np.uint64)
    for i, c in enumerate(cols):
        x = raw[c]
        if isinstance(x.dtype, pd.CategoricalDtype):
            v = x.cat.codes.to_numpy().astype(np.int64)
        elif pd.api.types.is_datetime64_any_dtype(x.dtype):
            v = x.to_numpy().astype("datetime64[s]").astype(np.int64)
        elif pd.api.types.is_float_dtype(x.dtype):
            v = np.round(x.to_numpy(dtype="float64", na_value=np.nan) * 1000)
            v = np.nan_to_num(v, nan=-1).astype(np.int64)
        else:
            v = x.to_numpy(dtype="float64", na_value=-1).astype(np.int64)
        with np.errstate(over="ignore"):
            h = (h ^ (v.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15) + (h << np.uint64(6)) + (h >> np.uint64(2)))) \
                * np.uint64(1099511628211 + 2 * i)
    return h


def _storno_linked(raw: pd.DataFrame, storno: np.ndarray, has_c: np.ndarray) -> np.ndarray:
    """Wie PB-02 (Kunde, Artikel, ≤ 90 Tage, Menge ≥ 50 %), aber nur auf den nötigen Spalten."""
    cust = raw["customer_id"].to_numpy(dtype="float64", na_value=np.nan)
    prod = raw["stock_code"].cat.codes.to_numpy()
    t = raw["invoice_date"].to_numpy()
    q = raw["quantity"].to_numpy()
    si, bi = np.flatnonzero(storno & has_c), np.flatnonzero(~storno & has_c)
    S = pd.DataFrame({"i": si, "cust": cust[si], "prod": prod[si], "time": t[si], "q": np.abs(q[si])}).sort_values("time")
    B = pd.DataFrame({"cust": cust[bi], "prod": prod[bi], "time": t[bi], "bq": q[bi]}).sort_values("time")
    m = pd.merge_asof(S, B, on="time", by=["cust", "prod"], direction="backward", tolerance=pd.Timedelta(days=90))
    ok = m["bq"].notna() & (m["bq"] >= m["q"] * 0.5)
    out = np.zeros(len(raw), bool)
    out[m.loc[ok, "i"].to_numpy()] = True
    return out


def data_quality_monthly(raw: pd.DataFrame) -> list[dict]:
    """Datenqualität je Monat auf den Rohdaten (so, wie sie aus dem Quellsystem kommen)."""
    from . import playbooks as pb
    dt = raw["invoice_date"].dt
    m = (dt.year * 100 + dt.month).to_numpy(dtype=np.int32)          # JJJJMM statt 1 Mio. Strings
    storno = np.asarray(pb._is_storno(raw))
    has_c = raw["customer_id"].notna().to_numpy()
    cats = raw["stock_code"].cat.categories
    non_art = np.array([pb._non_article(str(c)) for c in cats])[raw["stock_code"].cat.codes.to_numpy()]
    dup = pd.Series(_row_hash(raw, [c for c in ("invoice", "stock_code", "description", "quantity", "invoice_date",
                                                "price_gbp", "customer_id", "country") if c in raw.columns])
                    ).duplicated().to_numpy()                                        # wie PB-06
    linked = _storno_linked(raw, storno, has_c)
    df = pd.DataFrame({"m": m, "c": has_c, "d": raw["description"].notna().to_numpy(),
                       "p0": raw["price_gbp"].to_numpy() == 0, "na": non_art, "dup": dup,
                       "st": storno & has_c, "ln": linked})
    res = []
    for month, x in df.groupby("m"):
        n_st = int(x["st"].sum())
        res.append({"month": f"{month // 100}-{month % 100:02d}", "rows": int(len(x)), "cust_fill": round(float(x["c"].mean()), 4),
                    "desc_fill": round(float(x["d"].mean()), 4), "price0": round(float(x["p0"].mean()), 4),
                    "non_article": round(float(x["na"].mean()), 4), "dup": round(float(x["dup"].mean()), 4),
                    "storno_linked": round(float(x["ln"].sum() / n_st), 4) if n_st else None, "stornos": n_st})
    return res


def dq_alarms(months: list[dict]) -> list[dict]:
    out = []
    for n, row in enumerate(months):
        for k, (name, kind, warn, bad) in DQ_RULES.items():
            v = row.get(k)
            if v is None or (k == "storno_linked" and n == 0):   # erster Monat: keine Käufe davor zum Zuordnen
                continue
            lvl = None
            if kind == "min":
                lvl = "bad" if v < bad else "warn" if v < warn else None
            else:
                lvl = "bad" if v > bad else "warn" if v > warn else None
            if lvl:
                out.append({"month": row["month"], "metric": k, "name": name, "value": v, "status": lvl,
                            "threshold": warn})
    return out


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], uc: str) -> list[dict]:
    rows = []
    for f in features:
        if f not in ref or f not in cur:
            continue
        v = psi(ref[f].to_numpy(), cur[f].to_numpy())
        rows.append({"feature": f, "label": bt.FEATURE_LABELS.get(f, f), "psi": v, "status": psi_status(v),
                     "ref_median": _r(np.nanmedian(ref[f].to_numpy(dtype=float))),
                     "cur_median": _r(np.nanmedian(cur[f].to_numpy(dtype=float))), "uc": uc})
    return sorted(rows, key=lambda r: -(r["psi"] or 0))


def drift_timeline(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> dict:
    """PSI je Stichtag gegen die ersten sechs Trainings-Stichtage – zeigt, wie sich Drift aufbaut."""
    cut = sorted(ref["cutoff"].unique())
    base = ref[ref["cutoff"].isin(cut[:6])]
    series = {f: [] for f in features}
    labels = []
    frames = [(pd.Timestamp(T), ref[ref["cutoff"] == T]) for T in cut[6:]] + [(None, cur)]
    for T, fr in frames:
        labels.append(T.strftime("%m/%Y") if T is not None else "Stichtag")
        for f in features:
            series[f].append(psi(base[f].to_numpy(), fr[f].to_numpy()))
    return {"labels": labels, "series": [{"feature": f, "label": bt.FEATURE_LABELS.get(f, f), "psi": series[f]}
                                         for f in features]}


def _hist(p: np.ndarray, bins: int = 20) -> list[float]:
    h, _ = np.histogram(np.clip(p, 0, 1), bins=bins, range=(0, 1))
    return [round(float(x), 4) for x in h / max(h.sum(), 1)]


def _r(x, d=3):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, d)


def build_monitoring(lists: dict, log=print, raw: pd.DataFrame | None = None) -> dict:
    from .data_loader import get_registry
    t0 = time.time()
    raw = get_registry().raw_online_retail() if raw is None else raw
    months = data_quality_monthly(raw)
    drift, scores = {}, {}
    timeline = None
    for uc, L in lists.items():
        d = L.get("_drift")
        if not d:
            continue
        feats = [f for f in d["features"] if f in d["ref"].columns]
        drift[uc] = drift_table(d["ref"], d["cur"], feats, uc)
        sp = psi(d["p_ref"], d["p_cur"])
        scores[uc] = {"psi": sp, "status": psi_status(sp), "ref": _hist(d["p_ref"]), "cur": _hist(d["p_cur"]),
                      "ref_mean": _r(np.mean(d["p_ref"]), 4), "cur_mean": _r(np.mean(d["p_cur"]), 4),
                      "ref_label": "Rückblick-Testfenster 01.09.2011", "cur_label": f"Stichtag {L['anchor']}"}
        if uc == "churn":
            top = [r["feature"] for r in drift[uc][:5]]
            timeline = drift_timeline(d["ref"], d["cur"], top)
    alarms = dq_alarms(months)
    for uc, rows in drift.items():
        for r in rows:
            if r["status"] == "bad":
                alarms.append({"month": "Stichtag", "metric": r["feature"], "name": f"Drift {r['label']} ({UC_META[uc]['name']})",
                               "value": r["psi"], "status": "bad", "threshold": PSI_ALARM})
    for uc, s in scores.items():
        if s["status"] in ("warn", "bad"):
            alarms.append({"month": "Stichtag", "metric": "score", "name": f"Score-Verteilung {UC_META[uc]['name']}",
                           "value": s["psi"], "status": s["status"], "threshold": PSI_WARN})
    log(f"[monitoring] {len(months)} Monate · {sum(len(v) for v in drift.values())} Merkmale · "
        f"{len(alarms)} Alarme · {time.time() - t0:.0f} s")
    alarms.sort(key=lambda a: (a["status"] != "bad", a["month"] != "Stichtag"))
    return {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"), "months": months,
            "dq_rules": {k: {"name": v[0], "kind": v[1], "warn": v[2], "bad": v[3]} for k, v in DQ_RULES.items()},
            "drift": drift, "scores": scores, "timeline": timeline, "alarms": alarms,
            "psi_thresholds": {"warn": PSI_WARN, "alarm": PSI_ALARM},
            "psi_text": "PSI < 0,10 stabil · 0,10–0,25 beobachten · > 0,25 Alarm: neu trainieren bzw. Datenquelle prüfen",
            "season_note": SEASON_NOTE}


def recheck_monitoring(ctx=None) -> dict:
    """„Erneut prüfen“: Datenqualität der Rohdaten neu, Drift der Churn-Merkmale neu (ohne Modelltraining)."""
    prev = load_monitoring() or {}
    if ctx:
        ctx.progress(10, "Datenqualität je Monat …")
    from .data_loader import get_registry
    months = data_quality_monthly(get_registry().raw_online_retail())
    res = {**prev, "months": months}
    if ctx:
        ctx.progress(55, "Drift der Churn-Merkmale …")
    vid, _ = _variant()
    from .engines.base import on_render
    try:
        if on_render():          # Speicher: Pipeline + Merkmale brauchen ~350 MB extra – auf Render nur Datenqualität
            raise RuntimeError("auf Render nur Datenqualität neu, Drift aus der Vorberechnung")
        _ensure_variant(vid)
        b = bt.get_base(vid)
        audit = bt.Audit()
        ref_cuts = list(pd.date_range(bt.TRAIN_START, T_NOW - pd.Timedelta(days=bt.HORIZON), freq="MS"))
        frames = []
        for T in ref_cuts + [T_NOW]:
            f = bt.customer_features(b, T, audit, "monitoring")
            f = f[f["recency"] <= bt.HISTORY]
            g = bt.graph_customer_features(b, T, f, audit, "monitoring")
            frames.append(pd.concat([f, g], axis=1).assign(cutoff=T))
        data = pd.concat(frames)
        ref, cur = data[data["cutoff"] < T_NOW], data[data["cutoff"] == T_NOW]
        feats = [r["feature"] for r in prev.get("drift", {}).get("churn", [])] or \
            [c for c in data.columns if c not in ("cutoff", "country", "month")]
        res.setdefault("drift", {})["churn"] = drift_table(ref, cur, feats, "churn")
        res["timeline"] = drift_timeline(ref, cur, [r["feature"] for r in res["drift"]["churn"][:5]])
    except Exception as exc:
        log.warning("Drift-Neuprüfung: %s", exc)
        res["recheck_note"] = f"Drift aus der Vorberechnung ({exc})"
    else:
        res.pop("recheck_note", None)
    alarms = dq_alarms(months)
    for uc, rows in res.get("drift", {}).items():
        alarms += [{"month": "Stichtag", "metric": r["feature"], "name": f"Drift {r['label']} ({UC_META[uc]['name']})",
                    "value": r["psi"], "status": "bad", "threshold": PSI_ALARM} for r in rows if r["status"] == "bad"]
    alarms += [{"month": "Stichtag", "metric": "score", "name": f"Score-Verteilung {UC_META[uc]['name']}",
                "value": s["psi"], "status": s["status"], "threshold": PSI_WARN}
               for uc, s in res.get("scores", {}).items() if s["status"] in ("warn", "bad")]
    res["alarms"] = alarms
    res["rechecked_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    save_monitoring(res)
    if ctx:
        ctx.progress(100, "fertig")
    return {"alarms": len(alarms), "rechecked_at": res["rechecked_at"]}


# ── Gesamtlauf und Artefakte ──────────────────────────────────────────────────

def build(log=print) -> dict:
    t0 = time.time()
    vid, label = _variant()
    log(f"[aktivierung] Datenstand {vid} ({label}) · Stichtag {T_NOW.date()}")
    _ensure_variant(vid)
    lists = {uc: build_list(uc, vid, log) for uc in LIST_UCS}
    reco = build_reco(vid, log)
    control, ctrl_info = _control_from(lists)
    pilot = {}
    for uc, L in lists.items():
        r = L["rows"]
        r["control"] = [int(c in control) for c in r["cust"]]
        il = np.asarray(r["in_list"], bool)
        ct = np.asarray(r["control"], bool)
        p = np.asarray(r["p"])
        eff = UC_META[uc]["effect_sign"] * L["params"]["rate"] / 100
        pilot[uc] = pilot_size(float(p[il].mean()) if il.any() else 0.0, eff, int((il & ~ct).sum()),
                               int((il & ct).sum()))
        pilot[uc]["effect_text"] = UC_META[uc]["effect_text"]
    nba = build_nba(lists, reco, control)
    monitoring = build_monitoring(lists, log)
    b = bt.get_base(vid)
    art = {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"), "variant": vid, "variant_label": label,
           "t_now": T_NOW.date().isoformat(), "valid_until": (T_NOW + pd.Timedelta(days=bt.HORIZON)).date().isoformat(),
           "currency": "EUR", "fx": bc.FX_DEFAULT if b.currency == "GBP" else 1.0,
           "lists": {uc: {k: v for k, v in L.items() if not k.startswith("_")} for uc, L in lists.items()},
           "reco": reco, "nba": nba, "control": ctrl_info, "pilot": pilot,
           "country_names": [str(x) for x in b.country_names], "group_names": list(b.group_names),
           "svd_labels": svd_labels(b),
           "seconds": round(time.time() - t0, 1)}
    save(art)
    save_monitoring(monitoring)
    log(f"[aktivierung] fertig in {art['seconds']} s")
    return art


_ENC_NBA = ("text", "channel", "why")


def _encode(art: dict) -> dict:
    """Wiederholte Texte (NBA, Gruppennamen) als Wörterbuch speichern – hält das Artefakt klein."""
    out = json.loads(json.dumps(art, default=str))
    words: list[str] = []
    idx: dict[str, int] = {}

    def code(t: str) -> int:
        if t not in idx:
            idx[t] = len(words)
            words.append(t)
        return idx[t]

    r = out["nba"]["rows"]
    for k in _ENC_NBA:
        r[k] = [code(t) for t in r[k]]
    r["conflicts"] = [[code(t) for t in c] for c in r["conflicts"]]
    for L in out["lists"].values():
        for e in L["rows"]["extra"]:
            if "group_name" in e:
                e.pop("group_name")
    out["_words"] = words
    return out


def _decode(art: dict) -> dict:
    words = art.pop("_words", None)
    if words is None:
        return art
    r = art["nba"]["rows"]
    for k in _ENC_NBA:
        r[k] = [words[i] for i in r[k]]
    r["conflicts"] = [[words[i] for i in c] for c in r["conflicts"]]
    g = art.get("group_names") or []
    for L in art["lists"].values():
        for e in L["rows"]["extra"]:
            if "group" in e and "group_name" not in e:
                e["group_name"] = g[e["group"]] if 0 <= e["group"] < len(g) else ""
    return art


def save(art: dict) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "activation.json").write_text(json.dumps(_encode(art), ensure_ascii=False, separators=(",", ":"),
                                                          default=str), encoding="utf-8")
    _CACHE.pop("activation", None)


def load() -> dict | None:
    if "activation" not in _CACHE:
        p = ARTIFACTS / "activation.json"
        _CACHE["activation"] = _decode(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None
    return _CACHE["activation"]


def save_monitoring(mon: dict) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "monitoring.json").write_text(json.dumps(mon, ensure_ascii=False, separators=(",", ":"),
                                                          default=str), encoding="utf-8")
    _CACHE.pop("monitoring", None)


def load_monitoring() -> dict | None:
    if "monitoring" not in _CACHE:
        p = ARTIFACTS / "monitoring.json"
        _CACHE["monitoring"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return _CACHE["monitoring"]


# ── Sichten für die UI ────────────────────────────────────────────────────────

def _ctx(art: dict) -> dict:
    return {"fx": art.get("fx", 1.0), "country_names": art.get("country_names"), "group_names": art.get("group_names"),
            "svd_labels": art.get("svd_labels")}


def svd_labels(b) -> list[str]:
    """Sprechende Namen der Co-Kauf-Einbettungen: die Artikelgruppen mit den stärksten Ladungen je Dimension."""
    if b.svd_items is None:
        return []
    out = []
    glob = pd.Series(b.prod_group).value_counts(normalize=True)
    for i in range(b.svd_items.shape[1]):
        top = np.argsort(-b.svd_items[:, i])[:60]
        cnt = pd.Series(b.prod_group[top]).value_counts()
        lift = (cnt / len(top) / glob.reindex(cnt.index)).where(cnt >= 3).dropna().sort_values(ascending=False)
        names = [b.group_names[g] for g in lift.index[:2]] or [b.group_names[cnt.index[0]]]
        out.append(" · ".join(names))
    return out


def summary() -> dict | None:
    art = load()
    if not art:
        return None
    lists = {}
    for uc, L in art["lists"].items():
        lists[uc] = {k: L[k] for k in ("uc", "name", "entity", "event", "action", "channel", "engine", "notes",
                                       "calibration", "anchor", "train_until", "params", "value_text", "population",
                                       "capacity", "threshold_text", "summary", "effect_text")}
        lists[uc]["pilot"] = art["pilot"][uc]
    return {k: art[k] for k in ("generated_at", "variant", "variant_label", "t_now", "valid_until", "currency",
                                "fx", "control")} | {
        "lists": lists, "nba": {k: art["nba"][k] for k in ("rules", "counts", "ev", "conflicts")},
        "reco": {"name": art["reco"]["name"], "engine": art["reco"]["engine"], "customers": len(art["reco"]["rows"]["cust"]),
                 "value_text": art["reco"]["value_text"]},
        "unc_text": UNC_TEXT, "sure": SURE, "control_share": CONTROL_SHARE}


def _row(art: dict, uc: str, i: int, decisions: dict | None = None) -> dict:
    L = art["lists"][uc]
    r = L["rows"]
    ctx = _ctx(art)
    reasons = [t for t in (reason_text(L["features"][fi], val, ctx) for fi, val, _ in r["reasons"][i])
               if not t.endswith("unbekannt")]
    rid = r["id"][i]
    dec = (decisions or {}).get(f"{uc}|{rid}") or {}
    return {"rank": i + 1, "id": rid, "cust": r["cust"][i], "score": r["score"][i], "p": r["p"][i],
            "value": r["value"][i], "ev": r["ev"][i], "thr": r["thr"][i], "in_list": bool(r["in_list"][i]),
            "control": bool(r["control"][i]), "unc": r["unc"][i], "sure": r["p"][i] >= SURE and not r["unc"][i],
            "reasons": reasons, "extra": r["extra"][i], "status": dec.get("status", "offen"),
            "status_reason": dec.get("reason", "")}


def rows(uc: str, filt: str = "list", offset: int = 0, limit: int = 25, decisions: dict | None = None) -> dict:
    art = load()
    if not art:
        return {"rows": [], "total": 0}
    if uc == "next_best_action":
        return nba_rows(filt, offset, limit)
    if uc == "recommendation":
        r = art["reco"]["rows"]
        idx = sorted(range(len(r["cust"])), key=lambda i: -r["ev"][i])
        total = len(idx)
        out = [{"rank": k + 1, "cust": r["cust"][i], "items": r["names"][i], "p": r["p"][i], "ev": r["ev"][i]}
               for k, i in enumerate(idx[offset: offset + limit], start=offset)]
        return {"rows": out, "total": total}
    L = art["lists"][uc]
    r = L["rows"]
    n = len(r["id"])
    dec = decisions or {}

    def keep(i):
        if filt == "list":
            return r["in_list"][i] and not r["control"][i]
        if filt == "uncertain":
            return r["in_list"][i] and bool(r["unc"][i])
        if filt == "sure":
            return r["in_list"][i] and r["p"][i] >= SURE and not r["unc"][i]
        if filt == "control":
            return r["in_list"][i] and r["control"][i]
        if filt in ("übernommen", "verworfen"):
            return (dec.get(f"{uc}|{r['id'][i]}") or {}).get("status") == filt
        return True

    idx = [i for i in range(n) if keep(i)]
    return {"rows": [_row(art, uc, i, dec) for i in idx[offset: offset + limit]], "total": len(idx)}


def nba_rows(filt: str = "all", offset: int = 0, limit: int = 25) -> dict:
    art = load()
    r = art["nba"]["rows"]
    idx = [i for i in range(len(r["cust"])) if filt in ("all", "list") and r["action"][i] != "none"
           or r["action"][i] == filt]
    idx.sort(key=lambda i: (r["action"][i] == "control", -r["ev"][i]))
    return {"rows": [{"rank": k + 1, "cust": r["cust"][i], "action": r["action"][i], "text": r["text"][i],
                      "channel": r["channel"][i], "ev": r["ev"][i], "why": r["why"][i], "conflicts": r["conflicts"][i]}
                     for k, i in enumerate(idx[offset: offset + limit], start=offset)], "total": len(idx)}


# ── Export (generischer CRM-Import) ───────────────────────────────────────────

EXPORT_FIELDS = [
    ("record_id", "Eindeutige ID des Eintrags (für Upsert)"), ("list_id", "Liste / Use Case"),
    ("entity_type", "Kunde oder Position"), ("customer_id", "Kundennummer"), ("country", "Land"),
    ("object", "Gruppe, Artikel oder Beleg"), ("action", "empfohlene Aktion"), ("channel", "Kanal"),
    ("priority", "Priorität (High/Normal)"), ("rank", "Rang in der Liste"), ("score", "Score der Engine"),
    ("probability", "kalibrierte Wahrscheinlichkeit"), ("expected_value_eur", "erwarteter Wert in EUR"),
    ("reason_1", "Grund 1"), ("reason_2", "Grund 2"), ("reason_3", "Grund 3"), ("uncertain", "unsicher (bitte prüfen)"),
    ("uncertainty", "Art der Unsicherheit"), ("control_group", "Kontrollgruppe (nicht kontaktieren)"),
    ("decision_status", "Status: offen / übernommen / verworfen"), ("decision_reason", "Grund der Entscheidung"),
    ("list_date", "Stichtag der Liste"), ("valid_until", "gültig bis"), ("model", "Engine"),
    ("data_version", "Datenstand (Pipeline-Version)"), ("source", "Quelle")]

CRM_MAPPING = {
    "record_id": ("Task.GraphReady_Id__c (External ID, eigenes Feld)", "task.graphready_id (eigene Eigenschaft)",
                  "task.new_graphready_id (eigene Spalte, Alternativschlüssel)"),
    "list_id": ("Campaign.Name", "task.graphready_list (eigene Eigenschaft)", "campaign.name"),
    "customer_id": ("Account.External_Id__c → Task.WhatId", "company.graphready_customer_id → Association",
                    "account.accountnumber → regardingobjectid"),
    "action": ("Task.Subject", "task.hs_task_subject", "task.subject"),
    "channel": ("Task.TaskSubtype (Call/Email/Task)", "task.hs_task_type (CALL/EMAIL/TODO)",
                "Aktivitätstyp phonecall / email / task"),
    "priority": ("Task.Priority (High/Normal)", "task.hs_task_priority (HIGH/MEDIUM)", "task.prioritycode (2/1)"),
    "reason_1": ("Task.Description (Gründe 1–3)", "task.hs_task_body (Gründe 1–3)", "task.description (Gründe 1–3)"),
    "probability": ("Account.GraphReady_Probability__c", "company.graphready_probability", "account.new_graphready_probability"),
    "expected_value_eur": ("Account.GraphReady_Value__c", "company.graphready_value", "account.new_graphready_value"),
    "uncertain": ("Task.GraphReady_Uncertain__c", "task.graphready_uncertain", "task.new_graphready_uncertain"),
    "control_group": ("CampaignMember.Status = „Kontrollgruppe“", "company.graphready_control_group",
                      "account.new_graphready_control"),
    "decision_status": ("Task.Status (Not Started/Completed)", "task.hs_task_status (NOT_STARTED/COMPLETED)",
                        "task.statecode (0 offen / 1 erledigt)"),
    "valid_until": ("Task.ActivityDate", "task.hs_timestamp", "task.scheduledend"),
}
CRMS = {"salesforce": "Salesforce", "hubspot": "HubSpot", "dynamics": "MS Dynamics 365"}


def export_records(uc: str, decisions: dict | None = None, include: str = "actionable") -> tuple[list[dict], list[dict]]:
    """Datensätze im generischen CRM-Format. Kontrollgruppe getrennt (sie darf nicht kontaktiert werden)."""
    art = load()
    if not art:
        return [], []
    recs, ctrl = [], []
    if uc == "next_best_action":
        r = art["nba"]["rows"]
        for i in range(len(r["cust"])):
            if r["action"][i] == "none":
                continue
            d = {"record_id": f"nba|{r['cust'][i]}", "list_id": "next_best_action", "entity_type": "Kunde",
                 "customer_id": r["cust"][i], "country": "", "object": "", "action": r["text"][i],
                 "channel": r["channel"][i], "priority": "High" if r["action"][i] == "retention" else "Normal",
                 "rank": i + 1, "score": "", "probability": "", "expected_value_eur": r["ev"][i],
                 "reason_1": r["why"][i], "reason_2": "; ".join(r["conflicts"][i]), "reason_3": "",
                 "uncertain": False, "uncertainty": "", "control_group": r["action"][i] == "control",
                 "decision_status": "offen", "decision_reason": "", "list_date": art["t_now"],
                 "valid_until": art["valid_until"], "model": "Regelwerk über Churn, Neukunden, Cross-Selling, Empfehlung",
                 "data_version": art["variant"], "source": "GraphReady"}
            (ctrl if d["control_group"] else recs).append(d)
        return recs, ctrl
    L = art["lists"][uc]
    n = len(L["rows"]["id"])
    for i in range(n):
        row = _row(art, uc, i, decisions)
        if not row["in_list"]:
            continue
        if include == "actionable" and row["status"] == "verworfen":
            continue
        ex = row["extra"]
        obj = ex.get("group_name") or (f"{ex.get('product', '')} · Beleg {ex.get('invoice', '')}" if "product" in ex else "")
        d = {"record_id": f"{uc}|{row['id']}", "list_id": uc, "entity_type": L["entity"], "customer_id": row["cust"],
             "country": (art["country_names"][ex["country"]] if 0 <= ex.get("country", -1) < len(art["country_names"]) else ""),
             "object": obj, "action": L["action"] + (f": {ex['group_name']}" if "group_name" in ex else ""),
             "channel": L["channel"], "priority": "High" if row["rank"] <= max(1, L["capacity"] // 3) else "Normal",
             "rank": row["rank"], "score": row["score"], "probability": row["p"], "expected_value_eur": row["ev"],
             "reason_1": (row["reasons"] + ["", "", ""])[0], "reason_2": (row["reasons"] + ["", "", ""])[1],
             "reason_3": (row["reasons"] + ["", "", ""])[2], "uncertain": bool(row["unc"]),
             "uncertainty": "; ".join(UNC_TEXT[u] for u in row["unc"]), "control_group": row["control"],
             "decision_status": row["status"], "decision_reason": row["status_reason"], "list_date": L["anchor"],
             "valid_until": art["valid_until"], "model": L["engine"]["name"], "data_version": art["variant"],
             "source": "GraphReady"}
        (ctrl if row["control"] else recs).append(d)
    return recs, ctrl


def to_csv(records: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=[f for f, _ in EXPORT_FIELDS], extrasaction="ignore")
    w.writeheader()
    for r in records:
        w.writerow(r)
    return buf.getvalue()


def mapping_rows() -> list[dict]:
    out = []
    for f, desc in EXPORT_FIELDS:
        m = CRM_MAPPING.get(f) or CRM_MAPPING.get("reason_1" if f.startswith("reason_") else "", ("", "", ""))
        out.append({"field": f, "description": desc, "salesforce": m[0], "hubspot": m[1], "dynamics": m[2]})
    return out


def mapping_csv() -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["field", "description", "salesforce", "hubspot", "dynamics"])
    w.writeheader()
    for r in mapping_rows():
        w.writerow(r)
    return buf.getvalue()


def to_xlsx(uc: str, records: list[dict], ctrl: list[dict]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    art = load()
    wb = Workbook()
    head = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="4C3FB0")

    def sheet(ws, rows, fields):
        ws.append(fields)
        for c in ws[1]:
            c.font, c.fill = head, fill
        for r in rows:
            ws.append([r.get(f) if not isinstance(r.get(f), bool) else ("ja" if r.get(f) else "nein") for f in fields])
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = min(48, max(10, *(len(str(c.value or "")) for c in col[:200])))
            ws.column_dimensions[col[0].column_letter].width = width + 1

    fields = [f for f, _ in EXPORT_FIELDS]
    ws = wb.active
    ws.title = "Aktionsliste"
    sheet(ws, records, fields)
    sheet(wb.create_sheet("Kontrollgruppe"), ctrl, fields)
    sheet(wb.create_sheet("CRM-Mapping"), mapping_rows(), ["field", "description", "salesforce", "hubspot", "dynamics"])
    info = wb.create_sheet("Info")
    L = art["lists"].get(uc, {}) if art else {}
    for k, v in [("Liste", L.get("name", uc)), ("Stichtag", art["t_now"] if art else ""),
                 ("gültig bis", art["valid_until"] if art else ""), ("Datenstand", art["variant_label"] if art else ""),
                 ("Engine", (L.get("engine") or {}).get("name", "")), ("Schwelle", L.get("threshold_text", "")),
                 ("Kontrollgruppe", f"{CONTROL_SHARE * 100:.0f} %, stratifiziert, Seed {SEED} – nicht kontaktieren"),
                 ("Rückspielweg", "Nur Vorschau – GraphReady sendet nichts an externe Systeme."),
                 ("Hinweis", bc.DISCLAIMER)]:
        info.append([k, v])
    info.column_dimensions["A"].width = 18
    info.column_dimensions["B"].width = 110
    for row in info.iter_rows():
        row[0].font = Font(bold=True)
        row[1].alignment = Alignment(wrap_text=True)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def push_preview(crm: str, records: list[dict], n: int = 3) -> dict:
    """So sähe der Rückspielweg aus – reine Vorschau, es wird nichts gesendet."""
    sample = records[:n]

    def desc(r):
        return "\n".join(x for x in (r["reason_1"], r["reason_2"], r["reason_3"]) if x) + \
            (f"\nErwarteter Wert: {_eur(r['expected_value_eur'])}" if r.get("expected_value_eur") not in ("", None) else "")

    if crm == "salesforce":
        body = {"allOrNone": False, "records": [
            {"attributes": {"type": "Task"}, "Subject": r["action"], "Description": desc(r),
             "Priority": r["priority"], "Status": "Not Started", "ActivityDate": r["valid_until"],
             "TaskSubtype": "Call" if r["channel"].startswith("Telefon") else "Email",
             "GraphReady_Id__c": r["record_id"], "What": {"External_Id__c": str(r["customer_id"])}} for r in sample]}
        return {"method": "POST", "endpoint": "/services/data/v61.0/composite/sobjects", "body": body,
                "auth": "OAuth 2.0 (Connected App) – im Pilot einzurichten",
                "note": "Account-Verknüpfung über das External-ID-Feld External_Id__c"}
    if crm == "hubspot":
        body = {"inputs": [
            {"properties": {"hs_task_subject": r["action"], "hs_task_body": desc(r),
                            "hs_task_priority": "HIGH" if r["priority"] == "High" else "MEDIUM",
                            "hs_task_status": "NOT_STARTED", "hs_timestamp": f"{r['valid_until']}T09:00:00Z",
                            "hs_task_type": "CALL" if r["channel"].startswith("Telefon") else "EMAIL",
                            "graphready_id": r["record_id"]},
             "associations": [{"to": {"id": f"<Company-ID zu Kunde {r['customer_id']}>"},
                               "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 192}]}]}
            for r in sample]}
        return {"method": "POST", "endpoint": "/crm/v3/objects/tasks/batch/create", "body": body,
                "auth": "Private App Token – im Pilot einzurichten",
                "note": "Firmen-ID vorher über die Eigenschaft graphready_customer_id suchen"}
    body = [{"subject": r["action"], "description": desc(r), "prioritycode": 2 if r["priority"] == "High" else 1,
             "scheduledend": f"{r['valid_until']}T09:00:00Z", "new_graphready_id": r["record_id"],
             "regardingobjectid_account_task@odata.bind": f"/accounts(accountnumber='{r['customer_id']}')"}
            for r in sample]
    return {"method": "POST (je Datensatz bzw. als $batch)", "endpoint": "/api/data/v9.2/tasks", "body": body,
            "auth": "Microsoft Entra ID (App-Registrierung) – im Pilot einzurichten",
            "note": "Verknüpfung über den Alternativschlüssel accountnumber"}


# ── Nachbarschafts-Pfade der Produktempfehlung (Detailseite, Abschnitt 3) ─────

def build_reco_paths(n_customers: int = 40, n_neighbors: int = 5, log=print) -> dict:
    """Je Kunde: eigene Artikel → Nachbarkunden (meiste gemeinsame Artikel, 365 Tage vor dem Stichtag) →
    empfohlene Artikel und welche Nachbarn sie gekauft haben. Macht sichtbar, dass andere Kunden einfließen."""
    from scipy import sparse
    art = load()
    vid = art["variant"]
    _ensure_variant(vid)
    b = bt.get_base(vid)
    hist = b.lines[(b.lines["time"] < T_NOW) & (b.lines["time"] >= T_NOW - pd.Timedelta(days=bt.HISTORY))
                   & (b.lines["cust"] > 0)]
    cu = pd.Index(np.sort(hist["cust"].unique()))
    M = sparse.csr_matrix((np.ones(len(hist)), (cu.get_indexer(hist["cust"]), hist["prod"].to_numpy())),
                          shape=(len(cu), b.n_prod))
    M.data[:] = 1
    M.sum_duplicates()
    M.data[:] = 1
    cnt = hist.groupby(["cust", "prod"]).size()
    rr = art["reco"]["rows"]
    order = np.argsort(-np.asarray(rr["ev"]))
    out = []
    for i in order:
        if len(out) >= n_customers:
            break
        c = rr["cust"][i]
        ci = cu.get_loc(c) if c in cu else None
        if ci is None:
            continue
        own = M.getrow(ci).indices
        if len(own) < 8:                         # anschauliche Beispiele: Kunden mit Kaufhistorie
            continue
        overlap = np.asarray((M @ M.getrow(ci).T).todense()).ravel()
        overlap[ci] = 0
        nb = np.argsort(-overlap)[:n_neighbors]
        nb = nb[overlap[nb] > 0]
        top_own = cnt.loc[c].sort_values(ascending=False).index[:5] if c in cnt.index.get_level_values(0) else []
        recs = []
        for p, pr in zip(rr["items"][i], rr["p"][i]):
            buyers = [int(cu[j]) for j in nb if M[j, p]]
            recs.append({"prod": int(p), "name": str(b.prod_names[p])[:40].title(), "p": pr,
                         "own": bool(M[ci, p]), "neighbors": buyers,
                         "neighbor_share": round(len(buyers) / max(len(nb), 1), 3),
                         "buyers_total": int(M[:, p].sum())})
        out.append({"cust": int(c), "n_own": int(len(own)), "own": [{"prod": int(p), "name": str(b.prod_names[p])[:40].title()}
                                                                    for p in top_own],
                    "neighbors": [{"cust": int(cu[j]), "shared": int(overlap[j])} for j in nb],
                    "recs": recs})
    res = {"t_now": T_NOW.date().isoformat(), "variant": vid, "customers": out,
           "note": ("Nachbarn = Kunden mit den meisten gemeinsamen Artikeln in den 365 Tagen vor dem Stichtag. "
                    "Die Empfehlung kommt aus dem Ranker; der Pfad zeigt, welche Nachbarn den Artikel bereits kaufen.")}
    (ARTIFACTS / "reco_paths.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    share = np.mean([r["neighbor_share"] for c in out for r in c["recs"]]) if out else 0
    log(f"[aktivierung] Empfehlungs-Pfade: {len(out)} Kunden · Ø {share:.0%} der Nachbarn kaufen den empfohlenen Artikel")
    return res


def load_reco_paths() -> dict | None:
    p = ARTIFACTS / "reco_paths.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
