"""
Modul 4 · Baseline-Vergleich
============================
Lässt je Use Case alle Engines auf demselben zeitlichen Split laufen,
misst Hauptmetrik, Kurven, Bootstrap-Konfidenzintervalle und den Lift
gegenüber der Regel-Baseline. Daraus: Ampel „Vorhersagekraft“ und eine
Engine-Empfehlung (Backtest + Datenlage nach „Beyond IID“).

Ergebnisse liegen als Artefakt artifacts/backtest_<use_case>.json vor.
"""
from __future__ import annotations

import json
import logging
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from .engines import ENGINE_ORDER, make_engines
from .engines.base import Unavailable

log = logging.getLogger("graph_ml.evaluation")
ARTIFACTS = Path(__file__).parent / "artifacts"
RESULT_VERSION = 1
UC_ORDER = ["churn", "lead_scoring", "cross_selling", "recommendation", "returns_root_cause", "demand_forecast",
            "account_scoring", "territory_coverage"]
# Eigene Rückblicke außerhalb von run_use_case: NBA-Policy (nba_eval.py) und Pricing (pricing.py)
POLICY_UCS = ["next_best_action", "price_recommendation", "price_leakage", "quote_check", "installed_base", "service_contracts",
              "spare_parts_pricing", "quality_root_cause", "supplier_recourse"]          # eigene Rückblicke außerhalb von run_use_case (nba_eval, pricing, aftersales)
NOT_MEASURABLE = {
    "margin": "Einstandspreis fehlt – kein Label, Vorhersagekraft nicht messbar.",
    "warranty_root_cause": "Seriennummer, Charge und Garantiefälle fehlen – gemessen ersatzweise als Qualitäts-Frühwarnung (Ausschuss).",
    "warranty_fraud": "Keine offenen Garantiefälle mit Händler, Techniker und Prüfergebnis – nur Struktur bewertbar.",
}
MAIN_LABEL = {"recall_at_10": "Recall@Top-10 %", "map_at_12": "MAP@12", "wape": "WAPE", "auprc": "AUPRC",
              "value_captured_at_10": "Umsatz-Recall@Top-10 %"}


def _r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 5)
    return obj


# ── eine Engine auf einer Aufgabe ─────────────────────────────────────────────

def run_engine(task: bt.Task, engine, rule_scores=None) -> dict:
    t0 = time.time()
    engine.timed_fit(task)
    pred = engine.predict(task)
    runtime = time.time() - t0
    res = {"id": engine.id, "status": "ok", "runtime_s": round(runtime, 1), "meta": engine.meta()}
    metric = bt.MAIN_METRIC[task.uc]
    if task.kind == "binary":
        y = task.test[task.label].to_numpy()
        val = task.test[task.value_col].to_numpy() if task.value_col in task.test else None
        s = np.asarray(pred, dtype=float)
        rank = s * np.clip(val, 0, None) if task.extra.get("rank_by_value") and engine.proba else None
        if rank is not None:
            res["ranking"] = "erwarteter Umsatzverlust (Wahrscheinlichkeit × Umsatz)"
        res["metrics"] = bt.binary_metrics(y, s, val, proba=engine.proba, rank=rank)
        res["curves"] = bt.curves(y, s, proba=engine.proba, rank=rank, value=val)
        if rule_scores is not None and engine.id != "rule":
            res["ci"] = bt.bootstrap_lift(y, s, rule_scores, metric, value=val, r_model=rank)
        else:
            res["ci"] = bt.bootstrap_lift(y, s, s, metric, value=val, r_model=rank)
            res["ci"]["lift_ci"] = None
        res["_scores"] = s
    elif task.kind == "ranking":
        recs, truth = pred
        m = bt.ranking_metrics(recs, truth)
        res["metrics"] = {k: v for k, v in m.items() if not k.startswith("_")}
        res["ci"] = {"metric_ci": bt.metric_ci(m["_ap"])}
        res["_ap"] = m["_ap"]
        res["_recs"] = recs[:20]
        res["_hit_stats"] = _reco_hits(task, recs, truth)
    else:
        yhat, ytrue, scale = pred
        res["metrics"] = bt.forecast_metrics(ytrue, yhat, scale)
        err = np.abs(ytrue - yhat).sum(axis=1)
        res["_series_err"], res["_series_tot"] = err, np.abs(ytrue).sum(axis=1)
        res["_yhat"], res["_ytrue"] = yhat, ytrue
    try:
        res["explain"] = engine.explain(task)
    except Exception as exc:  # Erklärung ist optional
        res["explain"] = None
        log.warning("Erklärung %s/%s: %s", task.uc, engine.id, exc)
    return res


def _reco_hits(task, recs, truth) -> dict:
    """Treffer der Empfehlungen und deren Umsatz im 90-Tage-Fenster (für das Euro-Potenzial)."""
    b = bt.base_of(task)
    snap_users = bt.reco_snapshot(b, bt.T_TEST)["users"]
    fut = b.lines[(b.lines["time"] >= bt.T_TEST) & (b.lines["time"] < bt.T_TEST + pd.Timedelta(days=bt.HORIZON))]
    val = fut.groupby(["cust", "prod"])["value"].sum()
    hits = hit_value = 0.0
    for u, (r, t) in enumerate(zip(recs, truth)):
        h = [p for p in r[:12] if p in t]
        hits += len(h)
        hit_value += float(sum(val.get((snap_users[u], p), 0.0) for p in h))
    return {"hits": hits, "hit_value": hit_value, "customers": len(recs)}


def _whatif(task, results: list[dict]) -> dict | None:
    """Rohdaten für die Was-wäre-wenn-Rechnung: je Engine und Listengröße erkannte Fälle und deren Wert."""
    b = bt.base_of(task)
    ok = [r for r in results if r.get("status") == "ok"]
    cur = b.currency
    if task.kind == "binary":
        te = task.test
        y = te[task.label].to_numpy().astype(bool)
        if task.uc == "churn":
            v = te["monetary_365"].to_numpy() / 4          # Umsatz, der im 90-Tage-Fenster ausbleibt
            unit, period = "Umsatz der Abwanderer im Quartal (Ø Quartalsumsatz)", 90
        elif task.uc == "lead_scoring":
            fut = b.inv[(b.inv["time"] >= bt.T_TEST) & (b.inv["time"] < bt.T_TEST + pd.Timedelta(days=bt.HORIZON))]
            v = te["cust"].map(fut.groupby("cust")["value"].sum()).fillna(0).to_numpy()
            unit, period = "Umsatz der Wiederkäufer in 90 Tagen", 90
        elif task.uc == "territory_coverage":
            v = te["next_revenue"].to_numpy()
            unit, period = "Händlerumsatz im nächsten Quartal", 90
        elif task.uc == "account_scoring":
            v = te["value"].to_numpy()
            unit, period = "Mehrumsatz der wachsenden Accounts im Quartal", 90
        elif task.uc == "cross_selling":
            fut = b.lines[(b.lines["time"] >= bt.T_TEST) & (b.lines["time"] < bt.T_TEST + pd.Timedelta(days=bt.HORIZON))]
            gv = fut.groupby(["cust", "group"])["value"].sum()
            v = np.array([gv.get((c, g), 0.0) for c, g in zip(te["cust"], te["group"])])
            unit, period = "Umsatz in der neuen Gruppe in 90 Tagen", 90
        else:
            v = te["value"].to_numpy()
            unit, period = "Warenwert der stornierten Positionen", 61
        N = len(te)
        ks = list(range(1, 31))
        engines = {}
        for r in ok:
            s = r.get("_scores")
            if s is None:
                continue
            proba = r["id"] not in ("rule",) and r["meta"].get("klass") != "Regel"
            order_s = s * np.clip(te[task.value_col].to_numpy(), 0, None) if task.extra.get("rank_by_value") and proba else s
            order = np.argsort(-order_s, kind="stable")
            yy, vv = y[order], (v * y)[order]
            cy, cv = np.cumsum(yy), np.cumsum(vv)
            ns = [max(1, int(round(N * k / 100))) for k in ks]
            engines[r["id"]] = {"tp": [int(cy[n - 1]) for n in ns], "value": [round(float(cv[n - 1]), 2) for n in ns]}
        return {"kind": "list", "k": ks, "n": [max(1, int(round(N * k / 100))) for k in ks], "N": N,
                "positives": int(y.sum()), "value_total": round(float((v * y).sum()), 2), "unit": unit,
                "period_days": period, "currency": cur, "engines": engines}
    if task.kind == "ranking":
        return {"kind": "reco", "period_days": 90, "currency": cur,
                "engines": {r["id"]: r["_hit_stats"] for r in ok if r.get("_hit_stats")}}
    if task.kind == "forecast":
        prods = task.extra["prods"]
        price = b.lines[b.lines["time"] < bt.T_TEST].groupby("prod")["price"].median().reindex(prods).fillna(0).to_numpy()
        eng = {}
        for r in ok:
            yt, yh = r["_ytrue"], r["_yhat"]
            eng[r["id"]] = {"wape": float(np.abs(yt - yh).sum() / max(yt.sum(), 1e-9))}
        yt = ok[0]["_ytrue"] if ok else None
        return {"kind": "forecast", "period_days": 91, "currency": cur, "engines": eng,
                "demand_value": round(float((yt * price[:, None]).sum()), 2) if yt is not None else None,
                "series": len(prods)}
    return None


def _paired_lift(task, best: dict, rule: dict, n: int = 200) -> dict:
    rng = np.random.default_rng(bt.SEED)
    if task.kind == "ranking":
        a, r = best["_ap"], rule["_ap"]
        lifts = []
        for _ in range(n):
            i = rng.integers(0, len(a), len(a))
            lifts.append((a[i].mean() - r[i].mean()) / max(r[i].mean(), 1e-9))
        return {"lift_ci": [float(np.quantile(lifts, 0.05)), float(np.quantile(lifts, 0.95))]}
    if task.kind == "forecast":
        eb, er, tot = best["_series_err"], rule["_series_err"], best["_series_tot"]
        lifts, vals = [], []
        for _ in range(n):
            i = rng.integers(0, len(eb), len(eb))
            wb, wr = eb[i].sum() / tot[i].sum(), er[i].sum() / tot[i].sum()
            vals.append(wb)
            lifts.append((wr - wb) / max(wr, 1e-9))
        return {"lift_ci": [float(np.quantile(lifts, 0.05)), float(np.quantile(lifts, 0.95))],
                "metric_ci": [float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))]}
    return {"lift_ci": best.get("ci", {}).get("lift_ci")}


def vorhersagekraft(lift: float | None, lift_ci) -> dict:
    """Schwellen aus 0.7: Grün ≥ +20 % und untere KI-Grenze > 0 · Gelb +5…+20 % · Rot darunter."""
    if lift is None:
        return {"status": "na", "reason": "nicht messbar"}
    lo = lift_ci[0] if lift_ci else None
    significant = lo is None or lo > 0
    if lift >= 0.20 and significant:
        st = "ok"
    elif lift >= 0.05 and significant:
        st = "warn"
    else:
        st = "bad"                         # unter +5 % oder nicht signifikant (0.7)
    sig = "signifikant" if lo is not None and lo > 0 else "nicht signifikant"
    return {"status": st, "lift": round(lift, 4), "lift_ci": [round(x, 4) for x in lift_ci] if lift_ci else None,
            "significant": bool(lo is not None and lo > 0),
            "reason": f"{lift * 100:+.1f} % gegenüber der Regel-Baseline ({sig})".replace(".", ",")}


def _family(klass: str) -> str:
    """Modellfamilie: Baum · Graph (Graph-ML light, Relational, RFM) · TFM · Zeitreihen-FM · Regel."""
    k = klass or ""
    if "Relational" in k or "Graph" in k or "GNN" in k or "LightGCN" in k:
        return "graph"
    if "TFM" in k or "TabPFN" in k:
        return "tfm"
    if "Zeitreihen" in k or "Aurora" in k:
        return "tsfm"
    if "Baum" in k or "LightGBM" in k:
        return "tree"
    return "rule"


def recommend(uc: str, engines: list[dict], best: dict | None, profile_engine: dict | None,
              vk: dict | None = None) -> dict:
    ok = [e for e in engines if e["status"] == "ok" and e["id"] != "rule" and e["id"] != "popularity"]
    if not best:
        return {"engine": None, "text": "Kein Modell gerechnet."}
    m = bt.MAIN_METRIC[uc]
    lower = m in bt.LOWER_IS_BETTER
    by_class: dict[str, dict] = {}
    for e in ok:
        k = e["meta"]["klass"]
        v = e["metrics"][m]
        if k not in by_class or (v < by_class[k]["metrics"][m] if lower else v > by_class[k]["metrics"][m]):
            by_class[k] = e
    tree = by_class.get("Baum")
    text = f"Bestes Modell: {best['meta']['name']} ({MAIN_LABEL.get(m, m)} {best['metrics'][m]:.3f})."
    if vk and vk.get("status") == "bad":
        text += (" Es schlägt die Regel-Baseline aber nicht belastbar – Regel vorerst beibehalten und den Use Case "
                 "nach den Fix-Playbooks bzw. mit mehr Daten erneut prüfen.")
        return {"engine": "rule", "name": "Regel-Baseline", "klass": "Regel", "text": text, "model_best": best["id"]}
    if tree and best["id"] != tree["id"]:
        tv, bv = tree["metrics"][m], best["metrics"][m]
        gain = (tv - bv) / tv if lower else (bv - tv) / max(tv, 1e-9)
        text += f" Gegenüber der Baum-Baseline {gain * 100:+.1f} %.".replace(".", ",", 1)
        if abs(gain) < 0.02:
            text += " Der Unterschied ist klein – die einfachere Baum-Baseline genügt für den Pilot."
    if profile_engine:
        prim = profile_engine.get("primary", "")
        agrees = _family(prim) == _family(best["meta"]["klass"])
        text += (" Bestätigt die Einschätzung aus dem Profilcheck." if agrees else
                 f" Profilcheck hatte „{prim}“ erwartet – der Rückblick korrigiert die Empfehlung.")
    return {"engine": best["id"], "name": best["meta"]["name"], "klass": best["meta"]["klass"], "text": text,
            "by_class": {k: {"id": v["id"], "value": _r(v["metrics"][m])} for k, v in by_class.items()}}


# ── ein Use Case komplett ─────────────────────────────────────────────────────

def run_use_case(uc: str, engine_ids: list[str] | None = None, ctx=None, previous: dict | None = None,
                 variant: str = "v0") -> dict:
    """Ein Use Case komplett – auf der Zeitachse seines Datensatzes (z. B. Account-Scoring auf WideWorldImporters)."""
    if variant != "v0" and uc in bt.DATASET_OF:
        raise ValueError(f"{uc} wird auf {bt.DATASET_OF[uc]} gerechnet – Pipeline-Varianten gelten für Online Retail II")
    with bt.dataset_time(bt.dataset_key(uc, variant)):
        res = _run_use_case(uc, engine_ids, ctx, previous, variant)
    res["dataset"] = bt.DATASET_NAMES.get(bt.dataset_key(uc, variant), "online_retail_ii")
    return res


def _run_use_case(uc: str, engine_ids, ctx, previous, variant) -> dict:
    t0 = time.time()
    if ctx:
        ctx.progress(3, f"Baue zeitlichen Split für {uc} …")
    task = bt.get_task(uc, variant)
    engines = make_engines(task)
    if engine_ids:
        engines = [e for e in engines if e.id in engine_ids or e.id == "rule"]
    results, prev = [], {e["id"]: e for e in (previous or {}).get("engines", [])}
    rule_res = None
    for i, eng in enumerate(engines):
        if ctx:
            ctx.progress(5 + int(85 * i / max(len(engines), 1)), f"{eng.name} …")
        ok, why = eng.available()
        if not ok:
            keep = prev.get(eng.id)
            if keep and keep.get("status") == "ok":
                results.append(keep)          # z. B. GPU-Ergebnis aus der Vorberechnung behalten
            else:
                results.append({"id": eng.id, "status": "unavailable", "reason": why, "meta": eng.meta()})
            continue
        try:
            if getattr(eng, "remote", False) and variant == "v0":
                from .engines.runtime import run_remote
                r = run_remote(uc, eng.id, ctx)
            else:
                r = run_engine(task, eng, rule_scores=rule_res["_scores"] if rule_res and task.kind == "binary" else None)
            results.append(r)
            if eng.id == "rule":
                rule_res = r
            if ctx:
                ctx.log(f"{eng.name}: fertig in {r.get('runtime_s')} s")
        except Unavailable as exc:
            results.append({"id": eng.id, "status": "unavailable", "reason": str(exc), "meta": eng.meta()})
        except Exception as exc:
            log.error("Engine %s/%s: %s\n%s", uc, eng.id, exc, traceback.format_exc())
            results.append({"id": eng.id, "status": "failed", "reason": str(exc)[:300], "meta": eng.meta()})
    return assemble(uc, task, results, time.time() - t0)


def assemble(uc: str, task: bt.Task, results: list[dict], seconds: float) -> dict:
    m = bt.MAIN_METRIC[uc]
    lower = m in bt.LOWER_IS_BETTER
    rule = next((r for r in results if r["id"] == "rule" and r["status"] == "ok"), None)
    cands = [r for r in results if r["status"] == "ok" and r["id"] not in ("rule", "popularity")]
    best = (min if lower else max)(cands, key=lambda r: r["metrics"][m]) if cands else None
    for r in results:
        if r["status"] != "ok" or not rule or r is rule:
            continue
        v, rv = r["metrics"][m], rule["metrics"][m]
        r["lift"] = _r((rv - v) / rv if lower else (v - rv) / max(rv, 1e-9))
    vk = {"status": "na", "reason": "kein Modell"}
    if best and rule:
        lift = best["lift"]
        extra = _paired_lift(task, best, rule) if task.kind != "binary" else {"lift_ci": best["ci"].get("lift_ci")}
        best["ci"] = {**best.get("ci", {}), **extra}
        vk = vorhersagekraft(lift, best["ci"].get("lift_ci"))
    profile_engine = _profile_engine(uc)
    out = {
        "result_version": RESULT_VERSION, "use_case": uc, "kind": task.kind,
        "variant": "v0" if bt.DATASET_OF.get(uc) == task.extra.get("base_key") else task.extra.get("base_key", "v0"),
        "currency": task.extra.get("currency", "GBP"),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "host": platform.node(), "compute_seconds": round(seconds, 1),
        "main_metric": m, "main_metric_label": MAIN_LABEL.get(m, m), "lower_is_better": lower,
        "timeline": task.timeline, "rule_text": task.extra.get("rule_text"),
        "sizes": _sizes(task), "audit": {"checks": len(task.audit.rows), "violations": 0,
                                         "sample": task.audit.rows[:3]},
        "engines": [_order_key(r) for r in results],
        "best": best["id"] if best else None,
        "vorhersagekraft": vk,
        "recommendation": recommend(uc, results, best, profile_engine, vk),
        "notes": _notes(task),
    }
    out["top_list"] = _top_list(task, best) if best else []
    out["whatif"] = _whatif(task, results)
    if task.kind == "forecast" and best:
        out["series_examples"] = _series_examples(task, best, rule)
    if uc == "returns_root_cause" and best:
        out["cohorts"] = _cohorts(task, best)
    return _clean(out)


def _order_key(r):
    return {k: v for k, v in r.items()}


def _sizes(task) -> dict:
    if task.kind == "binary":
        return {"train_rows": int(len(task.train)), "test_rows": int(len(task.test)),
                "features": len(task.features), "graph_features": len(task.graph_features),
                "train_positive_rate": _r(task.train[task.label].mean()),
                "test_positive_rate": _r(task.test[task.label].mean())}
    if task.kind == "forecast":
        return {"series": int(task.extra["Y"].shape[0]), "horizon_weeks": task.extra["horizon"],
                "train_origins": len(task.extra["origins_train"])}
    return {"train_snapshots": len(task.snapshots)}


def _notes(task) -> list[str]:
    n = []
    if task.extra.get("groups_provisional"):
        n.append("Artikelgruppen vorläufig per Schlüsselwort-Regeln – exakt erst nach Playbook PB-05.")
    if task.uc == "lead_scoring":
        n.append("Leakage-sichere Definition: Erstkäufer mit genau einem Beleg vor dem Stichtag; Label = Kauf im "
                 "90-Tage-Fenster danach. Kleine Testmenge → breite Konfidenzintervalle.")
    if task.uc == "returns_root_cause":
        n.append("Trainingsmonate je höchstens 25.000 Positionen (Zufallsstichprobe); Test vollständig.")
    return n


def _profile_engine(uc: str) -> dict | None:
    try:
        from .data_loader import get_registry
        res = get_registry().checks("online_retail_ii", 2)
        return next((u.get("engine") for u in res["use_cases"] if u["id"] == uc), None)
    except Exception:
        return None


def _top_list(task, best) -> list[dict]:
    if task.kind == "binary":
        s = best["_scores"]
        order = s * np.clip(task.test[task.value_col].to_numpy(), 0, None) if task.extra.get("rank_by_value") else s
        te = task.test.assign(_s=s, _o=order).sort_values("_o", ascending=False).head(15)
        cols = {"churn": ["recency", "freq_365", "monetary_365"], "lead_scoring": ["first_value", "first_products"],
                "account_scoring": ["monetary_90", "trend_revenue", "freq_90"],
                "territory_coverage": ["recency", "monetary_365", "g_portfolio_active30"],
                "cross_selling": ["group", "group_pop"], "returns_root_cause": ["prod", "qty", "price"]}[task.uc]
        out = []
        b = bt.base_of(task)
        for _, r in te.iterrows():
            d = {"id": str(r[task.id_col]) if task.id_col in r else str(r.get("cust")), "score": _r(r["_s"]),
                 "label": int(r[task.label]), "cust": int(r["cust"]) if "cust" in r else None}
            for c in cols:
                d[c] = _r(r[c], 2) if c not in ("group", "prod") else int(r[c])
            if "group" in d:
                d["group_name"] = b.group_names[d["group"]]
            if "prod" in d:
                d["product_name"] = str(b.prod_names[d["prod"]])[:40]
            out.append(d)
        return out
    if task.kind == "ranking":
        b = bt.base_of(task)
        snap_users = bt.reco_snapshot(b, bt.T_TEST)["users"]
        return [{"cust": int(snap_users[u]), "items": [str(b.prod_names[p])[:32] for p in recs[:5]]}
                for u, recs in enumerate(best.get("_recs", [])[:8])]
    return []


def _series_examples(task, best, rule) -> list[dict]:
    Y = task.extra["Y"].to_numpy(dtype=float)
    weeks = task.extra["weeks"]
    o = weeks.get_loc(task.extra["origin_test"])
    H = task.extra["horizon"]
    out = []
    for s in range(min(6, Y.shape[0])):
        out.append({"name": str(task.extra["names"][s])[:40],
                    "weeks": [w.date().isoformat() for w in weeks[o - 26:o + H]],
                    "history": [float(x) for x in Y[s, o - 26:o]],
                    "actual": [float(x) for x in best["_ytrue"][s]],
                    "model": [round(float(x), 1) for x in best["_yhat"][s]],
                    "naive": [round(float(x), 1) for x in rule["_yhat"][s]] if rule else None})
    return out


def _cohorts(task, best) -> list[dict]:
    """Root Cause: Kohorten (Artikelgruppe × Land) mit erhöhtem vorhergesagtem und tatsächlichem Risiko."""
    b = bt.base_of(task)
    GROUPS = b.group_names
    te = task.test.assign(_s=best["_scores"])
    base = te[task.label].mean()
    g = te.groupby(["group", "country"]).agg(n=("y", "size"), rate=("y", "mean"), pred=("_s", "mean"))
    g = g[g["n"] >= 150].sort_values("pred", ascending=False).head(10).reset_index()
    return [{"group": GROUPS[int(r.group)], "country": str(b.country_names[int(r.country)]), "n": int(r.n),
             "rate": _r(r.rate), "pred": _r(r.pred), "lift": _r(r.rate / max(base, 1e-9))} for r in g.itertuples()]


# ── Artefakte ─────────────────────────────────────────────────────────────────

def artifact_path(uc: str, variant: str = "v0") -> Path:
    return ARTIFACTS / (f"backtest_{uc}.json" if variant == "v0" else f"backtest_{uc}~{variant}.json")


def save(res: dict) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    artifact_path(res["use_case"], res.get("variant", "v0")).write_text(
        json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load(uc: str, variant: str = "v0") -> dict | None:
    p = artifact_path(uc, variant)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def summary(variant: str = "v0") -> dict:
    """Übersicht für Schritt 4 und die Ampel: je Use Case Sieger, Lift, Vorhersagekraft."""
    out = {}
    for uc in UC_ORDER + POLICY_UCS:
        r = load(uc, variant)
        if not r:
            out[uc] = {"status": "pending"}
            continue
        best = next((e for e in r["engines"] if e["id"] == r.get("best")), None)
        out[uc] = {"status": "ok", "vorhersagekraft": r["vorhersagekraft"], "best": r.get("best"),
                   "best_name": best["meta"]["name"] if best else None, "main_metric_label": r["main_metric_label"],
                   "best_value": best["metrics"][r["main_metric"]] if best else None,
                   "generated_at": r["generated_at"], "host": r.get("host"),
                   "pending_engines": [e["id"] for e in r["engines"] if e["status"] != "ok"]}
    for uc, why in NOT_MEASURABLE.items():
        out[uc] = {"status": "na", "vorhersagekraft": {"status": "na", "reason": why}}
    for uc in POLICY_UCS:
        if out.get(uc, {}).get("status") == "pending":
            out[uc] = {"status": "na", "vorhersagekraft": {"status": "na", "reason": "Policy-Rückblick noch nicht gerechnet"}}
    return out
