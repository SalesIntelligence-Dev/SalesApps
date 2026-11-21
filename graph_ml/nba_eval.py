"""
Next Best Action – Rückblick als Offline-Policy-Vergleich
=========================================================
Ohne Reaktionsdaten lässt sich kein NBA-Modell trainieren. Messbar ist aber, welchen Wert die gewählten Aktionen
am Rückblick-Stichtag (01.09.2011) gehabt hätten, weil die Ausgänge der 90 Tage danach bekannt sind:

  Retention-Anruf  wertvoll, wenn der Kunde tatsächlich abwanderte   → Quartalsumsatz × Rettungsquote − Kosten
  Cross-Selling    wertvoll, wenn er die Gruppe tatsächlich kaufte    → Umsatz in der Gruppe × Quote − Kosten
  Neukunden-Impuls wertvoll, wenn er tatsächlich wiederkaufte        → Folgeumsatz × Quote − Kosten
  Empfehlung       wertvoll, wenn er die Artikel tatsächlich kaufte  → Umsatz der Treffer × Quote − Kosten

Verglichen werden drei Policies mit derselben Kontaktlogik (eine Aktion je Kunde und Quartal):
  NBA mit Modellen   Regelwerk aus activation.build_nba über die Scores von LightGBM + Graph-Merkmalen
  Regel-Policy       „seit 60 Tagen inaktiv, nach Umsatz“ anrufen, alle anderen bekommen den Topseller-Newsletter
  Einheitsaktion     alle bekommen den Topseller-Newsletter

Gemessen wird also die Treffsicherheit der Auswahl – die Wirkungsquoten bleiben Annahmen (wie in Schritt 5).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import backtest as bt
from . import business_case as bc

FX = bc.FX_DEFAULT


def _fit_scores(uc: str):
    from .engines.cpu import LGBMGraphEngine
    task = bt.get_task(uc)
    e = LGBMGraphEngine().timed_fit(task)
    return task, np.asarray(e.predict(task), dtype=float)


def _future(b, T, days=bt.HORIZON):
    return b.lines[(b.lines["time"] >= T) & (b.lines["time"] < T + pd.Timedelta(days=days))]


def evaluate(log=print, n_boot: int = 300) -> dict:
    t0 = time.time()
    T = bt.T_TEST
    b = bt.get_base("v0")
    P = {uc: bc.defaults(uc) for uc in ("churn", "cross_selling", "lead_scoring", "recommendation")}
    fut_l, fut_i = _future(b, T), b.inv[(b.inv["time"] >= T) & (b.inv["time"] < T + pd.Timedelta(days=bt.HORIZON))]

    # Churn
    tc, sc = _fit_scores("churn")
    ch = pd.DataFrame({"cust": tc.test["cust"].to_numpy(), "p": sc, "y": tc.test["y"].to_numpy(),
                       "v": tc.test["monetary_365"].to_numpy() / 4 * FX, "rule": tc.test["rule"].to_numpy()})
    pr = P["churn"]
    ch["gain"] = ch["v"] * pr["rate"] / 100 * pr["margin"] / 100
    ch["ev"] = ch["p"] * ch["gain"] - pr["cost"]
    ch["real"] = ch["y"] * ch["gain"] - pr["cost"]
    cap = int(round(len(ch) * pr["list_share"] / 100))
    log(f"[nba] Churn: {len(ch)} Kunden, Anruf-Kapazität {cap}")

    # Cross-Selling (beste Gruppe je Kunde)
    tx, sx = _fit_scores("cross_selling")
    te = tx.test
    gv = fut_l.groupby(["cust", "group"])["value"].sum()
    tv = te["cutoff"].iloc[0]
    val_rows = tx.train[tx.train["cutoff"] == tx.val_cutoff]
    fv = _future(b, tx.val_cutoff).groupby(["cust", "group"])["value"].sum()
    vpos = val_rows[val_rows["y"] == 1]
    est = pd.Series([fv.get((c, g), 0.0) for c, g in zip(vpos["cust"], vpos["group"])]).groupby(vpos["group"].to_numpy()).median()
    cx = pd.DataFrame({"cust": te["cust"].to_numpy(), "group": te["group"].to_numpy(), "p": sx, "y": te["y"].to_numpy()})
    cx["est"] = cx["group"].map(est).fillna(float(est.median()) if len(est) else 0.0) * FX
    cx["spend"] = [gv.get((c, g), 0.0) * FX for c, g in zip(cx["cust"], cx["group"])]
    pr = P["cross_selling"]
    cx["ev"] = cx["p"] * cx["est"] * pr["rate"] / 100 - pr["cost"]
    cx["real"] = cx["y"] * cx["spend"] * pr["rate"] / 100 - pr["cost"]
    cx = cx.sort_values(["cust", "ev"], ascending=[True, False]).drop_duplicates("cust")
    log(f"[nba] Cross-Selling: {len(cx)} Kunden (beste Gruppe) · Stichtag {pd.Timestamp(tv).date()}")

    # Neukunden
    tl, sl = _fit_scores("lead_scoring")
    rev = fut_i.groupby("cust")["value"].sum()
    vl = tl.train[tl.train["cutoff"] == tl.val_cutoff]
    vrev = b.inv[(b.inv["time"] >= tl.val_cutoff) & (b.inv["time"] < tl.val_cutoff + pd.Timedelta(days=bt.HORIZON))].groupby("cust")["value"].sum()
    vp = vl[vl["y"] == 1]
    ratio = float(np.median(vp["cust"].map(vrev).fillna(0) / vp["first_value"].clip(lower=1))) if len(vp) else 1.0
    ld = pd.DataFrame({"cust": tl.test["cust"].to_numpy(), "p": sl, "y": tl.test["y"].to_numpy(),
                       "est": tl.test["first_value"].to_numpy() * ratio * FX})
    ld["rev"] = ld["cust"].map(rev).fillna(0).to_numpy() * FX
    pr = P["lead_scoring"]
    ld["ev"] = ld["p"] * ld["est"] * pr["rate"] / 100 - pr["cost"]
    ld["real"] = ld["y"] * ld["rev"] * pr["rate"] / 100 - pr["cost"]

    # Empfehlung (Top-3 je Kunde aus dem LightGBM-Ranker) und Topseller-Newsletter
    from .engines.cpu import RANK_FEATS, LGBMRankEngine, _rank_pairs, _top_pop
    task_r = bt.get_task("recommendation")
    eng = LGBMRankEngine().timed_fit(task_r)
    snap = bt.reco_snapshot(b, T)
    pairs = _rank_pairs(snap, False, b)
    pairs["s"] = eng.model.predict(pairs[RANK_FEATS], num_iteration=eng.model.best_iteration)
    top = pairs.sort_values(["u", "s"], ascending=[True, False]).groupby("u").head(3)
    hist = b.lines[b.lines["time"] < T]
    lv = hist.groupby("prod")["value"].median() * FX
    bought = fut_l.groupby(["cust", "prod"])["value"].sum() * FX
    pr = P["recommendation"]
    users = snap["users"]

    def reco_value(items_by_user):
        ev, real = {}, {}
        for u, items in items_by_user.items():
            c = int(users[u])
            ps = items.get("p")
            ev[c] = float(np.sum(ps * lv.reindex(items["prod"]).fillna(0).to_numpy()) * pr["rate"] / 100 - pr["cost"]) \
                if ps is not None else np.nan
            real[c] = float(sum(bought.get((c, p), 0.0) for p in items["prod"]) * pr["rate"] / 100 - pr["cost"])
        return ev, real

    model_items = {u: {"prod": g["p"].to_numpy(), "p": g["s"].to_numpy()} for u, g in top.groupby("u")}
    rc_ev, rc_real = reco_value(model_items)
    pop3 = np.asarray(_top_pop(snap, 3))
    _, news_real = reco_value({u: {"prod": pop3} for u in range(len(users))})
    log(f"[nba] Empfehlung: {len(rc_ev)} Kunden · Topseller-Newsletter als Einheitsaktion")

    # Policies je Kunde
    custs = sorted(set(ch["cust"]) | set(cx["cust"]) | set(ld["cust"]) | set(rc_ev))
    chi, cxi, ldi = ch.set_index("cust"), cx.set_index("cust"), ld.set_index("cust")
    in_cap = set(ch.sort_values("ev", ascending=False).head(cap)["cust"])
    in_cap = {c for c in in_cap if chi.at[c, "ev"] > 0}
    rule_cap = set(ch.sort_values("rule", ascending=False).head(cap)["cust"])
    rows = []
    for c in custs:
        cand = {}
        if c in in_cap:
            cand["retention"] = (chi.at[c, "ev"], chi.at[c, "p"], chi.at[c, "real"])
        if c in ldi.index and ldi.at[c, "ev"] > 0:
            cand["onboarding"] = (ldi.at[c, "ev"], ldi.at[c, "p"], ldi.at[c, "real"])
        if c in cxi.index and cxi.at[c, "ev"] > 0:
            cand["cross"] = (cxi.at[c, "ev"], cxi.at[c, "p"], cxi.at[c, "real"])
        if c in rc_ev and rc_ev[c] > 0:
            cand["reco"] = (rc_ev[c], np.nan, rc_real[c])
        if "retention" in cand and cand["retention"][1] >= 0.5:
            act = "retention"
        elif "onboarding" in cand:
            act = "onboarding"
        elif cand:
            act = max(cand, key=lambda k: cand[k][0])
        else:
            act = "none"
        nba_real = cand[act][2] if act in cand else 0.0
        if c in rule_cap:
            rule_act, rule_real = "retention", (chi.at[c, "real"] if c in chi.index else 0.0)
        else:
            rule_act, rule_real = "newsletter", news_real.get(c, 0.0)
        rows.append({"cust": c, "nba": act, "nba_real": nba_real, "rule": rule_act, "rule_real": rule_real,
                     "single_real": news_real.get(c, -P["recommendation"]["cost"])})
    df = pd.DataFrame(rows)

    def stats(col, act_col=None):
        real = df[col].to_numpy(float)
        out = {"net": round(float(real.sum()), 2), "contacts": int((df[act_col] != "none").sum()) if act_col else len(df),
               "per_customer": round(float(real.mean()), 3)}
        if act_col:
            out["actions"] = {k: int(v) for k, v in df[act_col].value_counts().items()}
            out["by_action"] = {k: round(float(g[col].sum()), 2) for k, g in df.groupby(act_col)}
        return out

    pol = {"nba_model": stats("nba_real", "nba"), "rule_policy": stats("rule_real", "rule"),
           "single_action": stats("single_real")}
    rng = np.random.default_rng(bt.SEED)
    a, r = df["nba_real"].to_numpy(float), df["rule_real"].to_numpy(float)
    base = r.sum()
    lift = (a.sum() - base) / abs(base) if base else None
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(df), len(df))
        rb = r[i].sum()
        if rb:
            boots.append((a[i].sum() - rb) / abs(rb))
    ci = [round(float(np.percentile(boots, 5)), 4), round(float(np.percentile(boots, 95)), 4)] if boots else None
    log(f"[nba] Netto-Wert je Quartal: NBA {pol['nba_model']['net']:,.0f} € · Regel {pol['rule_policy']['net']:,.0f} € · "
        f"Einheit {pol['single_action']['net']:,.0f} € · Lift {lift:+.1%} · {time.time() - t0:.0f} s")
    return {"policies": pol, "lift": lift, "lift_ci": ci, "customers": len(df), "capacity": cap,
            "seconds": round(time.time() - t0, 1), "T": T.date().isoformat()}


def to_result(ev_res: dict) -> dict:
    """Im Format der Rückblick-Artefakte (evaluation), damit Schritt 5, Ampel und Business Case es lesen."""
    from . import evaluation as ev
    names = {"nba_model": ("NBA mit Modellen (LightGBM + Graph)", "Graph-ML light"),
             "rule_policy": ("Regel-Policy (Inaktive anrufen, sonst Newsletter)", "Regel"),
             "single_action": ("Einheitsaktion (Topseller-Newsletter für alle)", "Regel")}
    engines = [{"id": {"nba_model": "nba", "rule_policy": "rule", "single_action": "single"}[k], "status": "ok",
                "meta": {"name": names[k][0], "klass": names[k][1], "device": "CPU"},
                "metrics": {"net_value": v["net"], "per_customer": v["per_customer"]}, "policy": v}
               for k, v in ev_res["policies"].items()]
    vk = ev.vorhersagekraft(ev_res["lift"], ev_res["lift_ci"])
    w = {"kind": "policy", "currency": "EUR", "period_days": 90,
         "engines": {e["id"]: {"net": e["metrics"]["net_value"]} for e in engines},
         "unit": "realisierter Netto-Wert der gewählten Aktionen im Quartal"}
    return {"result_version": ev.RESULT_VERSION, "use_case": "next_best_action", "kind": "policy", "variant": "v0",
            "currency": "EUR", "dataset": "online_retail_ii", "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "host": "lokal", "compute_seconds": ev_res["seconds"], "main_metric": "net_value",
            "main_metric_label": "Netto-Wert der Aktionen je Quartal (€)", "lower_is_better": False,
            "timeline": {"test": ev_res["T"], "test_label_end": (pd.Timestamp(ev_res["T"]) + pd.Timedelta(days=89)).date().isoformat(),
                         "train": [], "val": None, "horizon_days": 90, "label": "Ausgang der gewählten Aktion in 90 Tagen"},
            "rule_text": "Inaktive (≥ 60 Tage) nach Umsatz anrufen, alle anderen bekommen den Topseller-Newsletter",
            "sizes": {"customers": ev_res["customers"], "capacity": ev_res["capacity"]}, "audit": [],
            "engines": engines, "best": "nba", "vorhersagekraft": vk,
            "recommendation": {"engine": "nba", "name": names["nba_model"][0], "klass": "Graph-ML light",
                               "text": ("Policy-Rückblick: " + vk.get("reason", "") + ". Gemessen ist die Treffsicherheit der Auswahl; "
                                        "die Wirkungsquoten bleiben Annahmen, belegt erst im Pilot mit Kontrollgruppe.")},
            "notes": ["Offline-Policy-Vergleich am Rückblick-Stichtag – keine Reaktionsdaten nötig, aber auch kein gelerntes NBA-Modell."],
            "top_list": [], "whatif": w, "policy_eval": ev_res}


def main() -> int:
    from . import evaluation as ev
    res = to_result(evaluate(log=lambda m: print(m, flush=True)))
    ev.save(res)
    print(json.dumps(res["vorhersagekraft"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
