"""
Wartung & Teile (Prompt 8) – Predictive Maintenance · Ersatzteilbedarf & Bestände · (First-Time-Fix: nur Struktur)
=================================================================================================================
  Predictive Maintenance  NASA C-MAPSS (Public Domain): fällt das Triebwerk in den nächsten 30 Zyklen aus?
                          Schnitte bis 10 je Triebwerk, Test auf 20 % der Triebwerke (keine Kalenderzeit).
                          Graph-Signal: Schnitt → Betriebsbedingung → andere Triebwerke (Sensoren relativ zum Normalwert der
                          Peers unter derselben Bedingung) und Schnitt → Flotte → Lebensdauer der Peer-Triebwerke.
  Ersatzteilbedarf        WideWorldImporters: Absatz je Lagerartikel im Folgemonat (Bestellungen), Test: die letzten
                          4 Monate. Graph-Signal: Artikel → Warengruppe → Geschwister-Artikel, Artikel → Kunden → deren
                          übrige Artikel (Co-Kauf). Dazu Bestandsampel für 06/2016 aus dem Endbestand.

  .venv/bin/python -m graph_ml.maintenance
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import connectors as cn
from . import quality as q

SEED = 7
H = 30                      # Zyklen
CUTS = 10
MIN_HIST = 10
FAIL_EUR = 50_000.0         # Annahme: Mehrkosten eines ungeplanten Ausfalls ggü. geplantem Tausch
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
FX = cn.WWI_USD_EUR


# ── Predictive Maintenance (C-MAPSS) ──────────────────────────────────────────

def pdm_rows(log=print):
    t = cn.load_tables("cmapss")
    c, e = t["cycles"].copy(), t["engines"].copy()
    c["engine_id"] = c["engine_id"].astype(str)
    e["engine_id"] = e["engine_id"].astype(str)
    c = c.merge(e[["engine_id", "fleet_id", "cycles_observed", "rul_at_end"]], on="engine_id")
    c["fleet_id"] = c["fleet_id"].astype(str)
    c = c.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    c["rul"] = c["rul_at_end"] + c["cycles_observed"] - c["cycle"]
    c["cond"] = (c["setting_1"].round(0).astype(int).astype(str) + "|" + (c["setting_2"] * 10).round(0).astype(int).astype(str)
                 + "|" + c["setting_3"].round(0).astype(int).astype(str))
    g = c.groupby("engine_id")
    extra = {}
    for s in SENSORS:                                            # Verlauf der eigenen Auslesungen (nur Vergangenheit)
        extra[f"m10_{s}"] = g[s].transform(lambda x: x.rolling(10, min_periods=3).mean())
        extra[f"d10_{s}"] = c[s] - g[s].shift(10)
    c = pd.concat([c, pd.DataFrame(extra)], axis=1)
    rng = np.random.default_rng(SEED)
    cand = c[c["cycle"] >= MIN_HIST].copy()
    cand["_r"] = rng.random(len(cand))
    rows = cand[cand.groupby("engine_id")["_r"].rank(method="first") <= CUTS].drop(columns="_r").reset_index(drop=True)
    rows["y"] = (rows["rul"] <= H).astype(int)
    eng = rows[["engine_id", "fleet_id"]].drop_duplicates()
    test_e = set(eng.groupby("fleet_id", group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=SEED), include_groups=False).index
                 .map(eng["engine_id"]))
    rows["test"] = rows["engine_id"].isin(test_e)
    log(f"[maint] C-MAPSS: {len(rows):,} Schnitte, {rows['engine_id'].nunique():,} Triebwerke, Positivrate {rows['y'].mean():.3f}")
    return c, rows


def pdm_features(c: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    f = {}
    f["cycle"] = rows["cycle"].to_numpy(float)
    for s in ("setting_1", "setting_2", "setting_3"):
        f[s] = rows[s].to_numpy(float)
    for s in SENSORS:
        f[s] = rows[s].to_numpy(float)
        f[f"m10_{s}"] = rows[f"m10_{s}"].to_numpy(float)
        f[f"d10_{s}"] = rows[f"d10_{s}"].to_numpy(float)
    # Graph 1: Schnitt → Betriebsbedingung → Peer-Triebwerke (nur Trainings-Triebwerke, frühe Lebensphase = Normalzustand)
    train_e = set(rows.loc[~rows["test"], "engine_id"])
    ref = c[c["engine_id"].isin(train_e) & (c["cycle"] <= 30)]
    mu, sd = ref.groupby("cond")[SENSORS].mean(), ref.groupby("cond")[SENSORS].std().replace(0, np.nan)
    for s in SENSORS:
        m, d = rows["cond"].map(mu[s]), rows["cond"].map(sd[s])
        f[f"g_z_{s}"] = ((rows[s] - m) / d).fillna(0).to_numpy()
        f[f"g_zm10_{s}"] = ((rows[f"m10_{s}"] - m) / d).fillna(0).to_numpy()
    # Graph 2: Schnitt → Flotte → Lebensdauer der Peer-Triebwerke (Run-to-failure im Training)
    life = c[c["engine_id"].isin(train_e) & (c["rul_at_end"] == 0)].groupby("engine_id").agg(fleet=("fleet_id", "first"), life=("cycle", "max"))
    haz, med = np.zeros(len(rows)), np.zeros(len(rows))
    for fl, L in life.groupby("fleet")["life"]:
        Ls = np.sort(L.to_numpy())
        m = (rows["fleet_id"] == fl).to_numpy()
        cy = rows.loc[m, "cycle"].to_numpy()
        alive = len(Ls) - np.searchsorted(Ls, cy, side="left")
        fail = np.searchsorted(Ls, cy + H, side="right") - np.searchsorted(Ls, cy, side="left")
        haz[m] = (fail + 1) / (alive + 5)
        med[m] = cy / np.median(Ls)
    f["g_fleet_hazard"], f["g_age_vs_fleet"] = haz, med
    return pd.DataFrame(f, index=rows.index)


PDM_BASE = None          # wird aus den Spalten bestimmt
PDM_LABELS = {"cycle": "Zyklen seit Inbetriebnahme", "g_fleet_hazard": "Graph: Ausfallquote der Flotten-Peers in diesem Alter",
              "g_age_vs_fleet": "Graph: Alter relativ zur Flotten-Lebensdauer"}


def _pdm_label(c: str) -> str:
    if c in PDM_LABELS:
        return PDM_LABELS[c]
    if c.startswith("g_zm10_"):
        return f"Graph: Ø Sensor {c[12:]} ggü. Peers gleicher Bedingung"
    if c.startswith("g_z_"):
        return f"Graph: Sensor {c[11:]} ggü. Peers gleicher Bedingung"
    if c.startswith("m10_"):
        return f"Ø Sensor {c[11:]} (10 Zyklen)"
    if c.startswith("d10_"):
        return f"Änderung Sensor {c[11:]} (10 Zyklen)"
    return c.replace("sensor_", "Sensor ").replace("setting_", "Einstellung ")


def predictive_maintenance(log=print) -> dict:
    c, rows = pdm_rows(log)
    f = pdm_features(c, rows)
    graph = [x for x in f.columns if x.startswith("g_")]
    base = [x for x in f.columns if not x.startswith("g_")]
    rng = np.random.default_rng(SEED)
    tr_e = rows.loc[~rows["test"], "engine_id"].unique()
    va_e = set(rng.choice(tr_e, len(tr_e) // 7, replace=False))
    test = rows["test"].to_numpy()
    va = (~test) & rows["engine_id"].isin(va_e).to_numpy()
    ec, ne = q._index(rows["engine_id"])
    cc, nc = q._index(rows["cond"])
    fc, nf = q._index(rows["fleet_id"])
    ix = np.arange(len(rows))
    links = {"triebwerk": (ix, ec, ne), "bedingung": (ix, cc, nc), "flotte": (ix, fc, nf)}
    names = {"rule": ("Regel: Triebwerke mit den meisten Zyklen zuerst", "Regel"),
             "lgbm": ("LightGBM (Sensoren + Verlauf)", "Baum"),
             "lgbm_graph": ("LightGBM + Peer-Normierung & Flotten-Lebensdauer (Graph)", "Graph-ML light"),
             "tabpfn": ("TabPFN v2", "TFM"), "gnn": ("GNN Schnitt ↔ Triebwerk ↔ Bedingung ↔ Flotte (GraphSAGE)", "Relational")}
    labels = {x: _pdm_label(x) for x in base + graph}
    y = rows["y"].to_numpy()
    value = y * FAIL_EUR
    res = q.evaluate_binary(f, y, value, pd.Series(pd.Timestamp("2000-01-01"), index=rows.index), pd.Timestamp("2000-01-01"),
                            pd.Series(pd.Timestamp("2000-01-01"), index=rows.index), base, graph, rows["cycle"].to_numpy(float),
                            links, names, labels, log, masks=(test, va))
    te = rows[test].assign(score=res["scores"][res["best"]])
    res["fleets"] = [{"fleet": fl, "cuts": int(len(g)), "observed": round(float(g["y"].mean()), 4),
                      "predicted": round(float(g["score"].mean()), 4)} for fl, g in te.groupby("fleet_id")]
    # Restlebensdauer-Ansicht: Trefferquote der Top-Liste nach wahrer RUL
    o = te.sort_values("score", ascending=False).head(max(1, len(te) // 10))
    res["rul_top"] = {"<=10": int((o["rul"] <= 10).sum()), "11-30": int(((o["rul"] > 10) & (o["rul"] <= H)).sum()),
                      "31-60": int(((o["rul"] > H) & (o["rul"] <= 60)).sum()), ">60": int((o["rul"] > 60).sum())}
    return res


# ── Ersatzteilbedarf & Bestände (WWI) ─────────────────────────────────────────

def demand(log=print) -> dict:
    t0 = time.time()
    t = cn.load_tables("wwi")
    ol = t["OrderLines"].merge(t["Orders"][["OrderID", "OrderDate", "CustomerID"]], on="OrderID")
    ol["m"] = pd.to_datetime(ol["OrderDate"]).dt.to_period("M").dt.start_time
    ol["Quantity"] = pd.to_numeric(ol["Quantity"].astype(str), errors="coerce").fillna(0)
    ol["UnitPrice"] = pd.to_numeric(ol["UnitPrice"].astype(str), errors="coerce").fillna(0)
    ms = sorted(ol["m"].unique())
    if ol[ol["m"] == ms[-1]]["OrderDate"].nunique() < 15:      # unvollständiger letzter Monat
        ms = ms[:-1]
    ol = ol[ol["m"].isin(ms)]
    items = sorted(ol["StockItemID"].unique())
    Q = ol.groupby(["StockItemID", "m"])["Quantity"].sum().unstack(fill_value=0).reindex(index=items, columns=ms, fill_value=0)
    price = ol.groupby("StockItemID")["UnitPrice"].median().reindex(items) * FX
    sg = t["StockItemStockGroups"]
    grp = sg.groupby("StockItemID")["StockGroupID"].apply(list).to_dict()
    # Co-Kauf: Artikel ↔ Artikel über gemeinsame Kunden (Jaccard, aus der Zeit vor dem Test)
    test_from = ms[-4]
    pre = ol[ol["m"] < test_from]
    ci = pre.groupby("StockItemID")["CustomerID"].apply(set).to_dict()
    nbrs = {}
    for a in items:
        sa = ci.get(a, set())
        sims = [(len(sa & ci.get(b, set())) / max(len(sa | ci.get(b, set())), 1), b) for b in items if b != a]
        nbrs[a] = [b for _, b in sorted(sims, reverse=True)[:5]]
    rows = []
    A = Q.to_numpy(float)
    gi = {g: [items.index(i) for i, gs in grp.items() if g in gs and i in items] for g in sg["StockGroupID"].unique()}
    for j in range(12, len(ms) - 1):
        for k, it in enumerate(items):
            h = A[k, : j + 1]
            sib = sorted({x for g in grp.get(it, []) for x in gi.get(g, []) if x != k})
            sh = A[sib, : j + 1] if sib else np.zeros((1, j + 1))
            nb = [items.index(b) for b in nbrs[it]]
            nh = A[nb, : j + 1]
            gr = lambda M: (M[:, -3:].sum() + 1) / (M[:, -6:-3].sum() + 1)  # noqa: E731
            rows.append({"item": it, "m": ms[j], "target": A[k, j + 1], "last1": h[-1], "mean3": h[-3:].mean(), "mean12": h[-12:].mean(),
                         "ly": h[-12], "moy": ms[j].month, "trend": (h[-3:].sum() + 1) / (h[-6:-3].sum() + 1), "price": price.iloc[k],
                         "g_sib_growth": gr(sh), "g_sib_last_rel": sh[:, -1].sum() / max(sh[:, -4:-1].sum() / 3, 1),
                         "g_nb_growth": gr(nh), "g_nb_last_rel": nh[:, -1].sum() / max(nh[:, -4:-1].sum() / 3, 1),
                         "g_n_groups": len(grp.get(it, []))})
    df = pd.DataFrame(rows)
    te, tr = df[df["m"] >= ms[-5]], df[df["m"] < ms[-5]]                # Ursprünge der letzten 4 Zielmonate
    va = tr[tr["m"] >= ms[-8]]
    a = tr[tr["m"] < ms[-8]]
    import lightgbm as lgb
    base = ["last1", "mean3", "mean12", "ly", "moy", "trend", "price"]
    graph = ["g_sib_growth", "g_sib_last_rel", "g_nb_growth", "g_nb_last_rel", "g_n_groups"]
    preds, imp = {"rule": te["mean3"].to_numpy(float)}, None
    den = lambda x: np.maximum(x["mean3"].to_numpy(float), 1.0)  # noqa: E731
    for eid, cols in (("lgbm", base), ("lgbm_graph", base + graph)):
        m = lgb.train(dict(objective="l1", learning_rate=0.03, num_leaves=7, min_data_in_leaf=20, feature_fraction=0.8, seed=SEED,
                           verbose=-1, num_threads=4), lgb.Dataset(a[cols], a["target"] / den(a), weight=den(a)), 1500,
                      valid_sets=[lgb.Dataset(va[cols], va["target"] / den(va), weight=den(va))],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        preds[eid] = np.clip(m.predict(te[cols], num_iteration=m.best_iteration) * den(te), 0, None)
        if eid == "lgbm_graph":
            imp = sorted([{"feature": c, "label": DEM_LABELS.get(c, c), "importance": round(float(i), 1)}
                          for c, i in zip(cols, m.feature_importance("gain"))], key=lambda r: -r["importance"])
    yv = te["target"].to_numpy(float)
    wv = te["price"].to_numpy(float)
    wape = {k: float(np.abs((v - yv) * wv).sum() / max((yv * wv).sum(), 1e-9)) for k, v in preds.items()}   # wertgewichtet
    best = min((k for k in wape if k != "rule"), key=wape.get)
    rng = np.random.default_rng(SEED)
    lifts, its = [], te["item"].to_numpy()
    for _ in range(300):
        s = np.isin(its, rng.choice(items, len(items)))
        r_, m_ = np.abs((preds["rule"][s] - yv[s]) * wv[s]).sum(), np.abs((preds[best][s] - yv[s]) * wv[s]).sum()
        if r_:
            lifts.append((r_ - m_) / r_)
    # Bestandsampel 06/2016: Endbestand gegen Prognose des Folgemonats (letzter Ursprung)
    hold = t["StockItemHoldings"].set_index("StockItemID")
    j = len(ms) - 1
    nxt = []
    for k, it in enumerate(items):
        h = A[k, : j + 1]
        nxt.append({"item": it, "fc_rule": h[-3:].mean()})
    nx = pd.DataFrame(nxt).set_index("item")
    names = t["StockItems"].set_index("StockItemID")["StockItemName"].astype(str)
    lead = t["StockItems"].set_index("StockItemID")["LeadTimeDays"]
    stock = []
    for it in items:
        oh = float(pd.to_numeric(pd.Series([hold.at[it, "QuantityOnHand"]]), errors="coerce").iloc[0]) if it in hold.index else np.nan
        fc = float(nx.at[it, "fc_rule"])
        cover = oh / fc * 30 if fc > 0 else np.inf
        stock.append({"item": names.get(it, str(it))[:48], "on_hand": oh, "forecast": round(fc, 0), "cover_days": round(cover, 0) if np.isfinite(cover) else None,
                      "lead_days": int(lead.get(it, 0)), "value_eur": round(oh * float(price.loc[it]), 0) if np.isfinite(oh) else None})
    st = pd.DataFrame(stock)
    short = st[(st["cover_days"].notna()) & (st["cover_days"] < st["lead_days"] + 7)].sort_values("cover_days")
    over = st[(st["cover_days"].notna()) & (st["cover_days"] > 365)].sort_values("value_eur", ascending=False)
    demand_q = float((te.groupby("m").apply(lambda g: (g["target"] * g["price"]).sum(), include_groups=False)).mean() * 3)
    log(f"[maint] WWI: {len(items)} Artikel · WAPE (wertgewichtet) Regel {wape['rule']:.3f} · LightGBM {wape['lgbm']:.3f} · "
        f"+Graph {wape['lgbm_graph']:.3f} · knapp {len(short)} · Überbestand {len(over)}")
    return {"wape": wape, "best": best, "lift": (wape["rule"] - wape[best]) / wape["rule"],
            "lift_ci": [round(float(np.percentile(lifts, 5)), 4), round(float(np.percentile(lifts, 95)), 4)] if lifts else None,
            "importance": imp, "demand_q_eur": round(demand_q, 2), "items": len(items),
            "test_months": [str(pd.Timestamp(m).date()) for m in ms[-4:]],
            "stock": {"as_of": str((pd.Timestamp(ms[-1]) + pd.offsets.MonthEnd(0)).date()), "short": short.head(8).to_dict("records"),
                      "over": over.head(8).to_dict("records"), "n_short": int(len(short)), "n_over": int(len(over))},
            "sizes": {"train_rows": int(len(tr)), "test_rows": int(len(te)), "items": len(items)}, "seconds": round(time.time() - t0, 1)}


DEM_LABELS = {"last1": "Menge Vormonat", "mean3": "Ø 3 Monate", "mean12": "Ø 12 Monate", "ly": "Vorjahresmonat", "moy": "Monat",
              "trend": "Trend 3 ggü. 3 Monate", "price": "Preis", "g_sib_growth": "Graph: Trend der Warengruppen-Geschwister",
              "g_sib_last_rel": "Graph: Vormonat der Geschwister ggü. Ø", "g_nb_growth": "Graph: Trend der Co-Kauf-Nachbarn",
              "g_nb_last_rel": "Graph: Vormonat der Co-Kauf-Nachbarn ggü. Ø", "g_n_groups": "Graph: Anzahl Warengruppen"}


# ── Artefakte ────────────────────────────────────────────────────────────────

def to_results(pdm: dict, dem: dict) -> dict[str, dict]:
    from . import evaluation as ev
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    common = {"result_version": ev.RESULT_VERSION, "variant": "v0", "currency": "EUR", "generated_at": now, "host": "lokal",
              "audit": [], "top_list": []}
    vk = ev.vorhersagekraft(pdm["lift"], pdm["lift_ci"])
    best = next(e for e in pdm["engines"] if e["id"] == pdm["best"])
    out = {"predictive_maintenance": {**common, "use_case": "predictive_maintenance", "kind": "binary", "dataset": "cmapss",
        "compute_seconds": pdm["seconds"], "main_metric": "auprc", "main_metric_label": "AUPRC", "lower_is_better": False,
        "timeline": {"test": "20 % der Triebwerke", "test_label_end": f"+{H} Zyklen", "train": [], "val": None, "horizon_days": H,
                     "label": f"Ausfall in den nächsten {H} Zyklen", "unit": "Triebwerk"},
        "rule_text": "Triebwerke mit den meisten Zyklen zuerst", "sizes": pdm["sizes"], "engines": pdm["engines"], "best": pdm["best"],
        "vorhersagekraft": vk,
        "whatif": {**pdm["whatif"], "unit": f"Ausfälle in {H} Zyklen (× {FAIL_EUR:,.0f} € Mehrkosten, Annahme)".replace(",", ".")},
        "recommendation": {"engine": pdm["best"], "name": best["meta"]["name"], "klass": best["meta"]["klass"],
                           "text": f"Bestes Modell: {best['meta']['name']} (AUPRC {best['metrics']['auprc']:.3f}".replace(".", ",")
                                   + f") – {vk.get('reason', '')}."},
        "notes": ["C-MAPSS ist eine NASA-Simulation (Public Domain) – Test auf anderen Triebwerken, keine Kalenderzeit.",
                  "Peer-Normalwerte nur aus Trainings-Triebwerken (erste 30 Zyklen = Normalzustand)."],
        "maintenance": {"fleets": pdm["fleets"], "rul_top": pdm["rul_top"], "fail_eur": FAIL_EUR}}}
    vk2 = ev.vorhersagekraft(dem["lift"], dem["lift_ci"])
    nm = {"rule": ("Regel: Ø der letzten 3 Monate", "Regel"), "lgbm": ("LightGBM (Verlauf)", "Baum"),
          "lgbm_graph": ("LightGBM + Warengruppe & Co-Kauf (Graph)", "Graph-ML light")}
    eng = [{"id": k, "status": "ok", "meta": {"name": nm[k][0], "klass": nm[k][1], "device": "CPU"}, "metrics": {"wape": round(v, 5)},
            **({"explain": {"global": dem["importance"]}} if k == "lgbm_graph" else {})} for k, v in dem["wape"].items()]
    out["spare_parts_demand"] = {**common, "use_case": "spare_parts_demand", "kind": "regression", "dataset": "wwi",
        "compute_seconds": dem["seconds"], "main_metric": "wape", "main_metric_label": "WAPE Bedarf Folgemonat (wertgewichtet)",
        "lower_is_better": True,
        "timeline": {"test": dem["test_months"][0], "test_label_end": dem["test_months"][-1], "train": [], "val": None, "horizon_days": 30,
                     "label": "Absatz je Lagerartikel im Folgemonat"},
        "rule_text": "Ø der letzten 3 Monate", "sizes": dem["sizes"], "engines": eng, "best": dem["best"], "vorhersagekraft": vk2,
        "whatif": {"kind": "forecast", "currency": "EUR", "demand_value": dem["demand_q_eur"],
                   "engines": {k: {"wape": v} for k, v in dem["wape"].items()}},
        "recommendation": {"engine": dem["best"], "name": nm[dem["best"]][0], "klass": nm[dem["best"]][1],
                           "text": f"Teilebedarf {vk2.get('reason', '')}. Bestandsampel: {dem['stock']['n_short']} Artikel knapp, "
                                   f"{dem['stock']['n_over']} mit mehr als einem Jahr Reichweite."},
        "notes": ["WideWorldImporters ist künstlich – die Methode ist übertragbar, Muster und Euro-Größenordnung nicht."],
        "maintenance": {"stock": dem["stock"], "items": dem["items"], "test_months": dem["test_months"]}}
    return out


def main() -> int:
    from . import evaluation as ev
    log = lambda m: print(m, flush=True)  # noqa: E731
    res = to_results(predictive_maintenance(log), demand(log))
    for r in res.values():
        ev.save(r)
        print(r["use_case"], r["vorhersagekraft"].get("reason"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
