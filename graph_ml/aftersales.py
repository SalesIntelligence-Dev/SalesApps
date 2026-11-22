"""
After Sales (Prompt 6) – Installierte Basis · Serviceverträge · Ersatzteilpreise
================================================================================
  Installierte Basis   SCANIA Component X (echte Lkw-Flotte, CC BY 4.0): Welches Fahrzeug braucht in den nächsten
                       48 Zeiteinheiten eine Reparatur der Komponente X? → proaktives Serviceangebot.
                       Schnitt je Fahrzeug an zufälligen Werkstatt-Auslesungen; Test auf anderen Fahrzeugen
                       (SCANIA hat keine Kalenderzeit, nur Zeit seit Studienbeginn je Fahrzeug).
                       Graph-Signal: Fahrzeug → Konfigurationswert → andere Fahrzeuge (Reparaturquote der Peers,
                       nur aus Trainingsfahrzeugen, außerhalb des eigenen Folds) und ein GNN über diesen Graphen.
  Serviceverträge      Keine Vertragsdaten → Abschluss/Verlängerung nicht messbar. Angeboten wird eine risikobasierte
                       Kalkulation je Konfiguration aus dem Reparaturmodell.
  Ersatzteilpreise     AdventureWorks: Komponenten, die auch einzeln verkauft werden. Graph-Pfad Teil → Stückliste →
                       verkaufte Fahrräder (installierte Basis). Prognose des Teilebedarfs im Folgequartal; Teile mit
                       starker Bindung an die installierte Basis haben Preisspielraum.

  .venv/bin/python -m graph_ml.aftersales
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import backtest as bt
from . import connectors as cn

SEED = 7
H = 48                         # Horizont in SCANIA-Zeiteinheiten
CUTS = 10                      # Schnitte je Fahrzeug
REPAIR_EUR = 2500.0            # Annahme: Umsatz je Reparaturauftrag Komponente X (Teile + Arbeit)
SPEC = [f"spec_{i}" for i in range(8)]


# ── Installierte Basis (SCANIA) ───────────────────────────────────────────────

def _scania_rows(log=print):
    t = cn.load_tables("scania")
    v = t["vehicles"].copy()
    for c in SPEC:
        v[c] = v[c].astype(str)
    z = np.load(cn.LOCAL_DIR / "scania_readouts.npz", allow_pickle=False)
    cols = [str(c) for c in z["columns"]]
    ro = pd.DataFrame(z["values"], columns=cols)
    ro["vehicle_id"] = ro["vehicle_id"].astype(np.int32)
    ro = ro.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)
    feats = [c for c in cols if c not in ("vehicle_id", "time_step")]
    d = ro.groupby("vehicle_id")[feats].diff()                      # Zuwachs seit der vorigen Auslesung
    ro = pd.concat([ro, pd.DataFrame(d.to_numpy(np.float32), columns=[f"d_{c}" for c in feats], index=ro.index)], axis=1)
    ro["n_read"] = ro.groupby("vehicle_id").cumcount() + 1
    ro = ro.merge(v[["vehicle_id", "study_length", "repaired"] + SPEC], on="vehicle_id", how="inner")
    ro = ro[ro["time_step"] < ro["study_length"]]
    rng = np.random.default_rng(SEED)
    ro["_r"] = rng.random(len(ro))                                    # je Fahrzeug bis zu CUTS zufällige Schnitte
    pick = ro[ro.groupby("vehicle_id")["_r"].rank(method="first") <= CUTS].drop(columns="_r")
    pick["y"] = ((pick["repaired"] == 1) & (pick["study_length"] - pick["time_step"] <= H)).astype(int)
    vids = np.sort(v["vehicle_id"].unique())
    rng.shuffle(vids)
    test_v = set(vids[: len(vids) // 5])
    pick["test"] = pick["vehicle_id"].isin(test_v)
    log(f"[aftersales] SCANIA: {len(pick):,} Schnitte, {pick['vehicle_id'].nunique():,} Fahrzeuge, "
        f"Positivrate {pick['y'].mean():.3f}, Test {len(test_v):,} Fahrzeuge")
    return pick.reset_index(drop=True), feats, v


def _peer_features(rows: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """Reparaturquote der Fahrzeuge mit demselben Konfigurationswert – nur Trainingsfahrzeuge, außerhalb des eigenen
    Folds (5 Folds nach Fahrzeug), geglättet. Dazu: Reparaturquote der identischen Konfiguration."""
    tv = v[v["vehicle_id"].isin(rows.loc[~rows["test"], "vehicle_id"])].copy()
    fold = pd.Series(np.random.default_rng(SEED).integers(0, 5, len(tv)), index=tv["vehicle_id"].to_numpy())
    k = 20.0                                                           # Glättung; Prior je Fold (sonst sieht er das eigene Label)
    out = pd.DataFrame(index=rows.index)
    rf = rows["vehicle_id"].map(fold)                                  # NaN für Testfahrzeuge
    tfold = fold.reindex(tv["vehicle_id"]).to_numpy()
    tv["combo"] = tv[SPEC].agg("|".join, axis=1)                       # ganze Konfiguration, ebenfalls fold-basiert
    key = rows[SPEC].agg("|".join, axis=1)
    cols = {f"g_rate_{c}": np.zeros(len(rows)) for c in SPEC}
    rate, cnt = np.zeros(len(rows)), np.zeros(len(rows))
    for f in list(range(5)) + [None]:
        ref = tv if f is None else tv[tfold != f]
        m = rows["test"].to_numpy() if f is None else (rf == f).to_numpy()
        prior = float(ref["repaired"].mean())
        for c in SPEC:
            st = ref.groupby(c)["repaired"].agg(["sum", "size"])
            cols[f"g_rate_{c}"][m] = rows.loc[m, c].map((st["sum"] + k * prior) / (st["size"] + k)).fillna(prior).to_numpy()
        st = ref.groupby("combo")["repaired"].agg(["sum", "size"])
        s_, n_ = key[m].map(st["sum"]).fillna(0).to_numpy(), key[m].map(st["size"]).fillna(0).to_numpy()
        rate[m], cnt[m] = (s_ + k * prior) / (n_ + k), n_ * (1.0 if f is None else 1.25)   # Fold sieht 4/5 der Peers
    for c, x in cols.items():
        out[c] = x
    out["g_rate_combo"] = rate
    out["g_n_combo"] = cnt
    return out


def _lgbm_fit(Xtr, ytr, Xva, yva, cats=()):
    import lightgbm as lgb
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
                       bagging_fraction=0.85, bagging_freq=1, seed=SEED, verbose=-1, num_threads=4),
                  lgb.Dataset(Xtr, ytr, categorical_feature=list(cats) or "auto"), 2000,
                  valid_sets=[lgb.Dataset(Xva, yva, categorical_feature=list(cats) or "auto")],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
    return m


def _gnn(rows, base_cols, dev_log):
    """Bipartiter Graph: Schnitt-Knoten (Fahrzeug zum Zeitpunkt) ↔ Konfigurationswert-Knoten (8 Kanten je Schnitt)."""
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import HeteroConv, SAGEConv
    except Exception as exc:
        return None, f"PyTorch Geometric nicht verfügbar ({type(exc).__name__})"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    X = rows[base_cols].astype(float).fillna(0).to_numpy()
    X = np.sign(X) * np.log1p(np.abs(X))
    mu, sd = X[~rows["test"].to_numpy()].mean(0), X[~rows["test"].to_numpy()].std(0) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    keys = pd.Index(sorted({f"{c}={v}" for c in SPEC for v in rows[c].unique()}))
    src, dst = [], []
    for c in SPEC:
        src.append(np.arange(len(rows)))
        dst.append(keys.get_indexer(c + "=" + rows[c].astype(str)))
    ei = torch.tensor(np.vstack([np.concatenate(src), np.concatenate(dst)]), device=dev)
    y = torch.tensor(rows["y"].to_numpy(np.float32), device=dev)
    test = rows["test"].to_numpy()
    rng = np.random.default_rng(SEED)
    va = (~test) & (rng.random(len(rows)) < 0.15)
    tr = (~test) & ~va
    Hd = 64

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.lin = nn.Linear(X.shape[1], Hd)
            s.spec = nn.Embedding(len(keys), Hd)
            s.c1 = HeteroConv({("v", "has", "s"): SAGEConv((Hd, Hd), Hd), ("s", "of", "v"): SAGEConv((Hd, Hd), Hd)}, aggr="mean")
            s.head = nn.Sequential(nn.Linear(2 * Hd, Hd), nn.ReLU(), nn.Dropout(0.2), nn.Linear(Hd, 1))

        def forward(s, x):
            hv = torch.relu(s.lin(x))
            h = s.c1({"v": hv, "s": s.spec.weight}, {("v", "has", "s"): ei, ("s", "of", "v"): ei.flip(0)})
            return s.head(torch.cat([hv, torch.relu(h["v"])], 1)).squeeze(-1)

    net = Net().to(dev)
    xt = torch.tensor(X, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    pw = torch.tensor(float((1 - rows["y"][tr].mean()) / max(rows["y"][tr].mean(), 1e-6)) ** 0.5, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    trm, vam = torch.tensor(tr, device=dev), torch.tensor(va, device=dev)
    best, state, pat = 1e9, None, 0
    for ep in range(200):
        net.train()
        opt.zero_grad()
        o = net(xt)
        lossf(o[trm], y[trm]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(xt)[vam], y[vam]))
        if vl < best - 1e-4:
            best, pat, state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            pat += 1
            if pat >= 20:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        p = torch.sigmoid(net(xt)).cpu().numpy()
    return p[test], f"bipartiter Graph Schnitt ↔ Konfigurationswert ({len(keys)} Werte), {ep + 1} Epochen, {dev.type}"


def installed_base(log=print) -> dict:
    t0 = time.time()
    rows, feats, v = _scania_rows(log)
    g = _peer_features(rows, v)
    rows = pd.concat([rows, g], axis=1)
    base_cols = ["time_step", "n_read"] + feats + [f"d_{c}" for c in feats]
    graph_cols = list(g.columns)
    test = rows["test"].to_numpy()
    rng = np.random.default_rng(SEED)
    trv = rows.loc[~test, "vehicle_id"].unique()
    va_v = set(rng.choice(trv, len(trv) // 7, replace=False))
    va = (~test) & rows["vehicle_id"].isin(va_v).to_numpy()
    tr = (~test) & ~va
    y, yt = rows["y"].to_numpy(), rows.loc[test, "y"].to_numpy()
    scores, notes = {"rule": rows.loc[test, "time_step"].to_numpy(float)}, {}
    for eid, cols in (("lgbm", base_cols), ("lgbm_graph", base_cols + graph_cols)):
        m = _lgbm_fit(rows.loc[tr, cols], y[tr], rows.loc[va, cols], y[va])
        scores[eid] = m.predict(rows.loc[test, cols], num_iteration=m.best_iteration)
        if eid == "lgbm_graph":
            imp = m.feature_importance("gain")
            notes["importance"] = sorted([{"feature": f, "label": _label(f), "importance": round(float(i), 1)}
                                          for f, i in zip(cols, imp)], key=lambda r: -r["importance"])[:12]
        log(f"[aftersales] {eid}: fertig")
    try:
        import torch
        from tabpfn import TabPFNClassifier
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from tabpfn.constants import ModelVersion
            tp = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=dev)
        except Exception:
            tp = TabPFNClassifier(device=dev)
        cols = base_cols + graph_cols
        idx = rng.choice(np.flatnonzero(tr), min(int(tr.sum()), 8000), replace=False)
        Xs = rows.loc[idx, cols].astype(float).fillna(0).to_numpy()
        top_var = np.argsort(-Xs.std(0))[:100]                       # TabPFN v2: höchstens ~500 Merkmale, hier 100
        tp.fit(Xs[:, top_var], y[idx])
        Xt = rows.loc[test, cols].astype(float).fillna(0).to_numpy()[:, top_var]
        scores["tabpfn"] = np.concatenate([tp.predict_proba(Xt[i:i + 4000])[:, 1] for i in range(0, len(Xt), 4000)])
        notes["tabpfn"] = f"Kontext 8.000 Schnitte, 100 Merkmale mit der größten Streuung, {dev}"
        log("[aftersales] tabpfn: fertig")
    except Exception as exc:
        notes["tabpfn"] = f"nicht verfügbar ({type(exc).__name__}: {str(exc)[:80]})"
    p, note = _gnn(rows, base_cols, log)
    if p is not None:
        scores["gnn"] = p
    notes["gnn"] = note
    names = {"rule": ("Regel: Fahrzeuge mit der längsten Laufzeit zuerst", "Regel", False),
             "lgbm": ("LightGBM (Auslesungen)", "Baum", True), "lgbm_graph": ("LightGBM + Konfigurations-Peers (Graph)", "Graph-ML light", True),
             "tabpfn": ("TabPFN v2", "TFM", True), "gnn": ("GNN Fahrzeug ↔ Konfiguration (GraphSAGE)", "Relational", True)}
    engines = []
    for eid, s in scores.items():
        r = {"id": eid, "status": "ok", "meta": {"name": names[eid][0], "klass": names[eid][1],
                                                  "device": "GPU" if eid in ("tabpfn", "gnn") else "CPU",
                                                  "notes": [notes[eid]] if eid in notes else []},
             "metrics": bt.binary_metrics(yt, s, proba=names[eid][2]), "curves": bt.curves(yt, s, proba=names[eid][2])}
        r["ci"] = bt.bootstrap_lift(yt, s, scores["rule"], "auprc") if eid != "rule" else {"lift_ci": None}
        if eid == "lgbm_graph":
            r["explain"] = {"global": notes["importance"]}
        engines.append(r)
        log(f"[aftersales] {eid:10s} AUPRC {r['metrics']['auprc']:.3f} · Recall@10 % {r['metrics']['recall_at_10']:.3f}")
    best = max((e for e in engines if e["id"] != "rule"), key=lambda e: e["metrics"]["auprc"])
    rule = next(e for e in engines if e["id"] == "rule")
    lift = (best["metrics"]["auprc"] - rule["metrics"]["auprc"]) / rule["metrics"]["auprc"]
    # Was-wäre-wenn (Listenlogik wie Churn): Treffer in der Top-k-%-Liste × Umsatz je Reparatur
    N, ks = len(yt), list(range(1, 31))
    ns = [max(1, int(round(N * k / 100))) for k in ks]
    wi = {}
    for e in engines:
        o = np.argsort(-scores[e["id"]], kind="stable")
        cy = np.cumsum(yt[o])
        wi[e["id"]] = {"tp": [int(cy[n - 1]) for n in ns], "value": [round(float(cy[n - 1] * REPAIR_EUR), 2) for n in ns]}
    whatif = {"kind": "list", "k": ks, "n": ns, "N": N, "positives": int(yt.sum()), "value_total": float(yt.sum() * REPAIR_EUR),
              "currency": "EUR", "unit": f"Reparaturaufträge Komponente X in {H} Zeiteinheiten (× {REPAIR_EUR:,.0f} € Annahme)".replace(",", "."),
              "period_days": None, "engines": wi}
    # Serviceverträge: risikobasierte Kalkulation je Konfigurationswert des wichtigsten Merkmals
    te = rows.loc[test].assign(p=scores[best["id"]])
    spec_imp = max(SPEC, key=lambda c: te.groupby(c)["y"].mean().std() if te[c].nunique() > 1 else 0)
    seg = te.groupby(spec_imp).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean")).reset_index()
    seg = seg[seg["n"] >= 30].sort_values("predicted", ascending=False)
    contract = [{"segment": f"{spec_imp} = {r[spec_imp]}", "n": int(r["n"]), "observed": round(float(r["observed"]), 4),
                 "predicted": round(float(r["predicted"]), 4),
                 "price": round(float(r["predicted"] * REPAIR_EUR * 1.2), 0)} for _, r in seg.head(8).iterrows()]
    return {"engines": engines, "best": best["id"], "lift": lift, "lift_ci": best["ci"]["lift_ci"], "whatif": whatif,
            "contract_segments": contract, "contract_spec": spec_imp,
            "sizes": {"train_rows": int(tr.sum()), "test_rows": int(test.sum()), "train_positive_rate": round(float(y[tr].mean()), 4),
                      "test_positive_rate": round(float(yt.mean()), 4), "vehicles_test": int(rows.loc[test, "vehicle_id"].nunique()),
                      "features": len(base_cols) + len(graph_cols), "graph_features": len(graph_cols)},
            "seconds": round(time.time() - t0, 1)}


def _label(f: str) -> str:
    if f.startswith("g_rate_spec"):
        return f"Graph: Reparaturquote der Peers mit gleichem Merkmal {f[-1]}"
    return {"g_rate_combo": "Graph: Reparaturquote gleicher Konfiguration", "g_n_combo": "Graph: Fahrzeuge gleicher Konfiguration",
            "time_step": "Laufzeit seit Studienbeginn", "n_read": "Anzahl Werkstatt-Auslesungen"}.get(
        f, f"Zuwachs Sensor {f[2:]}" if f.startswith("d_") else f"Sensor {f}")


# ── Ersatzteilpreise (AdventureWorks) ─────────────────────────────────────────

def spare_parts(log=print) -> dict:
    t0 = time.time()
    t = cn.load_tables("adventureworks_data")
    prod, sub, cat = t["Product"], t["ProductSubcategory"], t["ProductCategory"]
    pc = prod.set_index("ProductID")["ProductSubcategoryID"].map(sub.set_index("ProductSubcategoryID")["ProductCategoryID"]).map(
        cat.set_index("ProductCategoryID")["Name"])
    d = t["SalesOrderDetail"].merge(t["SalesOrderHeader"][["SalesOrderID", "OrderDate"]], on="SalesOrderID")
    d["q"] = pd.to_datetime(d["OrderDate"]).dt.to_period("Q").dt.start_time
    d["cat"] = d["ProductID"].map(pc)
    d["net"] = d["UnitPrice"] * (1 - d["UnitPriceDiscount"]) * d["OrderQty"] * cn.AW_USD_EUR
    comp = d[d["cat"] == "Components"]
    bikes = d[d["cat"] == "Bikes"]
    bom = t["BillOfMaterials"]
    link = bom[bom["ComponentID"].isin(comp["ProductID"].unique()) & bom["ProductAssemblyID"].isin(bikes["ProductID"].unique())]
    link = link.groupby(["ProductAssemblyID", "ComponentID"])["PerAssemblyQty"].max().reset_index()
    qs = sorted(d["q"].unique())
    qty = comp.groupby(["ProductID", "q"])["OrderQty"].sum().unstack(fill_value=0).reindex(columns=qs, fill_value=0)
    rev = comp.groupby(["ProductID", "q"])["net"].sum().unstack(fill_value=0).reindex(columns=qs, fill_value=0)
    bq = bikes.groupby(["ProductID", "q"])["OrderQty"].sum().unstack(fill_value=0).reindex(columns=qs, fill_value=0)
    # installierte Basis je Teil: über die Stückliste verbaute Stück in verkauften Rädern (Kante Teil → Rad → Kunden)
    L = link.pivot_table(index="ComponentID", columns="ProductAssemblyID", values="PerAssemblyQty", fill_value=0)
    L = L.reindex(index=qty.index, fill_value=0)
    B = bq.reindex(index=L.columns, fill_value=0)
    ib_q = pd.DataFrame(L.to_numpy() @ B.to_numpy(), index=qty.index, columns=qs)       # Teile in neu verkauften Rädern je Quartal
    rows = []
    for j in range(4, len(qs) - 1):
        Q, nxt = qs[j], qs[j + 1]
        for p in qty.index:
            h = qty.loc[p].to_numpy()[: j + 1]
            ib = ib_q.loc[p].to_numpy()[: j + 1]
            rows.append({"part": p, "q": Q, "target": float(qty.at[p, nxt]), "rev_next": float(rev.at[p, nxt]),
                         "last1": h[-1], "mean4": h[-4:].mean(), "ly": h[-4] if len(h) >= 4 else np.nan, "qoy": Q.quarter,
                         "g_ib_last": ib[-1], "g_ib_4q": ib[-4:].sum(), "g_ib_8q": ib[-8:].sum(),
                         "g_ib_growth": ib[-1] / max(ib[-5:-1].mean(), 1e-6), "g_linked_bikes": int((L.loc[p] > 0).sum())})
    df = pd.DataFrame(rows)
    test_q = qs[-3:-1]                                    # die letzten beiden vollständigen Ursprünge
    te, tr = df[df["q"].isin(test_q)], df[~df["q"].isin(test_q)]
    va_q = tr["q"].max()
    a, va = tr[tr["q"] < va_q], tr[tr["q"] == va_q]
    import lightgbm as lgb
    base = ["last1", "mean4", "ly", "qoy"]
    graph = ["g_ib_last", "g_ib_4q", "g_ib_8q", "g_ib_growth", "g_linked_bikes"]
    preds = {"rule": te["last1"].to_numpy(float)}
    imp = None
    # skalenfrei: Faktor gegenüber dem Vorquartal lernen (L1), sonst dominieren wenige Großteile
    def rel(x):
        r = x.copy()
        den = np.maximum(x["last1"], 1.0)
        for c in ("mean4", "ly", "g_ib_last", "g_ib_4q", "g_ib_8q"):
            r[c + "_r"] = x[c] / den
        r["d_ib"] = x["g_ib_last"] / np.maximum(x["g_ib_4q"] / 4, 1.0)
        return r
    a, va, te = rel(a), rel(va), rel(te)
    base = base + ["mean4_r", "ly_r"]
    graph = graph + ["g_ib_last_r", "g_ib_4q_r", "g_ib_8q_r", "d_ib"]
    for eid, cols in (("lgbm", base), ("lgbm_graph", base + graph)):
        den_a, den_v, den_t = (np.maximum(x["last1"], 1.0) for x in (a, va, te))
        m = lgb.train(dict(objective="l1", learning_rate=0.03, num_leaves=7, min_data_in_leaf=20, feature_fraction=0.8,
                           seed=SEED, verbose=-1, num_threads=4),
                      lgb.Dataset(a[cols], a["target"] / den_a, weight=den_a), 1000,
                      valid_sets=[lgb.Dataset(va[cols], va["target"] / den_v, weight=den_v)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        preds[eid] = np.clip(m.predict(te[cols], num_iteration=m.best_iteration) * den_t, 0, None)
        if eid == "lgbm_graph":
            gi = m.feature_importance("gain")
            imp = sorted([{"feature": f, "label": SP_LABELS.get(f, f), "importance": round(float(i), 1)} for f, i in zip(cols, gi)],
                         key=lambda r: -r["importance"])
    yv = te["target"].to_numpy(float)
    wape = {k: float(np.abs(v - yv).sum() / max(yv.sum(), 1e-9)) for k, v in preds.items()}
    best = min((k for k in wape if k != "rule"), key=wape.get)
    rngb = np.random.default_rng(SEED)
    lifts = []
    parts = te["part"].to_numpy()
    up = np.unique(parts)
    for _ in range(300):
        s = np.isin(parts, rngb.choice(up, len(up)))
        r_, m_ = np.abs(preds["rule"][s] - yv[s]).sum(), np.abs(preds[best][s] - yv[s]).sum()
        if r_:
            lifts.append((r_ - m_) / r_)
    # gebundene Teile: Anteil der installierten Basis am Bedarf hoch → Preisspielraum
    te = te.assign(pred=preds[best])
    with np.errstate(all="ignore"):
        corr = df.groupby("part").apply(lambda g: g["target"].corr(g["g_ib_4q"]) if g["g_ib_4q"].std() > 0 and g["target"].std() > 0 else np.nan,
                                           include_groups=False)
    captive = set(corr[corr >= 0.5].index)
    cap_rev = float(te.loc[te["part"].isin(captive), "rev_next"].sum())
    names = prod.set_index("ProductID")["Name"].astype(str)
    top = (te[te["part"].isin(captive)].groupby("part")["rev_next"].sum().sort_values(ascending=False).head(8))
    log(f"[aftersales] Ersatzteile: {len(qty)} Teile · WAPE Regel {wape['rule']:.3f} · LightGBM {wape['lgbm']:.3f} · "
        f"mit installierter Basis {wape['lgbm_graph']:.3f} · gebunden {len(captive)} Teile, {cap_rev:,.0f} € im Test")
    return {"wape": wape, "best": best, "lift": (wape["rule"] - wape[best]) / wape["rule"],
            "lift_ci": [round(float(np.percentile(lifts, 5)), 4), round(float(np.percentile(lifts, 95)), 4)] if lifts else None,
            "importance": imp, "captive_parts": len(captive), "parts": int(len(qty)), "captive_rev_test": round(cap_rev, 2),
            "test_quarters": [str(qs[qs.index(q) + 1].date()) for q in test_q],   # prognostizierte Quartale
            "linked_parts": int(link["ComponentID"].nunique()),
            "bom_edges": int(len(link)),
            "captive_top": [{"part": names.get(p, str(p)), "rev": round(float(v), 0), "corr": round(float(corr[p]), 2)} for p, v in top.items()],
            "sizes": {"train_rows": int(len(tr)), "test_rows": int(len(te)), "parts": int(len(qty))},
            "seconds": round(time.time() - t0, 1)}


SP_LABELS = {"last1": "Menge Vorquartal", "mean4": "Ø Menge 4 Quartale", "ly": "Menge Vorjahresquartal", "qoy": "Quartal",
             "g_ib_last": "Graph: Teil in neu verkauften Rädern (Vorquartal)", "g_ib_4q": "Graph: installierte Basis 4 Quartale (Stückliste)",
             "g_ib_8q": "Graph: installierte Basis 8 Quartale", "g_ib_growth": "Graph: Wachstum der installierten Basis",
             "g_linked_bikes": "Graph: Räder, in denen das Teil steckt", "mean4_r": "Ø 4 Quartale / Vorquartal",
             "ly_r": "Vorjahresquartal / Vorquartal", "g_ib_last_r": "Graph: neue installierte Basis / Vorquartal",
             "g_ib_4q_r": "Graph: installierte Basis 4 Q. / Vorquartal", "g_ib_8q_r": "Graph: installierte Basis 8 Q. / Vorquartal",
             "d_ib": "Graph: Beschleunigung der installierten Basis"}


# ── Artefakte im Format der Rückblicke ────────────────────────────────────────

def to_results(ib: dict, sp: dict) -> dict[str, dict]:
    from . import evaluation as ev
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    vk = ev.vorhersagekraft(ib["lift"], ib["lift_ci"])
    best = next(e for e in ib["engines"] if e["id"] == ib["best"])
    common = {"result_version": ev.RESULT_VERSION, "variant": "v0", "currency": "EUR", "generated_at": now, "host": "lokal",
              "audit": [], "top_list": []}
    out = {}
    out["installed_base"] = {**common, "use_case": "installed_base", "kind": "binary", "dataset": "scania",
        "compute_seconds": ib["seconds"], "main_metric": "auprc", "main_metric_label": "AUPRC", "lower_is_better": False,
        "timeline": {"test": "20 % der Fahrzeuge", "test_label_end": f"+{H} Zeiteinheiten", "train": [], "val": None, "horizon_days": H,
                     "label": f"Reparatur Komponente X in den nächsten {H} Zeiteinheiten", "unit": "Fahrzeug"},
        "rule_text": "Fahrzeuge mit der längsten Laufzeit zuerst prüfen", "sizes": ib["sizes"], "engines": ib["engines"],
        "best": ib["best"], "vorhersagekraft": vk, "whatif": ib["whatif"],
        "recommendation": {"engine": ib["best"], "name": best["meta"]["name"], "klass": best["meta"]["klass"],
                           "text": f"Bestes Modell: {best['meta']['name']} (AUPRC {best['metrics']['auprc']:.3f}".replace(".", ",")
                                   + f") – {vk.get('reason', '')}."},
        "notes": ["SCANIA hat keine Kalenderzeit – Test auf anderen Fahrzeugen statt nach Datum; Peer-Quoten nur aus Trainingsfahrzeugen."],
        "aftersales": {"contract_segments": ib["contract_segments"], "contract_spec": ib["contract_spec"], "repair_eur": REPAIR_EUR}}
    out["service_contracts"] = {**common, "use_case": "service_contracts", "kind": "check", "dataset": "scania",
        "compute_seconds": 0, "main_metric": "segments", "main_metric_label": "risikobasierte Vertragskalkulation", "lower_is_better": False,
        "timeline": out["installed_base"]["timeline"], "rule_text": "Pauschalpreis je Vertrag", "sizes": {},
        "engines": [{"id": "check", "status": "ok", "meta": {"name": "Kalkulation aus dem Reparaturmodell", "klass": "Regel", "device": "CPU"},
                     "metrics": {"segments": len(ib["contract_segments"])}}], "best": "check",
        "vorhersagekraft": {"status": "na", "reason": "Keine Vertragsdaten (Abschluss, Verlängerung, Kündigung) – nicht messbar; "
                                                      "Kalkulation je Konfiguration aus dem Reparaturmodell"},
        "recommendation": {"engine": "check", "name": "risikobasierte Kalkulation", "text": "Vertragsdaten nachliefern (PB-10), dann Abschluss "
                           "und Verlängerung wie Churn rechnen; bis dahin Preis je Konfiguration aus erwarteter Reparaturquote."},
        "aftersales": out["installed_base"]["aftersales"]}
    vk2 = ev.vorhersagekraft(sp["lift"], sp["lift_ci"])
    eng = [{"id": k, "status": "ok", "meta": {"name": {"rule": "Regel: Menge des Vorquartals", "lgbm": "LightGBM (Verlauf)",
                                                        "lgbm_graph": "LightGBM + installierte Basis (Stückliste)"}[k],
                                               "klass": {"rule": "Regel", "lgbm": "Baum", "lgbm_graph": "Graph-ML light"}[k], "device": "CPU"},
            "metrics": {"wape": round(v, 5)}, **({"explain": {"global": sp["importance"]}} if k == "lgbm_graph" else {})}
           for k, v in sp["wape"].items()]
    out["spare_parts_pricing"] = {**common, "use_case": "spare_parts_pricing", "kind": "regression", "dataset": "adventureworks_data",
        "compute_seconds": sp["seconds"], "main_metric": "wape", "main_metric_label": "WAPE Teilebedarf Folgequartal", "lower_is_better": True,
        "timeline": {"test": sp["test_quarters"][0], "test_label_end": sp["test_quarters"][-1], "train": [], "val": None, "horizon_days": 91,
                     "label": "Absatz je Komponente im Folgequartal"},
        "rule_text": "Menge des Vorquartals", "sizes": sp["sizes"], "engines": eng, "best": sp["best"], "vorhersagekraft": vk2,
        "recommendation": {"engine": sp["best"], "name": next(e["meta"]["name"] for e in eng if e["id"] == sp["best"]),
                           "text": f"Teilebedarf {vk2.get('reason', '')}. Preisspielraum bei {sp['captive_parts']} gebundenen Teilen "
                                   "(Bedarf folgt der installierten Basis)."},
        "whatif": {"kind": "uplift", "currency": "EUR", "engines": {k: {"rev": sp["captive_rev_test"], "n": sp["captive_parts"]} for k in sp["wape"]},
                   "unit": "Umsatz gebundener Teile im Testzeitraum"},
        "aftersales": {k: sp[k] for k in ("captive_parts", "parts", "captive_rev_test", "captive_top", "linked_parts", "bom_edges",
                                          "test_quarters")}}
    return out


def main() -> int:
    from . import evaluation as ev
    log = lambda m: print(m, flush=True)  # noqa: E731
    res = to_results(installed_base(log), spare_parts(log))
    for r in res.values():
        ev.save(r)
        print(r["use_case"], r["vorhersagekraft"].get("reason"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
