"""
Qualität & Regress (Prompt 7) – Qualitäts-Frühwarnung & Root Cause · Lieferantenregress · (Garantiebetrug: nur Struktur)
=====================================================================================================================
  Es gibt keinen offenen, relationalen Garantiedatensatz. AdventureWorks (künstlich) hat aber die Kette, über die sich
  Qualitätsfehler fortpflanzen: Lieferant → Bauteil (Wareneingang mit Beanstandungen) → Stückliste → Produkt →
  Fertigungsauftrag mit Ausschuss und Ausschussgrund.

  Lieferantenregress   je Bestellposition: wird beim Wareneingang beanstandet (RejectedQty > 0)? → gezielte Prüfung,
                       Beanstandung beim Lieferanten geltend machen. Test: Bestellungen ab 01.04.2025.
  Qualität / Root Cause je Fertigungsauftrag: entsteht Ausschuss (ScrappedQty > 0)? → Frühwarnung, Ursachen nach
                       Ausschussgrund, Produktgruppe und Lieferantenqualität der Bauteile. Test: Starts ab 01.01.2025.

  Leakage-Guard: Ein Ergebnis ist erst bekannt, wenn es feststeht – Beanstandung ab Fälligkeit (≈ Wareneingang), Ausschuss
  ab Auftragsende. Alle Quoten zählen nur Fälle, die vor dem Startzeitpunkt der abgefragten Zeile abgeschlossen waren.

  .venv/bin/python -m graph_ml.quality
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import backtest as bt
from . import connectors as cn

SEED = 7
K_SMOOTH = 10.0
SUP_TEST = pd.Timestamp("2025-04-01")
QUAL_TEST = pd.Timestamp("2025-01-01")
FX = cn.AW_USD_EUR


# ── Vergangenheits-Quoten (Leakage-Guard) ─────────────────────────────────────

def past_stats(key: pd.Series, avail: pd.Series, y: pd.Series, q_key: pd.Series, q_time: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Je Abfrage (q_key, q_time): Summe und Anzahl von y über alle Fälle desselben Schlüssels, die vor q_time feststanden."""
    ref = pd.DataFrame({"k": key.to_numpy(), "t": pd.to_datetime(avail).to_numpy(), "y": y.to_numpy(float)}).dropna(subset=["k"])
    ref = ref.sort_values("t")
    s_out, n_out = np.zeros(len(q_key)), np.zeros(len(q_key))
    qk, qt = q_key.to_numpy(), pd.to_datetime(q_time).to_numpy()
    groups = {k: (g["t"].to_numpy(), np.cumsum(g["y"].to_numpy())) for k, g in ref.groupby("k")}
    order = pd.Series(np.arange(len(qk))).groupby(qk)
    for k, idx in order.groups.items():
        if k not in groups:
            continue
        ts, cs = groups[k]
        idx = np.asarray(idx)
        pos = np.searchsorted(ts, qt[idx], side="left")          # nur strikt vorher abgeschlossene Fälle
        n_out[idx] = pos
        s_out[idx] = np.where(pos > 0, cs[np.maximum(pos - 1, 0)], 0.0)
    return s_out, n_out


def _rate(s, n, prior):
    return (s + K_SMOOTH * prior) / (n + K_SMOOTH)


# ── Daten ────────────────────────────────────────────────────────────────────

def _aw():
    t = cn.load_tables("adventureworks_data")
    prod = t["Product"].copy()
    prod["cat"] = prod["ProductSubcategoryID"].map(t["ProductSubcategory"].set_index("ProductSubcategoryID")["ProductCategoryID"])
    return t, prod


def supplier_lines(t, prod) -> pd.DataFrame:
    p = t["PurchaseOrderDetail"].merge(t["PurchaseOrderHeader"][["PurchaseOrderID", "VendorID", "OrderDate", "ShipMethodID"]],
                                       on="PurchaseOrderID")
    p["OrderDate"], p["DueDate"] = pd.to_datetime(p["OrderDate"]), pd.to_datetime(p["DueDate"])
    for c in ("OrderQty", "UnitPrice", "LineTotal", "ReceivedQty", "RejectedQty", "StockedQty"):   # teils als Text gespeichert
        p[c] = pd.to_numeric(p[c].astype(str), errors="coerce").fillna(0.0)
    p["y"] = (p["RejectedQty"] > 0).astype(int)
    p["rej_eur"] = p["RejectedQty"] * p["UnitPrice"] * FX
    v = t["Vendor"].set_index("BusinessEntityID")
    p["credit"] = p["VendorID"].map(v["CreditRating"])
    p["preferred"] = p["VendorID"].map(v["PreferredVendorStatus"]).astype(float)
    pv = t["ProductVendor"].set_index(["ProductID", "BusinessEntityID"])
    idx = pd.MultiIndex.from_arrays([p["ProductID"], p["VendorID"]])
    p["lead"] = pv["AverageLeadTime"].reindex(idx).to_numpy()
    p["std_price"] = pv["StandardPrice"].reindex(idx).to_numpy()
    p["price_dev"] = p["UnitPrice"] / p["std_price"] - 1
    pa = prod.set_index("ProductID")
    p["subcat"] = p["ProductID"].map(pa["ProductSubcategoryID"]).fillna(-1)
    p["make"] = p["ProductID"].map(pa["MakeFlag"]).astype(float)
    p["vp"] = p["VendorID"].astype(str) + "|" + p["ProductID"].astype(str)
    return p.sort_values("OrderDate").reset_index(drop=True)


def supplier_features(p: pd.DataFrame, bom: pd.DataFrame) -> pd.DataFrame:
    q_t, av = p["OrderDate"], p["DueDate"]
    prior = float(p.loc[p["DueDate"] < SUP_TEST, "y"].mean())
    f = pd.DataFrame(index=p.index)
    s, n = past_stats(p["VendorID"], av, p["y"], p["VendorID"], q_t)
    f["v_rate"], f["v_n"] = _rate(s, n, prior), n
    s2, n2 = past_stats(p["vp"], av, p["y"], p["vp"], q_t)
    f["vp_rate"], f["vp_n"] = _rate(s2, n2, prior), n2
    # Graph: dasselbe Bauteil bei anderen Lieferanten (Bauteil → andere Lieferanten)
    s3, n3 = past_stats(p["ProductID"], av, p["y"], p["ProductID"], q_t)
    f["g_prod_other_rate"], f["g_prod_other_n"] = _rate(s3 - s2, n3 - n2, prior), n3 - n2
    # Graph: Lieferant → seine anderen Bauteilgruppen bzw. Bauteilgruppe → andere Lieferanten
    vs = p["VendorID"].astype(str) + "|" + p["subcat"].astype(str)
    s4, n4 = past_stats(vs, av, p["y"], vs, q_t)
    s5, n5 = past_stats(p["subcat"], av, p["y"], p["subcat"], q_t)
    f["g_subcat_other_rate"] = _rate(s5 - s4, n5 - n4, prior)
    # Graph: Lieferanten-Peers – Lieferanten, die dieselben Bauteile liefern (Lieferant → Bauteil → anderer Lieferant)
    peers = p[["VendorID", "ProductID"]].drop_duplicates()
    pp = peers.merge(peers, on="ProductID", suffixes=("", "_peer"))
    pp = pp[pp["VendorID"] != pp["VendorID_peer"]][["VendorID", "VendorID_peer"]].drop_duplicates()
    f["g_n_peer_vendors"] = p["VendorID"].map(pp.groupby("VendorID").size()).fillna(0).to_numpy()
    vrate_now = pd.DataFrame({"v": p["VendorID"], "r": f["v_rate"], "t": q_t})
    peer_rate = np.full(len(p), prior)
    if len(pp):
        m = pp.groupby("VendorID")["VendorID_peer"].apply(list).to_dict()
        for d, g in vrate_now.groupby("t"):                     # je Bestelldatum: Peer-Quoten zum selben Zeitpunkt
            sv, nv = past_stats(p["VendorID"], av, p["y"], pd.Series(sum([m.get(v, []) for v in g["v"]], [])),
                                pd.Series([d] * sum(len(m.get(v, [])) for v in g["v"])))
            pos = 0
            for i, v in zip(g.index, g["v"]):
                k = len(m.get(v, []))
                if k:
                    peer_rate[i] = float(np.mean(_rate(sv[pos:pos + k], nv[pos:pos + k], prior)))
                pos += k
    f["g_peer_vendor_rate"] = peer_rate
    # Graph: Bauteil → Stückliste → Produkte (wie viele Räder/Baugruppen hängen an diesem Bauteil)
    f["g_bom_assemblies"] = p["ProductID"].map(bom.groupby("ComponentID")["ProductAssemblyID"].nunique()).fillna(0).to_numpy()
    for c in ("OrderQty", "UnitPrice", "LineTotal", "credit", "preferred", "lead", "price_dev", "make", "subcat", "ShipMethodID"):
        f[c] = p[c].astype(float).to_numpy()
    f["month"] = q_t.dt.month.to_numpy()
    f["due_days"] = (p["DueDate"] - q_t).dt.days.to_numpy()
    return f


SUP_BASE = ["OrderQty", "UnitPrice", "LineTotal", "credit", "preferred", "lead", "price_dev", "make", "subcat", "ShipMethodID",
            "month", "due_days", "v_rate", "v_n", "vp_rate", "vp_n"]
SUP_GRAPH = ["g_prod_other_rate", "g_prod_other_n", "g_subcat_other_rate", "g_n_peer_vendors", "g_peer_vendor_rate", "g_bom_assemblies"]


def work_orders(t, prod) -> pd.DataFrame:
    w = t["WorkOrder"].copy()
    for c in ("StartDate", "EndDate", "DueDate"):
        w[c] = pd.to_datetime(w[c])
    for c in ("OrderQty", "StockedQty", "ScrappedQty"):
        w[c] = pd.to_numeric(w[c].astype(str), errors="coerce").fillna(0.0)
    w["y"] = (w["ScrappedQty"] > 0).astype(int)
    pa = prod.set_index("ProductID")
    w["scrap_eur"] = w["ScrappedQty"] * w["ProductID"].map(pa["StandardCost"]) * FX
    for c in ("ListPrice", "StandardCost", "DaysToManufacture", "SafetyStockLevel"):
        w[c] = w["ProductID"].map(pa[c]).astype(float)
    w["subcat"] = w["ProductID"].map(pa["ProductSubcategoryID"]).fillna(-1)
    w["cat"] = w["ProductID"].map(pa["cat"]).fillna(-1)
    w["n_ops"] = w["WorkOrderID"].map(t["WorkOrderRouting"].groupby("WorkOrderID").size()).fillna(0)
    return w.sort_values("StartDate").reset_index(drop=True)


def quality_features(w: pd.DataFrame, p: pd.DataFrame, bom: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    q_t, av = w["StartDate"], w["EndDate"]
    prior = float(w.loc[w["EndDate"] < QUAL_TEST, "y"].mean())
    f = pd.DataFrame(index=w.index)
    s, n = past_stats(w["ProductID"], av, w["y"], w["ProductID"], q_t)
    f["p_rate"], f["p_n"] = _rate(s, n, prior), n
    # Graph: Geschwister-Produkte derselben Unterkategorie (ohne das Produkt selbst)
    s2, n2 = past_stats(w["subcat"], av, w["y"], w["subcat"], q_t)
    f["g_sibling_rate"] = _rate(s2 - s, n2 - n, prior)
    # Graph: Produkte, die dieselben Bauteile verbauen (Produkt → Bauteil → anderes Produkt)
    b = bom[["ProductAssemblyID", "ComponentID"]].dropna().drop_duplicates()
    share = b.merge(b, on="ComponentID", suffixes=("", "_o"))
    share = share[share["ProductAssemblyID"] != share["ProductAssemblyID_o"]][["ProductAssemblyID", "ProductAssemblyID_o"]].drop_duplicates()
    nb = share.groupby("ProductAssemblyID")["ProductAssemblyID_o"].apply(list).to_dict()
    # Graph: Lieferantenqualität der Bauteile (Lieferant → Bauteil → Stückliste → Produkt), zum Startzeitpunkt
    comp_v = pv[["ProductID", "BusinessEntityID"]].rename(columns={"ProductID": "ComponentID", "BusinessEntityID": "VendorID"})
    pc = b.merge(comp_v, on="ComponentID")
    vend_of = pc.groupby("ProductAssemblyID")["VendorID"].apply(lambda x: sorted(set(x))).to_dict()
    sup_prior = float(p["y"].mean())
    f["g_n_components"] = w["ProductID"].map(b.groupby("ProductAssemblyID").size()).fillna(0).to_numpy()
    f["g_n_comp_vendors"] = w["ProductID"].map({k: len(v) for k, v in vend_of.items()}).fillna(0).to_numpy()
    shared, vrate, vmax = np.full(len(w), prior), np.full(len(w), sup_prior), np.full(len(w), sup_prior)
    for d, g in w.groupby(q_t.dt.to_period("M")):               # Monatsraster – Quoten zum Monatsanfang (konservativ)
        d0 = d.start_time
        prods = g["ProductID"].unique()
        others = sorted({o for pr in prods for o in nb.get(pr, [])})
        if others:
            so, no = past_stats(w["ProductID"], av, w["y"], pd.Series(others), pd.Series([d0] * len(others)))
            ro = dict(zip(others, _rate(so, no, prior)))
        vends = sorted({v for pr in prods for v in vend_of.get(pr, [])})
        if vends:
            sv, nv = past_stats(p["VendorID"], p["DueDate"], p["y"], pd.Series(vends), pd.Series([d0] * len(vends)))
            rv = dict(zip(vends, _rate(sv, nv, sup_prior)))
        for pr in prods:
            m = (g["ProductID"] == pr).to_numpy()
            ix = g.index[m]
            if nb.get(pr):
                shared[ix] = np.mean([ro[o] for o in nb[pr]])
            if vend_of.get(pr):
                rr = [rv[v] for v in vend_of[pr]]
                vrate[ix], vmax[ix] = np.mean(rr), np.max(rr)
    f["g_shared_comp_rate"], f["g_comp_vendor_rate"], f["g_comp_vendor_max"] = shared, vrate, vmax
    for c in ("OrderQty", "ListPrice", "StandardCost", "DaysToManufacture", "SafetyStockLevel", "subcat", "cat", "n_ops"):
        f[c] = w[c].astype(float).to_numpy()
    f["log_qty"] = np.log1p(w["OrderQty"].to_numpy(float))
    f["due_days"] = (w["DueDate"] - q_t).dt.days.to_numpy()
    f["month"] = q_t.dt.month.to_numpy()
    return f


QUAL_BASE = ["OrderQty", "log_qty", "ListPrice", "StandardCost", "DaysToManufacture", "SafetyStockLevel", "subcat", "cat", "n_ops",
             "due_days", "month", "p_rate", "p_n"]
QUAL_GRAPH = ["g_sibling_rate", "g_shared_comp_rate", "g_comp_vendor_rate", "g_comp_vendor_max", "g_n_components", "g_n_comp_vendors"]


# ── Engines ──────────────────────────────────────────────────────────────────

def _lgbm(Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    m = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8,
                       bagging_fraction=0.85, bagging_freq=1, seed=SEED, verbose=-1, num_threads=4),
                  lgb.Dataset(Xtr, ytr), 2000, valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def _tabpfn(X, y, tr, test, log):
    try:
        import torch
        from tabpfn import TabPFNClassifier
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from tabpfn.constants import ModelVersion
            tp = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=dev)
        except Exception:
            tp = TabPFNClassifier(device=dev)
        rng = np.random.default_rng(SEED)
        idx = np.flatnonzero(tr)
        if len(idx) > 10000:
            idx = rng.choice(idx, 10000, replace=False)
        tp.fit(X[idx], y[idx])
        Xt = X[test]
        p = np.concatenate([tp.predict_proba(Xt[i:i + 4000])[:, 1] for i in range(0, len(Xt), 4000)])
        return p, f"Kontext {len(idx):,} Fälle, {dev}".replace(",", ".")
    except Exception as exc:
        return None, f"nicht verfügbar ({type(exc).__name__}: {str(exc)[:80]})"


def _gnn(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, test: np.ndarray, links: dict[str, tuple]) -> tuple:
    """Heterogener Graph: Fall-Knoten (Merkmale ohne Label) ↔ Entitäten (Lieferant, Bauteil, Produkt …) mit gelernten
    Einbettungen; zwei Runden GraphSAGE. links: name → (Fall-Index, Entitäts-Index, Anzahl Entitäten)."""
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import HeteroConv, SAGEConv
    except Exception as exc:
        return None, f"PyTorch Geometric nicht verfügbar ({type(exc).__name__})"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    Xs = np.sign(X) * np.log1p(np.abs(np.nan_to_num(X)))
    mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
    xt = torch.tensor(((Xs - mu) / sd).astype(np.float32), device=dev)
    Hd = 48
    eis = {k: torch.tensor(np.vstack([a, b]).astype(np.int64), device=dev) for k, (a, b, _) in links.items()}

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.lin = nn.Linear(X.shape[1], Hd)
            s.emb = nn.ModuleDict({k: nn.Embedding(n, Hd) for k, (_, _, n) in links.items()})
            mk = lambda: HeteroConv({**{("c", f"to_{k}", k): SAGEConv((Hd, Hd), Hd) for k in links},  # noqa: E731
                                     **{(k, f"of_{k}", "c"): SAGEConv((Hd, Hd), Hd) for k in links}}, aggr="mean")
            s.c1, s.c2 = mk(), mk()
            s.head = nn.Sequential(nn.Linear(2 * Hd, Hd), nn.ReLU(), nn.Dropout(0.2), nn.Linear(Hd, 1))

        def forward(s, x):
            h0 = torch.relu(s.lin(x))
            h = {"c": h0, **{k: s.emb[k].weight for k in links}}
            ed = {**{("c", f"to_{k}", k): eis[k] for k in links}, **{(k, f"of_{k}", "c"): eis[k].flip(0) for k in links}}
            h1 = {k: torch.relu(v) for k, v in s.c1(h, ed).items()}
            h1 = {**h, **h1}
            h2 = s.c2(h1, ed)
            return s.head(torch.cat([h0, torch.relu(h2["c"])], 1)).squeeze(-1)

    net = Net().to(dev)
    yt = torch.tensor(y.astype(np.float32), device=dev)
    trm, vam = torch.tensor(tr, device=dev), torch.tensor(va, device=dev)
    pos = max(float(y[tr].mean()), 1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(((1 - pos) / pos) ** 0.5, device=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best, state, pat, ep = 1e9, None, 0, 0
    for ep in range(300):
        net.train()
        opt.zero_grad()
        lossf(net(xt)[trm], yt[trm]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(xt)[vam], yt[vam]))
        if vl < best - 1e-4:
            best, pat, state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            pat += 1
            if pat >= 25:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        p = torch.sigmoid(net(xt)).cpu().numpy()
    return p[test], f"Fälle ↔ {' / '.join(links)} (GraphSAGE, 2 Runden), {ep + 1} Epochen, {dev.type}"


def _index(s: pd.Series) -> tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(s)
    return codes, len(uniq)


def evaluate_binary(f: pd.DataFrame, y: np.ndarray, value: np.ndarray, t_query: pd.Series, test_from: pd.Timestamp,
                    avail: pd.Series, base: list[str], graph: list[str], rule: np.ndarray, links: dict, names: dict,
                    labels: dict, log=print, masks: tuple | None = None) -> dict:
    """masks = (test, val) für Daten ohne Kalenderzeit (Split nach Einheiten); sonst zeitlicher Split ab test_from."""
    t0 = time.time()
    if masks is not None:
        test, va = masks
        trall = ~test
        tr = trall & ~va
        cut = None
    else:
        test = (t_query >= test_from).to_numpy()
        known = (avail < test_from).to_numpy()                   # Training nur mit Labels, die am Stichtag feststanden
        trall = (~test) & known
        tq = t_query[trall]
        cut = tq.quantile(0.82)
        va = trall & (t_query >= cut).to_numpy()
        tr = trall & ~va
    yt = y[test]
    scores, notes = {"rule": rule[test]}, {}
    imp = None
    for eid, cols in (("lgbm", base), ("lgbm_graph", base + graph)):
        m = _lgbm(f.loc[tr, cols], y[tr], f.loc[va, cols], y[va])
        scores[eid] = m.predict(f.loc[test, cols], num_iteration=m.best_iteration)
        if eid == "lgbm_graph":
            imp = sorted([{"feature": c, "label": labels.get(c, c), "importance": round(float(i), 1)}
                          for c, i in zip(cols, m.feature_importance("gain"))], key=lambda r: -r["importance"])[:12]
    X = f[base + graph].astype(float).fillna(0).to_numpy()
    p, notes["tabpfn"] = _tabpfn(X, y, tr | va, test, log)
    if p is not None:
        scores["tabpfn"] = p
    p, notes["gnn"] = _gnn(f[base].astype(float).fillna(0).to_numpy(), y, tr, va, test, links)
    if p is not None:
        scores["gnn"] = p
    engines = []
    vt = value[test]
    for eid, s in scores.items():
        proba = eid != "rule"
        r = {"id": eid, "status": "ok", "meta": {"name": names[eid][0], "klass": names[eid][1],
                                                  "device": "GPU" if eid in ("tabpfn", "gnn") else "CPU",
                                                  "notes": [notes[eid]] if eid in notes else []},
             "metrics": bt.binary_metrics(yt, s, value=vt, proba=proba), "curves": bt.curves(yt, s, proba=proba, value=vt)}
        r["ci"] = bt.bootstrap_lift(yt, s, scores["rule"], "auprc") if eid != "rule" else {"lift_ci": None}
        if eid == "lgbm_graph":
            r["explain"] = {"global": imp}
        engines.append(r)
        log(f"[quality] {eid:10s} AUPRC {r['metrics']['auprc']:.3f} · Recall@10 % {r['metrics']['recall_at_10']:.3f}")
    best = max((e for e in engines if e["id"] != "rule"), key=lambda e: e["metrics"]["auprc"])
    rl = next(e for e in engines if e["id"] == "rule")
    lift = (best["metrics"]["auprc"] - rl["metrics"]["auprc"]) / max(rl["metrics"]["auprc"], 1e-9)
    N, ks = len(yt), list(range(1, 31))
    ns = [max(1, int(round(N * k / 100))) for k in ks]
    wi = {}
    for e in engines:
        o = np.argsort(-scores[e["id"]], kind="stable")
        cy, cv = np.cumsum(yt[o]), np.cumsum(vt[o])
        wi[e["id"]] = {"tp": [int(cy[n - 1]) for n in ns], "value": [round(float(cv[n - 1]), 2) for n in ns]}
    whatif = {"kind": "list", "k": ks, "n": ns, "N": N, "positives": int(yt.sum()), "value_total": round(float(vt.sum()), 2),
              "currency": "EUR", "period_days": None, "engines": wi}
    sizes = {"train_rows": int((tr | va).sum()), "test_rows": int(test.sum()), "train_positive_rate": round(float(y[tr | va].mean()), 4),
             "test_positive_rate": round(float(yt.mean()), 4), "features": len(base) + len(graph), "graph_features": len(graph)}
    return {"engines": engines, "best": best["id"], "lift": lift, "lift_ci": best["ci"]["lift_ci"], "whatif": whatif, "sizes": sizes,
            "scores": scores, "test": test, "seconds": round(time.time() - t0, 1),
            "train_dates": [] if cut is None else [str(d.date()) for d in pd.date_range(t_query[trall].min(), cut, freq="QS")][:12],
            "val": None if cut is None else str(cut.date())}


SUP_LABELS = {"v_rate": "Beanstandungsquote des Lieferanten (bisher)", "v_n": "bisherige Positionen des Lieferanten",
              "vp_rate": "Quote Lieferant × Bauteil (bisher)", "vp_n": "bisherige Positionen Lieferant × Bauteil",
              "g_prod_other_rate": "Graph: dasselbe Bauteil bei anderen Lieferanten", "g_prod_other_n": "Graph: Positionen anderer Lieferanten",
              "g_subcat_other_rate": "Graph: Bauteilgruppe bei anderen Lieferanten", "g_n_peer_vendors": "Graph: Lieferanten mit gleichen Bauteilen",
              "g_peer_vendor_rate": "Graph: Quote der Lieferanten-Peers", "g_bom_assemblies": "Graph: Baugruppen mit diesem Bauteil",
              "OrderQty": "Bestellmenge", "UnitPrice": "Stückpreis", "LineTotal": "Positionswert", "credit": "Bonität Lieferant",
              "preferred": "Vorzugslieferant", "lead": "Ø Lieferzeit", "price_dev": "Abweichung vom Standardpreis", "make": "Eigenfertigung",
              "subcat": "Bauteilgruppe", "ShipMethodID": "Versandart", "month": "Monat", "due_days": "Tage bis Fälligkeit"}
QUAL_LABELS = {"p_rate": "Ausschussquote des Produkts (bisher)", "p_n": "bisherige Aufträge des Produkts",
               "g_sibling_rate": "Graph: Geschwister-Produkte (Unterkategorie)", "g_shared_comp_rate": "Graph: Produkte mit gleichen Bauteilen",
               "g_comp_vendor_rate": "Graph: Ø Lieferantenqualität der Bauteile", "g_comp_vendor_max": "Graph: schlechtester Bauteil-Lieferant",
               "g_n_components": "Graph: Bauteile laut Stückliste", "g_n_comp_vendors": "Graph: Lieferanten der Bauteile",
               "OrderQty": "Auftragsmenge", "log_qty": "Auftragsmenge (log)", "ListPrice": "Listenpreis", "StandardCost": "Standardkosten",
               "DaysToManufacture": "Fertigungstage", "SafetyStockLevel": "Sicherheitsbestand", "subcat": "Unterkategorie",
               "cat": "Kategorie", "n_ops": "Arbeitsgänge laut Plan", "due_days": "Tage bis Fälligkeit", "month": "Monat"}


# ── Use Cases ─────────────────────────────────────────────────────────────────

def supplier_recourse(t, prod, log=print) -> dict:
    p = supplier_lines(t, prod)
    f = supplier_features(p, t["BillOfMaterials"])
    log(f"[quality] Lieferanten: {len(p):,} Bestellpositionen, {p['VendorID'].nunique()} Lieferanten, Quote {p['y'].mean():.3f}")
    vc, nv = _index(p["VendorID"])
    pc, npr = _index(p["ProductID"])
    links = {"lieferant": (np.arange(len(p)), vc, nv), "bauteil": (np.arange(len(p)), pc, npr)}
    names = {"rule": ("Regel: Lieferanten mit der höchsten bisherigen Beanstandungsquote zuerst", "Regel"),
             "lgbm": ("LightGBM (Position + eigene Lieferantenhistorie)", "Baum"),
             "lgbm_graph": ("LightGBM + Bauteil-/Lieferanten-Peers (Graph)", "Graph-ML light"),
             "tabpfn": ("TabPFN v2", "TFM"), "gnn": ("GNN Position ↔ Lieferant ↔ Bauteil (GraphSAGE)", "Relational")}
    rule = f["v_rate"].to_numpy() + 1e-6 * f["LineTotal"].to_numpy() / (f["LineTotal"].max() + 1)
    res = evaluate_binary(f, p["y"].to_numpy(), p["rej_eur"].to_numpy(), p["OrderDate"], SUP_TEST, p["DueDate"],
                          SUP_BASE, SUP_GRAPH, rule, links, names, SUP_LABELS, log)
    te = p[res["test"]].assign(score=res["scores"][res["best"]])
    vn = t["Vendor"].set_index("BusinessEntityID")["Name"]
    g = te.groupby("VendorID").agg(lines=("y", "size"), predicted=("score", "mean"), observed=("y", "mean"),
                                   rejected_eur=("rej_eur", "sum"), value=("LineTotal", "sum")).reset_index()
    g = g[g["lines"] >= 5].sort_values("predicted", ascending=False)
    res["vendors"] = [{"vendor": str(vn.get(r.VendorID, r.VendorID)), "lines": int(r.lines), "predicted": round(float(r.predicted), 4),
                       "observed": round(float(r.observed), 4), "rejected_eur": round(float(r.rejected_eur), 0)}
                      for r in g.head(10).itertuples()]
    res["recourse"] = {"rejected_eur_test": round(float(te["rej_eur"].sum()), 0), "lines_rejected": int(te["y"].sum()),
                       "vendors": int(te["VendorID"].nunique()),
                       "top10_share": round(float(g.sort_values("rejected_eur", ascending=False).head(10)["rejected_eur"].sum()
                                                  / max(te["rej_eur"].sum(), 1e-9)), 3)}
    return res


def quality_root_cause(t, prod, log=print) -> dict:
    p = supplier_lines(t, prod)
    w = work_orders(t, prod)
    f = quality_features(w, p, t["BillOfMaterials"], t["ProductVendor"])
    log(f"[quality] Fertigung: {len(w):,} Aufträge, Ausschussquote {w['y'].mean():.4f}")
    pc, npr = _index(w["ProductID"])
    sc, nsc = _index(w["subcat"])
    links = {"produkt": (np.arange(len(w)), pc, npr), "unterkategorie": (np.arange(len(w)), sc, nsc)}
    names = {"rule": ("Regel: größte Auftragsmenge zuerst", "Regel"), "lgbm": ("LightGBM (Auftrag + Produkthistorie)", "Baum"),
             "lgbm_graph": ("LightGBM + Stückliste/Lieferanten (Graph)", "Graph-ML light"), "tabpfn": ("TabPFN v2", "TFM"),
             "gnn": ("GNN Auftrag ↔ Produkt ↔ Unterkategorie (GraphSAGE)", "Relational")}
    res = evaluate_binary(f, w["y"].to_numpy(), w["scrap_eur"].to_numpy(), w["StartDate"], QUAL_TEST, w["EndDate"],
                          QUAL_BASE, QUAL_GRAPH, w["OrderQty"].to_numpy(float), links, names, QUAL_LABELS, log)
    # Root Cause: Ausschussgründe × Unterkategorie im Testzeitraum, Kohorten mit erhöhter Quote
    te = w[res["test"]].assign(score=res["scores"][res["best"]])
    rs = t["ScrapReason"].set_index("ScrapReasonID")["Name"]
    sub = t["ProductSubcategory"].set_index("ProductSubcategoryID")["Name"]
    pos = te[te["y"] == 1]
    res["reasons"] = [{"reason": str(rs.get(k, k)), "orders": int(v), "eur": round(float(pos.loc[pos["ScrapReasonID"] == k, "scrap_eur"].sum()), 0)}
                      for k, v in pos["ScrapReasonID"].value_counts().head(8).items()]
    g = te.groupby("subcat").agg(orders=("y", "size"), observed=("y", "mean"), predicted=("score", "mean"),
                                 eur=("scrap_eur", "sum")).reset_index()
    g = g[g["orders"] >= 30].sort_values("observed", ascending=False)
    res["cohorts"] = [{"cohort": str(sub.get(r.subcat, "ohne Kategorie")), "orders": int(r.orders), "observed": round(float(r.observed), 4),
                       "predicted": round(float(r.predicted), 4), "eur": round(float(r.eur), 0)} for r in g.head(8).itertuples()]
    # Fortpflanzung messbar? Ausschussquote nach Lieferantenqualität der Bauteile (Terzile)
    q = pd.qcut(f.loc[res["test"], "g_comp_vendor_rate"].rank(method="first"), 3, labels=["gut", "mittel", "schwach"])
    res["propagation"] = {k: round(float(v), 4) for k, v in te.groupby(q.to_numpy())["y"].mean().items()}
    res["root"] = {"orders_scrap": int(te["y"].sum()), "scrap_eur_test": round(float(te["scrap_eur"].sum()), 0),
                   "orders": int(len(te)), "qty_threshold": int(w.loc[w["y"] == 1, "OrderQty"].min())}
    return res


# ── Artefakte ────────────────────────────────────────────────────────────────

def _artifact(uc: str, r: dict, label: str, rule_text: str, test_from: pd.Timestamp, test_end: str, unit: str, extra: dict) -> dict:
    from . import evaluation as ev
    vk = ev.vorhersagekraft(r["lift"], r["lift_ci"])
    best = next(e for e in r["engines"] if e["id"] == r["best"])
    return {"result_version": ev.RESULT_VERSION, "variant": "v0", "currency": "EUR", "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "host": "lokal", "audit": [], "top_list": [], "use_case": uc, "kind": "binary", "dataset": "adventureworks_data",
            "compute_seconds": r["seconds"], "main_metric": "auprc", "main_metric_label": "AUPRC", "lower_is_better": False,
            "timeline": {"train": r["train_dates"], "val": r["val"], "test": str(test_from.date()), "test_label_end": test_end,
                         "horizon_days": 0, "label": label, "unit": unit},
            "rule_text": rule_text, "sizes": r["sizes"], "engines": r["engines"], "best": r["best"], "vorhersagekraft": vk,
            "whatif": {**r["whatif"], "unit": unit},
            "recommendation": {"engine": r["best"], "name": best["meta"]["name"], "klass": best["meta"]["klass"],
                               "text": f"Bestes Modell: {best['meta']['name']} (AUPRC {best['metrics']['auprc']:.3f}".replace(".", ",")
                                       + f") – {vk.get('reason', '')}."},
            "notes": ["AdventureWorks ist künstlich – die Methode ist übertragbar, Muster und Euro-Größenordnung nicht.",
                      "Leakage-Guard: Quoten zählen nur Fälle, die vor dem Startzeitpunkt abgeschlossen waren."],
            "quality": extra}


def to_results(sup: dict, qual: dict) -> dict[str, dict]:
    out = {"supplier_recourse": _artifact(
        "supplier_recourse", sup, "Bestellposition wird beim Wareneingang beanstandet (RejectedQty > 0)",
        "höchste bisherige Beanstandungsquote des Lieferanten zuerst", SUP_TEST, "2025-09-21", "Bestellposition",
        {"vendors": sup["vendors"], "recourse": sup["recourse"]})}
    out["quality_root_cause"] = _artifact(
        "quality_root_cause", qual, "Fertigungsauftrag mit Ausschuss (ScrappedQty > 0)", "größte Auftragsmenge zuerst",
        QUAL_TEST, "2025-06-01", "Fertigungsauftrag",
        {"reasons": qual["reasons"], "cohorts": qual["cohorts"], "propagation": qual["propagation"], "root": qual["root"]})
    return out


def main() -> int:
    from . import evaluation as ev
    log = lambda m: print(m, flush=True)  # noqa: E731
    t, prod = _aw()
    res = to_results(supplier_recourse(t, prod, log), quality_root_cause(t, prod, log))
    for r in res.values():
        ev.save(r)
        print(r["use_case"], r["vorhersagekraft"].get("reason"))
    print("Fortpflanzung Lieferant → Ausschuss:", res["quality_root_cause"]["quality"]["propagation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
