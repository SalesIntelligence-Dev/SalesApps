"""
Pricing (Prompt 5) – Preis- und Rabattempfehlung · Preis-Leakage · Angebotsprüfung (AdventureWorks)
====================================================================================================
Händlerpositionen mit Listenpreis und Standardkosten zum Auftragsdatum (Preis- und Kostenhistorie).

  Preis- & Rabattempfehlung  Welcher Nettopreis (in % vom Listenpreis) ist für diesen Händler, Artikel und diese Menge
                             typisch erzielbar? Merkmale nur aus dem, was beim Angebot bekannt ist – der gewährte Rabatt
                             ist die Entscheidung und deshalb kein Merkmal. Graph: Preise der Peers (gleicher Artikel im
                             Gebiet, eigene Historie des Händlers, Portfolio des Vertrieblers).
  Preis-Leakage              Positionen, deren Nettopreis mehr als 3 Prozentpunkte unter dem erwartbaren liegt.
  Angebotsprüfung            Marge je Position und Auftrag mit den Kosten zum Datum, Stücklisten-Tiefe als Komplexität –
                             eine Regelprüfung, kein Modell nötig.

Zeitlich getrennt: Trainingsquartale ab 01.2023, Test 01.01.–29.06.2025; Merkmale eines Quartals nur aus der Zeit davor.

  .venv/bin/python -m graph_ml.pricing
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import connectors as cn

T_TRAIN = pd.Timestamp("2023-01-01")
T_TEST = pd.Timestamp("2025-01-01")
LEAK_TOL = 0.03                        # 3 Prozentpunkte unter dem erwartbaren Preis gelten als Leakage
MARGIN_FLOOR = 0.10                    # Mindestmarge der Angebotsprüfung
SEED = 7
FX = cn.AW_USD_EUR
UCS = ("price_recommendation", "price_leakage", "quote_check")


def _asof(lines: pd.DataFrame, hist: pd.DataFrame, col: str) -> np.ndarray:
    """Preis bzw. Kosten des Artikels, gültig am Auftragsdatum (Historie mit StartDate)."""
    h = hist.dropna(subset=["StartDate"]).sort_values("StartDate")
    m = pd.merge_asof(lines[["OrderDate", "ProductID"]].reset_index().sort_values("OrderDate"),
                      h[["StartDate", "ProductID", col]].rename(columns={"StartDate": "OrderDate"}),
                      on="OrderDate", by="ProductID", direction="backward")
    return m.set_index("index")[col].reindex(lines.index).to_numpy()


def base_lines() -> pd.DataFrame:
    t = cn.load_tables("adventureworks_data")
    d, h, cu, st = t["SalesOrderDetail"], t["SalesOrderHeader"], t["Customer"], t["Store"]
    x = d.merge(h[["SalesOrderID", "OrderDate", "CustomerID", "TerritoryID", "SalesPersonID"]], on="SalesOrderID")
    x["OrderDate"] = pd.to_datetime(x["OrderDate"])
    c = cu.set_index("CustomerID")
    x["store"] = x["CustomerID"].map(c["StoreID"]).notna()
    prod = t["Product"].set_index("ProductID")
    sub = t["ProductSubcategory"].set_index("ProductSubcategoryID")
    x["subcat"] = x["ProductID"].map(prod["ProductSubcategoryID"]).fillna(-1).astype(int)
    x["category"] = x["subcat"].map(sub["ProductCategoryID"]).fillna(-1).astype(int)
    x["list"] = _asof(x, t["ProductListPriceHistory"], "ListPrice")
    x["list"] = np.where(np.isfinite(x["list"]), x["list"], x["ProductID"].map(prod["ListPrice"]))
    x["cost"] = _asof(x, t["ProductCostHistory"], "StandardCost")
    x["cost"] = np.where(np.isfinite(x["cost"]), x["cost"], x["ProductID"].map(prod["StandardCost"]))
    x["net"] = x["UnitPrice"] * (1 - x["UnitPriceDiscount"])
    x = x[x["list"] > 0].copy()
    x["ratio"] = x["net"] / x["list"]
    x["margin"] = (x["net"] - x["cost"]) / x["net"].clip(lower=1e-6)
    so = t["SpecialOffer"]
    tiers = so[so["Type"] == "Volume Discount"].sort_values("MinQty")
    x["tier_disc"] = 0.0
    for _, r in tiers.iterrows():
        x.loc[x["OrderQty"] >= r["MinQty"], "tier_disc"] = float(r["DiscountPct"])
    bom = t["BillOfMaterials"]
    bom = bom[bom["EndDate"].isna()] if "EndDate" in bom else bom
    x["bom_parts"] = x["ProductID"].map(bom.groupby("ProductAssemblyID").size()).fillna(0).astype(int)
    x["bom_level"] = x["ProductID"].map(bom.groupby("ProductAssemblyID")["BOMLevel"].max()).fillna(0).astype(int)
    x["offer_type"] = x["SpecialOfferID"].map(so.set_index("SpecialOfferID")["Type"]).astype(str)
    x["store_sp"] = x["CustomerID"].map(c["StoreID"].map(st.set_index("BusinessEntityID")["SalesPersonID"])).fillna(-1).astype(int)
    return x.reset_index(drop=True)


BASE_FEATS = ["log_qty", "log_list", "tier_disc", "category", "subcat", "territory", "salesperson", "bom_parts", "month"]
GRAPH_FEATS = ["g_peer_prod_terr", "g_peer_prod", "g_peer_subcat_terr", "g_cust_ratio", "g_cust_n", "g_sp_ratio",
               "g_peer_n"]
FEATURE_LABELS = {"log_qty": "Menge (log)", "log_list": "Listenpreis (log)", "tier_disc": "Rabatt laut Mengenstaffel",
                  "category": "Kategorie", "subcat": "Unterkategorie", "territory": "Gebiet", "salesperson": "Vertriebler",
                  "bom_parts": "Bauteile laut Stückliste", "month": "Monat",
                  "g_peer_prod_terr": "Graph: Preis desselben Artikels im Gebiet (Peers)",
                  "g_peer_prod": "Graph: Preis desselben Artikels (alle Händler)",
                  "g_peer_subcat_terr": "Graph: Preisniveau der Unterkategorie im Gebiet",
                  "g_cust_ratio": "Graph: bisheriges Preisniveau des Händlers", "g_cust_n": "Positionen des Händlers bisher",
                  "g_sp_ratio": "Graph: Preisniveau im Portfolio des Vertrieblers", "g_peer_n": "Peer-Positionen (Artikel, Gebiet)"}


def _features(x: pd.DataFrame, Q: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Positionen im Zeitraum [Q, end) mit Peer-Merkmalen ausschließlich aus Aufträgen vor Q (365 Tage)."""
    cur = x[(x["OrderDate"] >= Q) & (x["OrderDate"] < end) & x["store"]].copy()
    hist = x[(x["OrderDate"] < Q) & (x["OrderDate"] >= Q - pd.Timedelta(days=365)) & x["store"]]
    assert hist["OrderDate"].max() < Q if len(hist) else True        # Leakage-Guard

    def med(keys, name):
        s = hist.groupby(keys)["ratio"].median()
        return cur.set_index(keys).index.map(s).to_numpy(dtype=float) if len(keys) > 1 else cur[keys[0]].map(s).to_numpy(dtype=float)

    cur["g_peer_prod_terr"] = med(["ProductID", "TerritoryID"], "a")
    cur["g_peer_prod"] = med(["ProductID"], "b")
    cur["g_peer_subcat_terr"] = med(["subcat", "TerritoryID"], "c")
    cur["g_cust_ratio"] = (cur["ratio"] * 0 + cur["CustomerID"].map(hist.groupby("CustomerID")["ratio"].median())).to_numpy()
    cur["g_cust_n"] = cur["CustomerID"].map(hist.groupby("CustomerID").size()).fillna(0).to_numpy()
    cur["g_sp_ratio"] = cur["store_sp"].map(hist.groupby("store_sp")["ratio"].median()).to_numpy()
    cur["g_peer_n"] = cur.set_index(["ProductID", "TerritoryID"]).index.map(
        hist.groupby(["ProductID", "TerritoryID"]).size()).to_numpy(dtype=float)
    cur["log_qty"], cur["log_list"] = np.log1p(cur["OrderQty"]), np.log1p(cur["list"])
    cur["territory"], cur["salesperson"] = cur["TerritoryID"].fillna(-1).astype(int), cur["store_sp"]
    cur["month"] = cur["OrderDate"].dt.month
    # Regel wie im Vertrieb: zuletzt erzielter Preis des Artikels, sonst Preisliste mit Mengenstaffel (60 % Händlerbasis)
    cur["rule"] = np.where(np.isfinite(cur["g_peer_prod"]), cur["g_peer_prod"], 0.6 * (1 - cur["tier_disc"]))
    cur["cutoff"] = Q
    return cur


def build(log=print):
    x = base_lines()
    quarters = list(pd.date_range(T_TRAIN, T_TEST - pd.Timedelta(days=1), freq="QS"))
    tr = pd.concat([_features(x, Q, Q + pd.DateOffset(months=3)) for Q in quarters], ignore_index=True)
    te = _features(x, T_TEST, x["OrderDate"].max() + pd.Timedelta(days=1))
    log(f"[pricing] {len(tr):,} Trainings- und {len(te):,} Testpositionen (Händler) · {len(quarters)} Quartale")
    return x, tr, te, quarters


def _lgbm(tr, te, feats, cats):
    import lightgbm as lgb
    va_q = tr["cutoff"].max()
    a, v = tr[tr["cutoff"] < va_q], tr[tr["cutoff"] == va_q]
    X = lambda d: d[feats].astype({c: "category" for c in cats if c in feats})  # noqa: E731
    m = lgb.train(dict(objective="l1", learning_rate=0.05, num_leaves=31, min_data_in_leaf=30, feature_fraction=0.9,
                       seed=SEED, verbose=-1, num_threads=4),
                  lgb.Dataset(X(a), a["ratio"]), 2000, valid_sets=[lgb.Dataset(X(v), v["ratio"])],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
    imp = m.feature_importance("gain")
    return m.predict(X(te), num_iteration=m.best_iteration), sorted(
        [{"feature": f, "label": FEATURE_LABELS.get(f, f), "importance": round(float(i), 1)} for f, i in zip(feats, imp)],
        key=lambda r: -r["importance"])[:12]


def _tabpfn(tr, te, feats):
    try:
        import torch
        from tabpfn import TabPFNRegressor
    except Exception as exc:
        return None, f"TabPFN nicht verfügbar ({type(exc).__name__})"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = tr.sample(min(len(tr), 8000), random_state=SEED)
    try:
        from tabpfn.constants import ModelVersion
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=dev)
    except Exception:
        m = TabPFNRegressor(device=dev)
    Xs, Xt = s[feats].astype(float).fillna(-1).to_numpy(), te[feats].astype(float).fillna(-1).to_numpy()
    m.fit(Xs, s["ratio"].to_numpy())
    out = np.concatenate([m.predict(Xt[i:i + 4000]) for i in range(0, len(Xt), 4000)])
    return out, f"Kontext {len(s):,} Positionen, {dev}".replace(",", ".")


def _boot_lift(err_rule, err_model, groups, n=300):
    rng = np.random.default_rng(SEED)
    g = pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).apply(list).tolist()
    out = []
    for _ in range(n):
        idx = np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])
        r, m = err_rule[idx].mean(), err_model[idx].mean()
        out.append((r - m) / r if r else 0.0)
    return [round(float(np.percentile(out, 5)), 4), round(float(np.percentile(out, 95)), 4)]


def evaluate(log=print) -> dict:
    t0 = time.time()
    x, tr, te, quarters = build(log)
    cats = ["category", "subcat", "territory", "salesperson", "month"]
    preds, notes, imps = {"rule": te["rule"].to_numpy()}, {}, {}
    preds["lgbm"], imps["lgbm"] = _lgbm(tr, te, BASE_FEATS, cats)
    preds["lgbm_graph"], imps["lgbm_graph"] = _lgbm(tr, te, BASE_FEATS + GRAPH_FEATS, cats)
    p, note = _tabpfn(tr, te, BASE_FEATS + GRAPH_FEATS)
    if p is not None:
        preds["tabpfn"] = p
    notes["tabpfn"] = note
    y = te["ratio"].to_numpy()
    listv = (te["list"] * te["OrderQty"]).to_numpy() * FX
    res = {}
    for k, pr in preds.items():
        err = np.abs(pr - y)
        gap = pr - y - LEAK_TOL
        flag = gap > 0
        res[k] = {"mae": round(float(err.mean()), 5), "mae_pp": round(float(err.mean() * 100), 2),
                  "leak_eur": round(float((gap.clip(min=0) * listv).sum()), 2), "n_flag": int(flag.sum()),
                  "orders_flag": int(te.loc[flag, "SalesOrderID"].nunique()), "_err": err, "_pred": pr}
        log(f"[pricing] {k:11s} MAE {res[k]['mae_pp']:.2f} Pp. · Leakage {res[k]['leak_eur']:,.0f} € in {res[k]['n_flag']:,} Positionen")
    best = min((k for k in res if k != "rule"), key=lambda k: res[k]["mae"])
    lift = (res["rule"]["mae"] - res[best]["mae"]) / res["rule"]["mae"]
    ci = _boot_lift(res["rule"]["_err"], res[best]["_err"], te["SalesOrderID"].to_numpy())
    # Leakage-Beispiele aus dem besten Modell
    te = te.assign(pred=res[best]["_pred"], gap=res[best]["_pred"] - y)
    t = cn.load_tables("adventureworks_data")
    pname = t["Product"].set_index("ProductID")["Name"].astype(str)
    terr = t["SalesTerritory"].set_index("TerritoryID")["Name"].astype(str)
    lk = te[te["gap"] > LEAK_TOL].assign(eur=lambda d: (d["gap"] - LEAK_TOL) * d["list"] * d["OrderQty"] * FX)
    by_sp = lk.groupby("store_sp")["eur"].sum().sort_values(ascending=False)
    examples = [{"order": int(r.SalesOrderID), "product": pname.get(r.ProductID, str(r.ProductID)), "qty": int(r.OrderQty),
                 "territory": terr.get(r.TerritoryID, ""), "actual": round(float(r.ratio), 3), "expected": round(float(r.pred), 3),
                 "eur": round(float(r.eur), 0), "offer": r.offer_type}
                for r in lk.sort_values("eur", ascending=False).head(12).itertuples()]
    months = (x["OrderDate"].max() - T_TEST).days / 30.4
    periods = round(12 / months, 2)
    # Angebotsprüfung: Marge mit Kosten zum Datum (alle Aufträge im Testzeitraum, Händler und Online)
    q = x[x["OrderDate"] >= T_TEST].copy()
    q["dm"] = (q["net"] - q["cost"]) * q["OrderQty"] * FX
    q["floor_gap"] = ((MARGIN_FLOOR * q["net"] - (q["net"] - q["cost"])).clip(lower=0) * q["OrderQty"] * FX)
    orders = q.groupby("SalesOrderID").agg(net=("net", lambda s: float((s * q.loc[s.index, "OrderQty"]).sum() * FX)),
                                           dm=("dm", "sum"), gap=("floor_gap", "sum"), store=("store", "first"),
                                           bom=("bom_level", "max"), offer=("offer_type", lambda s: s.mode().iat[0]))
    orders["margin"] = orders["dm"] / orders["net"].clip(lower=1e-6)
    flag_o = orders[orders["margin"] < MARGIN_FLOOR]
    loss_lines = q[q["margin"] < 0]
    by_offer = loss_lines.groupby("offer_type")["dm"].sum().sort_values().head(6)
    by_bom = q.groupby("bom_level").apply(lambda g: float((g["margin"] < 0).mean()), include_groups=False)
    check = {"orders": int(len(orders)), "orders_flag": int(len(flag_o)), "lines_loss": int(len(loss_lines)),
             "loss_eur": round(float(-loss_lines["dm"].sum()), 2), "floor_gap_eur": round(float(flag_o["gap"].sum()), 2),
             "share_store_flag": round(float(flag_o["store"].mean()), 3) if len(flag_o) else None,
             "by_offer": {k: round(float(v), 0) for k, v in by_offer.items()},
             "loss_share_by_bom_level": {int(k): round(v, 3) for k, v in by_bom.items()},
             "floor": MARGIN_FLOOR, "periods": periods}
    log(f"[pricing] Angebotsprüfung: {check['orders_flag']:,} von {check['orders']:,} Aufträgen unter {MARGIN_FLOOR:.0%} Marge · "
        f"Verlustpositionen {check['lines_loss']:,} ({check['loss_eur']:,.0f} €) · {time.time() - t0:.0f} s")
    for k in res:
        res[k].pop("_err")
        res[k].pop("_pred")
    return {"engines": res, "best": best, "lift": lift, "lift_ci": ci, "importance": imps, "notes": notes,
            "examples": examples, "leak_by_salesperson": {f"Vertriebler {int(k)}": round(float(v), 0) for k, v in by_sp.head(8).items()},
            "check": check, "sizes": {"train_rows": int(len(tr)), "test_rows": int(len(te)), "quarters": len(quarters)},
            "test_from": T_TEST.date().isoformat(), "test_to": x["OrderDate"].max().date().isoformat(), "periods": periods,
            "seconds": round(time.time() - t0, 1)}


NAMES = {"rule": ("Regel: zuletzt erzielter Artikelpreis", "Regel"), "lgbm": ("LightGBM", "Baum"),
         "lgbm_graph": ("LightGBM + Peer-Preise (Graph)", "Graph-ML light"), "tabpfn": ("TabPFN v2 (Regression)", "TFM")}


def to_results(r: dict) -> dict[str, dict]:
    from . import evaluation as ev
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    tl = {"test": r["test_from"], "test_label_end": r["test_to"], "train": [], "val": None, "horizon_days": 0,
          "label": "Nettopreis in % vom Listenpreis je Händlerposition"}
    vk = ev.vorhersagekraft(r["lift"], r["lift_ci"])
    common = {"result_version": ev.RESULT_VERSION, "variant": "v0", "currency": "EUR", "dataset": "adventureworks_data",
              "generated_at": now, "host": "lokal", "compute_seconds": r["seconds"], "timeline": tl, "audit": [],
              "sizes": r["sizes"], "top_list": [], "pricing": {k: v for k, v in r.items() if k != "engines"}}
    eng = lambda metrics: [{"id": k, "status": "ok", "meta": {"name": NAMES[k][0], "klass": NAMES[k][1], "device": "CPU" if k != "tabpfn" else "GPU",  # noqa: E731
                                                              "notes": [r["notes"].get(k)] if r["notes"].get(k) else []},
                            "metrics": metrics(v), "explain": {"global": r["importance"].get(k)} if k in r["importance"] else None}
                           for k, v in r["engines"].items()]
    best_name = NAMES[r["best"]][0]
    out = {}
    out["price_recommendation"] = {**common, "use_case": "price_recommendation", "kind": "regression", "main_metric": "mae",
        "main_metric_label": "Abweichung vom erzielten Preis (MAE, Anteil des Listenpreises)", "lower_is_better": True,
        "rule_text": NAMES["rule"][0], "engines": eng(lambda v: {"mae": v["mae"], "mae_pp": v["mae_pp"]}), "best": r["best"],
        "vorhersagekraft": vk, "recommendation": {"engine": r["best"], "name": best_name, "text":
            f"Bestes Modell: {best_name} – Preisempfehlung {vk.get('reason', '')}."},
        "whatif": {"kind": "leakage", "currency": "EUR", "periods_default": r["periods"],
                   "engines": {k: {"leak_eur": v["leak_eur"], "n": v["orders_flag"]} for k, v in r["engines"].items()},
                   "unit": "Preislücke zum erwartbaren Preis im Testzeitraum"}}
    out["price_leakage"] = {**out["price_recommendation"], "use_case": "price_leakage",
        "main_metric_label": "Treffsicherheit des erwartbaren Preises (MAE)",
        "rule_text": "Leakage = Nettopreis mehr als 3 Prozentpunkte unter dem erwartbaren Preis"}
    c = r["check"]
    out["quote_check"] = {**common, "use_case": "quote_check", "kind": "check", "main_metric": "orders_flag",
        "main_metric_label": f"Aufträge unter {c['floor']:.0%} Marge", "lower_is_better": False,
        "rule_text": f"Marge je Auftrag mit Standardkosten zum Datum < {c['floor']:.0%}", "best": "check",
        "engines": [{"id": "check", "status": "ok", "meta": {"name": "Regelprüfung Marge (Kosten zum Datum)", "klass": "Regel", "device": "CPU"},
                     "metrics": {"orders_flag": c["orders_flag"], "floor_gap_eur": c["floor_gap_eur"]}}],
        "vorhersagekraft": {"status": "na", "reason": "Regelprüfung mit bekannten Kosten – kein Modell nötig, Vorhersagekraft nicht anwendbar"},
        "recommendation": {"engine": "check", "name": "Regelprüfung", "text": "Angebotsprüfung ist eine Regel mit Kosten zum Datum und Stückliste; "
                           "Graph ML bringt hier wenig – die Lücke ist Datenqualität (Kosten im Angebot), nicht Vorhersage."},
        "whatif": {"kind": "check", "currency": "EUR", "periods_default": r["periods"],
                   "engines": {"check": {"gap_eur": c["floor_gap_eur"], "n": c["orders_flag"]}}}}
    return out


def main() -> int:
    from . import evaluation as ev
    for res in to_results(evaluate(log=lambda m: print(m, flush=True))).values():
        ev.save(res)
        print(res["use_case"], res["vorhersagekraft"].get("reason"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
