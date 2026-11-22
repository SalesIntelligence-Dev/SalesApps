"""
Gewinnwahrscheinlichkeit & Umsatzprognose (Prompt 9) – WideWorldImporters (künstlich)
=====================================================================================
  WWI hat keine Angebote/Opportunities, aber die Kette Auftrag → Rechnung. Fast alle nicht fakturierten Aufträge sind
  Rückstandsaufträge, die auf Ware warten. Vorhersage je Rückstandsauftrag: wird er NICHT innerhalb von 30 Tagen geliefert
  und fakturiert (Umsatz gefährdet)? Daraus die Umsatzprognose der Pipeline je Woche (Σ Auftragswert × Lieferwahrscheinlichkeit).

  Graph-Signal: Auftrag → Artikel → andere wartende Aufträge (Rückstandsdruck), Artikel → Lieferant → dessen Lieferquote,
  Artikel → offene Einkaufsbestellungen (geplanter Nachschub) – alles zum Auftragsdatum bekannt.
  Leakage-Guard: Ein Label steht 30 Tage nach Auftragsdatum fest; Quoten zählen nur Aufträge, deren Label vorher feststand.

  .venv/bin/python -m graph_ml.sales_pipeline
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import connectors as cn
from . import quality as q

SEED = 7
WIN = 30
TEST_FROM = pd.Timestamp("2016-01-01")
FX = cn.WWI_USD_EUR


def backorders(log=print) -> tuple[pd.DataFrame, dict]:
    t = cn.load_tables("wwi")
    o = t["Orders"].copy()
    o["OrderDate"] = pd.to_datetime(o["OrderDate"])
    inv = t["Invoices"][["OrderID", "InvoiceDate"]].copy()
    inv["InvoiceDate"] = pd.to_datetime(inv["InvoiceDate"])
    o = o.merge(inv, on="OrderID", how="left")
    end = max(o["OrderDate"].max(), o["InvoiceDate"].max())
    ol = t["OrderLines"].copy()
    for c in ("Quantity", "UnitPrice"):
        ol[c] = pd.to_numeric(ol[c].astype(str), errors="coerce").fillna(0)
    ol["value"] = ol["Quantity"] * ol["UnitPrice"] * FX
    agg = ol.groupby("OrderID").agg(value=("value", "sum"), qty=("Quantity", "sum"), lines=("OrderLineID", "size"),
                                    item=("StockItemID", "first"))
    b = o[o["OrderID"].isin(o["BackorderOrderID"])].merge(agg, left_on="OrderID", right_index=True)
    b = b[b["OrderDate"] <= end - pd.Timedelta(days=WIN)].copy()        # sonst ist das Label noch offen
    b["avail"] = b["OrderDate"] + pd.Timedelta(days=WIN)
    b["delivered_at"] = b["InvoiceDate"]
    b["y"] = (b["InvoiceDate"].isna() | (b["InvoiceDate"] > b["avail"])).astype(int)    # 1 = Umsatz gefährdet
    si = t["StockItems"].set_index("StockItemID")
    b["supplier"] = b["item"].map(si["SupplierID"])
    b["lead"] = b["item"].map(si["LeadTimeDays"]).astype(float)
    b["chiller"] = b["item"].map(si["IsChillerStock"]).astype(str).isin(["1", "True", "true"]).astype(float)
    b["price"] = b["item"].map(pd.to_numeric(si["UnitPrice"].astype(str), errors="coerce")).astype(float)
    cu = t["Customers"].set_index("CustomerID")
    b["cat"] = b["CustomerID"].map(cu["CustomerCategoryID"]).astype(float)
    b["bgroup"] = b["CustomerID"].map(cu["BuyingGroupID"]).fillna(0).astype(float)
    b["credit"] = b["CustomerID"].map(pd.to_numeric(cu["CreditLimit"].astype(str), errors="coerce")).fillna(0).astype(float)
    b = b.sort_values("OrderDate").reset_index(drop=True)
    log(f"[pipeline] WWI: {len(b):,} Rückstandsaufträge, gefährdet {b['y'].mean():.3f}, Test ab {TEST_FROM.date()}: "
        f"{int((b['OrderDate'] >= TEST_FROM).sum()):,}")
    return b, t


def features(b: pd.DataFrame, t: dict) -> pd.DataFrame:
    qt, av, y = b["OrderDate"], b["avail"], b["y"]
    prior = float(b.loc[b["avail"] < TEST_FROM, "y"].mean())
    f = {}
    s, n = q.past_stats(b["CustomerID"], av, y, b["CustomerID"], qt)
    f["c_rate"], f["c_n"] = q._rate(s, n, prior), n
    # Graph: Auftrag → Artikel → frühere Rückstände desselben Artikels
    s, n = q.past_stats(b["item"], av, y, b["item"], qt)
    f["g_item_rate"], f["g_item_n"] = q._rate(s, n, prior), n
    # Graph: Artikel → Lieferant → Rückstände aller Artikel dieses Lieferanten
    s, n = q.past_stats(b["supplier"], av, y, b["supplier"], qt)
    f["g_supplier_rate"] = q._rate(s, n, prior)
    # Graph: Kunde → Einkaufsgemeinschaft → Rückstände der Gemeinschaft
    s, n = q.past_stats(b["bgroup"], av, y, b["bgroup"], qt)
    f["g_group_rate"] = q._rate(s, n, prior)
    # Graph: Artikel → andere zum Auftragsdatum noch offene Rückstände (Druck auf den Bestand)
    od, dl = b["OrderDate"].to_numpy(), b["delivered_at"].to_numpy()
    open_same = np.zeros(len(b))
    for it, g in b.groupby("item"):
        ix = g.index.to_numpy()
        d0, d1 = od[ix], dl[ix]
        for k, i in enumerate(ix):
            prev = slice(0, k)
            open_same[i] = np.sum(np.isnat(d1[prev]) | (d1[prev] > d0[k]))
    f["g_open_same_item"] = open_same
    # Graph: Artikel → offene Einkaufsbestellungen (bestellt vor dem Auftrag, erwartet ab Auftragsdatum)
    po = t["PurchaseOrders"][["PurchaseOrderID", "OrderDate", "ExpectedDeliveryDate"]].copy()
    po["OrderDate"], po["ExpectedDeliveryDate"] = pd.to_datetime(po["OrderDate"]), pd.to_datetime(po["ExpectedDeliveryDate"])
    pol = t["PurchaseOrderLines"][["PurchaseOrderID", "StockItemID", "OrderedOuters"]].merge(po, on="PurchaseOrderID")
    pol["OrderedOuters"] = pd.to_numeric(pol["OrderedOuters"].astype(str), errors="coerce").fillna(0)
    n_po, days_po = np.zeros(len(b)), np.full(len(b), 120.0)
    for it, g in b.groupby("item"):
        p = pol[pol["StockItemID"] == it]
        if p.empty:
            continue
        for i, d in zip(g.index, g["OrderDate"]):
            m = (p["OrderDate"] < d) & (p["ExpectedDeliveryDate"] >= d)
            n_po[i] = p.loc[m, "OrderedOuters"].sum()
            if m.any():
                days_po[i] = (p.loc[m, "ExpectedDeliveryDate"].min() - d).days
    f["g_open_po_outers"], f["g_days_to_next_po"] = n_po, days_po
    for c in ("value", "qty", "lines", "lead", "chiller", "price", "cat", "bgroup", "credit"):
        f[c] = b[c].astype(float).to_numpy()
    f["month"] = qt.dt.month.to_numpy()
    f["weekday"] = qt.dt.weekday.to_numpy()
    return pd.DataFrame(f, index=b.index)


BASE = ["value", "qty", "lines", "lead", "chiller", "price", "cat", "bgroup", "credit", "month", "weekday", "c_rate", "c_n"]
GRAPH = ["g_item_rate", "g_item_n", "g_supplier_rate", "g_group_rate", "g_open_same_item", "g_open_po_outers", "g_days_to_next_po"]
LABELS = {"value": "Auftragswert", "qty": "Menge", "lines": "Positionen", "lead": "Lieferzeit des Artikels", "chiller": "Kühlware",
          "price": "Artikelpreis", "cat": "Kundenkategorie", "bgroup": "Einkaufsgemeinschaft", "credit": "Kreditlimit", "month": "Monat",
          "weekday": "Wochentag", "c_rate": "Kunde: bisherige Quote gefährdeter Rückstände", "c_n": "Kunde: bisherige Rückstände",
          "g_item_rate": "Graph: Artikel – Quote früherer Rückstände", "g_item_n": "Graph: Artikel – frühere Rückstände",
          "g_supplier_rate": "Graph: Lieferant des Artikels – Quote", "g_group_rate": "Graph: Einkaufsgemeinschaft – Quote",
          "g_open_same_item": "Graph: offene Rückstände desselben Artikels", "g_open_po_outers": "Graph: offener Nachschub (Einkauf)",
          "g_days_to_next_po": "Graph: Tage bis zum nächsten Wareneingang"}


def evaluate(log=print) -> dict:
    b, t = backorders(log)
    f = features(b, t)
    ix = np.arange(len(b))
    links = {"artikel": (ix, *q._index(b["item"])), "kunde": (ix, *q._index(b["CustomerID"])),
             "lieferant": (ix, *q._index(b["supplier"].fillna(-1)))}
    links = {k: (a, c, n) for k, (a, c, n) in links.items()}
    names = {"rule": ("Regel: größte Rückstandsmenge zuerst", "Regel"),
             "lgbm": ("LightGBM (Auftrag + Kundenhistorie)", "Baum"),
             "lgbm_graph": ("LightGBM + Artikel/Lieferant/Nachschub (Graph)", "Graph-ML light"),
             "tabpfn": ("TabPFN v2", "TFM"), "gnn": ("GNN Auftrag ↔ Artikel ↔ Kunde ↔ Lieferant (GraphSAGE)", "Relational")}
    res = q.evaluate_binary(f, b["y"].to_numpy(), b["value"].to_numpy(), b["OrderDate"], TEST_FROM, b["avail"], BASE, GRAPH,
                            f["qty"].to_numpy() + 1e-9 * f["value"].to_numpy(), links, names, LABELS, log)
    # Umsatzprognose der Pipeline je Woche: Σ Wert × (1 − p_gefährdet) gegen tatsächlich binnen 30 Tagen fakturiert
    te = b[res["test"]].copy()
    te["week"] = te["OrderDate"].dt.to_period("W").dt.start_time
    hist = b[(b["avail"] < TEST_FROM) & (b["OrderDate"] >= TEST_FROM - pd.Timedelta(days=182))]
    base_rate = 1 - float(hist["y"].mean())
    actual = te.assign(a=te["value"] * (1 - te["y"])).groupby("week")["a"].sum()
    edges = np.unique(np.quantile(hist["qty"], np.linspace(0, 1, 11)))          # Regel: Lieferquote je Mengenklasse (Vergangenheit)
    cls = lambda x: np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)  # noqa: E731
    rate_cls = pd.Series(1 - hist["y"].to_numpy()).groupby(cls(hist["qty"].to_numpy())).mean()
    p_rule = pd.Series(cls(te["qty"].to_numpy())).map(rate_cls).fillna(base_rate).to_numpy()
    fc = {"rule": te.assign(v=te["value"] * p_rule).groupby("week")["v"].sum()}
    for e in res["engines"]:
        if e["id"] == "rule":
            continue
        p = res["scores"][e["id"]]
        if e["id"] in ("lgbm", "lgbm_graph", "tabpfn", "gnn"):
            fc[e["id"]] = te.assign(v=te["value"] * (1 - p)).groupby("week")["v"].sum()
    wape = {k: round(float(np.abs(v - actual).sum() / max(actual.sum(), 1e-9)), 4) for k, v in fc.items()}
    res["forecast"] = {"weeks": [str(w.date()) for w in actual.index], "actual": [round(float(x), 0) for x in actual],
                       "rule": [round(float(x), 0) for x in fc["rule"]], "best": [round(float(x), 0) for x in fc[res["best"]]],
                       "wape": wape, "base_rate": round(base_rate, 4), "pipeline_eur": round(float(te["value"].sum()), 0),
                       "at_risk_eur": round(float((te["value"] * te["y"]).sum()), 0)}
    log(f"[pipeline] Umsatzprognose je Woche, WAPE: {wape}")
    sn = t["Suppliers"].set_index("SupplierID")["SupplierName"].astype(str) if "Suppliers" in t else pd.Series(dtype=str)
    te["score"] = res["scores"][res["best"]]
    g = te.groupby("supplier").agg(orders=("y", "size"), observed=("y", "mean"), predicted=("score", "mean"),
                                   at_risk=("value", lambda v: float((v * te.loc[v.index, "y"]).sum()))).reset_index()
    res["suppliers"] = [{"supplier": str(sn.get(r.supplier, r.supplier)), "orders": int(r.orders), "observed": round(float(r.observed), 4),
                         "predicted": round(float(r.predicted), 4), "at_risk": round(float(r.at_risk), 0)}
                        for r in g.sort_values("at_risk", ascending=False).head(8).itertuples()]
    return res


def to_result(r: dict) -> dict:
    from . import evaluation as ev
    vk = ev.vorhersagekraft(r["lift"], r["lift_ci"])
    best = next(e for e in r["engines"] if e["id"] == r["best"])
    return {"result_version": ev.RESULT_VERSION, "variant": "v0", "currency": "EUR", "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "host": "lokal", "audit": [], "top_list": [], "use_case": "win_probability", "kind": "binary", "dataset": "wwi",
            "compute_seconds": r["seconds"], "main_metric": "auprc", "main_metric_label": "AUPRC", "lower_is_better": False,
            "timeline": {"train": r["train_dates"], "val": r["val"], "test": str(TEST_FROM.date()), "test_label_end": "2016-05-31",
                         "horizon_days": WIN, "label": f"Rückstandsauftrag wird nicht binnen {WIN} Tagen geliefert und fakturiert",
                         "unit": "Rückstandsauftrag"},
            "rule_text": "größte Rückstandsmenge zuerst", "sizes": r["sizes"], "engines": r["engines"],
            "best": r["best"], "vorhersagekraft": vk, "whatif": {**r["whatif"], "unit": "gefährdeter Auftragswert"},
            "recommendation": {"engine": r["best"], "name": best["meta"]["name"], "klass": best["meta"]["klass"],
                               "text": f"Bestes Modell: {best['meta']['name']} (AUPRC {best['metrics']['auprc']:.3f}".replace(".", ",")
                                       + f") – {vk.get('reason', '')}."},
            "notes": ["WWI ist künstlich und hat keine Angebote – gemessen wird, ob ein Rückstandsauftrag zu Umsatz wird.",
                      "Der Generator folgt fast einer Regel: Rückstände ab etwa 50 Stück bleiben liegen. Deshalb ist schon die "
                      "Mengen-Regel stark – echte Daten sind deutlich unschärfer.",
                      f"Leakage-Guard: Ein Label steht {WIN} Tage nach Auftragsdatum fest; Quoten zählen nur vorher feststehende Fälle."],
            "pipeline": {"forecast": r["forecast"], "suppliers": r["suppliers"]}}


def main() -> int:
    from . import evaluation as ev
    res = to_result(evaluate(log=lambda m: print(m, flush=True)))
    ev.save(res)
    print(res["use_case"], res["vorhersagekraft"].get("reason"), res["pipeline"]["forecast"]["wape"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
