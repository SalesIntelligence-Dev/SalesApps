"""
Tests für GraphReady v2 – Use-Case-Detailseiten und Seitenaufbau.

  python -m unittest tests.test_graph_ml_v2 -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_ml.usecases import STORY, list_use_cases  # noqa: E402

KEYS = {"v2", "effect", "fit", "graph", "path", "tabular", "unit", "euro"}


class StoryTest(unittest.TestCase):
    def test_every_use_case_has_a_detail_page(self):
        for u in list_use_cases():
            self.assertIsNotNone(u.get("story"), u["id"])
            self.assertTrue(KEYS <= set(u["story"]), u["id"])
            self.assertGreaterEqual(len(u["story"]["path"]), 3, u["id"])
            self.assertIn(u["story"]["fit"], ("hoch", "mittel", "gering"), u["id"])

    def test_page_structure(self):
        html = (Path(__file__).resolve().parents[1] / "templates" / "graph_ml.html").read_text(encoding="utf-8")
        self.assertNotIn("Readiness-Trichter", html)
        self.assertNotIn("Demo-Tour: Pitch", html)
        self.assertLess(html.index("SCHRITT 3 · FIX-PLAYBOOKS"), html.index("SCHRITT 4 · READINESS"))
        self.assertLess(html.index("SCHRITT 4 · READINESS"), html.index("SCHRITT 5 · RÜCKBLICK"))
        self.assertIn('data-tour="s5-whatif"', html[html.index("SCHRITT 5 · RÜCKBLICK"):html.index("SCHRITT 6 · AKTIVIERUNG")])
        for sec in ("Worum geht es", "Input-Daten", "Warum relationales Lernen", "Was wird vorhergesagt", "Nutzen in Euro",
                    "Readiness-Ampel"):
            self.assertIn(sec, html)                                     # sechs Abschnitte der Detailseite
        self.assertEqual(set(STORY), {u["id"] for u in list_use_cases()})


class ConnectorTest(unittest.TestCase):
    def test_roundtrip_and_declared_keys(self):
        import tempfile

        import numpy as np
        import pandas as pd

        from graph_ml import connectors as cn
        from graph_ml.graph_mapping import _measure_declared
        tables = {"orders": pd.DataFrame({"id": [1, 2, 3, 4], "cust": [10.0, 11.0, np.nan, 99.0],
                                          "day": pd.to_datetime(["2024-01-01", None, "2024-01-03", "2024-01-04"]),
                                          "note": ["a", None, "c", "a"]}),
                  "lines": pd.DataFrame({"order": [1, 1, 2], "pos": [1, 2, 1]}),
                  "customers": pd.DataFrame({"cust": [10, 11, 12]})}
        old = cn.DATA_DIR
        with tempfile.TemporaryDirectory() as d:
            cn.DATA_DIR = Path(d)
            try:
                cn.save_tables("unit_test", tables, {"name": "x"})
                back = cn.load_tables("unit_test")
            finally:
                cn.DATA_DIR = old
        self.assertEqual(back["orders"]["note"].tolist()[0], "a")
        self.assertTrue(pd.isna(back["orders"]["day"].iloc[1]))
        r = _measure_declared({"from_table": "orders", "from_columns": ["cust"], "to_table": "customers",
                               "to_columns": ["cust"]}, back)
        self.assertAlmostEqual(r["linkage_rate"], 2 / 3, places=3)      # 10.0 == 10, 99 verwaist, NaN = leer
        self.assertEqual(r["null_rows"], 1)
        r2 = _measure_declared({"from_table": "lines", "from_columns": ["order", "pos"], "to_table": "lines",
                                "to_columns": ["order", "pos"]}, back)
        self.assertTrue(r2["composite"])
        self.assertEqual(r2["linkage_rate"], 1.0)

    def test_prepared_datasets_are_registered(self):
        from graph_ml import connectors as cn
        from graph_ml.data_loader import get_registry
        ids = {d["id"] for d in get_registry().list()}
        for ds in cn.available():
            self.assertIn(ds, ids)
            meta = cn.load_meta(ds)
            self.assertTrue(meta["license"] and meta["source"])


class SalesCoreTest(unittest.TestCase):
    """Prompt 3: Account-Scoring (WWI), GNN für Cross-Selling und Empfehlung, Datensatz-Trennung."""

    def test_peer_growth_excludes_self(self):
        import numpy as np
        import pandas as pd

        from graph_ml.backtest import _peer_growth
        f = pd.DataFrame({"monetary_90": [200.0, 100.0, 100.0], "prev_rev90": [100.0, 100.0, 100.0]}, index=[1, 2, 3])
        g = _peer_growth(f, pd.Series(["A", "A", "B"], index=[1, 2, 3]))
        self.assertAlmostEqual(g[0], 0.0)                         # Peer von 1 ist 2 (kein Wachstum)
        self.assertAlmostEqual(g[1], np.log1p(200) - np.log1p(100), places=6)
        self.assertTrue(np.isnan(g[2]))                           # allein in der Gruppe → kein Peer-Signal

    def test_account_task_no_leakage_and_dates(self):
        import pandas as pd

        from graph_ml import backtest as bt
        task = bt.get_task("account_scoring")
        self.assertEqual(task.timeline["test"], "2016-02-01")
        self.assertEqual(bt.T_TEST, pd.Timestamp("2011-09-01"))     # Zeitachse danach wieder Online Retail II
        for r in task.audit.rows:
            if r["max_feature_time"]:
                self.assertLess(pd.Timestamp(r["max_feature_time"]), pd.Timestamp(r["cutoff"]))
        self.assertTrue({"g_peer_group_growth", "g_peer_category_growth"} <= set(task.graph_features))

    def test_artifacts_compare_graph_models(self):
        from graph_ml import business_case as bc
        from graph_ml import evaluation as ev
        ids = lambda uc: {e["id"]: e["status"] for e in ev.load(uc)["engines"]}  # noqa: E731
        self.assertEqual(ids("cross_selling").get("gnn"), "ok")
        self.assertEqual(ids("recommendation").get("gnn_rank"), "ok")
        acc = ev.load("account_scoring")
        self.assertEqual(acc["dataset"], "wwi")
        ov = bc.overview()
        self.assertEqual(ov["cases"]["account_scoring"]["dataset"], "wwi")
        self.assertNotEqual(ov["next_step"].get("use_case"), "account_scoring")   # keine Werte aus anderen Daten


class Prompt4Test(unittest.TestCase):
    """Next Best Action (Policy-Rückblick), Absatzprognose mit Gruppen-Graph, Vertriebsabdeckung auf AdventureWorks."""

    def test_nba_is_measurable_as_policy(self):
        from graph_ml import business_case as bc
        from graph_ml import evaluation as ev
        self.assertNotIn("next_best_action", ev.NOT_MEASURABLE)
        r = ev.load("next_best_action")
        self.assertEqual(r["kind"], "policy")
        pol = r["policy_eval"]["policies"]
        self.assertGreater(pol["nba_model"]["net"], pol["single_action"]["net"])
        w = r["whatif"]
        a = bc.compute("next_best_action", w, {"periods": 4}, "nba")["annual"]
        self.assertAlmostEqual(a, w["engines"]["nba"]["net"] * 4, places=2)   # identisch zur JS-Formel
        self.assertIn(ev.summary()["next_best_action"]["vorhersagekraft"]["status"], ("ok", "warn", "bad"))

    def test_territory_task_and_table(self):
        import pandas as pd

        from graph_ml import activation as act
        from graph_ml import backtest as bt
        task = bt.get_task("territory_coverage")
        self.assertEqual(task.timeline["test"], "2025-03-01")
        self.assertNotIn("next_revenue", task.features)              # Zukunftswert nur für Was-wäre-wenn
        for r in task.audit.rows:
            if r["max_feature_time"]:
                self.assertLess(pd.Timestamp(r["max_feature_time"]), pd.Timestamp(r["cutoff"]))
        tab = act.load_territory_table()
        self.assertEqual(len(tab["territories"]), 10)
        self.assertTrue(all("quota_quarter" in t and "underserved" in t for t in tab["territories"]))

    def test_stage3_repeated_calls_keep_backtest(self):
        from graph_ml.data_loader import Registry
        r = Registry()
        first = r.checks("online_retail_ii", 3)
        second = r.checks("online_retail_ii", 3)             # früher: Cache lieferte die Ampel ohne Rückblick
        f = {u["id"]: u["ampel"]["gesamt"]["status"] for u in first["use_cases"]}
        g = {u["id"]: u["ampel"]["gesamt"]["status"] for u in second["use_cases"]}
        self.assertEqual(f, g)
        self.assertTrue(second.get("backtest_merged"))

    def test_forecast_graph_features(self):
        import numpy as np

        from graph_ml import backtest as bt
        from graph_ml.engines.cpu import _fc_graph_cols
        task = bt.get_task("demand_forecast")
        Y, o = task.extra["Y"].to_numpy(float), task.extra["weeks"].get_loc(task.extra["origin_test"])
        g = _fc_graph_cols(task, Y, o)
        self.assertEqual(len(g), Y.shape[0])
        grp = bt.base_of(task).prod_group[np.asarray(task.extra["prods"])]
        i = 0
        sib = [j for j in range(len(grp)) if grp[j] == grp[i] and j != i]
        if sib:
            self.assertAlmostEqual(g["g_grp_mean4"].iloc[i], Y[sib, o - 4:o].mean(1).mean(), places=6)  # ohne sich selbst


class PricingTest(unittest.TestCase):
    """Prompt 5: Preis- & Rabattempfehlung, Preis-Leakage, Angebotsprüfung (AdventureWorks)."""

    def test_no_decision_leakage_in_features(self):
        from graph_ml import pricing as pr
        feats = pr.BASE_FEATS + pr.GRAPH_FEATS
        self.assertFalse({"UnitPriceDiscount", "SpecialOfferID", "ratio", "net", "UnitPrice"} & set(feats))

    def test_peer_features_from_past_only(self):
        import pandas as pd

        from graph_ml import pricing as pr
        x = pr.base_lines()
        Q = pd.Timestamp("2024-07-01")
        cur = pr._features(x, Q, Q + pd.DateOffset(months=3))
        self.assertTrue((cur["OrderDate"] >= Q).all())
        late = x[(x["OrderDate"] >= Q) & x["store"]].copy()
        late["ratio"] = 0.0                                       # Zukunft verändern …
        x2 = pd.concat([x[x["OrderDate"] < Q], late])
        cur2 = pr._features(x2, Q, Q + pd.DateOffset(months=3))
        self.assertTrue(cur["g_peer_prod"].fillna(-1).equals(cur2["g_peer_prod"].fillna(-1)))   # … Merkmale bleiben gleich

    def test_artifacts_and_business_case(self):
        from graph_ml import business_case as bc
        from graph_ml import evaluation as ev
        rec = ev.load("price_recommendation")
        self.assertEqual(rec["kind"], "regression")
        self.assertTrue(rec["lower_is_better"])
        maes = {e["id"]: e["metrics"]["mae"] for e in rec["engines"]}
        self.assertLess(maes[rec["best"]], maes["rule"])
        w = rec["whatif"]
        e = w["engines"][rec["best"]]
        a = bc.compute("price_leakage", w, {"rate": 40, "cost": 10, "periods": 2}, rec["best"])["annual"]
        self.assertAlmostEqual(a, (e["leak_eur"] * 0.4 - e["n"] * 10) * 2, places=2)       # identisch zur JS-Formel
        q = ev.load("quote_check")
        self.assertEqual(q["vorhersagekraft"]["status"], "na")
        self.assertGreater(q["pricing"]["check"]["orders_flag"], 0)
        ov = bc.overview()
        self.assertTrue(all(ov["cases"][u]["dataset"] == "adventureworks_data" for u in ("price_recommendation", "quote_check")))


if __name__ == "__main__":
    unittest.main()

class AfterSalesTest(unittest.TestCase):
    """Prompt 6: Installierte Basis (SCANIA), Serviceverträge, Ersatzteilpreise (AdventureWorks)."""

    def test_peer_rates_never_see_own_label(self):
        import numpy as np
        import pandas as pd

        from graph_ml import aftersales as af
        rng = np.random.default_rng(0)
        n = 400
        v = pd.DataFrame({"vehicle_id": np.arange(n), "repaired": rng.integers(0, 2, n),
                          **{c: rng.choice(["a", "b"], n).astype(str) for c in af.SPEC}})
        rows = v.assign(test=v["vehicle_id"] >= 320)
        g1 = af._peer_features(rows, v)
        v2 = v.copy()
        v2.loc[5, "repaired"] = 1 - v2.loc[5, "repaired"]          # eigenes Label kippen …
        g2 = af._peer_features(rows.assign(repaired=v2["repaired"]), v2)
        self.assertTrue(np.allclose(g1.loc[5].to_numpy(), g2.loc[5].to_numpy()))   # … eigene Merkmale bleiben gleich
        self.assertFalse(np.allclose(g1.loc[rows["test"]].to_numpy(), g2.loc[rows["test"]].to_numpy()))

    def test_artifacts_and_business_case(self):
        from graph_ml import business_case as bc
        from graph_ml import evaluation as ev
        ib = ev.load("installed_base")
        self.assertEqual((ib["kind"], ib["dataset"]), ("binary", "scania"))
        auprc = {e["id"]: e["metrics"]["auprc"] for e in ib["engines"]}
        self.assertGreater(auprc[ib["best"]], auprc["rule"])
        self.assertIn("gnn", auprc)
        sp = ev.load("spare_parts_pricing")
        self.assertTrue(sp["lower_is_better"])
        w = sp["whatif"]
        e = w["engines"][sp["best"]]
        a = bc.compute("spare_parts_pricing", w, {"uplift": 3, "loss": 30, "cost": 50, "periods": 2}, sp["best"])["annual"]
        self.assertAlmostEqual(a, (e["rev"] * 0.03 * 0.7 - e["n"] * 50) * 2, places=2)        # identisch zur JS-Formel
        sc = ev.load("service_contracts")
        self.assertEqual(sc["vorhersagekraft"]["status"], "na")
        self.assertTrue(sc["aftersales"]["contract_segments"])
        ov = bc.overview()["cases"]
        self.assertEqual(ov["service_contracts"]["euro"]["status"], "na")
        self.assertEqual(ov["installed_base"]["dataset"], "scania")
        self.assertIsNone(ov["spare_parts_pricing"]["scenarios"]["realistisch"]["added"])

    def test_stage3_on_scania(self):
        from graph_ml.data_loader import get_registry
        res = get_registry().checks("scania", 3)
        by = {u["id"]: u for u in res["use_cases"]}
        self.assertEqual(by["installed_base"]["ampel"]["vorhersagekraft"]["status"], "ok")
        self.assertEqual(by["service_contracts"]["ampel"]["datenreife"]["status"], "warn")
        bound = {f["field"]: (f["binding"] or {}).get("column") for f in by["installed_base"]["fields"]}
        self.assertEqual(bound["serial_number"], "vehicle_id")
        self.assertEqual(bound["service_event"], "repair_id")
        aw = {u["id"]: u for u in get_registry().checks("adventureworks_data", 3)["use_cases"]}
        self.assertNotEqual((next(f for f in aw["installed_base"]["fields"] if f["field"] == "service_event")["binding"] or {})
                            .get("column"), "WorkOrderID")                        # Fertigungsauftrag ≠ Serviceauftrag

class QualityTest(unittest.TestCase):
    """Prompt 7: Qualitäts-Frühwarnung & Root Cause, Lieferantenregress (AdventureWorks), Garantiebetrug (nur Struktur)."""

    def test_past_stats_only_count_finished_cases(self):
        import pandas as pd

        from graph_ml import quality as q
        key = pd.Series(["a", "a", "a", "b"])
        avail = pd.Series(pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-01-01"]))
        y = pd.Series([1, 0, 1, 1])
        s, n = q.past_stats(key, avail, y, pd.Series(["a", "a", "b"]), pd.Series(pd.to_datetime(["2025-02-01", "2025-12-01", "2025-01-01"])))
        self.assertEqual(list(n), [1, 3, 0])                    # am selben Tag fertig = noch unbekannt
        self.assertEqual(list(s), [1, 2, 0])

    def test_supplier_features_ignore_future_labels(self):
        from graph_ml import quality as q
        t, prod = q._aw()
        p = q.supplier_lines(t, prod)
        f1 = q.supplier_features(p, t["BillOfMaterials"])
        p2 = p.copy()
        late = p2["DueDate"] >= q.SUP_TEST
        p2.loc[late, "y"] = 1 - p2.loc[late, "y"]               # Zukunft verändern …
        f2 = q.supplier_features(p2, t["BillOfMaterials"])
        early = (p["OrderDate"] < q.SUP_TEST).to_numpy()
        cols = [c for c in q.SUP_BASE + q.SUP_GRAPH]
        self.assertTrue(f1.loc[early, cols].fillna(-1).equals(f2.loc[early, cols].fillna(-1)))   # … Merkmale davor bleiben gleich

    def test_artifacts_and_ampel(self):
        from graph_ml import business_case as bc
        from graph_ml import evaluation as ev
        from graph_ml.data_loader import get_registry
        for uc in ("quality_root_cause", "supplier_recourse"):
            r = ev.load(uc)
            self.assertEqual((r["kind"], r["dataset"]), ("binary", "adventureworks_data"))
            self.assertEqual({e["id"] for e in r["engines"]} >= {"rule", "lgbm", "lgbm_graph", "gnn"}, True)
        sup = ev.load("supplier_recourse")
        au = {e["id"]: e["metrics"]["auprc"] for e in sup["engines"]}
        self.assertGreater(au[sup["best"]], au["rule"])
        self.assertIn("warranty_fraud", ev.NOT_MEASURABLE)
        ov = bc.overview()["cases"]
        self.assertEqual(ov["warranty_fraud"]["euro"]["status"], "na")
        by = {u["id"]: u for u in get_registry().checks("adventureworks_data", 3)["use_cases"]}
        rq = next(f for f in by["supplier_recourse"]["fields"] if f["field"] == "rejected_qty")
        self.assertEqual(rq["binding"]["column"], "RejectedQty")
        self.assertEqual(by["supplier_recourse"]["ampel"]["vorhersagekraft"]["status"], "ok")

class MaintenanceTest(unittest.TestCase):
    """Prompt 8: Predictive Maintenance (C-MAPSS), Ersatzteilbedarf & Bestände (WWI), First-Time-Fix (nur Struktur)."""

    def test_peer_normal_values_only_from_training_engines(self):
        import numpy as np

        from graph_ml import maintenance as mt
        c, rows = mt.pdm_rows(log=lambda m: None)
        f1 = mt.pdm_features(c, rows)
        c2 = c.copy()
        test_e = set(rows.loc[rows["test"], "engine_id"])
        c2.loc[c2["engine_id"].isin(test_e), mt.SENSORS] += 1000.0       # Test-Triebwerke verändern …
        f2 = mt.pdm_features(c2, rows)
        tr = (~rows["test"]).to_numpy()
        g = [x for x in f1.columns if x.startswith("g_")]
        self.assertTrue(np.allclose(f1.loc[tr, g].to_numpy(), f2.loc[tr, g].to_numpy()))   # … Peer-Werte der anderen bleiben gleich
        self.assertEqual(set(rows.loc[rows["test"], "engine_id"]) & set(rows.loc[~rows["test"], "engine_id"]), set())

    def test_artifacts_and_ampel(self):
        from graph_ml import business_case as bc
        from graph_ml import evaluation as ev
        from graph_ml.data_loader import get_registry
        pdm = ev.load("predictive_maintenance")
        self.assertEqual((pdm["kind"], pdm["dataset"]), ("binary", "cmapss"))
        au = {e["id"]: e["metrics"]["auprc"] for e in pdm["engines"]}
        self.assertGreater(au[pdm["best"]], au["rule"])
        dem = ev.load("spare_parts_demand")
        self.assertEqual((dem["whatif"]["kind"], dem["dataset"]), ("forecast", "wwi"))
        self.assertIn("first_time_fix", ev.NOT_MEASURABLE)
        ov = bc.overview()["cases"]
        self.assertEqual(ov["first_time_fix"]["euro"]["status"], "na")
        self.assertIsNotNone(ov["spare_parts_demand"]["euro"].get("annual"))
        by = {u["id"]: u for u in get_registry().checks("cmapss", 3)["use_cases"]}
        self.assertEqual(by["predictive_maintenance"]["ampel"]["vorhersagekraft"]["status"], "ok")
        fe = next(f for f in by["predictive_maintenance"]["fields"] if f["field"] == "failure_event")
        self.assertIsNotNone(fe["binding"])

class PipelineTest(unittest.TestCase):
    """Prompt 9: Gewinnwahrscheinlichkeit & Umsatzprognose (WWI), Service-Triage (Struktur), Abgleich der Altfälle."""

    def test_labels_only_after_window_and_past_features(self):
        from graph_ml import sales_pipeline as sp
        b, t = sp.backorders(log=lambda m: None)
        end = max(b["OrderDate"].max(), b["InvoiceDate"].max())
        self.assertTrue((b["avail"] <= end).all())                    # nur Aufträge mit feststehendem Label
        f1 = sp.features(b, t)
        b2 = b.copy()
        late = b2["avail"] >= sp.TEST_FROM
        b2.loc[late, "y"] = 1 - b2.loc[late, "y"]                     # Zukunft verändern …
        f2 = sp.features(b2, t)
        early = (b["OrderDate"] < sp.TEST_FROM).to_numpy()
        cols = ["c_rate", "g_item_rate", "g_supplier_rate", "g_group_rate"]
        self.assertTrue(f1.loc[early, cols].equals(f2.loc[early, cols]))   # … Quoten davor bleiben gleich

    def test_artifact_and_legacy_reconciliation(self):
        from graph_ml import business_case as bc
        from graph_ml import evaluation as ev
        r = ev.load("win_probability")
        self.assertEqual((r["kind"], r["dataset"]), ("binary", "wwi"))
        fc = r["pipeline"]["forecast"]
        self.assertLess(fc["wape"][r["best"]], fc["wape"]["rule"])
        self.assertEqual(len(fc["weeks"]), len(fc["actual"]))
        self.assertIn("service_triage", ev.NOT_MEASURABLE)
        self.assertIn("AdventureWorks", ev.NOT_MEASURABLE["margin"])
        ov = bc.overview()["cases"]
        self.assertEqual(ov["win_probability"]["dataset"], "wwi")
        self.assertEqual(ov["service_triage"]["euro"]["status"], "na")

class MatrixTest(unittest.TestCase):
    """Prompt 10: Gesamtabgleich Use Cases × Datensätze, Report."""

    def test_matrix_artifact_consistent(self):
        from graph_ml import evaluation as ev
        from graph_ml import matrix as mx
        from graph_ml import usecases
        m = mx.load()
        self.assertEqual(len(m["use_cases"]), len(usecases.USE_CASES))
        ids = {d["id"] for d in m["datasets"]}
        self.assertTrue({"online_retail_ii", "adventureworks_data", "wwi", "scania", "cmapss"} <= ids)
        for u in m["use_cases"]:
            homes = [d for d in ids if (m["cells"].get(f"{u['id']}|{d}") or {}).get("home")]
            self.assertEqual(len(homes), 0 if u["home"] is None else 1, u["id"])    # genau ein Heimat-Datensatz
            if u["home"] and ev.load(u["id"]):
                self.assertTrue(u["measured"], u["id"])
        self.assertEqual(sum(m["counts"].values()), len(m["use_cases"]))

    def test_report_contains_matrix(self):
        from app import app
        c = app.test_client()
        r = c.get("/graph_ml/report")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Gesamtabgleich".encode(), r.data)
        self.assertEqual(c.get("/graph_ml/api/matrix").status_code, 200)

class ClassifierScanTest(unittest.TestCase):
    """Schritt 3: Data-Classifier-Analyse – manuell gepflegte Felder, die ML vorschlagen kann."""

    def test_single_field_leak_is_detected(self):
        import numpy as np
        import pandas as pd

        from graph_ml import classifier_scan as cs
        rng = np.random.default_rng(0)
        y = rng.choice(["a", "b"], 300)
        X = pd.DataFrame({"folge": pd.Series(np.where(y == "a", "x", "z")).astype("category"), "rauschen": rng.random(300)})
        rules = cs.single_field_rules(X, y, cs._folds(y))
        self.assertEqual(rules[0]["feature"], "folge")
        self.assertGreaterEqual(rules[0]["ba"], cs.LEAK_BA)
        self.assertLess(next(r for r in rules if r["feature"] == "rauschen")["ba"], 0.7)

    def test_group_share_never_sees_own_label(self):
        import numpy as np
        import pandas as pd

        from graph_ml import classifier_scan as cs
        keys = pd.Series(range(6))
        groups = [{"name": "g", "edges": pd.DataFrame({"e": [0, 1, 2, 3, 4, 5], "g": [1, 1, 1, 2, 2, 2]})}]
        y = np.array(["1", "1", "0", "0", "0", "1"])
        train = np.ones(6, bool)
        a = cs._group_share(groups, keys, y, train, ["0", "1"])
        y2 = y.copy()
        y2[0] = "0"                                                  # eigenes Label kippen …
        b = cs._group_share(groups, keys, y2, train, ["0", "1"])
        self.assertTrue(np.allclose(a[0], b[0]))                    # … eigene Nachbarschaft bleibt gleich
        self.assertAlmostEqual(a[0, 1], 0.5)                        # Geschwister 1 („1“) und 2 („0“)

    def test_artifacts_and_spare_part_example(self):
        from graph_ml import classifier_scan as cs
        aw = cs.load("adventureworks_data")
        self.assertIsNotNone(aw)
        et = aw["results"][0]
        self.assertTrue(et["special"])
        self.assertEqual(et["id"], "Product.FinishedGoodsFlag@BOM")
        self.assertEqual(et["labeled"], 325)
        self.assertIn("SpecialOfferProduct", et["leak_tables"])           # Folgetabelle erkannt und entfernt
        self.assertTrue(all("SpecialOfferProduct" not in f["feature"] for f in et["importance"]))
        self.assertAlmostEqual(et["alpha_95"], 1 - 0.05 ** (1 / et["n_positive"]), places=5)
        e = cs.euro(et)
        m = next(x for x in et["engines"] if x["id"] == et["best"])["metrics"]
        v = et["value"]
        labor = v["entries_per_year"] * et["minutes"] / 60 * 60 * (m["auto_positive"] + m["auto_negative"])
        self.assertAlmostEqual(e["labor_year"], round(labor), places=0)
        ids = {r["id"] for r in aw["results"]}
        self.assertFalse(any(PII in i for i in ids for PII in ("Gender", "MaritalStatus", "BirthDate")))
        for ds in ("scania", "cmapss"):                                      # Sensordaten: keine manuellen Felder
            self.assertEqual(cs.load(ds)["n_likely"], 0)

    def test_route(self):
        from app import app
        c = app.test_client()
        r = c.get("/graph_ml/api/classifier/adventureworks_data").get_json()
        self.assertTrue(r["available"])
        self.assertFalse(c.get("/graph_ml/api/classifier/unbekannt").get_json()["available"])

class DataPredictionTest(unittest.TestCase):
    """Sammel-Kachel „Data Prediction“: je manuell gepflegtem Feld ein Use Case, genau eine Kachel."""

    def test_tile_counts_field_use_cases(self):
        from graph_ml import business_case as bc
        from graph_ml import data_prediction as dpm
        from graph_ml import evaluation as ev
        from graph_ml.data_loader import get_registry
        r = ev.load("data_prediction")
        self.assertEqual(r["kind"], "classifier")
        aw = r["data_prediction"]["datasets"]["adventureworks_data"]
        self.assertGreater(aw["count"], 0)
        self.assertTrue(aw["fields"][0]["special"])                              # Ersatzteil zuerst
        self.assertFalse(any(f["inherited"] for f in aw["fields"]))             # übernommene Werte zählen nicht
        w = r["whatif"]
        self.assertAlmostEqual(bc.compute("data_prediction", w, {"rate": 60}, "cascade")["annual"], w["engines"]["cascade"]["hours"] * 60)
        for ds, n in (("adventureworks_data", aw["count"]), ("online_retail_ii", 0)):
            u = next(x for x in get_registry().checks(ds, 3)["use_cases"] if x["id"] == "data_prediction")
            self.assertEqual(u["dp"]["count"], n)
        self.assertEqual(dpm.readiness(None)["status"], "bad")
