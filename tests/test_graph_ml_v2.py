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
