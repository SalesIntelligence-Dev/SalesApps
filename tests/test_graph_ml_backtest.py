"""
Tests für /graph_ml – Rückblick und Baseline-Vergleich (Modul 4, Stufe 3).

  python -m unittest tests.test_graph_ml_backtest -v
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_ml import backtest as bt  # noqa: E402
from graph_ml import evaluation as ev  # noqa: E402
from graph_ml.engines import engine_catalog, make_engines  # noqa: E402
from graph_ml.engines.cpu import LGBMEngine  # noqa: E402


def _truncated(b: bt.Base, T: pd.Timestamp) -> bt.Base:
    """Basis ohne jedes Ereignis ab T – Merkmale zum Stichtag T dürfen sich dadurch nicht ändern."""
    c = copy.copy(b)
    c.lines = b.lines[b.lines["time"] < T]
    c.inv = b.inv[b.inv["time"] < T]
    c.storno = b.storno[b.storno["time"] < T]
    return c


class LeakageTest(unittest.TestCase):
    def test_features_ignore_future_events(self):
        b = bt.get_base()
        for T in (pd.Timestamp("2010-09-01"), bt.T_TEST):
            full = bt.customer_features(b, T, bt.Audit(), "t")
            cut = bt.customer_features(_truncated(b, T), T, bt.Audit(), "t")
            pd.testing.assert_frame_equal(full, cut.reindex(full.index))
            gf = bt.graph_customer_features(b, T, full, bt.Audit(), "t")
            gc = bt.graph_customer_features(_truncated(b, T), T, full, bt.Audit(), "t")
            pd.testing.assert_frame_equal(gf, gc)

    def test_audit_rejects_future_feature_time(self):
        with self.assertRaises(AssertionError):
            bt.Audit().check("x", pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02"))
        with self.assertRaises(AssertionError):     # Trainings-Label darf nicht über den Test-Stichtag reichen
            bt.Audit().check("x", pd.Timestamp("2011-07-01"), None, pd.Timestamp("2011-09-29"))

    def test_training_labels_end_before_test_cutoff(self):
        task = bt.get_task("churn")
        for T in task.train["cutoff"].unique():
            self.assertLessEqual(pd.Timestamp(T) + pd.Timedelta(days=bt.HORIZON), bt.T_TEST)
        self.assertTrue((task.test["cutoff"] == bt.T_TEST).all())
        self.assertGreater(len(task.audit.rows), 10)
        for row in task.audit.rows:
            if row["max_feature_time"]:
                self.assertLess(pd.Timestamp(row["max_feature_time"]), pd.Timestamp(row["cutoff"]))


class EngineTest(unittest.TestCase):
    def test_interface(self):
        for e in engine_catalog():
            for k in ("id", "name", "klass", "kind", "available", "reason"):
                self.assertIn(k, e)
        task = bt.get_task("churn")
        for eng in make_engines(task):
            ok, why = eng.available()
            self.assertIsInstance(ok, bool)
            self.assertTrue(hasattr(eng, "fit") and hasattr(eng, "predict") and hasattr(eng, "explain"))

    def test_lightgbm_deterministic(self):
        task = bt.get_task("lead_scoring")
        a = LGBMEngine().fit(task).predict(task)
        b = LGBMEngine().fit(task).predict(task)
        np.testing.assert_allclose(a, b)

    def test_vorhersagekraft_thresholds(self):
        self.assertEqual(ev.vorhersagekraft(0.25, [0.05, 0.4])["status"], "ok")
        self.assertEqual(ev.vorhersagekraft(0.10, [0.02, 0.2])["status"], "warn")
        self.assertEqual(ev.vorhersagekraft(0.10, [-0.02, 0.2])["status"], "bad")    # nicht signifikant
        self.assertEqual(ev.vorhersagekraft(0.03, [0.01, 0.05])["status"], "bad")
        self.assertEqual(ev.vorhersagekraft(None, None)["status"], "na")


class ArtifactTest(unittest.TestCase):
    def test_all_use_cases_precomputed(self):
        for uc in ev.UC_ORDER:
            r = ev.load(uc)
            self.assertIsNotNone(r, uc)
            best = next(e for e in r["engines"] if e["id"] == r["best"])
            self.assertIsNotNone(best["metrics"][r["main_metric"]])
            self.assertIn(r["vorhersagekraft"]["status"], ("ok", "warn", "bad"))

    def test_stage3_fills_vorhersagekraft(self):
        from graph_ml.data_loader import Registry
        res = Registry().checks("online_retail_ii", 3)
        churn = next(u for u in res["use_cases"] if u["id"] == "churn")
        self.assertNotEqual(churn["ampel"]["vorhersagekraft"]["status"], "pending")
        nba = next(u for u in res["use_cases"] if u["id"] == "next_best_action")
        self.assertEqual(nba["ampel"]["vorhersagekraft"]["status"], "na")

    def test_api(self):
        from app import app
        c = app.test_client()
        r = c.get("/graph_ml/api/backtest").get_json()
        self.assertEqual(set(r["use_cases"]), set(ev.UC_ORDER))
        self.assertEqual(c.get("/graph_ml/api/backtest/churn").status_code, 200)
        self.assertEqual(c.get("/graph_ml/api/backtest/margin").status_code, 404)


if __name__ == "__main__":
    unittest.main()
