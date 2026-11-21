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


if __name__ == "__main__":
    unittest.main()
