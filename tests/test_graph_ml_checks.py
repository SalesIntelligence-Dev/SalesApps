"""
Tests für /graph_ml – Readiness-Checks (Modul 3), Stufe 1 und 2.

  python -m unittest tests.test_graph_ml_checks -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_ml.checks import run_checks  # noqa: E402
from graph_ml.data_loader import Registry  # noqa: E402


def _forbid_data(reg: Registry) -> Registry:
    """Jeder Zugriff auf Datenzeilen schlägt fehl."""
    def boom(*_a, **_k):
        raise AssertionError("Datenzeilen gelesen")
    reg.raw_online_retail = boom
    reg.tables = boom
    return reg


class ChecksTest(unittest.TestCase):
    def test_customer_linkage_about_77_percent(self):
        res = Registry().checks("online_retail_ii", 2)
        churn = next(u for u in res["use_cases"] if u["id"] == "churn")
        structure = next(c for c in churn["checks"] if c["dim"] == "structure")
        value = next(m["value"] for m in structure["metrics"] if m["label"] == "Positionen mit Kundennummer")
        self.assertEqual(value, "77,2 %")

    def test_stage1_reads_no_data_rows(self):
        reg = _forbid_data(Registry())
        res = reg.checks("online_retail_ii", 1)
        self.assertEqual(res["stage_used"], 1)
        self.assertTrue(all(u["readiness"]["provisional"] for u in res["use_cases"]))
        aw = _forbid_data(Registry()).checks("adventureworks", 1)
        self.assertEqual(aw["stage_used"], 1)

    def test_stage2_uses_only_profile(self):
        reg = _forbid_data(Registry())
        profile = reg.profile("online_retail_ii")          # aus dem Artefakt, ohne Datenzugriff
        res = run_checks(stage=2, view={}, profile=profile)
        self.assertEqual(res["stage_used"], 2)
        self.assertIsNotNone(res["privacy"])
        self.assertFalse(res["privacy"]["contains_rows"])

    def test_expected_traffic_lights(self):
        res = Registry().checks("online_retail_ii", 2)
        st = {u["id"]: u["readiness"]["status"] for u in res["use_cases"]}
        self.assertEqual(st["warranty_root_cause"], "bad")   # Seriennummer/Charge/Garantiefälle fehlen
        self.assertEqual(st["margin"], "bad")                # Einstandspreis fehlt
        self.assertEqual(st["next_best_action"], "warn")     # höchstens Gelb ohne Reaktionsdaten
        self.assertIn(st["churn"], ("ok", "warn"))

    def test_schema_dataset_is_capped_at_stage1(self):
        res = Registry().checks("adventureworks", 3)
        self.assertEqual(res["stage_used"], 1)
        self.assertTrue(res["downgraded"])
        margin = next(u for u in res["use_cases"] if u["id"] == "margin")
        self.assertNotEqual(margin["checks"][0]["status"], "bad")   # StandardCost vorhanden


if __name__ == "__main__":
    unittest.main()
