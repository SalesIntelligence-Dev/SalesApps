"""
Tests für /graph_ml – Fix-Playbooks, Business Case und Ampel aus drei Dimensionen (Modul 5).

  python -m unittest tests.test_graph_ml_business -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_ml import business_case as bc  # noqa: E402
from graph_ml import playbooks as pb  # noqa: E402


class BusinessCaseTest(unittest.TestCase):
    def test_pdf_example(self):
        """2.000 Kunden, 150 Abwanderer à 20.000 €, Top-10-%-Liste enthält 60, 25 % gerettet → 300.000 €."""
        r = bc.intervention(value_in_list=60 * 20_000, n_list=200, rate=0.25, margin=1.0, cost_per_contact=0,
                            periods=1)
        self.assertEqual(r["annual"], 300_000)

    def test_list_compute(self):
        w = {"kind": "list", "k": list(range(1, 31)), "n": [i * 20 for i in range(1, 31)], "positives": 150,
             "value_total": 3_000_000, "currency": "EUR",
             "engines": {"m": {"tp": [6 * i for i in range(1, 31)], "value": [120_000.0 * i for i in range(1, 31)]}}}
        p = {"list_share": 10, "rate": 25, "cost": 0, "margin": 100, "periods": 1}
        self.assertAlmostEqual(bc.compute("churn", w, p, "m")["annual"], 300_000)
        p["cost"] = 100        # 200 Kontakte × 100 € Kosten
        self.assertAlmostEqual(bc.compute("churn", w, p, "m")["annual"], 280_000)
        cons = bc.compute("churn", w, p, "m", scen=bc.SCENARIOS["konservativ"])
        self.assertLess(cons["annual"], 280_000)

    def test_euro_thresholds(self):
        self.assertEqual(bc.euro_status(150_000)["status"], "ok")
        self.assertEqual(bc.euro_status(50_000)["status"], "warn")
        self.assertEqual(bc.euro_status(10_000)["status"], "bad")
        self.assertEqual(bc.euro_status(None)["status"], "na")

    def test_gesamt_is_weakest_dimension(self):
        u = {"ampel": {"datenreife": {"status": "ok"}, "vorhersagekraft": {"status": "warn"}}}
        bc.merge_ampel(u, {"euro": {"status": "bad"}})
        self.assertEqual(u["ampel"]["gesamt"]["status"], "bad")
        u = {"ampel": {"datenreife": {"status": "ok"}, "vorhersagekraft": {"status": "na"}}}
        bc.merge_ampel(u, {"euro": {"status": "ok"}})
        self.assertEqual(u["ampel"]["gesamt"]["status"], "ok")

    def test_overview_from_artifacts(self):
        ov = bc.overview()
        self.assertIn("churn", ov["cases"])
        c = ov["cases"]["churn"]
        self.assertIn(c["euro"]["status"], ("ok", "warn", "bad"))
        self.assertEqual(ov["cases"]["margin"]["euro"]["status"], "na")
        self.assertTrue(ov["next_step"]["text"])


class PlaybookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from graph_ml.data_loader import get_registry
        cls.raw = get_registry().raw_online_retail()

    def test_raw_untouched_and_idempotent(self):
        n_raw = len(self.raw)
        steps = [{"id": "PB-04"}, {"id": "PB-06"}, {"id": "PB-03"}, {"id": "PB-07"}]
        once, _ = pb.run_pipeline(self.raw, steps)
        twice, eff = pb.run_pipeline(once, steps)
        self.assertEqual(len(self.raw), n_raw)                    # Rohdaten v0 bleiben unverändert
        self.assertEqual(len(once), len(twice))
        self.assertTrue(all(e["rows_removed"] == 0 for e in eff))

    def test_order_and_variant_id(self):
        a = pb.normalize_steps([{"id": "PB-08"}, {"id": "PB-06"}, {"id": "PB-09"}])
        self.assertEqual([s["id"] for s in a], ["PB-06", "PB-08"])   # feste Reihenfolge, PB-09 ist keine Datenänderung
        self.assertEqual(pb.variant_id([{"id": "PB-06"}, {"id": "PB-08"}]),
                         pb.variant_id([{"id": "PB-08"}, {"id": "PB-06"}]))

    def test_storno_matching_and_currency(self):
        df, eff = pb.run_pipeline(self.raw, [{"id": "PB-02"}, {"id": "PB-08"}])
        e = {x["id"]: x for x in eff}
        self.assertGreater(e["PB-02"]["match_rate"], 0.6)
        self.assertAlmostEqual(float(df["price_eur"].iloc[0]), float(df["price_gbp"].iloc[0]) * 1.15, places=3)

    def test_data_request_covers_blocked_use_cases(self):
        from graph_ml.data_loader import get_registry
        req = pb.data_request(get_registry().checks("online_retail_ii", 2))
        ids = {s["use_case"] for s in req["sections"]}
        self.assertTrue({"margin", "warranty_root_cause"} <= ids)
        self.assertIn("Einstandspreis", req["markdown"])


if __name__ == "__main__":
    unittest.main()
