"""
Tests für /graph_ml – Aktivierung, Kontrollgruppe, Fallzahl, Next Best Action, Monitoring und Report (Modul 6).

  python -m unittest tests.test_graph_ml_activation -v
"""
from __future__ import annotations

import io
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_ml import activation as act  # noqa: E402


class ControlAndPilotTest(unittest.TestCase):
    def test_control_group_stratified_and_deterministic(self):
        custs = list(range(1000))
        strata = [f"S{c % 5}" for c in custs]
        a = act.assign_control(custs, strata)
        b = act.assign_control(list(reversed(custs)), list(reversed(strata)))
        self.assertEqual(a, b)                                       # fester Seed, reihenfolgeunabhängig
        for s in range(5):
            members = [c for c in custs if c % 5 == s]
            self.assertEqual(sum(c in a for c in members), 40)       # exakt 20 % je Schicht
        self.assertNotEqual(a, act.assign_control(custs, strata, seed=8))

    def test_pilot_size_formula(self):
        r = act.pilot_size(0.5, -0.25, 800, 200)
        z = 1.959964 + 0.841621
        self.assertAlmostEqual(r["mde_abs"], z * math.sqrt(0.25 * (1 / 800 + 1 / 200)), places=3)
        self.assertTrue(r["detectable"])                             # 12,5 Prozentpunkte bei 1.000 Fällen
        small = act.pilot_size(0.5, -0.25, 80, 20)
        self.assertFalse(small["detectable"])
        self.assertGreater(small["n_required"], 100)
        self.assertAlmostEqual(small["n_required"], r["n_required"], delta=1)   # Bedarf hängt nicht von n ab


class CalibrationAndDriftTest(unittest.TestCase):
    def test_isotonic_calibration_fixes_overconfidence(self):
        rng = np.random.default_rng(1)
        true_p = rng.uniform(0, 1, 4000)
        y = (rng.uniform(0, 1, 4000) < true_p).astype(int)
        raw = np.clip(true_p ** 0.3, 0, 1)                           # zu „sicher“
        cal = act.calibrate(raw, y, proba=True)
        self.assertLess(cal["ece_after"], cal["ece_before"] / 2)
        p = act.apply_calibration(cal, raw)
        self.assertLessEqual(p.max(), act.P_MAX)                     # nie 100 %

    def test_psi(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0, 1, 5000)
        self.assertLess(act.psi(a, rng.normal(0, 1, 5000)), 0.02)
        self.assertGreater(act.psi(a, rng.normal(1.2, 1, 5000)), act.PSI_ALARM)
        self.assertEqual(act.psi_status(0.3), "bad")


class NextBestActionTest(unittest.TestCase):
    def test_conflict_resolution(self):
        def L(rows):
            keys = ["cust", "in_list", "ev", "p", "extra"]
            return {"rows": {k: [r[i] for r in rows] for i, k in enumerate(keys)}}

        lists = {"churn": L([(1, 1, 300.0, 0.8, {}), (2, 1, 20.0, 0.3, {}), (3, 0, -5.0, 0.1, {}), (4, 1, 50.0, 0.9, {})]),
                 "lead_scoring": L([(5, 1, 40.0, 0.6, {})]),
                 "cross_selling": L([(1, 1, 500.0, 0.5, {"group_name": "Licht"}), (2, 1, 80.0, 0.4, {"group_name": "Kerzen"})])}
        reco = {"rows": {"cust": [2, 3], "ev": [30.0, 12.0], "p": [[0.3], [0.2]], "names": [["A"], ["B"]]}}
        nba = act.build_nba(lists, reco, control={4})
        r = dict(zip(nba["rows"]["cust"], nba["rows"]["action"]))
        self.assertEqual(r[1], "retention")          # Bindung vor Verkauf, obwohl Cross-Selling mehr Wert hätte
        self.assertEqual(r[2], "cross")              # höchster erwarteter Wert (80 > 30 > 20)
        self.assertEqual(r[3], "reco")
        self.assertEqual(r[4], "control")            # Kontrollgruppe bleibt unberührt
        self.assertEqual(r[5], "onboarding")
        i1 = nba["rows"]["cust"].index(1)
        self.assertTrue(any("Regel 2" in c for c in nba["rows"]["conflicts"][i1]))


@unittest.skipUnless(act.load(), "Aktivierung nicht vorberechnet")
class ArtifactAndRoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as appmod
        cls.c = appmod.app.test_client()

    def test_lists_have_reasons_and_calibration(self):
        s = act.summary()
        self.assertEqual(set(s["lists"]), set(act.LIST_UCS))
        for uc in act.LIST_UCS:
            rows = act.rows(uc, "list", 0, 5)["rows"]
            self.assertTrue(rows, uc)
            self.assertTrue(all(0 < r["p"] < 1 and r["reasons"] for r in rows), uc)
            self.assertFalse(any(r["control"] for r in rows), uc)

    def test_export_excludes_control_and_rejected(self):
        recs, ctrl = act.export_records("churn")
        self.assertTrue(recs and ctrl)
        self.assertFalse(any(r["control_group"] for r in recs))
        first = recs[0]["record_id"].split("|", 1)[1]
        recs2, _ = act.export_records("churn", {f"churn|{first}": {"status": "verworfen", "reason": "Test"}})
        self.assertEqual(len(recs2), len(recs) - 1)
        csv = act.to_csv(recs)
        self.assertIn("expected_value_eur", csv.splitlines()[0])

    def test_routes(self):
        c = self.c
        self.assertEqual(c.get("/graph_ml/api/activation").get_json()["available"], True)
        r = c.post("/graph_ml/api/activation/preview", json={"uc": "churn", "crm": "hubspot"}).get_json()
        self.assertFalse(r["sent"])
        r = c.post("/graph_ml/api/activation/export", json={"uc": "churn", "format": "xlsx"})
        self.assertEqual(r.status_code, 200)
        from openpyxl import load_workbook
        self.assertIn("Kontrollgruppe", load_workbook(io.BytesIO(r.data)).sheetnames)
        self.assertEqual(c.get("/graph_ml/api/monitoring").status_code, 200)
        self.assertIn(b"Readiness-Report", c.get("/graph_ml/report").data)
        wb = load_workbook(io.BytesIO(c.get("/graph_ml/report.xlsx").data))
        self.assertTrue({"Ampel", "Reparaturplan", "Business Case"} <= set(wb.sheetnames))


class DataQualityTest(unittest.TestCase):
    def test_monthly_quality_finds_december_duplicates(self):
        from graph_ml.data_loader import get_registry
        months = {m["month"]: m for m in act.data_quality_monthly(get_registry().raw_online_retail())}
        self.assertEqual(len(months), 25)
        self.assertGreater(months["2010-12"]["dup"], 0.3)            # Dubletten-Häufung, die PB-06 bereinigt
        self.assertTrue(any(a["metric"] == "dup" and a["month"] == "2010-12" for a in act.dq_alarms(list(months.values()))))


if __name__ == "__main__":
    unittest.main()
