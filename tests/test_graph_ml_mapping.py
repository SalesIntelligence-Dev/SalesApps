"""
Tests für /graph_ml – Konnektoren (Modul 1) und Graph-Mapping (Modul 2).

  python -m unittest tests.test_graph_ml_mapping -v
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_ml.data_loader import get_registry  # noqa: E402
from graph_ml.graph_mapping import build_mapping, build_schema_mapping, name_match  # noqa: E402

EXISTING_ROUTES = [
    "/", "/forecasting", "/forecasting_model", "/computer_vision_quality", "/margenoptimierung",
    "/cross-selling", "/konfiguration", "/retrieval", "/lead-scoring", "/recommender",
    "/opportunity", "/next-action", "/churn", "/api/health",
]


class OnlineRetailMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = get_registry()
        cls.mapping = cls.reg.mapping("online_retail_ii")
        cls.facts = cls.reg.get("online_retail_ii").facts

    def test_raw_facts_match_source(self):
        self.assertEqual(self.facts["rows"], 1_067_371)
        self.assertEqual(self.facts["rows_without_customer"], 243_007)
        self.assertAlmostEqual(self.facts["share_without_customer"], 0.2277, places=3)

    def test_relations_detected(self):
        rel = {r["id"]: r for r in self.mapping["relations"]}
        for rid in ("invoice_lines.invoice->invoices.invoice",
                    "invoice_lines.stock_code->products.stock_code",
                    "invoices.customer_id->customers.customer_id",
                    "invoices.country->countries.country"):
            self.assertIn(rid, rel)
        self.assertAlmostEqual(rel["invoice_lines.invoice->invoices.invoice"]["linkage_rate"], 1.0)
        self.assertEqual(rel["invoices.customer_id->customers.customer_id"]["status"], "warn")

    def test_multi_hop_coverage_equals_missing_customer_share(self):
        path = next(p for p in self.mapping["paths"] if p["via"] == ["invoice_lines", "invoices", "customers"])
        self.assertAlmostEqual(path["coverage"], 1 - self.facts["share_without_customer"], places=3)

    def test_storno_reference_reported_missing(self):
        missing = [r for r in self.mapping["relations"] if r["kind"] == "missing"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["rows"], self.facts["storno_invoices"])

    def test_roles(self):
        tables = {t["name"]: t for t in self.mapping["tables"]}
        self.assertEqual(tables["invoices"]["time_columns"], ["invoice_date"])
        desc = next(c for c in tables["products"]["columns"] if c["name"] == "description")
        self.assertIn("Freitext", desc["roles"])
        self.assertEqual(tables["invoice_lines"]["primary_key"], "line_id")

    def test_subgraph_is_bounded(self):
        cid = self.reg.sample_customers()[0]["customer_id"]
        sg = self.reg.customer_subgraph(cid)
        self.assertLessEqual(len(sg["nodes"]), 150)
        ids = {n["data"]["id"] for n in sg["nodes"]}
        for e in sg["edges"]:
            self.assertIn(e["data"]["source"], ids)
            self.assertIn(e["data"]["target"], ids)


class SyntheticMappingTest(unittest.TestCase):
    def test_fk_linkage_counts_nulls_and_orphans(self):
        kunden = pd.DataFrame({"kunde_id": np.arange(1, 101), "land": ["DE"] * 100})
        ids = np.r_[np.arange(1, 81), [np.nan] * 10, np.arange(500, 510)].astype(float)
        auftraege = pd.DataFrame({"auftrag_id": np.arange(1, 101), "kunde_id": ids,
                                  "datum": pd.date_range("2024-01-01", periods=100)})
        m = build_mapping({"kunden": kunden, "auftraege": auftraege}, dataset_id="t")
        rel = next(r for r in m["relations"] if r["from_table"] == "auftraege")
        self.assertEqual(rel["to_table"], "kunden")
        self.assertEqual(rel["null_rows"], 10)
        self.assertEqual(rel["orphan_rows"], 10)
        self.assertAlmostEqual(rel["linkage_rate"], 0.8)

    def test_no_false_fk_on_surrogate_ranges(self):
        # customer_id-Werte liegen im Bereich von line_id – darf trotzdem kein FK auf lines sein
        lines = pd.DataFrame({"line_id": np.arange(1, 20_001), "menge": 1})
        orders = pd.DataFrame({"order_id": np.arange(1, 101), "customer_id": np.arange(1, 101)})
        m = build_mapping({"lines": lines, "orders": orders}, dataset_id="t")
        self.assertFalse(any(r["to_table"] == "lines" for r in m["relations"]))

    def test_name_match(self):
        self.assertEqual(name_match("customer_id", "customer_id", "customers"), 1.0)
        self.assertGreaterEqual(name_match("CustomerID", "ID", "Sales.Customer"), 0.85)
        self.assertEqual(name_match("quantity", "customer_id", "customers"), 0.0)


class SchemaMappingTest(unittest.TestCase):
    def test_adventureworks_stage1(self):
        m = get_registry().mapping("adventureworks")
        self.assertEqual(m["mode"], "schema")
        self.assertEqual(len(m["tables"]), 68)
        self.assertEqual(len(m["relations"]), 88)
        self.assertTrue(all(r["linkage_rate"] is None for r in m["relations"]))

    def test_sap_style_schema_without_declared_keys(self):
        schema = {"tables": [
            {"name": "VBAK", "columns": [{"name": "VBELN", "type": "string"}, {"name": "KUNNR", "type": "string"},
                                         {"name": "ERDAT", "type": "datetime"}]},
            {"name": "KNA1", "columns": [{"name": "KUNNR", "type": "string"}, {"name": "NAME1", "type": "string"}]},
        ], "foreign_keys": []}
        m = build_schema_mapping(schema, dataset_id="sap")
        pks = {t["name"]: t["primary_key"] for t in m["tables"]}
        self.assertEqual(pks, {"VBAK": "VBELN", "KNA1": "KUNNR"})
        self.assertTrue(any(r["from_table"] == "VBAK" and r["to_table"] == "KNA1" for r in m["relations"]))


class RoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def test_graph_ml_page_and_api(self):
        self.assertEqual(self.client.get("/graph_ml").status_code, 200)
        sysinfo = self.client.get("/graph_ml/api/system?health=0").get_json()
        self.assertEqual({c["id"] for c in sysinfo["chips"]}, {"cpu", "gpu", "kumo", "tabpfn"})
        ids = [d["id"] for d in self.client.get("/graph_ml/api/datasets").get_json()]
        self.assertEqual(ids[0], "online_retail_ii")
        self.assertIn("adventureworks", ids)
        self.assertEqual(len(self.client.get("/graph_ml/api/usecases").get_json()), 23)

    def test_upload_roundtrip(self):
        csv = b"kunde_id;name\n1;A\n2;B\n"
        r = self.client.post("/graph_ml/api/upload", data={"files": (io.BytesIO(csv), "kunden.csv")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 201)
        ds_id = r.get_json()["id"]
        m = self.client.get(f"/graph_ml/api/datasets/{ds_id}/mapping").get_json()
        self.assertEqual(m["tables"][0]["primary_key"], "kunde_id")
        self.assertEqual(self.client.delete(f"/graph_ml/api/datasets/{ds_id}").status_code, 200)

    def test_existing_routes_unchanged(self):
        for route in EXISTING_ROUTES:
            with self.subTest(route=route):
                self.assertEqual(self.client.get(route).status_code, 200)


if __name__ == "__main__":
    unittest.main()
