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


if __name__ == "__main__":
    unittest.main()
