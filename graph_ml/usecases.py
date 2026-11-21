"""
Use-Case-Bibliothek (Grundgerüst)
=================================
Die 9 Vorlagen aus GRAPH_ML_PROMPTS.md, Abschnitt 0.4. Prompt 2 ergänzt je
Vorlage die benötigten Tabellen/Felder, Synonyme und Mindestfallzahlen.
"""
from __future__ import annotations

USE_CASES: list[dict] = [
    {"id": "churn", "name": "Churn & Retention", "area": "Sales", "lighthouse": True,
     "app_route": "/churn", "app_name": "Churn & Retention",
     "label": "Aktiver Kunde kauft in den nächsten 90 Tagen nicht",
     "baseline": "„seit 60 Tagen keine Bestellung“", "lever": "Umsatz halten", "max_stage": 3},
    {"id": "cross_selling", "name": "Graph Cross-Selling", "area": "Sales", "lighthouse": False,
     "app_route": "/cross-selling", "app_name": "Graph Cross-Selling",
     "label": "Kauf aus einer bisher nicht gekauften Artikelgruppe (90 Tage)",
     "baseline": "meistverkaufte noch nicht gekaufte Gruppe", "lever": "Umsatz", "max_stage": 3,
     "note": "nach Playbook „Artikelgruppen aus Freitext“"},
    {"id": "recommendation", "name": "Produkt-Empfehlung", "area": "Sales", "lighthouse": False,
     "app_route": "/recommender", "app_name": "Produkt-Empfehlung",
     "label": "Rangliste der in 90 Tagen gekauften Artikel",
     "baseline": "Topseller der letzten 30 Tage + Wiederkauf", "lever": "Warenkorb", "max_stage": 3},
    {"id": "lead_scoring", "name": "Neukunden- / Lead-Scoring", "area": "Sales", "lighthouse": False,
     "app_route": "/lead-scoring", "app_name": "Lead Scoring",
     "label": "Erstkäufer kauft innerhalb von 90 Tagen erneut",
     "baseline": "Wert des ersten Warenkorbs", "lever": "Umsatz, Vertriebsfokus", "max_stage": 3,
     "note": "Cold-Start-Check"},
    {"id": "next_best_action", "name": "Next Best Action", "area": "Sales", "lighthouse": False,
     "app_route": "/next-action", "app_name": "Next Best Action",
     "label": "Regelwerk über die Scores 1–4 (keine Reaktionshistorie)",
     "baseline": "–", "lever": "Umsatz", "max_stage": 2,
     "note": "höchstens Gelb: Kampagnen- und Reaktionsdaten fehlen"},
    {"id": "demand_forecast", "name": "Absatzprognose", "area": "Sales", "lighthouse": False,
     "app_route": "/forecasting_model", "app_name": "Zeitreihen-Forecasting",
     "label": "Wochenmenge je Artikelgruppe und Land",
     "baseline": "saisonal-naiv (Vorjahreswoche)", "lever": "Bestand, Kosten", "max_stage": 3},
    {"id": "margin", "name": "Margenoptimierung", "area": "Sales", "lighthouse": False,
     "app_route": "/margenoptimierung", "app_name": "Margenoptimierung",
     "label": "nicht berechenbar – Einstandspreis fehlt",
     "baseline": "Preisstreuung je Artikel und Kunde", "lever": "Marge", "max_stage": 2},
    {"id": "returns_root_cause", "name": "Retouren- & Storno-Root-Cause", "area": "After Sales",
     "lighthouse": True, "app_route": None, "app_name": "neu",
     "label": "Position wird innerhalb von 30 Tagen storniert oder retourniert",
     "baseline": "Retourenquote je Artikel (Vorperiode)", "lever": "Kosten, Umsatzerhalt", "max_stage": 3},
    {"id": "warranty_root_cause", "name": "Warranty Root Cause", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu (PDF-Vorlage)",
     "label": "Garantiefall in X Monaten je Maschine",
     "baseline": "–", "lever": "Kosten", "max_stage": 1,
     "note": "Seriennummer, Charge und Garantiefälle fehlen"},
]


def list_use_cases() -> list[dict]:
    return USE_CASES
