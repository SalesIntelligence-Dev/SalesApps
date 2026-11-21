"""
Use-Case-Bibliothek
===================
Jeder Use Case ist eine Vorlage: welche Felder nötig sind (mit Synonymen für
verbreitete ERP-/CRM-Feldnamen), wie das Label berechnet wird, welche Regel
der Vertrieb heute nutzt und wo es typischerweise hakt (PDF S. 5–6).

Felder sind *semantisch* (z. B. „Kundennummer“), nicht an Spaltennamen gebunden:
semantics.bind_fields() ordnet sie den Spalten eines Schemas zu – für
Online Retail II (customer_id), SAP (KUNNR) oder AdventureWorks (CustomerID).
"""
from __future__ import annotations

# ── Feldkatalog ───────────────────────────────────────────────────────────────
# syn        exakte Spaltennamen (normalisiert: klein, ohne Sonderzeichen)
# contains   Teilstrings ab 5 Zeichen, die im Spaltennamen vorkommen dürfen
# tables     Tabellennamen-Hinweise (Bonus bzw. Pflicht bei table_required)
# table_level  Feld gilt als vorhanden, wenn eine Tabelle so heißt
# kinds      zulässige Spaltentypen (datetime · numeric · any)
FIELDS: dict[str, dict] = {
    "customer_key": {
        "label": "Kundennummer",
        "syn": ["customerid", "customer", "kunde", "kundeid", "kundennr", "kundennummer", "kunnr", "debitor",
                "debitorennr", "accountid", "clientid", "kdnr", "partnerid", "businesspartner"],
        "contains": ["customer", "kunden", "debitor"],
        "exclude": ["type", "typ", "name", "group", "gruppe"],
        "kinds": "any",
    },
    "order_key": {
        "label": "Beleg-/Auftragsnummer",
        "syn": ["invoice", "invoiceid", "invoiceno", "orderid", "salesorderid", "auftrag", "auftragsnr", "vbeln",
                "beleg", "belegnr", "belegnummer", "rechnung", "rechnungsnr", "rechnungsnummer", "ordernumber"],
        "contains": ["invoice", "salesorder", "auftrag", "rechnung"],
        "exclude": ["detail", "line", "date", "datum", "type", "typ", "status"],
        "kinds": "any",
    },
    "event_time": {
        "label": "Ereigniszeit (Beleg-/Auftragsdatum)",
        "syn": ["invoicedate", "orderdate", "datum", "erdat", "audat", "fkdat", "belegdatum", "auftragsdatum",
                "rechnungsdatum", "createdat", "date", "transactiondate", "buchungsdatum"],
        "contains": ["invoicedate", "orderdate", "belegdatum", "transactiondate", "auftragsdatum"],
        "exclude": ["modified", "birth", "hire", "due", "ship", "start", "end", "sellend", "discontinued",
                    "expir", "valid", "rate"],
        "kinds": "datetime",
    },
    "product_key": {
        "label": "Artikelnummer",
        "syn": ["stockcode", "productid", "product", "artikel", "artikelnr", "artikelnummer", "matnr", "material",
                "sku", "itemid", "item", "materialnummer"],
        "contains": ["stockcode", "productid", "artikel", "material"],
        "exclude": ["category", "subcategory", "model", "photo", "description", "name", "group", "gruppe",
                    "vendor", "review", "document", "type"],
        "kinds": "any",
    },
    "quantity": {
        "label": "Menge",
        "syn": ["quantity", "qty", "orderqty", "menge", "anzahl", "kwmeng", "fkimg", "stueck"],
        "contains": ["quantity", "menge"],
        "exclude": ["safety", "reorder", "stocked", "scrapped", "received", "rejected", "min", "max"],
        "kinds": "numeric",
        "prefer_with": "product_key",
    },
    "price": {
        "label": "Verkaufspreis",
        "syn": ["price", "unitprice", "pricegbp", "preis", "netpr", "vkpreis", "listprice", "verkaufspreis"],
        "contains": ["price", "preis"],
        "exclude": ["discount", "rabatt", "cost"],
        "kinds": "numeric",
        "prefer_with": "product_key",
    },
    "cost": {
        "label": "Einstandspreis / Kosten",
        "syn": ["standardcost", "cost", "einstandspreis", "ekpreis", "costprice", "stprs", "verpr", "herstellkosten",
                "selbstkosten", "unitcost"],
        "contains": ["standardcost", "einstand", "costprice", "unitcost", "herstellkosten"],
        "exclude": ["shipping", "freight", "fracht"],
        "kinds": "numeric",
    },
    "product_group": {
        "label": "Artikelgruppe",
        "syn": ["productsubcategoryid", "productcategoryid", "warengruppe", "artikelgruppe", "matkl", "productgroup",
                "category", "kategorie", "productline", "sparte"],
        "contains": ["productcategory", "productsubcategory", "warengruppe", "artikelgruppe"],
        "tables": ["product", "artikel", "material", "item", "category", "kategorie", "warengruppe"],
        "table_required": True,
        "table_level": ["productcategory", "productsubcategory", "warengruppe", "artikelgruppe"],
        "kinds": "any",
    },
    "product_text": {
        "label": "Artikelbeschreibung (Text)",
        "syn": ["description", "name", "bezeichnung", "maktx", "beschreibung", "artikeltext", "productname"],
        "contains": ["description", "bezeichnung", "beschreibung"],
        "tables": ["product", "artikel", "material", "item"],
        "table_required": True,
        "kinds": "any",
    },
    "country": {
        "label": "Land / Region",
        "syn": ["country", "land", "countryregioncode", "region", "territoryid", "landesschluessel", "land1"],
        "contains": ["country", "region"],
        "kinds": "any",
    },
    "account_group": {
        "label": "Einkaufsgemeinschaft / Konzern / Kundengruppe",
        "syn": ["buyinggroupid", "buyinggroup", "einkaufsgemeinschaft", "konzern", "konzernid", "parentaccountid",
                "customercategoryid", "kundengruppe", "kdgrp", "verband", "billtocustomerid"],
        "contains": ["buyinggroup", "einkaufsgem", "konzern", "kundengruppe", "customercategory"],
        "kinds": "any",
    },
    "doc_type": {
        "label": "Belegart / Status (Storno, Retoure)",
        "syn": ["belegtyp", "doctype", "belegart", "auart", "fkart", "orderstatus", "status", "documenttype"],
        "contains": ["belegtyp", "belegart", "doctype", "orderstatus"],
        "kinds": "any",
    },
    "return_link": {
        "label": "Bezug Retoure → Originalbeleg",
        "syn": ["originalinvoice", "refinvoice", "bezugsbeleg", "vgbel", "originalorder", "referenceinvoice",
                "originalbeleg", "returnof"],
        "contains": ["originalinvoice", "bezugsbeleg", "originalorder", "originalbeleg"],
        "kinds": "any",
    },
    "campaign": {
        "label": "Kampagnen / Aktionen",
        "syn": ["campaignid", "campaign", "kampagne", "specialofferid", "promotionid", "aktion", "aktionsnr"],
        "contains": ["campaign", "kampagne", "promotion", "specialoffer"],
        "table_level": ["campaign", "kampagne", "specialoffer", "promotion", "aktion", "marketing"],
        "kinds": "any",
    },
    "response": {
        "label": "Reaktion auf Aktionen (Antwort, Abschluss)",
        "syn": ["response", "reaktion", "responded", "converted", "conversion", "opened", "clicked", "outcome"],
        "contains": ["response", "reaktion", "conversion", "outcome"],
        "table_level": ["campaignresponse", "kampagnenreaktion", "activityoutcome"],
        "kinds": "any",
    },
    "serial_number": {
        "label": "Seriennummer",
        "syn": ["serialnumber", "serial", "seriennummer", "sernr", "equnr", "equipment", "geraetenr"],
        "contains": ["serial", "seriennummer", "equipment"],
        "table_level": ["serialnumber", "equipment", "installedbase", "installiertebasis"],
        "kinds": "any",
    },
    "batch": {
        "label": "Charge / Los",
        "syn": ["batch", "batchid", "charge", "charg", "lot", "lotnumber", "losnr"],
        "contains": ["batch", "charge", "lotnumber"],
        "kinds": "any",
    },
    "warranty_claim": {
        "label": "Garantie-/Reklamationsfälle",
        "syn": ["claimid", "warrantyclaim", "garantiefall", "reklamation", "qmnum", "complaintid"],
        "contains": ["warranty", "garantie", "reklamation", "complaint", "claim"],
        "table_level": ["warranty", "garantie", "claim", "reklamation", "complaint", "qualitynotification"],
        "kinds": "any",
    },
    "defect_code": {
        "label": "Fehlercode / Schadensursache",
        "syn": ["scrapreasonid", "defectcode", "fehlercode", "schadenscode", "failurecode", "fecod", "reasoncode"],
        "contains": ["scrapreason", "defect", "fehlercode", "failure", "schaden"],
        "table_level": ["scrapreason", "defect", "fehlercode", "failure"],
        "kinds": "any",
    },
    "supplier": {
        "label": "Lieferant",
        "syn": ["vendorid", "supplierid", "lieferant", "lifnr", "vendor", "supplier"],
        "contains": ["vendor", "supplier", "lieferant"],
        "table_level": ["vendor", "supplier", "lieferant"],
        "kinds": "any",
    },
    "bom": {
        "label": "Stückliste",
        "syn": ["billofmaterialsid", "bomid", "stueckliste", "stlnr"],
        "contains": ["billofmaterial", "stueckliste"],
        "table_level": ["billofmaterials", "bom", "stueckliste"],
        "kinds": "any",
    },
    "usage_data": {
        "label": "Einsatz-/Betriebsdaten",
        "syn": ["operatinghours", "betriebsstunden", "usage", "laufzeit", "telemetry", "sensor"],
        "contains": ["operatinghour", "betriebsstunden", "telemetry", "sensor", "usage"],
        "table_level": ["usage", "telemetry", "sensordata", "betriebsdaten", "einsatzdaten"],
        "kinds": "any",
    },
}

# ── Use Cases ─────────────────────────────────────────────────────────────────
# requires: (feld, wichtigkeit, ersatzfeld, playbook)
#   pflicht   – ohne das Feld ist der Use Case nicht abbildbar
#   optional  – verbessert Modell / Erklärung
#   messung   – nötig, um ein Label zu lernen (fehlt es: nur Regelwerk, Ampel max. Gelb)
# links: Felder, die über Schlüssel verbunden sein müssen
USE_CASES: list[dict] = [
    {"id": "churn", "name": "Churn & Retention", "area": "Sales", "lighthouse": True,
     "app_route": "/churn", "app_name": "Churn & Retention",
     "label": "Aktiver Kunde kauft in den nächsten 90 Tagen nicht",
     "label_detail": "Aktiv = mindestens ein Kauf in den 365 Tagen vor dem Stichtag. Positiv = kein Kauf im 90-Tage-Fenster danach.",
     "window_days": 90, "min_positives": 200, "entity": "customer",
     "baseline": "„seit 60 Tagen keine Bestellung“", "lever": "Umsatz halten",
     "blockers": ["Churn im B2B unscharf definiert", "Kundendubletten", "Konzernstrukturen"],
     "requires": [("customer_key", "pflicht", None, "PB-01"), ("event_time", "pflicht", None, None),
                  ("order_key", "pflicht", None, None), ("price", "optional", None, None),
                  ("quantity", "optional", None, None), ("country", "optional", None, None),
                  ("doc_type", "optional", None, "PB-02")],
     "links": [("customer_key", "event_time")], "learnable": True, "max_stage": 3},
    {"id": "cross_selling", "name": "Graph Cross-Selling", "area": "Sales", "lighthouse": False,
     "app_route": "/cross-selling", "app_name": "Graph Cross-Selling",
     "label": "Kauf aus einer bisher nicht gekauften Artikelgruppe (90 Tage)",
     "label_detail": "Positiv = aktiver Kunde kauft im 90-Tage-Fenster erstmals eine Artikelgruppe, die er vorher nie gekauft hat.",
     "window_days": 90, "min_positives": 200, "entity": "customer",
     "baseline": "meistverkaufte noch nicht gekaufte Gruppe", "lever": "Umsatz",
     "blockers": ["Uneinheitlicher Artikelstamm", "Varianten", "wenig Daten bei Kleinkunden"],
     "requires": [("customer_key", "pflicht", None, "PB-01"), ("event_time", "pflicht", None, None),
                  ("product_key", "pflicht", None, "PB-04"), ("product_group", "pflicht", "product_text", "PB-05"),
                  ("price", "optional", None, None), ("quantity", "optional", None, None)],
     "links": [("customer_key", "product_key")], "learnable": True, "max_stage": 3,
     "note": "nach Playbook „Artikelgruppen aus Freitext“"},
    {"id": "recommendation", "name": "Produkt-Empfehlung", "area": "Sales", "lighthouse": False,
     "app_route": "/recommender", "app_name": "Produkt-Empfehlung",
     "label": "Rangliste der in 90 Tagen gekauften Artikel",
     "label_detail": "Für jeden aktiven Kunden: welche Artikel kauft er im 90-Tage-Fenster? Bewertet mit MAP@12 / Recall@12.",
     "window_days": 90, "min_positives": 300, "entity": "customer",
     "baseline": "Topseller der letzten 30 Tage + Wiederkauf", "lever": "Warenkorb",
     "blockers": ["Cold Start bei neuen Artikeln", "Sortimentswechsel"],
     "requires": [("customer_key", "pflicht", None, "PB-01"), ("event_time", "pflicht", None, None),
                  ("product_key", "pflicht", None, "PB-04"), ("product_group", "optional", None, "PB-05"),
                  ("product_text", "optional", None, None), ("price", "optional", None, None)],
     "links": [("customer_key", "product_key")], "learnable": True, "max_stage": 3},
    {"id": "lead_scoring", "name": "Neukunden- / Lead-Scoring", "area": "Sales", "lighthouse": False,
     "app_route": "/lead-scoring", "app_name": "Lead Scoring",
     "label": "Erstkäufer kauft innerhalb von 90 Tagen erneut",
     "label_detail": "Kohorte = Kunden mit Erstkauf in den 90 Tagen vor dem Stichtag. Positiv = zweiter Beleg innerhalb von 90 Tagen nach dem Erstkauf.",
     "window_days": 90, "min_positives": 150, "entity": "customer",
     "baseline": "Wert des ersten Warenkorbs", "lever": "Umsatz, Vertriebsfokus",
     "blockers": ["Nur der erste Warenkorb als Kontext (Cold Start)", "Kundendubletten"],
     "requires": [("customer_key", "pflicht", None, "PB-01"), ("event_time", "pflicht", None, None),
                  ("order_key", "pflicht", None, None), ("product_key", "optional", None, None),
                  ("price", "optional", None, None), ("country", "optional", None, None)],
     "links": [("customer_key", "event_time")], "learnable": True, "max_stage": 3, "note": "Cold-Start-Check"},
    {"id": "account_scoring", "name": "Account-Scoring (Wachstum)", "area": "Sales", "lighthouse": False,
     "app_route": "/lead-scoring", "app_name": "Lead Scoring",
     "label": "Bestandskunde steigert seinen Quartalsumsatz um mindestens 20 %",
     "label_detail": "Aktiv = Kauf in den 180 Tagen vor dem Stichtag. Positiv = Umsatz der nächsten 90 Tage ≥ 120 % der letzten 90 Tage.",
     "window_days": 90, "min_positives": 100, "entity": "customer",
     "baseline": "Umsatztrend des letzten Quartals", "lever": "Umsatz, Key-Account-Fokus",
     "blockers": ["Konzern- und Einkaufsverbund-Strukturen fehlen im CRM", "Kundendubletten"],
     "requires": [("customer_key", "pflicht", None, None), ("event_time", "pflicht", None, None),
                  ("order_key", "pflicht", None, None), ("account_group", "optional", None, None),
                  ("price", "optional", None, None), ("quantity", "optional", None, None),
                  ("country", "optional", None, None)],
     "links": [("customer_key", "event_time")], "learnable": True, "max_stage": 3,
     "note": "gerechnet auf WideWorldImporters (Einkaufsgemeinschaften)"},
    {"id": "territory_coverage", "name": "Vertriebsabdeckung & Gebiete", "area": "Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Händler bestellt im nächsten Quartal – Besuchsliste nach erwartetem Umsatz",
     "label_detail": "Händler = Kunde mit Store und Vertriebler. Positiv = mindestens ein Auftrag in den 90 Tagen nach dem Stichtag.",
     "window_days": 90, "min_positives": 100, "entity": "customer",
     "baseline": "Händler mit dem höchsten Jahresumsatz zuerst besuchen", "lever": "Vertriebsproduktivität",
     "blockers": ["Gebiets- und Vertriebler-Zuordnung nicht historisiert", "Besuchsberichte fehlen im CRM"],
     "requires": [("customer_key", "pflicht", None, None), ("event_time", "pflicht", None, None),
                  ("order_key", "pflicht", None, None), ("account_group", "optional", None, None),
                  ("country", "pflicht", None, None), ("price", "optional", None, None)],
     "links": [("customer_key", "event_time")], "learnable": True, "max_stage": 3,
     "note": "gerechnet auf AdventureWorks (Gebiete, Vertriebler, Quoten)"},
    {"id": "next_best_action", "name": "Next Best Action", "area": "Sales", "lighthouse": False,
     "app_route": "/next-action", "app_name": "Next Best Action",
     "label": "Wert der gewählten Aktion je Kunde (Policy-Rückblick)",
     "label_detail": "Ein lernendes NBA-Modell braucht Kampagnen- und Reaktionsdaten. Messbar ohne sie: Welche Aktion hätte das Regelwerk am Rückblick-Stichtag gewählt, und trat das Zielereignis (Abwanderung, Gruppenkauf, Wiederkauf, Artikelkauf) danach ein?",
     "window_days": 90, "min_positives": 200, "entity": "customer",
     "baseline": "Inaktive anrufen, sonst Topseller-Newsletter", "lever": "Umsatz",
     "blockers": ["Keine Kampagnen- und Reaktionsdaten", "Aktionen nicht protokolliert"],
     "requires": [("customer_key", "pflicht", None, "PB-01"), ("event_time", "pflicht", None, None),
                  ("product_key", "pflicht", None, None), ("campaign", "messung", None, "PB-10"),
                  ("response", "messung", None, "PB-10")],
     "links": [("customer_key", "event_time")], "learnable": False, "cap": "warn", "max_stage": 3,
     "note": "Datenreife höchstens Gelb ohne Reaktionsdaten · Rückblick als Policy-Vergleich"},
    {"id": "demand_forecast", "name": "Absatzprognose", "area": "Sales", "lighthouse": False,
     "app_route": "/forecasting_model", "app_name": "Zeitreihen-Forecasting",
     "label": "Wochenmenge je Artikel bzw. Artikelgruppe",
     "label_detail": "Menge je Artikel und Kalenderwoche; Prognose der nächsten 13 Wochen. Mindestens 52 Wochen Historie, besser zwei Saisons.",
     "window_days": 91, "min_positives": 50, "entity": "series",
     "baseline": "saisonal-naiv (Vorjahreswoche)", "lever": "Bestand, Kosten",
     "blockers": ["Sondereffekte", "starke Zeitverschiebung (hier gewinnen oft Baum-Modelle)"],
     "requires": [("product_key", "pflicht", None, "PB-04"), ("event_time", "pflicht", None, None),
                  ("quantity", "pflicht", None, "PB-07"), ("product_group", "optional", "product_text", "PB-05"),
                  ("country", "optional", None, None), ("campaign", "optional", None, None)],
     "links": [("product_key", "event_time")], "learnable": True, "max_stage": 3},
    {"id": "margin", "name": "Margenoptimierung", "area": "Sales", "lighthouse": False,
     "app_route": "/margenoptimierung", "app_name": "Margenoptimierung",
     "label": "Deckungsbeitrag je Position, Rabatt-Leckage",
     "label_detail": "Marge = Verkaufspreis − Einstandspreis je Position. Ohne Einstandspreis bleibt nur die Preisstreuung je Artikel.",
     "window_days": 90, "min_positives": 200, "entity": "line",
     "baseline": "Preisstreuung je Artikel und Kunde", "lever": "Marge",
     "blockers": ["Einstandspreis nicht im Export", "Rabatte nur als Nettopreis"],
     "requires": [("product_key", "pflicht", None, None), ("price", "pflicht", None, None),
                  ("quantity", "pflicht", None, None), ("cost", "pflicht", None, "PB-10"),
                  ("customer_key", "optional", None, None), ("event_time", "optional", None, None),
                  ("product_group", "optional", None, None)],
     "links": [("product_key", "price")], "learnable": True, "max_stage": 2},
    {"id": "returns_root_cause", "name": "Retouren- & Storno-Root-Cause", "area": "After Sales",
     "lighthouse": True, "app_route": None, "app_name": "neu",
     "label": "Position wird innerhalb von 30 Tagen storniert oder retourniert",
     "label_detail": "Positiv = Verkaufsposition, zu der binnen 30 Tagen ein Storno desselben Kunden für denselben Artikel folgt. Daraus: Kohorten mit erhöhtem Risiko.",
     "window_days": 30, "min_positives": 200, "entity": "line",
     "baseline": "Retourenquote je Artikel (Vorperiode)", "lever": "Kosten, Umsatzerhalt",
     "blockers": ["Retoure ohne Bezug zum Originalbeleg", "Retourengrund nur als Freitext"],
     "requires": [("product_key", "pflicht", None, "PB-04"), ("event_time", "pflicht", None, None),
                  ("doc_type", "pflicht", "quantity", None), ("return_link", "pflicht", "customer_key", "PB-02"),
                  ("customer_key", "optional", None, "PB-01"), ("country", "optional", None, None),
                  ("defect_code", "optional", None, "PB-05")],
     "links": [("product_key", "event_time")], "learnable": True, "max_stage": 3},
    {"id": "warranty_root_cause", "name": "Warranty Root Cause", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu (PDF-Vorlage)",
     "label": "Garantiefall in X Monaten je Maschine",
     "label_detail": "Positiv = Garantiefall innerhalb von 12 Monaten nach Auslieferung je Seriennummer; Kohorten über Charge, Lieferant und Stückliste.",
     "window_days": 365, "min_positives": 100, "entity": "serial",
     "baseline": "–", "lever": "Kosten",
     "blockers": ["Seriennummer–Charge-Verknüpfung fehlt", "Fehlercodes als Freitext"],
     "requires": [("serial_number", "pflicht", None, "PB-10"), ("batch", "pflicht", None, "PB-10"),
                  ("warranty_claim", "pflicht", None, "PB-10"), ("product_key", "pflicht", None, None),
                  ("event_time", "pflicht", None, None), ("supplier", "optional", None, None),
                  ("bom", "optional", None, None), ("defect_code", "optional", None, "PB-05"),
                  ("usage_data", "optional", None, None)],
     "links": [("serial_number", "warranty_claim"), ("serial_number", "batch")], "learnable": True,
     "max_stage": 1, "note": "Seriennummer, Charge und Garantiefälle fehlen"},
]

PLAYBOOKS = {
    "PB-01": "Fehlende Kundennummern",
    "PB-02": "Stornos ohne Originalbezug",
    "PB-03": "Nicht-Artikel-Codes",
    "PB-04": "Artikelstamm konsolidieren",
    "PB-05": "Freitext → Kategorien",
    "PB-06": "Dubletten entfernen",
    "PB-07": "Preis 0 / negative Mengen",
    "PB-08": "Währung umrechnen",
    "PB-09": "Governance-Checkliste",
    "PB-10": "Fehlende Tabellen anfordern",
}


def get_use_case(uc_id: str) -> dict:
    return next(u for u in USE_CASES if u["id"] == uc_id)


# ── Detailseite je Use Case (v2): Hebel laut Literatur, Input, Graph-Signal, Vorhersage, Euro ──────────────
# v2      Bezeichnung in der Use-Case-Bibliothek v2 (GRAPH_ML_V2_PROMPTS.md, 0.4)
# effect  Wirkung laut Literatur bzw. Praxisfällen – Größenordnung, kein Versprechen
# graph   welches Signal über mehrere Tabellen läuft (das sieht relationales Lernen zusätzlich)
# path    Pfad im Graphen, der dieses Signal trägt (für die Mini-Grafik)
# tabular was ein flaches Modell (LightGBM, TabPFN) aus einer Zeile je Fall schon sieht
# unit    wofür vorhergesagt wird · euro  wie der Nutzen in Euro bewertet wird
STORY: dict[str, dict] = {
    "churn": {
        "v2": "Abwanderung früh erkennen", "effect": "5 % mehr Kundenbindung → 25–95 % mehr Gewinn", "fit": "hoch",
        "graph": "Kunden, die dieselben Artikel kaufen wie inaktive Kunden, wandern häufiger ab – das Signal läuft über "
                 "Kunde → Artikel → anderer Kunde und ist in keiner einzelnen Kundenzeile enthalten.",
        "path": ["Kunde", "Rechnung", "Artikel", "Nachbarkunde"],
        "tabular": "Recency, Frequenz, Umsatz, Bestellrhythmus, Stornoquote je Kunde",
        "unit": "je aktivem Kunde, 90-Tage-Fenster ab Stichtag",
        "euro": "erkannte Abwanderer in der Top-Liste × Ø Quartalsumsatz × Rettungsquote − Kontakte × Kosten je Kontakt, × Kampagnen pro Jahr"},
    "cross_selling": {
        "v2": "Cross-Selling und Produktempfehlung", "effect": "meist 10–15 % mehr Umsatz", "fit": "hoch",
        "graph": "Welche Artikelgruppen gemeinsam gekauft werden, ergibt sich aus allen Kunden: Kunde → Gruppe ← "
                 "andere Kunden → nächste Gruppe (Co-Kauf-Affinität).",
        "path": ["Kunde", "Artikelgruppe", "Kunden derselben Gruppe", "neue Gruppe"],
        "tabular": "Beliebtheit und Wachstum der Gruppe, Umsatz und Breite des Kunden",
        "unit": "je Kunde × noch nicht gekaufter Artikelgruppe, 90 Tage",
        "euro": "Umsatz der Erstkäufe in neuen Gruppen, die die Liste erfasst × ausgelöster Anteil − Angebote × Kosten"},
    "recommendation": {
        "v2": "Cross-Selling und Produktempfehlung", "effect": "meist 10–15 % mehr Umsatz", "fit": "hoch",
        "graph": "Heute: Kandidaten aus dem Co-Kauf-Graphen aller Kunden („wer A kaufte, kaufte auch B“) – andere Kunden "
                 "fließen als Artikel-Ähnlichkeit ein (Graph-ML light). Ein gelerntes Kunden-Graph-Modell (LightGCN) "
                 "verlor im Rückblick; ein heterogenes GNN folgt in Prompt 3.",
        "path": ["Kunde", "gekaufter Artikel", "Kunden mit demselben Artikel", "empfohlener Artikel"],
        "tabular": "eigene Wiederkäufe, Beliebtheit 30/90 Tage",
        "unit": "je aktivem Kunde: Top-12-Artikel für die nächsten 90 Tage",
        "euro": "Umsatz der getroffenen Empfehlungen × Abdeckung × ausgelöster Anteil − Kosten je Kunde"},
    "lead_scoring": {
        "v2": "Lead- und Account-Scoring", "effect": "Lead-ROI +77 % (ältere Umfrage)", "fit": "mittel",
        "graph": "Neukunden ähneln früheren Erstkäufern derselben Artikel und desselben Landes – deren Wiederkaufquote "
                 "ist ein Graph-Merkmal (Erstkäufer → Artikel → frühere Erstkäufer).",
        "path": ["Neukunde", "Erstkauf-Artikel", "frühere Erstkäufer", "Wiederkauf"],
        "tabular": "Wert, Breite und Ø Preis des ersten Warenkorbs, Land",
        "unit": "je Erstkäufer der letzten 90 Tage: Wiederkauf in 90 Tagen",
        "euro": "Umsatz der Wiederkäufer in der Liste × Mehrumsatz durch Betreuung − Kontakte × Kosten"},
    "account_scoring": {
        "v2": "Lead- und Account-Scoring", "effect": "Lead-ROI +77 % (ältere Umfrage)", "fit": "mittel",
        "graph": "Accounts derselben Einkaufsgemeinschaft, Kundenkategorie und Region wachsen gemeinsam – deren Wachstum "
                 "(ohne den Account selbst) ist ein Peer-Signal über den Graphen, dazu die Kunde–Artikel-Nachbarschaft.",
        "path": ["Account", "Einkaufsgemeinschaft", "Peer-Accounts", "Wachstum"],
        "tabular": "Umsatz, Frequenz und Trend des Accounts, Kreditlimit, Kategorie",
        "unit": "je aktivem Account: Quartalsumsatz wächst um ≥ 20 %",
        "euro": "Mehrumsatz wachsender Accounts in der Liste × zusätzlich gesicherter Anteil − Accounts × Betreuungskosten, × 4 Quartale"},
    "territory_coverage": {
        "v2": "Vertriebsabdeckung und Gebiete", "effect": "Vertriebsproduktivität +5 bis 10 %", "fit": "mittel",
        "graph": "Händler eines Vertrieblers bestellen in Wellen (Besuchsrunden), Händler eines Gebiets folgen gemeinsamen "
                 "Saisons – der Anteil der Portfolio- und Gebiets-Peers mit frischem Auftrag ist ein Signal über die Kanten "
                 "Händler → Vertriebler → andere Händler.",
        "path": ["Händler", "Vertriebler", "Portfolio-Händler", "Bestellwelle"],
        "tabular": "Recency, Frequenz, Jahresumsatz und Bestellrhythmus je Händler",
        "unit": "je aktivem Händler: Auftrag im nächsten Quartal, sortiert nach erwartetem Umsatz",
        "euro": "Händlerumsatz des Folgequartals in der Besuchsliste × Mehrumsatz durch gezielten Besuch − Besuche × Kosten, × 4 Runden"},
    "next_best_action": {
        "v2": "Next Best Action im Vertrieb", "effect": "Umsatz +5–8 %, Betreuungskosten −20–30 %", "fit": "hoch",
        "graph": "Kombiniert die Scores aus Churn, Cross-Selling, Empfehlung und Neukunden (LightGBM + Graph-Merkmale) je Kunde "
                 "mit Konfliktauflösung. Der Rückblick vergleicht Policies: welcher Wert wäre mit den gewählten Aktionen erzielt "
                 "worden – gegen „Inaktive anrufen, sonst Newsletter“. Ein lernendes Modell braucht Reaktionsdaten.",
        "path": ["Kunde", "Scores je Use Case", "Regelwerk", "eine Aktion"],
        "tabular": "Scores und erwartete Werte der Einzel-Use-Cases",
        "unit": "je Kunde: eine Aktion je Quartal",
        "euro": "realisierter Netto-Wert der gewählten Aktionen im Rückblick-Quartal (Wirkungsquoten wie in den Einzel-Use-Cases), × 4"},
    "demand_forecast": {
        "v2": "Absatzprognose", "effect": "Prognosefehler −20 bis −50 %", "fit": "mittel",
        "graph": "Artikel derselben Gruppe teilen Saison und Trends (Artikel → Gruppe → Artikel). Gemessen: LightGBM mit "
                 "Verzögerungs-Merkmalen gewinnt – der Graph bringt hier wenig.",
        "path": ["Artikel", "Artikelgruppe", "Geschwister-Artikel"],
        "tabular": "Vorwochen, Vorjahreswoche, gleitende Mittel, Kalender",
        "unit": "je Top-100-Artikel und Woche, 13 Wochen voraus",
        "euro": "Sicherheitsbestand × relative Fehlerreduktion ggü. Regel × Lagerhaltungskostensatz"},
    "margin": {
        "v2": "Preis- und Rabattempfehlung · Preis-Leakage", "effect": "+2–7 Pp. Umsatzrendite; > 6 % des Umsatzes versickern",
        "fit": "mittel",
        "graph": "Rabatte ähnlicher Kunden für ähnliche Artikel (Kunde → Artikel → Peer-Kunden) zeigen, wo Preise unter "
                 "Peer-Niveau liegen. Ohne Einstandspreis keine Marge – deshalb hier nur Profil.",
        "path": ["Kunde", "Position", "Artikel", "Peer-Kunden"],
        "tabular": "Preisstreuung je Artikel und Kunde",
        "unit": "je Position: Preis unter Peer-Niveau (Leakage)",
        "euro": "Umsatz der Leakage-Positionen × schließbare Lücke zum Peer-Preis (Prompt 5)"},
    "returns_root_cause": {
        "v2": "Garantie: Frühwarnung und Root Cause (Retouren)", "effect": "Garantie-/Retourenkosten −4 bis −20 %",
        "fit": "hoch",
        "graph": "Stornos häufen sich bei Artikeln, die dieselben Kunden kaufen (Position → Artikel → Kunden → "
                 "Nachbarartikel) – die Retourenquote der Nachbarn ist ein Graph-Merkmal.",
        "path": ["Position", "Artikel", "Kunden", "Nachbarartikel"],
        "tabular": "Menge, Preis vs. üblich, Positionen im Beleg, Kundenhistorie",
        "unit": "je Verkaufsposition: Storno binnen 30 Tagen",
        "euro": "erkannte Retouren × Vermeidungsquote × (Prozesskosten + Wertverlust) − Prüfkosten"},
    "warranty_root_cause": {
        "v2": "Garantie: Frühwarnung und Root Cause", "effect": "Garantiekosten −4 bis −20 % (Anbieter)", "fit": "hoch",
        "graph": "Fehler pflanzen sich über Charge → Bauteil → Lieferant fort: gleiche Charge, ähnliche Ausfälle. "
                 "Mit Online Retail II nicht berechenbar (keine Seriennummern, Chargen, Garantiefälle).",
        "path": ["Maschine", "Bauteil", "Charge", "Lieferant"],
        "tabular": "Alter, Laufzeit, Einsatzland je Maschine",
        "unit": "je Maschine: Garantiefall in X Monaten",
        "euro": "vermiedene Garantiefälle × Ø Kosten je Fall (Prompt 7)"},
}


def list_use_cases() -> list[dict]:
    """UI-Sicht: Felder mit Klartext-Labels statt Tupeln."""
    out = []
    for u in USE_CASES:
        d = {k: v for k, v in u.items() if k not in ("requires", "links")}
        d["story"] = STORY.get(u["id"])
        d["requires"] = [{"field": f, "label": FIELDS[f]["label"], "importance": imp,
                          "alt": alt, "alt_label": FIELDS[alt]["label"] if alt else None, "playbook": pb}
                         for f, imp, alt, pb in u["requires"]]
        out.append(d)
    return out
