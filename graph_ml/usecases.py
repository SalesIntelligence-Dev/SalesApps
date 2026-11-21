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
    {"id": "next_best_action", "name": "Next Best Action", "area": "Sales", "lighthouse": False,
     "app_route": "/next-action", "app_name": "Next Best Action",
     "label": "Reaktion auf die empfohlene Aktion",
     "label_detail": "Lernbar nur mit Kampagnen- und Reaktionsdaten. Ohne sie: Regelwerk über die Scores aus Churn, Cross-Selling, Empfehlung und Neukunden.",
     "window_days": 90, "min_positives": 200, "entity": "customer",
     "baseline": "–", "lever": "Umsatz",
     "blockers": ["Keine Kampagnen- und Reaktionsdaten", "Aktionen nicht protokolliert"],
     "requires": [("customer_key", "pflicht", None, "PB-01"), ("event_time", "pflicht", None, None),
                  ("product_key", "pflicht", None, None), ("campaign", "messung", None, "PB-10"),
                  ("response", "messung", None, "PB-10")],
     "links": [("customer_key", "event_time")], "learnable": False, "cap": "warn", "max_stage": 2,
     "note": "höchstens Gelb ohne Kampagnen- und Reaktionsdaten"},
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


def list_use_cases() -> list[dict]:
    """UI-Sicht: Felder mit Klartext-Labels statt Tupeln."""
    out = []
    for u in USE_CASES:
        d = {k: v for k, v in u.items() if k not in ("requires", "links")}
        d["requires"] = [{"field": f, "label": FIELDS[f]["label"], "importance": imp,
                          "alt": alt, "alt_label": FIELDS[alt]["label"] if alt else None, "playbook": pb}
                         for f, imp, alt, pb in u["requires"]]
        out.append(d)
    return out
