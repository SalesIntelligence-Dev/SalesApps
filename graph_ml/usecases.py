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
                "sku", "itemid", "item", "materialnummer", "stockitemid"],
        "contains": ["stockcode", "productid", "artikel", "material", "stockitemid"],
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
    "sales_rep": {
        "label": "Vertriebler / Betreuer (Gebiet)",
        "syn": ["salespersonid", "salesperson", "vertriebler", "verkaeufer", "aussendienst", "salesrepid", "salesrep",
                "accountmanager", "betreuer", "vkbur", "territoryid"],
        "contains": ["salesperson", "vertriebler", "salesrep", "accountmanager"],
        "kinds": "any",
    },
    "list_price": {
        "label": "Listenpreis (zum Datum)",
        "syn": ["listprice", "listenpreis", "recommendedretailprice", "bruttopreis", "katalogpreis", "kbetr"],
        "contains": ["listprice", "listenpreis", "retailprice"],
        "kinds": "numeric",
    },
    "discount": {
        "label": "Rabatt / Sonderkondition",
        "syn": ["unitpricediscount", "discount", "discountpct", "rabatt", "discountpercentage", "specialofferid", "kondition"],
        "contains": ["discount", "rabatt", "specialoffer"],
        "kinds": "any",
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
        "syn": ["serialnumber", "serial", "seriennummer", "sernr", "equnr", "equipment", "geraetenr", "vehicleid", "fahrzeugid",
                "vin", "fin", "machineid", "maschinenid", "assetid", "unitid", "engineid", "triebwerk"],
        "contains": ["serial", "seriennummer", "equipment", "vehicleid", "machineid", "assetid", "engineid"],
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
        "syn": ["operatinghours", "betriebsstunden", "usage", "laufzeit", "telemetry", "sensor", "readoutid", "nreadouts",
                "timestep", "cycle", "zyklus"],
        "contains": ["operatinghour", "betriebsstunden", "telemetry", "sensor", "usage", "readout"],
        "table_level": ["usage", "telemetry", "sensordata", "betriebsdaten", "einsatzdaten", "readout", "readouts", "auslesung"],
        "kinds": "any",
    },
    "service_text": {
        "label": "Fehlerbeschreibung (Freitext)",
        "syn": ["problemdescription", "fehlerbeschreibung", "symptom", "complainttext", "ticketdescription", "servicetext", "meldungstext"],
        "contains": ["problemdescription", "fehlerbeschreibung", "symptom", "tickettext", "meldungstext"],
        "kinds": "any",
    },
    "service_event": {
        "label": "Reparatur-/Serviceereignis",
        "syn": ["repairid", "repairtimestep", "serviceorderid", "serviceorder", "wartung", "reparatur", "maintenanceid"],
        "contains": ["repair", "reparatur", "serviceorder", "maintenance", "wartung"],
        "exclude": ["workorder", "production", "fertigung"],
        "table_level": ["repair", "repairs", "reparatur", "serviceorder", "maintenance", "wartung"],
        "kinds": "any",
    },
    "rejected_qty": {
        "label": "Beanstandete Menge (Wareneingangsprüfung)",
        "syn": ["rejectedqty", "rejected", "beanstandet", "beanstandetemenge", "reklamiertemenge", "rejectqty", "insmk"],
        "contains": ["rejected", "beanstand", "reject"],
        "kinds": "any",
    },
    "failure_event": {
        "label": "Ausfall / Fehlerereignis",
        "syn": ["failed", "failure", "failureid", "ausfall", "breakdown", "rulatend", "faultid", "errorid", "stoerung"],
        "contains": ["failure", "ausfall", "breakdown", "rulatend", "stoerung"],
        "table_level": ["failure", "failures", "ausfall", "breakdown", "errors", "faults"],
        "kinds": "any",
    },
    "technician": {
        "label": "Techniker / Servicepartner",
        "syn": ["technicianid", "techniker", "servicepartnerid", "servicepartner", "mechanicid", "fieldengineerid"],
        "contains": ["technician", "techniker", "servicepartner", "mechanic"],
        "table_level": ["technician", "techniker", "servicepartner", "mechanic"],
        "kinds": "any",
    },
    "contract": {
        "label": "Servicevertrag (Abschluss, Laufzeit, Kündigung)",
        "syn": ["contractid", "servicecontractid", "vertragsnr", "vertrag", "vbeln_contract", "subscriptionid"],
        "contains": ["contract", "vertrag", "subscription"],
        "table_level": ["contract", "servicecontract", "vertrag", "servicevertrag", "subscription"],
        "kinds": "any",
    },
    "config": {
        "label": "Konfiguration / Ausstattung",
        "syn": ["spec", "configid", "configuration", "konfiguration", "ausstattung", "variant", "variante", "modelcode"],
        "contains": ["spec", "config", "konfiguration", "ausstattung"],
        "table_level": ["config", "configuration", "konfiguration", "ausstattung", "variant"],
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
                  ("order_key", "pflicht", None, None), ("sales_rep", "pflicht", None, None),
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
    {"id": "price_recommendation", "name": "Preis- & Rabattempfehlung", "area": "Sales", "lighthouse": False,
     "app_route": "/margenoptimierung", "app_name": "Margenoptimierung",
     "label": "Erzielbarer Nettopreis (in % vom Listenpreis) je Händler, Artikel und Menge",
     "label_detail": "Vorhergesagt wird der typische Preis zum Angebotszeitpunkt – der tatsächlich gewährte Rabatt ist die Entscheidung "
                     "und deshalb kein Merkmal. Bewertet mit der Abweichung vom erzielten Preis (MAE).",
     "window_days": 0, "min_positives": 0, "entity": "line",
     "baseline": "zuletzt erzielter Preis des Artikels", "lever": "Marge",
     "blockers": ["Listenpreise nicht historisiert", "Rabatte nur als Nettopreis"],
     "requires": [("product_key", "pflicht", None, None), ("price", "pflicht", None, None), ("list_price", "pflicht", None, None),
                  ("quantity", "pflicht", None, None), ("customer_key", "pflicht", None, None), ("event_time", "pflicht", None, None),
                  ("discount", "optional", None, None), ("country", "optional", None, None)],
     "links": [("product_key", "price")], "learnable": True, "max_stage": 3, "note": "gerechnet auf AdventureWorks"},
    {"id": "price_leakage", "name": "Preis-Leakage", "area": "Sales", "lighthouse": False,
     "app_route": "/margenoptimierung", "app_name": "Margenoptimierung",
     "label": "Position mit Nettopreis mehr als 3 Prozentpunkte unter dem erwartbaren Preis",
     "label_detail": "Erwartbarer Preis aus der Preisempfehlung (Peers, Händlerhistorie, Mengenstaffel); die Lücke in Euro ist das Leakage.",
     "window_days": 0, "min_positives": 0, "entity": "line",
     "baseline": "Abweichung vom zuletzt erzielten Artikelpreis", "lever": "Marge",
     "blockers": ["Freigaben für Sonderrabatte nicht protokolliert"],
     "requires": [("product_key", "pflicht", None, None), ("price", "pflicht", None, None), ("list_price", "pflicht", None, None),
                  ("quantity", "pflicht", None, None), ("customer_key", "pflicht", None, None), ("discount", "optional", None, None)],
     "links": [("product_key", "price")], "learnable": True, "max_stage": 3, "note": "gerechnet auf AdventureWorks"},
    {"id": "quote_check", "name": "Konfigurations- & Angebotsprüfung", "area": "Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Auftrag mit Marge unter 10 % (Standardkosten zum Datum)",
     "label_detail": "Regelprüfung je Auftrag: Nettopreis − Standardkosten zum Auftragsdatum, Stücklisten-Tiefe als Komplexität. Kein Modell nötig.",
     "window_days": 0, "min_positives": 0, "entity": "order",
     "baseline": "Prüfung nur gegen Listenpreis", "lever": "Marge, Kosten",
     "blockers": ["Kosten im Angebot nicht bekannt", "Stücklisten nicht gepflegt"],
     "requires": [("product_key", "pflicht", None, None), ("price", "pflicht", None, None), ("cost", "pflicht", None, "PB-10"),
                  ("quantity", "pflicht", None, None), ("bom", "optional", None, None)],
     "links": [("product_key", "price")], "learnable": False, "max_stage": 3, "note": "Regelprüfung · AdventureWorks"},
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
    {"id": "installed_base", "name": "Installierte Basis & Servicebedarf", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Fahrzeug braucht in den nächsten 48 Zeiteinheiten eine Reparatur der Komponente X",
     "label_detail": "Schnitt an einer Werkstatt-Auslesung je Fahrzeug; positiv, wenn die Reparatur innerhalb von 48 Zeiteinheiten folgt. "
                     "Test auf anderen Fahrzeugen (SCANIA hat keine Kalenderzeit). Ergebnis: Liste für proaktive Serviceangebote.",
     "window_days": 0, "min_positives": 100, "entity": "serial",
     "baseline": "längste Laufzeit zuerst", "lever": "Serviceumsatz, Verfügbarkeit",
     "blockers": ["Seriennummer nicht mit Auslesungen verknüpft", "Reparaturen ohne Seriennummer"],
     "requires": [("serial_number", "pflicht", None, "PB-10"), ("usage_data", "pflicht", None, "PB-10"),
                  ("service_event", "pflicht", None, "PB-10"), ("config", "optional", None, None)],
     "links": [("serial_number", "usage_data"), ("serial_number", "service_event")], "learnable": True, "max_stage": 3,
     "note": "gerechnet auf SCANIA Component X"},
    {"id": "service_contracts", "name": "Serviceverträge", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Abschluss bzw. Verlängerung eines Servicevertrags – hier: risikobasierte Vertragskalkulation",
     "label_detail": "Ohne Vertragsdaten ist Abschluss oder Verlängerung nicht messbar. Gezeigt wird die erwartete Reparaturquote je "
                     "Konfiguration aus dem Modell der installierten Basis als Grundlage für den Vertragspreis.",
     "window_days": 0, "min_positives": 100, "entity": "serial",
     "baseline": "Pauschalpreis je Vertrag", "lever": "Serviceumsatz, Marge",
     "blockers": ["Vertragsdaten (Abschluss, Laufzeit, Kündigung) nicht im Export"],
     "requires": [("serial_number", "pflicht", None, "PB-10"), ("service_event", "pflicht", None, "PB-10"),
                  ("contract", "messung", None, "PB-10"), ("config", "optional", None, None), ("usage_data", "optional", None, None)],
     "links": [("serial_number", "service_event")], "learnable": False, "max_stage": 3, "cap": "warn",
     "nolabel": {"label": "Kein lernbares Label: Vertragsdaten (Abschluss, Laufzeit, Kündigung) fehlen – nur Kalkulation aus dem "
                          "Reparaturmodell der installierten Basis.",
                 "action": "Servicevertragsdaten je Seriennummer exportieren (Beginn, Ende, Kündigung, Preis)",
                 "model": "Kein lernbares Label – Vertragspreis aus der erwarteten Reparaturquote je Konfiguration.",
                 "engine": {"primary": "Reparaturmodell der installierten Basis (LightGBM) → Risikoprämie je Konfiguration",
                            "challengers": ["Churn-Modell für Verlängerung, sobald Vertragsdaten vorliegen"],
                            "baseline": "Pauschalpreis", "reason": "Ohne Vertragsdaten keine Trainingsbasis für Abschluss/Verlängerung"}},
     "note": "Kalkulation · SCANIA, Vertragsdaten fehlen"},
    {"id": "spare_parts_pricing", "name": "Ersatzteilbedarf & -preise", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Absatz je Komponente im Folgequartal; Preisspielraum bei Teilen, deren Bedarf der installierten Basis folgt",
     "label_detail": "Installierte Basis je Teil = verkaufte Fahrräder × Stückmenge laut Stückliste. Bewertet mit WAPE auf den letzten "
                     "beiden Quartalen; gebundene Teile (Korrelation Bedarf ↔ installierte Basis ≥ 0,5) tragen den Preishebel.",
     "window_days": 91, "min_positives": 0, "entity": "series",
     "baseline": "Menge des Vorquartals", "lever": "Ersatzteilmarge",
     "blockers": ["Stückliste nicht gepflegt", "Ersatzteilverkäufe ohne Artikelbezug"],
     "requires": [("product_key", "pflicht", None, None), ("quantity", "pflicht", None, None), ("event_time", "pflicht", None, None),
                  ("bom", "pflicht", None, "PB-10"), ("price", "optional", None, None), ("product_group", "optional", None, None)],
     "links": [("product_key", "quantity")], "learnable": True, "max_stage": 3, "note": "gerechnet auf AdventureWorks"},
    {"id": "quality_root_cause", "name": "Qualitäts-Frühwarnung & Root Cause", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Fertigungsauftrag mit Ausschuss – Frühwarnung vor dem Start, Ursachen nach Grund, Produktgruppe und Bauteil-Lieferant",
     "label_detail": "Ersatz für Garantiefälle (die es offen nicht gibt): Ausschuss ist der früheste messbare Qualitätsfehler. "
                     "Quoten zählen nur Aufträge, die vor dem Start abgeschlossen waren. Test: Starts ab 01.01.2025.",
     "window_days": 0, "min_positives": 100, "entity": "order",
     "baseline": "größte Auftragsmenge zuerst", "lever": "Kosten (Ausschuss, Garantie)",
     "blockers": ["Ausschussgrund nur als Freitext", "Bauteil-Lieferant je Charge nicht protokolliert"],
     "requires": [("product_key", "pflicht", None, None), ("quantity", "pflicht", None, None), ("defect_code", "pflicht", None, "PB-05"),
                  ("event_time", "pflicht", None, None), ("bom", "optional", None, None), ("supplier", "optional", None, None)],
     "links": [("product_key", "defect_code")], "learnable": True, "max_stage": 3, "note": "gerechnet auf AdventureWorks (Ausschuss)"},
    {"id": "supplier_recourse", "name": "Lieferantenregress", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Bestellposition wird beim Wareneingang beanstandet – gezielt prüfen und beim Lieferanten geltend machen",
     "label_detail": "Positiv = beanstandete Menge > 0. Beanstandungen zählen erst ab Fälligkeit (≈ Wareneingang). "
                     "Test: Bestellungen ab 01.04.2025. Regress-Volumen = beanstandete Menge × Stückpreis.",
     "window_days": 0, "min_positives": 100, "entity": "line",
     "baseline": "Lieferanten mit der höchsten bisherigen Beanstandungsquote zuerst", "lever": "Kosten, Regress",
     "blockers": ["Beanstandungen nicht je Bestellposition erfasst", "Garantiefälle ohne Bauteil-/Lieferantenbezug"],
     "requires": [("supplier", "pflicht", None, "PB-10"), ("product_key", "pflicht", None, None), ("quantity", "pflicht", None, None),
                  ("rejected_qty", "pflicht", None, "PB-10"), ("event_time", "pflicht", None, None), ("price", "optional", None, None),
                  ("bom", "optional", None, None)],
     "links": [("supplier", "product_key")], "learnable": True, "max_stage": 3, "note": "gerechnet auf AdventureWorks (Wareneingang)"},
    {"id": "predictive_maintenance", "name": "Predictive Maintenance", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Einheit fällt in den nächsten 30 Zyklen aus – Wartung einplanen, bevor es ungeplant wird",
     "label_detail": "Schnitt an einem Betriebszyklus je Triebwerk; positiv, wenn die Restlebensdauer ≤ 30 Zyklen ist. "
                     "Test auf anderen Triebwerken (keine Kalenderzeit). Peer-Normalwerte nur aus Trainings-Triebwerken.",
     "window_days": 0, "min_positives": 100, "entity": "serial",
     "baseline": "meiste Zyklen zuerst", "lever": "Stillstand, Servicekosten",
     "blockers": ["Telemetrie nicht je Seriennummer", "Ausfälle nicht dokumentiert (nur Reparaturen)"],
     "requires": [("serial_number", "pflicht", None, "PB-10"), ("usage_data", "pflicht", None, "PB-10"),
                  ("failure_event", "pflicht", None, "PB-10"), ("config", "optional", None, None)],
     "links": [("serial_number", "usage_data")], "learnable": True, "max_stage": 3, "note": "gerechnet auf NASA C-MAPSS"},
    {"id": "spare_parts_demand", "name": "Ersatzteilbedarf & Bestände", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Bedarf je Lagerartikel im Folgemonat – Grundlage für Sicherheitsbestand und Nachbestellung",
     "label_detail": "Bestellmengen je Artikel und Monat; Test auf den letzten 4 Monaten, WAPE wertgewichtet. Dazu Bestandsampel: "
                     "Reichweite des Endbestands gegen Lieferzeit.",
     "window_days": 30, "min_positives": 0, "entity": "series",
     "baseline": "Ø der letzten 3 Monate", "lever": "Bestand, Kapitalbindung",
     "blockers": ["Bestände nicht historisiert", "Lieferzeiten nicht gepflegt"],
     "requires": [("product_key", "pflicht", None, None), ("quantity", "pflicht", None, None), ("event_time", "pflicht", None, None),
                  ("product_group", "optional", None, None), ("customer_key", "optional", None, None), ("supplier", "optional", None, None)],
     "links": [("product_key", "quantity")], "learnable": True, "max_stage": 3, "note": "gerechnet auf WideWorldImporters"},
    {"id": "first_time_fix", "name": "First-Time-Fix im Außendienst", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Benötigte Teile vor dem Einsatz – Einsatz ohne Folgebesuch",
     "label_detail": "Braucht Serviceaufträge mit Fehlercode, Techniker, verbauten Teilen und Folgebesuch ja/nein. "
                     "Kein offener Datensatz vorhanden – bewertet wird nur die Struktur.",
     "window_days": 0, "min_positives": 100, "entity": "order",
     "baseline": "Standard-Teilekoffer je Fehlercode", "lever": "Servicekosten, Kundenzufriedenheit",
     "blockers": ["Teileverbrauch nicht je Serviceauftrag", "Folgebesuche nicht verknüpft"],
     "requires": [("service_event", "pflicht", None, "PB-10"), ("defect_code", "pflicht", None, "PB-05"),
                  ("technician", "pflicht", None, "PB-10"), ("product_key", "optional", None, None), ("serial_number", "optional", None, None)],
     "links": [("service_event", "defect_code")], "learnable": True, "max_stage": 1, "note": "keine offenen Daten – nur Struktur"},
    {"id": "win_probability", "name": "Gewinnwahrscheinlichkeit & Umsatzprognose", "area": "Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Auftrag wird nicht binnen 30 Tagen zu Umsatz (Lieferung + Rechnung) – plus Pipeline-Umsatz je Woche",
     "label_detail": "Ohne Angebotsdaten gemessen an Rückstandsaufträgen (WWI): positiv, wenn binnen 30 Tagen keine Rechnung folgt. "
                     "Umsatzprognose = Σ Auftragswert × Lieferwahrscheinlichkeit je Woche, verglichen mit dem tatsächlich fakturierten Wert.",
     "window_days": 30, "min_positives": 100, "entity": "order",
     "baseline": "größte Rückstandsmenge zuerst", "lever": "Umsatz, Planbarkeit",
     "blockers": ["Angebote/Opportunities nicht im Export", "Auftrag ohne Bezug zur Rechnung"],
     "requires": [("order_key", "pflicht", None, None), ("customer_key", "pflicht", None, "PB-01"), ("event_time", "pflicht", None, None),
                  ("product_key", "pflicht", None, None), ("quantity", "pflicht", None, None), ("account_group", "optional", None, None),
                  ("supplier", "optional", None, None)],
     "links": [("order_key", "customer_key")], "learnable": True, "max_stage": 3, "note": "gerechnet auf WideWorldImporters (Rückstände)"},
    {"id": "service_triage", "name": "Service-Triage & Diagnose", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Kategorie und Lösung je Servicemeldung aus dem Fehlertext",
     "label_detail": "Braucht Servicemeldungen mit Freitext und gelöster Kategorie. Eher ein Sprachmodell-Thema als Graph ML; "
                     "kein offener Datensatz mit Texten vorhanden (WWI- und AdventureWorks-Kommentare sind leer).",
     "window_days": 0, "min_positives": 100, "entity": "ticket",
     "baseline": "Schlagwortregeln", "lever": "Produktivität im Service",
     "blockers": ["Fehlertexte nicht exportiert", "Lösungskategorie nicht gepflegt"],
     "requires": [("service_event", "pflicht", None, "PB-10"), ("service_text", "pflicht", None, "PB-05"),
                  ("defect_code", "optional", None, "PB-05"), ("serial_number", "optional", None, None)],
     "links": [("service_event", "service_text")], "learnable": True, "max_stage": 1, "note": "Sprachmodell-Thema · keine offenen Texte"},
    {"id": "data_prediction", "name": "Data Prediction · Stammdaten-Vorschläge", "area": "Stammdaten", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Je manuell gepflegtem Feld ein Use Case: Wert vorschlagen statt eintippen (z. B. Ersatzteil ja/nein)",
     "label_detail": "Sammel-Kachel der Data-Classifier-Analyse aus Schritt 3: jedes Kennzeichen, jede Klassifikation und jeder "
                     "Parameter, den Regel oder ML sicher vorschlagen kann, zählt als eigener Use Case.",
     "window_days": 0, "min_positives": 0, "entity": "field",
     "baseline": "Regel je Feld (häufigster Wert je bestem Einzelfeld)", "lever": "Pflegeaufwand, Datenqualität, Bestand",
     "blockers": ["Manuelle Felder ohne Historie", "Folgefelder statt Ursachen im Datenmodell"],
     "requires": [("product_key", "optional", None, None)], "links": [], "learnable": True, "max_stage": 3,
     "note": "Anzahl Feld-Use-Cases je Datensatz"},
    {"id": "warranty_fraud", "name": "Garantiebetrug & Fehlbuchungen", "area": "After Sales", "lighthouse": False,
     "app_route": None, "app_name": "neu",
     "label": "Auffälliger Garantiefall (Händler–Techniker–Teil-Muster)",
     "label_detail": "Braucht Garantiefälle mit Händler, Techniker, getauschten Teilen und Prüfergebnis (Betrug ja/nein). "
                     "Kein offener Datensatz vorhanden – bewertet wird nur die Struktur.",
     "window_days": 0, "min_positives": 100, "entity": "claim",
     "baseline": "Regeln (Kosten über Schwelle, Häufung je Händler)", "lever": "Garantiekosten",
     "blockers": ["Garantiefälle ohne Techniker/Teile", "Prüfergebnis (Betrug ja/nein) nicht erfasst"],
     "requires": [("warranty_claim", "pflicht", None, "PB-10"), ("serial_number", "pflicht", None, "PB-10"),
                  ("technician", "pflicht", None, "PB-10"), ("customer_key", "optional", None, None), ("product_key", "optional", None, None),
                  ("defect_code", "optional", None, "PB-05")],
     "links": [("warranty_claim", "serial_number")], "learnable": True, "max_stage": 1, "note": "keine offenen Daten – nur Struktur"},
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
    "price_recommendation": {
        "v2": "Preis- und Rabattempfehlung", "effect": "+2 bis 7 Prozentpunkte Umsatzrendite", "fit": "mittel",
        "graph": "Was Peers für denselben Artikel im selben Gebiet zahlen, was der Händler bisher bekam und welches Preisniveau "
                 "im Portfolio seines Vertrieblers üblich ist – Signale über Händler → Artikel → andere Händler bzw. Händler → Vertriebler.",
        "path": ["Händler", "Artikel", "Peer-Händler im Gebiet", "erzielter Preis"],
        "tabular": "Artikel, Kategorie, Menge und Mengenstaffel, Listenpreis, Gebiet",
        "unit": "je Händlerposition: Nettopreis in % vom Listenpreis zum Angebotszeitpunkt",
        "euro": "Preislücke zum erwartbaren Preis × geschlossener Anteil − geprüfte Aufträge × Kosten, × 2 Halbjahre"},
    "price_leakage": {
        "v2": "Preis-Leakage aufdecken", "effect": "über 6 % des Umsatzes versickern", "fit": "gering",
        "graph": "Leakage ist Abweichung vom Peer-Niveau: gleicher Artikel, ähnliche Menge, gleiches Gebiet. Die Peer-Preise "
                 "kommen aus dem Händler–Artikel-Graphen; gemessen hilft hier aber vor allem ein genaueres Tabellenmodell (TabPFN).",
        "path": ["Position", "Artikel", "Peer-Positionen", "Preislücke"],
        "tabular": "erwartbarer Preis aus der Preisempfehlung",
        "unit": "je Händlerposition: Preis > 3 Pp. unter dem erwartbaren",
        "euro": "Leakage im Testzeitraum × vermiedener Anteil − geprüfte Aufträge × Kosten, × 2 Halbjahre"},
    "quote_check": {
        "v2": "Konfigurations- und Angebotsprüfung", "effect": "62 % melden Margenverlust zwischen Angebot und Lieferung", "fit": "gering",
        "graph": "Konfiguration → Stückliste → Bauteile → Kosten: die Marge entsteht entlang der Stückliste. Mit bekannten Kosten "
                 "ist die Prüfung eine Regel – Graph ML hilft erst, wenn Kosten im Angebot fehlen und geschätzt werden müssen.",
        "path": ["Angebot", "Konfiguration", "Bauteile (Stückliste)", "Kosten"],
        "tabular": "Nettopreis, Standardkosten zum Datum, Sonderangebot",
        "unit": "je Auftrag: Marge unter 10 %",
        "euro": "Margenlücke bis 10 % in markierten Aufträgen × zurückgeholter Anteil − geprüfte Angebote × Kosten, × 2"},
    "installed_base": {
        "v2": "Installierte Basis analysieren", "effect": "Serviceumsatz +10–20 % durch proaktive Angebote", "fit": "mittel",
        "graph": "Fahrzeuge mit gleicher Konfiguration fallen ähnlich aus: Fahrzeug → Konfigurationswert → andere Fahrzeuge. "
                 "Die Reparaturquote dieser Peers (nur aus Trainingsfahrzeugen) und ein GNN über den Graphen Fahrzeug ↔ "
                 "Konfiguration ergänzen die eigenen Auslesungen. Gemessen: Die Auslesungen tragen das Signal – LightGBM gewinnt, "
                 "Konfigurations-Peers und GNN bringen auf SCANIA nichts zusätzlich (Konfiguration erklärt Ausfälle kaum).",
        "path": ["Fahrzeug", "Konfiguration", "Peer-Fahrzeuge", "Reparatur"],
        "tabular": "Werkstatt-Auslesungen (Zähler, Histogramme) zum Schnitt, Zuwachs seit der vorigen Auslesung, Laufzeit",
        "unit": "je Fahrzeug und Auslesung: Reparatur der Komponente X in den nächsten 48 Zeiteinheiten",
        "euro": "erkannte Reparaturen in der Top-Liste × Auftragswert (Annahme 2.500 €) × Abschlussquote des proaktiven Angebots "
                "× Ergebnisanteil − Angebote × Kosten, × Runden pro Jahr"},
    "service_contracts": {
        "v2": "Serviceverträge", "effect": "wiederkehrender Umsatz, Marge planbarer", "fit": "mittel",
        "graph": "Der Vertragspreis folgt dem erwarteten Servicebedarf der Konfiguration – derselbe Graph wie bei der "
                 "installierten Basis. Für Abschluss und Verlängerung bräuchte es zusätzlich Vertragsdaten (Kunde → Vertrag → Maschine).",
        "path": ["Vertrag", "Fahrzeug", "Konfiguration", "Reparaturrisiko"],
        "tabular": "Laufzeit, Auslesungen, Konfiguration je Fahrzeug",
        "unit": "je Konfigurationssegment: erwartete Reparaturquote → Risikoprämie",
        "euro": "nicht bewertbar ohne Vertragsdaten – Kalkulation: erwartete Reparaturquote × Auftragswert × 1,2 Sicherheitsaufschlag"},
    "spare_parts_pricing": {
        "v2": "Ersatzteilpreise optimieren", "effect": "+2 bis 5 % Ersatzteilmarge bei gebundenen Teilen", "fit": "mittel",
        "graph": "Der Teilebedarf folgt der installierten Basis: Teil → Stückliste → Fahrräder → verkaufte Stück. Dieser Pfad "
                 "läuft über drei Tabellen (Stückliste, Auftragspositionen, Produkte) und fehlt jeder Zeile der Teile-Historie. "
                 "Gemessen: leicht besser als ohne (WAPE 0,27 statt 0,29).",
        "path": ["Ersatzteil", "Stückliste", "Fahrrad", "verkaufte Räder"],
        "tabular": "Menge Vorquartal, Ø vier Quartale, Vorjahresquartal",
        "unit": "je Komponente: Absatz im Folgequartal",
        "euro": "Umsatz gebundener Teile × Preisaufschlag × verbleibende Menge (Elastizität) − Pflegekosten, hochgerechnet auf ein Jahr"},
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
                 "Peer-Niveau liegen. Ohne Einstandspreis keine Marge – deshalb auf Online Retail II nur Profil. Abgleich v2: "
                 "gemessen in Preis- & Rabattempfehlung, Preis-Leakage und Angebotsprüfung (AdventureWorks mit Kosten zum Datum).",
        "path": ["Kunde", "Position", "Artikel", "Peer-Kunden"],
        "tabular": "Preisstreuung je Artikel und Kunde",
        "unit": "je Position: Preis unter Peer-Niveau (Leakage)",
        "euro": "Umsatz der Leakage-Positionen × schließbare Lücke zum Peer-Preis (Prompt 5)"},
    "returns_root_cause": {
        "v2": "Garantie: Frühwarnung und Root Cause (Retouren)", "effect": "Garantie-/Retourenkosten −4 bis −20 %",
        "fit": "hoch",
        "graph": "Stornos häufen sich bei Artikeln, die dieselben Kunden kaufen (Position → Artikel → Kunden → "
                 "Nachbarartikel) – die Retourenquote der Nachbarn ist ein Graph-Merkmal. Abgleich v2: bleibt auf Online Retail II "
                 "(WWI hat keine Gutschriften, AdventureWorks keine Retouren); Garantie-Pendant ist die Qualitäts-Frühwarnung.",
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
        "euro": "vermiedene Garantiefälle × Ø Kosten je Fall – gemessen ersatzweise als Qualitäts-Frühwarnung (Ausschuss, AdventureWorks)"},
    "quality_root_cause": {
        "v2": "Garantie: Frühwarnung und Root Cause (über Ausschuss)", "effect": "Garantie- und Qualitätskosten −4 bis −20 %", "fit": "mittel",
        "graph": "Fehler können sich fortpflanzen: Lieferant → Bauteil → Stückliste → Produkt → Fertigungsauftrag. Graph-Merkmale: "
                 "Ausschuss der Geschwister-Produkte und der Produkte mit gleichen Bauteilen, Beanstandungsquote der Bauteil-Lieferanten. "
                 "Gemessen auf AdventureWorks: Ausschuss entsteht nur bei großen Aufträgen und folgt nicht der Lieferantenqualität – "
                 "die Regel „große Aufträge zuerst“ ist so gut wie jedes Modell.",
        "path": ["Lieferant", "Bauteil", "Produkt", "Fertigungsauftrag"],
        "tabular": "Auftragsmenge, Fertigungstage, Kosten, Arbeitsgänge, bisherige Ausschussquote des Produkts",
        "unit": "je Fertigungsauftrag: Ausschuss ja/nein, Ursachen nach Grund und Produktgruppe",
        "euro": "Ausschusskosten (Menge × Standardkosten) in der Top-Liste × vermiedener Anteil − Prüfungen × Kosten, × 2 Halbjahre"},
    "supplier_recourse": {
        "v2": "Lieferantenregress", "effect": "Zulieferer tragen ~10 % der Garantiekosten, fair wären bis 37 %", "fit": "mittel",
        "graph": "Beanstandungen hängen am Lieferanten und am Bauteil: dasselbe Bauteil bei anderen Lieferanten, die Bauteilgruppe, "
                 "Lieferanten mit gleichen Bauteilen (Lieferant → Bauteil → Lieferant). Gemessen: Die eigene Historie des Lieferanten "
                 "trägt das Signal, ein Tabellen-Foundation-Model (TabPFN) nutzt sie am besten – die Peers bringen kaum etwas.",
        "path": ["Bestellposition", "Lieferant", "Bauteil", "andere Lieferanten"],
        "tabular": "Bestellmenge, Preis, Lieferzeit, Bonität, bisherige Beanstandungsquote des Lieferanten",
        "unit": "je Bestellposition: Beanstandung beim Wareneingang",
        "euro": "beanstandeter Warenwert in der Prüfliste × zurückgeholter Anteil (Regress inkl. Folgekosten) − Prüfungen × Kosten, × 2"},
    "predictive_maintenance": {
        "v2": "Predictive Maintenance als Service", "effect": "ungeplanter Stillstand −30 bis −50 %", "fit": "mittel",
        "graph": "Was „normal“ ist, hängt von der Betriebsbedingung ab: Schnitt → Bedingung → Peer-Triebwerke liefert den Normalwert, "
                 "Schnitt → Flotte → Lebensdauer der Peers das Ausfallrisiko in diesem Alter. Gemessen: Die Peer-Normierung hilft "
                 "LightGBM leicht, am besten ist TabPFN; ein gelerntes GNN bleibt deutlich zurück – oft genügt Zeitreihe + Baum.",
        "path": ["Triebwerk", "Betriebsbedingung", "Peer-Triebwerke", "Normalwert"],
        "tabular": "21 Sensoren, Ø und Änderung über 10 Zyklen, Zyklenzahl, Einstellungen",
        "unit": "je Triebwerk und Zyklus: Ausfall in den nächsten 30 Zyklen",
        "euro": "erkannte Ausfälle in der Top-Liste × Mehrkosten ungeplanter Ausfall (Annahme 50.000 €) × vermiedener Anteil − "
                "Inspektionen × Kosten (Testflotte über ihre ganze Lebensdauer)"},
    "spare_parts_demand": {
        "v2": "Ersatzteilbedarf und Bestände", "effect": "Bestand −20 bis −27 %", "fit": "mittel",
        "graph": "Artikel derselben Warengruppe und Artikel, die dieselben Kunden kaufen, bewegen sich gemeinsam (Artikel → "
                 "Warengruppe/Kunden → andere Artikel). Gemessen: LightGBM mit dem eigenen Verlauf gewinnt, die Graph-Signale bringen nichts.",
        "path": ["Artikel", "Warengruppe", "Geschwister-Artikel", "Trend"],
        "tabular": "Vormonat, Ø 3 und 12 Monate, Vorjahresmonat, Trend, Preis",
        "unit": "je Lagerartikel: Menge im Folgemonat",
        "euro": "Sicherheitsbestand × relative Fehlerreduktion ggü. Regel × Lagerhaltungskostensatz"},
    "first_time_fix": {
        "v2": "First-Time-Fix im Außendienst", "effect": "Fehlbesuch: 1,6 Zusatzfahrten à 200–300 $", "fit": "mittel",
        "graph": "Welche Teile ein Einsatz braucht, zeigt das Muster ähnlicher Anlagen: Fehlercode → frühere Aufträge → verbaute Teile, "
                 "Anlage → Konfiguration → Peer-Anlagen. Ohne offene Serviceaufträge bleibt es bei der Struktur.",
        "path": ["Serviceauftrag", "Fehlercode", "frühere Aufträge", "Teile"],
        "tabular": "Fehlercode, Anlagentyp, Alter, Techniker",
        "unit": "je Serviceauftrag: benötigte Teile",
        "euro": "vermiedene Folgebesuche × Kosten je Besuch − Mehrbestand im Fahrzeug (sobald Serviceaufträge mit Teilen vorliegen)"},
    "win_probability": {
        "v2": "Gewinnwahrscheinlichkeit und Umsatzprognose", "effect": "nur 45 % der Vertriebsleiter vertrauen ihrer Prognose", "fit": "mittel",
        "graph": "Ob ein Auftrag zu Umsatz wird, hängt am Artikel und seinem Nachschub: Auftrag → Artikel → andere wartende Aufträge, "
                 "Artikel → Lieferant → dessen Lieferquote, Artikel → offene Einkaufsbestellungen. Gemessen auf WWI: Der Generator "
                 "folgt fast einer Mengenregel (ab ~50 Stück bleibt der Rückstand liegen) – Modelle sind nur +12 % besser, die "
                 "Graph-Merkmale verbessern aber die Wochenprognose (WAPE 0,15 → 0,11).",
        "path": ["Auftrag", "Artikel", "Lieferant", "Nachschub"],
        "tabular": "Auftragswert, Menge, Kunde, Kreditlimit, bisherige Quote des Kunden",
        "unit": "je Rückstandsauftrag: Umsatz binnen 30 Tagen gefährdet; je Woche: Pipeline-Umsatz",
        "euro": "gefährdeter Auftragswert in der Top-Liste × geretteter Anteil (Nachschub, Ersatzartikel) × Marge − Aktionen × Kosten, × 2"},
    "service_triage": {
        "v2": "Service-Triage und Diagnose", "effect": "Produktivität +30–45 % (GenAI)", "fit": "gering",
        "graph": "Der Fehlertext trägt das Signal – das ist ein Sprachmodell-Thema. Graph hilft wenig (ähnliche Anlagen → frühere "
                 "Lösungen als Zusatzmerkmal). Ehrlich Rot/Gelb; ohne offene Servicetexte nur Struktur.",
        "path": ["Meldung", "Fehlertext", "ähnliche Meldungen", "Lösung"],
        "tabular": "Fehlercode, Produkt, Alter",
        "unit": "je Servicemeldung: Kategorie/Lösung",
        "euro": "eingesparte Bearbeitungszeit je Meldung × Meldungen × Stundensatz (sobald Texte und Lösungen vorliegen)"},
    "data_prediction": {
        "v2": "Data Prediction (Stammdaten-Vorschläge)", "effect": "Pflegeaufwand und Fehleingaben sinken, Bestände folgen sauberen Stammdaten",
        "fit": "hoch",
        "graph": "Ein manuell gepflegter Wert hängt selten nur an der eigenen Zeile: Stückliste (Vater, Geschwister), Lieferant je "
                 "Artikel, Warengruppe und Benennung tragen ihn. Die Kaskade prüft je Feld Regel, LightGBM, Graph-Kontext, Text und "
                 "TabPFN und weist Folgefelder (Leakage) aus.",
        "path": ["Feld", "eigener Datensatz", "verbundene Stammdaten", "ähnliche Datensätze"],
        "tabular": "übrige Felder derselben Zeile",
        "unit": "je Feld: automatisch entscheidbarer Anteil bei ≥ 98 % Trefferquote",
        "euro": "Eingaben pro Jahr × Minuten je Eingabe × Stundensatz × automatisch entschiedener Anteil, summiert über die Feld-Use-Cases"},
    "warranty_fraud": {
        "v2": "Garantiebetrug und Fehlbuchungen", "effect": "3–15 % der Garantiekosten", "fit": "hoch",
        "graph": "Betrug tritt in Ringen auf: dieselben Händler, Techniker und Teile wiederholen sich (Fall → Händler → Techniker → Teil). "
                 "Genau diese Muster sieht ein Graph-Modell, eine einzelne Fallzeile nicht. Ohne offene Daten bleibt es bei der Struktur.",
        "path": ["Garantiefall", "Händler", "Techniker", "Teil"],
        "tabular": "Kosten, Laufzeit, Teileanzahl, Fehlercode je Fall",
        "unit": "je Garantiefall: auffällig ja/nein",
        "euro": "aufgedeckte Fehlbuchungen × Ø Kosten je Fall − Prüfungen × Kosten (sobald Garantiefälle mit Prüfergebnis vorliegen)"},
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
