"""Generiert synthetische Angebots-/Margendaten für die Margenoptimierungs-Demo."""
import random
import csv
from pathlib import Path
from datetime import date, timedelta

random.seed(42)

KUNDEN = ["Müller GmbH", "Weber AG", "Schmidt & Co.", "Braun Industrie", "Fischer Tech"]
PRODUKTE = [
    ("Hydraulikpumpe",    "Maschinen",  2800, 0.58),
    ("Steuerungsmodul",   "Elektronik", 1200, 0.52),
    ("Wartungsvertrag",   "Service",     900, 0.72),
    ("Präzisionssensor",  "Elektronik",  480, 0.55),
]
MITARBEITER = [
    ("Meyer",    0.18),   # rabattiert systematisch mehr
    ("Schulz",   0.08),
    ("Hoffmann", 0.12),
]
REGIONEN = ["Nord", "Süd"]

start = date(2023, 1, 1)

rows = []
for i in range(1, 201):
    kunde      = random.choice(KUNDEN)
    prod_name, kategorie, listenpreis, kostenpct = random.choice(PRODUKTE)
    mitarb, base_rabatt = random.choice(MITARBEITER)
    region     = random.choice(REGIONEN)
    menge      = random.randint(1, 20)
    datum      = start + timedelta(days=random.randint(0, 730))

    # Rabatt mit individuellem Rauschen
    rabatt = round(max(0, min(0.40, base_rabatt + random.gauss(0, 0.05))), 3)
    verkaufspreis = round(listenpreis * (1 - rabatt), 2)
    herstellkosten = round(listenpreis * kostenpct, 2)
    deckungsbeitrag = round((verkaufspreis - herstellkosten) / verkaufspreis * 100, 1)
    umsatz = round(verkaufspreis * menge, 2)

    rows.append({
        "angebot_id":          f"ANG-{i:04d}",
        "datum":               datum.isoformat(),
        "kunde":               kunde,
        "produkt":             prod_name,
        "kategorie":           kategorie,
        "listenpreis":         listenpreis,
        "verkaufspreis":       verkaufspreis,
        "rabatt_pct":          round(rabatt * 100, 1),
        "menge":               menge,
        "umsatz":              umsatz,
        "deckungsbeitrag_pct": deckungsbeitrag,
        "vertriebsmitarbeiter": mitarb,
        "region":              region,
    })

out = Path(__file__).parent / "margen_beispiel.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ {len(rows)} Zeilen → {out}")
