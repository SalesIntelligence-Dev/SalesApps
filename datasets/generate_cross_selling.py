"""Synthetische B2B-Transaktionsdaten für Cross-Selling-Demo."""
import random, csv
from pathlib import Path
from datetime import date, timedelta

random.seed(7)

KUNDEN = [
    ("K01", "Müller GmbH"),
    ("K02", "Weber AG"),
    ("K03", "Schmidt & Co."),
    ("K04", "Braun Industrie"),
    ("K05", "Fischer Tech"),
    ("K06", "Klein Systems"),
    ("K07", "Groß Automation"),
    ("K08", "Ritter Technik"),
]

PRODUKTE = [
    # (id, name, kategorie, preis_min, preis_max)
    ("P01", "Hydraulikpumpe HX-200",   "Maschinen", 2400, 3200),
    ("P02", "Kompressor KA-500",        "Maschinen", 1800, 2600),
    ("P03", "Druckluftanlage DA-100",   "Maschinen", 3500, 4800),
    ("P04", "Förderpumpe FP-350",       "Maschinen", 2100, 2900),
    ("P05", "Steuerungsmodul SM-Pro",   "Zubehör",    900, 1400),
    ("P06", "Präzisionssensor PS-200",  "Zubehör",    450,  700),
    ("P07", "Druckregler DR-80",        "Zubehör",    320,  480),
    ("P08", "Schlauchpaket SP-Set",     "Zubehör",    180,  280),
    ("P09", "Wartungsvertrag Basic",    "Service",    600,  800),
    ("P10", "Wartungsvertrag Premium",  "Service",   1100, 1500),
    ("P11", "Inbetriebnahme & Setup",   "Service",    700,  950),
    ("P12", "Anwenderschulung",         "Service",    500,  750),
]

# Kaufprofile: Welcher Kunde kauft welche Produkte (mit relativer Häufigkeit)
PROFILE = {
    "K01": ["P01", "P01", "P05", "P09", "P11"],        # Hydraulik-Fokus + Wartung
    "K02": ["P02", "P05", "P06", "P10", "P10"],        # Kompressor + Sensor + Premium
    "K03": ["P01", "P01", "P07", "P08", "P09"],        # Pumpe + Zubehör + Basic
    "K04": ["P03", "P05", "P06", "P10", "P11", "P12"], # Full-Stack Druckluft
    "K05": ["P02", "P06", "P07", "P12"],                # Kompressor + Sensor + Schulung
    "K06": ["P01", "P05", "P11", "P12"],                # Pumpe + Steuerung + Service
    "K07": ["P04", "P05", "P06", "P10", "P12"],        # Förderpumpe + Premium
    "K08": ["P02", "P05", "P09", "P11"],                # Kompressor + Steuerung + Basic
}

START = date(2023, 1, 1)
prod_map = {p[0]: p for p in PRODUKTE}

rows = []
tx_id = 1
for kid, kname in KUNDEN:
    pool = PROFILE[kid]
    # Zwischen 15 und 22 Einzeltransaktionen pro Kunde
    n_tx = random.randint(15, 22)
    for _ in range(n_tx):
        pid = random.choice(pool)
        _, pname, kat, pmin, pmax = prod_map[pid]
        datum = START + timedelta(days=random.randint(0, 700))
        menge = random.randint(1, 4)
        ep    = random.randint(pmin, pmax)
        rows.append({
            "transaktions_id": f"TX-{tx_id:04d}",
            "datum":      datum.isoformat(),
            "kunde_id":   kid,
            "kunde_name": kname,
            "produkt_id": pid,
            "produkt_name": pname,
            "kategorie":  kat,
            "menge":      menge,
            "einzelpreis": ep,
            "umsatz":     ep * menge,
        })
        tx_id += 1

# Shuffle und auf 150 Zeilen begrenzen (mehr gibt schönere Dichte)
random.shuffle(rows)
rows = sorted(rows[:150], key=lambda r: r["datum"])

out = Path(__file__).parent / "cross_selling_beispiel.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"✓ {len(rows)} Zeilen → {out}")
