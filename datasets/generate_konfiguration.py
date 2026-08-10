"""Synthetische Maschinenkonfigurations-Daten für Graph-Demo."""
import csv, random
from pathlib import Path

random.seed(42)

OUT = Path(__file__).parent

# ── Stückliste (60 Komponenten) ────────────────────────────────────────────

STUECKLISTE = [
    # id, name, kategorie, listenpreis, herstellkosten, kompatibel_mit (filled below)
    # Basis (10)
    ("B01", "Maschinenrahmen Kompakt MK-100",    "Basis",     4200, 2100),
    ("B02", "Maschinenrahmen Standard MS-200",   "Basis",     5800, 2900),
    ("B03", "Maschinenrahmen Heavy MH-400",      "Basis",     8400, 4200),
    ("B04", "Basismodul Automation BA-150",      "Basis",     6200, 3100),
    ("B05", "Basismodul Prozess BP-250",         "Basis",     7100, 3600),
    ("B06", "Grundgestell Kompakt GK-80",        "Basis",     3900, 1950),
    ("B07", "Plattformmodul Universal PM-300",   "Basis",     9200, 4600),
    ("B08", "Trägerrahmen Mittel TR-180",        "Basis",     5400, 2700),
    ("B09", "Modulrahmen Industrie MI-350",      "Basis",     7800, 3900),
    ("B10", "Grundgestell Schwer GS-500",        "Basis",    10200, 5100),

    # Antrieb (14)
    ("A01", "Servomotor 2,5 kW SM-250",         "Antrieb",   2800, 1400),
    ("A02", "Servomotor 5 kW SM-500",           "Antrieb",   3900, 1950),
    ("A03", "Servomotor 7,5 kW SM-750",         "Antrieb",   5200, 2600),
    ("A04", "Servomotor 11 kW SM-1100",         "Antrieb",   7400, 3700),
    ("A05", "Frequenzumrichter 4 kW FU-400",    "Antrieb",   1800,  900),
    ("A06", "Frequenzumrichter 7,5 kW FU-750",  "Antrieb",   2400, 1200),
    ("A07", "Frequenzumrichter 15 kW FU-1500",  "Antrieb",   3200, 1600),
    ("A08", "Hydraulikantrieb 200 bar HA-200",  "Antrieb",   4600, 2300),
    ("A09", "Hydraulikantrieb 400 bar HA-400",  "Antrieb",   6800, 3400),
    ("A10", "Direktantrieb 3 kW DA-300",        "Antrieb",   7200, 3600),
    ("A11", "Linearmotor 500 N LM-500",         "Antrieb",   8900, 4450),
    ("A12", "Schrittmotor 1,5 Nm SK-150",       "Antrieb",   1200,  600),
    ("A13", "Getriebemotor 3,5 kW GR-350",      "Antrieb",   2200, 1100),
    ("A14", "Torquemotor 8 kW TM-800",          "Antrieb",   9600, 4800),

    # Steuerung (12)
    ("S01", "SPS Basic SB-100",                 "Steuerung", 1400,  700),
    ("S02", "SPS Standard SS-200",              "Steuerung", 2200, 1100),
    ("S03", "SPS Advanced SA-300",              "Steuerung", 3800, 1900),
    ("S04", "SPS Cloud-Connected SC-400",       "Steuerung", 5400, 2700),
    ("S05", "Edge-Controller EC-250",           "Steuerung", 4200, 2100),
    ("S06", "Industrie-PC IPC-500",             "Steuerung", 6800, 3400),
    ("S07", "CNC-Steuerung CNC-300",            "Steuerung", 7200, 3600),
    ("S08", "Robotersteuerung RS-500",          "Steuerung", 8400, 4200),
    ("S09", "Sicherheits-SPS FSP-200",          "Steuerung", 3200, 1600),
    ("S10", "Kompaktsteuerung KS-100",          "Steuerung", 1800,  900),
    ("S11", "Motion-Controller MC-400",         "Steuerung", 5900, 2950),
    ("S12", "HMI-Panel HP-700",                 "Steuerung", 4600, 2300),

    # Gehäuse (10)
    ("G01", "Schutzgehäuse IP54 Standard",      "Gehäuse",    800,  400),
    ("G02", "Schutzgehäuse IP65 Standard",      "Gehäuse",   1100,  550),
    ("G03", "Schutzgehäuse IP67 Premium",       "Gehäuse",   1600,  800),
    ("G04", "Edelstahlgehäuse V2A-IP65",        "Gehäuse",   2400, 1200),
    ("G05", "Edelstahlgehäuse V4A-IP67",        "Gehäuse",   3200, 1600),
    ("G06", "Explosionsschutz Ex-II-2G",        "Gehäuse",   4800, 2400),
    ("G07", "Freiluft-Gehäuse IP65 FL-400",     "Gehäuse",   2100, 1050),
    ("G08", "Minimalgehäuse IP54 MG-50",        "Gehäuse",    600,  300),
    ("G09", "Großgehäuse GR-IP54",              "Gehäuse",   1800,  900),
    ("G10", "Rackgehäuse 19\" RK-400",          "Gehäuse",   2600, 1300),

    # Zubehör (14)
    ("Z01", "Touch-Display 7\" TD-700",         "Zubehör",    650,  325),
    ("Z02", "Touch-Display 15\" TD-1500",       "Zubehör",   1200,  600),
    ("Z03", "Temperatursensor TS-200",           "Zubehör",    280,  140),
    ("Z04", "Drucksensor DS-100",               "Zubehör",    320,  160),
    ("Z05", "Vibrationssensor VS-300",           "Zubehör",    480,  240),
    ("Z06", "Ethernet-Interface 1G ETH-100",    "Zubehör",    380,  190),
    ("Z07", "OPC-UA Gateway OG-200",            "Zubehör",    890,  445),
    ("Z08", "Kabelsatz Kompakt KK-100",         "Zubehör",    240,  120),
    ("Z09", "Kabelsatz Premium KP-300",         "Zubehör",    580,  290),
    ("Z10", "Sicherheits-Relais SR-200",        "Zubehör",    420,  210),
    ("Z11", "Not-Aus-Set NA-300",               "Zubehör",    360,  180),
    ("Z12", "Remote-Access-Modul RA-100",       "Zubehör",    720,  360),
    ("Z13", "Energiemessmodul EM-200",          "Zubehör",    390,  195),
    ("Z14", "UPS-Modul UP-500",                 "Zubehör",    840,  420),
]

# Kompatibilitätsregeln: Basis → erlaubte Antriebe, Antrieb → erlaubte Steuerungen, Steuerung → erlaubte Gehäuse
# (Zubehör ist immer optional und wird separat behandelt)

COMPAT = {
    # Basis → Antrieb (welche Antriebe passen auf diesen Rahmen)
    "B01": ["A01", "A05", "A12", "A13"],
    "B02": ["A01", "A02", "A05", "A06", "A13"],
    "B03": ["A02", "A03", "A06", "A07", "A08", "A09"],
    "B04": ["A02", "A03", "A10", "A11", "A14"],
    "B05": ["A08", "A09", "A10"],
    "B06": ["A01", "A05", "A12"],
    "B07": ["A03", "A04", "A07", "A09", "A11", "A14"],
    "B08": ["A01", "A02", "A05", "A06"],
    "B09": ["A03", "A04", "A07", "A08", "A09"],
    "B10": ["A04", "A07", "A09", "A11", "A14"],

    # Antrieb → Steuerung
    "A01": ["S01", "S02", "S10"],
    "A02": ["S02", "S03", "S09"],
    "A03": ["S03", "S04", "S05"],
    "A04": ["S04", "S05", "S11"],
    "A05": ["S01", "S02", "S10"],
    "A06": ["S02", "S03"],
    "A07": ["S03", "S04", "S05"],
    "A08": ["S03", "S05", "S06"],
    "A09": ["S04", "S05", "S06"],
    "A10": ["S05", "S06", "S07", "S11"],
    "A11": ["S06", "S07", "S08", "S11"],
    "A12": ["S01", "S10"],
    "A13": ["S01", "S02", "S12"],
    "A14": ["S07", "S08", "S11"],

    # Steuerung → Gehäuse
    "S01": ["G01", "G08"],
    "S02": ["G01", "G02", "G08"],
    "S03": ["G02", "G03", "G07"],
    "S04": ["G02", "G03", "G07", "G10"],
    "S05": ["G03", "G04", "G10"],
    "S06": ["G03", "G04", "G07", "G09", "G10"],
    "S07": ["G04", "G05", "G09"],
    "S08": ["G04", "G05", "G06"],
    "S09": ["G02", "G03", "G06"],
    "S10": ["G01", "G08"],
    "S11": ["G03", "G04", "G09", "G10"],
    "S12": ["G01", "G02", "G09"],
}

ZUBEHOER_IDS = ["Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07",
                 "Z08", "Z09", "Z10", "Z11", "Z12", "Z13", "Z14"]


def build_kompatibel_mit(kid):
    """Gibt kompatible IDs für eine Komponente zurück."""
    return COMPAT.get(kid, [])


# ── Stückliste schreiben ────────────────────────────────────────────────────

rows_sl = []
for kid, kname, kat, lp, hk in STUECKLISTE:
    rows_sl.append({
        "komponente_id":   kid,
        "komponente_name": kname,
        "kategorie":       kat,
        "listenpreis":     lp,
        "herstellkosten":  hk,
        "kompatibel_mit":  ",".join(build_kompatibel_mit(kid)),
    })

with open(OUT / "stückliste.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows_sl[0].keys())
    w.writeheader()
    w.writerows(rows_sl)

print(f"✓ stückliste.csv  ({len(rows_sl)} Zeilen)")


# ── Angebote historisch ─────────────────────────────────────────────────────

# Valide Konfigs = (Basis, Antrieb, Steuerung, Gehäuse)
def valid_configs():
    configs = []
    for b, antriebe in COMPAT.items():
        if not b.startswith("B"):
            continue
        for a in antriebe:
            for s in COMPAT.get(a, []):
                for g in COMPAT.get(s, []):
                    configs.append((b, a, s, g))
    return configs

ALLE_CONFIGS = valid_configs()

BRANCHEN = ["Automotive", "Maschinenbau", "Chemie", "Lebensmittel", "Pharma", "Energie"]

# Marge-Bonus je Branche (realistische Unterschiede)
BRANCHE_MARGE = {
    "Automotive":  -3,
    "Maschinenbau": 0,
    "Chemie":       2,
    "Lebensmittel":-1,
    "Pharma":       5,
    "Energie":      3,
}

# Gewinn-Wahrscheinlichkeit: Teure Configs werden weniger oft gewonnen
def calc_win_prob(preis, marge_pct, branche):
    base = 0.55
    if preis > 30000:
        base -= 0.10
    elif preis < 15000:
        base += 0.08
    if marge_pct > 32:
        base -= 0.05  # hohe Marge = teurer = seltener gewonnen
    if branche == "Pharma":
        base += 0.05
    if branche == "Automotive":
        base -= 0.08
    return max(0.15, min(0.85, base + random.gauss(0, 0.06)))


stueck_dict = {k: {"listenpreis": lp, "herstellkosten": hk}
               for k, _, _, lp, hk in STUECKLISTE}

rows_ang = []
selected_configs = random.choices(ALLE_CONFIGS, k=80)

for i, (b, a, s, g) in enumerate(selected_configs, 1):
    # Zufällige Zubehör-Ergänzung (1-3 Teile)
    n_zusatz = random.randint(1, 3)
    zusatz = random.sample(ZUBEHOER_IDS, n_zusatz)
    config_ids = [b, a, s, g] + zusatz

    # Preise & Kosten
    gesamtpreis = sum(stueck_dict[k]["listenpreis"] for k in config_ids)
    gesamtkosten = sum(stueck_dict[k]["herstellkosten"] for k in config_ids)

    # Aufschlag realistisch variieren
    aufschlag = random.uniform(0.92, 1.12)
    gesamtpreis = round(gesamtpreis * aufschlag, -2)

    branche = random.choice(BRANCHEN)
    marge_pct = round((gesamtpreis - gesamtkosten) / gesamtpreis * 100
                      + BRANCHE_MARGE[branche]
                      + random.gauss(0, 1.5), 1)
    marge_pct = max(5.0, min(45.0, marge_pct))

    win_prob = calc_win_prob(gesamtpreis, marge_pct, branche)
    gewonnen = 1 if random.random() < win_prob else 0

    menge = random.choice([1, 1, 1, 2, 2, 3, 5])

    rows_ang.append({
        "angebot_id":         f"ANG-{i:04d}",
        "gewonnen":           gewonnen,
        "konfiguration":      ",".join(config_ids),
        "gesamtpreis":        gesamtpreis,
        "gesamtkosten":       round(gesamtkosten, -2),
        "deckungsbeitrag_pct": marge_pct,
        "branche":            branche,
        "menge":              menge,
    })

with open(OUT / "angebote_historisch.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows_ang[0].keys())
    w.writeheader()
    w.writerows(rows_ang)

print(f"✓ angebote_historisch.csv  ({len(rows_ang)} Zeilen)")
