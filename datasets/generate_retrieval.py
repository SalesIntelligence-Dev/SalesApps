"""Synthetische Firmenstammdaten mit bewussten Duplikaten für Entity-Resolution-Demo."""
import csv, random
from datetime import date, timedelta
from pathlib import Path

random.seed(99)

OUT = Path(__file__).parent

BRANCHEN = [
    "Maschinenbau", "Automotive", "Chemie", "Handel", "IT-Dienstleistung",
    "Logistik", "Pharma", "Lebensmittel", "Metallindustrie", "Bau",
]

UMSATZ_KLASSEN = ["< 1 Mio", "1–10 Mio", "10–50 Mio", "50–250 Mio", "> 250 Mio"]
MA_KLASSEN     = ["1–9", "10–49", "50–249", "250–999", "> 1000"]

STAEDTE = [
    ("40213", "Düsseldorf"), ("70173", "Stuttgart"), ("80331", "München"),
    ("20095", "Hamburg"),    ("60311", "Frankfurt"),  ("50667", "Köln"),
    ("10115", "Berlin"),     ("30159", "Hannover"),   ("01067", "Dresden"),
    ("28195", "Bremen"),     ("90402", "Nürnberg"),   ("04103", "Leipzig"),
    ("48143", "Münster"),    ("44135", "Dortmund"),   ("45127", "Essen"),
]

RECHTSFORMEN = ["GmbH", "AG", "GmbH & Co. KG", "KG", "OHG", "SE", "GbR"]

BASE_FIRMEN = [
    # (id_prefix, name_base, branche, umsatz, mitarbeiter, plz, ort)
    (  1, "Müller",         "Maschinenbau",      "10–50 Mio", "50–249",  "40213", "Düsseldorf"),
    (  2, "Weber",          "Automotive",         "50–250 Mio","250–999", "70173", "Stuttgart"),
    (  3, "Schmidt",        "Chemie",             "1–10 Mio",  "10–49",   "80331", "München"),
    (  4, "Braun",          "IT-Dienstleistung",  "1–10 Mio",  "10–49",   "20095", "Hamburg"),
    (  5, "Fischer",        "Logistik",           "10–50 Mio", "50–249",  "60311", "Frankfurt"),
    (  6, "Wagner",         "Handel",             "50–250 Mio","250–999", "50667", "Köln"),
    (  7, "Becker",         "Metallindustrie",    "1–10 Mio",  "10–49",   "10115", "Berlin"),
    (  8, "Schulz",         "Pharma",             "10–50 Mio", "50–249",  "30159", "Hannover"),
    (  9, "Hoffmann",       "Lebensmittel",       "10–50 Mio", "50–249",  "01067", "Dresden"),
    ( 10, "Koch",           "Bau",                "1–10 Mio",  "10–49",   "28195", "Bremen"),
    ( 11, "Bauer",          "Maschinenbau",       "50–250 Mio","250–999", "90402", "Nürnberg"),
    ( 12, "Richter",        "Automotive",         "10–50 Mio", "50–249",  "04103", "Leipzig"),
    ( 13, "Klein",          "IT-Dienstleistung",  "< 1 Mio",   "1–9",     "48143", "Münster"),
    ( 14, "Wolf",           "Chemie",             "50–250 Mio","250–999", "44135", "Dortmund"),
    ( 15, "Schröder",       "Handel",             "1–10 Mio",  "10–49",   "45127", "Essen"),
    ( 16, "Neumann",        "Logistik",           "10–50 Mio", "50–249",  "40213", "Düsseldorf"),
    ( 17, "Schwarz",        "Pharma",             "> 250 Mio", "> 1000",  "70173", "Stuttgart"),
    ( 18, "Zimmermann",     "Metallindustrie",    "1–10 Mio",  "10–49",   "80331", "München"),
    ( 19, "Krüger",         "Lebensmittel",       "10–50 Mio", "50–249",  "20095", "Hamburg"),
    ( 20, "Hartmann",       "Maschinenbau",       "50–250 Mio","250–999", "60311", "Frankfurt"),
    ( 21, "Lange",          "Bau",                "1–10 Mio",  "10–49",   "50667", "Köln"),
    ( 22, "Köhler",         "IT-Dienstleistung",  "10–50 Mio", "50–249",  "10115", "Berlin"),
    ( 23, "Maier",          "Automotive",         "50–250 Mio","250–999", "90402", "Nürnberg"),
    ( 24, "Meyer",          "Chemie",             "10–50 Mio", "50–249",  "04103", "Leipzig"),
    ( 25, "Lehmann",        "Handel",             "1–10 Mio",  "10–49",   "30159", "Hannover"),
    ( 26, "Herrmann",       "Logistik",           "50–250 Mio","250–999", "01067", "Dresden"),
    ( 27, "König",          "Pharma",             "10–50 Mio", "50–249",  "28195", "Bremen"),
    ( 28, "Peters",         "Metallindustrie",    "< 1 Mio",   "1–9",     "48143", "Münster"),
    ( 29, "Huber",          "Lebensmittel",       "10–50 Mio", "50–249",  "44135", "Dortmund"),
    ( 30, "Günther",        "Maschinenbau",       "1–10 Mio",  "10–49",   "45127", "Essen"),
    ( 31, "Brandt",         "Bau",                "10–50 Mio", "50–249",  "40213", "Düsseldorf"),
    ( 32, "Haas",           "IT-Dienstleistung",  "1–10 Mio",  "10–49",   "70173", "Stuttgart"),
    ( 33, "Schäfer",        "Automotive",         "> 250 Mio", "> 1000",  "80331", "München"),
    ( 34, "Vogt",           "Chemie",             "1–10 Mio",  "10–49",   "20095", "Hamburg"),
    ( 35, "Möller",         "Handel",             "10–50 Mio", "50–249",  "60311", "Frankfurt"),
    ( 36, "Krause",         "Logistik",           "50–250 Mio","250–999", "50667", "Köln"),
    ( 37, "Jung",           "Pharma",             "10–50 Mio", "50–249",  "10115", "Berlin"),
    ( 38, "Hahn",           "Metallindustrie",    "1–10 Mio",  "10–49",   "30159", "Hannover"),
    ( 39, "Franke",         "Lebensmittel",       "10–50 Mio", "50–249",  "04103", "Leipzig"),
    ( 40, "Albrecht",       "Maschinenbau",       "1–10 Mio",  "10–49",   "90402", "Nürnberg"),
    ( 41, "Ziegler",        "Bau",                "50–250 Mio","250–999", "28195", "Bremen"),
    ( 42, "Roth",           "IT-Dienstleistung",  "< 1 Mio",   "1–9",     "01067", "Dresden"),
    ( 43, "Winkler",        "Automotive",         "10–50 Mio", "50–249",  "48143", "Münster"),
    ( 44, "Simon",          "Chemie",             "10–50 Mio", "50–249",  "44135", "Dortmund"),
    ( 45, "Pfeiffer",       "Handel",             "1–10 Mio",  "10–49",   "45127", "Essen"),
]


def make_website(name):
    slug = name.lower().replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("ß", "ss")
    slug = slug.replace(" ", "-").replace(".", "").replace("&", "")
    return f"www.{slug}.de"


def make_strasse():
    straßen = ["Hauptstraße", "Industriestraße", "Bahnhofstraße", "Parkstraße",
               "Münchener Straße", "Berliner Allee", "Rheinstraße", "Gartenweg",
               "Am Werk", "Gewerbepark", "Technologiepark", "Hansastraße"]
    return f"{random.choice(straßen)} {random.randint(1, 150)}"


def make_datum():
    start = date(2015, 1, 1)
    return (start + timedelta(days=random.randint(0, 3000))).isoformat()


rows = []
fid = 1

# ── Echte Firmen (45 Basis-Einträge) ────────────────────────────────────────
base_records = {}
for bnum, name, branche, umsatz, ma, plz, ort in BASE_FIRMEN:
    rf = random.choice(["GmbH", "GmbH", "AG", "GmbH & Co. KG"])
    firma_name = f"{name} {rf}"
    rec = {
        "firma_id":      f"F{fid:04d}",
        "firma_name":    firma_name,
        "strasse":       make_strasse(),
        "plz":           plz,
        "ort":           ort,
        "land":          "DE",
        "branche":       branche,
        "umsatz_klasse": umsatz,
        "mitarbeiter":   ma,
        "website":       make_website(name),
        "erstellt_am":   make_datum(),
    }
    rows.append(rec)
    base_records[bnum] = rec
    fid += 1

# ── Duplikat-Paare (17 echte Duplikate, verschiedene Varianten) ─────────────

TIPPFEHLER = {
    "Müller":      ["Müler",    "Muller",     "Müllerr"],
    "Weber":       ["Webber",   "Wber",       "Weber "],
    "Schmidt":     ["Schmitt",  "Schmmidt",   "Shmidt"],
    "Braun":       ["Brown",    "Braun ",     "Brann"],
    "Fischer":     ["Ficher",   "Fiscer",     "Fischer "],
    "Wagner":      ["Vagner",   "Wagnerr",    "Wagener"],
    "Becker":      ["Bekker",   "Beker",      "Beckerr"],
    "Schulz":      ["Schultz",  "Shulz",      "Schultz"],
    "Hoffmann":    ["Hofmann",  "Hoffman",    "Hofman"],
    "Koch":        ["Kock",     "Kooch",      "Kok"],
    "Bauer":       ["Bour",     "Baur",       "Bauerm"],
    "Richter":     ["Richter",  "Richterr",   "Rihter"],
    "Klein":       ["Clein",    "Kleim",      "Kein"],
    "Wolf":        ["Wölf",     "Wolff",      "Wof"],
    "Schröder":    ["Schroeder","Shröder",    "Schröderr"],
    "Neumann":     ["Neuman",   "Neumannn",   "Neymann"],
    "Schwarz":     ["Schwarz",  "Schwars",    "Swarz"],
}

# 17 Duplikat-Szenarien
DUPLIKAT_SZENARIEN = [
    # (base_num, variante_typ, rf_override)
    ( 1, "name_typo",     None),           # Müller GmbH → Müler GmbH
    ( 1, "rechtsform",    "GmbH & Co. KG"),# Müller GmbH → Müller GmbH & Co. KG
    ( 2, "name_typo",     None),           # Weber AG → Webber AG
    ( 3, "umlaut",        None),           # Schmidt → Schmitt
    ( 4, "name_typo",     None),           # Braun
    ( 5, "alte_adresse",  None),           # Fischer – alte Adresse
    ( 6, "name_typo",     None),           # Wagner
    ( 7, "name_typo",     None),           # Becker
    ( 8, "umlaut",        None),           # Schulz → Schultz
    (10, "name_typo",     None),           # Koch
    (11, "rechtsform",    "AG"),           # Bauer GmbH → Bauer AG
    (12, "name_typo",     None),           # Richter
    (14, "name_typo",     None),           # Wolf
    (15, "umlaut",        None),           # Schröder → Schroeder
    (17, "name_typo",     None),           # Schwarz
    (20, "alte_adresse",  None),           # Hartmann – andere PLZ
    (23, "rechtsform",    "GmbH"),         # Maier AG → Maier GmbH
]

for base_num, variante, rf_override in DUPLIKAT_SZENARIEN:
    base = base_records[base_num]
    base_name_part = BASE_FIRMEN[base_num - 1][1]  # z.B. "Müller"

    if variante == "name_typo":
        typos = TIPPFEHLER.get(base_name_part, [base_name_part + "x"])
        typo = random.choice(typos)
        rf = base["firma_name"].replace(base_name_part, "").strip()
        firma_name = f"{typo} {rf}".strip()
        plz = base["plz"]
        ort = base["ort"]
        strasse = base["strasse"]
    elif variante == "umlaut":
        typos = TIPPFEHLER.get(base_name_part, [])
        uml = typos[0] if typos else base_name_part
        rf = base["firma_name"].replace(base_name_part, "").strip()
        firma_name = f"{uml} {rf}".strip()
        plz = base["plz"]
        ort = base["ort"]
        strasse = base["strasse"]
    elif variante == "rechtsform":
        firma_name = f"{base_name_part} {rf_override}"
        plz = base["plz"]
        ort = base["ort"]
        strasse = base["strasse"]
    elif variante == "alte_adresse":
        firma_name = base["firma_name"]
        # Andere Straße, aber gleiche PLZ
        plz = base["plz"]
        ort = base["ort"]
        strasse = make_strasse()
    else:
        firma_name = base["firma_name"]
        plz = base["plz"]
        ort = base["ort"]
        strasse = base["strasse"]

    rows.append({
        "firma_id":      f"F{fid:04d}",
        "firma_name":    firma_name,
        "strasse":       strasse,
        "plz":           plz,
        "ort":           ort,
        "land":          "DE",
        "branche":       base["branche"],
        "umsatz_klasse": base["umsatz_klasse"],
        "mitarbeiter":   base["mitarbeiter"],
        "website":       make_website(base_name_part),
        "erstellt_am":   make_datum(),
    })
    fid += 1

# ── Weitere eindeutige Firmen (Auffüllen auf 120) ──────────────────────────

WEITERE = [
    ("Technovation", "IT-Dienstleistung", "48143", "Münster"),
    ("ProMetall",    "Metallindustrie",   "44135", "Dortmund"),
    ("GreenChem",    "Chemie",            "45127", "Essen"),
    ("FastLog",      "Logistik",          "40213", "Düsseldorf"),
    ("FoodPro",      "Lebensmittel",      "70173", "Stuttgart"),
    ("BauKraft",     "Bau",               "80331", "München"),
    ("AutoParts",    "Automotive",        "20095", "Hamburg"),
    ("MedTech",      "Pharma",            "60311", "Frankfurt"),
    ("Handelshaus Bergmann", "Handel",    "50667", "Köln"),
    ("Industriewerke Ost",   "Maschinenbau","10115","Berlin"),
    ("Logistikcenter Nord",  "Logistik",  "30159", "Hannover"),
    ("Chemiewerk Elbe",      "Chemie",    "01067", "Dresden"),
    ("Stahlbau Weser",       "Metallindustrie","28195","Bremen"),
    ("Software Solutions",   "IT-Dienstleistung","90402","Nürnberg"),
    ("Autoteile Express",    "Automotive","04103", "Leipzig"),
    ("MaschBau Rhein",       "Maschinenbau","48143","Münster"),
    ("Pharmawerk Süd",       "Pharma",    "44135", "Dortmund"),
    ("Einzelhandel König",   "Handel",    "45127", "Essen"),
    ("TechGroup International", "IT-Dienstleistung","40213","Düsseldorf"),
    ("Bauunternehmen Müller-Bau", "Bau",  "70173", "Stuttgart"),
    ("Energiesysteme GmbH",  "Energie",   "80331", "München"),
    ("Versorgungslogistik",  "Logistik",  "20095", "Hamburg"),
    ("Chemcompound AG",      "Chemie",    "60311", "Frankfurt"),
    ("GlobalTrade GmbH",     "Handel",    "50667", "Köln"),
    ("MetalCraft OHG",       "Metallindustrie","10115","Berlin"),
    ("DigiSolutions KG",     "IT-Dienstleistung","30159","Hannover"),
    ("Nahrungswerk Nord",    "Lebensmittel","01067","Dresden"),
    ("PharmaPlus GmbH",      "Pharma",    "28195", "Bremen"),
    ("Spedition Franken",    "Logistik",  "90402", "Nürnberg"),
    ("Autowerk Leipzig",     "Automotive","04103", "Leipzig"),
    ("Maschinenfabrik Thüringen","Maschinenbau","48143","Münster"),
    ("HVAC Solutions",       "Maschinenbau","44135","Dortmund"),
    ("RealEstate Invest",    "Bau",       "45127", "Essen"),
    ("DataVault GmbH",       "IT-Dienstleistung","40213","Düsseldorf"),
    ("Retail Chain SE",      "Handel",    "70173", "Stuttgart"),
    ("ChemPure AG",          "Chemie",    "80331", "München"),
    ("FlexLog GmbH",         "Logistik",  "20095", "Hamburg"),
    ("Foodservice Rhein",    "Lebensmittel","60311","Frankfurt"),
    ("Bauwerk & Söhne",      "Bau",       "50667", "Köln"),
    ("Stahlverarbeitungs AG","Metallindustrie","10115","Berlin"),
    ("BioMedical GmbH",      "Pharma",    "30159", "Hannover"),
    ("AutoSupplement OHG",   "Automotive","01067", "Dresden"),
    ("Werkzeugbau Osten",    "Maschinenbau","28195","Bremen"),
    ("NetServices GmbH",     "IT-Dienstleistung","90402","Nürnberg"),
    ("FreshFood GmbH",       "Lebensmittel","04103","Leipzig"),
    ("Logistik Center Süd",  "Logistik",  "48143", "Münster"),
    ("ElektroTech GmbH",     "Metallindustrie","44135","Dortmund"),
    ("WholesaleHub AG",      "Handel",    "45127", "Essen"),
    ("IntelliSystems KG",    "IT-Dienstleistung","40213","Düsseldorf"),
    ("PharmaRhein GmbH",     "Pharma",    "70173", "Stuttgart"),
    ("AutoCare AG",          "Automotive","80331", "München"),
    ("Baustoffe Süd GmbH",   "Bau",       "20095", "Hamburg"),
    ("ClearChem GmbH",       "Chemie",    "60311", "Frankfurt"),
    ("MachPrecision OHG",    "Maschinenbau","50667","Köln"),
    ("FastDeli GmbH",        "Logistik",  "10115", "Berlin"),
    ("BioFoods AG",          "Lebensmittel","30159","Hannover"),
    ("DrugStore Network",    "Pharma",    "01067", "Dresden"),
    ("AutoMate Systems",     "Automotive","28195", "Bremen"),
]

umsatz_kl = random.choices(UMSATZ_KLASSEN, k=len(WEITERE))
ma_kl     = random.choices(MA_KLASSEN, k=len(WEITERE))

for idx, (name, branche, plz, ort) in enumerate(WEITERE):
    rf = random.choice(["GmbH", "AG", "GmbH & Co. KG"])
    rows.append({
        "firma_id":      f"F{fid:04d}",
        "firma_name":    f"{name} {rf}",
        "strasse":       make_strasse(),
        "plz":           plz,
        "ort":           ort,
        "land":          "DE",
        "branche":       branche,
        "umsatz_klasse": umsatz_kl[idx],
        "mitarbeiter":   ma_kl[idx],
        "website":       make_website(name),
        "erstellt_am":   make_datum(),
    })
    fid += 1

# Auf 120 trimmen und shuffeln
random.shuffle(rows)
rows = rows[:120]

with open(OUT / "firmenstamm_beispiel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"✓ firmenstamm_beispiel.csv  ({len(rows)} Zeilen)")
