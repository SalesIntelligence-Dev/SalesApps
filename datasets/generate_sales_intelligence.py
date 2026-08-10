"""
Gemeinsamer Datengenerator für die Sales-Intelligence-Ebene (Apps 7–9).

Alle drei Apps arbeiten auf demselben Entitäten-Roster (Kunden, Produkte,
Vertriebsmitarbeiter) – das ist die Grundidee des "Sales Graph":
ein Unternehmensgraph, auf dem mehrere Modelle laufen.

Erzeugt:
  growth_beispiel.csv        – Kunde × Produkt × Jahr (Customer Growth Engine)
  opportunities.csv          – offene + historische Deals (Opportunity Intelligence)
  opportunity_kontakte.csv   – Buying-Center je Deal
  next_action_beispiel.csv   – Signal-Tabelle je Kunde (Next Best Action)
"""
import csv, random
from pathlib import Path

random.seed(2026)
OUT = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════════
# GEMEINSAMER ENTITÄTEN-ROSTER
# ══════════════════════════════════════════════════════════════════════════

REPS = ["Meyer", "Schulz", "Hoffmann"]

# (id, name, branche, rep, groesse_klasse, wachstumstyp)
KUNDEN = [
    ("K01", "Müller GmbH",        "Maschinenbau",    "Meyer",    "Groß",   "wachstum"),
    ("K02", "Schneider AG",       "Automotive",      "Schulz",   "Groß",   "wachstum"),
    ("K03", "Meier GmbH",         "Chemie",          "Meyer",    "Mittel", "rueckgang"),
    ("K04", "Schmidt AG",         "Lebensmittel",    "Hoffmann", "Mittel", "stabil"),
    ("K05", "Weber Industrie",    "Maschinenbau",    "Schulz",   "Groß",   "stabil"),
    ("K06", "Braun Systems",      "Pharma",          "Meyer",    "Mittel", "wachstum"),
    ("K07", "Fischer Tech",       "Automotive",      "Hoffmann", "Klein",  "wachstum"),
    ("K08", "Klein Automation",   "Maschinenbau",    "Schulz",   "Klein",  "stabil"),
    ("K09", "Groß Maschinenbau",  "Metallindustrie", "Meyer",    "Groß",   "rueckgang"),
    ("K10", "Ritter Technik",     "Chemie",          "Hoffmann", "Mittel", "stabil"),
    ("K11", "Hoffmann Werke",     "Lebensmittel",    "Schulz",   "Groß",   "wachstum"),
    ("K12", "Bauer Industrie",    "Pharma",          "Meyer",    "Mittel", "rueckgang"),
    ("K13", "Wagner Systems",     "Metallindustrie", "Hoffmann", "Klein",  "stabil"),
    ("K14", "Koch Anlagen",       "Maschinenbau",    "Schulz",   "Mittel", "wachstum"),
]

# (id, name, gruppe, listenpreis)
PRODUKTE = [
    ("P01", "Hydraulikpumpe HX-200",  "Maschinen", 2800),
    ("P02", "Kompressor KA-500",       "Maschinen", 2200),
    ("P03", "Druckluftanlage DA-100",  "Maschinen", 4100),
    ("P04", "Förderpumpe FP-350",      "Maschinen", 2500),
    ("P05", "Steuerungsmodul SM-Pro",  "Zubehör",   1150),
    ("P06", "Präzisionssensor PS-200", "Zubehör",    580),
    ("P07", "Druckregler DR-80",       "Zubehör",    400),
    ("P08", "Schlauchpaket SP-Set",    "Zubehör",    230),
    ("P09", "Wartungsvertrag Basic",   "Service",    700),
    ("P10", "Wartungsvertrag Premium", "Service",   1300),
    ("P11", "Inbetriebnahme & Setup",  "Service",    825),
    ("P12", "Anwenderschulung",        "Service",    625),
]

PROD = {p[0]: p for p in PRODUKTE}
KUNDE = {k[0]: k for k in KUNDEN}

# Portfolio je Kunde: welche Produkte werden gekauft (bewusste Lücken!)
PORTFOLIO = {
    "K01": ["P01", "P02", "P05", "P06", "P09", "P11"],              # Lücke: P10 Premium, P12
    "K02": ["P02", "P03", "P05", "P06", "P07", "P10", "P11", "P12"],# fast voll
    "K03": ["P01", "P04", "P07", "P09"],                             # große Lücken
    "K04": ["P02", "P05", "P06", "P08", "P09", "P12"],
    "K05": ["P01", "P02", "P03", "P05", "P06", "P07", "P10", "P11"],
    "K06": ["P03", "P05", "P06", "P10"],                             # Lücke: P11, P12
    "K07": ["P02", "P06", "P07", "P12"],                             # klein, wachsend
    "K08": ["P01", "P05", "P08", "P09"],
    "K09": ["P01", "P02", "P04", "P07", "P08", "P09", "P11"],
    "K10": ["P03", "P04", "P05", "P06", "P09", "P10"],
    "K11": ["P02", "P03", "P05", "P06", "P07", "P10", "P11", "P12"],
    "K12": ["P01", "P03", "P06", "P09"],
    "K13": ["P04", "P07", "P08", "P09"],
    "K14": ["P01", "P02", "P05", "P06", "P09", "P10", "P11"],
}

GROESSE_FAKTOR = {"Klein": 0.5, "Mittel": 1.0, "Groß": 2.2}
WACHSTUM_FAKTOR = {
    #          2023   2024   2025
    "wachstum":  [0.72, 0.88, 1.00],
    "stabil":    [0.97, 1.00, 0.99],
    "rueckgang": [1.55, 1.37, 1.00],   # 2024→2025 ≈ −27 %
}
JAHRE = [2023, 2024, 2025]


# ══════════════════════════════════════════════════════════════════════════
# 1) growth_beispiel.csv  – Customer Growth Engine
# ══════════════════════════════════════════════════════════════════════════

rows_growth = []
for kid, kname, branche, rep, groesse, wtyp in KUNDEN:
    gf = GROESSE_FAKTOR[groesse]
    # Basismenge einmal je Kunde/Produkt festlegen – sonst überdeckt der Zufall
    # den Wachstumstrend und die Jahresvergleiche werden unbrauchbar.
    basis_mengen = {pid: random.randint(4, 12) for pid in PORTFOLIO[kid]}
    for jahr_idx, jahr in enumerate(JAHRE):
        wf = WACHSTUM_FAKTOR[wtyp][jahr_idx]
        for pid in PORTFOLIO[kid]:
            _, pname, gruppe, preis = PROD[pid]
            # Menge skaliert mit Größe und Wachstumsphase (+ leichtes Rauschen)
            menge = max(1, round(basis_mengen[pid] * gf * wf * random.uniform(0.94, 1.06)))
            # Preisvariation (Rabatte)
            ep = round(preis * random.uniform(0.86, 1.02))
            rows_growth.append({
                "kunde_id":      kid,
                "kunde_name":    kname,
                "branche":       branche,
                "vertriebsmitarbeiter": rep,
                "groesse_klasse": groesse,
                "jahr":          jahr,
                "produkt_id":    pid,
                "produkt_name":  pname,
                "produktgruppe": gruppe,
                "menge":         menge,
                "einzelpreis":   ep,
                "umsatz":        ep * menge,
            })

with open(OUT / "growth_beispiel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows_growth[0].keys())
    w.writeheader()
    w.writerows(rows_growth)
print(f"✓ growth_beispiel.csv        ({len(rows_growth)} Zeilen)")


# ══════════════════════════════════════════════════════════════════════════
# 2) opportunities.csv + opportunity_kontakte.csv – Opportunity Intelligence
# ══════════════════════════════════════════════════════════════════════════

STAGES = [
    ("Qualifizierung",  1, 20),
    ("Bedarfsanalyse",  2, 40),
    ("Angebot",         3, 60),
    ("Verhandlung",     4, 80),
    ("Abschluss",       5, 92),
]
WETTBEWERBER = ["Nordtech AG", "Vertec Systems", "Industrial One", "Hydrotek GmbH", "—"]

ROLLEN = ["Geschäftsführung", "CFO / Einkaufsleitung", "Einkauf", "Technik", "Produktion"]
VORNAMEN = ["Andreas", "Birgit", "Christian", "Daniela", "Erik", "Franziska",
            "Gerd", "Heike", "Ingo", "Julia", "Klaus", "Lena", "Martin",
            "Nadine", "Oliver", "Petra", "Ralf", "Sabine", "Thomas", "Ute"]
NACHNAMEN = ["Berger", "Krämer", "Lindner", "Ostermann", "Pfeiffer", "Reinhardt",
             "Sander", "Thiele", "Ullmann", "Vogel", "Wendt", "Zeller"]


def contact_name():
    return f"{random.choice(VORNAMEN)} {random.choice(NACHNAMEN)}"


rows_opp, rows_kontakt = [], []
opp_counter = 1

# ── Handgebaute Kern-Deals (die Story-Deals aus dem Konzept) ──────────────
# (kunde_id, produkt, wert, stage_nr, tage_in_stage, letzter_kontakt, wettbewerb,
#  rabatt, cfo_involviert, wettbewerber_kontakt, status)
STORY_DEALS = [
    # Müller GmbH – groß, sieht gut aus, aber CFO fehlt + Wettbewerberkontakt
    ("K01", "P03", 420000, 3,  38, 17, "Nordtech AG",     12, False, True,  "offen"),
    # Schneider AG – sauber aufgestellt, hohe Chance
    ("K02", "P10", 210000, 4,  11,  4, "—",                6, True,  False, "offen"),
    # Meier GmbH – Rückgangskunde, Deal steht still
    ("K03", "P01", 145000, 2,  62, 41, "Vertec Systems",  18, False, True,  "offen"),
    # Weber Industrie – Verhandlung, hoher Rabattdruck
    ("K05", "P03", 380000, 4,  24,  9, "Industrial One",  22, True,  True,  "offen"),
    # Braun Systems – frisch, Technik überzeugt
    ("K06", "P11", 95000,  3,  14,  6, "—",                8, False, False, "offen"),
    # Groß Maschinenbau – langer Stillstand, Risiko
    ("K09", "P02", 265000, 3,  71, 55, "Hydrotek GmbH",   15, False, True,  "offen"),
    # Hoffmann Werke – kurz vor Abschluss
    ("K11", "P03", 310000, 5,   8,  3, "—",                9, True,  False, "offen"),
    # Koch Anlagen – Expansion
    ("K14", "P10", 175000, 3,  19, 12, "Nordtech AG",     11, True,  False, "offen"),
]

for kid, pid, wert, stage_nr, tage_stage, kontakt_tage, wb, rabatt, cfo, wb_kontakt, status in STORY_DEALS:
    _, kname, branche, rep, _, _ = KUNDE[kid]
    stage_name, _, crm_prob = STAGES[stage_nr - 1]
    opp_id = f"OPP-{opp_counter:04d}"
    rows_opp.append({
        "opp_id":               opp_id,
        "kunde_id":             kid,
        "kunde_name":           kname,
        "branche":              branche,
        "vertriebsmitarbeiter": rep,
        "produkt_id":           pid,
        "produkt_name":         PROD[pid][1],
        "deal_wert":            wert,
        "stage":                stage_name,
        "stage_nr":             stage_nr,
        "tage_in_stage":        tage_stage,
        "tage_seit_erstellung": tage_stage + random.randint(20, 90),
        "letzter_kontakt_tage": kontakt_tage,
        "wettbewerber":         wb,
        "rabatt_gefordert_pct": rabatt,
        "crm_probability":      crm_prob,
        "status":               status,
    })

    # Buying Center aufbauen
    rollen_aktiv = ["Einkauf", "Technik"]
    if cfo:
        rollen_aktiv.append("CFO / Einkaufsleitung")
    if wert > 250000:
        rollen_aktiv.append("Geschäftsführung")
    if random.random() < 0.6:
        rollen_aktiv.append("Produktion")

    for rolle in ROLLEN:
        involviert = 1 if rolle in rollen_aktiv else 0
        # Wettbewerberkontakt sitzt typischerweise im Einkauf
        wbk = 1 if (wb_kontakt and rolle == "Einkauf") else 0
        if involviert:
            sentiment = random.choice(["positiv", "positiv", "neutral"]) if not wbk else "neutral"
        else:
            sentiment = "unbekannt"
        rows_kontakt.append({
            "opp_id":              opp_id,
            "kontakt_name":        contact_name() if involviert else "—",
            "rolle":               rolle,
            "involviert":          involviert,
            "sentiment":           sentiment,
            "wettbewerber_kontakt": wbk,
        })
    opp_counter += 1

# ── Weitere offene Deals (aufgefüllt) ────────────────────────────────────
for _ in range(22):
    kid = random.choice([k[0] for k in KUNDEN])
    _, kname, branche, rep, groesse, _ = KUNDE[kid]
    pid = random.choice([p[0] for p in PRODUKTE])
    stage_name, stage_nr, crm_prob = random.choice(STAGES)
    wert = round(random.uniform(40000, 350000) * GROESSE_FAKTOR[groesse] / 1.4, -3)
    wb = random.choice(WETTBEWERBER)
    tage_stage = random.randint(5, 80)
    opp_id = f"OPP-{opp_counter:04d}"

    rows_opp.append({
        "opp_id":               opp_id,
        "kunde_id":             kid,
        "kunde_name":           kname,
        "branche":              branche,
        "vertriebsmitarbeiter": rep,
        "produkt_id":           pid,
        "produkt_name":         PROD[pid][1],
        "deal_wert":            int(wert),
        "stage":                stage_name,
        "stage_nr":             stage_nr,
        "tage_in_stage":        tage_stage,
        "tage_seit_erstellung": tage_stage + random.randint(15, 120),
        "letzter_kontakt_tage": random.randint(2, 60),
        "wettbewerber":         wb,
        "rabatt_gefordert_pct": random.randint(3, 25),
        "crm_probability":      crm_prob,
        "status":               "offen",
    })

    cfo = random.random() < 0.5
    rollen_aktiv = ["Einkauf"]
    if random.random() < 0.75:
        rollen_aktiv.append("Technik")
    if cfo:
        rollen_aktiv.append("CFO / Einkaufsleitung")
    if wert > 250000 and random.random() < 0.6:
        rollen_aktiv.append("Geschäftsführung")
    if random.random() < 0.5:
        rollen_aktiv.append("Produktion")

    wb_kontakt = wb != "—" and random.random() < 0.45
    for rolle in ROLLEN:
        involviert = 1 if rolle in rollen_aktiv else 0
        wbk = 1 if (wb_kontakt and rolle == "Einkauf") else 0
        sentiment = (random.choice(["positiv", "neutral", "neutral", "negativ"])
                     if involviert else "unbekannt")
        rows_kontakt.append({
            "opp_id":              opp_id,
            "kontakt_name":        contact_name() if involviert else "—",
            "rolle":               rolle,
            "involviert":          involviert,
            "sentiment":           sentiment,
            "wettbewerber_kontakt": wbk,
        })
    opp_counter += 1

# ── Historische (abgeschlossene) Deals für die Referenz-Win-Rate ─────────
for _ in range(70):
    kid = random.choice([k[0] for k in KUNDEN])
    _, kname, branche, rep, groesse, _ = KUNDE[kid]
    pid = random.choice([p[0] for p in PRODUKTE])
    stage_name, stage_nr, crm_prob = STAGES[-1]  # abgeschlossen
    wert = round(random.uniform(40000, 320000) * GROESSE_FAKTOR[groesse] / 1.4, -3)
    wb = random.choice(WETTBEWERBER)
    kontakt_tage = random.randint(2, 50)
    rabatt = random.randint(3, 26)

    # Realistische Gewinnlogik: Wettbewerb, Rabattdruck und Kontaktabstand zählen
    p = 0.55
    if wb != "—":            p -= 0.14
    if rabatt > 18:          p -= 0.10
    if kontakt_tage > 30:    p -= 0.12
    if wert > 250000:        p -= 0.06
    gewonnen = 1 if random.random() < max(0.08, p) else 0

    opp_id = f"OPP-{opp_counter:04d}"
    rows_opp.append({
        "opp_id":               opp_id,
        "kunde_id":             kid,
        "kunde_name":           kname,
        "branche":              branche,
        "vertriebsmitarbeiter": rep,
        "produkt_id":           pid,
        "produkt_name":         PROD[pid][1],
        "deal_wert":            int(wert),
        "stage":                "Abgeschlossen",
        "stage_nr":             5,
        "tage_in_stage":        0,
        "tage_seit_erstellung": random.randint(60, 400),
        "letzter_kontakt_tage": kontakt_tage,
        "wettbewerber":         wb,
        "rabatt_gefordert_pct": rabatt,
        "crm_probability":      crm_prob,
        "status":               "gewonnen" if gewonnen else "verloren",
    })
    opp_counter += 1

with open(OUT / "opportunities.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows_opp[0].keys())
    w.writeheader()
    w.writerows(rows_opp)
print(f"✓ opportunities.csv          ({len(rows_opp)} Zeilen)")

with open(OUT / "opportunity_kontakte.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows_kontakt[0].keys())
    w.writeheader()
    w.writerows(rows_kontakt)
print(f"✓ opportunity_kontakte.csv   ({len(rows_kontakt)} Zeilen)")


# ══════════════════════════════════════════════════════════════════════════
# 3) next_action_beispiel.csv – Next Best Action
# ══════════════════════════════════════════════════════════════════════════

# Umsatz je Kunde aus growth-Daten ableiten (2025 vs. 2024)
umsatz_2025, umsatz_2024 = {}, {}
for r in rows_growth:
    if r["jahr"] == 2025:
        umsatz_2025[r["kunde_id"]] = umsatz_2025.get(r["kunde_id"], 0) + r["umsatz"]
    elif r["jahr"] == 2024:
        umsatz_2024[r["kunde_id"]] = umsatz_2024.get(r["kunde_id"], 0) + r["umsatz"]

# Offene Opportunities je Kunde
opp_agg = {}
for r in rows_opp:
    if r["status"] == "offen":
        e = opp_agg.setdefault(r["kunde_id"], {"n": 0, "wert": 0})
        e["n"] += 1
        e["wert"] += r["deal_wert"]

TRIGGER_EVENTS = {
    "K01": "Neue Produktionsstätte angekündigt",
    "K02": "Werkserweiterung genehmigt",
    "K06": "Neue Produktlinie in Aufbau",
    "K07": "Finanzierungsrunde abgeschlossen",
    "K11": "Übernahme eines Wettbewerbers",
    "K14": "Zweites Werk in Planung",
}

# Wettbewerber aktiv (aus Opportunities ableitbar, hier explizit)
WETTBEWERB_AKTIV = {"K01", "K03", "K05", "K09", "K14"}

rows_na = []
for kid, kname, branche, rep, groesse, wtyp in KUNDEN:
    u25 = umsatz_2025.get(kid, 0)
    u24 = umsatz_2024.get(kid, 0)
    opp = opp_agg.get(kid, {"n": 0, "wert": 0})

    # Portfolio-Abdeckung: wie viel der 12 Produkte werden gekauft
    abdeckung = round(len(PORTFOLIO[kid]) / len(PRODUKTE) * 100)
    # Wert der Produktlücken (Peer-Median-Näherung)
    luecken = [p for p in PROD if p not in PORTFOLIO[kid]]
    luecken_wert = sum(PROD[p][3] for p in luecken) * (6 * GROESSE_FAKTOR[groesse])

    # Service-Tickets: Rückgangskunden haben mehr
    if wtyp == "rueckgang":
        tickets, tickets_vq = random.randint(9, 18), random.randint(3, 7)
    elif wtyp == "wachstum":
        tickets, tickets_vq = random.randint(1, 5), random.randint(2, 6)
    else:
        tickets, tickets_vq = random.randint(3, 8), random.randint(3, 8)

    # Kontaktpflege
    kontakte_gesamt = random.randint(3, 7)
    if wtyp == "rueckgang":
        kontakte_aktiv = max(1, kontakte_gesamt - random.randint(2, 3))
        letzter_kontakt = random.randint(28, 62)
    elif wtyp == "wachstum":
        kontakte_aktiv = kontakte_gesamt - random.randint(0, 1)
        letzter_kontakt = random.randint(3, 22)
    else:
        kontakte_aktiv = kontakte_gesamt - random.randint(0, 2)
        letzter_kontakt = random.randint(10, 38)

    rows_na.append({
        "kunde_id":                 kid,
        "kunde_name":               kname,
        "branche":                  branche,
        "vertriebsmitarbeiter":     rep,
        "groesse_klasse":           groesse,
        "umsatz_ytd":               int(u25),
        "umsatz_vorjahr":           int(u24),
        "portfolio_abdeckung_pct":  abdeckung,
        "produktluecken_wert":      int(luecken_wert),
        "letzter_kontakt_tage":     letzter_kontakt,
        "offene_opps":              opp["n"],
        "opp_wert":                 int(opp["wert"]),
        "service_tickets":          tickets,
        "service_tickets_vorquartal": tickets_vq,
        "wettbewerber_aktiv":       1 if kid in WETTBEWERB_AKTIV else 0,
        "kontakte_aktiv":           kontakte_aktiv,
        "kontakte_gesamt":          kontakte_gesamt,
        "trigger_event":            TRIGGER_EVENTS.get(kid, ""),
    })

with open(OUT / "next_action_beispiel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows_na[0].keys())
    w.writeheader()
    w.writerows(rows_na)
print(f"✓ next_action_beispiel.csv   ({len(rows_na)} Zeilen)")


# ══════════════════════════════════════════════════════════════════════════
# 4) churn_beispiel.csv – Churn & Retention Intelligence
# ══════════════════════════════════════════════════════════════════════════
# Churn ist die Kehrseite von Cross-Selling: dieselben Kunden, dieselbe
# Produktwelt – nur die umgekehrte Frage. Deshalb derselbe Roster.

PRODUKTGRUPPEN = ["Maschinen", "Zubehör", "Service"]

# Produktanzahl im Vorjahr (2024) je Kunde – für Portfolio-Schrumpfung
produkte_2024 = {}
for r in rows_growth:
    if r["jahr"] == 2024:
        produkte_2024.setdefault(r["kunde_id"], set()).add(r["produkt_id"])

rows_churn = []
for kid, kname, branche, rep, groesse, wtyp in KUNDEN:
    u25 = umsatz_2025.get(kid, 0)
    u24 = umsatz_2024.get(kid, 0)
    prod_akt = len(PORTFOLIO[kid])
    prod_vor = len(produkte_2024.get(kid, PORTFOLIO[kid]))

    if wtyp == "rueckgang":
        # Kunden im Rückgang zeigen das volle Warnbild
        bestell_akt   = random.randint(6, 12)
        bestell_vor   = bestell_akt + random.randint(5, 12)
        letzter_kauf  = random.randint(58, 96)
        service       = random.randint(11, 19)
        service_vor   = random.randint(3, 7)
        reklam        = random.randint(4, 9)
        reklam_vor    = random.randint(0, 2)
        ap_wechsel    = random.randint(2, 4)
        verzug        = random.randint(18, 52)
        mahnungen     = random.randint(1, 4)
        preis_anp     = round(random.uniform(4.5, 9.5), 1)
        angebote_akt  = random.randint(1, 3)
        angebote_vor  = angebote_akt + random.randint(3, 7)
        kontakte_akt  = random.randint(3, 8)
        kontakte_vor  = kontakte_akt + random.randint(6, 14)
        # Portfolio ist geschrumpft
        prod_vor      = prod_akt + random.randint(1, 3)
        wb_gruppe     = random.choice(PRODUKTGRUPPEN)
    elif wtyp == "stabil":
        # Stabile Kunden unter Wettbewerbsdruck zeigen Frühindikatoren, ohne
        # dass der Umsatz schon eingebrochen wäre – das ist der interessante
        # Graubereich, in dem Retention noch billig ist.
        unter_druck = kid in WETTBEWERB_AKTIV
        bestell_akt   = random.randint(9, 14) if unter_druck else random.randint(12, 20)
        bestell_vor   = bestell_akt + (random.randint(4, 8) if unter_druck else random.randint(-2, 3))
        letzter_kauf  = random.randint(38, 58) if unter_druck else random.randint(14, 40)
        service       = random.randint(8, 13)  if unter_druck else random.randint(3, 8)
        service_vor   = random.randint(3, 6)   if unter_druck else random.randint(3, 8)
        reklam        = random.randint(2, 5)   if unter_druck else random.randint(0, 3)
        reklam_vor    = random.randint(0, 2)
        ap_wechsel    = random.randint(1, 3)   if unter_druck else random.randint(0, 1)
        verzug        = random.randint(8, 24)  if unter_druck else random.randint(0, 12)
        mahnungen     = random.randint(0, 2)   if unter_druck else 0
        preis_anp     = round(random.uniform(4.0, 7.5) if unter_druck
                              else random.uniform(1.5, 4.0), 1)
        angebote_akt  = random.randint(2, 4)   if unter_druck else random.randint(3, 7)
        angebote_vor  = angebote_akt + (random.randint(2, 5) if unter_druck
                                        else random.randint(-1, 2))
        kontakte_akt  = random.randint(7, 11)  if unter_druck else random.randint(10, 18)
        kontakte_vor  = kontakte_akt + (random.randint(4, 9) if unter_druck
                                        else random.randint(-3, 3))
        wb_gruppe     = random.choice(PRODUKTGRUPPEN) if unter_druck else ""
    else:  # wachstum
        bestell_akt   = random.randint(16, 26)
        bestell_vor   = bestell_akt - random.randint(2, 7)
        letzter_kauf  = random.randint(3, 24)
        service       = random.randint(1, 5)
        service_vor   = random.randint(2, 6)
        reklam        = random.randint(0, 1)
        reklam_vor    = random.randint(0, 2)
        ap_wechsel    = random.randint(0, 1)
        verzug        = random.randint(0, 8)
        mahnungen     = 0
        preis_anp     = round(random.uniform(1.0, 3.5), 1)
        angebote_akt  = random.randint(5, 11)
        angebote_vor  = angebote_akt - random.randint(1, 4)
        kontakte_akt  = random.randint(14, 24)
        kontakte_vor  = kontakte_akt - random.randint(2, 6)
        wb_gruppe     = random.choice(PRODUKTGRUPPEN) if kid in WETTBEWERB_AKTIV else ""

    rows_churn.append({
        "kunde_id":                  kid,
        "kunde_name":                kname,
        "branche":                   branche,
        "vertriebsmitarbeiter":      rep,
        "groesse_klasse":            groesse,
        "umsatz_ytd":                int(u25),
        "umsatz_vorjahr":            int(u24),
        "bestellungen_ytd":          bestell_akt,
        "bestellungen_vorjahr":      bestell_vor,
        "produkte_aktuell":          prod_akt,
        "produkte_vorjahr":          prod_vor,
        "tage_seit_letztem_einkauf": letzter_kauf,
        "service_faelle":            service,
        "service_faelle_vorjahr":    service_vor,
        "reklamationen":             reklam,
        "reklamationen_vorjahr":     reklam_vor,
        "ansprechpartner_wechsel":   ap_wechsel,
        "zahlungsverzug_tage":       verzug,
        "mahnungen":                 mahnungen,
        "wettbewerber_aktiv":        1 if kid in WETTBEWERB_AKTIV else 0,
        "wettbewerber_produktgruppe": wb_gruppe,
        "preisanpassung_pct":        preis_anp,
        "angebote_ytd":              angebote_akt,
        "angebote_vorjahr":          angebote_vor,
        "kontakte_ytd":              kontakte_akt,
        "kontakte_vorjahr":          kontakte_vor,
    })

with open(OUT / "churn_beispiel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows_churn[0].keys())
    w.writeheader()
    w.writerows(rows_churn)
print(f"✓ churn_beispiel.csv         ({len(rows_churn)} Zeilen)")
