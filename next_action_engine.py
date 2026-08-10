"""
Next Best Action – AI Sales Copilot.

Kernfrage: "Was sind die wichtigsten Aktionen, die dieser Verkäufer heute
durchführen sollte – und warum?"

Das ist der Übergang von Analytics zu Decision Intelligence: Das System
liefert keine Kennzahl, sondern eine priorisierte Handlungsliste mit
Begründung, erwartetem Wert und Frist.

Ansatz (Signal-Fusion, vollständig erklärbar):
  1. Signaldetektion je Kunde über sieben Signalklassen
  2. Jedes Signal hat Gewicht, Dringlichkeit und einen Wert-am-Risiko
  3. Priority Score = Σ (Gewicht × Dringlichkeit) × log-normierter Wert
  4. Das dominante Signal bestimmt die empfohlene Aktion
  5. Ranking → "Heutige Prioritäten"
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import pandas as pd

# Signalklassen: (Gewicht, Dringlichkeit 1–3, Farbe, Icon)
SIGNAL_META = {
    "churn":       {"gewicht": 30, "dringlichkeit": 3, "farbe": "#f87171", "icon": "📉", "label": "Umsatzrückgang"},
    "wettbewerb":  {"gewicht": 28, "dringlichkeit": 3, "farbe": "#f87171", "icon": "⚔️", "label": "Wettbewerbsdruck"},
    "deal_risk":   {"gewicht": 26, "dringlichkeit": 3, "farbe": "#fb923c", "icon": "⏳", "label": "Deal in Gefahr"},
    "trigger":     {"gewicht": 24, "dringlichkeit": 2, "farbe": "#34d399", "icon": "🚀", "label": "Expansionssignal"},
    "cross_sell":  {"gewicht": 18, "dringlichkeit": 1, "farbe": "#60a5fa", "icon": "🎯", "label": "Cross-Sell-Chance"},
    "service":     {"gewicht": 16, "dringlichkeit": 2, "farbe": "#fbbf24", "icon": "🛠️", "label": "Service-Eskalation"},
    "beziehung":   {"gewicht": 14, "dringlichkeit": 2, "farbe": "#a78bfa", "icon": "👥", "label": "Beziehungsrisiko"},
}

DRINGLICHKEIT_FRIST = {3: "innerhalb 48 h", 2: "diese Woche", 1: "in 2 Wochen"}


@dataclass
class ActionResult:
    kpi: dict                    = field(default_factory=dict)
    aktionen: list[dict]         = field(default_factory=list)
    reps: list[str]              = field(default_factory=list)
    selected_rep: str | None     = None
    signal_verteilung: list[dict] = field(default_factory=list)


def _detect_signals(r: pd.Series) -> list[dict]:
    """Erkennt alle zutreffenden Signale für einen Kunden."""
    signale = []

    umsatz     = float(r.get("umsatz_ytd", 0))
    vorjahr    = float(r.get("umsatz_vorjahr", 0))
    wachstum   = ((umsatz - vorjahr) / vorjahr * 100) if vorjahr > 0 else 0.0
    kontakt    = float(r.get("letzter_kontakt_tage", 0))
    opps       = int(r.get("offene_opps", 0))
    opp_wert   = float(r.get("opp_wert", 0))
    tickets    = float(r.get("service_tickets", 0))
    tickets_vq = float(r.get("service_tickets_vorquartal", 0))
    wb_aktiv   = int(r.get("wettbewerber_aktiv", 0))
    luecken    = float(r.get("produktluecken_wert", 0))
    abdeckung  = float(r.get("portfolio_abdeckung_pct", 0))
    k_aktiv    = float(r.get("kontakte_aktiv", 0))
    k_gesamt   = float(r.get("kontakte_gesamt", 1))
    trigger    = str(r.get("trigger_event", "") or "").strip()
    if trigger.lower() in {"nan", "none", "-", "—"}:
        trigger = ""

    # ── 1. Umsatzrückgang ─────────────────────────────────────────────────
    if wachstum <= -12:
        verlust = vorjahr - umsatz
        signale.append({
            "typ": "churn",
            "text": f"Umsatz {wachstum:+.0f} % gegenüber Vorjahr ({verlust:,.0f} € Rückgang).",
            "wert": verlust,
            "staerke": min(2.0, abs(wachstum) / 15),
        })

    # ── 2. Wettbewerbsdruck ───────────────────────────────────────────────
    if wb_aktiv and wachstum < 0:
        signale.append({
            "typ": "wettbewerb",
            "text": "Wettbewerber aktiv im Account bei gleichzeitig sinkendem Bestellvolumen.",
            "wert": max(umsatz * 0.35, 0),
            "staerke": 1.6,
        })
    elif wb_aktiv:
        signale.append({
            "typ": "wettbewerb",
            "text": "Wettbewerber ist im Account aktiv – Position verteidigen.",
            "wert": umsatz * 0.18,
            "staerke": 1.0,
        })

    # ── 3. Deal in Gefahr ─────────────────────────────────────────────────
    if opps > 0 and kontakt > 25:
        signale.append({
            "typ": "deal_risk",
            "text": f"{opps} offene Opportunity(s) über {opp_wert:,.0f} €, "
                    f"aber seit {kontakt:.0f} Tagen kein Kontakt.",
            "wert": opp_wert,
            "staerke": min(2.0, kontakt / 30),
        })
    elif opps > 0 and opp_wert > 150_000:
        signale.append({
            "typ": "deal_risk",
            "text": f"{opps} offene Opportunity(s) über {opp_wert:,.0f} € in aktiver Bearbeitung.",
            "wert": opp_wert * 0.5,
            "staerke": 0.8,
        })

    # ── 4. Expansionssignal ───────────────────────────────────────────────
    if trigger:
        signale.append({
            "typ": "trigger",
            "text": f"{trigger} – konkreter Anlass für ein Erweiterungsgespräch.",
            "wert": umsatz * 0.30,
            "staerke": 1.5,
        })

    # ── 5. Cross-Sell-Chance ──────────────────────────────────────────────
    if luecken > 0 and abdeckung < 70:
        signale.append({
            "typ": "cross_sell",
            "text": f"Portfolio-Abdeckung nur {abdeckung:.0f} % – "
                    f"Produktlücken im Wert von {luecken:,.0f} €.",
            "wert": luecken,
            "staerke": 1.2 if wachstum > 0 else 0.8,
        })

    # ── 6. Service-Eskalation ─────────────────────────────────────────────
    if tickets_vq > 0 and tickets > tickets_vq * 1.6:
        signale.append({
            "typ": "service",
            "text": f"Service-Tickets von {tickets_vq:.0f} auf {tickets:.0f} gestiegen "
                    f"(+{(tickets/tickets_vq-1)*100:.0f} %) – Zufriedenheitsrisiko.",
            "wert": umsatz * 0.20,
            "staerke": min(1.8, tickets / max(tickets_vq, 1) / 2),
        })

    # ── 7. Beziehungsrisiko ───────────────────────────────────────────────
    inaktiv = k_gesamt - k_aktiv
    if k_gesamt > 0 and (k_aktiv / k_gesamt) < 0.65:
        signale.append({
            "typ": "beziehung",
            "text": f"Nur {k_aktiv:.0f} von {k_gesamt:.0f} Kontakten aktiv – "
                    f"{inaktiv:.0f} Ansprechpartner nicht mehr erreichbar.",
            "wert": umsatz * 0.15,
            "staerke": 1.0 + inaktiv * 0.2,
        })

    return signale


def _empfehlung(dominant: str, r: pd.Series, signale: list[dict]) -> dict:
    """Leitet die konkrete Handlungsempfehlung aus dem dominanten Signal ab."""
    name    = r.get("kunde_name", "Kunde")
    luecken = float(r.get("produktluecken_wert", 0))
    trigger = str(r.get("trigger_event", "") or "").strip()
    if trigger.lower() in {"nan", "none", "-", "—"}:
        trigger = "Expansionssignal erkannt"

    mapping = {
        "churn": {
            "aktion": "Eskalationsgespräch mit Einkaufsleitung",
            "detail": "Ursachen des Volumenrückgangs klären, Wettbewerbssituation offen "
                      "ansprechen und Rahmenvereinbarung neu verhandeln.",
        },
        "wettbewerb": {
            "aktion": "Verteidigungsgespräch führen",
            "detail": "Battlecard gegen den aktiven Wettbewerber einsetzen, technische "
                      "Differenzierung und Servicequalität in den Vordergrund stellen.",
        },
        "deal_risk": {
            "aktion": "Offene Opportunity reaktivieren",
            "detail": "Verbindlichen nächsten Schritt mit Datum vereinbaren und "
                      "Entscheidungsstand im Buying Center verifizieren.",
        },
        "trigger": {
            "aktion": "Termin innerhalb 48 h vereinbaren",
            "detail": f"{trigger} – Bedarf frühzeitig besetzen, bevor der Wettbewerb "
                      f"reagiert. Kapazitätserweiterung und Servicepaket anbieten.",
        },
        "cross_sell": {
            "aktion": "Produkterweiterung vorstellen",
            "detail": f"Portfolio-Lücken im Wert von {luecken:,.0f} € gezielt adressieren – "
                      f"vergleichbare Kunden nutzen diese Produkte bereits.",
        },
        "service": {
            "aktion": "Service-Review ansetzen",
            "detail": "Gemeinsame Ticket-Analyse mit Technik und Kunde, danach Upgrade "
                      "auf höherwertigen Servicevertrag anbieten.",
        },
        "beziehung": {
            "aktion": "Beziehungsnetz neu aufbauen",
            "detail": "Neue Ansprechpartner identifizieren und persönlich vorstellen, "
                      "bevor der Account beziehungslos wird.",
        },
    }
    return mapping.get(dominant, {
        "aktion": "Kundengespräch führen",
        "detail": f"Aktuellen Status bei {name} klären.",
    })


def analyse(df: pd.DataFrame, selected_rep: str | None = None) -> ActionResult:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    result = ActionResult()

    reps = sorted(df["vertriebsmitarbeiter"].dropna().unique().tolist()) \
        if "vertriebsmitarbeiter" in df.columns else []
    result.reps = reps
    result.selected_rep = selected_rep

    aktionen = []
    for _, r in df.iterrows():
        signale = _detect_signals(r)
        if not signale:
            continue

        # Signale anreichern und Score berechnen
        for s in signale:
            meta = SIGNAL_META[s["typ"]]
            s["label"]         = meta["label"]
            s["icon"]          = meta["icon"]
            s["farbe"]         = meta["farbe"]
            s["dringlichkeit"] = meta["dringlichkeit"]
            s["beitrag"]       = meta["gewicht"] * meta["dringlichkeit"] * s["staerke"]

        signale.sort(key=lambda x: -x["beitrag"])
        dominant = signale[0]

        # Wert am Risiko / Chance
        wert_gesamt = sum(s["wert"] for s in signale)
        # Log-Normierung dämpft Größeneffekte, ohne sie zu ignorieren
        wert_faktor = math.log10(max(wert_gesamt, 1000) / 1000 + 1) + 0.4

        roh_score = sum(s["beitrag"] for s in signale) * wert_faktor
        dringlichkeit = max(s["dringlichkeit"] for s in signale)

        empf = _empfehlung(dominant["typ"], r, signale)

        umsatz  = float(r.get("umsatz_ytd", 0))
        vorjahr = float(r.get("umsatz_vorjahr", 0))
        wachstum = ((umsatz - vorjahr) / vorjahr * 100) if vorjahr > 0 else 0.0

        aktionen.append({
            "kunde_id":    r.get("kunde_id", ""),
            "kunde_name":  r.get("kunde_name", ""),
            "branche":     r.get("branche", ""),
            "rep":         r.get("vertriebsmitarbeiter", ""),
            "umsatz":      round(umsatz, 0),
            "wachstum":    round(wachstum, 1),
            "abdeckung":   round(float(r.get("portfolio_abdeckung_pct", 0)), 0),
            "letzter_kontakt": int(float(r.get("letzter_kontakt_tage", 0))),
            "offene_opps": int(r.get("offene_opps", 0)),
            "opp_wert":    round(float(r.get("opp_wert", 0)), 0),
            "signale":     signale,
            "n_signale":   len(signale),
            "dominant":    dominant["typ"],
            "dominant_label": dominant["label"],
            "dominant_farbe": dominant["farbe"],
            "wert_am_risiko": round(wert_gesamt, 0),
            "score":       round(roh_score, 1),
            "dringlichkeit": dringlichkeit,
            "frist":       DRINGLICHKEIT_FRIST[dringlichkeit],
            "aktion":      empf["aktion"],
            "aktion_detail": empf["detail"],
        })

    # Score auf 0–100 normieren
    if aktionen:
        max_score = max(a["score"] for a in aktionen) or 1
        for a in aktionen:
            a["score"] = round(a["score"] / max_score * 100, 0)

    aktionen.sort(key=lambda x: (-x["dringlichkeit"], -x["score"]))

    # Filter nach Vertriebsmitarbeiter
    if selected_rep:
        aktionen = [a for a in aktionen if a["rep"] == selected_rep]

    # Rang vergeben
    for i, a in enumerate(aktionen, 1):
        a["rang"] = i

    result.aktionen = aktionen

    # ── Signalverteilung (für Balkendiagramm) ─────────────────────────────
    verteilung: dict[str, int] = {}
    for a in aktionen:
        for s in a["signale"]:
            verteilung[s["typ"]] = verteilung.get(s["typ"], 0) + 1
    result.signal_verteilung = sorted(
        [{"typ": k, "label": SIGNAL_META[k]["label"], "icon": SIGNAL_META[k]["icon"],
          "farbe": SIGNAL_META[k]["farbe"], "anzahl": v}
         for k, v in verteilung.items()],
        key=lambda x: -x["anzahl"]
    )

    # ── KPIs ──────────────────────────────────────────────────────────────
    result.kpi = {
        "n_aktionen":     len(aktionen),
        "n_kritisch":     sum(1 for a in aktionen if a["dringlichkeit"] == 3),
        "wert_gesamt":    round(sum(a["wert_am_risiko"] for a in aktionen), 0),
        "wert_kritisch":  round(sum(a["wert_am_risiko"] for a in aktionen
                                    if a["dringlichkeit"] == 3), 0),
        "n_signale":      sum(a["n_signale"] for a in aktionen),
        "avg_score":      round(sum(a["score"] for a in aktionen) / len(aktionen), 0) if aktionen else 0,
    }

    return result
