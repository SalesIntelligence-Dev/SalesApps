"""
Customer Churn & Retention Intelligence.

Kernfrage: "Welche Kunden drohen wir zu verlieren – und warum?"

Das ist die inverse Seite der Customer Growth Engine: dieselben Kunden,
derselbe Produktgraph, die umgekehrte Frage.
  Growth : Wo können wir wachsen?
  Churn  : Wo verlieren wir Umsatz?

Ansatz (gewichtetes Multi-Signal-Scoring, vollständig erklärbar):
  11 Signaldimensionen, jede liefert einen Schweregrad 0–1.
  Churn Risk = Σ (Gewicht × Schweregrad)  →  0–100 %
  Umsatzrisiko = Jahresumsatz × Churn Risk
  Die dominante Ursache bestimmt die empfohlene Gegenmaßnahme.

Bewusst kein Blackbox-Klassifikator: Ein Vertriebsleiter muss die Ursache
sehen können, sonst ist die Zahl im Kundengespräch wertlos.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import pandas as pd

# Gewichte der Signaldimensionen (Summe = 100)
GEWICHTE = {
    "umsatz":        18,
    "bestellfreq":   12,
    "letzter_kauf":  12,
    "wettbewerb":    12,
    "portfolio":     10,
    "service":        9,
    "ansprechpartner": 8,
    "zahlung":        6,
    "angebote":       5,
    "preis":          5,
    "kommunikation":  3,
}

SEGMENT_FARBE = {
    "Kritisch":   "#f87171",
    "Gefährdet":  "#fb923c",
    "Beobachten": "#fbbf24",
    "Stabil":     "#34d399",
}


@dataclass
class ChurnResult:
    kpi: dict                    = field(default_factory=dict)
    kunden: list[dict]           = field(default_factory=list)
    scatter_json: str            = "{}"
    selected_kunde: dict | None  = None
    ursachen_verteilung: list[dict] = field(default_factory=list)


def _seg(risk: float) -> str:
    if risk >= 65:
        return "Kritisch"
    if risk >= 40:
        return "Gefährdet"
    if risk >= 25:
        return "Beobachten"
    return "Stabil"


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _signale(r: pd.Series) -> list[dict]:
    """Bewertet alle elf Signaldimensionen für einen Kunden."""
    def num(key, default=0.0):
        try:
            v = float(r.get(key, default))
            return default if pd.isna(v) else v
        except (TypeError, ValueError):
            return default

    s = []

    # ── 1. Umsatzentwicklung ──────────────────────────────────────────────
    u_akt, u_vor = num("umsatz_ytd"), num("umsatz_vorjahr")
    delta_u = ((u_akt - u_vor) / u_vor * 100) if u_vor > 0 else 0.0
    sev = _clamp(-delta_u / 30)          # −30 % ⇒ voller Ausschlag
    s.append({
        "key": "umsatz", "label": "Umsatzentwicklung", "icon": "📉",
        "severity": sev,
        "wert": f"{delta_u:+.0f} %",
        "text": (f"Umsatz {delta_u:+.0f} % gegenüber Vorjahr "
                 f"({u_vor - u_akt:,.0f} € Rückgang)." if delta_u < 0
                 else f"Umsatz {delta_u:+.0f} % gegenüber Vorjahr – stabil bis wachsend."),
    })

    # ── 2. Bestellfrequenz ────────────────────────────────────────────────
    b_akt, b_vor = num("bestellungen_ytd"), num("bestellungen_vorjahr")
    delta_b = ((b_akt - b_vor) / b_vor * 100) if b_vor > 0 else 0.0
    sev = _clamp(-delta_b / 40)
    s.append({
        "key": "bestellfreq", "label": "Bestellfrequenz", "icon": "🔄",
        "severity": sev,
        "wert": f"{delta_b:+.0f} %",
        "text": f"{b_akt:.0f} Bestellungen im laufenden Jahr gegenüber {b_vor:.0f} im Vorjahr.",
    })

    # ── 3. Letzter Einkauf ────────────────────────────────────────────────
    tage = num("tage_seit_letztem_einkauf")
    sev = _clamp((tage - 30) / 70)       # ab 30 Tagen kritisch werdend
    s.append({
        "key": "letzter_kauf", "label": "Letzter Einkauf", "icon": "📅",
        "severity": sev,
        "wert": f"{tage:.0f} Tage",
        "text": f"Letzter Auftragseingang vor {tage:.0f} Tagen.",
    })

    # ── 4. Wettbewerb ─────────────────────────────────────────────────────
    wb = int(num("wettbewerber_aktiv"))
    wb_gruppe = str(r.get("wettbewerber_produktgruppe", "") or "").strip()
    if wb_gruppe.lower() in {"nan", "none", "-"}:
        wb_gruppe = ""
    sev = 0.0
    if wb and wb_gruppe:
        sev = 1.0 if delta_u < 0 else 0.6
    elif wb:
        sev = 0.5
    s.append({
        "key": "wettbewerb", "label": "Wettbewerbsdruck", "icon": "⚔️",
        "severity": sev,
        "wert": wb_gruppe if wb_gruppe else ("aktiv" if wb else "—"),
        "text": (f"Wettbewerber gewinnt Anteile in der Produktgruppe „{wb_gruppe}“."
                 if wb and wb_gruppe else
                 "Wettbewerber im Account aktiv." if wb else
                 "Kein Wettbewerbsdruck erkennbar."),
    })

    # ── 5. Portfolio-Schrumpfung ──────────────────────────────────────────
    p_akt, p_vor = num("produkte_aktuell"), num("produkte_vorjahr")
    verloren = max(0.0, p_vor - p_akt)
    sev = _clamp(verloren / 3)
    s.append({
        "key": "portfolio", "label": "Produktportfolio", "icon": "📦",
        "severity": sev,
        "wert": f"{p_akt:.0f} von {p_vor:.0f}",
        "text": (f"{verloren:.0f} Produktgruppe(n) gegenüber Vorjahr abbestellt."
                 if verloren > 0 else "Portfolio unverändert oder ausgebaut."),
    })

    # ── 6. Service & Reklamationen ────────────────────────────────────────
    sf, sf_v = num("service_faelle"), num("service_faelle_vorjahr")
    rk, rk_v = num("reklamationen"), num("reklamationen_vorjahr")
    anstieg_sf = (sf / sf_v) if sf_v > 0 else (2.0 if sf > 3 else 1.0)
    sev = _clamp((anstieg_sf - 1.2) / 1.8) * 0.6 + _clamp((rk - rk_v) / 5) * 0.4
    sev = _clamp(sev)
    s.append({
        "key": "service", "label": "Service & Reklamationen", "icon": "🛠️",
        "severity": sev,
        "wert": f"{sf:.0f} / {rk:.0f}",
        "text": (f"{sf:.0f} Servicefälle (Vorjahr {sf_v:.0f}), "
                 f"{rk:.0f} Reklamationen (Vorjahr {rk_v:.0f})."),
    })

    # ── 7. Ansprechpartnerwechsel ─────────────────────────────────────────
    apw = num("ansprechpartner_wechsel")
    sev = _clamp(apw / 3)
    s.append({
        "key": "ansprechpartner", "label": "Ansprechpartnerwechsel", "icon": "👤",
        "severity": sev,
        "wert": f"{apw:.0f}",
        "text": (f"{apw:.0f} Wechsel auf Kundenseite – gewachsene Beziehungen gehen verloren."
                 if apw > 0 else "Ansprechpartner stabil."),
    })

    # ── 8. Zahlungsverhalten ──────────────────────────────────────────────
    verzug, mahn = num("zahlungsverzug_tage"), num("mahnungen")
    sev = _clamp(verzug / 45) * 0.6 + _clamp(mahn / 3) * 0.4
    sev = _clamp(sev)
    s.append({
        "key": "zahlung", "label": "Zahlungsverhalten", "icon": "💳",
        "severity": sev,
        "wert": f"{verzug:.0f} Tage",
        "text": (f"Ø {verzug:.0f} Tage Zahlungsverzug, {mahn:.0f} Mahnung(en)."
                 if verzug > 0 or mahn > 0 else "Zahlungsverhalten unauffällig."),
    })

    # ── 9. Angebotsaktivität ──────────────────────────────────────────────
    a_akt, a_vor = num("angebote_ytd"), num("angebote_vorjahr")
    delta_a = ((a_akt - a_vor) / a_vor * 100) if a_vor > 0 else 0.0
    sev = _clamp(-delta_a / 50)
    s.append({
        "key": "angebote", "label": "Angebotsaktivität", "icon": "📄",
        "severity": sev,
        "wert": f"{delta_a:+.0f} %",
        "text": f"{a_akt:.0f} Anfragen im laufenden Jahr gegenüber {a_vor:.0f} im Vorjahr.",
    })

    # ── 10. Preisentwicklung ──────────────────────────────────────────────
    preis = num("preisanpassung_pct")
    sev = _clamp((preis - 3) / 6)
    s.append({
        "key": "preis", "label": "Preisentwicklung", "icon": "💶",
        "severity": sev,
        "wert": f"+{preis:.1f} %",
        "text": (f"Preise um {preis:.1f} % angehoben – bei aktivem Wettbewerb ein Abwanderungsanlass."
                 if preis > 3 else f"Preisanpassung von {preis:.1f} % im üblichen Rahmen."),
    })

    # ── 11. Kommunikationsintensität ──────────────────────────────────────
    k_akt, k_vor = num("kontakte_ytd"), num("kontakte_vorjahr")
    delta_k = ((k_akt - k_vor) / k_vor * 100) if k_vor > 0 else 0.0
    sev = _clamp(-delta_k / 50)
    s.append({
        "key": "kommunikation", "label": "Kommunikationsintensität", "icon": "💬",
        "severity": sev,
        "wert": f"{delta_k:+.0f} %",
        "text": f"{k_akt:.0f} Kundenkontakte gegenüber {k_vor:.0f} im Vorjahr.",
    })

    # Beitrag zum Gesamtscore
    for x in s:
        x["gewicht"] = GEWICHTE[x["key"]]
        x["beitrag"] = round(x["gewicht"] * x["severity"], 2)
        x["severity"] = round(x["severity"], 3)

    return s


def _gegenmassnahme(dominant: str, r: pd.Series, risk: float) -> dict:
    wb_gruppe = str(r.get("wettbewerber_produktgruppe", "") or "").strip()
    if wb_gruppe.lower() in {"nan", "none", "-"}:
        wb_gruppe = "der betroffenen Produktgruppe"

    mapping = {
        "umsatz": (
            "Executive Review ansetzen",
            "Volumenrückgang auf Geschäftsführungsebene adressieren. Rahmenvertrag, "
            "Mengenstaffel und Lieferperformance gemeinsam auf den Tisch legen."),
        "bestellfreq": (
            "Bestellprozess prüfen",
            "Rückgang der Bestellfrequenz deutet auf abgewanderte Teilbedarfe. "
            "Abrufvereinbarung oder Konsignationslager anbieten."),
        "letzter_kauf": (
            "Reaktivierung sofort",
            "Der Kunde hat seit Monaten nicht bestellt. Persönlicher Besuch statt "
            "E-Mail – klären, ob bereits ein Wettbewerber liefert."),
        "wettbewerb": (
            f"Produktgruppe {wb_gruppe} verteidigen",
            f"Wettbewerber gewinnt Anteile in {wb_gruppe}. Battlecard einsetzen, "
            f"Referenzen und Total-Cost-Argumentation gezielt gegenhalten."),
        "portfolio": (
            "Abbestellte Produkte zurückgewinnen",
            "Der Kunde hat Produktgruppen abbestellt. Ursache klären: Preis, Qualität "
            "oder Wettbewerber – und ein Rückgewinnungsangebot aufsetzen."),
        "service": (
            "Service-Eskalation auflösen",
            "Gestiegene Servicefälle und Reklamationen sind der häufigste Vorläufer von "
            "Abwanderung. Gemeinsames Review mit Technik und Qualitätssicherung."),
        "ansprechpartner": (
            "Beziehungsnetz neu aufbauen",
            "Nach Ansprechpartnerwechseln fehlt die gewachsene Bindung. Neue Kontakte "
            "aktiv vorstellen und Entscheidungswege neu kartieren."),
        "zahlung": (
            "Kaufmännisches Gespräch führen",
            "Zahlungsverzug kann Liquiditätsprobleme oder Unzufriedenheit signalisieren. "
            "Gemeinsam mit dem Innendienst klären, bevor der Kunde still abwandert."),
        "angebote": (
            "Anfrageverhalten hinterfragen",
            "Deutlich weniger Anfragen bedeutet, dass der Bedarf woanders platziert wird. "
            "Bedarfsplanung des Kunden aktiv erfragen."),
        "preis": (
            "Preisgespräch vorbereiten",
            "Die Preisanhebung trifft auf aktiven Wettbewerb. Wertargumentation und "
            "Mehrjahresvereinbarung als Alternative zum Nachlass vorbereiten."),
        "kommunikation": (
            "Kontaktfrequenz erhöhen",
            "Die Kommunikation ist stark zurückgegangen. Feste Besuchsfrequenz "
            "vereinbaren, bevor die Beziehung abreißt."),
    }
    aktion, detail = mapping.get(dominant, ("Kundengespräch führen", "Status klären."))
    frist = "innerhalb 48 h" if risk >= 65 else ("diese Woche" if risk >= 40 else "in 2 Wochen")
    return {"aktion": aktion, "detail": detail, "frist": frist}


def _build_scatter(kunden: list[dict]) -> str:
    if not kunden:
        return json.dumps({"data": [], "layout": {}})

    max_risk_eur = max((k["umsatzrisiko"] for k in kunden), default=1) or 1
    traces = []
    for seg in ["Kritisch", "Gefährdet", "Beobachten", "Stabil"]:
        grp = [k for k in kunden if k["segment"] == seg]
        if not grp:
            continue
        traces.append({
            "type": "scatter", "mode": "markers",
            "name": seg,
            "x": [k["umsatz"] for k in grp],
            "y": [k["churn_risk"] for k in grp],
            "hovertext": [
                f"<b>{k['name']}</b><br>"
                f"Churn-Risiko: {k['churn_risk']:.0f} %<br>"
                f"Umsatz: {k['umsatz']:,.0f} €<br>"
                f"Umsatzrisiko: {k['umsatzrisiko']:,.0f} €<br>"
                f"Hauptursache: {k['hauptursache']}"
                for k in grp
            ],
            "hoverinfo": "text",
            "marker": {
                "size": [16 + 44 * (k["umsatzrisiko"] / max_risk_eur) for k in grp],
                "color": SEGMENT_FARBE[seg],
                "opacity": 0.75,
                "line": {"color": "#0f172a", "width": 1.5},
            },
        })

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(20,30,50,0.4)",
        "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
        "xaxis": {"title": "Jahresumsatz (€)", "gridcolor": "rgba(148,163,184,0.1)", "color": "#94a3b8"},
        "yaxis": {"title": "Churn-Risiko (%)", "gridcolor": "rgba(148,163,184,0.1)",
                  "color": "#94a3b8", "range": [0, 100]},
        "font": {"family": "Inter, sans-serif", "size": 11, "color": "#94a3b8"},
        "legend": {"orientation": "h", "y": -0.18, "bgcolor": "rgba(0,0,0,0)"},
        "shapes": [
            {"type": "line", "y0": 65, "y1": 65, "x0": 0, "x1": 1,
             "xref": "paper", "yref": "y",
             "line": {"color": "rgba(248,113,113,0.45)", "dash": "dot"}},
            {"type": "line", "y0": 40, "y1": 40, "x0": 0, "x1": 1,
             "xref": "paper", "yref": "y",
             "line": {"color": "rgba(251,146,60,0.35)", "dash": "dot"}},
        ],
        "hovermode": "closest",
    }
    return json.dumps({"data": traces, "layout": layout}, default=float)


def analyse(df: pd.DataFrame, selected_kunde_id: str | None = None) -> ChurnResult:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    result = ChurnResult()

    kunden = []
    for _, r in df.iterrows():
        sig = _signale(r)
        risk = sum(x["beitrag"] for x in sig)          # 0–100
        risk = max(0.0, min(100.0, risk))

        umsatz = float(r.get("umsatz_ytd", 0) or 0)
        umsatzrisiko = umsatz * risk / 100

        sig_sorted = sorted(sig, key=lambda x: -x["beitrag"])
        dominant = sig_sorted[0]
        massnahme = _gegenmassnahme(dominant["key"], r, risk)

        u_vor = float(r.get("umsatz_vorjahr", 0) or 0)
        delta_u = ((umsatz - u_vor) / u_vor * 100) if u_vor > 0 else 0.0

        wb_gruppe = str(r.get("wettbewerber_produktgruppe", "") or "").strip()
        if wb_gruppe.lower() in {"nan", "none", "-"}:
            wb_gruppe = ""

        kunden.append({
            "id":            r.get("kunde_id", ""),
            "name":          r.get("kunde_name", ""),
            "branche":       r.get("branche", ""),
            "rep":           r.get("vertriebsmitarbeiter", ""),
            "groesse":       r.get("groesse_klasse", ""),
            "umsatz":        round(umsatz, 0),
            "umsatz_vorjahr": round(u_vor, 0),
            "wachstum":      round(delta_u, 1),
            "churn_risk":    round(risk, 0),
            "umsatzrisiko":  round(umsatzrisiko, 0),
            "segment":       _seg(risk),
            "hauptursache":  dominant["label"],
            "hauptursache_key": dominant["key"],
            "hauptursache_text": dominant["text"],
            "signale":       sig_sorted,
            "top_signale":   [x for x in sig_sorted if x["severity"] >= 0.25][:4],
            "letzter_kauf":  int(float(r.get("tage_seit_letztem_einkauf", 0) or 0)),
            "wb_gruppe":     wb_gruppe,
            "aktion":        massnahme["aktion"],
            "aktion_detail": massnahme["detail"],
            "frist":         massnahme["frist"],
        })

    kunden.sort(key=lambda x: -x["umsatzrisiko"])
    result.kunden = kunden

    # ── KPIs ──────────────────────────────────────────────────────────────
    gesamt_umsatz = sum(k["umsatz"] for k in kunden)
    gesamt_risiko = sum(k["umsatzrisiko"] for k in kunden)
    kritisch = [k for k in kunden if k["segment"] == "Kritisch"]

    result.kpi = {
        "n_kunden":        len(kunden),
        "gesamt_umsatz":   round(gesamt_umsatz, 0),
        "umsatzrisiko":    round(gesamt_risiko, 0),
        "risiko_pct":      round(gesamt_risiko / gesamt_umsatz * 100, 1) if gesamt_umsatz else 0,
        "n_kritisch":      len(kritisch),
        "risiko_kritisch": round(sum(k["umsatzrisiko"] for k in kritisch), 0),
        "n_gefaehrdet":    sum(1 for k in kunden if k["segment"] == "Gefährdet"),
        "n_stabil":        sum(1 for k in kunden if k["segment"] == "Stabil"),
        "avg_risk":        round(sum(k["churn_risk"] for k in kunden) / len(kunden), 0) if kunden else 0,
    }

    # ── Ursachenverteilung ────────────────────────────────────────────────
    ursachen: dict[str, dict] = {}
    for k in kunden:
        e = ursachen.setdefault(k["hauptursache_key"], {
            "label": k["hauptursache"], "anzahl": 0, "umsatzrisiko": 0.0,
        })
        e["anzahl"] += 1
        e["umsatzrisiko"] += k["umsatzrisiko"]
    result.ursachen_verteilung = sorted(
        [{"key": kk, **vv, "umsatzrisiko": round(vv["umsatzrisiko"], 0)}
         for kk, vv in ursachen.items()],
        key=lambda x: -x["umsatzrisiko"]
    )

    result.scatter_json = _build_scatter(kunden)

    if selected_kunde_id:
        result.selected_kunde = next(
            (k for k in kunden if k["id"] == selected_kunde_id), None)

    return result
