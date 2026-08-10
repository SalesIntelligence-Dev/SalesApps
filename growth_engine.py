"""
Customer Growth Engine – Recommender System für Kundenpotenzial.

Kernfrage: "Wo liegt bei welchem Kunden wie viel Umsatzpotenzial – und mit
welchem Produkt hebe ich es?"

Ansatz (Peer-Group Collaborative Filtering, erklärbar):
  1. Share of Wallet: Portfolio-Abdeckung je Kunde
  2. Peer-Group: Kunden gleicher Branche + Größenklasse
  3. Produktlücken: Produkte, die ≥ 50 % der Peers kaufen, der Kunde aber nicht
  4. Potenzial in EUR = Median-Umsatz der Peers mit diesem Produkt,
     skaliert auf die Größenklasse des Kunden
  5. Wachstumstrend (YoY) → Segmentierung Wachstum / Stabil / Rückgang
  6. Ranking der Kunden nach realisierbarem Gesamtpotenzial
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import pandas as pd

# Anteil der Peers, ab dem ein Produkt als "Standard im Segment" gilt
PEER_SCHWELLE = 0.50
# Wie viel des theoretischen Potenzials realistisch abrufbar ist
REALISIERUNGSQUOTE = 0.45

SEGMENT_FARBE = {
    "Wachstum":  "#34d399",
    "Stabil":    "#60a5fa",
    "Rückgang":  "#f87171",
}


@dataclass
class GrowthResult:
    kpi: dict                       = field(default_factory=dict)
    kunden: list[dict]              = field(default_factory=list)
    produkt_potenzial: list[dict]   = field(default_factory=list)
    bubble_json: str                = "{}"
    selected_kunde: dict | None     = None
    selected_luecken: list[dict]    = field(default_factory=list)
    selected_portfolio: list[dict]  = field(default_factory=list)
    selected_trend: list[dict]      = field(default_factory=list)


def _segment(wachstum_pct: float) -> str:
    if wachstum_pct >= 5:
        return "Wachstum"
    if wachstum_pct <= -5:
        return "Rückgang"
    return "Stabil"


def _build_bubble(kunden: list[dict]) -> str:
    """Bubble-Chart: Umsatz (x) vs. Portfolio-Abdeckung (y), Größe = Potenzial."""
    if not kunden:
        return json.dumps({"data": [], "layout": {}})

    max_pot = max((k["potenzial_eur"] for k in kunden), default=1) or 1
    traces = []

    for seg in ["Wachstum", "Stabil", "Rückgang"]:
        grp = [k for k in kunden if k["segment"] == seg]
        if not grp:
            continue
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "name": seg,
            "x": [k["umsatz_aktuell"] for k in grp],
            "y": [k["abdeckung_pct"] for k in grp],
            "text": [k["name"] for k in grp],
            "hovertext": [
                f"<b>{k['name']}</b><br>"
                f"Umsatz: {k['umsatz_aktuell']:,.0f} €<br>"
                f"Abdeckung: {k['abdeckung_pct']:.0f} %<br>"
                f"Potenzial: {k['potenzial_eur']:,.0f} €<br>"
                f"Trend: {k['wachstum_pct']:+.1f} %"
                for k in grp
            ],
            "hoverinfo": "text",
            "marker": {
                "size": [18 + 46 * (k["potenzial_eur"] / max_pot) for k in grp],
                "color": SEGMENT_FARBE[seg],
                "opacity": 0.72,
                "line": {"color": "#0f172a", "width": 1.5},
            },
        })

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(20,30,50,0.4)",
        "margin": {"l": 60, "r": 20, "t": 20, "b": 50},
        "xaxis": {"title": "Aktueller Jahresumsatz (€)",
                  "gridcolor": "rgba(148,163,184,0.1)", "color": "#94a3b8"},
        "yaxis": {"title": "Portfolio-Abdeckung (%)",
                  "gridcolor": "rgba(148,163,184,0.1)", "color": "#94a3b8",
                  "range": [0, 105]},
        "font": {"family": "Inter, sans-serif", "size": 11, "color": "#94a3b8"},
        "legend": {"orientation": "h", "y": -0.18, "bgcolor": "rgba(0,0,0,0)"},
        "hovermode": "closest",
    }
    return json.dumps({"data": traces, "layout": layout}, default=float)


def analyse(df: pd.DataFrame, selected_kunde_id: str | None = None) -> GrowthResult:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    result = GrowthResult()

    jahre = sorted(df["jahr"].unique())
    jahr_akt = jahre[-1]
    jahr_vor = jahre[-2] if len(jahre) > 1 else jahre[-1]

    alle_produkte = sorted(df["produkt_id"].unique())
    prod_meta = {
        pid: {
            "name":   grp["produkt_name"].iloc[0],
            "gruppe": grp["produktgruppe"].iloc[0],
        }
        for pid, grp in df.groupby("produkt_id")
    }

    # ── Umsatz je Kunde/Jahr ──────────────────────────────────────────────
    ku_jahr = df.groupby(["kunde_id", "jahr"])["umsatz"].sum().unstack(fill_value=0)
    # Umsatz je Kunde/Produkt (aktuelles Jahr)
    df_akt = df[df["jahr"] == jahr_akt]
    kp_akt = df_akt.groupby(["kunde_id", "produkt_id"])["umsatz"].sum().unstack(fill_value=0)

    # ── Kundenstammdaten ──────────────────────────────────────────────────
    stamm = (df.groupby("kunde_id")
               .agg(name=("kunde_name", "first"),
                    branche=("branche", "first"),
                    rep=("vertriebsmitarbeiter", "first"),
                    groesse=("groesse_klasse", "first"))
               .reset_index())

    # ── Peer-Group-Definition ─────────────────────────────────────────────
    def peers_of(kid: str) -> list[str]:
        row = stamm[stamm["kunde_id"] == kid].iloc[0]
        same = stamm[(stamm["branche"] == row["branche"]) & (stamm["kunde_id"] != kid)]
        if len(same) < 2:  # Fallback: gleiche Größenklasse
            same = stamm[(stamm["groesse"] == row["groesse"]) & (stamm["kunde_id"] != kid)]
        return same["kunde_id"].tolist()

    # Größen-Skalierung für Potenzialrechnung
    umsatz_akt_map = {kid: float(ku_jahr.loc[kid, jahr_akt]) for kid in ku_jahr.index}

    kunden_out, luecken_detail = [], {}

    for _, s in stamm.iterrows():
        kid = s["kunde_id"]
        u_akt = umsatz_akt_map.get(kid, 0.0)
        u_vor = float(ku_jahr.loc[kid, jahr_vor]) if jahr_vor in ku_jahr.columns else u_akt
        wachstum = ((u_akt - u_vor) / u_vor * 100) if u_vor > 0 else 0.0

        gekauft = set(kp_akt.columns[kp_akt.loc[kid] > 0]) if kid in kp_akt.index else set()
        abdeckung = len(gekauft) / len(alle_produkte) * 100

        peer_ids = peers_of(kid)
        peer_umsatz_gesamt = sum(umsatz_akt_map.get(p, 0) for p in peer_ids) or 1

        # ── Produktlücken bestimmen ───────────────────────────────────────
        luecken = []
        for pid in alle_produkte:
            if pid in gekauft:
                continue
            # Wie viele Peers kaufen dieses Produkt?
            peers_mit = [p for p in peer_ids
                         if p in kp_akt.index and kp_akt.loc[p, pid] > 0]
            if not peer_ids:
                continue
            peer_anteil = len(peers_mit) / len(peer_ids)
            if peer_anteil < PEER_SCHWELLE:
                continue

            # Peer-Median-Umsatz mit diesem Produkt, normiert auf Peer-Gesamtumsatz
            peer_werte = [float(kp_akt.loc[p, pid]) for p in peers_mit]
            peer_median = sorted(peer_werte)[len(peer_werte) // 2]
            # Skalierung: Verhältnis Kundenumsatz zu Peer-Durchschnittsumsatz
            peer_avg_umsatz = peer_umsatz_gesamt / len(peer_ids)
            skalierung = (u_akt / peer_avg_umsatz) if peer_avg_umsatz > 0 else 1.0
            skalierung = max(0.35, min(2.5, skalierung))

            potenzial = peer_median * skalierung * REALISIERUNGSQUOTE

            luecken.append({
                "produkt_id":   pid,
                "produkt_name": prod_meta[pid]["name"],
                "gruppe":       prod_meta[pid]["gruppe"],
                "peer_anteil":  round(peer_anteil * 100),
                "peer_anzahl":  len(peers_mit),
                "peer_namen":   [
                    stamm[stamm["kunde_id"] == p]["name"].iloc[0] for p in peers_mit[:3]
                ],
                "potenzial":    round(potenzial, -2),
            })

        luecken.sort(key=lambda x: -x["potenzial"])
        luecken_detail[kid] = luecken
        potenzial_gesamt = sum(l["potenzial"] for l in luecken)

        kunden_out.append({
            "id":              kid,
            "name":            s["name"],
            "branche":         s["branche"],
            "rep":             s["rep"],
            "groesse":         s["groesse"],
            "umsatz_aktuell":  round(u_akt, 0),
            "umsatz_vorjahr":  round(u_vor, 0),
            "wachstum_pct":    round(wachstum, 1),
            "segment":         _segment(wachstum),
            "abdeckung_pct":   round(abdeckung, 0),
            "n_produkte":      len(gekauft),
            "n_luecken":       len(luecken),
            "potenzial_eur":   round(potenzial_gesamt, 0),
            "top_luecke":      luecken[0]["produkt_name"] if luecken else "—",
        })

    kunden_out.sort(key=lambda x: -x["potenzial_eur"])
    result.kunden = kunden_out

    # ── KPIs ──────────────────────────────────────────────────────────────
    gesamt_potenzial = sum(k["potenzial_eur"] for k in kunden_out)
    gesamt_umsatz    = sum(k["umsatz_aktuell"] for k in kunden_out)
    result.kpi = {
        "n_kunden":         len(kunden_out),
        "gesamt_umsatz":    round(gesamt_umsatz, 0),
        "gesamt_potenzial": round(gesamt_potenzial, 0),
        "potenzial_pct":    round(gesamt_potenzial / gesamt_umsatz * 100, 1) if gesamt_umsatz else 0,
        "avg_abdeckung":    round(sum(k["abdeckung_pct"] for k in kunden_out) / len(kunden_out), 0) if kunden_out else 0,
        "n_wachstum":       sum(1 for k in kunden_out if k["segment"] == "Wachstum"),
        "n_rueckgang":      sum(1 for k in kunden_out if k["segment"] == "Rückgang"),
        "jahr_aktuell":     int(jahr_akt),
        "jahr_vorjahr":     int(jahr_vor),
    }

    # ── Potenzial je Produkt (aggregiert über alle Kunden) ───────────────
    prod_pot: dict[str, dict] = {}
    for kid, luecken in luecken_detail.items():
        for l in luecken:
            e = prod_pot.setdefault(l["produkt_id"], {
                "produkt_name": l["produkt_name"],
                "gruppe":       l["gruppe"],
                "potenzial":    0.0,
                "n_kunden":     0,
            })
            e["potenzial"] += l["potenzial"]
            e["n_kunden"]  += 1
    result.produkt_potenzial = sorted(
        [{"produkt_id": k, **v, "potenzial": round(v["potenzial"], 0)}
         for k, v in prod_pot.items()],
        key=lambda x: -x["potenzial"]
    )

    result.bubble_json = _build_bubble(kunden_out)

    # ── Kundenspezifische Detailansicht ──────────────────────────────────
    if selected_kunde_id:
        k_info = next((k for k in kunden_out if k["id"] == selected_kunde_id), None)
        if k_info:
            result.selected_kunde   = k_info
            result.selected_luecken = luecken_detail.get(selected_kunde_id, [])[:5]

            # Aktuelles Portfolio
            if selected_kunde_id in kp_akt.index:
                port = []
                for pid in alle_produkte:
                    val = float(kp_akt.loc[selected_kunde_id, pid])
                    if val > 0:
                        port.append({
                            "produkt_id":   pid,
                            "produkt_name": prod_meta[pid]["name"],
                            "gruppe":       prod_meta[pid]["gruppe"],
                            "umsatz":       round(val, 0),
                        })
                port.sort(key=lambda x: -x["umsatz"])
                result.selected_portfolio = port

            # Umsatztrend über alle Jahre
            result.selected_trend = [
                {"jahr": int(j), "umsatz": round(float(ku_jahr.loc[selected_kunde_id, j]), 0)}
                for j in jahre if selected_kunde_id in ku_jahr.index
            ]

    return result
