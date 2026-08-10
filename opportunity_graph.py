"""
Graph Opportunity Intelligence – Deal Intelligence über einen heterogenen Graphen.

Kernidee: Ein CRM sieht nur "Opportunity, 250.000 €, Stage 3, 60 %".
Ein B2B-Deal ist aber ein Netzwerk aus Buying Center, Wettbewerbern,
Produkten und vergangenen Deals. Genau dort liegen die Risikosignale.

Heterogener Graph (NetworkX):
  Knotentypen : Opportunity, Kunde, Kontakt, Produkt, Wettbewerber, Vertriebsmitarbeiter
  Kantentypen : GEHOERT_ZU, ARBEITET_FUER, INVOLVIERT, BETRIFFT,
                KONKURRIERT_UM, BETREUT, AEHNLICH_ZU

Abgeleitete Vorhersagen (Heuristik auf Graph-Features, GNN als Next Step):
  P(Win)            – graph-adjustierte Gewinnwahrscheinlichkeit
  P(Stall)          – Stillstandsrisiko
  P(Competitor Win) – Wettbewerbsrisiko
  P(Discount)       – Rabattdruck
  Expected Value    – Deal-Wert × P(Win)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import pandas as pd
import networkx as nx

# Rollen, die ein vollständiges Buying Center ausmachen
KRITISCHE_ROLLEN = ["Geschäftsführung", "CFO / Einkaufsleitung", "Einkauf", "Technik"]
# Ab diesem Deal-Wert ist die Finanzfreigabe zwingend
CFO_SCHWELLE = 150_000

NODE_FARBE = {
    "Opportunity":  "#a78bfa",
    "Kunde":        "#60a5fa",
    "Kontakt":      "#34d399",
    "Produkt":      "#fbbf24",
    "Wettbewerber": "#f87171",
    "Vertrieb":     "#22d3ee",
}


@dataclass
class OppResult:
    kpi: dict                      = field(default_factory=dict)
    deals: list[dict]              = field(default_factory=list)
    graph_json: str                = "{}"
    selected_deal: dict | None     = None
    selected_kontakte: list[dict]  = field(default_factory=list)
    selected_aehnliche: list[dict] = field(default_factory=list)
    benchmark: dict                = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
# Graph-Aufbau
# ══════════════════════════════════════════════════════════════════════════

def _build_hetero_graph(df_opp: pd.DataFrame, df_kon: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for _, o in df_opp.iterrows():
        oid = o["opp_id"]
        G.add_node(oid, typ="Opportunity", label=oid,
                   wert=float(o["deal_wert"]), stage=o["stage"],
                   status=o["status"])

        kid = f"KD_{o['kunde_id']}"
        G.add_node(kid, typ="Kunde", label=o["kunde_name"], branche=o["branche"])
        G.add_edge(oid, kid, rel="GEHOERT_ZU")

        pid = f"PR_{o['produkt_id']}"
        G.add_node(pid, typ="Produkt", label=o["produkt_name"])
        G.add_edge(oid, pid, rel="BETRIFFT")

        rep = f"VM_{o['vertriebsmitarbeiter']}"
        G.add_node(rep, typ="Vertrieb", label=o["vertriebsmitarbeiter"])
        G.add_edge(oid, rep, rel="BETREUT")

        wb = str(o.get("wettbewerber", "—"))
        if wb and wb != "—":
            wid = f"WB_{wb}"
            G.add_node(wid, typ="Wettbewerber", label=wb)
            G.add_edge(oid, wid, rel="KONKURRIERT_UM")

    for _, k in df_kon.iterrows():
        if not int(k.get("involviert", 0)):
            continue
        oid = k["opp_id"]
        if oid not in G:
            continue
        cid = f"KT_{oid}_{k['rolle']}"
        G.add_node(cid, typ="Kontakt", label=k["kontakt_name"],
                   rolle=k["rolle"], sentiment=k.get("sentiment", "neutral"),
                   wb_kontakt=int(k.get("wettbewerber_kontakt", 0)))
        G.add_edge(oid, cid, rel="INVOLVIERT")

    return G


# ══════════════════════════════════════════════════════════════════════════
# Risiko-Features aus dem Graphen
# ══════════════════════════════════════════════════════════════════════════

def _risiko_faktoren(o: pd.Series,
                     kontakte: pd.DataFrame,
                     median_stage_tage: float,
                     hist_winrate_branche: float,
                     hist_winrate_gesamt: float) -> list[dict]:
    """Gibt eine Liste von Risiko-/Chancen-Faktoren mit Wahrscheinlichkeits-Delta zurück."""
    f = []
    wert = float(o["deal_wert"])
    involviert = kontakte[kontakte["involviert"] == 1]
    rollen_aktiv = set(involviert["rolle"].tolist())

    # ── 1. Buying-Center-Abdeckung ────────────────────────────────────────
    if "CFO / Einkaufsleitung" not in rollen_aktiv and wert >= CFO_SCHWELLE:
        f.append({
            "typ": "risiko", "delta": -14,
            "titel": "Finanzfreigabe nicht eingebunden",
            "text": f"Bei {wert:,.0f} € Deal-Wert ist keine CFO-/Einkaufsleitung im Buying Center aktiv. "
                    f"Historisch scheitern Deals dieser Größe ohne Finanzfreigabe deutlich häufiger.",
        })
    elif "CFO / Einkaufsleitung" in rollen_aktiv:
        f.append({
            "typ": "chance", "delta": +8,
            "titel": "Finanzfreigabe eingebunden",
            "text": "CFO / Einkaufsleitung ist aktiv im Entscheidungsprozess.",
        })

    if "Geschäftsführung" not in rollen_aktiv and wert >= 250_000:
        f.append({
            "typ": "risiko", "delta": -7,
            "titel": "Kein Zugang zur Geschäftsführung",
            "text": "Deals über 250.000 € werden in der Regel auf Geschäftsführungsebene entschieden.",
        })

    if "Technik" not in rollen_aktiv:
        f.append({
            "typ": "risiko", "delta": -6,
            "titel": "Technische Entscheidung offen",
            "text": "Kein technischer Ansprechpartner involviert – die Lösungsvalidierung fehlt.",
        })

    # ── 2. Wettbewerberkontakt im Buying Center ──────────────────────────
    wb_kontakte = involviert[involviert["wettbewerber_kontakt"] == 1]
    if len(wb_kontakte) > 0:
        rollen = ", ".join(wb_kontakte["rolle"].tolist())
        f.append({
            "typ": "risiko", "delta": -16,
            "titel": "Wettbewerber im Buying Center aktiv",
            "text": f"Direkter Wettbewerberkontakt über: {rollen}. "
                    f"Wettbewerber: {o.get('wettbewerber', '—')}.",
        })
    elif str(o.get("wettbewerber", "—")) != "—":
        f.append({
            "typ": "risiko", "delta": -6,
            "titel": "Wettbewerber im Rennen",
            "text": f"{o['wettbewerber']} bietet ebenfalls an, hat aber keinen erkennbaren "
                    f"Draht ins Buying Center.",
        })

    # ── 3. Stillstand ─────────────────────────────────────────────────────
    tage_stage = float(o.get("tage_in_stage", 0))
    if tage_stage > median_stage_tage * 1.8:
        f.append({
            "typ": "risiko", "delta": -12,
            "titel": "Deal steht still",
            "text": f"{tage_stage:.0f} Tage in Stage „{o['stage']}“ – "
                    f"Median liegt bei {median_stage_tage:.0f} Tagen.",
        })

    # ── 4. Kontaktabstand ─────────────────────────────────────────────────
    kontakt_tage = float(o.get("letzter_kontakt_tage", 0))
    if kontakt_tage > 30:
        f.append({
            "typ": "risiko", "delta": -10,
            "titel": "Kontakt eingeschlafen",
            "text": f"Letzter Kontakt vor {kontakt_tage:.0f} Tagen. "
                    f"Ab 30 Tagen sinkt die Abschlussquote messbar.",
        })
    elif kontakt_tage <= 7:
        f.append({
            "typ": "chance", "delta": +5,
            "titel": "Enger Kundenkontakt",
            "text": f"Letzter Kontakt vor {kontakt_tage:.0f} Tagen – der Deal ist aktiv in Bewegung.",
        })

    # ── 5. Rabattdruck ────────────────────────────────────────────────────
    rabatt = float(o.get("rabatt_gefordert_pct", 0))
    if rabatt >= 18:
        f.append({
            "typ": "risiko", "delta": -8,
            "titel": "Hoher Rabattdruck",
            "text": f"{rabatt:.0f} % Nachlass gefordert – Indiz für reinen Preiswettbewerb "
                    f"statt Wertargumentation.",
        })

    # ── 6. Negatives Sentiment ────────────────────────────────────────────
    neg = involviert[involviert["sentiment"] == "negativ"]
    if len(neg) > 0:
        f.append({
            "typ": "risiko", "delta": -9,
            "titel": "Widerstand im Buying Center",
            "text": f"Negatives Sentiment bei: {', '.join(neg['rolle'].tolist())}.",
        })

    # ── 7. Branchen-Referenz ──────────────────────────────────────────────
    diff = (hist_winrate_branche - hist_winrate_gesamt) * 100
    if abs(diff) >= 8:
        f.append({
            "typ": "chance" if diff > 0 else "risiko",
            "delta": round(diff * 0.4),
            "titel": f"Branchen-Historie {o['branche']}",
            "text": f"Win-Rate in dieser Branche: {hist_winrate_branche*100:.0f} % "
                    f"(Gesamt: {hist_winrate_gesamt*100:.0f} %).",
        })

    return f


def _next_action(deal: dict, faktoren: list[dict]) -> dict:
    """Leitet die konkret empfohlene nächste Handlung aus dem stärksten Risiko ab."""
    risiken = sorted([f for f in faktoren if f["typ"] == "risiko"], key=lambda x: x["delta"])
    if not risiken:
        return {
            "aktion": "Abschluss forcieren",
            "detail": "Keine wesentlichen Risikosignale. Verbindliches Abschlussdatum vereinbaren.",
            "frist":  "diese Woche",
        }

    top = risiken[0]["titel"]
    mapping = {
        "Finanzfreigabe nicht eingebunden": (
            "CFO-Termin vereinbaren",
            "Business Case mit TCO-Rechnung aufbereiten und Finanzentscheider direkt einbinden.",
            "innerhalb 48 h"),
        "Wettbewerber im Buying Center aktiv": (
            "Wettbewerbs-Battlecard einsetzen",
            "Differenzierung gegenüber dem Wettbewerber schriftlich adressieren und "
            "technische Entscheider erneut überzeugen.",
            "innerhalb 48 h"),
        "Deal steht still": (
            "Deal reaktivieren",
            "Verbindlichen nächsten Schritt mit Datum vereinbaren – sonst Stage zurückstufen.",
            "innerhalb 3 Tagen"),
        "Kontakt eingeschlafen": (
            "Kontakt reaktivieren",
            "Persönlichen Termin vereinbaren und Entscheidungsstand verifizieren.",
            "innerhalb 48 h"),
        "Kein Zugang zur Geschäftsführung": (
            "Zugang zur Geschäftsleitung schaffen",
            "Über bestehenden Sponsor eine Vorstellung auf Geschäftsführungsebene erwirken.",
            "diese Woche"),
        "Technische Entscheidung offen": (
            "Technische Validierung ansetzen",
            "Proof of Concept oder Referenzbesuch mit dem technischen Entscheider terminieren.",
            "diese Woche"),
        "Hoher Rabattdruck": (
            "Wertargumentation stärken",
            "Statt Preisnachlass Zusatzleistungen anbieten (Service, Schulung, Verfügbarkeit).",
            "vor nächstem Angebot"),
        "Widerstand im Buying Center": (
            "Einwände strukturiert adressieren",
            "Einzelgespräch mit dem kritischen Stakeholder führen und Bedenken dokumentieren.",
            "innerhalb 1 Woche"),
    }
    aktion, detail, frist = mapping.get(
        top, ("Deal-Review durchführen", "Risikosignale mit dem Vertriebsleiter besprechen.", "diese Woche"))
    return {"aktion": aktion, "detail": detail, "frist": frist}


# ══════════════════════════════════════════════════════════════════════════
# Graph-Visualisierung für einen Deal
# ══════════════════════════════════════════════════════════════════════════

def _build_deal_graph(G: nx.Graph, opp_id: str) -> str:
    if opp_id not in G:
        return json.dumps({"data": [], "layout": {}})

    # Ego-Graph: Deal + direkte Nachbarn
    sub = nx.ego_graph(G, opp_id, radius=1)
    pos = nx.spring_layout(sub, seed=11, k=1.6)
    pos[opp_id] = [0.0, 0.0]  # Deal ins Zentrum

    edge_traces = []
    for u, v, d in sub.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append({
            "type": "scatter", "mode": "lines",
            "x": [x0, x1, None], "y": [y0, y1, None],
            "line": {"width": 1.4, "color": "rgba(148,163,184,0.35)"},
            "hoverinfo": "none", "showlegend": False,
        })

    by_typ: dict[str, dict] = {}
    for nid in sub.nodes():
        d = sub.nodes[nid]
        typ = d.get("typ", "?")
        e = by_typ.setdefault(typ, {"x": [], "y": [], "text": [], "hover": [], "size": []})
        x, y = pos[nid]
        e["x"].append(x); e["y"].append(y)
        e["text"].append(d.get("label", nid))

        if typ == "Opportunity":
            hover = f"<b>{d.get('label')}</b><br>Deal: {d.get('wert', 0):,.0f} €<br>Stage: {d.get('stage','–')}"
            size = 34
        elif typ == "Kontakt":
            sent = d.get("sentiment", "–")
            wbk = " · Wettbewerberkontakt" if d.get("wb_kontakt") else ""
            hover = f"<b>{d.get('label')}</b><br>{d.get('rolle','–')}<br>Sentiment: {sent}{wbk}"
            size = 20
        else:
            hover = f"<b>{d.get('label')}</b><br>{typ}"
            size = 22
        e["hover"].append(hover)
        e["size"].append(size)

    node_traces = []
    for typ, e in by_typ.items():
        node_traces.append({
            "type": "scatter", "mode": "markers+text",
            "name": typ,
            "x": e["x"], "y": e["y"],
            "text": e["text"], "textposition": "bottom center",
            "textfont": {"size": 9, "color": "#cbd5e1"},
            "hovertext": e["hover"], "hoverinfo": "text",
            "marker": {
                "size": e["size"],
                "color": NODE_FARBE.get(typ, "#94a3b8"),
                "line": {"color": "#0f172a", "width": 2},
                "opacity": 0.92,
            },
        })

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "margin": {"l": 0, "r": 0, "t": 10, "b": 30},
        "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "font": {"family": "Inter, sans-serif", "size": 11, "color": "#94a3b8"},
        "legend": {"orientation": "h", "y": -0.02, "font": {"size": 10},
                   "bgcolor": "rgba(0,0,0,0)"},
        "hovermode": "closest",
    }
    return json.dumps({"data": edge_traces + node_traces, "layout": layout}, default=float)


# ══════════════════════════════════════════════════════════════════════════
# Hauptanalyse
# ══════════════════════════════════════════════════════════════════════════

def analyse(df_opp: pd.DataFrame,
            df_kon: pd.DataFrame,
            selected_opp_id: str | None = None) -> OppResult:

    df_opp = df_opp.copy(); df_opp.columns = [c.lower().strip() for c in df_opp.columns]
    df_kon = df_kon.copy(); df_kon.columns = [c.lower().strip() for c in df_kon.columns]
    result = OppResult()

    G = _build_hetero_graph(df_opp, df_kon)

    offen  = df_opp[df_opp["status"] == "offen"].copy()
    hist   = df_opp[df_opp["status"].isin(["gewonnen", "verloren"])].copy()

    # ── Historische Referenzwerte ─────────────────────────────────────────
    if len(hist) > 0:
        hist["won"] = (hist["status"] == "gewonnen").astype(int)
        winrate_gesamt = float(hist["won"].mean())
        winrate_branche = hist.groupby("branche")["won"].mean().to_dict()
    else:
        winrate_gesamt, winrate_branche = 0.45, {}

    median_stage_tage = float(offen["tage_in_stage"].median()) if len(offen) else 30.0

    # ── Deals bewerten ────────────────────────────────────────────────────
    deals = []
    for _, o in offen.iterrows():
        oid = o["opp_id"]
        kon = df_kon[df_kon["opp_id"] == oid]
        wb_branche = winrate_branche.get(o["branche"], winrate_gesamt)

        faktoren = _risiko_faktoren(o, kon, median_stage_tage, wb_branche, winrate_gesamt)

        crm_prob = float(o["crm_probability"])
        delta = sum(f["delta"] for f in faktoren)
        graph_prob = max(3.0, min(96.0, crm_prob + delta))

        wert = float(o["deal_wert"])
        rabatt = float(o.get("rabatt_gefordert_pct", 0))
        tage_stage = float(o.get("tage_in_stage", 0))
        kontakt_tage = float(o.get("letzter_kontakt_tage", 0))
        hat_wb = str(o.get("wettbewerber", "—")) != "—"
        wb_im_bc = len(kon[(kon["involviert"] == 1) & (kon["wettbewerber_kontakt"] == 1)]) > 0

        # Zusatzwahrscheinlichkeiten
        p_stall = min(95, max(4,
            18 + (tage_stage / max(median_stage_tage, 1) - 1) * 32 + kontakt_tage * 0.55))
        p_comp = min(92, max(2,
            (34 if hat_wb else 5) + (26 if wb_im_bc else 0) + (rabatt - 10) * 0.9))
        p_disc = min(95, max(5,
            rabatt * 2.6 + (16 if hat_wb else 0) + (10 if wb_im_bc else 0)))

        risiken = [f for f in faktoren if f["typ"] == "risiko"]
        if graph_prob < crm_prob - 12:
            ampel, ampel_text = "rot", "Deutlich riskanter als im CRM"
        elif graph_prob < crm_prob - 4:
            ampel, ampel_text = "gelb", "Risiken erkannt"
        else:
            ampel, ampel_text = "grün", "Auf Kurs"

        d = {
            "opp_id":        oid,
            "kunde_id":      o["kunde_id"],
            "kunde_name":    o["kunde_name"],
            "branche":       o["branche"],
            "rep":           o["vertriebsmitarbeiter"],
            "produkt_name":  o["produkt_name"],
            "deal_wert":     round(wert, 0),
            "stage":         o["stage"],
            "stage_nr":      int(o["stage_nr"]),
            "tage_in_stage": int(tage_stage),
            "letzter_kontakt_tage": int(kontakt_tage),
            "wettbewerber":  o.get("wettbewerber", "—"),
            "rabatt_pct":    round(rabatt, 0),
            "crm_prob":      round(crm_prob, 0),
            "graph_prob":    round(graph_prob, 0),
            "delta":         round(graph_prob - crm_prob, 0),
            "p_stall":       round(p_stall, 0),
            "p_competitor":  round(p_comp, 0),
            "p_discount":    round(p_disc, 0),
            "expected_value": round(wert * graph_prob / 100, 0),
            "crm_value":     round(wert * crm_prob / 100, 0),
            "faktoren":      faktoren,
            "n_risiken":     len(risiken),
            "ampel":         ampel,
            "ampel_text":    ampel_text,
            "buying_center": int(kon["involviert"].sum()) if len(kon) else 0,
        }
        d["next_action"] = _next_action(d, faktoren)
        deals.append(d)

    deals.sort(key=lambda x: -x["expected_value"])
    result.deals = deals

    # ── KPIs ──────────────────────────────────────────────────────────────
    pipeline    = sum(d["deal_wert"] for d in deals)
    gewichtet   = sum(d["expected_value"] for d in deals)
    crm_gewicht = sum(d["crm_value"] for d in deals)
    at_risk     = sum(d["deal_wert"] for d in deals if d["ampel"] == "rot")

    result.kpi = {
        "n_deals":        len(deals),
        "pipeline":       round(pipeline, 0),
        "gewichtet":      round(gewichtet, 0),
        "crm_gewichtet":  round(crm_gewicht, 0),
        "korrektur":      round(gewichtet - crm_gewicht, 0),
        "at_risk":        round(at_risk, 0),
        "n_rot":          sum(1 for d in deals if d["ampel"] == "rot"),
        "n_gelb":         sum(1 for d in deals if d["ampel"] == "gelb"),
        "n_gruen":        sum(1 for d in deals if d["ampel"] == "grün"),
        "winrate_hist":   round(winrate_gesamt * 100, 1),
        "n_historisch":   len(hist),
        "n_knoten":       G.number_of_nodes(),
        "n_kanten":       G.number_of_edges(),
    }
    result.benchmark = {
        "winrate_branche": {k: round(v * 100, 1) for k, v in winrate_branche.items()},
        "median_stage_tage": round(median_stage_tage, 0),
    }

    # ── Deal-Detailansicht ────────────────────────────────────────────────
    if selected_opp_id:
        d = next((x for x in deals if x["opp_id"] == selected_opp_id), None)
        if d:
            result.selected_deal = d
            kon = df_kon[df_kon["opp_id"] == selected_opp_id]
            result.selected_kontakte = [
                {
                    "rolle":       r["rolle"],
                    "name":        r["kontakt_name"],
                    "involviert":  int(r["involviert"]),
                    "sentiment":   r["sentiment"],
                    "wb_kontakt":  int(r["wettbewerber_kontakt"]),
                    "kritisch":    r["rolle"] in KRITISCHE_ROLLEN,
                }
                for _, r in kon.iterrows()
            ]
            # Ähnliche historische Deals (gleiche Branche, ähnlicher Wert)
            if len(hist) > 0:
                wert = d["deal_wert"]
                aehn = hist[hist["branche"] == d["branche"]].copy()
                if len(aehn) == 0:
                    aehn = hist.copy()
                aehn["wert_diff"] = (aehn["deal_wert"] - wert).abs()
                aehn = aehn.nsmallest(5, "wert_diff")
                result.selected_aehnliche = [
                    {
                        "opp_id":     r["opp_id"],
                        "kunde_name": r["kunde_name"],
                        "deal_wert":  round(float(r["deal_wert"]), 0),
                        "status":     r["status"],
                        "wettbewerber": r.get("wettbewerber", "—"),
                        "rabatt_pct": round(float(r.get("rabatt_gefordert_pct", 0)), 0),
                    }
                    for _, r in aehn.iterrows()
                ]
            result.graph_json = _build_deal_graph(G, selected_opp_id)

    return result
