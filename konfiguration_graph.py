"""
Graph-basierte Produktkonfiguration mit Margen-Optimierung.

Ansatz:
  1. Kompatibilitätsgraph mit NetworkX (Knoten = Komponenten, Kanten = kompatibel_mit)
  2. Gegeben Basiskomponente → alle validen Configs (Basis + Antrieb + Steuerung + Gehäuse)
  3. Für jede Config: Marge berechnen, historische Gewinnwahrscheinlichkeit ermitteln
  4. Ranking nach Score = Marge% × Gewinnwahrscheinlichkeit
  5. Top-5 zurückgeben
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json, itertools
import pandas as pd
import networkx as nx

KAT_FARBE = {
    "Basis":     "#60a5fa",
    "Antrieb":   "#f59e0b",
    "Steuerung": "#34d399",
    "Gehäuse":   "#a78bfa",
    "Zubehör":   "#f87171",
}

KAT_REIHENFOLGE = ["Basis", "Antrieb", "Steuerung", "Gehäuse", "Zubehör"]


@dataclass
class KonfigResult:
    kpi: dict                        = field(default_factory=dict)
    basis_komponenten: list[dict]    = field(default_factory=list)
    top_configs: list[dict]          = field(default_factory=list)
    alle_configs: list[dict]         = field(default_factory=list)  # für Scatter (alle validen)
    scatter_json: str                = "{}"
    selected_basis: dict | None      = None


def _build_graph(df_sl: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in df_sl.iterrows():
        G.add_node(row["komponente_id"],
                   name=row["komponente_name"],
                   kat=row["kategorie"],
                   preis=float(row["listenpreis"]),
                   kosten=float(row["herstellkosten"]))
    for _, row in df_sl.iterrows():
        compat = str(row.get("kompatibel_mit", "") or "")
        for other_id in [s.strip() for s in compat.split(",") if s.strip()]:
            if G.has_node(other_id):
                G.add_edge(row["komponente_id"], other_id)
    return G


def _calc_win_prob(config_ids: list[str], df_ang: pd.DataFrame) -> float:
    """Historische Gewinnwahrscheinlichkeit: Angebote mit ≥2 überlappenden Komponenten."""
    config_set = set(config_ids)
    mask = df_ang["konfiguration"].apply(
        lambda x: len(config_set & set(str(x).split(","))) >= 2
    )
    matching = df_ang[mask]
    if len(matching) == 0:
        return 0.45  # Fallback-Durchschnitt
    return round(float(matching["gewonnen"].mean()), 3)


def _build_scatter(configs: list[dict]) -> str:
    if not configs:
        return json.dumps({"data": [], "layout": {}})

    # Rank-Farben
    rank_colors = ["#fbbf24", "#94a3b8", "#b45309", "#60a5fa", "#34d399"]

    traces = []
    for i, c in enumerate(configs):
        is_top5 = i < 5
        color = rank_colors[i] if i < 5 else "#334155"
        size  = 18 if is_top5 else 10
        opacity = 1.0 if is_top5 else 0.5
        label = f"#{i+1} {c['label']}" if is_top5 else ""
        traces.append({
            "type": "scatter",
            "mode": "markers+text" if is_top5 else "markers",
            "x": [round(c["win_prob"] * 100, 1)],
            "y": [round(c["marge_pct"], 1)],
            "text": [label],
            "textposition": "top center",
            "textfont": {"size": 9, "color": "#cbd5e1"},
            "hovertext": [
                f"<b>{c['label']}</b><br>"
                f"Marge: {c['marge_pct']:.1f}%<br>"
                f"Gewinnchance: {c['win_prob']*100:.0f}%<br>"
                f"Score: {c['score']:.2f}<br>"
                f"Preis: {c['gesamtpreis']:,.0f} €"
            ],
            "hoverinfo": "text",
            "marker": {"size": size, "color": color, "opacity": opacity,
                       "line": {"color": "#1e293b", "width": 1}},
            "showlegend": False,
        })

    # Quadrant-Linien
    avg_wp = sum(c["win_prob"] for c in configs) / len(configs) * 100
    avg_mg = sum(c["marge_pct"] for c in configs) / len(configs)

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(20,30,50,0.4)",
        "margin": {"l": 50, "r": 20, "t": 30, "b": 50},
        "xaxis": {
            "title": "Gewinnwahrscheinlichkeit (%)",
            "gridcolor": "rgba(148,163,184,0.1)",
            "color": "#94a3b8",
            "range": [0, 105],
        },
        "yaxis": {
            "title": "Deckungsbeitrag (%)",
            "gridcolor": "rgba(148,163,184,0.1)",
            "color": "#94a3b8",
        },
        "font": {"family": "Inter, sans-serif", "size": 11, "color": "#94a3b8"},
        "shapes": [
            {"type": "line", "x0": avg_wp, "x1": avg_wp, "y0": 0, "y1": 1,
             "xref": "x", "yref": "paper",
             "line": {"color": "rgba(148,163,184,0.3)", "dash": "dot"}},
            {"type": "line", "y0": avg_mg, "y1": avg_mg, "x0": 0, "x1": 1,
             "xref": "paper", "yref": "y",
             "line": {"color": "rgba(148,163,184,0.3)", "dash": "dot"}},
        ],
        "annotations": [
            {"x": 95, "y": max(c["marge_pct"] for c in configs),
             "text": "Optimal", "showarrow": False,
             "font": {"size": 9, "color": "#64748b"}, "xref": "x", "yref": "y"},
        ],
        "hovermode": "closest",
    }
    return json.dumps({"data": traces, "layout": layout}, default=float)


def analyse(df_sl: pd.DataFrame,
            df_ang: pd.DataFrame,
            selected_basis_id: str | None = None) -> KonfigResult:

    result = KonfigResult()

    # ── Graph aufbauen ────────────────────────────────────────────────────
    G = _build_graph(df_sl)

    # Komponenten-Index nach Kategorie
    by_kat: dict[str, list[str]] = {}
    for nid, data in G.nodes(data=True):
        by_kat.setdefault(data["kat"], []).append(nid)

    basis_list = [
        {
            "id":    nid,
            "name":  G.nodes[nid]["name"],
            "preis": G.nodes[nid]["preis"],
            "kat":   G.nodes[nid]["kat"],
        }
        for nid in sorted(by_kat.get("Basis", []))
    ]
    result.basis_komponenten = basis_list

    n_komponenten = G.number_of_nodes()
    n_configs_total = 0

    # ── KPI ───────────────────────────────────────────────────────────────
    gewinn_rate = float(df_ang["gewonnen"].mean()) if len(df_ang) > 0 else 0.0
    avg_marge   = float(df_ang["deckungsbeitrag_pct"].mean()) if len(df_ang) > 0 else 0.0
    result.kpi = {
        "n_komponenten":  n_komponenten,
        "n_angebote":     len(df_ang),
        "gewinn_rate":    round(gewinn_rate * 100, 1),
        "avg_marge":      round(avg_marge, 1),
    }

    if not selected_basis_id or not G.has_node(selected_basis_id):
        return result

    result.selected_basis = {
        "id":   selected_basis_id,
        "name": G.nodes[selected_basis_id]["name"],
    }

    # ── Valide Configs für gewählte Basis ────────────────────────────────
    # Nachbarn nach Kategorie gruppieren
    def neighbors_by_kat(node_id: str, kat: str) -> list[str]:
        return [n for n in G.neighbors(node_id) if G.nodes[n]["kat"] == kat]

    antriebe  = neighbors_by_kat(selected_basis_id, "Antrieb")
    alle_valide = []

    for antrieb_id in antriebe:
        steuerungen = neighbors_by_kat(antrieb_id, "Steuerung")
        for steuer_id in steuerungen:
            gehaeuse_list = neighbors_by_kat(steuer_id, "Gehäuse")
            for geh_id in gehaeuse_list:
                alle_valide.append((selected_basis_id, antrieb_id, steuer_id, geh_id))

    n_configs_total = len(alle_valide)
    result.kpi["n_valide_configs"] = n_configs_total

    # ── Bewerten ──────────────────────────────────────────────────────────
    scored = []
    for basis_id, a_id, s_id, g_id in alle_valide:
        ids = [basis_id, a_id, s_id, g_id]
        # Top 2 Zubehör: die teuersten kompatiblen (einfache Heuristik)
        zub_candidates = []
        for z_id in by_kat.get("Zubehör", []):
            if G.has_edge(basis_id, z_id) or any(G.has_edge(x, z_id) for x in [a_id, s_id]):
                zub_candidates.append(z_id)
        # nach Preis sortiert, top-2 nehmen
        zub_candidates.sort(key=lambda z: G.nodes[z]["preis"], reverse=True)
        top_zub = zub_candidates[:2]
        ids_with_zub = ids + top_zub

        gesamtpreis = sum(G.nodes[k]["preis"]  for k in ids_with_zub)
        gesamtkosten = sum(G.nodes[k]["kosten"] for k in ids_with_zub)
        marge_pct    = (gesamtpreis - gesamtkosten) / gesamtpreis * 100

        win_prob = _calc_win_prob(ids, df_ang)
        score    = marge_pct * win_prob

        label = (f"{G.nodes[a_id]['name'].split()[0]} + "
                 f"{G.nodes[s_id]['name'].split()[0]} + "
                 f"{G.nodes[g_id]['name'].split()[0]}")

        scored.append({
            "label":        label,
            "komponenten":  [
                {"id": k, "name": G.nodes[k]["name"], "kat": G.nodes[k]["kat"],
                 "preis": G.nodes[k]["preis"], "kosten": G.nodes[k]["kosten"]}
                for k in ids_with_zub
            ],
            "ids":          ids_with_zub,
            "gesamtpreis":  round(gesamtpreis, 2),
            "gesamtkosten": round(gesamtkosten, 2),
            "marge_pct":    round(marge_pct, 1),
            "win_prob":     win_prob,
            "score":        round(score, 2),
            "n_zub":        len(top_zub),
        })

    scored.sort(key=lambda x: -x["score"])

    result.alle_configs = scored
    result.top_configs  = scored[:5]
    result.scatter_json = _build_scatter(scored)

    return result
