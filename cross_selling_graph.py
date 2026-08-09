"""
Graph-basiertes Cross-Selling mit NetworkX.

Ansatz:
  1. Bipartiter Graph: Kunden ↔ Produkte
  2. Projektion auf Produkt-Produkt-Graph (Co-Occurrence)
  3. Lift-Score für statistisch signifikante Produktpaare
  4. Item-based Collaborative Filtering für Kunden-Empfehlungen
  5. Community Detection (Louvain-ähnlich via greedy modularity)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json, math
import pandas as pd
import networkx as nx

KAT_FARBE = {
    "Maschinen": "#60a5fa",   # blau
    "Zubehör":   "#34d399",   # grün
    "Service":   "#f59e0b",   # orange
}


@dataclass
class CrossSellingResult:
    kpi: dict                        = field(default_factory=dict)
    kunden: list[dict]               = field(default_factory=list)
    top_paare: list[dict]            = field(default_factory=list)
    alle_empfehlungen: list[dict]    = field(default_factory=list)
    graph_json: str                  = "{}"
    selected_kunde: dict | None      = None
    selected_empfehlungen: list[dict]= field(default_factory=list)
    selected_kaeufe: list[dict]      = field(default_factory=list)
    communities: list[list[str]]     = field(default_factory=list)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    cl = {c.lower(): c for c in df.columns}
    for canon, aliases in {
        "kunde_id":    ["kunde_id", "customer_id", "kunden_id"],
        "kunde_name":  ["kunde_name", "kundenname", "customer_name", "kunde"],
        "produkt_id":  ["produkt_id", "product_id"],
        "produkt_name":["produkt_name", "produktname", "product_name", "produkt"],
        "kategorie":   ["kategorie", "category", "kat"],
        "umsatz":      ["umsatz", "revenue", "betrag"],
        "menge":       ["menge", "quantity", "qty"],
    }.items():
        if canon in df.columns:
            continue
        for a in aliases:
            if a.lower() in cl:
                rename[cl[a.lower()]] = canon
                break
    if rename:
        df = df.rename(columns=rename)

    if "produkt_id" not in df.columns:
        df["produkt_id"] = df["produkt_name"].astype(str).str[:4].str.upper()
    if "kunde_id" not in df.columns:
        df["kunde_id"] = df["kunde_name"].astype(str).str[:3].str.upper()
    if "umsatz" not in df.columns:
        df["umsatz"] = 1.0
    if "kategorie" not in df.columns:
        df["kategorie"] = "Unbekannt"

    return df


def _build_graph_figure(G_prod: nx.Graph,
                         meta: dict,
                         highlight_ids: set[str] | None = None) -> str:
    """Baut Plotly-Figure für den Produkt-Graphen und gibt JSON zurück."""
    if len(G_prod) == 0:
        return json.dumps({"data": [], "layout": {}})

    pos = nx.spring_layout(G_prod, seed=42, k=2.5)

    # Kanten
    edge_traces = []
    max_w = max((d.get("weight", 1) for _, _, d in G_prod.edges(data=True)), default=1)
    for u, v, d in G_prod.edges(data=True):
        w = d.get("weight", 1)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        opacity = 0.25 + 0.55 * (w / max_w)
        width   = 1 + 4 * (w / max_w)
        edge_traces.append({
            "type": "scatter", "mode": "lines",
            "x": [x0, x1, None], "y": [y0, y1, None],
            "line": {"width": width, "color": f"rgba(148,163,184,{opacity:.2f})"},
            "hoverinfo": "none", "showlegend": False,
        })

    # Knoten
    node_x, node_y, node_text, node_hover = [], [], [], []
    node_color, node_size, node_border = [], [], []

    for pid in G_prod.nodes():
        m = meta.get(pid, {})
        x, y = pos[pid]
        node_x.append(x); node_y.append(y)
        node_text.append(m.get("name", pid))
        node_hover.append(
            f"<b>{m.get('name', pid)}</b><br>"
            f"Kategorie: {m.get('kat', '–')}<br>"
            f"Käufer: {G_prod.nodes[pid].get('n_kunden', '–')}<br>"
            f"Umsatz: {G_prod.nodes[pid].get('umsatz', 0):,.0f} €"
        )
        kat   = m.get("kat", "Unbekannt")
        color = KAT_FARBE.get(kat, "#94a3b8")
        is_hl = highlight_ids and pid in highlight_ids
        node_color.append("#fff" if is_hl else color)
        node_size.append(28 if is_hl else 20)
        node_border.append(color)

    node_trace = {
        "type": "scatter", "mode": "markers+text",
        "x": node_x, "y": node_y,
        "text": node_text,
        "textposition": "top center",
        "textfont": {"size": 10, "color": "#cbd5e1"},
        "hovertext": node_hover, "hoverinfo": "text",
        "marker": {
            "size": node_size,
            "color": node_color,
            "line": {"color": node_border, "width": 2},
            "opacity": 0.92,
        },
        "showlegend": False,
    }

    # Legende als Dummy-Traces
    legend_traces = []
    for kat, col in KAT_FARBE.items():
        legend_traces.append({
            "type": "scatter", "mode": "markers",
            "x": [None], "y": [None],
            "name": kat,
            "marker": {"size": 10, "color": col},
            "showlegend": True,
        })

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "margin": {"l": 0, "r": 0, "t": 20, "b": 20},
        "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "font": {"family": "Inter, sans-serif", "size": 11, "color": "#94a3b8"},
        "legend": {"orientation": "h", "y": -0.05,
                   "font": {"size": 11}, "bgcolor": "rgba(0,0,0,0)"},
        "hovermode": "closest",
    }

    fig = {"data": edge_traces + [node_trace] + legend_traces, "layout": layout}
    return json.dumps(fig, default=float)


def analyse(df: pd.DataFrame,
            selected_kunde_id: str | None = None) -> CrossSellingResult:

    df = _normalise(df.copy())
    result = CrossSellingResult()

    # ── Basisdaten ────────────────────────────────────────────────────────
    n_kunden   = df["kunde_id"].nunique()
    n_produkte = df["produkt_id"].nunique()
    n_tx       = len(df)
    umsatz_ges = float(df["umsatz"].sum())

    result.kpi = {
        "n_kunden":   n_kunden,
        "n_produkte": n_produkte,
        "n_tx":       n_tx,
        "umsatz":     round(umsatz_ges, 0),
    }

    # ── Produktmeta ───────────────────────────────────────────────────────
    prod_meta: dict[str, dict] = {}
    for pid, grp in df.groupby("produkt_id"):
        prod_meta[pid] = {
            "name":  grp["produkt_name"].iloc[0],
            "kat":   grp["kategorie"].iloc[0] if "kategorie" in grp.columns else "–",
            "umsatz": float(grp["umsatz"].sum()),
        }

    # ── Kunden-Produkt-Matrix (binär) ─────────────────────────────────────
    kp_matrix = (
        df.groupby(["kunde_id", "produkt_id"])["umsatz"]
        .sum().unstack(fill_value=0)
    )
    kp_binary = (kp_matrix > 0).astype(int)

    # ── Co-Occurrence-Matrix ──────────────────────────────────────────────
    cooc = kp_binary.T @ kp_binary   # produkt × produkt; diagonal = #Käufer

    all_prods = list(cooc.index)
    n_k = n_kunden

    # ── Lift-Scores & Top-Paare ───────────────────────────────────────────
    paare = []
    for i, pa in enumerate(all_prods):
        for pb in all_prods[i+1:]:
            n_ab = int(cooc.loc[pa, pb])
            if n_ab == 0:
                continue
            n_a = int(cooc.loc[pa, pa])
            n_b = int(cooc.loc[pb, pb])
            support_a  = n_a  / n_k
            support_b  = n_b  / n_k
            support_ab = n_ab / n_k
            lift = support_ab / (support_a * support_b) if support_a * support_b > 0 else 0
            konfidenz_ab = support_ab / support_a if support_a > 0 else 0
            paare.append({
                "produkt_a":  pa,
                "name_a":     prod_meta.get(pa, {}).get("name", pa),
                "produkt_b":  pb,
                "name_b":     prod_meta.get(pb, {}).get("name", pb),
                "gemeinsame_kunden": n_ab,
                "lift":       round(lift, 2),
                "konfidenz":  round(konfidenz_ab * 100, 1),
                "support":    round(support_ab * 100, 1),
            })

    paare.sort(key=lambda x: (-x["lift"], -x["gemeinsame_kunden"]))
    result.top_paare = paare[:15]

    # ── Bipartiter Graph → Produkt-Graph ──────────────────────────────────
    B = nx.Graph()
    for kid in kp_binary.index:
        for pid in kp_binary.columns:
            if kp_binary.loc[kid, pid]:
                B.add_edge(f"K_{kid}", pid)

    G_prod = nx.Graph()
    for pid in all_prods:
        n_k_pid = int(cooc.loc[pid, pid])
        G_prod.add_node(pid, n_kunden=n_k_pid, umsatz=prod_meta.get(pid, {}).get("umsatz", 0))

    for p in paare:
        if p["gemeinsame_kunden"] >= 1:
            G_prod.add_edge(p["produkt_a"], p["produkt_b"],
                            weight=p["gemeinsame_kunden"],
                            lift=p["lift"])

    # ── Community Detection ───────────────────────────────────────────────
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G_prod))
        result.communities = [
            [prod_meta.get(p, {}).get("name", p) for p in sorted(c)]
            for c in comms
        ]
    except Exception:
        result.communities = []

    # ── Kundenliste ───────────────────────────────────────────────────────
    kunden_list = []
    for kid, kgrp in df.groupby("kunde_id"):
        prods_bought = set(kgrp["produkt_id"].unique())
        kunden_list.append({
            "id":        kid,
            "name":      kgrp["kunde_name"].iloc[0],
            "n_produkte": len(prods_bought),
            "umsatz":    round(float(kgrp["umsatz"].sum()), 0),
            "produkte":  [prod_meta.get(p, {}).get("name", p) for p in sorted(prods_bought)],
        })
    kunden_list.sort(key=lambda x: -x["umsatz"])
    result.kunden = kunden_list

    # ── Empfehlungen je Kunde (alle) ──────────────────────────────────────
    alle_empf = []
    for k in kunden_list:
        kid  = k["id"]
        bought = set(kp_binary.loc[kid][kp_binary.loc[kid] == 1].index) \
            if kid in kp_binary.index else set()
        missing = [p for p in all_prods if p not in bought]

        empf_k = []
        for pid in missing:
            # Score = Σ Lift(gekauftes Produkt, pid) × #gemeinsame Kunden
            score = 0.0
            evidence_kunden = set()
            for pb in bought:
                w = float(cooc.loc[pb, pid]) if pb in cooc.index and pid in cooc.columns else 0
                if w > 0:
                    row = next((x for x in paare if
                                (x["produkt_a"] == pb and x["produkt_b"] == pid) or
                                (x["produkt_a"] == pid and x["produkt_b"] == pb)), None)
                    lift_val = row["lift"] if row else 1.0
                    score += w * lift_val
                    # Finde welche Kunden pid UND pb gekauft haben
                    for ok in kp_binary.index:
                        if ok != kid and kp_binary.loc[ok, pid] and kp_binary.loc[ok, pb]:
                            evidence_kunden.add(ok)

            if score > 0:
                # Evidence-Kunden-Namen
                ev_namen = [
                    next((k2["name"] for k2 in kunden_list if k2["id"] == ek), ek)
                    for ek in sorted(evidence_kunden)
                ][:3]
                empf_k.append({
                    "produkt_id":   pid,
                    "produkt_name": prod_meta.get(pid, {}).get("name", pid),
                    "kategorie":    prod_meta.get(pid, {}).get("kat", "–"),
                    "score":        round(score, 2),
                    "evidence":     ev_namen,
                    "begruendung":  (
                        f"Weil {', '.join(ev_namen)} dieses Produkt "
                        f"zusammen mit Ihren bisherigen Produkten kauften."
                        if ev_namen else "Basierend auf Kaufmuster-Ähnlichkeit."
                    ),
                })

        empf_k.sort(key=lambda x: -x["score"])
        for e in empf_k[:3]:
            alle_empf.append({"kunde_id": kid, "kunde_name": k["name"], **e})

    result.alle_empfehlungen = alle_empf
    result.kpi["n_empfehlungen"] = len(alle_empf)

    # ── Kundenspezifische Ansicht ──────────────────────────────────────────
    highlight_ids: set[str] = set()
    if selected_kunde_id:
        k_info = next((k for k in kunden_list if k["id"] == selected_kunde_id), None)
        if k_info:
            result.selected_kunde = k_info
            result.selected_empfehlungen = [
                e for e in alle_empf if e["kunde_id"] == selected_kunde_id
            ][:3]
            bought_ids = set(kp_binary.loc[selected_kunde_id][
                kp_binary.loc[selected_kunde_id] == 1
            ].index) if selected_kunde_id in kp_binary.index else set()
            highlight_ids = bought_ids

            # Käufe des gewählten Kunden (für Tabelle)
            bought_df = df[df["kunde_id"] == selected_kunde_id].copy()
            grp = (
                bought_df.groupby(["produkt_id", "produkt_name", "kategorie"])
                .agg(n_tx=("umsatz", "count"), umsatz=("umsatz", "sum"))
                .reset_index()
            )
            result.selected_kaeufe = grp.sort_values("umsatz", ascending=False).to_dict(orient="records")

    # ── Graph JSON ────────────────────────────────────────────────────────
    result.graph_json = _build_graph_figure(G_prod, prod_meta, highlight_ids or None)

    return result
