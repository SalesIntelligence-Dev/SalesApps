"""
Produkt-Recommender auf Basis von Graph ML.

Sicht: der Kunde öffnet die App und sieht die Produkte, die für ihn als
nächster Kauf infrage kommen – inklusive Begründung.

Zwei Verfahren laufen parallel und werden zu einem Hybrid-Score kombiniert:

  1. Personalized PageRank auf dem bipartiten Kunde-Produkt-Graphen
     Der Random Walk startet beim Kunden, läuft über seine Produkte zu
     ähnlichen Kunden und von dort zu deren Produkten. Dadurch werden auch
     mehrstufige Zusammenhänge erfasst, die eine reine Co-Occurrence-Zählung
     nicht sieht (Kunde → Produkt → Kunde → Produkt → Kunde → Produkt).

  2. Latent-Factor-Modell über Truncated SVD der Interaktionsmatrix
     Klassische Matrixfaktorisierung: R ≈ U·Σ·Vᵀ. Die Rekonstruktion sagt
     eine Affinität auch für nie gekaufte Produkte vorher.

  Hybrid = 0.6 × PageRank + 0.4 × SVD

Erklärbarkeit: Für jede Empfehlung wird der konkrete Pfad im Graphen
zurückgegeben – welche eigenen Produkte über welche ähnlichen Kunden zur
Empfehlung führen. Ohne diese Begründung ist eine Empfehlung im B2B-Vertrieb
nicht verwendbar.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import numpy as np
import pandas as pd
import networkx as nx

PPR_GEWICHT = 0.60
SVD_GEWICHT = 0.40
SVD_RANK    = 4          # Latente Faktoren
PPR_ALPHA   = 0.85       # Dämpfung des Random Walks

GRUPPE_FARBE = {
    "Maschinen": "#60a5fa",
    "Zubehör":   "#34d399",
    "Service":   "#f59e0b",
}
GRUPPE_ICON = {
    "Maschinen": "⚙️",
    "Zubehör":   "🔩",
    "Service":   "🛠️",
}


@dataclass
class RecoResult:
    kpi: dict                     = field(default_factory=dict)
    kunden: list[dict]            = field(default_factory=list)
    selected_kunde: dict | None   = None
    empfehlungen: list[dict]      = field(default_factory=list)
    portfolio: list[dict]         = field(default_factory=list)
    graph_json: str               = "{}"
    algo: dict                    = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════
# Graph & Modelle
# ══════════════════════════════════════════════════════════════════════════

def _build_bipartite(kp: pd.DataFrame) -> nx.Graph:
    """Bipartiter Graph: Kunden ↔ Produkte, Kantengewicht = normierter Umsatz."""
    G = nx.Graph()
    for kid in kp.index:
        G.add_node(f"C_{kid}", typ="kunde")
    for pid in kp.columns:
        G.add_node(f"P_{pid}", typ="produkt")

    # Umsatz je Kunde auf 1 normieren, damit Großkunden den Walk nicht dominieren
    for kid in kp.index:
        row = kp.loc[kid]
        total = row.sum()
        if total <= 0:
            continue
        for pid, val in row.items():
            if val > 0:
                G.add_edge(f"C_{kid}", f"P_{pid}", weight=float(val / total))
    return G


def _personalized_pagerank(G: nx.Graph, kid: str) -> dict[str, float]:
    start = f"C_{kid}"
    if start not in G:
        return {}
    try:
        pr = nx.pagerank(G, alpha=PPR_ALPHA,
                         personalization={start: 1.0},
                         weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(G, alpha=PPR_ALPHA,
                         personalization={start: 1.0},
                         weight="weight", max_iter=1000, tol=1e-4)
    return {n[2:]: v for n, v in pr.items() if n.startswith("P_")}


def _svd_scores(kp: pd.DataFrame, kid: str) -> dict[str, float]:
    """Truncated SVD auf der log-skalierten Interaktionsmatrix."""
    R = np.log1p(kp.values.astype(float))
    if R.shape[0] < 2 or R.shape[1] < 2:
        return {}

    # Zeilen zentrieren (Kundenniveau herausrechnen)
    row_mean = R.mean(axis=1, keepdims=True)
    Rc = R - row_mean

    k = int(min(SVD_RANK, min(Rc.shape) - 1))
    if k < 1:
        return {}

    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    R_hat = (U[:, :k] * S[:k]) @ Vt[:k, :] + row_mean

    try:
        i = list(kp.index).index(kid)
    except ValueError:
        return {}
    return {pid: float(R_hat[i, j]) for j, pid in enumerate(kp.columns)}


def _norm(d: dict[str, float], keys: list[str]) -> dict[str, float]:
    """Min-Max-Normierung auf die betrachteten Kandidaten."""
    vals = [d.get(k, 0.0) for k in keys]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.5 for k in keys}
    return {k: (d.get(k, 0.0) - lo) / (hi - lo) for k in keys}


# ══════════════════════════════════════════════════════════════════════════
# Erklärbarkeit: Pfade im Graphen
# ══════════════════════════════════════════════════════════════════════════

def _erklaerung(kp_bin: pd.DataFrame, kid: str, pid: str,
                namen: dict[str, str], prod_namen: dict[str, str]) -> dict:
    """
    Findet die Pfade Kunde → eigenes Produkt → ähnlicher Kunde → Zielprodukt.
    Gibt die stärksten Belege zurück.
    """
    eigene = set(kp_bin.columns[kp_bin.loc[kid] > 0])
    # Kunden, die das Zielprodukt haben
    andere = [k for k in kp_bin.index
              if k != kid and kp_bin.loc[k, pid] > 0]

    belege = []
    for ok in andere:
        deren = set(kp_bin.columns[kp_bin.loc[ok] > 0])
        gemeinsam = eigene & deren
        if not gemeinsam:
            continue
        # Jaccard als Kundenähnlichkeit
        union = eigene | deren
        sim = len(gemeinsam) / len(union) if union else 0
        belege.append({
            "kunde":      namen.get(ok, ok),
            "gemeinsam":  sorted(prod_namen.get(g, g) for g in gemeinsam),
            "n_gemeinsam": len(gemeinsam),
            "sim":        round(sim, 3),
        })

    belege.sort(key=lambda x: (-x["sim"], -x["n_gemeinsam"]))
    top = belege[:3]

    # Welches eigene Produkt taucht in den Belegen am häufigsten auf?
    zaehler: dict[str, int] = {}
    for b in top:
        for g in b["gemeinsam"]:
            zaehler[g] = zaehler.get(g, 0) + 1
    bruecke = max(zaehler, key=zaehler.get) if zaehler else None

    if top:
        namen_liste = ", ".join(b["kunde"] for b in top)
        text = (f"{len(andere)} vergleichbare Kunden setzen dieses Produkt bereits ein – "
                f"darunter {namen_liste}.")
        if bruecke:
            text += f" Die Verbindung läuft vor allem über Ihr Produkt „{bruecke}“."
    else:
        text = "Das Modell leitet die Empfehlung aus latenten Kaufmustern ab."

    return {
        "text":     text,
        "belege":   top,
        "bruecke":  bruecke,
        "n_kunden": len(andere),
    }


# ══════════════════════════════════════════════════════════════════════════
# Empfehlungsgraph für die Visualisierung
# ══════════════════════════════════════════════════════════════════════════

def _build_graph_figure(kp_bin: pd.DataFrame, kid: str,
                        empf: list[dict], namen: dict[str, str],
                        prod_namen: dict[str, str], prod_gruppe: dict[str, str]) -> str:
    top_pids = [e["produkt_id"] for e in empf[:4]]
    eigene   = list(kp_bin.columns[kp_bin.loc[kid] > 0])

    # Relevante Nachbarkunden: die Belegkunden der Top-Empfehlungen
    nachbarn = set()
    for e in empf[:4]:
        for b in e["erklaerung"]["belege"]:
            for k2, n2 in namen.items():
                if n2 == b["kunde"]:
                    nachbarn.add(k2)
    nachbarn = list(nachbarn)[:5]

    G = nx.Graph()
    me = f"C_{kid}"
    G.add_node(me, rolle="ich", label=namen.get(kid, kid))
    for p in eigene:
        G.add_node(f"P_{p}", rolle="meins", label=prod_namen.get(p, p),
                   gruppe=prod_gruppe.get(p, ""))
        G.add_edge(me, f"P_{p}")
    for nk in nachbarn:
        G.add_node(f"C_{nk}", rolle="peer", label=namen.get(nk, nk))
        for p in kp_bin.columns[kp_bin.loc[nk] > 0]:
            node = f"P_{p}"
            if node in G:
                G.add_edge(f"C_{nk}", node)
    for p in top_pids:
        node = f"P_{p}"
        G.add_node(node, rolle="empfehlung", label=prod_namen.get(p, p),
                   gruppe=prod_gruppe.get(p, ""))
        for nk in nachbarn:
            if kp_bin.loc[nk, p] > 0:
                G.add_edge(f"C_{nk}", node)

    if len(G) == 0:
        return json.dumps({"data": [], "layout": {}})

    pos = nx.spring_layout(G, seed=13, k=1.9)
    pos[me] = [0.0, 0.0]

    edges = []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        highlight = (G.nodes[u].get("rolle") == "empfehlung"
                     or G.nodes[v].get("rolle") == "empfehlung")
        edges.append({
            "type": "scatter", "mode": "lines",
            "x": [x0, x1, None], "y": [y0, y1, None],
            "line": {"width": 2.0 if highlight else 1.0,
                     "color": "rgba(52,211,153,0.45)" if highlight
                              else "rgba(148,163,184,0.22)"},
            "hoverinfo": "none", "showlegend": False,
        })

    STIL = {
        "ich":        ("#a78bfa", 32, "Sie"),
        "meins":      ("#60a5fa", 18, "Ihre Produkte"),
        "peer":       ("#94a3b8", 20, "Ähnliche Kunden"),
        "empfehlung": ("#34d399", 28, "Empfohlen"),
    }
    gruppen: dict[str, dict] = {}
    for n in G.nodes():
        rolle = G.nodes[n].get("rolle", "peer")
        farbe, size, legende = STIL[rolle]
        e = gruppen.setdefault(rolle, {"x": [], "y": [], "t": [], "h": [],
                                        "farbe": farbe, "size": size, "name": legende})
        x, y = pos[n]
        e["x"].append(x); e["y"].append(y)
        e["t"].append(G.nodes[n].get("label", n))
        e["h"].append(f"<b>{G.nodes[n].get('label', n)}</b><br>{legende}")

    nodes = []
    for rolle, e in gruppen.items():
        nodes.append({
            "type": "scatter", "mode": "markers+text",
            "name": e["name"],
            "x": e["x"], "y": e["y"],
            "text": e["t"], "textposition": "bottom center",
            "textfont": {"size": 9, "color": "#cbd5e1"},
            "hovertext": e["h"], "hoverinfo": "text",
            "marker": {"size": e["size"], "color": e["farbe"],
                       "line": {"color": "#0f172a", "width": 2}, "opacity": 0.93},
        })

    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 0, "r": 0, "t": 10, "b": 30},
        "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "font": {"family": "Inter, sans-serif", "size": 11, "color": "#94a3b8"},
        "legend": {"orientation": "h", "y": -0.02, "font": {"size": 10},
                   "bgcolor": "rgba(0,0,0,0)"},
        "hovermode": "closest",
    }
    return json.dumps({"data": edges + nodes, "layout": layout}, default=float)


# ══════════════════════════════════════════════════════════════════════════
# Hauptanalyse
# ══════════════════════════════════════════════════════════════════════════

def analyse(df: pd.DataFrame, selected_kunde_id: str | None = None) -> RecoResult:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    result = RecoResult()

    # ── Interaktionsmatrix (Umsatz über alle Jahre) ──────────────────────
    kp = (df.groupby(["kunde_id", "produkt_id"])["umsatz"]
            .sum().unstack(fill_value=0.0))
    kp_bin = (kp > 0).astype(int)

    namen      = df.groupby("kunde_id")["kunde_name"].first().to_dict()
    prod_namen = df.groupby("produkt_id")["produkt_name"].first().to_dict()
    prod_grp   = df.groupby("produkt_id")["produktgruppe"].first().to_dict()
    prod_preis = df.groupby("produkt_id")["einzelpreis"].median().to_dict()

    # Typische Jahresmenge je Produkt (für den geschätzten Wert)
    jahr_max = df["jahr"].max()
    df_akt   = df[df["jahr"] == jahr_max]
    prod_menge = df_akt.groupby("produkt_id")["menge"].median().to_dict()

    # ── Kundenliste ───────────────────────────────────────────────────────
    stamm = (df.groupby("kunde_id")
               .agg(name=("kunde_name", "first"),
                    branche=("branche", "first"),
                    groesse=("groesse_klasse", "first"),
                    rep=("vertriebsmitarbeiter", "first"))
               .reset_index())
    result.kunden = [
        {"id": r["kunde_id"], "name": r["name"], "branche": r["branche"],
         "n_produkte": int(kp_bin.loc[r["kunde_id"]].sum())
                       if r["kunde_id"] in kp_bin.index else 0}
        for _, r in stamm.iterrows()
    ]
    result.kunden.sort(key=lambda x: x["name"])

    result.kpi = {
        "n_kunden":   len(result.kunden),
        "n_produkte": int(kp.shape[1]),
        "n_kanten":   int(kp_bin.values.sum()),
        "dichte":     round(kp_bin.values.sum() / (kp.shape[0] * kp.shape[1]) * 100, 1),
    }
    result.algo = {
        "ppr_gewicht": int(PPR_GEWICHT * 100),
        "svd_gewicht": int(SVD_GEWICHT * 100),
        "svd_rank":    SVD_RANK,
        "alpha":       PPR_ALPHA,
    }

    if not selected_kunde_id or selected_kunde_id not in kp.index:
        return result

    kid = selected_kunde_id
    k_row = stamm[stamm["kunde_id"] == kid].iloc[0]

    # ── Graph ML ──────────────────────────────────────────────────────────
    G = _build_bipartite(kp)
    ppr = _personalized_pagerank(G, kid)
    svd = _svd_scores(kp, kid)

    besessen  = set(kp_bin.columns[kp_bin.loc[kid] > 0])
    kandidaten = [p for p in kp.columns if p not in besessen]

    if not kandidaten:
        result.selected_kunde = {
            "id": kid, "name": k_row["name"], "branche": k_row["branche"],
            "groesse": k_row["groesse"], "rep": k_row["rep"],
            "n_produkte": len(besessen), "n_gesamt": int(kp.shape[1]),
            "abdeckung": 100,
        }
        return result

    ppr_n = _norm(ppr, kandidaten)
    svd_n = _norm(svd, kandidaten)

    scored = []
    for pid in kandidaten:
        hybrid = PPR_GEWICHT * ppr_n.get(pid, 0) + SVD_GEWICHT * svd_n.get(pid, 0)
        scored.append((pid, hybrid, ppr_n.get(pid, 0), svd_n.get(pid, 0)))
    scored.sort(key=lambda x: -x[1])

    # Match-Score auf 45–97 spreizen, damit die Rangfolge lesbar bleibt
    hi = scored[0][1] if scored else 1.0
    lo = scored[-1][1] if scored else 0.0
    spanne = (hi - lo) or 1.0

    empfehlungen = []
    for rang, (pid, hybrid, p_ppr, p_svd) in enumerate(scored, 1):
        match = 45 + (hybrid - lo) / spanne * 52
        erkl = _erklaerung(kp_bin, kid, pid, namen, prod_namen)

        preis = float(prod_preis.get(pid, 0) or 0)
        menge = float(prod_menge.get(pid, 1) or 1)
        jahreswert = preis * menge

        gruppe = prod_grp.get(pid, "")
        empfehlungen.append({
            "rang":         rang,
            "produkt_id":   pid,
            "produkt_name": prod_namen.get(pid, pid),
            "gruppe":       gruppe,
            "farbe":        GRUPPE_FARBE.get(gruppe, "#94a3b8"),
            "icon":         GRUPPE_ICON.get(gruppe, "📦"),
            "match":        round(match, 0),
            "score_ppr":    round(p_ppr * 100, 0),
            "score_svd":    round(p_svd * 100, 0),
            "preis":        round(preis, 0),
            "menge":        round(menge, 0),
            "jahreswert":   round(jahreswert, -1),
            "erklaerung":   erkl,
            "passt_zu":     erkl["bruecke"],
        })

    result.empfehlungen = empfehlungen

    # ── Aktuelles Portfolio ───────────────────────────────────────────────
    port = []
    for pid in sorted(besessen):
        umsatz = float(kp.loc[kid, pid])
        gruppe = prod_grp.get(pid, "")
        port.append({
            "produkt_id":   pid,
            "produkt_name": prod_namen.get(pid, pid),
            "gruppe":       gruppe,
            "farbe":        GRUPPE_FARBE.get(gruppe, "#94a3b8"),
            "icon":         GRUPPE_ICON.get(gruppe, "📦"),
            "umsatz":       round(umsatz, 0),
        })
    port.sort(key=lambda x: -x["umsatz"])
    result.portfolio = port

    result.selected_kunde = {
        "id":          kid,
        "name":        k_row["name"],
        "branche":     k_row["branche"],
        "groesse":     k_row["groesse"],
        "rep":         k_row["rep"],
        "n_produkte":  len(besessen),
        "n_gesamt":    int(kp.shape[1]),
        "abdeckung":   round(len(besessen) / kp.shape[1] * 100),
        "umsatz":      round(float(kp.loc[kid].sum()), 0),
        "potenzial":   round(sum(e["jahreswert"] for e in empfehlungen[:3]), -1),
    }

    result.graph_json = _build_graph_figure(
        kp_bin, kid, empfehlungen, namen, prod_namen, prod_grp)

    return result
