"""Generiert Plotly-JSON-Daten für die Aurora Web App."""
import numpy as np
import pandas as pd
import json
from aurora_inference import ForecastResult

WD_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

_COLORS = dict(
    hist   = "#7eb8f7",
    band   = "rgba(109, 40, 217, 0.25)",
    band_l = "#6d28d9",
    mean   = "#f472b6",
    median = "#34d399",
    samp   = "rgba(167, 139, 250, 0.07)",
    bg     = "#0a0a18",
    grid   = "#1a1a32",
    text   = "#e2e8f0",
    sub    = "#94a3b8",
)


def make_forecast_chart(result: ForecastResult, show_history: int = 168) -> str:
    """Gibt Plotly-Figure als JSON-String zurück."""
    H   = result.horizon
    sh  = min(show_history, len(result.history_values))
    his = result.history_values[-sh:]

    if result.history_timestamps is not None:
        hist_x = [str(t) for t in result.history_timestamps[-sh:]]
        fore_x = [str(t) for t in result.forecast_timestamps] if result.forecast_timestamps is not None else list(range(H))
    else:
        hist_x = list(range(-sh, 0))
        fore_x = list(range(H))

    traces = []

    # Sample-Linien (transparent, nur 25)
    for i in range(min(25, result.num_samples)):
        traces.append({
            "x": fore_x, "y": result.samples_orig[i].tolist(),
            "type": "scatter", "mode": "lines",
            "line": {"color": "rgba(167,139,250,0.06)", "width": 0.6},
            "showlegend": False, "hoverinfo": "skip",
        })

    # Konfidenzband Q10-Q90
    traces.append({
        "x": fore_x + fore_x[::-1],
        "y": result.q_lo.tolist() + result.q_hi.tolist()[::-1],
        "type": "scatter", "fill": "toself",
        "fillcolor": _COLORS["band"],
        "line": {"color": "transparent"},
        "name": "Q10–Q90 Konfidenz",
        "showlegend": True,
    })

    # History
    traces.append({
        "x": hist_x, "y": his.tolist(),
        "type": "scatter", "mode": "lines",
        "line": {"color": _COLORS["hist"], "width": 1.8},
        "name": "Historie",
    })

    # Mittelwert
    traces.append({
        "x": fore_x, "y": result.mean.tolist(),
        "type": "scatter", "mode": "lines",
        "line": {"color": _COLORS["mean"], "width": 2.2},
        "name": "Erwartungswert",
    })

    # Median
    traces.append({
        "x": fore_x, "y": result.median.tolist(),
        "type": "scatter", "mode": "lines",
        "line": {"color": _COLORS["median"], "width": 1.6, "dash": "dash"},
        "name": "Median",
    })

    layout = {
        "paper_bgcolor": _COLORS["bg"],
        "plot_bgcolor":  _COLORS["bg"],
        "font": {"color": _COLORS["text"], "family": "Inter, sans-serif"},
        "title": {
            "text": f"Aurora Forecast – {result.target_column}  "
                    f"| H={H}  | {result.num_samples} Szenarien",
            "font": {"size": 15},
        },
        "xaxis": {
            "gridcolor": _COLORS["grid"],
            "linecolor": _COLORS["grid"],
            "tickfont": {"color": _COLORS["sub"]},
        },
        "yaxis": {
            "gridcolor": _COLORS["grid"],
            "linecolor": _COLORS["grid"],
            "tickfont": {"color": _COLORS["sub"]},
        },
        "legend": {
            "bgcolor": "#0f0f28",
            "bordercolor": "#2a2a4a",
            "font": {"color": _COLORS["text"]},
        },
        "shapes": [{
            "type": "line",
            "x0": fore_x[0], "x1": fore_x[0],
            "y0": 0, "y1": 1, "yref": "paper",
            "line": {"color": "white", "width": 0.8, "dash": "dot"},
        }],
        "margin": {"l": 50, "r": 20, "t": 60, "b": 50},
        "hovermode": "x unified",
    }

    return json.dumps({"data": traces, "layout": layout})


def make_daily_bar_chart(result: ForecastResult) -> str:
    """Balkendiagramm mit Tages-Aggregation. Gibt Plotly-JSON zurück."""
    if result.daily_mean is None:
        return ""

    days  = result.daily_dates
    dm    = result.daily_mean
    dq10  = result.daily_q10
    dq90  = result.daily_q90

    bar_colors = [
        "#fb923c" if d.startswith("Sa") or d.startswith("So") else "#f472b6"
        for d in days
    ]

    traces = [
        {
            "x": days, "y": dm.tolist(),
            "type": "bar",
            "name": "Tages-Erwartung",
            "marker": {"color": bar_colors, "opacity": 0.88},
            "error_y": {
                "type": "data",
                "symmetric": False,
                "array":      (dq90 - dm).tolist(),
                "arrayminus": (dm - dq10).tolist(),
                "color": "#e2e8f0",
                "thickness": 1.5,
                "width": 5,
            },
        }
    ]

    layout = {
        "paper_bgcolor": _COLORS["bg"],
        "plot_bgcolor":  _COLORS["bg"],
        "font": {"color": _COLORS["text"], "family": "Inter, sans-serif"},
        "title": {
            "text": f"Tägliche Prognose – {result.target_column}  (Balken = Erwartung, Balken = Q10–Q90)",
            "font": {"size": 14},
        },
        "xaxis": {
            "gridcolor": _COLORS["grid"],
            "tickfont": {"color": _COLORS["sub"]},
        },
        "yaxis": {
            "gridcolor": _COLORS["grid"],
            "tickfont": {"color": _COLORS["sub"]},
        },
        "bargap": 0.25,
        "margin": {"l": 50, "r": 20, "t": 55, "b": 60},
        "showlegend": False,
    }

    return json.dumps({"data": traces, "layout": layout})
