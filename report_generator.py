"""
Generiert deutsche Geschäftsberichte aus Aurora-Forecast-Ergebnissen.
Der Bericht beantwortet konkrete Szenario-Fragen basierend auf den
probabilistischen Vorhersagen des Modells.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from aurora_inference import ForecastResult

WD_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
MONTH_DE = [
    "", "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]

# Vordefinierte Report-Templates nach Anwendungsfall
SCENARIO_TEMPLATES = {
    "einkauf":      "Einkaufs- & Bestandsplanung",
    "personal":     "Personalplanung",
    "energie":      "Energiemanagement",
    "umsatz":       "Umsatzplanung",
    "produktion":   "Produktionsplanung",
    "allgemein":    "Allgemeine Prognose",
}

UNIT_HINTS = {
    "liter":  ("L",  "Litern",       50,   "Fässer à 50 L"),
    "kwh":    ("kWh","kWh",          None, None),
    "eur":    ("€",  "Euro",         None, None),
    "pieces": ("Stk","Stück",        None, None),
    "orders": ("Bst","Bestellungen", None, None),
    "count":  ("",   "Einheiten",    None, None),
}


def _detect_unit(column_name: str):
    col = column_name.lower()
    if any(x in col for x in ["liter", "beer", "bier", "ml", "cl"]):
        return UNIT_HINTS["liter"]
    if any(x in col for x in ["kwh", "energie", "energy", "strom", "power"]):
        return UNIT_HINTS["kwh"]
    if any(x in col for x in ["eur", "revenue", "umsatz", "sales", "price"]):
        return UNIT_HINTS["eur"]
    if any(x in col for x in ["order", "bestell"]):
        return UNIT_HINTS["orders"]
    return UNIT_HINTS["count"]


def generate_report(
    result: ForecastResult,
    scenario_question: str = "",
    scenario_type: str = "allgemein",
    freq: str = "h",
    keg_size: float = 50.0,
    safety_buffer: float = 0.15,
    custom_unit: str = "",
    horizon_weeks: int = 1,
) -> dict:
    """
    Generiert einen strukturierten Bericht aus den Forecast-Ergebnissen.
    Gibt ein dict zurück mit 'html', 'summary', 'tables', 'kpis'.
    """
    unit_short, unit_long, default_keg, keg_label = _detect_unit(
        custom_unit or result.target_column
    )

    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")
    scenario_title = SCENARIO_TEMPLATES.get(scenario_type, "Allgemeine Prognose")

    # ── Gesamtstatistik ──────────────────────────────────────────────────
    total_mean   = float(result.mean.sum())
    total_q10    = float(result.q_lo.sum())
    total_q90    = float(result.q_hi.sum())
    total_safe   = total_q90 * (1 + safety_buffer)

    mean_per_step   = float(result.mean.mean())
    spread_pct      = ((total_q90 - total_q10) / (total_mean + 1e-8)) * 100

    # ── Tages-Aggregation (wenn vorhanden) ──────────────────────────────
    daily_section = ""
    daily_table   = []
    peak_day_label = ""
    low_day_label  = ""

    if result.daily_mean is not None and len(result.daily_mean) > 0:
        dm  = result.daily_mean
        dq10 = result.daily_q10
        dq90 = result.daily_q90
        days = result.daily_dates or [f"Tag {i+1}" for i in range(len(dm))]

        max_idx = int(np.argmax(dm))
        min_idx = int(np.argmin(dm))
        peak_day_label = days[max_idx]
        low_day_label  = days[min_idx]
        safe_daily = dq90 * (1 + safety_buffer)

        rows_html = ""
        for i, (d, m, lo, hi, sf) in enumerate(zip(days, dm, dq10, dq90, safe_daily)):
            badge = ""
            if i == max_idx:
                badge = ' <span class="badge-peak">Peak</span>'
            elif i == min_idx:
                badge = ' <span class="badge-low">Tief</span>'
            we_cls = " weekend-row" if i < len(days) and _is_weekend(days[i]) else ""
            rows_html += f"""
            <tr class="{we_cls}">
                <td><strong>{d}</strong>{badge}</td>
                <td class="num">{m:,.1f}</td>
                <td class="num">{lo:,.0f} – {hi:,.0f}</td>
                <td class="num safe">{sf:,.1f}</td>
            </tr>"""
            daily_table.append({
                "tag": d,
                "erwartet": round(float(m), 1),
                "q10":      round(float(lo), 0),
                "q90":      round(float(hi), 0),
                "sicher":   round(float(sf), 1),
            })

        total_safe_row = np.array(dq90).sum() * (1 + safety_buffer)
        rows_html += f"""
            <tr class="total-row">
                <td><strong>GESAMT</strong></td>
                <td class="num"><strong>{dm.sum():,.1f}</strong></td>
                <td class="num">{dq10.sum():,.0f} – {dq90.sum():,.0f}</td>
                <td class="num safe"><strong>{total_safe_row:,.1f}</strong></td>
            </tr>"""

        daily_section = f"""
        <h3>Tägliche Aufschlüsselung</h3>
        <div class="table-wrapper">
        <table class="forecast-table">
            <thead>
                <tr>
                    <th>Tag</th>
                    <th>Erwartet ({unit_short})</th>
                    <th>Bereich Q10–Q90 ({unit_short})</th>
                    <th>Sicher +{safety_buffer*100:.0f}% ({unit_short})</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>"""

    # ── Einkaufsempfehlung (Liter/Fässer) ───────────────────────────────
    purchase_section = ""
    kpis_purchase = {}
    if unit_short == "L" and keg_size > 0:
        n_kegs_mean = int(np.ceil(total_mean / keg_size))
        n_kegs_safe = int(np.ceil(total_safe / keg_size))
        kpis_purchase = {
            "kegs_mean": n_kegs_mean,
            "kegs_safe": n_kegs_safe,
            "liters_mean": round(total_mean, 1),
            "liters_safe": round(total_safe, 1),
        }
        purchase_section = f"""
        <div class="purchase-box">
            <h3>🛒 Einkaufsempfehlung</h3>
            <div class="purchase-grid">
                <div class="purchase-item conservative">
                    <div class="purchase-label">Erwartungswert</div>
                    <div class="purchase-value">{n_kegs_mean} Fässer</div>
                    <div class="purchase-sub">≈ {total_mean:,.0f} {unit_short}</div>
                </div>
                <div class="purchase-item recommended">
                    <div class="purchase-label">Empfehlung (+{safety_buffer*100:.0f}% Puffer)</div>
                    <div class="purchase-value">{n_kegs_safe} Fässer</div>
                    <div class="purchase-sub">≈ {total_safe:,.0f} {unit_short}</div>
                </div>
            </div>
            <p class="purchase-note">
                Die Empfehlung deckt das 90%-Quantil aller {result.num_samples} KI-Szenarien
                zuzüglich {safety_buffer*100:.0f}% Sicherheitspuffer ab.
            </p>
        </div>"""

    # ── KPI-Kacheln ─────────────────────────────────────────────────────
    kpis = {
        "total_mean":    round(total_mean, 1),
        "total_q90":     round(total_q90, 1),
        "total_safe":    round(total_safe, 1),
        "mean_per_step": round(mean_per_step, 2),
        "spread_pct":    round(spread_pct, 1),
        "num_samples":   result.num_samples,
        "horizon":       result.horizon,
        "peak_day":      peak_day_label,
        "low_day":       low_day_label,
        **kpis_purchase,
    }

    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Gesamt-Erwartung</div>
            <div class="kpi-value">{total_mean:,.1f}</div>
            <div class="kpi-unit">{unit_long}</div>
        </div>
        <div class="kpi-card kpi-warn">
            <div class="kpi-label">Worst-Case Q90</div>
            <div class="kpi-value">{total_q90:,.1f}</div>
            <div class="kpi-unit">{unit_long}</div>
        </div>
        <div class="kpi-card kpi-safe">
            <div class="kpi-label">Sicherer Bedarf (+{safety_buffer*100:.0f}%)</div>
            <div class="kpi-value">{total_safe:,.1f}</div>
            <div class="kpi-unit">{unit_long}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Modell-Unsicherheit</div>
            <div class="kpi-value">{spread_pct:.1f}%</div>
            <div class="kpi-unit">Q10–Q90 Spanne</div>
        </div>
    </div>"""

    # ── Narrativer Text ──────────────────────────────────────────────────
    peak_info = f"Der stärkste Tag wird <strong>{peak_day_label}</strong> sein." if peak_day_label else ""
    low_info  = f"Den geringsten Bedarf erwarten wir am <strong>{low_day_label}</strong>." if low_day_label else ""

    horizon_text = _horizon_label(result.horizon, freq)
    scenario_q   = f'"{scenario_question}"' if scenario_question else "Ihre Prognose"

    narrative = f"""
    <div class="narrative">
        <h3>Zusammenfassung</h3>
        <p>
            Das Modell prognostiziert basierend auf <strong>{result.num_samples} Szenarien</strong>
            einen Gesamtbedarf von <strong>{total_mean:,.1f} {unit_long}</strong>
            über {horizon_text}.
            Das 90%-Quantil liegt bei <strong>{total_q90:,.1f} {unit_long}</strong>
            – dieser Wert sollte als Planungsgrundlage dienen.
        </p>
        <p>
            {peak_info} {low_info}
            Die Modell-Unsicherheit beträgt {spread_pct:.1f}% (Q10–Q90 Spanne),
            was auf {'hohe' if spread_pct > 40 else 'moderate' if spread_pct > 20 else 'geringe'}
            Vorhersage-Variabilität hindeutet.
        </p>
        {"<p>Das Prognose-Modell wurde mit folgendem Kontext-Prompt betrieben: <em>" + result.text_used + "</em></p>" if result.text_used else ""}
    </div>"""

    # ── Zusammenbau des vollständigen HTML-Berichts ──────────────────────
    html = f"""
    <div class="report">
        <div class="report-header">
            <div class="report-meta">
                <span class="report-date">Erstellt: {now_str}</span>
                <span class="report-badge">{scenario_title}</span>
            </div>
            <h2 class="report-title">{scenario_q}</h2>
            <p class="report-subtitle">
                KI-gestützte Zeitreihenvorhersage |
                Horizont: {horizon_text} | {result.num_samples} Szenarien
            </p>
        </div>

        {kpi_html}
        {narrative}
        {daily_section}
        {purchase_section}

        <div class="report-footer">
            <p>
                <strong>Modell:</strong> KI Foundation Model |
                <strong>Ziel-Variable:</strong> {result.target_column} |
                <strong>Lookback:</strong> 528 Zeitschritte |
                <strong>Quantile:</strong> Q10 / Q90
            </p>
        </div>
    </div>"""

    summary = (
        f"Prognose für {horizon_text}: "
        f"Erwartet {total_mean:,.0f} {unit_long}, "
        f"sicherer Bedarf {total_safe:,.0f} {unit_long}."
    )

    return {
        "html":        html,
        "summary":     summary,
        "daily_table": daily_table,
        "kpis":        kpis,
    }


def _is_weekend(day_label: str) -> bool:
    return day_label.startswith("Sa") or day_label.startswith("So")


def _horizon_label(horizon: int, freq: str) -> str:
    if freq == "h":
        days = horizon // 24
        if days == 7:
            return "1 Woche (7 Tage)"
        if days == 14:
            return "2 Wochen (14 Tage)"
        if days == 28:
            return "4 Wochen (28 Tage)"
        return f"{days} Tage ({horizon} Stunden)"
    if freq == "D":
        return f"{horizon} Tage"
    return f"{horizon} Schritte"
