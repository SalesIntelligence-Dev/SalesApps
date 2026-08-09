"""Deskriptive Margenleckage- und Rabattanalyse. Nur pandas, kein ML."""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

MINDESTMARGE = 15.0   # %


@dataclass
class MargenResult:
    kpi: dict                           = field(default_factory=dict)
    top_lecks: list[dict]               = field(default_factory=list)
    mitarbeiter_vergleich: list[dict]   = field(default_factory=list)
    produkt_margen: list[dict]          = field(default_factory=list)
    kunden_rabatte: list[dict]          = field(default_factory=list)
    insights: list[dict]                = field(default_factory=list)
    zeilen_gesamt: int                  = 0


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Bringt flexibel benannte Spalten auf kanonische Namen."""
    rename = {}
    cols_lower = {c.lower(): c for c in df.columns}

    _alias = {
        "angebot_id":           ["angebot_id", "angebotsnummer", "id", "nr"],
        "datum":                ["datum", "date"],
        "kunde":                ["kunde", "kundenname", "customer", "kunden"],
        "produkt":              ["produkt", "produktname", "product"],
        "kategorie":            ["kategorie", "category", "cat"],
        "listenpreis":          ["listenpreis", "list_price", "brutto"],
        "verkaufspreis":        ["verkaufspreis", "vk", "netto", "preis", "sell_price"],
        "rabatt_pct":           ["rabatt_pct", "rabatt", "discount", "discount_pct"],
        "menge":                ["menge", "quantity", "qty"],
        "umsatz":               ["umsatz", "revenue", "erlös"],
        "deckungsbeitrag_pct":  ["deckungsbeitrag_pct", "deckungsbeitrag", "marge", "margin"],
        "vertriebsmitarbeiter": ["vertriebsmitarbeiter", "mitarbeiter", "sales_rep", "vma"],
        "region":               ["region", "gebiet"],
    }

    for canonical, aliases in _alias.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in cols_lower:
                rename[cols_lower[alias.lower()]] = canonical
                break

    if rename:
        df = df.rename(columns=rename)

    # deckungsbeitrag_pct aus Preisen berechnen, wenn nicht vorhanden
    if "deckungsbeitrag_pct" not in df.columns:
        if "verkaufspreis" in df.columns and "listenpreis" in df.columns:
            herstellkosten = df["listenpreis"] * 0.55
            df["deckungsbeitrag_pct"] = (
                (df["verkaufspreis"] - herstellkosten)
                / df["verkaufspreis"].replace(0, float("nan")) * 100
            ).round(1)
        elif "rabatt_pct" in df.columns:
            df["deckungsbeitrag_pct"] = (30 - df["rabatt_pct"]).round(1)
        else:
            df["deckungsbeitrag_pct"] = 25.0

    # rabatt_pct aus Preisen berechnen, wenn nicht vorhanden
    if "rabatt_pct" not in df.columns:
        if "listenpreis" in df.columns and "verkaufspreis" in df.columns:
            df["rabatt_pct"] = (
                (df["listenpreis"] - df["verkaufspreis"])
                / df["listenpreis"].replace(0, float("nan")) * 100
            ).clip(0, 100).round(1)
        else:
            df["rabatt_pct"] = 0.0

    if "umsatz" not in df.columns and "verkaufspreis" in df.columns and "menge" in df.columns:
        df["umsatz"] = (df["verkaufspreis"] * df["menge"]).round(2)

    return df


def _fmt_eur(val: float) -> str:
    return f"{val:,.0f} €".replace(",", ".")


def analyse(df: pd.DataFrame) -> MargenResult:
    df = _normalise(df.copy())
    result = MargenResult(zeilen_gesamt=len(df))

    has_umsatz = "umsatz" in df.columns

    # ── Margenlecks identifizieren ─────────────────────────────────────────
    lecks = df[df["deckungsbeitrag_pct"] < MINDESTMARGE].copy()

    # Einsparpotenzial: wie viel DB-EUR kann zurückgewonnen werden,
    # wenn alle Leck-Angebote auf 15 % Mindestmarge angehoben werden?
    if has_umsatz:
        leck_umsatz = lecks["umsatz"].fillna(0)
        leck_db_ist = lecks["deckungsbeitrag_pct"].fillna(0) / 100 * leck_umsatz
        leck_db_ziel = MINDESTMARGE / 100 * leck_umsatz
        einsparpotenzial_eur = float((leck_db_ziel - leck_db_ist).clip(lower=0).sum())
        leck_volumen = float(leck_umsatz.sum())
    else:
        einsparpotenzial_eur = 0.0
        leck_volumen = float(len(lecks))

    avg_rabatt = float(df["rabatt_pct"].mean()) if "rabatt_pct" in df.columns else 0.0
    avg_marge  = float(df["deckungsbeitrag_pct"].mean())
    ausreisser = int(
        (df["rabatt_pct"] > df["rabatt_pct"].quantile(0.90)).sum()
    ) if "rabatt_pct" in df.columns else 0

    result.kpi = {
        "avg_rabatt":          round(avg_rabatt, 1),
        "avg_marge":           round(avg_marge, 1),
        "einsparpotenzial_eur": round(einsparpotenzial_eur, 0),
        "leck_anzahl":         len(lecks),
        "leck_volumen":        round(leck_volumen, 0),
        "ausreisser":          ausreisser,
        "gesamt":              len(df),
    }

    # ── Top-5 Margenlecks ─────────────────────────────────────────────────
    id_col = "angebot_id" if "angebot_id" in df.columns else df.columns[0]
    leck_sorted = lecks.sort_values("deckungsbeitrag_pct").head(5)
    top_lecks = []
    for _, row in leck_sorted.iterrows():
        umsatz_val  = float(row.get("umsatz", 0)) if has_umsatz else 0.0
        db_ist      = float(row["deckungsbeitrag_pct"]) / 100 * umsatz_val
        db_ziel     = MINDESTMARGE / 100 * umsatz_val
        potenzial   = max(0.0, db_ziel - db_ist)

        rabatt_val  = float(row.get("rabatt_pct", 0))
        # Konkrete Empfehlung: Preiserhöhung oder Rabattkürzung
        fehlende_pp = MINDESTMARGE - float(row["deckungsbeitrag_pct"])
        empfehlung  = f"Preis um ≈{round(fehlende_pp * 0.9, 1)} % erhöhen"

        top_lecks.append({
            "angebot":      str(row.get(id_col, "—")),
            "kunde":        str(row.get("kunde", "—")),
            "produkt":      str(row.get("produkt", "—")),
            "rabatt":       round(rabatt_val, 1),
            "marge":        round(float(row["deckungsbeitrag_pct"]), 1),
            "umsatz":       round(umsatz_val, 0),
            "potenzial_eur": round(potenzial, 0),
            "empfehlung":   empfehlung,
        })
    result.top_lecks = top_lecks

    # ── Vertriebsmitarbeiter-Vergleich ────────────────────────────────────
    mitarb_potenzial_eur = 0.0
    if "vertriebsmitarbeiter" in df.columns:
        grp = df.groupby("vertriebsmitarbeiter").agg(
            avg_rabatt=("rabatt_pct", "mean"),
            avg_marge=("deckungsbeitrag_pct", "mean"),
            anzahl=("deckungsbeitrag_pct", "count"),
            leck_quote=(
                "deckungsbeitrag_pct",
                lambda x: round((x < MINDESTMARGE).mean() * 100, 1)
            ),
            umsatz_sum=("umsatz", "sum") if has_umsatz else ("deckungsbeitrag_pct", "count"),
        ).reset_index().round({"avg_rabatt": 1, "avg_marge": 1})

        grp_sorted = grp.sort_values("avg_rabatt", ascending=False)
        result.mitarbeiter_vergleich = grp_sorted.to_dict(orient="records")

        # Einsparpotenzial: Wenn der schwächste Mitarbeiter auf Benchmark-Niveau käme
        if len(grp_sorted) >= 2:
            worst = grp_sorted.iloc[0]
            best  = grp_sorted.iloc[-1]
            if has_umsatz:
                delta_marge_pp = float(best["avg_marge"]) - float(worst["avg_marge"])
                umsatz_worst   = float(worst["umsatz_sum"])
                mitarb_potenzial_eur = round(max(0.0, umsatz_worst * delta_marge_pp / 100), 0)
            result.kpi["mitarb_worst"]         = str(worst["vertriebsmitarbeiter"])
            result.kpi["mitarb_best"]          = str(best["vertriebsmitarbeiter"])
            result.kpi["mitarb_potenzial_eur"] = mitarb_potenzial_eur
            result.kpi["mitarb_delta_rabatt"]  = round(float(worst["avg_rabatt"]) - float(best["avg_rabatt"]), 1)
            result.kpi["mitarb_delta_marge"]   = round(float(best["avg_marge"]) - float(worst["avg_marge"]), 1)

    # ── Marge je Produkt ─────────────────────────────────────────────────
    if "produkt" in df.columns:
        grp_p = df.groupby("produkt").agg(
            avg_marge=("deckungsbeitrag_pct", "mean"),
            avg_rabatt=("rabatt_pct", "mean"),
            umsatz=("umsatz", "sum") if has_umsatz else ("deckungsbeitrag_pct", "count"),
            anzahl=("deckungsbeitrag_pct", "count"),
        ).reset_index().round({"avg_marge": 1, "avg_rabatt": 1})
        result.produkt_margen = grp_p.sort_values("avg_marge").to_dict(orient="records")

    # ── Kunden-Rabatte ────────────────────────────────────────────────────
    if "kunde" in df.columns:
        grp_k = df.groupby("kunde").agg(
            avg_rabatt=("rabatt_pct", "mean"),
            avg_marge=("deckungsbeitrag_pct", "mean"),
            umsatz=("umsatz", "sum") if has_umsatz else ("deckungsbeitrag_pct", "count"),
            anzahl=("deckungsbeitrag_pct", "count"),
        ).reset_index().round({"avg_rabatt": 1, "avg_marge": 1})
        result.kunden_rabatte = grp_k.sort_values("avg_rabatt", ascending=False).to_dict(orient="records")

    # ── Handlungsempfehlungen (plain-text Insights) ───────────────────────
    insights = []

    # Insight 1: Einsparpotenzial aus Margenlecks
    if len(lecks) > 0 and einsparpotenzial_eur > 0:
        insights.append({
            "typ":   "leck",
            "icon":  "🎯",
            "titel": f"{len(lecks)} Angebote unter Mindestmarge",
            "text":  (
                f"Durch Preiskorrektur auf die Mindestmarge von {MINDESTMARGE:.0f} % "
                f"ließen sich <strong>{_fmt_eur(einsparpotenzial_eur)}</strong> zusätzlicher "
                f"Deckungsbeitrag erzielen. Das entspricht dem wirtschaftlich sinnvollsten "
                f"kurzfristigen Hebel."
            ),
        })

    # Insight 2: Mitarbeiter-Delta
    if mitarb_potenzial_eur > 0 and "mitarb_worst" in result.kpi:
        insights.append({
            "typ":   "mitarbeiter",
            "icon":  "👤",
            "titel": f"{result.kpi['mitarb_worst']} rabattiert {result.kpi['mitarb_delta_rabatt']} % mehr als {result.kpi['mitarb_best']}",
            "text":  (
                f"Wenn <strong>{result.kpi['mitarb_worst']}</strong> auf das Rabattniveau von "
                f"<strong>{result.kpi['mitarb_best']}</strong> käme ({result.kpi['mitarb_delta_rabatt']} %-Punkte weniger), "
                f"entstünde ein Mehrertrag von ca. <strong>{_fmt_eur(mitarb_potenzial_eur)}</strong> Deckungsbeitrag "
                f"– ohne einen einzigen neuen Auftrag."
            ),
        })

    # Insight 3: Schwächstes Produkt
    if result.produkt_margen:
        worst_p = result.produkt_margen[0]
        if float(worst_p["avg_marge"]) < 25:
            insights.append({
                "typ":   "produkt",
                "icon":  "📦",
                "titel": f"{worst_p['produkt']} – niedrigste Produktmarge",
                "text":  (
                    f"<strong>{worst_p['produkt']}</strong> erzielt im Schnitt nur "
                    f"<strong>{worst_p['avg_marge']} %</strong> Deckungsbeitrag bei einem Ø Rabatt von "
                    f"{worst_p['avg_rabatt']} %. Prüfen Sie Preisliste oder Kostenbasis für dieses Produkt."
                ),
            })

    result.insights = insights
    return result
