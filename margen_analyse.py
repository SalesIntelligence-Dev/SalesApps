"""Deskriptive Margenleckage- und Rabattanalyse. Nur pandas, kein ML."""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class MargenResult:
    kpi: dict                           = field(default_factory=dict)
    top_lecks: list[dict]               = field(default_factory=list)
    mitarbeiter_vergleich: list[dict]   = field(default_factory=list)
    produkt_margen: list[dict]          = field(default_factory=list)
    kunden_rabatte: list[dict]          = field(default_factory=list)
    zeilen_gesamt: int                  = 0


MINDESTMARGE = 15.0   # %


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Bringt flexibel benannte Spalten auf kanonische Namen."""
    rename = {}
    cols_lower = {c.lower(): c for c in df.columns}

    _alias = {
        "angebot_id":           ["angebot_id", "angebotsnummer", "id", "nr"],
        "datum":                ["datum", "date", "Datum"],
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
            # Rohkosten ≈ 55 % des Listenpreises (branchenübliche Schätzung)
            herstellkosten = df["listenpreis"] * 0.55
            df["deckungsbeitrag_pct"] = (
                (df["verkaufspreis"] - herstellkosten) / df["verkaufspreis"].replace(0, float("nan")) * 100
            ).round(1)
        elif "rabatt_pct" in df.columns:
            df["deckungsbeitrag_pct"] = (30 - df["rabatt_pct"]).round(1)
        else:
            df["deckungsbeitrag_pct"] = 25.0

    # rabatt_pct aus Preisen berechnen, wenn nicht vorhanden
    if "rabatt_pct" not in df.columns:
        if "listenpreis" in df.columns and "verkaufspreis" in df.columns:
            df["rabatt_pct"] = (
                (df["listenpreis"] - df["verkaufspreis"]) / df["listenpreis"].replace(0, float("nan")) * 100
            ).clip(0, 100).round(1)
        else:
            df["rabatt_pct"] = 0.0

    if "umsatz" not in df.columns and "verkaufspreis" in df.columns and "menge" in df.columns:
        df["umsatz"] = (df["verkaufspreis"] * df["menge"]).round(2)

    return df


def analyse(df: pd.DataFrame) -> MargenResult:
    df = _normalise(df.copy())

    result = MargenResult(zeilen_gesamt=len(df))

    # ── KPI-Kacheln ────────────────────────────────────────────────────────
    avg_rabatt     = float(df["rabatt_pct"].mean()) if "rabatt_pct" in df.columns else 0.0
    avg_marge      = float(df["deckungsbeitrag_pct"].mean())
    lecks          = df[df["deckungsbeitrag_pct"] < MINDESTMARGE]
    leck_volumen   = float(lecks["umsatz"].sum()) if "umsatz" in df.columns else float(len(lecks))
    ausreisser     = int(
        (df["rabatt_pct"] > df["rabatt_pct"].quantile(0.90)).sum()
    ) if "rabatt_pct" in df.columns else 0

    result.kpi = {
        "avg_rabatt":   round(avg_rabatt, 1),
        "avg_marge":    round(avg_marge, 1),
        "leck_volumen": round(leck_volumen, 0),
        "ausreisser":   ausreisser,
        "leck_anzahl":  len(lecks),
        "gesamt":       len(df),
    }

    # ── Top-5 Margenlecks ─────────────────────────────────────────────────
    id_col = "angebot_id" if "angebot_id" in df.columns else df.columns[0]
    leck_sorted = lecks.sort_values("deckungsbeitrag_pct").head(5)
    top_lecks = []
    for _, row in leck_sorted.iterrows():
        fehlende_marge = MINDESTMARGE - float(row["deckungsbeitrag_pct"])
        empfehlung = (
            f"Rabatt um {round(fehlende_marge * 0.8, 1)} % reduzieren"
            if "rabatt_pct" in df.columns
            else "Preis überprüfen"
        )
        top_lecks.append({
            "angebot":    str(row.get(id_col, "—")),
            "kunde":      str(row.get("kunde", "—")),
            "produkt":    str(row.get("produkt", "—")),
            "rabatt":     round(float(row.get("rabatt_pct", 0)), 1),
            "marge":      round(float(row["deckungsbeitrag_pct"]), 1),
            "umsatz":     round(float(row.get("umsatz", 0)), 0),
            "empfehlung": empfehlung,
        })
    result.top_lecks = top_lecks

    # ── Vertriebsmitarbeiter-Vergleich ────────────────────────────────────
    if "vertriebsmitarbeiter" in df.columns:
        grp = df.groupby("vertriebsmitarbeiter").agg(
            avg_rabatt=("rabatt_pct", "mean"),
            avg_marge=("deckungsbeitrag_pct", "mean"),
            anzahl=("deckungsbeitrag_pct", "count"),
            leck_quote=(
                "deckungsbeitrag_pct",
                lambda x: round((x < MINDESTMARGE).mean() * 100, 1)
            ),
        ).reset_index().round(1)
        result.mitarbeiter_vergleich = grp.sort_values("avg_rabatt", ascending=False).to_dict(orient="records")

    # ── Marge je Produkt ─────────────────────────────────────────────────
    if "produkt" in df.columns:
        grp_p = df.groupby("produkt").agg(
            avg_marge=("deckungsbeitrag_pct", "mean"),
            avg_rabatt=("rabatt_pct", "mean"),
            umsatz=("umsatz", "sum"),
            anzahl=("deckungsbeitrag_pct", "count"),
        ).reset_index().round(1)
        result.produkt_margen = grp_p.sort_values("avg_marge").to_dict(orient="records")

    # ── Kunden-Rabatte ────────────────────────────────────────────────────
    if "kunde" in df.columns:
        grp_k = df.groupby("kunde").agg(
            avg_rabatt=("rabatt_pct", "mean"),
            avg_marge=("deckungsbeitrag_pct", "mean"),
            umsatz=("umsatz", "sum"),
            anzahl=("deckungsbeitrag_pct", "count"),
        ).reset_index().round(1)
        result.kunden_rabatte = grp_k.sort_values("avg_rabatt", ascending=False).to_dict(orient="records")

    return result
