"""
Entity Resolution – Ähnlichkeitssuche im Firmenstamm.

Ansatz:
  1. Blocking: Nur Firmen mit gleicher PLZ-Prefix (erste 2 Ziffern) oder gleichem Anfangsbuchstaben vergleichen
  2. Name-Ähnlichkeit: difflib.SequenceMatcher (Levenshtein-ähnlich)
  3. Adress-Score: PLZ-Übereinstimmung (hart), Ort-Ähnlichkeit
  4. Gesamtscore = 0.6 × Name + 0.3 × Adresse + 0.1 × Branche
  5. Flag > 0.75 als wahrscheinliches Duplikat
"""
from __future__ import annotations
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import pandas as pd

SCHWELLENWERT = 0.75


def _name_score(a: str, b: str) -> float:
    """Normierter Ähnlichkeits-Score für Firmennamen (0–1)."""
    a_clean = _clean_name(a)
    b_clean = _clean_name(b)
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def _clean_name(name: str) -> str:
    """Rechtsform-Suffixe entfernen, Umlaute normalisieren, lowercase."""
    suffixe = ["gmbh & co. kg", "gmbh & co kg", "gmbh co kg",
                "ag & co. kg", "ag & co kg",
                "gmbh", "ag", "kg", "ohg", "gbr", "se",
                "und söhne", "& söhne", "& co.", "& co"]
    n = name.lower()
    for s in suffixe:
        n = n.replace(s, "")
    n = (n.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae")
           .replace("ß", "ss").replace("-", " ").strip())
    return " ".join(n.split())  # mehrfache Leerzeichen kollabieren


def _addr_score(row_a: pd.Series, row_b: pd.Series) -> float:
    """Adress-Ähnlichkeit: PLZ + Ort."""
    plz_match = 1.0 if str(row_a["plz"]) == str(row_b["plz"]) else (
        0.5 if str(row_a["plz"])[:2] == str(row_b["plz"])[:2] else 0.0
    )
    ort_match = SequenceMatcher(None,
                                str(row_a["ort"]).lower(),
                                str(row_b["ort"]).lower()).ratio()
    return 0.6 * plz_match + 0.4 * ort_match


def _branche_score(row_a: pd.Series, row_b: pd.Series) -> float:
    return 1.0 if str(row_a.get("branche", "")) == str(row_b.get("branche", "")) else 0.0


def _gesamtscore(ns: float, as_: float, bs: float) -> float:
    return round(0.6 * ns + 0.3 * as_ + 0.1 * bs, 4)


def _ampel(score: float) -> str:
    if score >= SCHWELLENWERT:
        return "rot"
    if score >= 0.55:
        return "gelb"
    return "grün"


def _match_grund(ns: float, as_: float, bs: float) -> str:
    gründe = []
    if ns >= 0.80:
        gründe.append(f"Name sehr ähnlich ({ns*100:.0f}%)")
    elif ns >= 0.60:
        gründe.append(f"Name ähnlich ({ns*100:.0f}%)")
    if as_ >= 0.80:
        gründe.append("gleiche Adresse")
    elif as_ >= 0.50:
        gründe.append("ähnliche Adresse")
    if bs == 1.0:
        gründe.append("gleiche Branche")
    return " · ".join(gründe) if gründe else "Geringe Übereinstimmung"


@dataclass
class RetrievalResult:
    kpi: dict                          = field(default_factory=dict)
    suchergebnisse: list[dict]         = field(default_factory=list)  # Treffer für Suchanfrage
    duplikate: list[dict]              = field(default_factory=list)  # Auto-erkannte Duplikate
    suchbegriff: str                   = ""


def analyse(df: pd.DataFrame,
            suchbegriff: str | None = None) -> RetrievalResult:

    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Pflichtfelder sicherstellen
    for col in ["firma_id", "firma_name", "plz", "ort"]:
        if col not in df.columns:
            df[col] = ""
    if "branche" not in df.columns:
        df["branche"] = ""
    if "strasse" not in df.columns:
        df["strasse"] = ""

    result = RetrievalResult()

    # ── KPI ───────────────────────────────────────────────────────────────
    result.kpi = {
        "n_firmen":   len(df),
        "n_branchen": df["branche"].nunique(),
        "n_orte":     df["ort"].nunique(),
        "schwelle":   int(SCHWELLENWERT * 100),
    }

    # ── Freie Textsuche ────────────────────────────────────────────────────
    if suchbegriff and suchbegriff.strip():
        result.suchbegriff = suchbegriff.strip()
        treffer = []
        for _, row in df.iterrows():
            ns = _name_score(suchbegriff, row["firma_name"])
            as_ = _addr_score(
                pd.Series({"plz": "", "ort": ""}), row
            )
            bs = 0.0
            score = round(0.7 * ns + 0.3 * as_, 4)  # bei Freisuche: Name dominiert
            if score >= 0.30:
                treffer.append({
                    "firma_id":   row["firma_id"],
                    "firma_name": row["firma_name"],
                    "strasse":    row.get("strasse", ""),
                    "plz":        row["plz"],
                    "ort":        row["ort"],
                    "branche":    row.get("branche", ""),
                    "name_score": round(ns * 100, 1),
                    "score":      round(score * 100, 1),
                    "ampel":      _ampel(score),
                    "match_grund": _match_grund(ns, 0.0, 0.0),
                })
        treffer.sort(key=lambda x: -x["score"])
        result.suchergebnisse = treffer[:20]

    # ── Duplikat-Erkennung (Blocking) ─────────────────────────────────────
    duplikate = []
    seen = set()
    n = len(df)

    for i in range(n):
        row_a = df.iloc[i]
        plz_a  = str(row_a["plz"])
        ini_a  = _clean_name(str(row_a["firma_name"]))[:1]

        for j in range(i + 1, n):
            row_b = df.iloc[j]
            plz_b  = str(row_b["plz"])
            ini_b  = _clean_name(str(row_b["firma_name"]))[:1]

            # Blocking: gleiche PLZ-Vorwahl (2 Ziffern) ODER gleicher Anfangsbuchstabe
            if plz_a[:2] != plz_b[:2] and ini_a != ini_b:
                continue

            pair_key = tuple(sorted([row_a["firma_id"], row_b["firma_id"]]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            ns  = _name_score(str(row_a["firma_name"]), str(row_b["firma_name"]))
            as_ = _addr_score(row_a, row_b)
            bs  = _branche_score(row_a, row_b)
            score = _gesamtscore(ns, as_, bs)

            if score >= 0.55:  # auch "prüfen"-Kandidaten aufnehmen
                duplikate.append({
                    "firma_a_id":   row_a["firma_id"],
                    "firma_a_name": row_a["firma_name"],
                    "firma_a_ort":  f"{row_a['plz']} {row_a['ort']}",
                    "firma_a_br":   row_a.get("branche", ""),
                    "firma_b_id":   row_b["firma_id"],
                    "firma_b_name": row_b["firma_name"],
                    "firma_b_ort":  f"{row_b['plz']} {row_b['ort']}",
                    "firma_b_br":   row_b.get("branche", ""),
                    "name_score":   round(ns * 100, 1),
                    "addr_score":   round(as_ * 100, 1),
                    "gesamt_score": round(score * 100, 1),
                    "ampel":        _ampel(score),
                    "match_grund":  _match_grund(ns, as_, bs),
                })

    duplikate.sort(key=lambda x: -x["gesamt_score"])
    result.duplikate = duplikate[:40]  # max. 40 Paare anzeigen

    result.kpi["n_duplikate_rot"]  = sum(1 for d in duplikate if d["ampel"] == "rot")
    result.kpi["n_duplikate_gelb"] = sum(1 for d in duplikate if d["ampel"] == "gelb")

    return result
