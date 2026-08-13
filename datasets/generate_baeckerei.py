"""
Synthetischer Bäckerei-Datensatz für LightGBM Nachfrageprognose.
Zeitraum: 2024-08-01 bis 2026-08-05 (≈ 2 Jahre).
5 Filialen × 8 Artikel × tägliche Granularität.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

DATASETS_DIR = Path(__file__).parent

FILIALEN = ["Innenstadt", "Arkaden", "Bahnhof", "Penny-Markt", "Ferno-Center"]
ARTIKEL   = ["Roggenbrot", "Vollkornbrot", "Brötchen", "Croissant",
              "Butterkuchen", "Käsekuchen", "Laugenstange", "Mohnschnecke"]

# Tagesfrisch: Basismengen bei normalem Wochentag (Innenstadt als Referenz)
BASE_VOLUMES = {
    "Roggenbrot":   20,
    "Vollkornbrot": 15,
    "Brötchen":    120,
    "Croissant":    35,
    "Butterkuchen": 25,
    "Käsekuchen":   18,
    "Laugenstange": 40,
    "Mohnschnecke": 22,
}

FILIALE_MULT = {
    "Innenstadt":  1.00,
    "Arkaden":     1.30,   # Einkaufszentrum, mehr Laufkundschaft
    "Bahnhof":     0.80,
    "Penny-Markt": 0.60,
    "Ferno-Center":0.90,
}

# 0=Montag … 6=Sonntag
DOW_MULT = {0: 0.85, 1: 0.88, 2: 0.90, 3: 0.95, 4: 1.15, 5: 1.38, 6: 0.68}

MONTH_MULT = {
    1: 0.90, 2: 0.88, 3: 0.95, 4: 1.00,
    5: 0.98, 6: 0.92, 7: 0.86, 8: 0.84,   # Sommer schwächer
    9: 0.95, 10: 1.05, 11: 1.07, 12: 1.22, # Weihnachten stark
}

# NRW Feiertage 2024–2026
NRW_HOLIDAYS = {
    # 2024
    date(2024,  1,  1), date(2024,  3, 29), date(2024,  4,  1),
    date(2024,  5,  1), date(2024,  5,  9), date(2024,  5, 20),
    date(2024,  5, 30), date(2024, 10,  3), date(2024, 11,  1),
    date(2024, 12, 25), date(2024, 12, 26),
    # 2025
    date(2025,  1,  1), date(2025,  4, 18), date(2025,  4, 21),
    date(2025,  5,  1), date(2025,  5, 29), date(2025,  6,  9),
    date(2025,  6, 19), date(2025, 10,  3), date(2025, 11,  1),
    date(2025, 12, 25), date(2025, 12, 26),
    # 2026
    date(2026,  1,  1), date(2026,  4,  3), date(2026,  4,  6),
    date(2026,  5,  1), date(2026,  5, 14), date(2026,  5, 25),
    date(2026,  6,  4), date(2026, 10,  3), date(2026, 11,  1),
    date(2026, 12, 25), date(2026, 12, 26),
}

# NRW Schulferien 2024–2026 (Näherungswerte)
SCHOOL_HOLIDAYS = [
    # 2024
    (date(2024,  7, 22), date(2024,  9,  3)),   # Sommerferien
    (date(2024, 10, 14), date(2024, 10, 26)),
    (date(2024, 12, 23), date(2025,  1,  7)),
    # 2025
    (date(2025,  2, 17), date(2025,  2, 21)),
    (date(2025,  4, 14), date(2025,  4, 26)),
    (date(2025,  5, 30), date(2025,  6,  6)),
    (date(2025,  7,  7), date(2025,  8, 19)),   # Sommerferien 2025
    (date(2025, 10, 13), date(2025, 10, 25)),
    (date(2025, 12, 22), date(2026,  1,  7)),
    # 2026
    (date(2026,  2, 16), date(2026,  2, 20)),
    (date(2026,  4,  7), date(2026,  4, 18)),
    (date(2026,  5, 22), date(2026,  5, 29)),
    (date(2026,  7,  6), date(2026,  8, 18)),   # Sommerferien 2026 → deckt Prognose-Woche ab
]


def _is_schulferien(d: date) -> bool:
    return any(s <= d <= e for s, e in SCHOOL_HOLIDAYS)


def _synthetic_temp(d: date) -> float:
    """Jahresgang NRW-Temperatur (°C), reproduzierbar per Tag."""
    doy  = d.timetuple().tm_yday
    base = 10.0 + 12.0 * np.sin((doy - 80) / 365.0 * 2 * np.pi)
    rng  = np.random.default_rng(d.toordinal())
    return float(base + rng.normal(0, 2.5))


def _temp_mult(temp: float) -> float:
    """Heiße Sommer → weniger Kauf von Brot/Gebäck; kalte Winter → mehr."""
    if temp > 28:
        return max(0.78, 1.0 - (temp - 28) * 0.018)
    if temp > 22:
        return 1.0 - (temp - 22) * 0.010
    if temp < 2:
        return min(1.12, 1.0 + (2 - temp) * 0.010)
    return 1.0


def generate() -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    rows  = []
    start = date(2024, 8, 1)
    end   = date(2026, 8, 5)

    current = start
    while current <= end:
        dow         = current.weekday()
        month       = current.month
        is_feiertag = current in NRW_HOLIDAYS
        is_schulfer = _is_schulferien(current)
        temp        = _synthetic_temp(current)
        temp_m      = _temp_mult(temp)
        feiertag_m  = 1.40 if is_feiertag else 1.0   # Feiertags-Früheinsatz
        schulfer_m  = 0.93 if is_schulfer  else 1.0   # leicht weniger Familienbedarf

        for filiale in FILIALEN:
            for artikel in ARTIKEL:
                mu = (BASE_VOLUMES[artikel]
                      * FILIALE_MULT[filiale]
                      * DOW_MULT[dow]
                      * MONTH_MULT[month]
                      * temp_m
                      * feiertag_m
                      * schulfer_m)

                # ~5 % Aktionstage – deterministisch, alle Wochentage möglich
                is_aktion = 1 if (hash((current.toordinal(), filiale, artikel)) % 20 == 0) else 0
                if is_aktion:
                    mu *= 1.32

                menge = int(rng.poisson(max(1, mu)))

                rows.append({
                    "datum":           current.isoformat(),
                    "filiale":         filiale,
                    "artikel":         artikel,
                    "menge":           menge,
                    "temperatur":      round(temp, 1),
                    "is_feiertag_nrw": int(is_feiertag),
                    "is_schulferien":  int(is_schulfer),
                    "is_aktion":       is_aktion,
                })

        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    df.to_csv(DATASETS_DIR / "baeckerei_beispiel.csv", index=False)
    n_days = (end - start).days + 1
    print(f"Bäckerei-Datensatz: {len(df):,} Zeilen · {n_days} Tage · {start} → {end} → baeckerei_beispiel.csv")
    return df


if __name__ == "__main__":
    generate()
