"""
Synthetischer Bäckerei-Datensatz für LightGBM Nachfrageprognose.
Modelliert 5 Filialen × 8 Artikel über 2 Jahre (2023-01-01 bis 2025-01-04).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

DATASETS_DIR = Path(__file__).parent

FILIALEN = ["Innenstadt", "Arkaden", "Bahnhof", "Penny-Markt", "Ferno-Center"]
ARTIKEL   = ["Roggenbrot", "Vollkornbrot", "Brötchen", "Croissant",
              "Butterkuchen", "Käsekuchen", "Laugenstange", "Mohnschnecke"]

BASE_VOLUMES = {
    "Roggenbrot":  20,
    "Vollkornbrot": 15,
    "Brötchen":    120,
    "Croissant":    35,
    "Butterkuchen": 25,
    "Käsekuchen":   18,
    "Laugenstange": 40,
    "Mohnschnecke": 22,
}

FILIALE_MULT = {
    "Innenstadt": 1.00,
    "Arkaden":    1.30,
    "Bahnhof":    0.80,
    "Penny-Markt":0.60,
    "Ferno-Center":0.90,
}

# 0=Montag … 6=Sonntag
DOW_MULT = {0: 0.85, 1: 0.90, 2: 0.90, 3: 0.95, 4: 1.15, 5: 1.35, 6: 0.70}

MONTH_MULT = {
    1: 0.90, 2: 0.88, 3: 0.95, 4: 1.00,
    5: 0.98, 6: 0.93, 7: 0.88, 8: 0.85,
    9: 0.95, 10: 1.05, 11: 1.05, 12: 1.20,
}

NRW_HOLIDAYS = {
    date(2023, 1,  1), date(2023, 4,  7), date(2023, 4, 10),
    date(2023, 5,  1), date(2023, 5, 18), date(2023, 5, 29),
    date(2023, 6,  8), date(2023, 10, 3), date(2023, 11, 1),
    date(2023, 12, 25), date(2023, 12, 26),
    date(2024, 1,  1), date(2024, 3, 29), date(2024, 4,  1),
    date(2024, 5,  1), date(2024, 5,  9), date(2024, 5, 20),
    date(2024, 5, 30), date(2024, 10, 3), date(2024, 11, 1),
    date(2024, 12, 25), date(2024, 12, 26),
}

SCHOOL_HOLIDAYS = [
    (date(2023, 1, 30), date(2023, 2,  3)),
    (date(2023, 4,  3), date(2023, 4, 15)),
    (date(2023, 5, 30), date(2023, 6,  2)),
    (date(2023, 6, 29), date(2023, 8, 11)),
    (date(2023, 10, 2), date(2023, 10, 14)),
    (date(2023, 12, 27), date(2024, 1,  6)),
    (date(2024, 2, 12), date(2024, 2, 16)),
    (date(2024, 3, 25), date(2024, 4,  6)),
    (date(2024, 5, 21), date(2024, 5, 24)),
    (date(2024, 7, 22), date(2024, 9,  3)),
    (date(2024, 10, 14), date(2024, 10, 26)),
    (date(2024, 12, 23), date(2024, 12, 31)),
]


def _is_schulferien(d: date) -> bool:
    return any(s <= d <= e for s, e in SCHOOL_HOLIDAYS)


def _synthetic_temp(d: date) -> float:
    """Jahresgang Temperatur NRW (°C), reproduzierbar per Tag."""
    doy = d.timetuple().tm_yday
    base = 10.0 + 12.0 * np.sin((doy - 80) / 365.0 * 2 * np.pi)
    rng = np.random.default_rng(d.toordinal())
    return float(base + rng.normal(0, 2.5))


def _temp_mult(temp: float) -> float:
    if temp > 25:
        return max(0.80, 1.0 - (temp - 25) * 0.015)
    if temp < 5:
        return min(1.10, 1.0 + (5 - temp) * 0.008)
    return 1.0


def generate() -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    rows  = []
    start = date(2023, 1, 1)
    end   = date(2025, 1, 4)

    current = start
    while current <= end:
        dow          = current.weekday()
        month        = current.month
        is_feiertag  = current in NRW_HOLIDAYS
        is_schulfer  = _is_schulferien(current)
        temp         = _synthetic_temp(current)
        temp_m       = _temp_mult(temp)
        feiertag_m   = 1.35 if is_feiertag else 1.0
        schulfer_m   = 0.94 if is_schulfer  else 1.0

        for filiale in FILIALEN:
            for artikel in ARTIKEL:
                mu = (BASE_VOLUMES[artikel]
                      * FILIALE_MULT[filiale]
                      * DOW_MULT[dow]
                      * MONTH_MULT[month]
                      * temp_m
                      * feiertag_m
                      * schulfer_m)

                # ~5 % Aktionstage, deterministisch
                is_aktion = 1 if (hash((current.toordinal(), filiale, artikel)) % 20 == 0) else 0
                if is_aktion:
                    mu *= 1.30

                menge = int(rng.poisson(max(1, mu)))

                rows.append({
                    "datum":          current.isoformat(),
                    "filiale":        filiale,
                    "artikel":        artikel,
                    "menge":          menge,
                    "temperatur":     round(temp, 1),
                    "is_feiertag_nrw": int(is_feiertag),
                    "is_schulferien": int(is_schulfer),
                    "is_aktion":      is_aktion,
                })

        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    df.to_csv(DATASETS_DIR / "baeckerei_beispiel.csv", index=False)
    print(f"Bäckerei-Datensatz: {len(df):,} Zeilen → baeckerei_beispiel.csv")
    return df


if __name__ == "__main__":
    generate()
