"""
Generiert 3 Default-Datensätze für die Aurora Web App.
Wird einmalig beim ersten App-Start ausgeführt.
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent


def make_gastro_bier(seed=42):
    """Stündliche Gastronomie-Zeitreihe – Restaurant 'Zum Goldenen Löwen'"""
    rng = np.random.default_rng(seed)
    ts  = pd.date_range("2022-01-01 06:00", "2023-12-31 23:00", freq="h")

    H_PROFILE = {
        6:0.05, 7:0.08, 8:0.12, 9:0.10, 10:0.08, 11:0.15,
        12:0.45, 13:0.55, 14:0.35, 15:0.18, 16:0.15, 17:0.22,
        18:0.50, 19:0.70, 20:0.75, 21:0.65, 22:0.45, 23:0.25,
    }
    HOLIDAYS = {(1,1),(1,6),(5,1),(10,3),(11,1),(12,25),(12,26)}

    beer_l, food_o, rev_e, staff = [], [], [], []
    for t in ts:
        base = H_PROFILE.get(t.hour, 0.05)
        doy  = t.day_of_year
        we   = 1.0 + (0.6 if t.dayofweek >= 5 else 0.0)
        summ = 1.0 + 0.28 * max(0, np.sin(np.pi*(doy-100)/190)) * (120 < doy < 280)
        ofest = 1.9 if (t.month == 9 and t.day >= 17) or (t.month == 10 and t.day <= 3) else 1.0
        hol  = 1.35 if (t.month, t.day) in HOLIDAYS else 1.0
        trd  = 1.0 + 0.00012 * (t - ts[0]).days
        nz   = rng.normal(1.0, 0.08)
        f    = base * we * summ * ofest * hol * trd * nz

        beer_l.append(max(0, round(f * 18.0 + rng.normal(0, 0.4), 2)))
        food_o.append(max(0, int(   f * 22.0 + rng.normal(0, 1.2))))
        rev_e.append( max(0, round(beer_l[-1]*4.80 + food_o[-1]*12.50 + rng.normal(0,4), 2)))
        staff.append( max(1, int(   f *  6.5 + 1.5)))

    df = pd.DataFrame({
        "timestamp":   ts,
        "beer_liters": beer_l,
        "food_orders": food_o,
        "revenue_eur": rev_e,
        "staff_count": staff,
    })
    path = OUT / "gastro_bier.csv"
    df.to_csv(path, index=False)
    (OUT / "gastro_bier.meta.txt").write_text(
        "Restaurant 'Zum Goldenen Löwen' – Stündlicher Bierverbrauch, Speisen, Umsatz & Personal (2022–2023)",
        encoding="utf-8",
    )
    print(f"  OK  gastro_bier.csv   ({len(df):,} Zeilen)")
    return path


def make_supermarkt_energie(seed=7):
    """Stündlicher Stromverbrauch eines Supermarkts"""
    rng = np.random.default_rng(seed)
    ts  = pd.date_range("2022-01-01", "2023-12-31 23:00", freq="h")

    OPEN_H    = set(range(7, 21))    # Öffnungszeiten 07–20 Uhr
    OPEN_SA   = set(range(7, 19))
    HOLIDAYS  = {(1,1),(5,1),(10,3),(12,25),(12,26)}

    strom, kaelte, beleuch = [], [], []
    for t in ts:
        doy   = t.day_of_year
        is_we = t.dayofweek >= 5
        is_sa = t.dayofweek == 5
        is_ho = (t.month, t.day) in HOLIDAYS
        open_h = OPEN_SA if is_sa else OPEN_H
        is_open = t.hour in open_h and not (is_we and not is_sa) and not is_ho

        # Saisonalität: Kältebedarf im Sommer höher, Heizung im Winter
        summer = 1.0 + 0.45 * np.sin(np.pi * (doy - 80) / 183) * (80 < doy < 263)
        winter = 1.0 + 0.30 * np.cos(np.pi * (doy - 1)  / 180) * (doy < 60 or doy > 300)

        base_open  = 85.0 if is_open else 12.0
        base_kaelt = 30.0 * summer if is_open else 15.0 * summer
        base_bel   = 22.0 if is_open else 3.0

        nz = rng.normal(1.0, 0.06)
        strom.append(   max(5,  round((base_open  + base_kaelt + base_bel) * winter * nz, 2)))
        kaelte.append(  max(0,  round(base_kaelt * nz, 2)))
        beleuch.append( max(0,  round(base_bel   * nz, 2)))

    df = pd.DataFrame({
        "timestamp":        ts,
        "stromverbrauch_kwh": strom,
        "kaelteverbrauch_kwh": kaelte,
        "beleuchtung_kwh":    beleuch,
    })
    path = OUT / "supermarkt_energie.csv"
    df.to_csv(path, index=False)
    (OUT / "supermarkt_energie.meta.txt").write_text(
        "Supermarkt Energieverbrauch – Stündlich Strom, Kälte & Beleuchtung (2022–2023)",
        encoding="utf-8",
    )
    print(f"  OK  supermarkt_energie.csv  ({len(df):,} Zeilen)")
    return path


def make_ecommerce_umsatz(seed=13):
    """Täglicher E-Commerce Umsatz"""
    rng  = np.random.default_rng(seed)
    ts   = pd.date_range("2021-01-01", "2023-12-31", freq="D")

    HOLIDAYS = {
        (1,1),(12,24),(12,25),(12,26),(10,3),(5,1),(1,6),(11,1),
    }
    # Cyber-Monday-Woche (letzter Montag November)
    cyber_dates = set()
    for yr in [2021, 2022, 2023]:
        nov = pd.date_range(f"{yr}-11-01", f"{yr}-11-30", freq="D")
        mondays = [d for d in nov if d.dayofweek == 0]
        last_mo = mondays[-1]
        for i in range(-2, 7):
            cyber_dates.add((last_mo + pd.Timedelta(days=i)).strftime("%m-%d"))

    umsatz, bestellungen, neue_kunden = [], [], []
    for t in ts:
        doy  = t.day_of_year
        dow  = t.dayofweek

        # Trend: 15% Wachstum pro Jahr
        trd = 1.0 + 0.15 * (t - ts[0]).days / 365

        # Saisonalität: Q4 stark (Black Friday, Weihnachten)
        q4_boost = 1.0
        if t.month == 11:
            q4_boost = 1.8
        if t.month == 12 and t.day <= 23:
            q4_boost = 2.4
        # Sommer-Delle
        sommer_delle = 0.85 if 175 < doy < 230 else 1.0

        # Cyber-Monday
        cyber = 3.2 if t.strftime("%m-%d") in cyber_dates else 1.0

        # Wochentag: Mo-Do stärker als Fr, schwach am WE
        we_fak = {0:1.1, 1:1.15, 2:1.1, 3:1.0, 4:0.85, 5:0.6, 6:0.55}[dow]

        # Feiertag
        hol = 0.3 if (t.month, t.day) in HOLIDAYS else 1.0

        base = 12_500
        f = base * trd * q4_boost * sommer_delle * cyber * we_fak * hol

        nz = rng.normal(1.0, 0.1)
        u  = max(0, round(f * nz, 2))
        b  = max(0, int(u / rng.uniform(38, 52)))
        nk = max(0, int(b * rng.uniform(0.12, 0.22)))

        umsatz.append(u)
        bestellungen.append(b)
        neue_kunden.append(nk)

    df = pd.DataFrame({
        "datum":        ts.strftime("%Y-%m-%d"),
        "umsatz_eur":   umsatz,
        "bestellungen": bestellungen,
        "neue_kunden":  neue_kunden,
    })
    path = OUT / "ecommerce_umsatz.csv"
    df.to_csv(path, index=False)
    (OUT / "ecommerce_umsatz.meta.txt").write_text(
        "Online-Shop Tagesumsatz – Umsatz EUR, Bestellungen & Neukunden (2021–2023)",
        encoding="utf-8",
    )
    print(f"  OK  ecommerce_umsatz.csv    ({len(df):,} Zeilen)")
    return path


if __name__ == "__main__":
    print("Generiere Default-Datensätze …")
    make_gastro_bier()
    make_supermarkt_energie()
    make_ecommerce_umsatz()
    print("Fertig.")
