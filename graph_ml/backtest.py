"""
Rückblick (Stufe 3) – zeitlich strikt getrennter Backtest
=========================================================
RelBench-Stil: monatliche Stichtage (Snapshots). Merkmale je Stichtag
ausschließlich aus Ereignissen VOR dem Stichtag, Labels aus dem Fenster
danach. Training nur auf Stichtagen, deren Label-Fenster vor dem
Test-Stichtag endet; Validierung = letzter Trainings-Stichtag.

  Test-Stichtag  T_test = 2011-09-01, Label-Fenster bis 2011-11-30
  Training       monatlich ab 2010-03-01, letzter = 2011-06-01

Use Cases: churn · lead_scoring · cross_selling · returns_root_cause
(binär) · recommendation (Rangliste) · demand_forecast (Zeitreihen).
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

T_TEST = pd.Timestamp("2011-09-01")
TRAIN_START = pd.Timestamp("2010-03-01")
SVD_REF_END = pd.Timestamp("2010-03-01")      # feste Artikel-Faktoren aus der Zeit vor allen Stichtagen
HORIZON = 90
HISTORY = 365
SEED = 7
TOP_SHARE = 0.10

BINARY_UCS = ("churn", "lead_scoring", "cross_selling", "returns_root_cause", "account_scoring", "territory_coverage")

# Zeitachse je Datensatz: (Test-Stichtag, erster Trainings-Stichtag, Ende des SVD-Referenzfensters)
DATASET_DATES = {"v0": ("2011-09-01", "2010-03-01", "2010-03-01"),
                 "wwi": ("2016-02-01", "2014-01-01", "2014-01-01"),
                 "aw": ("2025-03-01", "2023-06-01", "2023-06-01")}
DATASET_OF = {"account_scoring": "wwi", "territory_coverage": "aw", "price_recommendation": "aw", "price_leakage": "aw",
              "quote_check": "aw"}
DATASET_NAMES = {"v0": "online_retail_ii", "wwi": "wwi", "aw": "adventureworks_data"}          # Use Cases, die auf einem anderen Datensatz gerechnet werden
_TIME_LOCK = threading.RLock()


@contextmanager
def dataset_time(key: str):
    """Stichtage eines Datensatzes für die Dauer eines Laufs setzen (Builder und Engines lesen sie global)."""
    global T_TEST, TRAIN_START, SVD_REF_END
    dates = DATASET_DATES.get(key) or DATASET_DATES["v0"]
    with _TIME_LOCK:
        old = (T_TEST, TRAIN_START, SVD_REF_END)
        if key in DATASET_DATES and key != "v0":
            T_TEST, TRAIN_START, SVD_REF_END = (pd.Timestamp(x) for x in dates)
        try:
            yield
        finally:
            T_TEST, TRAIN_START, SVD_REF_END = old


def dataset_key(uc: str, key: str = "v0") -> str:
    return DATASET_OF.get(uc, key) if key == "v0" else key
# Churn: Hebel ist „Umsatz halten“ → umsatzgewichteter Recall der Top-10-%-Liste (Recall bleibt sichtbar)
MAIN_METRIC = {"churn": "value_captured_at_10", "lead_scoring": "auprc", "cross_selling": "recall_at_10",
               "returns_root_cause": "recall_at_10", "recommendation": "map_at_12", "demand_forecast": "wape",
               "account_scoring": "auprc", "territory_coverage": "value_captured_at_10"}
LOWER_IS_BETTER = {"wape", "mase"}

FEATURE_LABELS = {
    "recency": "Tage seit letztem Beleg", "freq_30": "Belege 30 Tage", "freq_90": "Belege 90 Tage",
    "freq_365": "Belege 365 Tage", "monetary_90": "Umsatz 90 Tage", "monetary_365": "Umsatz 365 Tage",
    "avg_basket": "Ø Warenkorb", "tenure": "Kundentreue (Tage seit Erstkauf)", "n_products": "Artikel 365 Tage",
    "storno_cnt": "Stornos 365 Tage", "storno_share": "Storno-Anteil Umsatz", "trend_orders": "Trend Belege",
    "trend_revenue": "Trend Umsatz", "gap_mean": "Ø Tage zwischen Belegen", "gap_std": "Streuung Abstände",
    "overdue": "Überfälligkeit (Recency / Ø Abstand)", "country": "Land", "month": "Monat",
    "n_groups": "Artikelgruppen gekauft", "first_value": "Wert Erstkauf", "first_lines": "Positionen Erstkauf",
    "first_products": "Artikel Erstkauf", "first_avg_price": "Ø Preis Erstkauf", "first_storno": "Storno nach Erstkauf",
    "days_since_first": "Tage seit Erstkauf",
    # Graph
    "g_nb_inactive": "Graph: inaktive Nachbarkunden (über gemeinsame Artikel)",
    "g_nb_size": "Graph: Anzahl Nachbarkunden (log)", "g_prod_pop": "Graph: Ø Beliebtheit der Artikel",
    "g_pagerank": "Graph: PageRank im Kunde–Artikel-Graph", "g_repeat_prop": "Graph: Wiederkaufquote ähnlicher Erstkäufer",
    "g_country_repeat": "Graph: Wiederkaufquote im Land", "g_affinity_max": "Graph: max. Co-Kauf-Affinität zur Gruppe",
    "g_affinity_mean": "Graph: Ø Co-Kauf-Affinität zur Gruppe", "g_prod_return": "Graph: Retourenquote des Artikels",
    "g_nb_return": "Graph: Retourenquote der Nachbarartikel", "g_cust_return": "Graph: Retourenquote des Kunden",
    # Paare / Positionen
    "group_pop": "Beliebtheit der Gruppe (90 Tage)", "group_growth": "Wachstum der Gruppe", "group": "Artikelgruppe",
    "qty": "Menge", "price": "Preis", "value": "Positionswert", "price_rel": "Preis relativ zum Artikel-Median",
    "basket_lines": "Positionen im Beleg", "new_for_customer": "Artikel neu für Kunden", "new_product": "Neuer Artikel",
    "prod_sales": "Verkäufe des Artikels (log)", "cust_orders": "Belege des Kunden",
}
FEATURE_LABELS.update({
    "g_portfolio_active30": "Graph: Anteil Portfolio-Händler mit Auftrag (30 T.)",
    "g_portfolio_active90": "Graph: Anteil Portfolio-Händler mit Auftrag (90 T.)",
    "g_territory_active30": "Graph: Anteil Gebiets-Händler mit Auftrag (30 T.)",
    "g_portfolio_growth": "Graph: Wachstum im Vertriebler-Portfolio", "salesperson": "Vertriebler",
    "g_peer_group_growth": "Graph: Wachstum der Einkaufsgemeinschaft (Peers)",
    "g_peer_category_growth": "Graph: Wachstum der Kundenkategorie (Peers)",
    "g_peer_region_growth": "Graph: Wachstum der Region (Peers)",
    "buying_group": "Einkaufsgemeinschaft", "customer_category": "Kundenkategorie", "credit_limit": "Kreditlimit",
})
for i in range(8):
    FEATURE_LABELS[f"g_svd_{i}"] = f"Graph: Co-Kauf-Einbettung {i + 1}"


# ── Basisdaten ────────────────────────────────────────────────────────────────

@dataclass
class Base:
    lines: pd.DataFrame          # Käufe mit Kunde: cust, inv, time, prod, qty, price, value, country, group
    inv: pd.DataFrame            # je Beleg: cust, inv, time, value, n_lines
    storno: pd.DataFrame         # Stornos mit Kunde: cust, time, prod, value
    n_prod: int
    n_country: int
    prod_group: np.ndarray
    prod_names: np.ndarray
    country_names: np.ndarray
    tmax: pd.Timestamp
    svd_items: np.ndarray | None = None
    key: str = "v0"
    group_names: list = field(default_factory=list)
    currency: str = "GBP"
    groups_provisional: bool = True
    cust_attrs: pd.DataFrame | None = None     # optionale Kundenattribute (z. B. Einkaufsgemeinschaft, Kategorie)


_BASES: dict[str, Base] = {}
_VARIANTS: dict[str, pd.DataFrame] = {}


def register_variant(key: str, raw: pd.DataFrame) -> None:
    """Bereinigten Datenstand (Pipeline-Version) für Backtests bereitstellen. Rohdaten bleiben unverändert."""
    _VARIANTS[key] = raw
    _BASES.pop(key, None)
    for k in [k for k in _TASKS if k[1] == key]:
        _TASKS.pop(k)


def base_of(task) -> "Base":
    return get_base(task.extra.get("base_key", "v0"))


def get_base(key: str = "v0") -> Base:
    if key not in _BASES:
        from .data_loader import get_registry
        if key == "v0":
            raw = get_registry().raw_online_retail()
        elif key == "wwi" and key not in _VARIANTS:
            from .connectors import wwi_as_retail
            raw = wwi_as_retail()
        elif key == "aw" and key not in _VARIANTS:
            from .connectors import aw_as_retail
            raw = aw_as_retail()
        else:
            raw = _VARIANTS[key]
        _BASES[key] = _build_base(raw, key)
    return _BASES[key]


def _build_base(raw: pd.DataFrame, key: str) -> Base:
    if True:
        from .data_loader import cat_startswith, product_descriptions
        from .product_groups import GROUPS, NON_ARTICLE, product_group_codes
        storno = cat_startswith(raw["invoice"], "C")
        cust = raw["customer_id"].to_numpy(dtype="float64", na_value=np.nan)
        has_c = ~np.isnan(cust)
        prod = raw["stock_code"].cat.codes.to_numpy().astype(np.int32)
        ctry = raw["country"].cat.codes.to_numpy().astype(np.int16)
        inv = raw["invoice"].cat.codes.to_numpy().astype(np.int32)
        qty = raw["quantity"].to_numpy().astype(np.float64)
        currency = "EUR" if "price_eur" in raw.columns else "GBP"
        price = raw["price_eur" if currency == "EUR" else "price_gbp"].to_numpy().astype(np.float64)
        t = raw["invoice_date"].to_numpy()
        stock_cats = raw["stock_code"].cat.categories
        desc = product_descriptions(raw)
        if "pb05_group" in raw.columns:
            # Gruppen aus Playbook PB-05 (Freitext → Kategorien) statt der vorläufigen Schlüsselwort-Regeln
            gcat = raw["pb05_group"].astype("category")
            group_names = list(gcat.cat.categories)
            if NON_ARTICLE not in group_names:
                group_names.append(NON_ARTICLE)
            from .data_loader import _mode_codes
            k, v, _ = _mode_codes(raw["stock_code"].cat.codes.to_numpy().astype(np.int64),
                                  gcat.cat.codes.to_numpy().astype(np.int64), len(gcat.cat.categories))
            groups = np.full(len(stock_cats), group_names.index(NON_ARTICLE), dtype=np.int16)
            groups[k] = v
            provisional = False
        else:
            groups = product_group_codes(stock_cats, desc)
            group_names, provisional = list(GROUPS), True
        m = has_c & ~storno & (qty > 0)
        lines = pd.DataFrame({"cust": cust[m].astype(np.int64), "inv": inv[m], "time": t[m], "prod": prod[m],
                              "qty": qty[m], "price": price[m], "value": qty[m] * price[m], "country": ctry[m]})
        lines["group"] = groups[lines["prod"].to_numpy()]
        lines = lines.sort_values("time", kind="stable").reset_index(drop=True)
        # Gast-Kunden (PB-01 B, negative IDs) nur für Warenkorb/Empfehlung/Prognose – nicht als Kunden
        g = lines[lines["cust"] > 0].groupby(["cust", "inv"], sort=False)
        invt = g.agg(time=("time", "min"), value=("value", "sum"), n_lines=("prod", "size")).reset_index()
        invt = invt.sort_values("time", kind="stable").reset_index(drop=True)
        ms = has_c & storno
        st = pd.DataFrame({"cust": cust[ms].astype(np.int64), "time": t[ms], "prod": prod[ms],
                           "value": np.abs(qty[ms] * price[ms])}).sort_values("time").reset_index(drop=True)
        base = Base(lines=lines, inv=invt, storno=st, n_prod=len(stock_cats),
                    n_country=len(raw["country"].cat.categories), prod_group=groups,
                    prod_names=np.asarray(desc.reindex(stock_cats).fillna("").to_numpy()),
                    country_names=np.asarray(raw["country"].cat.categories), tmax=pd.Timestamp(t.max()),
                    key=key, group_names=group_names, currency=currency, groups_provisional=provisional)
        extra_cols = [c for c in ("buying_group", "customer_category", "credit_limit") if c in raw.columns]
        if extra_cols:
            ca = raw.loc[has_c, ["customer_id"] + extra_cols].drop_duplicates("customer_id")
            ca = ca.set_index(ca["customer_id"].astype(np.int64)).drop(columns="customer_id")
            base.cust_attrs = ca
        with dataset_time(key):
            base.svd_items = _fit_reference_svd(base)
    return base


def _fit_reference_svd(b: Base, k: int = 8) -> np.ndarray:
    """Artikel-Faktoren aus dem Zeitraum VOR allen Stichtagen – stabil und leakage-frei."""
    from sklearn.decomposition import TruncatedSVD
    ref = b.lines[b.lines["time"] < SVD_REF_END]
    cu = pd.factorize(ref["cust"])[0]
    M = sparse.csr_matrix((np.log1p(ref["qty"].to_numpy()), (cu, ref["prod"].to_numpy())),
                          shape=(cu.max() + 1, b.n_prod))
    svd = TruncatedSVD(n_components=k, random_state=SEED).fit(M)
    comp = svd.components_
    comp *= np.sign(comp[np.arange(k), np.abs(comp).argmax(axis=1)])[:, None]
    return comp.T.astype(np.float32)            # n_prod × k


def snapshots(horizon_days: int = HORIZON, step_months: int = 1) -> list[pd.Timestamp]:
    out, T = [], TRAIN_START
    while T + pd.Timedelta(days=horizon_days) <= T_TEST:
        out.append(T)
        T = T + pd.DateOffset(months=step_months)
    return out


# ── Leakage-Guard ─────────────────────────────────────────────────────────────

@dataclass
class Audit:
    rows: list = field(default_factory=list)

    def check(self, what: str, T: pd.Timestamp, used_max, label_end=None) -> None:
        used_max = pd.Timestamp(used_max) if used_max is not None and not pd.isna(used_max) else None
        if used_max is not None and used_max >= T:
            raise AssertionError(f"Leakage: {what} nutzt Ereignis {used_max} ≥ Stichtag {T}")
        if label_end is not None and T < T_TEST and label_end > T_TEST:
            raise AssertionError(f"Leakage: Trainings-Label {what} endet {label_end} nach dem Test-Stichtag")
        self.rows.append({"what": what, "cutoff": T.date().isoformat(),
                          "max_feature_time": used_max.isoformat() if used_max is not None else None,
                          "label_end": label_end.date().isoformat() if label_end is not None else None})


# ── Kunden- und Graph-Merkmale zum Stichtag ───────────────────────────────────

def customer_features(b: Base, T: pd.Timestamp, audit: Audit, what: str) -> pd.DataFrame:
    """Merkmale aller Kunden mit Kauf vor T – nur Ereignisse < T."""
    inv = b.inv[b.inv["time"] < T]
    audit.check(f"{what}:belege", T, inv["time"].max())
    t_ns = T.value
    it = inv["time"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    day = 86400 * 10**9
    age = (t_ns - it) / day
    df = pd.DataFrame({"cust": inv["cust"].to_numpy(), "age": age, "value": inv["value"].to_numpy()})
    g = df.groupby("cust")
    out = pd.DataFrame({"recency": g["age"].min(), "tenure": g["age"].max()})
    for w in (30, 90, 365):
        sub = df[df["age"] <= w]
        out[f"freq_{w}"] = sub.groupby("cust").size()
    for w in (90, 365):
        sub = df[df["age"] <= w]
        out[f"monetary_{w}"] = sub.groupby("cust")["value"].sum()
    prev = df[(df["age"] > 90) & (df["age"] <= 180)]
    out["prev_orders"] = prev.groupby("cust").size()
    out["prev_rev"] = prev.groupby("cust")["value"].sum()
    out = out.fillna(0.0)
    out["avg_basket"] = out["monetary_365"] / out["freq_365"].clip(lower=1)
    out["trend_orders"] = out["freq_90"] / (out["prev_orders"] + 1)
    out["trend_revenue"] = out["monetary_90"] / (out["prev_rev"] + 1)
    srt = df.sort_values(["cust", "age"], ascending=[True, False])
    gaps = srt.groupby("cust")["age"].diff().abs()
    gs = gaps.groupby(srt["cust"])
    out["gap_mean"] = gs.mean().reindex(out.index)
    out["gap_std"] = gs.std().reindex(out.index)
    out["overdue"] = out["recency"] / out["gap_mean"].clip(lower=1)
    ln = b.lines[(b.lines["time"] < T) & (b.lines["time"] >= T - pd.Timedelta(days=HISTORY))]
    audit.check(f"{what}:positionen", T, ln["time"].max())
    out["n_products"] = ln.groupby("cust")["prod"].nunique().reindex(out.index).fillna(0)
    last_country = b.lines[b.lines["time"] < T].groupby("cust")["country"].last()
    out["country"] = last_country.reindex(out.index).fillna(-1).astype(int)
    st = b.storno[(b.storno["time"] < T) & (b.storno["time"] >= T - pd.Timedelta(days=HISTORY))]
    audit.check(f"{what}:stornos", T, st["time"].max())
    out["storno_cnt"] = st.groupby("cust").size().reindex(out.index).fillna(0)
    out["storno_share"] = (st.groupby("cust")["value"].sum().reindex(out.index).fillna(0)
                           / out["monetary_365"].clip(lower=1))
    out["month"] = T.month
    return out.drop(columns=["prev_orders", "prev_rev"])


def _bipartite(b: Base, T: pd.Timestamp, custs: np.ndarray) -> tuple[sparse.csr_matrix, dict]:
    ln = b.lines[(b.lines["time"] < T) & (b.lines["time"] >= T - pd.Timedelta(days=HISTORY))]
    idx = {c: i for i, c in enumerate(custs)}
    rows = ln["cust"].map(idx)
    keep = rows.notna().to_numpy()
    B = sparse.csr_matrix((np.ones(keep.sum()), (rows[keep].astype(int).to_numpy(), ln["prod"].to_numpy()[keep])),
                          shape=(len(custs), b.n_prod))
    B.data[:] = 1.0
    B.sum_duplicates()
    B.data[:] = 1.0
    return B, idx


def _pagerank(B: sparse.csr_matrix, iters: int = 25, alpha: float = 0.85) -> np.ndarray:
    nc, npd = B.shape
    A = sparse.bmat([[None, B], [B.T, None]]).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    P = sparse.diags(1 / deg) @ A
    r = np.full(nc + npd, 1 / (nc + npd))
    for _ in range(iters):
        r = (1 - alpha) / (nc + npd) + alpha * (P.T @ r)
    return r[:nc]


def graph_customer_features(b: Base, T: pd.Timestamp, feats: pd.DataFrame, audit: Audit, what: str) -> pd.DataFrame:
    """2-Sprung-Merkmale über den Kunde–Artikel-Graphen (nur Kanten vor T)."""
    custs = feats.index.to_numpy()
    B, _ = _bipartite(b, T, custs)
    audit.check(f"{what}:graph", T, b.lines.loc[(b.lines["time"] < T), "time"].max())
    deg_c = np.asarray(B.sum(axis=1)).ravel()
    deg_p = np.asarray(B.sum(axis=0)).ravel()
    inactive = (feats["recency"].to_numpy() >= 90).astype(float)
    num = B @ (B.T @ inactive) - deg_c * inactive
    den = B @ deg_p - deg_c
    g = pd.DataFrame(index=feats.index)
    g["g_nb_inactive"] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    g["g_nb_size"] = np.log1p(np.maximum(den, 0))
    g["g_prod_pop"] = np.where(deg_c > 0, (B @ np.log1p(deg_p)) / np.maximum(deg_c, 1), np.nan)
    g["g_pagerank"] = np.log(_pagerank(B) * len(custs) + 1e-9)
    emb = B @ b.svd_items
    emb = emb / np.maximum(deg_c, 1)[:, None]
    for i in range(emb.shape[1]):
        g[f"g_svd_{i}"] = emb[:, i]
    return g


# ── Aufgaben ──────────────────────────────────────────────────────────────────

@dataclass
class Task:
    uc: str
    kind: str                                   # binary · ranking · forecast
    train: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    features: list = field(default_factory=list)
    graph_features: list = field(default_factory=list)
    cat_features: list = field(default_factory=list)
    label: str = "y"
    rule_col: str = "rule"
    value_col: str = "value"
    id_col: str = "cust"
    snapshots: list = field(default_factory=list)
    val_cutoff: pd.Timestamp | None = None
    audit: Audit = field(default_factory=Audit)
    extra: dict = field(default_factory=dict)
    timeline: dict = field(default_factory=dict)

    @property
    def base_features(self) -> list:
        return [f for f in self.features if f not in self.graph_features]


def _timeline(snaps, test_cutoff, horizon_days, label_text) -> dict:
    return {"train": [s.date().isoformat() for s in snaps[:-1]], "val": snaps[-1].date().isoformat(),
            "test": test_cutoff.date().isoformat(),
            "test_label_end": (test_cutoff + pd.Timedelta(days=horizon_days - 1)).date().isoformat(),
            "horizon_days": horizon_days, "label": label_text}


def build_churn(b: Base) -> Task:
    task = Task(uc="churn", kind="binary")
    snaps = snapshots()
    frames = []
    for T in snaps + [T_TEST]:
        f = customer_features(b, T, task.audit, "churn")
        f = f[f["recency"] <= HISTORY]                       # aktiv: Kauf in den letzten 365 Tagen
        g = graph_customer_features(b, T, f, task.audit, "churn")
        fut = b.inv[(b.inv["time"] >= T) & (b.inv["time"] < T + pd.Timedelta(days=HORIZON))]
        task.audit.check("churn:label", T, None, T + pd.Timedelta(days=HORIZON))
        y = (~f.index.isin(fut["cust"].unique())).astype(int)
        # Regel wie im Vertrieb: erst alle seit ≥ 60 Tagen Inaktiven, nach Umsatz sortiert, dann der Rest nach Recency
        rec, mon = f["recency"].to_numpy(), f["monetary_365"].to_numpy()
        rule = np.where(rec >= 60, 1e7 + mon, rec)
        d = pd.concat([f, g], axis=1).assign(y=y, cutoff=T, rule=rule, value=mon)
        frames.append(d.reset_index().rename(columns={"index": "cust"}))
    data = pd.concat(frames, ignore_index=True)
    task.train, task.test = data[data["cutoff"] < T_TEST], data[data["cutoff"] == T_TEST]
    task.graph_features = [c for c in data.columns if c.startswith("g_")]
    task.features = ["recency", "freq_30", "freq_90", "freq_365", "monetary_90", "monetary_365", "avg_basket",
                     "tenure", "n_products", "storno_cnt", "storno_share", "trend_orders", "trend_revenue",
                     "gap_mean", "gap_std", "overdue", "country"] + task.graph_features
    task.cat_features = ["country"]
    task.snapshots, task.val_cutoff = snaps, snaps[-1]
    # Entscheidungsregel für Modelle: Liste nach erwartetem Umsatzverlust (Wahrscheinlichkeit × Umsatz 365 Tage)
    task.extra["rank_by_value"] = True
    task.timeline = _timeline(snaps, T_TEST, HORIZON, "kein Beleg in 90 Tagen")
    task.extra["rule_text"] = "„seit 60 Tagen keine Bestellung“, nach Umsatz sortiert – so priorisiert der Vertrieb heute"
    return task


def _repeat_propensity(b: Base, T: pd.Timestamp, audit: Audit) -> tuple[pd.Series, pd.Series, float]:
    """Je Artikel/Land: Anteil früherer Erstkäufer (Label vor T bekannt), die binnen 90 Tagen wiederkauften."""
    inv = b.inv[b.inv["time"] < T].sort_values(["cust", "time"])
    audit.check("lead:repeat", T, inv["time"].max())
    k = inv.groupby("cust").cumcount()
    first = inv[k == 0].set_index("cust")
    second = inv[k == 1].set_index("cust")["time"].reindex(first.index)
    known = first["time"] < T - pd.Timedelta(days=HORIZON)
    rep = ((second - first["time"]) <= pd.Timedelta(days=HORIZON)).astype(float)[known]
    glob = float(rep.mean()) if len(rep) else 0.3
    fl = b.lines.merge(first.loc[known, ["inv"]].reset_index(), on=["cust", "inv"])
    fl = fl.assign(rep=fl["cust"].map(rep))
    pg = fl.groupby("prod")["rep"].agg(["sum", "count"])
    prod_rate = (pg["sum"] + 5 * glob) / (pg["count"] + 5)
    cg = fl.drop_duplicates("cust").groupby("country")["rep"].agg(["sum", "count"])
    ctry_rate = (cg["sum"] + 10 * glob) / (cg["count"] + 10)
    return prod_rate, ctry_rate, glob


def build_lead(b: Base) -> Task:
    task = Task(uc="lead_scoring", kind="binary")
    snaps = snapshots()
    frames = []
    for T in snaps + [T_TEST]:
        f = customer_features(b, T, task.audit, "lead")
        inv = b.inv[b.inv["time"] < T]
        n_orders = inv.groupby("cust").size()
        first_t = inv.groupby("cust")["time"].min()
        new = first_t[(first_t >= T - pd.Timedelta(days=HORIZON))].index
        new = new[n_orders.reindex(new).to_numpy() == 1]
        if len(new) == 0:
            continue
        f = f.loc[new]
        fi = inv[inv["cust"].isin(new)].drop_duplicates("cust").set_index("cust")
        fl = b.lines[(b.lines["time"] < T) & b.lines["cust"].isin(new)]
        task.audit.check("lead:erstkauf", T, fl["time"].max())
        f["first_value"] = fi["value"].reindex(new)
        f["first_lines"] = fi["n_lines"].reindex(new)
        f["first_products"] = fl.groupby("cust")["prod"].nunique().reindex(new)
        f["first_avg_price"] = fl.groupby("cust")["price"].mean().reindex(new)
        f["first_storno"] = (f["storno_cnt"] > 0).astype(int)
        f["days_since_first"] = f["tenure"]
        prod_rate, ctry_rate, glob = _repeat_propensity(b, T, task.audit)
        g = pd.DataFrame(index=new)
        g["g_repeat_prop"] = fl.assign(r=fl["prod"].map(prod_rate).fillna(glob)).groupby("cust")["r"].mean().reindex(new)
        g["g_country_repeat"] = f["country"].map(ctry_rate).fillna(glob).to_numpy()
        g["g_prod_pop"] = fl.assign(p=fl.groupby("prod")["cust"].transform("size")).groupby("cust")["p"].mean().reindex(new)
        emb = sparse.csr_matrix((np.ones(len(fl)), (pd.Index(new).get_indexer(fl["cust"]), fl["prod"].to_numpy())),
                                shape=(len(new), b.n_prod)) @ b.svd_items
        emb = emb / np.maximum(f["first_products"].to_numpy(), 1)[:, None]
        for i in range(emb.shape[1]):
            g[f"g_svd_{i}"] = emb[:, i]
        fut = b.inv[(b.inv["time"] >= T) & (b.inv["time"] < T + pd.Timedelta(days=HORIZON))]
        task.audit.check("lead:label", T, None, T + pd.Timedelta(days=HORIZON))
        y = f.index.isin(fut["cust"].unique()).astype(int)
        d = pd.concat([f, g], axis=1).assign(y=y, cutoff=T, rule=f["first_value"].to_numpy(),
                                             value=f["first_value"].to_numpy())
        frames.append(d.reset_index().rename(columns={"index": "cust"}))
    data = pd.concat(frames, ignore_index=True)
    task.train, task.test = data[data["cutoff"] < T_TEST], data[data["cutoff"] == T_TEST]
    task.graph_features = [c for c in data.columns if c.startswith("g_")]
    task.features = ["first_value", "first_lines", "first_products", "first_avg_price", "first_storno",
                     "days_since_first", "country"] + task.graph_features
    task.cat_features = ["country"]
    task.snapshots, task.val_cutoff = snaps, snaps[-1]
    task.timeline = _timeline(snaps, T_TEST, HORIZON, "Erstkäufer kauft in 90 Tagen erneut")
    task.extra["rule_text"] = "Wert des ersten Warenkorbs"
    return task


def build_cross_selling(b: Base) -> Task:
    from .product_groups import NON_ARTICLE
    GROUPS = b.group_names
    task = Task(uc="cross_selling", kind="binary", id_col="pair")
    snaps = snapshots(step_months=2)
    groups = [i for i, g in enumerate(GROUPS) if g != NON_ARTICLE]
    frames = []
    for T in snaps + [T_TEST]:
        f = customer_features(b, T, task.audit, "cross")
        f = f[f["recency"] <= HISTORY]
        hist = b.lines[b.lines["time"] < T]
        task.audit.check("cross:gruppen", T, hist["time"].max())
        cg = hist[hist["cust"].isin(f.index)][["cust", "group"]].drop_duplicates()
        bought = cg.groupby("cust")["group"].apply(set)
        # Co-Kauf-Affinität zwischen Gruppen (Graph Kunde–Gruppe, nur vor T)
        allcg = hist[["cust", "group"]].drop_duplicates()
        ci = pd.factorize(allcg["cust"])[0]
        M = sparse.csr_matrix((np.ones(len(allcg)), (ci, allcg["group"].to_numpy())), shape=(ci.max() + 1, len(GROUPS)))
        co = (M.T @ M).toarray()
        cond = co / np.maximum(np.diag(co)[:, None], 1)       # P(g | h)
        recent = hist[hist["time"] >= T - pd.Timedelta(days=90)]
        pop = recent.groupby("group")["cust"].nunique().reindex(range(len(GROUPS))).fillna(0).to_numpy()
        prev = hist[(hist["time"] < T - pd.Timedelta(days=90)) & (hist["time"] >= T - pd.Timedelta(days=180))]
        pop_prev = prev.groupby("group")["cust"].nunique().reindex(range(len(GROUPS))).fillna(0).to_numpy()
        fut = b.lines[(b.lines["time"] >= T) & (b.lines["time"] < T + pd.Timedelta(days=HORIZON))]
        task.audit.check("cross:label", T, None, T + pd.Timedelta(days=HORIZON))
        fut_pairs = set(zip(fut["cust"].to_numpy(), fut["group"].to_numpy()))
        rows = []
        for c, bset in bought.items():
            bl = list(bset)
            for gidx in groups:
                if gidx in bset:
                    continue
                aff = cond[bl, gidx]
                rows.append((c, gidx, aff.max(), aff.mean(), len(bset), int((c, gidx) in fut_pairs)))
        p = pd.DataFrame(rows, columns=["cust", "group", "g_affinity_max", "g_affinity_mean", "n_groups", "y"])
        p["group_pop"] = pop[p["group"].to_numpy()]
        p["group_growth"] = (pop / np.maximum(pop_prev, 1))[p["group"].to_numpy()]
        p = p.join(f[["recency", "freq_365", "monetary_365", "n_products", "tenure", "country", "month"]], on="cust")
        p = p.assign(cutoff=T, rule=p["group_pop"], value=p["monetary_365"],
                     pair=p["cust"].astype(str) + ":" + p["group"].astype(str))
        frames.append(p)
    data = pd.concat(frames, ignore_index=True)
    task.train, task.test = data[data["cutoff"] < T_TEST], data[data["cutoff"] == T_TEST]
    task.graph_features = ["g_affinity_max", "g_affinity_mean"]
    task.features = ["group", "group_pop", "group_growth", "n_groups", "recency", "freq_365", "monetary_365",
                     "n_products", "tenure", "country"] + task.graph_features
    task.cat_features = ["group", "country"]
    task.snapshots, task.val_cutoff = snaps, snaps[-1]
    task.timeline = _timeline(snaps, T_TEST, HORIZON, "Erstkauf einer neuen Artikelgruppe in 90 Tagen")
    task.extra["rule_text"] = "meistverkaufte noch nicht gekaufte Gruppe (Beliebtheit 90 Tage)"
    task.extra["groups_provisional"] = b.groups_provisional
    task.extra["groups"] = GROUPS
    return task


def build_returns(b: Base) -> Task:
    """Positionsebene: Verkauf im Monat M, Label = Storno desselben Kunden/Artikels binnen 30 Tagen."""
    task = Task(uc="returns_root_cause", kind="binary", id_col="line")
    rng = np.random.default_rng(SEED)
    L = b.lines
    S = b.storno.rename(columns={"time": "st_time"})[["cust", "prod", "st_time"]].sort_values("st_time")
    months = list(pd.date_range("2010-06-01", "2011-07-01", freq="MS"))
    tests = [T_TEST, T_TEST + pd.DateOffset(months=1)]
    frames = []
    for M in months + tests:
        end = M + pd.DateOffset(months=1)
        cur = L[(L["time"] >= M) & (L["time"] < end) & (L["cust"] > 0)].copy()
        if M in months and len(cur) > 25_000:
            cur = cur.iloc[np.sort(rng.choice(len(cur), 25_000, replace=False))]
        hist = L[L["time"] < M]
        hs = b.storno[b.storno["time"] < M]
        task.audit.check("returns:historie", M, max(hist["time"].max(), hs["time"].max()))
        # Label: Storno binnen 30 Tagen nach der Position (Vorwärts-Zuordnung)
        cs = cur.sort_values("time")
        mm = pd.merge_asof(cs[["time", "cust", "prod"]].reset_index(), S, left_on="time", right_on="st_time",
                           by=["cust", "prod"], direction="forward", tolerance=pd.Timedelta(days=30))
        lab = pd.Series(mm["st_time"].notna().astype(int).to_numpy(), index=mm["index"])
        task.audit.check("returns:label", M if M < T_TEST else T_TEST, None, end + pd.Timedelta(days=30))
        cur["y"] = lab.reindex(cur.index).to_numpy()
        ps = hist.groupby("prod").size()
        pr = hs.groupby("prod").size()
        prod_ret = ((pr.reindex(ps.index).fillna(0) + 0.2) / (ps + 10))
        glob_ret = float(len(hs) / max(len(hist), 1))
        cust_sales = hist.groupby("cust").size()
        cust_ret = ((hs.groupby("cust").size().reindex(cust_sales.index).fillna(0) + 0.2) / (cust_sales + 10))
        med = hist.groupby("prod")["price"].median()
        cur["g_prod_return"] = cur["prod"].map(prod_ret).fillna(glob_ret).to_numpy()
        cur["g_cust_return"] = cur["cust"].map(cust_ret).fillna(glob_ret).to_numpy()
        # 2 Sprünge: Retourenquote der Artikel, die dieselben Kunden kaufen
        recent = hist[hist["time"] >= M - pd.Timedelta(days=HISTORY)][["cust", "prod"]].drop_duplicates()
        recent = recent.assign(r=recent["prod"].map(prod_ret).fillna(glob_ret))
        cust_nb = recent.groupby("cust")["r"].mean()
        prod_nb = recent.assign(cr=recent["cust"].map(cust_nb)).groupby("prod")["cr"].mean()
        cur["g_nb_return"] = cur["prod"].map(prod_nb).fillna(glob_ret).to_numpy()
        cur["prod_sales"] = np.log1p(cur["prod"].map(ps).fillna(0).to_numpy())
        cur["cust_orders"] = cur["cust"].map(hist.groupby("cust")["inv"].nunique()).fillna(0).to_numpy()
        cur["price_rel"] = (cur["price"] / cur["prod"].map(med)).fillna(1.0).to_numpy()
        cur["basket_lines"] = cur.groupby("inv")["prod"].transform("size").to_numpy()
        known = set(zip(hist["cust"].to_numpy(), hist["prod"].to_numpy()))
        cur["new_for_customer"] = [int((c, p) not in known) for c, p in zip(cur["cust"], cur["prod"])]
        cur["new_product"] = (~cur["prod"].isin(ps.index)).astype(int).to_numpy()
        cur["month"] = M.month
        cur = cur.assign(cutoff=M if M < T_TEST else T_TEST, rule=cur["g_prod_return"], line=cur.index)
        frames.append(cur)
    data = pd.concat(frames, ignore_index=True)
    task.train, task.test = data[data["cutoff"] < T_TEST], data[data["cutoff"] == T_TEST]
    task.graph_features = ["g_prod_return", "g_cust_return", "g_nb_return"]
    task.features = ["qty", "price", "value", "price_rel", "basket_lines", "new_for_customer", "new_product",
                     "prod_sales", "cust_orders", "country", "month", "group"] + task.graph_features
    task.cat_features = ["country", "group"]
    task.snapshots, task.val_cutoff = months, months[-1]
    task.timeline = {"train": [m.date().isoformat() for m in months[:-1]], "val": months[-1].date().isoformat(),
                     "test": T_TEST.date().isoformat(), "test_label_end": "2011-11-30", "horizon_days": 30,
                     "label": "Storno derselben Position binnen 30 Tagen", "unit": "Position"}
    task.extra["rule_text"] = "Retourenquote des Artikels in der Vorperiode"
    return task


def _peer_growth(f: pd.DataFrame, key: pd.Series) -> np.ndarray:
    """Ø Umsatzwachstum der anderen Kunden derselben Gruppe (ohne den Kunden selbst) – Peer-Signal über den Graphen."""
    g = np.log1p(f["monetary_90"]) - np.log1p(f["prev_rev90"])
    k = key.reindex(f.index).astype(str).to_numpy()
    df = pd.DataFrame({"k": k, "g": g.to_numpy()})
    s, n = df.groupby("k")["g"].transform("sum"), df.groupby("k")["g"].transform("size")
    return np.where(n > 1, (s - df["g"]) / np.maximum(n - 1, 1), np.nan)


def build_account(b: Base) -> Task:
    """Account-Scoring (WideWorldImporters): wächst der Kunde im nächsten Quartal um ≥ 20 %?

    Graph-Signal: Wachstum der Peers in derselben Einkaufsgemeinschaft, Kundenkategorie und Region
    sowie die Kunde–Artikel-Nachbarschaft (alles nur aus Ereignissen vor dem Stichtag).
    """
    task = Task(uc="account_scoring", kind="binary")
    snaps = snapshots()
    attrs = b.cust_attrs if b.cust_attrs is not None else pd.DataFrame()
    frames = []
    for T in snaps + [T_TEST]:
        f = customer_features(b, T, task.audit, "account")
        f = f[f["recency"] <= 180]                                    # aktive Accounts
        inv = b.inv[(b.inv["time"] < T) & (b.inv["time"] >= T - pd.Timedelta(days=180))]
        prev = inv[inv["time"] < T - pd.Timedelta(days=90)]
        f["prev_rev90"] = prev.groupby("cust")["value"].sum().reindex(f.index).fillna(0.0)
        g = graph_customer_features(b, T, f, task.audit, "account")
        for col, name in (("buying_group", "g_peer_group_growth"), ("customer_category", "g_peer_category_growth")):
            if col in attrs:
                g[name] = _peer_growth(f, attrs[col])
        g["g_peer_region_growth"] = _peer_growth(f, f["country"])
        for col in ("buying_group", "customer_category"):
            if col in attrs:
                f[col] = attrs[col].reindex(f.index).astype("category").cat.codes.to_numpy()
        if "credit_limit" in attrs:
            f["credit_limit"] = attrs["credit_limit"].reindex(f.index).to_numpy()
        fut = b.inv[(b.inv["time"] >= T) & (b.inv["time"] < T + pd.Timedelta(days=HORIZON))]
        task.audit.check("account:label", T, None, T + pd.Timedelta(days=HORIZON))
        nxt = fut.groupby("cust")["value"].sum().reindex(f.index).fillna(0.0)
        y = (nxt >= 1.2 * f["monetary_90"].clip(lower=1)).astype(int).to_numpy()
        gain = np.clip(nxt.to_numpy() - f["monetary_90"].to_numpy(), 0, None)
        d = pd.concat([f, g], axis=1).assign(y=y, cutoff=T, rule=f["trend_revenue"].to_numpy(), value=gain)
        frames.append(d.reset_index().rename(columns={"index": "cust"}))
    data = pd.concat(frames, ignore_index=True)
    task.train, task.test = data[data["cutoff"] < T_TEST], data[data["cutoff"] == T_TEST]
    task.graph_features = [c for c in data.columns if c.startswith("g_")]
    cats = [c for c in ("buying_group", "customer_category") if c in data.columns]
    task.features = ["recency", "freq_90", "freq_365", "monetary_90", "monetary_365", "avg_basket", "tenure",
                     "n_products", "trend_orders", "trend_revenue", "gap_mean", "overdue", "country"] + \
        [c for c in ("credit_limit",) if c in data.columns] + cats + task.graph_features
    task.cat_features = ["country"] + cats
    task.snapshots, task.val_cutoff = snaps, snaps[-1]
    task.timeline = _timeline(snaps, T_TEST, HORIZON, "Quartalsumsatz wächst um ≥ 20 %")
    task.extra["rule_text"] = "Umsatztrend des letzten Quartals – Accounts mit steigendem Umsatz zuerst"
    return task


def _peer_active(f: pd.DataFrame, key: pd.Series, days: int) -> np.ndarray:
    """Anteil der anderen Kunden derselben Gruppe mit Kauf in den letzten `days` Tagen (ohne den Kunden selbst)."""
    a = (f["recency"] <= days).astype(float).to_numpy()
    k = key.reindex(f.index).astype(str).to_numpy()
    df = pd.DataFrame({"k": k, "a": a})
    s, n = df.groupby("k")["a"].transform("sum"), df.groupby("k")["a"].transform("size")
    return np.where(n > 1, (s - df["a"]) / np.maximum(n - 1, 1), np.nan)


def build_territory(b: Base) -> Task:
    """Vertriebsabdeckung (AdventureWorks): bestellt der Händler im nächsten Quartal? Liste nach erwartetem Umsatz.

    Graph-Signal: Bestellwellen im Portfolio desselben Vertrieblers und im Gebiet (Peers ohne den Händler selbst)
    sowie die Händler–Artikel-Nachbarschaft – Händler und Vertriebler sind über Kanten verbunden, nicht nur Zeilen.
    """
    task = Task(uc="territory_coverage", kind="binary")
    snaps = snapshots()
    attrs = b.cust_attrs if b.cust_attrs is not None else pd.DataFrame()
    stores = attrs.index[attrs.get("customer_category", pd.Series(dtype=str)).astype(str) == "Händler"] \
        if "customer_category" in attrs else attrs.index
    frames = []
    for T in snaps + [T_TEST]:
        f = customer_features(b, T, task.audit, "territory")
        f = f[(f["recency"] <= HISTORY) & f.index.isin(stores)]
        inv = b.inv[(b.inv["time"] < T) & (b.inv["time"] >= T - pd.Timedelta(days=180))]
        f["prev_rev90"] = inv[inv["time"] < T - pd.Timedelta(days=90)].groupby("cust")["value"].sum().reindex(f.index).fillna(0.0)
        g = graph_customer_features(b, T, f, task.audit, "territory")
        port = attrs["buying_group"] if "buying_group" in attrs else pd.Series("–", index=f.index)
        g["g_portfolio_active30"] = _peer_active(f, port, 30)
        g["g_portfolio_active90"] = _peer_active(f, port, 90)
        g["g_territory_active30"] = _peer_active(f, f["country"], 30)
        g["g_portfolio_growth"] = _peer_growth(f, port)
        f["salesperson"] = port.reindex(f.index).astype("category").cat.codes.to_numpy()
        fut = b.inv[(b.inv["time"] >= T) & (b.inv["time"] < T + pd.Timedelta(days=HORIZON))]
        task.audit.check("territory:label", T, None, T + pd.Timedelta(days=HORIZON))
        nxt = fut.groupby("cust")["value"].sum().reindex(f.index).fillna(0.0)
        y = (nxt > 0).astype(int).to_numpy()
        d = pd.concat([f, g], axis=1).assign(y=y, cutoff=T, rule=f["monetary_365"].to_numpy(),
                                             value=f["monetary_365"].to_numpy(), next_revenue=nxt.to_numpy())
        frames.append(d.reset_index().rename(columns={"index": "cust"}))
    data = pd.concat(frames, ignore_index=True)
    task.train, task.test = data[data["cutoff"] < T_TEST], data[data["cutoff"] == T_TEST]
    task.graph_features = [c for c in data.columns if c.startswith("g_")]
    task.features = ["recency", "freq_90", "freq_365", "monetary_90", "monetary_365", "avg_basket", "tenure",
                     "n_products", "trend_orders", "trend_revenue", "gap_mean", "overdue", "country", "salesperson"] + \
        task.graph_features
    task.cat_features = ["country", "salesperson"]
    task.snapshots, task.val_cutoff = snaps, snaps[-1]
    task.extra["rank_by_value"] = True
    task.timeline = _timeline(snaps, T_TEST, HORIZON, "Händler bestellt im nächsten Quartal")
    task.extra["rule_text"] = "Händler mit dem höchsten Jahresumsatz zuerst besuchen"
    return task


def build_recommendation(b: Base) -> Task:
    task = Task(uc="recommendation", kind="ranking")
    snaps = [pd.Timestamp("2011-03-01"), pd.Timestamp("2011-06-01")]
    task.snapshots, task.val_cutoff = snaps, snaps[-1]
    task.extra["cutoffs"] = snaps + [T_TEST]
    task.timeline = _timeline(snaps, T_TEST, HORIZON, "gekaufte Artikel in 90 Tagen (MAP@12)")
    task.extra["rule_text"] = "Wiederkauf der eigenen Artikel, aufgefüllt mit Topsellern der letzten 30 Tage"
    for T in task.extra["cutoffs"]:
        task.audit.check("reco:historie", T, b.lines.loc[b.lines["time"] < T, "time"].max(),
                         T + pd.Timedelta(days=HORIZON))
    return task


def reco_snapshot(b: Base, T: pd.Timestamp) -> dict:
    """Historie (365 Tage vor T) und Ground Truth (90 Tage ab T) für aktive, kaufende Kunden."""
    hist = b.lines[(b.lines["time"] < T) & (b.lines["time"] >= T - pd.Timedelta(days=HISTORY))]
    fut = b.lines[(b.lines["time"] >= T) & (b.lines["time"] < T + pd.Timedelta(days=HORIZON))]
    if T > b.tmax:          # Aktivierungs-Stichtag ohne Zukunft: alle aktiven Kunden bewerten (Truth bleibt leer)
        users = np.sort(hist.loc[hist["cust"] > 0, "cust"].unique())
    else:
        users = np.intersect1d(hist["cust"].unique(), fut["cust"].unique())
    uidx = {u: i for i, u in enumerate(users)}
    h = hist[hist["cust"].isin(users)]
    R = sparse.csr_matrix((np.ones(len(h)), (h["cust"].map(uidx).to_numpy(), h["prod"].to_numpy())),
                          shape=(len(users), b.n_prod))
    R.sum_duplicates()
    last = h.groupby(["cust", "prod"])["time"].max()
    pop30 = hist[hist["time"] >= T - pd.Timedelta(days=30)].groupby("prod").size()
    pop90 = hist[hist["time"] >= T - pd.Timedelta(days=90)].groupby("prod").size()
    truth = fut[fut["cust"].isin(users)].groupby("cust")["prod"].apply(lambda s: set(s.unique()))
    # vollständige Interaktionen für Item-Ähnlichkeit (alle Kunden, 365 Tage)
    ai = pd.factorize(hist["cust"])[0]
    A = sparse.csr_matrix((np.ones(len(hist)), (ai, hist["prod"].to_numpy())), shape=(ai.max() + 1, b.n_prod))
    A.data[:] = 1
    A.sum_duplicates()
    A.data[:] = 1
    return {"T": T, "users": users, "R": R, "A": A, "last": last, "pop30": pop30, "pop90": pop90,
            "truth": [truth.get(u, set()) for u in users]}


def build_forecast(b: Base, n_series: int = 100) -> Task:
    task = Task(uc="demand_forecast", kind="forecast")
    L = b.lines
    week = L["time"].dt.to_period("W-SUN").dt.start_time
    top = L[L["time"] < T_TEST].groupby("prod")["qty"].sum().sort_values(ascending=False).head(n_series).index
    sub = pd.DataFrame({"prod": L["prod"], "week": week, "qty": L["qty"]})
    sub = sub[sub["prod"].isin(top)]
    weeks = pd.date_range(week.min(), pd.Timestamp("2011-11-28"), freq="W-MON")
    Y = sub.groupby(["prod", "week"])["qty"].sum().unstack(fill_value=0).reindex(index=top, columns=weeks,
                                                                                  fill_value=0)
    origin_test = weeks[weeks >= T_TEST][0]
    task.extra.update(Y=Y, weeks=weeks, origin_test=origin_test, horizon=13,
                      names=[b.prod_names[p] for p in top], prods=list(top))
    o_idx = weeks.get_loc(origin_test)
    task.extra["origins_train"] = [weeks[i] for i in range(o_idx - 13 - 26, o_idx - 13 + 1, 2) if i >= 56]
    task.audit.check("forecast:historie", origin_test, weeks[o_idx - 1] + pd.Timedelta(days=6), None)
    task.timeline = {"train": [o.date().isoformat() for o in task.extra["origins_train"]],
                     "val": task.extra["origins_train"][-1].date().isoformat(),
                     "test": origin_test.date().isoformat(), "test_label_end": weeks[o_idx + 12].date().isoformat(),
                     "horizon_days": 91, "label": "Wochenmenge der Top-100-Artikel, 13 Wochen", "unit": "Reihe×Woche"}
    task.extra["rule_text"] = "saisonal-naiv: Menge der Vorjahreswoche"
    return task


BUILDERS = {"churn": build_churn, "lead_scoring": build_lead, "cross_selling": build_cross_selling,
            "returns_root_cause": build_returns, "recommendation": build_recommendation, "account_scoring": build_account,
            "territory_coverage": build_territory,
            "demand_forecast": build_forecast}
_TASKS: dict[tuple, Task] = {}


def get_task(uc: str, key: str = "v0") -> Task:
    key = dataset_key(uc, key)
    if (uc, key) not in _TASKS:
        t0 = time.time()
        b = get_base(key)
        with dataset_time(key):
            task = BUILDERS[uc](b)
        task.extra["base_key"] = key
        task.extra["currency"] = b.currency
        task.extra["build_seconds"] = round(time.time() - t0, 1)
        _TASKS[(uc, key)] = task
    return _TASKS[(uc, key)]


# ── Metriken ──────────────────────────────────────────────────────────────────

def _topk_mask(s: np.ndarray, share: float = TOP_SHARE) -> np.ndarray:
    k = max(1, int(round(len(s) * share)))
    idx = np.argsort(-s, kind="stable")[:k]
    m = np.zeros(len(s), bool)
    m[idx] = True
    return m


def binary_metrics(y: np.ndarray, s: np.ndarray, value: np.ndarray | None = None, proba: bool = True,
                   rank: np.ndarray | None = None) -> dict:
    """s: Score/Wahrscheinlichkeit (AUROC, AUPRC, Brier). rank: Sortierung der Top-10-%-Liste (Standard = s)."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    top = _topk_mask(np.asarray(rank, dtype=float) if rank is not None else s)
    pos = y.sum()
    out = {
        "auroc": float(roc_auc_score(y, s)) if 0 < pos < len(y) else None,
        "auprc": float(average_precision_score(y, s)) if pos else None,
        "precision_at_10": float(y[top].mean()),
        "recall_at_10": float(y[top].sum() / max(pos, 1)),
        "base_rate": float(y.mean()), "n": int(len(y)), "positives": int(pos),
    }
    if value is not None:
        v = np.asarray(value, dtype=float)
        out["value_captured_at_10"] = float(v[top & (y == 1)].sum() / max(v[y == 1].sum(), 1e-9))
    if proba and np.nanmin(s) >= 0 and np.nanmax(s) <= 1:
        out["brier"] = float(brier_score_loss(y, s))
    return out


def curves(y: np.ndarray, s: np.ndarray, proba: bool = True, points: int = 40, rank: np.ndarray | None = None,
           value: np.ndarray | None = None) -> dict:
    from sklearn.metrics import roc_curve
    y = np.asarray(y).astype(int)
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.unique(np.linspace(0, len(fpr) - 1, points).astype(int))
    order = np.argsort(-(np.asarray(rank, dtype=float) if rank is not None else s), kind="stable")
    cum = np.cumsum(y[order])
    ks = np.unique(np.linspace(1, len(y), points).astype(int))
    pak = [{"k_share": round(k / len(y), 4), "precision": round(float(cum[k - 1] / k), 4),
            "recall": round(float(cum[k - 1] / max(y.sum(), 1)), 4)} for k in ks]
    if value is not None:
        v = np.clip(np.asarray(value, dtype=float), 0, None) * y
        cv = np.cumsum(v[order])
        for p_, k in zip(pak, ks):
            p_["value_recall"] = round(float(cv[k - 1] / max(v.sum(), 1e-9)), 4)
    out = {"roc": [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx],
           "precision_at_k": pak}
    if proba and np.nanmin(s) >= 0 and np.nanmax(s) <= 1:
        bins = np.linspace(0, 1, 11)
        b = np.clip(np.digitize(s, bins) - 1, 0, 9)
        out["calibration"] = [{"pred": round(float(s[b == i].mean()), 4), "obs": round(float(y[b == i].mean()), 4),
                               "n": int((b == i).sum())} for i in range(10) if (b == i).sum() >= 20]
    return out


def bootstrap_lift(y: np.ndarray, s_model: np.ndarray, s_rule: np.ndarray, metric: str, n: int = 200,
                   value: np.ndarray | None = None, r_model: np.ndarray | None = None) -> dict:
    """Gepaartes Bootstrap: relative Verbesserung der Hauptmetrik gegenüber der Regel."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y).astype(int)
    vals, lifts = [], []
    N = len(y)
    for _ in range(n):
        i = rng.integers(0, N, N)
        if y[i].sum() == 0:
            continue
        v = value[i] if value is not None else None
        mm = binary_metrics(y[i], s_model[i], v, proba=False,
                            rank=r_model[i] if r_model is not None else None)[metric]
        mr = binary_metrics(y[i], s_rule[i], v, proba=False)[metric]
        vals.append(mm)
        lifts.append((mm - mr) / mr if mr else 0.0)
    return _ci(vals, lifts)


def _ci(vals, lifts) -> dict:
    vals, lifts = np.asarray(vals), np.asarray(lifts)
    return {"metric_ci": [float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))] if len(vals) else None,
            "lift_ci": [float(np.quantile(lifts, 0.05)), float(np.quantile(lifts, 0.95))] if len(lifts) else None}


def metric_ci(values_per_unit: np.ndarray, n: int = 200) -> list:
    rng = np.random.default_rng(SEED)
    v = np.asarray(values_per_unit, dtype=float)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))]


def ranking_metrics(recs: list[list[int]], truth: list[set], k: int = 12) -> dict:
    ap, rec, hit = [], [], []
    for r, t in zip(recs, truth):
        if not t:
            continue
        r = r[:k]
        hits, s = 0, 0.0
        for i, p in enumerate(r):
            if p in t:
                hits += 1
                s += hits / (i + 1)
        ap.append(s / min(len(t), k))
        rec.append(hits / len(t))
        hit.append(1.0 if hits else 0.0)
    ap, rec, hit = map(np.asarray, (ap, rec, hit))
    return {"map_at_12": float(ap.mean()), "recall_at_12": float(rec.mean()), "hit_rate_at_12": float(hit.mean()),
            "n": int(len(ap)), "_ap": ap}


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray) -> dict:
    """y_* : Reihen × Horizont. scale: mittlerer saisonal-naiver Fehler je Reihe (In-Sample) für MASE."""
    err = np.abs(y_true - y_pred)
    wape = err.sum() / max(np.abs(y_true).sum(), 1e-9)
    mase = np.nanmean(err.mean(axis=1) / np.maximum(scale, 1e-9))
    bias = (y_pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-9)
    return {"wape": float(wape), "mase": float(mase), "bias": float(bias), "n": int(y_true.size),
            "series": int(y_true.shape[0])}
