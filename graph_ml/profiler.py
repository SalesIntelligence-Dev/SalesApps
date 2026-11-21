"""
Profil für Stufe 2 (Profilcheck)
================================
Erzeugt aus einem Datenexport ein Profil mit ausschließlich aggregierten
Kennzahlen: Zeilenzahlen, Füllgrad, Verknüpfungsquoten, Zeitraum, Fallzahlen
je Zeitfenster, Cold-Start-Anteile, Verteilungsverschiebung (PSI).

Das Profil enthält KEINE Datenzeilen, keine Schlüsselwerte und keine
Beispielwerte. Kategoriewerte erscheinen nur für Spalten mit wenigen
Ausprägungen (keine Schlüssel) und nur ab MIN_GROUP Vorkommen.
Damit verlassen keine personenbezogenen Daten das Unternehmen.
"""
from __future__ import annotations

import re
from collections import deque

import numpy as np
import pandas as pd

from .graph_mapping import _key_values
from .semantics import bind_fields, schema_view

PROFILE_VERSION = 1
MIN_GROUP = 10
HISTORY_DAYS = 365
STORNO_RE = re.compile(r"storno|cancel|retour|return|gutschrift|credit|annul", re.I)
NON_ARTICLE = {"POST", "DOT", "M", "D", "C2", "C3", "BANK CHARGES", "AMAZONFEE", "CRUK", "S", "B", "PADS",
               "ADJUST", "ADJUST2"}
KEY_ROLES = {"PK", "FK", "Schlüssel-Kandidat"}

EVENT_FIELDS = {"customer_key": "customer", "order_key": "order", "event_time": "time", "product_key": "product",
                "quantity": "qty", "price": "price", "product_group": "group", "product_text": "text",
                "country": "country", "doc_type": "doc_type"}


# ── Ereignistabelle aus den Tabellen zusammensetzen ───────────────────────────

def _lookup(keys: pd.Series, target: pd.DataFrame, key_col: str, value_col: str) -> pd.Series:
    lut = pd.Series(target[value_col].to_numpy(), index=_key_values(target[key_col]).to_numpy())
    lut = lut[~lut.index.duplicated()]
    return pd.Series(lut.reindex(keys.to_numpy()).to_numpy(), index=keys.index)


def _fk_path(relations: list[dict], src: str, dst: str, max_hops: int = 3) -> list[dict] | None:
    out: dict[str, list[dict]] = {}
    for r in relations:
        if r["kind"] in ("fk", "declared"):
            out.setdefault(r["from_table"], []).append(r)
    queue, seen = deque([(src, [])]), {src}
    while queue:
        t, path = queue.popleft()
        if t == dst:
            return path
        if len(path) >= max_hops:
            continue
        for r in out.get(t, []):
            if r["to_table"] not in seen:
                seen.add(r["to_table"])
                queue.append((r["to_table"], path + [r]))
    return None


def assemble_events(tables: dict[str, pd.DataFrame], mapping: dict, bindings: dict) -> pd.DataFrame | None:
    """Flache Ereignistabelle (eine Zeile je Position) über Fremdschlüssel-Pfade."""
    base_field = next((f for f in ("product_key", "order_key", "event_time") if bindings.get(f)), None)
    if not base_field:
        return None
    base = bindings[base_field]["table"]
    df = tables[base]
    ev = pd.DataFrame(index=df.index)
    for field, name in EVENT_FIELDS.items():
        b = bindings.get(field)
        if not b or b.get("via") != "column" or b["table"] not in tables:
            continue
        if b["table"] == base:
            ev[name] = df[b["column"]].to_numpy()
            continue
        path = _fk_path(mapping["relations"], base, b["table"])
        if not path:
            continue
        keys = _key_values(df[path[0]["from_column"]])
        for i, r in enumerate(path):
            target = tables[r["to_table"]]
            value_col = path[i + 1]["from_column"] if i + 1 < len(path) else b["column"]
            vals = _lookup(keys, target, r["to_column"], value_col)
            keys = _key_values(vals) if i + 1 < len(path) else vals
        ev[name] = keys.to_numpy()
    return finalize_events(ev)


def finalize_events(ev: pd.DataFrame) -> pd.DataFrame:
    if "time" in ev:
        ev["time"] = pd.to_datetime(ev["time"], errors="coerce")
    for c in ("qty", "price"):
        if c in ev:
            ev[c] = pd.to_numeric(ev[c], errors="coerce").astype("float64")
    if "customer" in ev:
        ev["customer"] = _key_values(pd.Series(ev["customer"].to_numpy()))
    if "storno" not in ev:
        if "doc_type" in ev:
            ev["storno"] = ev["doc_type"].astype("string").str.contains(STORNO_RE).fillna(False).to_numpy(bool)
        elif "qty" in ev:
            ev["storno"] = (ev["qty"] < 0).to_numpy()
        else:
            ev["storno"] = False
    for c in ("product", "group", "text", "country", "order"):
        if c in ev and not isinstance(ev[c].dtype, pd.CategoricalDtype):
            ev[c] = ev[c].astype("string").astype("category")
    return ev


def events_from_online_retail(raw: pd.DataFrame) -> pd.DataFrame:
    """Online Retail II liegt bereits flach vor – direkt übernehmen (speicherschonend)."""
    from .data_loader import cat_startswith
    ev = pd.DataFrame({
        "customer": raw["customer_id"].to_numpy(dtype="float64", na_value=np.nan),
        "order": raw["invoice"], "time": raw["invoice_date"], "product": raw["stock_code"],
        "qty": raw["quantity"].astype("float64"), "price": raw["price_gbp"].astype("float64"),
        "text": raw["description"], "country": raw["country"],
    })
    ev["storno"] = cat_startswith(raw["invoice"], "C")
    return ev


# ── Kennzahlen ────────────────────────────────────────────────────────────────

def _cutoffs(tmin: pd.Timestamp, tmax: pd.Timestamp, horizon_days: int = 90) -> list[pd.Timestamp]:
    first = (tmin.normalize() + pd.Timedelta(days=HISTORY_DAYS))
    if first.day != 1:
        first = (first + pd.offsets.MonthBegin(1)).normalize()
    last = (tmax.normalize() - pd.Timedelta(days=horizon_days))
    last = last.replace(day=1)
    return list(pd.date_range(first, last, freq="MS")) if last >= first else []


def _psi(a: np.ndarray, b: np.ndarray) -> float:
    a = np.clip(a / max(a.sum(), 1e-12), 1e-4, None)
    b = np.clip(b / max(b.sum(), 1e-12), 1e-4, None)
    return float(np.sum((a - b) * np.log(a / b)))


def _cat_psi(sa: pd.Series, sb: pd.Series, top: int = 20) -> float:
    cats = sa.value_counts().head(top).index
    ca = sa.where(sa.isin(cats), "__other__").value_counts()
    cb = sb.where(sb.isin(cats), "__other__").value_counts()
    idx = ca.index.union(cb.index)
    return _psi(ca.reindex(idx, fill_value=0).to_numpy(float), cb.reindex(idx, fill_value=0).to_numpy(float))


def _product_strings(ev: pd.DataFrame) -> pd.Series:
    if isinstance(ev["product"].dtype, pd.CategoricalDtype):
        return pd.Series(ev["product"].cat.categories.astype(str))
    return pd.Series(ev["product"].dropna().astype(str).unique())


def event_stats(ev: pd.DataFrame) -> dict:
    """Alle Kennzahlen aus der Ereignistabelle – nur Aggregate."""
    out: dict = {"rows": int(len(ev)), "fields": [c for c in ev.columns if c != "storno"]}
    has = set(ev.columns)
    storno = ev["storno"].to_numpy(bool) if "storno" in has else np.zeros(len(ev), bool)
    out["storno_rows"] = int(storno.sum())

    # Qualität
    q: dict = {}
    cols = [c for c in ("order", "product", "text", "qty", "time", "price", "customer", "country") if c in has]
    q["duplicate_rows"] = int(ev.duplicated(subset=cols).sum()) if cols else 0
    q["duplicate_share"] = round(q["duplicate_rows"] / max(len(ev), 1), 5)
    if "price" in has:
        q["price_zero_rows"] = int((ev["price"] == 0).sum())
        q["price_negative_rows"] = int((ev["price"] < 0).sum())
    if "qty" in has:
        q["negative_qty_without_storno"] = int(((ev["qty"] < 0).to_numpy() & ~storno).sum())
    if "product" in has:
        prods = _product_strings(ev)
        upper = prods.str.upper()
        q["product_keys"] = int(len(prods))
        q["product_case_variants"] = int(upper.duplicated(keep=False).sum())
        non_art_codes = set(prods[upper.isin(NON_ARTICLE) | ~prods.str.contains(r"\d") |
                                  upper.str.startswith("TEST")])
        pcol = ev["product"].astype(str) if not isinstance(ev["product"].dtype, pd.CategoricalDtype) else None
        if pcol is None:
            mask_codes = ev["product"].cat.categories.astype(str).isin(non_art_codes)
            codes = ev["product"].cat.codes.to_numpy()
            q["non_article_rows"] = int(np.where(codes >= 0, mask_codes[np.clip(codes, 0, None)], False).sum())
        else:
            q["non_article_rows"] = int(pcol.isin(non_art_codes).sum())
        q["non_article_codes"] = int(len(non_art_codes))
    if "product" in has and "text" in has:
        pt = ev[["product", "text"]].dropna().drop_duplicates()
        n = pt.groupby("product", observed=True).size()
        q["products_multi_text"] = int((n > 1).sum())
        q["products_with_text"] = int((n >= 1).sum())
    for c in ("customer", "product", "time", "price", "qty"):
        if c in has:
            q[f"null_share_{c}"] = round(float(pd.isna(ev[c]).mean()), 5)
    out["quality"] = q

    if "time" not in has:
        return out
    t = ev["time"]
    tmin, tmax = t.min(), t.max()
    months = (tmax - tmin).days / 30.44
    monthly = t.dt.to_period("M").value_counts().sort_index()
    full = pd.period_range(tmin.to_period("M"), tmax.to_period("M"), freq="M")
    out["time"] = {"min": tmin.isoformat(), "max": tmax.isoformat(), "months": round(months, 1),
                   "months_with_events": int(len(monthly)), "months_without_events": int(len(full) - len(monthly))}
    cut = _cutoffs(tmin, tmax)
    out["cutoffs"] = [c.date().isoformat() for c in cut]

    stats: dict = {}
    # Käufe (ohne Storno) mit Kundenbezug, je Beleg
    if "customer" in has:
        pm = ~storno & ev["customer"].notna().to_numpy()
        P = ev.loc[pm, [c for c in ("customer", "order", "time", "product", "group") if c in has]]
        key = ["customer", "order"] if "order" in has else ["customer", "time"]
        inv = P.drop_duplicates(key)[["customer", "time"]].sort_values(["customer", "time"])
        c_arr, t_arr = inv["customer"].to_numpy(), inv["time"].to_numpy()

        # Churn
        rows = []
        for T in cut:
            T64 = np.datetime64(T)
            hist = (t_arr >= T64 - np.timedelta64(HISTORY_DAYS, "D")) & (t_arr < T64)
            fut = (t_arr >= T64) & (t_arr < T64 + np.timedelta64(90, "D"))
            active = np.unique(c_arr[hist])
            pos = np.setdiff1d(active, np.unique(c_arr[fut]), assume_unique=True).size
            rows.append({"cutoff": T.date().isoformat(), "eligible": int(active.size), "positives": int(pos)})
        stats["churn"] = {"windows": rows}

        # Neukunden: Erstkauf und zweiter Beleg
        k = inv.groupby("customer").cumcount().to_numpy()
        first = pd.Series(t_arr[k == 0], index=c_arr[k == 0])
        second = pd.Series(t_arr[k == 1], index=c_arr[k == 1]).reindex(first.index)
        rows = []
        for T in cut:
            new = (first >= T - pd.Timedelta(days=90)) & (first < T)
            rep = new & ((second - first) <= pd.Timedelta(days=90))
            rows.append({"cutoff": T.date().isoformat(), "eligible": int(new.sum()), "positives": int(rep.sum())})
        stats["lead_scoring"] = {"windows": rows}

        # Cold Start (Kunden)
        n_orders = inv.groupby("customer").size()
        last_year = inv[inv["time"] >= tmax - pd.Timedelta(days=HISTORY_DAYS)]["customer"].unique()
        new_last90 = (first >= tmax - pd.Timedelta(days=90)).sum()
        out["cold_start"] = {
            "customers": int(n_orders.size),
            "customers_lt3_orders": int((n_orders < 3).sum()),
            "share_lt3_orders": round(float((n_orders < 3).mean()), 4),
            "share_single_order": round(float((n_orders == 1).mean()), 4),
            "new_customers_last90": int(new_last90),
            "share_new_of_active": round(float(new_last90 / max(len(last_year), 1)), 4),
            "avg_orders_per_customer": round(float(n_orders.mean()), 2),
        }

        # Empfehlung / Cross-Selling im Testfenster
        if cut and "product" in has:
            Tt = cut[-1]
            hist = P[(P["time"] < Tt) & (P["time"] >= Tt - pd.Timedelta(days=HISTORY_DAYS))]
            fut = P[(P["time"] >= Tt) & (P["time"] < Tt + pd.Timedelta(days=90))]
            fut = fut[fut["customer"].isin(hist["customer"].unique())]
            pairs_f = fut[["customer", "product"]].drop_duplicates()
            all_hist = P[P["time"] < Tt]
            known_products = set(all_hist["product"].unique())
            pairs_h = all_hist[["customer", "product"]].drop_duplicates()
            m = pairs_f.merge(pairs_h.assign(_r=1), on=["customer", "product"], how="left")
            repeat = m["_r"].notna().to_numpy()
            eligible = int(pairs_f["customer"].nunique())
            stats["recommendation"] = {
                "cutoff": Tt.date().isoformat(), "eligible": eligible, "positives": int(len(pairs_f)),
                "avg_products_per_customer": round(len(pairs_f) / max(eligible, 1), 1),
                "share_repeat_pairs": round(float(repeat.mean()), 4) if len(repeat) else None,
                "share_new_products": round(float((~pairs_f["product"].isin(known_products)).mean()), 4)
                if len(pairs_f) else None,
            }
            grp_col = "group" if "group" in has else None
            if grp_col:
                gf = fut[["customer", "group"]].drop_duplicates()
                gh = all_hist[["customer", "group"]].drop_duplicates().assign(_r=1)
                gm = gf.merge(gh, on=["customer", "group"], how="left")
                pos = int(gm.loc[gm["_r"].isna(), "customer"].nunique())
                stats["cross_selling"] = {"cutoff": Tt.date().isoformat(), "eligible": eligible, "positives": pos,
                                          "basis": "Artikelgruppe"}
            else:
                pos = int(m.loc[~repeat, "customer"].nunique())
                stats["cross_selling"] = {"cutoff": Tt.date().isoformat(), "eligible": eligible,
                                          "positives": None, "proxy_positives": pos,
                                          "basis": "keine Artikelgruppe – Proxy: Kauf eines für den Kunden neuen Artikels"}
            # Artikel-Kontext
            sales_per_product = P.groupby("product", observed=True).size()
            out["cold_start"]["products"] = int(sales_per_product.size)
            out["cold_start"]["share_products_lt5_sales"] = round(float((sales_per_product < 5).mean()), 4)
            out["cold_start"]["share_new_products_test"] = stats["recommendation"]["share_new_products"]

        # Retouren: Storno → vorheriger Kauf desselben Kunden und Artikels
        if "product" in has and storno.any():
            S = ev.loc[storno & ev["customer"].notna().to_numpy(), ["customer", "product", "time"]].copy()
            B = ev.loc[pm, ["customer", "product", "time"]].copy()
            for d in (S, B):
                d["product"] = d["product"].astype(str) if not isinstance(d["product"].dtype, pd.CategoricalDtype) \
                    else d["product"].cat.codes
            B["bt"] = B["time"]
            mm = pd.merge_asof(S.sort_values("time"), B.sort_values("time"), on="time", by=["customer", "product"],
                               direction="backward", tolerance=pd.Timedelta(days=90))
            lag = (mm["time"] - mm["bt"]).dt.days
            matched90 = int(mm["bt"].notna().sum())
            matched30 = int((lag <= 30).sum())
            rows = []
            for T in cut:
                w = (mm["time"] >= T) & (mm["time"] < T + pd.Timedelta(days=90))
                sales = ((B["time"] >= T) & (B["time"] < T + pd.Timedelta(days=90))).sum()
                rows.append({"cutoff": T.date().isoformat(), "eligible": int(sales),
                             "positives": int((w & (lag <= 30)).sum())})
            stats["returns_root_cause"] = {
                "storno_rows": int(storno.sum()), "storno_rows_with_customer": int(len(S)),
                "matched_90d": matched90, "matched_30d": matched30,
                "match_rate_90d": round(matched90 / max(len(S), 1), 4), "windows": rows,
            }

    # Absatzprognose: Wochenreihen je Artikel
    if "product" in has and "qty" in has:
        sm = ~storno
        wk = ev.loc[sm, "time"].dt.to_period("W")
        pw = pd.DataFrame({"p": ev.loc[sm, "product"].to_numpy(), "w": wk.to_numpy(), "q": ev.loc[sm, "qty"].to_numpy()})
        active_weeks = pw.drop_duplicates(["p", "w"]).groupby("p", observed=True).size()
        n_weeks = int(round((tmax - tmin).days / 7))
        top = pw.groupby("p", observed=True)["q"].sum().sort_values(ascending=False).head(100).index
        zero_share = float(1 - active_weeks.reindex(top).fillna(0).mean() / max(n_weeks, 1)) if len(top) else None
        vol = pw.groupby("p", observed=True)["q"].sum()
        long_ids = active_weeks[active_weeks >= 52].index
        vol_share = float(vol.reindex(long_ids).sum() / vol.sum()) if vol.sum() > 0 else None
        stats["demand_forecast"] = {
            "history_weeks": n_weeks, "series": int(active_weeks.size),
            "volume_share_ge52": round(vol_share, 4) if vol_share is not None else None,
            "series_ge26_weeks": int((active_weeks >= 26).sum()), "series_ge52_weeks": int((active_weeks >= 52).sum()),
            "zero_share_top100": round(zero_share, 4) if zero_share is not None else None,
            "positives": int((active_weeks >= 52).sum()),
        }

    # Margen: nur Preisstreuung (Einstandspreis wird über Bindung geprüft)
    if "product" in has and "price" in has:
        pp = ev.loc[~storno & (ev["price"] > 0).to_numpy(), ["product", "price"]]
        g = pp.groupby("product", observed=True)["price"]
        cnt, std, mean, nun = g.size(), g.std(), g.mean(), g.nunique()
        cv = (std / mean)[cnt >= 10]
        stats["margin"] = {"products_multi_price_share": round(float((nun > 1).mean()), 4),
                           "median_price_cv": round(float(cv.median()), 4) if len(cv) else None}

    # Verteilungsverschiebung (PSI) erstes vs. letztes Jahr
    if months >= 18:
        a = ev["time"] < tmin + pd.Timedelta(days=HISTORY_DAYS)
        b = ev["time"] >= tmax - pd.Timedelta(days=HISTORY_DAYS)
        shift = {}
        if "country" in has:
            shift["country_mix"] = round(_cat_psi(ev.loc[a, "country"].astype(str), ev.loc[b, "country"].astype(str)), 4)
        if "product" in has:
            shift["product_mix"] = round(_cat_psi(ev.loc[a, "product"].astype(str), ev.loc[b, "product"].astype(str),
                                                  top=100), 4)
        if {"order", "qty", "price"} <= has:
            val = ev.loc[~storno, ["order", "time"]].assign(v=(ev["qty"] * ev["price"])[~storno].to_numpy())
            basket = val.groupby("order", observed=True).agg(v=("v", "sum"), t=("time", "min"))
            va, vb = basket.loc[basket["t"] < tmin + pd.Timedelta(days=HISTORY_DAYS), "v"], \
                basket.loc[basket["t"] >= tmax - pd.Timedelta(days=HISTORY_DAYS), "v"]
            if len(va) > 50 and len(vb) > 50:
                edges = np.unique(np.quantile(va, np.linspace(0, 1, 11)))
                ha = np.histogram(va, bins=edges)[0]
                hb = np.histogram(np.clip(vb, edges[0], edges[-1]), bins=edges)[0]
                shift["basket_value"] = round(_psi(ha.astype(float), hb.astype(float)), 4)
        shift["max"] = max(shift.values()) if shift else None
        out["shift"] = shift
    out["usecase_stats"] = stats
    return out


# ── Profil ────────────────────────────────────────────────────────────────────

def _column_profile(c: dict, df: pd.DataFrame | None) -> dict:
    keep = {k: c.get(k) for k in ("name", "kind", "roles", "null_rate", "n_unique", "unique_ratio",
                                  "avg_len", "space_share", "sql_type") if c.get(k) is not None}
    is_key = bool(KEY_ROLES & set(c.get("roles", [])))
    if c.get("kind") == "datetime":
        keep["min"], keep["max"] = c.get("min"), c.get("max")
    elif c.get("kind") in ("int", "float") and not is_key and df is not None:
        s = pd.to_numeric(df[c["name"]], errors="coerce").dropna()
        if len(s):
            keep.update(zero_share=round(float((s == 0).mean()), 5), negative_share=round(float((s < 0).mean()), 5),
                        p01=float(s.quantile(0.01)), p50=float(s.median()), p99=float(s.quantile(0.99)))
    if (df is not None and c.get("kind") == "string" and not is_key and "Freitext" not in c.get("roles", [])
            and (c.get("n_unique") or 0) <= 25):
        vc = df[c["name"]].value_counts()
        vc = vc[vc >= MIN_GROUP]
        keep["top_values"] = {str(k): int(v) for k, v in vc.head(25).items()}
    return keep


def build_profile(*, dataset_name: str, mapping: dict, tables: dict[str, pd.DataFrame] | None,
                  events: pd.DataFrame | None, source: str, extra_findings: list | None = None) -> dict:
    view = schema_view(mapping)
    bindings = bind_fields(view)
    tables_out = []
    for t in mapping["tables"]:
        df = (tables or {}).get(t["name"])
        tables_out.append({
            "name": t["name"], "label": t.get("label", t["name"]), "rows": t.get("rows"),
            "n_columns": t.get("n_columns"), "primary_key": t.get("primary_key"),
            "time_columns": t.get("time_columns", []),
            "columns": [_column_profile(c, df) for c in t["columns"]],
        })
    ev_stats = event_stats(events) if events is not None and len(events) else None
    findings = [{k: f[k] for k in ("severity", "check", "playbook", "title", "text")} for f in mapping.get("findings", [])]
    return {
        "profile_version": PROFILE_VERSION,
        "dataset_name": dataset_name,
        "source": source,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "privacy": {
            "contains_rows": False, "contains_key_values": False, "contains_examples": False,
            "min_group_size": MIN_GROUP,
            "note": "Nur aggregierte Kennzahlen – es verlassen keine personenbezogenen Daten das Unternehmen.",
        },
        "tables": tables_out,
        "relations": [{k: r.get(k) for k in ("id", "kind", "label", "from_table", "from_column", "to_table",
                                              "to_column", "rows", "linked_rows", "null_rows", "orphan_rows",
                                              "linkage_rate", "status", "note")} for r in mapping["relations"]],
        "paths": mapping.get("paths", []),
        "graph": mapping.get("graph"),
        "summary": mapping.get("summary"),
        "findings": findings + list(extra_findings or []),
        "insights": mapping.get("insights", []),
        "bindings": bindings,
        "derived_columns": list(mapping.get("derived_columns", [])),
        "events": ev_stats,
    }


def profile_to_mapping(profile: dict) -> dict:
    """Profil → Mapping-Struktur für Schritt 2 (Graph-Ansicht ohne Rohdaten)."""
    return {
        "artifact_version": None, "dataset_id": None, "mode": "data", "from_profile": True,
        "generated_at": profile.get("generated_at"),
        "tables": [{**t, "columns": [{**c, "roles": c.get("roles", [])} for c in t["columns"]]}
                   for t in profile["tables"]],
        "relations": profile["relations"], "paths": profile.get("paths", []),
        "graph": profile.get("graph") or {"node_types": [], "edge_types": [], "n_nodes": None, "n_edges": None},
        "findings": profile.get("findings", []), "insights": profile.get("insights", []),
        "summary": profile.get("summary") or {},
        "derived_columns": profile.get("derived_columns", []),
    }


def assert_privacy(profile: dict) -> None:
    """Grobe Selbstprüfung vor dem Export: keine Beispielwerte, keine Werte aus Schlüsselspalten."""
    for t in profile["tables"]:
        for c in t["columns"]:
            if "examples" in c:
                raise ValueError(f"Beispielwerte in {t['name']}.{c['name']}")
            if KEY_ROLES & set(c.get("roles", [])) and "top_values" in c:
                raise ValueError(f"Schlüsselwerte in {t['name']}.{c['name']}")
