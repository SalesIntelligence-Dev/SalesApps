"""
CPU-Engines: Regel-Baselines, LightGBM, LightGBM + Graph-Features, ItemKNN,
LightGBM-Ranker (Empfehlung) und saisonal-naive bzw. LightGBM-Prognose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .. import backtest as bt
from .base import Engine


def _lgb():
    import lightgbm as lgb
    return lgb


def _frame(df: pd.DataFrame, cols: list[str], cats: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cats:
        if c in X:
            X[c] = X[c].where(X[c] >= 0).astype("category")
    return X.astype({c: "float32" for c in X.columns if c not in cats})


# ── Regel-Baselines ───────────────────────────────────────────────────────────

class RuleEngine(Engine):
    id, name, klass, kind, tasks, proba = "rule", "Regel-Baseline", "Regel", "cpu", ("binary",), False

    def fit(self, task):
        return self

    def predict(self, task):
        return task.test[task.rule_col].to_numpy(dtype=float)


# ── LightGBM (binär) ──────────────────────────────────────────────────────────

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40,
                  feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
                  seed=bt.SEED, verbose=-1, num_threads=4)


class LGBMEngine(Engine):
    id, name, klass, kind, tasks = "lgbm", "LightGBM", "Baum", "cpu", ("binary",)
    license = "MIT"
    use_graph = False

    def _cols(self, task):
        return task.features if self.use_graph else task.base_features

    def fit(self, task):
        lgb = _lgb()
        cols = self._cols(task)
        tr = task.train[task.train["cutoff"] < task.val_cutoff]
        va = task.train[task.train["cutoff"] == task.val_cutoff]
        wcol = task.extra.get("weight")
        w = (lambda d: 1 + np.log1p(d[wcol].clip(lower=0).to_numpy())) if wcol else (lambda d: None)  # noqa: E731
        dtr = lgb.Dataset(_frame(tr, cols, task.cat_features), tr[task.label], weight=w(tr), categorical_feature=
                          [c for c in task.cat_features if c in cols] or "auto", free_raw_data=False)
        dva = lgb.Dataset(_frame(va, cols, task.cat_features), va[task.label], weight=w(va), reference=dtr)
        self.model = lgb.train(LGB_PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(60, verbose=False)])
        self.best_iter = self.model.best_iteration or self.model.current_iteration()
        if wcol:
            self.notes.append("umsatzgewichtet trainiert (Gewicht 1 + log(Umsatz))")
        self.cols = cols
        return self

    def predict(self, task):
        return self.model.predict(_frame(task.test, self.cols, task.cat_features), num_iteration=self.best_iter)

    def explain(self, task):
        X = _frame(task.test, self.cols, task.cat_features)
        contrib = self.model.predict(X, num_iteration=self.best_iter, pred_contrib=True)[:, :-1]
        imp = np.abs(contrib).mean(axis=0)
        order = np.argsort(-imp)[:15]
        glob = [{"feature": self.cols[i], "label": bt.FEATURE_LABELS.get(self.cols[i], self.cols[i]),
                 "importance": round(float(imp[i]), 4)} for i in order]
        scores = self.predict(task)
        top = int(np.argmax(scores))
        row = task.test.iloc[top]
        c = contrib[top]
        ex_order = np.argsort(-np.abs(c))[:6]
        example = {"id": str(row.get(task.id_col, top)), "score": round(float(scores[top]), 4),
                   "label": int(row[task.label]),
                   "contributions": [{"feature": self.cols[i], "label": bt.FEATURE_LABELS.get(self.cols[i], self.cols[i]),
                                      "value": _fmt_val(row[self.cols[i]]), "effect": round(float(c[i]), 4)}
                                     for i in ex_order]}
        return {"global": glob, "example": example, "best_iteration": int(self.best_iter)}


def _fmt_val(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(f):
        return None
    return round(f, 3) if abs(f) < 1000 else round(f)


class LGBMGraphEngine(LGBMEngine):
    id, name, klass = "lgbm_graph", "LightGBM + Graph-Features", "Graph-ML light"
    use_graph = True


# ── Empfehlung (Rangliste) ────────────────────────────────────────────────────

def _top_pop(snap, k=40):
    return list(snap["pop30"].sort_values(ascending=False).index[:k])


def _rule_recs(snap, k=12):
    pop = _top_pop(snap, 60)
    R = snap["R"].tocsr()
    recs = []
    for u in range(R.shape[0]):
        row = R.getrow(u)
        own = row.indices[np.argsort(-row.data, kind="stable")]
        r = list(own[:k])
        r += [p for p in pop if p not in set(r)][: k - len(r)]
        recs.append(r)
    return recs


class RecoRuleEngine(Engine):
    id, name, klass, kind, tasks, proba = "rule", "Regel-Baseline", "Regel", "cpu", ("ranking",), False

    def fit(self, task):
        return self

    def predict(self, task):
        snap = bt.reco_snapshot(bt.get_base(), bt.T_TEST)
        return _rule_recs(snap), snap["truth"]


class PopularityEngine(Engine):
    id, name, klass, kind, tasks, proba = "popularity", "Topseller 30 Tage", "Regel", "cpu", ("ranking",), False

    def fit(self, task):
        return self

    def predict(self, task):
        snap = bt.reco_snapshot(bt.get_base(), bt.T_TEST)
        pop = _top_pop(snap, 12)
        return [pop for _ in range(len(snap["users"]))], snap["truth"]


def _itemknn_scores(snap) -> np.ndarray:
    A = snap["A"].astype(np.float32)
    co = (A.T @ A).tocsr()
    norm = np.sqrt(np.asarray(A.sum(axis=0)).ravel()) + 1e-6
    S = sparse.diags(1 / norm) @ co @ sparse.diags(1 / norm)
    Rb = snap["R"].copy()
    Rb.data = np.log1p(Rb.data)
    return np.asarray((Rb @ S).todense(), dtype=np.float32)


class ItemKNNEngine(Engine):
    id, name, klass, kind, tasks, proba = "itemknn", "ItemKNN (Co-Kauf-Graph)", "Graph-ML light", "cpu", ("ranking",), False

    def fit(self, task):
        return self

    def predict(self, task):
        snap = bt.reco_snapshot(bt.get_base(), bt.T_TEST)
        sc = _itemknn_scores(snap)
        recs = np.argsort(-sc, axis=1)[:, :12]
        return [list(r) for r in recs], snap["truth"]


def _rank_pairs(snap, with_label: bool):
    b = bt.get_base()
    sc = _itemknn_scores(snap)
    R = snap["R"].tocsr()
    pop30, pop90 = snap["pop30"], snap["pop90"]
    gpop = _top_pop(snap, 30)
    T = snap["T"]
    users = snap["users"]
    last = snap["last"]
    emb_u = (R @ b.svd_items) / np.maximum(np.asarray(R.sum(axis=1)), 1)
    rows = []
    knn_top = np.argsort(-sc, axis=1)[:, :60]
    for u in range(len(users)):
        row = R.getrow(u)
        own = dict(zip(row.indices, row.data))
        cand = set(knn_top[u]) | set(sorted(own, key=own.get, reverse=True)[:40]) | set(gpop)
        cand = np.fromiter(cand, dtype=np.int64)
        knn_rank = pd.Series(np.arange(60), index=knn_top[u])
        cust = users[u]
        for p in cand:
            lt = last.get((cust, p))
            rows.append((u, p, own.get(p, 0.0), (T - lt).days if lt is not None else 999, sc[u, p],
                         knn_rank.get(p, 99), pop30.get(p, 0), pop90.get(p, 0), float(emb_u[u] @ b.svd_items[p]),
                         len(own)))
    df = pd.DataFrame(rows, columns=["u", "p", "own_cnt", "own_recency", "knn", "knn_rank", "pop30", "pop90",
                                     "svd_dot", "u_items"])
    if with_label:
        truth = snap["truth"]
        df["y"] = [int(p in truth[u]) for u, p in zip(df["u"], df["p"])]
    return df


RANK_FEATS = ["own_cnt", "own_recency", "knn", "knn_rank", "pop30", "pop90", "svd_dot", "u_items"]


class LGBMRankEngine(Engine):
    id, name, klass, kind, tasks, proba = "lgbm_rank", "LightGBM-Ranker (Graph-Kandidaten)", "Graph-ML light", "cpu", ("ranking",), False
    license = "MIT"

    def fit(self, task):
        lgb = _lgb()
        b = bt.get_base()
        tr = _rank_pairs(bt.reco_snapshot(b, task.snapshots[0]), True)
        va = _rank_pairs(bt.reco_snapshot(b, task.snapshots[1]), True)
        p = dict(LGB_PARAMS, objective="binary")
        self.model = lgb.train(p, lgb.Dataset(tr[RANK_FEATS], tr["y"]), 800,
                               valid_sets=[lgb.Dataset(va[RANK_FEATS], va["y"])],
                               callbacks=[lgb.early_stopping(40, verbose=False)])
        imp = self.model.feature_importance("gain")
        self.importance = [{"feature": f, "label": RANK_LABELS[f], "importance": round(float(v), 1)}
                           for f, v in sorted(zip(RANK_FEATS, imp), key=lambda x: -x[1])]
        return self

    def predict(self, task):
        snap = bt.reco_snapshot(bt.get_base(), bt.T_TEST)
        te = _rank_pairs(snap, False)
        te["s"] = self.model.predict(te[RANK_FEATS], num_iteration=self.model.best_iteration)
        top = te.sort_values(["u", "s"], ascending=[True, False]).groupby("u").head(12)
        recs = top.groupby("u")["p"].apply(list).reindex(range(len(snap["users"]))).tolist()
        return [r if isinstance(r, list) else [] for r in recs], snap["truth"]

    def explain(self, task):
        return {"global": self.importance}


RANK_LABELS = {"own_cnt": "eigene Käufe des Artikels", "own_recency": "Tage seit eigenem Kauf",
               "knn": "Graph: Co-Kauf-Score (ItemKNN)", "knn_rank": "Graph: Rang im Co-Kauf-Score",
               "pop30": "Beliebtheit 30 Tage", "pop90": "Beliebtheit 90 Tage",
               "svd_dot": "Graph: Einbettungs-Ähnlichkeit", "u_items": "Artikel des Kunden"}


# ── Absatzprognose ────────────────────────────────────────────────────────────

def forecast_matrix(task, origin):
    Y = task.extra["Y"].to_numpy(dtype=float)
    weeks = task.extra["weeks"]
    o = weeks.get_loc(origin)
    return Y, o


def seasonal_naive(Y: np.ndarray, o: int, H: int) -> np.ndarray:
    out = np.zeros((Y.shape[0], H))
    for h in range(H):
        t = o + h
        out[:, h] = Y[:, t - 52] if t - 52 >= 0 else Y[:, max(0, o - 4):o].mean(axis=1)
    return out


def naive_scale(Y: np.ndarray, o: int) -> np.ndarray:
    hist = Y[:, :o]
    if hist.shape[1] <= 52:
        return np.abs(np.diff(hist, axis=1)).mean(axis=1)
    return np.abs(hist[:, 52:] - hist[:, :-52]).mean(axis=1)


class NaiveForecastEngine(Engine):
    id, name, klass, kind, tasks, proba = "rule", "saisonal-naiv (Vorjahreswoche)", "Regel", "cpu", ("forecast",), False

    def fit(self, task):
        return self

    def predict(self, task):
        Y, o = forecast_matrix(task, task.extra["origin_test"])
        H = task.extra["horizon"]
        return seasonal_naive(Y, o, H), Y[:, o:o + H], naive_scale(Y, o)


FC_FEATS = ["h", "last1", "mean4", "mean8", "mean13", "mean26", "std13", "zero13", "seas", "seas3", "level_ly",
            "woy", "series"]
FC_LABELS = {"h": "Horizont (Wochen)", "last1": "Vorwoche", "mean4": "Ø 4 Wochen", "mean8": "Ø 8 Wochen",
             "mean13": "Ø 13 Wochen", "mean26": "Ø 26 Wochen", "std13": "Streuung 13 Wochen",
             "zero13": "Nullwochen 13 Wochen", "seas": "Vorjahreswoche", "seas3": "Vorjahr ±1 Woche",
             "level_ly": "Niveau Vorjahr", "woy": "Kalenderwoche", "series": "Artikel"}


def _fc_rows(Y, o, H, weeks, with_target=True):
    rows = []
    for s in range(Y.shape[0]):
        y = Y[s]
        base = dict(last1=y[o - 1], mean4=y[o - 4:o].mean(), mean8=y[o - 8:o].mean(), mean13=y[o - 13:o].mean(),
                    mean26=y[o - 26:o].mean(), std13=y[o - 13:o].std(), zero13=(y[o - 13:o] == 0).mean(),
                    level_ly=y[max(0, o - 65):max(1, o - 52)].mean())
        for h in range(H):
            t = o + h
            r = dict(base, h=h + 1, seas=y[t - 52] if t >= 52 else np.nan,
                     seas3=y[max(0, t - 53):t - 50].mean() if t >= 53 else np.nan,
                     woy=weeks[t].isocalendar()[1] if t < len(weeks) else 0, series=s)
            if with_target:
                r["y"] = y[t]
            rows.append(r)
    return pd.DataFrame(rows)


class LGBMForecastEngine(Engine):
    id, name, klass, kind, tasks, proba = "lgbm", "LightGBM (global, Lag-Features)", "Baum", "cpu", ("forecast",), False
    license = "MIT"

    def fit(self, task):
        lgb = _lgb()
        Y = task.extra["Y"].to_numpy(dtype=float)
        weeks, H = task.extra["weeks"], task.extra["horizon"]
        origins = task.extra["origins_train"]
        frames = [_fc_rows(Y, weeks.get_loc(o), H, weeks) for o in origins]
        tr = pd.concat(frames[:-2], ignore_index=True)
        va = pd.concat(frames[-2:], ignore_index=True)
        params = dict(LGB_PARAMS, objective="tweedie", tweedie_variance_power=1.2, metric="l1")
        X = lambda d: d[FC_FEATS].astype({"series": "category"})  # noqa: E731
        self.model = lgb.train(params, lgb.Dataset(X(tr), tr["y"]), 1500, valid_sets=[lgb.Dataset(X(va), va["y"])],
                               callbacks=[lgb.early_stopping(60, verbose=False)])
        self._X = X
        imp = self.model.feature_importance("gain")
        self.importance = [{"feature": f, "label": FC_LABELS[f], "importance": round(float(v), 1)}
                           for f, v in sorted(zip(FC_FEATS, imp), key=lambda x: -x[1])]
        return self

    def predict(self, task):
        Y, o = forecast_matrix(task, task.extra["origin_test"])
        H, weeks = task.extra["horizon"], task.extra["weeks"]
        te = _fc_rows(Y, o, H, weeks, with_target=False)
        p = np.clip(self.model.predict(self._X(te), num_iteration=self.model.best_iteration), 0, None)
        return p.reshape(Y.shape[0], H), Y[:, o:o + H], naive_scale(Y, o)

    def explain(self, task):
        return {"global": self.importance}
