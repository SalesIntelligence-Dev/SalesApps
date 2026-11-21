"""
GPU-Engines: TabPFN v2 (Tabular Foundation Model), heterogener GraphSAGE
(Relational Deep Learning) und LightGCN (Empfehlung).

Lokal mit CUDA (z. B. .venv auf der RTX 3060) laufen sie im Prozess; im
Render-Webprozess delegieren sie an den Graph-ML-Worker auf RunPod
(`remote = True`). torch wird erst beim Rechnen importiert.
"""
from __future__ import annotations

import importlib.util
import time

import numpy as np
import pandas as pd
from scipy import sparse

from .. import backtest as bt
from .base import Engine, Unavailable, on_render
from .runtime import local_gpu, runpod_ready, torch_device

TABPFN_MAX_TRAIN = 10_000


class _GPUEngine(Engine):
    kind = "gpu"
    needs = ("torch",)

    @property
    def remote(self) -> bool:
        return not local_gpu()[0] and runpod_ready()

    def available(self):
        if local_gpu()[0]:
            missing = [p for p in self.needs if importlib.util.find_spec(p) is None]
            if missing:
                return False, f"Paket fehlt in der GPU-Umgebung: {', '.join(missing)}"
            return True, ""
        if runpod_ready():
            return True, ""
        if on_render():
            return False, "GPU-Worker nicht konfiguriert (RUNPOD_GRAPHML_ENDPOINT_ID) – Ergebnis aus der Vorberechnung"
        return False, "keine CUDA-GPU in dieser Umgebung und kein RunPod-Worker – mit .venv/bin/python vorberechnen"

    def _device(self):
        dev = torch_device()
        self.device = f"Lokale GPU · {local_gpu()[1]}" if dev.type == "cuda" else "CPU (torch)"
        return dev


# ── TabPFN v2 ─────────────────────────────────────────────────────────────────

class TabPFNEngine(_GPUEngine):
    id, name, klass, tasks = "tabpfn", "TabPFN v2", "TFM", ("binary",)
    needs = ("torch", "tabpfn")
    license = "Prior Labs License – vor kommerzieller Nutzung prüfen"

    def fit(self, task):
        dev = self._device()
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise Unavailable(f"tabpfn nicht installiert: {exc}")
        cols = task.features
        tr = task.train
        rng = np.random.default_rng(bt.SEED)
        if len(tr) > TABPFN_MAX_TRAIN:
            # stratifizierte Stichprobe, damit die Positivrate erhalten bleibt
            pos, neg = np.flatnonzero(tr[task.label].to_numpy() == 1), np.flatnonzero(tr[task.label].to_numpy() == 0)
            n_pos = int(round(TABPFN_MAX_TRAIN * len(pos) / len(tr)))
            idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                                  rng.choice(neg, TABPFN_MAX_TRAIN - n_pos, replace=False)])
            tr = tr.iloc[np.sort(idx)]
            self.notes.append(f"Kontext: stratifizierte Stichprobe {TABPFN_MAX_TRAIN:,} von {len(task.train):,} "
                              "Trainingszeilen".replace(",", "."))
        X = tr[cols].astype("float32").fillna(-1).to_numpy()
        self.clf = _make_tabpfn(TabPFNClassifier, dev)
        self.clf.fit(X, tr[task.label].to_numpy())
        self.cols = cols
        return self

    def predict(self, task):
        X = task.test[self.cols].astype("float32").fillna(-1).to_numpy()
        out = [self.clf.predict_proba(X[i:i + 4000])[:, 1] for i in range(0, len(X), 4000)]
        return np.concatenate(out)


def _make_tabpfn(cls, dev):
    """TabPFN v2 bevorzugen (frei herunterladbare Gewichte); neuere Paketversionen kennen mehrere Modelle."""
    errors = []
    try:
        from tabpfn.constants import ModelVersion
        return cls.create_default_for_version(ModelVersion.V2, device=str(dev))
    except Exception as exc:          # ältere/andere Versionen
        errors.append(str(exc))
    try:
        return cls(device=str(dev), n_estimators=4, random_state=bt.SEED)
    except Exception as exc:
        errors.append(str(exc))
    raise Unavailable("TabPFN-Modell nicht ladbar: " + " | ".join(e[:120] for e in errors))


# ── Heterogener GraphSAGE (RDL) ───────────────────────────────────────────────

GNN_CUST_FEATS = ["recency", "freq_30", "freq_90", "freq_365", "monetary_90", "monetary_365", "avg_basket", "tenure",
                  "n_products", "storno_cnt", "storno_share", "trend_orders", "trend_revenue", "overdue"]


def _snapshot_graph(b, T, task, audit) -> dict:
    """Kunde–Artikel-Graph zum Stichtag T (nur Käufe < T, 365 Tage). Zeitlich korrekt je Snapshot."""
    feats = bt.customer_features(b, T, audit, f"gnn:{task.uc}")
    feats = feats[feats["recency"] <= bt.HISTORY]
    ln = b.lines[(b.lines["time"] < T) & (b.lines["time"] >= T - pd.Timedelta(days=bt.HISTORY))]
    audit.check(f"gnn:{task.uc}:kanten", T, ln["time"].max())
    cidx = pd.Index(feats.index)
    ln = ln[ln["cust"].isin(cidx)]
    pidx = pd.Index(np.sort(ln["prod"].unique()))
    e = ln[["cust", "prod"]].drop_duplicates()
    ec, ep = cidx.get_indexer(e["cust"]), pidx.get_indexer(e["prod"])
    Xc = np.log1p(np.clip(feats[GNN_CUST_FEATS].fillna(0).to_numpy(float), 0, None))
    pg = ln.groupby("prod")
    deg = pg["cust"].nunique().reindex(pidx).to_numpy(float)
    price = pg["price"].median().reindex(pidx).to_numpy(float)
    last = (T - pg["time"].max().reindex(pidx)).dt.days.to_numpy(float)
    grp = np.eye(int(b.prod_group.max()) + 1, dtype=np.float32)[b.prod_group[pidx.to_numpy()]]
    Xp = np.column_stack([np.log1p(deg), np.log1p(price), np.log1p(last), grp])
    country = feats["country"].clip(lower=0).to_numpy()
    return {"T": T, "cust": cidx.to_numpy(), "prod": pidx.to_numpy(), "Xc": Xc.astype(np.float32), "Xp": Xp.astype(np.float32),
            "country": country.astype(np.int64), "ec": ec.astype(np.int64), "ep": ep.astype(np.int64)}


PAIR_FEATS = ["g_affinity_max", "g_affinity_mean", "group_pop", "group_growth", "n_groups"]


def _labels_for(task, g):
    """Zeilen des Stichtags auf Graph-Knoten abbilden. Cross-Selling: Paare Kunde × Gruppe (+ Paar-Merkmale)."""
    rows = task.train if g["T"] < bt.T_TEST else task.test
    rows = rows[rows["cutoff"] == g["T"]]
    pos = pd.Index(g["cust"]).get_indexer(rows["cust"])
    keep = pos >= 0
    pair = None
    if task.uc == "cross_selling":
        r = rows[keep]
        pair = {"group": r["group"].to_numpy().astype(np.int64),
                "x": np.log1p(np.clip(r[PAIR_FEATS].fillna(0).to_numpy(float), 0, None)).astype(np.float32),
                "key": r[task.id_col].to_numpy()}
    return pos[keep], rows[task.label].to_numpy()[keep], pair


class GNNEngine(_GPUEngine):
    id, name, klass, tasks = "gnn", "Hetero-GraphSAGE (RDL)", "Relational", ("binary",)
    use_cases = ("churn", "lead_scoring", "cross_selling", "account_scoring", "territory_coverage")
    needs = ("torch", "torch_geometric")
    license = "MIT (PyTorch Geometric)"

    def fit(self, task):
        import torch
        from torch import nn
        from torch_geometric.nn import HeteroConv, SAGEConv
        dev = self._device()
        torch.manual_seed(bt.SEED)
        b = bt.base_of(task)
        audit = bt.Audit()
        snaps = task.snapshots[-9:]                      # letzte 9 Trainings-Stichtage + Test
        graphs = [_snapshot_graph(b, T, task, audit) for T in snaps + [bt.T_TEST]]
        allc = np.concatenate([g["Xc"] for g in graphs[:-1]])
        self.mu, self.sd = allc.mean(0), allc.std(0) + 1e-6
        n_ctry = int(max(g["country"].max() for g in graphs)) + 1

        n_grp = int(b.prod_group.max()) + 1

        def to_t(g):
            ei = torch.tensor(np.vstack([g["ec"], g["ep"]]), device=dev)
            idx, y, pair = _labels_for(task, g)
            d = {"xc": torch.tensor((g["Xc"] - self.mu) / self.sd, device=dev),
                 "xp": torch.tensor(g["Xp"], device=dev), "ctry": torch.tensor(g["country"], device=dev),
                 "ei": ei, "eir": ei.flip(0), "idx": torch.tensor(idx, device=dev),
                 "y": torch.tensor(y, dtype=torch.float32, device=dev), "cust": g["cust"][idx]}
            if pair is not None:
                d.update(grp=torch.tensor(pair["group"], device=dev), px=torch.tensor(pair["x"], device=dev),
                         key=pair["key"])
            return d

        self.tg = [to_t(g) for g in graphs]
        self.pairs = task.uc == "cross_selling"
        n_px = len(PAIR_FEATS)
        H = 64

        class Net(nn.Module):
            def __init__(s, dc, dp):
                super().__init__()
                s.lc, s.lp, s.emb = nn.Linear(dc, H), nn.Linear(dp, H), nn.Embedding(n_ctry, 8)
                s.lc2 = nn.Linear(H + 8, H)
                s.c1 = HeteroConv({("c", "b", "p"): SAGEConv((H, H), H), ("p", "r", "c"): SAGEConv((H, H), H)}, aggr="mean")
                s.c2 = HeteroConv({("c", "b", "p"): SAGEConv((H, H), H), ("p", "r", "c"): SAGEConv((H, H), H)}, aggr="mean")
                s.pairs = pairs
                if pairs:                                  # Kunde × Gruppe: Kunden-Einbettung + Gruppe + Paar-Merkmale
                    s.gemb = nn.Embedding(n_grp, 16)
                    s.head = nn.Sequential(nn.Linear(2 * H + 16 + n_px, H), nn.ReLU(), nn.Dropout(0.2), nn.Linear(H, 1))
                else:
                    s.head = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Dropout(0.2), nn.Linear(H, 1))

            def forward(s, d):
                hc = torch.relu(s.lc2(torch.cat([torch.relu(s.lc(d["xc"])), s.emb(d["ctry"])], 1)))
                x = {"c": hc, "p": torch.relu(s.lp(d["xp"]))}
                ed = {("c", "b", "p"): d["ei"], ("p", "r", "c"): d["eir"]}
                x1 = {k: torch.relu(v) for k, v in s.c1(x, ed).items()}
                x2 = s.c2(x1, ed)
                z = torch.cat([hc, x2["c"]], 1)
                if s.pairs:                                # Ausgabe je Paar (in Zeilenreihenfolge)
                    zi = z[d["idx"]]
                    return s.head(torch.cat([zi, s.gemb(d["grp"]), d["px"]], 1)).squeeze(-1)
                return s.head(z).squeeze(-1)

        pairs = self.pairs
        self.net = Net(self.tg[0]["xc"].shape[1], self.tg[0]["xp"].shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        train, val = self.tg[:-2], self.tg[-2]
        best, best_state, patience = 1e9, None, 0
        for epoch in range(120):
            self.net.train()
            for d in train:
                if len(d["idx"]) == 0:
                    continue
                opt.zero_grad()
                loss = lossf(self._out(d), d["y"])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(lossf(self._out(val), val["y"]))
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                patience += 1
                if patience >= 15:
                    break
        self.net.load_state_dict(best_state)
        self.epochs = epoch + 1
        self.notes.append(f"{len(train)} Trainings-Graphen (je Stichtag nur Kanten davor), {self.epochs} Epochen, "
                          "Full-Batch statt Neighbor-Sampling (Graph klein genug)")
        return self

    def _out(self, d):
        o = self.net(d)
        return o if self.pairs else o[d["idx"]]

    def predict(self, task):
        import torch
        d = self.tg[-1]
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self._out(d)).cpu().numpy()
        if self.pairs:
            s = pd.Series(p, index=d["key"])
            return s.reindex(task.test[task.id_col].to_numpy()).fillna(float(np.median(p))).to_numpy()
        s = pd.Series(p, index=d["cust"])
        return s.reindex(task.test["cust"].to_numpy()).fillna(float(np.median(p))).to_numpy()


# ── GNN-Ranker (Empfehlung über den Kunde–Artikel-Graphen) ────────────────────

class GNNRankEngine(_GPUEngine):
    """Hetero-GraphSAGE lernt Kunden- und Artikel-Einbettungen aus dem Kaufgraphen (2 Sprünge: Kunde → Artikel →
    andere Kunden → Artikel). Bewertet werden dieselben Kandidaten wie beim LightGBM-Ranker – fairer Vergleich."""
    id, name, klass, tasks, proba = "gnn_rank", "GNN-Ranker (Hetero-GraphSAGE)", "Relational", ("ranking",), False
    needs = ("torch", "torch_geometric")
    license = "MIT (PyTorch Geometric)"

    def _prep(self, task, T, with_label):
        from .cpu import RANK_FEATS, _rank_pairs
        b = bt.base_of(task)
        snap = bt.reco_snapshot(b, T)
        g = _snapshot_graph(b, T, task, self.audit)
        pr = _rank_pairs(snap, with_label, b)
        ci = pd.Index(g["cust"]).get_indexer(snap["users"][pr["u"].to_numpy()])
        pi = pd.Index(g["prod"]).get_indexer(pr["p"].to_numpy())
        pi = np.where(pi < 0, len(g["prod"]), pi)                 # unbekannter Artikel → eigener Platzhalter
        X = np.log1p(np.clip(pr[RANK_FEATS].to_numpy(float), 0, None)).astype(np.float32)
        return {"g": g, "snap": snap, "pairs": pr, "ci": ci, "pi": pi, "X": X,
                "y": pr["y"].to_numpy(np.float32) if with_label else None}

    def fit(self, task):
        import torch
        from torch import nn
        from torch_geometric.nn import HeteroConv, SAGEConv
        dev = self._device()
        torch.manual_seed(bt.SEED)
        self.audit = bt.Audit()
        t0 = time.time()
        tr, va = (self._prep(task, T, True) for T in task.snapshots[:2])
        self.te = self._prep(task, bt.T_TEST, False)
        mu = np.concatenate([tr["g"]["Xc"], va["g"]["Xc"]]).mean(0)
        sd = np.concatenate([tr["g"]["Xc"], va["g"]["Xc"]]).std(0) + 1e-6
        xm, xs = tr["X"].mean(0), tr["X"].std(0) + 1e-6
        n_ctry = int(max(d["g"]["country"].max() for d in (tr, va, self.te))) + 1

        def to_t(d):
            g = d["g"]
            ei = torch.tensor(np.vstack([g["ec"], g["ep"]]), device=dev)
            keep = d["ci"] >= 0
            out = {"xc": torch.tensor((g["Xc"] - mu) / sd, device=dev), "xp": torch.tensor(g["Xp"], device=dev),
                   "ctry": torch.tensor(g["country"], device=dev), "ei": ei, "eir": ei.flip(0),
                   "ci": torch.tensor(np.where(keep, d["ci"], 0), device=dev), "pi": torch.tensor(d["pi"], device=dev),
                   "x": torch.tensor((d["X"] - xm) / xs, device=dev), "keep": torch.tensor(keep, device=dev)}
            if d["y"] is not None:
                out["y"] = torch.tensor(d["y"], device=dev)
            return out

        H = 64

        class Net(nn.Module):
            def __init__(s, dc, dp, dx):
                super().__init__()
                s.lc, s.lp, s.emb = nn.Linear(dc, H), nn.Linear(dp, H), nn.Embedding(n_ctry, 8)
                s.lc2 = nn.Linear(H + 8, H)
                conv = lambda: HeteroConv({("c", "b", "p"): SAGEConv((H, H), H),  # noqa: E731
                                           ("p", "r", "c"): SAGEConv((H, H), H)}, aggr="mean")
                s.c1, s.c2 = conv(), conv()
                s.unk = nn.Parameter(torch.zeros(1, H))
                s.head = nn.Sequential(nn.Linear(3 * H + dx, H), nn.ReLU(), nn.Dropout(0.1), nn.Linear(H, 1))

            def forward(s, d):
                hc = torch.relu(s.lc2(torch.cat([torch.relu(s.lc(d["xc"])), s.emb(d["ctry"])], 1)))
                x = {"c": hc, "p": torch.relu(s.lp(d["xp"]))}
                ed = {("c", "b", "p"): d["ei"], ("p", "r", "c"): d["eir"]}
                x1 = {k: torch.relu(v) for k, v in s.c1(x, ed).items()}
                x2 = s.c2(x1, ed)
                zp = torch.cat([x2["p"], s.unk], 0)
                zc, zpp = x2["c"][d["ci"]], zp[d["pi"]]
                return s.head(torch.cat([zc, zpp, zc * zpp, d["x"]], 1)).squeeze(-1)

        self.T = [to_t(d) for d in (tr, va, self.te)]
        self.net = Net(tr["g"]["Xc"].shape[1], tr["g"]["Xp"].shape[1], tr["X"].shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=1e-4)
        pos_w = torch.tensor(float((1 - tr["y"].mean()) / max(tr["y"].mean(), 1e-6)) ** 0.5, device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, best_state, patience = 1e9, None, 0
        dtr, dva = self.T[0], self.T[1]
        for epoch in range(80):
            self.net.train()
            opt.zero_grad()
            o = self.net(dtr)
            loss = lossf(o[dtr["keep"]], dtr["y"][dtr["keep"]])
            loss.backward()
            opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(lossf(self.net(dva)[dva["keep"]], dva["y"][dva["keep"]]))
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                patience += 1
                if patience >= 10:
                    break
        self.net.load_state_dict(best_state)
        self.notes.append(f"Kaufgraph je Stichtag (nur Kanten davor), {len(tr['pairs']):,} Trainings-Kandidaten, "
                          f"{epoch + 1} Epochen in {time.time() - t0:.0f} s".replace(",", "."))
        return self

    def predict(self, task):
        import torch
        d, te = self.T[2], self.te
        self.net.eval()
        with torch.no_grad():
            sc = self.net(d).cpu().numpy()
        sc[te["ci"] < 0] = -1e9                                   # Kunde ohne Graph-Knoten
        pr = te["pairs"].assign(s=sc)
        top = pr.sort_values(["u", "s"], ascending=[True, False]).groupby("u").head(12)
        recs = top.groupby("u")["p"].apply(list).reindex(range(len(te["snap"]["users"]))).tolist()
        return [r if isinstance(r, list) else [] for r in recs], te["snap"]["truth"]


# ── LightGCN (Empfehlung) ─────────────────────────────────────────────────────

class LightGCNEngine(_GPUEngine):
    id, name, klass, tasks, proba = "lightgcn", "LightGCN", "Relational", ("ranking",), False
    license = "Methode frei (He et al. 2020), eigene Implementierung"

    def fit(self, task):
        import torch
        dev = self._device()
        torch.manual_seed(bt.SEED)
        b = bt.base_of(task)
        T = bt.T_TEST
        hist = b.lines[(b.lines["time"] < T) & (b.lines["time"] >= T - pd.Timedelta(days=bt.HISTORY))]
        task.audit.check("lightgcn:kanten", T, hist["time"].max())
        pairs = hist[["cust", "prod"]].drop_duplicates()
        uidx = pd.Index(np.sort(pairs["cust"].unique()))
        u, i = uidx.get_indexer(pairs["cust"]), pairs["prod"].to_numpy()
        nu, ni = len(uidx), b.n_prod
        A = sparse.coo_matrix((np.ones(len(u)), (u, nu + i)), shape=(nu + ni, nu + ni))
        A = (A + A.T).tocsr()
        d = np.asarray(A.sum(1)).ravel()
        dinv = 1 / np.sqrt(np.maximum(d, 1))
        An = sparse.diags(dinv) @ A @ sparse.diags(dinv)
        An = An.tocoo()
        G = torch.sparse_coo_tensor(np.vstack([An.row, An.col]), An.data.astype(np.float32), An.shape, device=dev)
        E = torch.nn.Embedding(nu + ni, 64).to(dev)
        torch.nn.init.normal_(E.weight, std=0.1)
        opt = torch.optim.Adam(E.parameters(), lr=5e-3)
        pu, pi = torch.tensor(u, device=dev), torch.tensor(i + nu, device=dev)
        rng = torch.Generator(device=dev).manual_seed(bt.SEED)

        def propagate():
            x, out = E.weight, E.weight
            for _ in range(3):
                x = torch.sparse.mm(G, x)
                out = out + x
            return out / 4

        t0 = time.time()
        for epoch in range(150):
            perm = torch.randperm(len(pu), device=dev, generator=rng)
            for s in range(0, len(pu), 16384):
                bidx = perm[s:s + 16384]
                X = propagate()
                neg = torch.randint(nu, nu + ni, (len(bidx),), device=dev, generator=rng)
                xu, xp, xn = X[pu[bidx]], X[pi[bidx]], X[neg]
                loss = -torch.nn.functional.logsigmoid((xu * xp).sum(1) - (xu * xn).sum(1)).mean()
                loss = loss + 1e-4 * (E.weight[pu[bidx]].pow(2).sum() + E.weight[pi[bidx]].pow(2).sum()) / len(bidx)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            self.X = propagate()
        self.uidx, self.nu = uidx, nu
        self.notes.append(f"{nu:,} Kunden × {ni:,} Artikel, {len(pu):,} Kanten, 150 Epochen BPR in {time.time() - t0:.0f} s"
                          .replace(",", "."))
        return self

    def predict(self, task):
        import torch
        snap = bt.reco_snapshot(bt.base_of(task), bt.T_TEST)
        rows = self.uidx.get_indexer(snap["users"])
        with torch.no_grad():
            U = self.X[torch.tensor(rows, device=self.X.device)]
            S = U @ self.X[self.nu:].T
            top = torch.topk(S, 12, dim=1).indices.cpu().numpy()
        return [list(r) for r in top], snap["truth"]
