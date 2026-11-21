"""
API-Engines: KumoRFM (relationales Foundation Model, In-Context) und Aurora
(Zeitreihen-Foundation-Model über den bestehenden RunPod-Endpoint).

Beide sind optional: ohne Schlüssel bzw. Endpoint werden sie ausgegraut,
nicht als Fehler gewertet.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd

from .. import backtest as bt
from .base import Engine, Unavailable


class KumoRFMEngine(Engine):
    id, name, klass, kind, tasks = "kumorfm", "KumoRFM", "Relational (Foundation Model)", "api", ("binary",)
    use_cases = ("churn",)
    license = "Kumo/Nvidia – Nutzungsbedingungen vor kommerzieller Demo klären"

    def available(self):
        if not os.environ.get("KUMO_API_KEY"):
            return False, "KUMO_API_KEY nicht gesetzt – optional, Nutzungsbedingungen vorab klären"
        if importlib.util.find_spec("kumoai") is None:
            return False, "kumoai-SDK nicht installiert"
        return True, ""

    def fit(self, task):
        import kumoai.experimental.rfm as rfm          # noqa: WPS433 – nur mit Key
        rfm.init(api_key=os.environ["KUMO_API_KEY"])
        b = bt.get_base()
        hist = b.inv[b.inv["time"] < bt.T_TEST]
        task.audit.check("kumorfm:belege", bt.T_TEST, hist["time"].max())
        customers = pd.DataFrame({"customer_id": np.sort(hist["cust"].unique())})
        invoices = hist.rename(columns={"cust": "customer_id", "inv": "invoice_id", "time": "invoice_date"})[
            ["invoice_id", "customer_id", "invoice_date", "value"]]
        graph = rfm.LocalGraph.from_data({"customers": customers, "invoices": invoices})
        self.model = rfm.KumoRFM(graph)
        self.device = "Kumo-API"
        return self

    def predict(self, task):
        ids = task.test["cust"].astype(int).tolist()
        scores = {}
        for i in range(0, len(ids), 500):            # Anfragen klein halten (Limit je Anfrage)
            chunk = ", ".join(str(x) for x in ids[i:i + 500])
            q = (f"PREDICT COUNT(invoices.*, 0, {bt.HORIZON}, days)=0 FOR EACH customers.customer_id IN ({chunk})")
            df = self.model.predict(q, anchor_time=bt.T_TEST)
            scores.update(dict(zip(df.iloc[:, 0].astype(int), df.iloc[:, -1].astype(float))))
        return np.array([scores.get(c, np.nan) for c in ids], dtype=float)


class AuroraEngine(Engine):
    id, name, klass, kind, tasks, proba = "aurora", "Aurora (Zeitreihen-FM)", "Zeitreihen-FM", "api", ("forecast",), False
    license = "bestehender Aurora-Endpoint (wie /forecasting)"
    LOOKBACK = 528

    def available(self):
        if not (os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_ENDPOINT_ID")):
            return False, "Aurora-Endpoint (RUNPOD_ENDPOINT_ID) nicht gesetzt – läuft auf Render wie /forecasting"
        return True, ""

    def fit(self, task):
        from aurora_inference import AuroraConfig, AuroraInference   # unverändert wiederverwendet
        self.client = AuroraInference(AuroraConfig.from_env())
        if not self.client.ready:
            raise Unavailable("Aurora-Endpoint nicht bereit")
        self.device = "RunPod · Aurora-Endpoint"
        return self

    def predict(self, task):
        """Tagesreihen (≥ 528 Tage Historie) zero-shot prognostizieren und auf 13 Wochen summieren."""
        b = bt.get_base()
        origin = task.extra["origin_test"]
        H = task.extra["horizon"]
        days = pd.date_range(origin - pd.Timedelta(days=self.LOOKBACK), periods=self.LOOKBACK, freq="D")
        L = b.lines[(b.lines["time"] < origin) & b.lines["prod"].isin(task.extra["prods"])]
        task.audit.check("aurora:historie", origin, L["time"].max())
        daily = (L.assign(d=L["time"].dt.normalize()).groupby(["prod", "d"])["qty"].sum()
                 .unstack(fill_value=0).reindex(index=task.extra["prods"], columns=days, fill_value=0))
        Y, o = task.extra["Y"].to_numpy(dtype=float), task.extra["weeks"].get_loc(origin)
        out = np.zeros((len(task.extra["prods"]), H))
        for s, prod in enumerate(task.extra["prods"]):
            frame = pd.DataFrame({"t": days, "qty": daily.loc[prod].to_numpy(dtype=float)})
            inp = self.client.prepare(frame, target_col="qty", time_col="t", freq="D")
            fc = self.client.forecast(inp, horizon=H * 7, num_samples=20)
            out[s] = np.clip(np.asarray(fc.mean), 0, None).reshape(H, 7).sum(axis=1)
        from .cpu import naive_scale
        return out, Y[:, o:o + H], naive_scale(Y, o)
