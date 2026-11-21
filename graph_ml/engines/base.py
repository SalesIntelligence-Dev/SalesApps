"""
Einheitliches Engine-Interface: fit · predict · explain · meta
"""
from __future__ import annotations

import os
import time


class Unavailable(Exception):
    """Engine kann in dieser Umgebung nicht laufen (fehlendes Paket, kein Key, keine GPU)."""


class Engine:
    id: str = ""
    name: str = ""
    klass: str = ""            # Regel · Baum · Graph-ML light · TFM · Relational · Zeitreihen-FM
    kind: str = "cpu"          # cpu · gpu · api
    tasks: tuple = ()          # unterstützte Aufgabentypen: binary · ranking · forecast
    use_cases: tuple | None = None
    proba: bool = True
    license: str = ""

    def __init__(self):
        self.device = "CPU"
        self.fit_seconds = 0.0
        self.notes: list[str] = []

    # Verfügbarkeit -----------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        return True, ""

    def supports(self, task) -> bool:
        return task.kind in self.tasks and (self.use_cases is None or task.uc in self.use_cases)

    # Lebenszyklus --------------------------------------------------------------
    def fit(self, task) -> "Engine":
        raise NotImplementedError

    def predict(self, task):
        raise NotImplementedError

    def explain(self, task) -> dict | None:
        return None

    def meta(self) -> dict:
        return {"id": self.id, "name": self.name, "klass": self.klass, "kind": self.kind,
                "device": self.device, "license": self.license, "notes": self.notes}

    def timed_fit(self, task) -> "Engine":
        t0 = time.time()
        self.fit(task)
        self.fit_seconds = time.time() - t0
        return self


def on_render() -> bool:
    return bool(os.environ.get("RENDER"))
