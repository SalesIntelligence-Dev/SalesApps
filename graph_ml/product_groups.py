"""
Vorläufige Artikelgruppen per Schlüsselwort-Regeln (für den Rückblick)
=====================================================================
Online Retail II hat keine Artikelgruppen – nur Freitext-Beschreibungen.
Bis Fix-Playbook PB-05 (Freitext → Kategorien, Prompt 4) die Gruppen sauber
ableitet, ordnet diese Regelliste jede Beschreibung der ersten passenden
Gruppe zu. Klar als „vorläufig“ gekennzeichnet.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

GROUP_RULES: list[tuple[str, list[str]]] = [
    ("Weihnachten", ["CHRISTMAS", "XMAS", "ADVENT", "SANTA", "REINDEER", "SNOWFLAKE", "NATIVITY", "BAUBLE"]),
    ("Kerzen & Licht", ["CANDLE", "T-LIGHT", "TEA LIGHT", "TLIGHT", "LANTERN", "LIGHT", "LAMP"]),
    ("Taschen", ["BAG", "TOTE", "SHOPPER"]),
    ("Tassen & Geschirr", ["MUG", "CUP", "PLATE", "BOWL", "SAUCER", "TEAPOT", "JUG", "TUMBLER", "GLASS"]),
    ("Küche & Backen", ["CAKE", "BAKING", "CUTLERY", "SPOON", "APRON", "OVEN", "TEA TOWEL", "JAM", "LUNCH BOX",
                        "TRAY", "EGG", "COOK", "KITCHEN", "BREAD"]),
    ("Dosen & Aufbewahrung", ["TIN", "BOX", "JAR", "CONTAINER", "STORAGE", "BASKET", "HAMPER", "DRAWER"]),
    ("Papier & Karten", ["CARD", "PAPER", "NAPKIN", "WRAP", "ENVELOPE", "NOTEBOOK", "STICKER", "CALENDAR", "TAPE",
                         "GIFT TAG", "JOURNAL"]),
    ("Schreibwaren", ["PEN", "PENCIL", "ERASER", "RULER", "CHALK", "CRAYON", "SHARPENER"]),
    ("Bilderrahmen & Schilder", ["FRAME", "SIGN", "PLAQUE", "CLOCK", "MIRROR", "DOORMAT"]),
    ("Textilien", ["CUSHION", "BLANKET", "TOWEL", "HOT WATER BOTTLE", "SLIPPER", "GLOVE", "SCARF", "UMBRELLA"]),
    ("Schmuck & Accessoires", ["NECKLACE", "BRACELET", "EARRING", "RING", "JEWEL", "HAIR", "CLIP", "PURSE",
                               "WALLET", "KEYRING", "BADGE"]),
    ("Spielzeug & Kinder", ["TOY", "DOLL", "GAME", "PUZZLE", "SPACEBOY", "CHILDREN", "CHILDS", "BABY",
                            "SOLDIER", "TEDDY", "PLAYHOUSE"]),
    ("Garten", ["GARDEN", "PLANT", "FLOWER", "WATERING", "BIRD HOUSE", "BIRDHOUSE", "SEED"]),
    ("Party", ["PARTY", "BALLOON", "BIRTHDAY", "PICNIC", "BUNTING"]),
    ("Deko & Hänger", ["HANGING", "DECORATION", "ORNAMENT", "GARLAND", "WREATH", "HEART", "STAR", "BIRD",
                       "BUTTERFLY", "VINTAGE"]),
]
OTHER = "Sonstiges"
NON_ARTICLE = "Gebühren & Porto"
GROUPS = [g for g, _ in GROUP_RULES] + [OTHER, NON_ARTICLE]


def _compile() -> list[tuple[str, re.Pattern]]:
    return [(g, re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")S?\b")) for g, kws in GROUP_RULES]


_RULES = _compile()


def group_of(description: str | None, stock_code: str | None = None) -> str:
    code = (stock_code or "").upper()
    if code and not re.search(r"\d", code) or code.startswith(("POST", "DOT", "BANK", "AMAZON", "ADJUST", "TEST")):
        return NON_ARTICLE
    d = (description or "").upper()
    for g, rx in _RULES:
        if rx.search(d):
            return g
    return OTHER


def product_group_codes(stock_codes: pd.Index, desc_by_code: pd.Series) -> np.ndarray:
    """Gruppen-Index (in GROUPS) je Artikelnummer."""
    idx = {g: i for i, g in enumerate(GROUPS)}
    return np.array([idx[group_of(desc_by_code.get(c), c)] for c in stock_codes], dtype=np.int16)
