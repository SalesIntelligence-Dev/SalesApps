"""
Semantische Feldzuordnung
=========================
Ordnet die Felder der Use-Case-Bibliothek (Kundennummer, Ereigniszeit, …)
den Spalten eines Schemas zu – über Synonyme verbreiteter ERP-/CRM-Namen,
Spaltentyp und Tabellenkontext. Arbeitet nur auf der Schema-Sicht
(Tabellen, Spalten, Rollen, Beziehungen) und liest nie Datenzeilen.
"""
from __future__ import annotations

import re
from collections import deque

from .graph_mapping import _norm
from .usecases import FIELDS

# Reihenfolge zählt: spätere Felder bevorzugen Tabellen früherer Bindungen
FIELD_ORDER = ["order_key", "event_time", "customer_key", "product_key", "quantity", "price", "list_price", "discount", "cost",
               "product_group", "product_text", "country", "account_group", "sales_rep", "doc_type", "return_link", "campaign", "response",
               "serial_number", "batch", "warranty_claim", "defect_code", "supplier", "bom", "usage_data",
               "service_event", "contract", "config",
               "rejected_qty", "technician", "failure_event"]

PREFER_WITH = {"quantity": "product_key", "price": "product_key", "event_time": "order_key",
               "doc_type": "order_key", "country": "customer_key", "product_text": "product_key_master",
               "product_group": "product_key_master"}

NUMERIC = {"int", "float", "integer", "number"}
PERSONAL_RE = re.compile(r"(firstname|lastname|vorname|nachname|email|e_mail|phone|telefon|mobile|street|strasse|"
                         r"address|adresse|birth|geburt|iban|bic|creditcard|cardnumber|kreditkarte|nationalid|"
                         r"loginid|passwor|salary|gehalt|gender|geschlecht|maritalstatus)", re.I)
SENSITIVE_RE = re.compile(r"(creditcard|cardnumber|kreditkarte|iban|nationalid|passwor|salary|gehalt)", re.I)


def schema_view(mapping: dict) -> dict:
    """Mapping → reine Struktur: Tabellen, Spalten, Typen, Rollen, Beziehungen. Keine Zahlen, keine Werte."""
    tables = []
    for t in mapping["tables"]:
        tables.append({
            "name": t["name"], "label": t.get("label", t["name"]), "schema": t.get("schema"),
            "primary_key": t.get("primary_key"), "time_columns": list(t.get("time_columns", [])),
            "only_modified_date": bool(t.get("only_modified_date")),
            "columns": [{"name": c["name"], "kind": c.get("kind"), "roles": list(c.get("roles", [])),
                         "sql_type": c.get("sql_type")} for c in t["columns"]],
        })
    relations = [{"from_table": r["from_table"], "from_column": r["from_column"], "to_table": r["to_table"],
                  "to_column": r["to_column"], "kind": r["kind"], "label": r.get("label")}
                 for r in mapping["relations"]]
    return {"dataset_id": mapping.get("dataset_id"), "mode": mapping.get("mode"), "tables": tables,
            "relations": relations, "derived_columns": list(mapping.get("derived_columns", []))}


def _is_event_table(t: dict) -> bool:
    return any(not re.search(r"modif|geaendert|changed|updated", c, re.I) for c in t.get("time_columns", []))


def _kind_ok(spec: dict, col: dict) -> bool:
    k = spec.get("kinds", "any")
    if k == "datetime":
        return col.get("kind") == "datetime" or "Zeit" in col.get("roles", [])
    if k == "numeric":
        return col.get("kind") in NUMERIC and "PK" not in col.get("roles", [])
    return col.get("kind") not in ("datetime",)


def _col_score(spec: dict, cname: str) -> float:
    n = _norm(cname)
    if any(x in n for x in spec.get("exclude", [])):
        return 0.0
    syn = spec.get("syn", [])
    if n in syn:
        return 1.0 - 0.005 * syn.index(n)      # frühere Synonyme sind spezifischer
    if any(len(x) >= 5 and x in n for x in spec.get("contains", [])):
        return 0.7
    return 0.0


def bind_fields(view: dict) -> dict:
    tables = view["tables"]
    out: dict[str, dict | None] = {}
    master_of_product = None
    for f in FIELD_ORDER:
        spec = FIELDS[f]
        best = None
        for t in tables:
            tn = _norm(t["name"].split(".")[-1])
            hint = any(h in tn for h in spec.get("tables", []))
            if spec.get("table_required") and not hint and t["name"] != master_of_product:
                continue
            colnames = [_norm(c["name"]) for c in t["columns"]]
            for c in t["columns"]:
                s = _col_score(spec, c["name"])
                if s == 0 or not _kind_ok(spec, c):
                    continue
                bonus = 0.05 if hint else 0.0
                if f in ("customer_key", "order_key") and _is_event_table(t):
                    bonus += 0.1
                if f == "product_key":
                    if any(x in FIELDS["order_key"]["syn"] for x in colnames):
                        bonus += 0.15
                    if any(x in FIELDS["quantity"]["syn"] for x in colnames):
                        bonus += 0.1
                pref = PREFER_WITH.get(f)
                if pref == "product_key_master":
                    if t["name"] == master_of_product:
                        bonus += 0.2
                elif pref and out.get(pref) and out[pref]["table"] == t["name"]:
                    bonus += 0.2
                score = s + bonus
                if best is None or score > best["score"]:
                    best = {"table": t["name"], "column": c["name"], "score": round(score, 2), "via": "column"}
            if spec.get("table_level") and any(x in tn for x in spec["table_level"]):
                score = 0.65 + (0.05 if hint else 0)
                if best is None or score > best["score"]:
                    best = {"table": t["name"], "column": t.get("primary_key") or (t["columns"][0]["name"]
                                                                                   if t["columns"] else None),
                            "score": round(score, 2), "via": "table"}
        if best:
            best["label"] = spec["label"]
        out[f] = best
        if f == "product_key" and best:
            master_of_product = _master_table(view, best)
    return out


def _master_table(view: dict, binding: dict) -> str | None:
    """Tabelle, deren Primärschlüssel die Artikelnummer ist (Artikelstamm)."""
    for r in view["relations"]:
        if r["from_table"] == binding["table"] and r["from_column"] == binding["column"] and r["kind"] != "missing":
            return r["to_table"]
    for t in view["tables"]:
        if t.get("primary_key") and _norm(t["primary_key"]) == _norm(binding["column"]):
            return t["name"]
    return binding["table"]


def connected(view: dict, a: str, b: str, max_hops: int = 3) -> list[str] | None:
    """Kürzester Pfad zwischen zwei Tabellen über Schlüsselbeziehungen (ungerichtet)."""
    if a == b:
        return [a]
    adj: dict[str, set] = {}
    for r in view["relations"]:
        if r["kind"] == "missing":
            continue
        adj.setdefault(r["from_table"], set()).add(r["to_table"])
        adj.setdefault(r["to_table"], set()).add(r["from_table"])
    queue, seen = deque([[a]]), {a}
    while queue:
        path = queue.popleft()
        if len(path) > max_hops + 1:
            continue
        for nxt in adj.get(path[-1], ()):
            if nxt == b:
                return path + [b]
            if nxt not in seen:
                seen.add(nxt)
                queue.append(path + [nxt])
    return None


def personal_columns(view: dict) -> tuple[list[str], list[str]]:
    personal, sensitive = [], []
    for t in view["tables"]:
        for c in t["columns"]:
            name = f"{t['name']}.{c['name']}"
            if SENSITIVE_RE.search(c["name"]):
                sensitive.append(name)
            elif PERSONAL_RE.search(c["name"]):
                personal.append(name)
    return personal, sensitive
