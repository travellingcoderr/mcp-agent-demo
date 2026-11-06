"""Thin, read-only client for the public openFDA drug APIs.

Every function returns {"results": [...], "sources": [{"id", "url"}...], "total": int}.
The agent may only cite ids that appear in "sources"; that is how answers are grounded.
Set OPENFDA_API_KEY for a higher rate limit (optional).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from urllib.parse import urlencode

import httpx

BASE = "https://api.fda.gov/drug"
SAFE_TERM = re.compile(r"^[A-Za-z0-9 .,'&()/+-]{2,80}$")


class InvalidInput(ValueError):
    pass


def _clean(term: str) -> str:
    term = (term or "").strip()
    if not SAFE_TERM.match(term):
        raise InvalidInput(f"unsupported characters or length in search term: {term!r}")
    return term.replace('"', "")


def _url(endpoint: str, params: dict) -> str:
    return f"{BASE}/{endpoint}.json?{urlencode(params)}"


@lru_cache(maxsize=512)
def _get(endpoint: str, frozen: tuple) -> dict:
    params = dict(frozen)
    if os.getenv("OPENFDA_API_KEY"):
        params["api_key"] = os.environ["OPENFDA_API_KEY"]
    r = httpx.get(f"{BASE}/{endpoint}.json", params=params, timeout=30)
    if r.status_code == 404:  # openFDA uses 404 for "no matches"
        return {"meta": {"results": {"total": 0}}, "results": []}
    r.raise_for_status()
    return r.json()


def _call(endpoint: str, **params) -> tuple[dict, str]:
    frozen = tuple(sorted((k, str(v)) for k, v in params.items()))
    return _get(endpoint, frozen), _url(endpoint, dict(frozen))


def _recall(x: dict) -> dict:
    return {
        "recall_number": x.get("recall_number"),
        "product": (x.get("product_description") or "")[:300],
        "reason": (x.get("reason_for_recall") or "")[:300],
        "classification": x.get("classification"),
        "firm": x.get("recalling_firm"),
        "recall_initiation_date": x.get("recall_initiation_date"),
        "status": x.get("status"),
    }


def _recalls(search: str, limit: int) -> dict:
    data, url = _call("enforcement", search=search, limit=max(1, min(int(limit), 25)), sort="recall_initiation_date:desc")
    results = [_recall(x) for x in data.get("results", [])]
    return {"total": data.get("meta", {}).get("results", {}).get("total", 0), "results": results,
            "sources": [{"id": r["recall_number"], "url": url} for r in results if r["recall_number"]]}


def search_recalls_by_drug(drug_name: str, limit: int = 10) -> dict:
    return _recalls(f'product_description:"{_clean(drug_name)}"', limit)


def search_recalls_by_firm(firm_name: str, limit: int = 10) -> dict:
    return _recalls(f'recalling_firm:"{_clean(firm_name)}"', limit)


def recent_class1_recalls(limit: int = 10) -> dict:
    return _recalls('classification:"Class I"', limit)


def recall_counts_by_class(drug_name: str) -> dict:
    data, url = _call("enforcement", search=f'product_description:"{_clean(drug_name)}"', count="classification.exact")
    results = data.get("results", [])
    sid = f"recall-counts:{drug_name.lower()}"
    return {"total": sum(r["count"] for r in results), "results": results,
            "sources": [{"id": sid, "url": url}] if results else []}


def top_adverse_reactions(drug_name: str, limit: int = 10) -> dict:
    data, url = _call("event", search=f'patient.drug.openfda.generic_name:"{_clean(drug_name)}"',
                      count="patient.reaction.reactionmeddrapt.exact", limit=max(1, min(int(limit), 25)))
    results = data.get("results", [])
    sid = f"faers-reactions:{drug_name.lower()}"
    return {"total": len(results), "results": results,
            "note": "FAERS report counts. Reports do not prove the drug caused the reaction.",
            "sources": [{"id": sid, "url": url}] if results else []}


def label_warnings(drug_name: str) -> dict:
    data, url = _call("label", search=f'openfda.generic_name:"{_clean(drug_name)}"', limit=1)
    results = []
    for x in data.get("results", []):
        def first(key):
            v = x.get(key) or [""]
            return v[0][:1200]
        results.append({"set_id": x.get("set_id"), "effective_time": x.get("effective_time"),
                        "brand_name": (x.get("openfda", {}).get("brand_name") or [None])[0],
                        "boxed_warning": first("boxed_warning"),
                        "warnings": first("warnings_and_cautions") or first("warnings"),
                        "contraindications": first("contraindications")})
    return {"total": len(results), "results": results,
            "sources": [{"id": f"label:{r['set_id']}", "url": url} for r in results if r["set_id"]]}


TOOLS = {
    "search_recalls_by_drug": search_recalls_by_drug,
    "search_recalls_by_firm": search_recalls_by_firm,
    "recent_class1_recalls": recent_class1_recalls,
    "recall_counts_by_class": recall_counts_by_class,
    "top_adverse_reactions": top_adverse_reactions,
    "label_warnings": label_warnings,
}
