"""Read-only MCP server exposing openFDA drug recall data as tools."""
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("openfda-drug-safety")
BASE = "https://api.fda.gov/drug/enforcement.json"

def _query(search: str, limit: int) -> list[dict]:
    r = httpx.get(BASE, params={"search": search, "limit": max(1, min(limit, 25))}, timeout=30)
    if r.status_code == 404:  # openFDA returns 404 when there are no matches
        return []
    r.raise_for_status()
    return [{
        "recall_number": x.get("recall_number"),
        "product": x.get("product_description", "")[:300],
        "reason": x.get("reason_for_recall", "")[:300],
        "classification": x.get("classification"),
        "firm": x.get("recalling_firm"),
        "date": x.get("recall_initiation_date"),
        "status": x.get("status"),
    } for x in r.json().get("results", [])]

@mcp.tool()
def search_recalls_by_drug(drug_name: str, limit: int = 10) -> list[dict]:
    """Find FDA drug recalls whose product description mentions the given drug name."""
    return _query(f'product_description:"{drug_name}"', limit)

@mcp.tool()
def search_recalls_by_firm(firm_name: str, limit: int = 10) -> list[dict]:
    """Find FDA drug recalls initiated by a given company."""
    return _query(f'recalling_firm:"{firm_name}"', limit)

@mcp.tool()
def recent_class1_recalls(limit: int = 10) -> list[dict]:
    """Most serious (Class I) drug recalls."""
    return _query('classification:"Class I"', limit)

if __name__ == "__main__":
    mcp.run()
