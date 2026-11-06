"""Read-only MCP server exposing openFDA drug safety data as tools (stdio transport).

  python src/fda_mcp_server.py
  npx @modelcontextprotocol/inspector python src/fda_mcp_server.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402  (mcp>=2; was FastMCP in 1.x)

import openfda  # noqa: E402

mcp = MCPServer("openfda-drug-safety")


@mcp.tool()
def search_recalls_by_drug(drug_name: str, limit: int = 10) -> dict:
    """Most recent FDA drug recalls (enforcement reports) whose product description mentions the drug name."""
    return openfda.search_recalls_by_drug(drug_name, limit)


@mcp.tool()
def search_recalls_by_firm(firm_name: str, limit: int = 10) -> dict:
    """Most recent FDA drug recalls initiated by a given company."""
    return openfda.search_recalls_by_firm(firm_name, limit)


@mcp.tool()
def recent_class1_recalls(limit: int = 10) -> dict:
    """Most recent Class I drug recalls (the most serious class: reasonable probability of serious harm)."""
    return openfda.recent_class1_recalls(limit)


@mcp.tool()
def recall_counts_by_class(drug_name: str) -> dict:
    """Number of recalls for a drug broken down by classification (Class I / II / III)."""
    return openfda.recall_counts_by_class(drug_name)


@mcp.tool()
def top_adverse_reactions(drug_name: str, limit: int = 10) -> dict:
    """Most frequently reported adverse reactions in FAERS for a drug (generic name). Counts, not causation."""
    return openfda.top_adverse_reactions(drug_name, limit)


@mcp.tool()
def label_warnings(drug_name: str) -> dict:
    """Boxed warning, warnings and contraindications from the FDA drug label (generic name)."""
    return openfda.label_warnings(drug_name)


if __name__ == "__main__":
    mcp.run()
