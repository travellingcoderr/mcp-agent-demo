# Project 04: Drug Safety Agent with MCP (Public Live Demo)

A public, clickable version of your Eli Lilly-style agent work, built only on **public data**. An agent answers questions about FDA drug recalls and adverse-event trends using an **MCP server** that wraps the free **openFDA API**, with grounding, structured outputs, and "abstain when unsure" logic.

**Why it matters:** your client work is confidential. This lets recruiters and hiring managers see the same skills (MCP, tool-calling agents, anti-hallucination controls, evaluation) in a live demo.

## Architecture
```
Next.js chat UI ──► FastAPI + LangGraph agent ──► Claude / Azure OpenAI
                                 │
                                 └──► MCP server (src/fda_mcp_server.py) ──► api.fda.gov (read-only)
                    Langfuse tracing on every call
```

## Milestones (about 2–3 weeks)
1. Run the MCP server locally and test it with an MCP client (Claude Desktop or the MCP Inspector)
2. Build a LangGraph agent that calls the MCP tools; force Pydantic structured answers with citations to recall numbers
3. Add a 30-question golden set + evaluation (grounding rate, hallucination rate), reusing Project 02's approach
4. Add Langfuse tracing and the Project 03 red-team gate
5. Deploy (e.g., Azure Container Apps or AWS App Runner) and put the live link on your resume and LinkedIn

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/fda_mcp_server.py            # stdio MCP server
npx @modelcontextprotocol/inspector python src/fda_mcp_server.py   # optional: test tools in a browser
```

## Resume bullet once finished
> Built and deployed a public drug-safety AI agent (LangGraph, MCP server over openFDA, Claude) with grounded, cited answers, golden-set evaluation (__% grounding, __% hallucination), and Langfuse tracing. Live demo: <link>
