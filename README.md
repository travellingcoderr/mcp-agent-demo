# Drug Safety Agent with MCP (openFDA)

A tool-calling agent that answers questions about FDA drug recalls, adverse-event reports and label warnings. The tools live in an **MCP server** (Python SDK v2) that wraps the free, read-only [openFDA API](https://open.fda.gov/apis/). The agent cites the record ids it used, refuses personal medical advice, and **abstains instead of guessing** when the data doesn't support an answer. Public data only.

```
browser ──► FastAPI (/api/ask, static UI) ──► DrugSafetyAgent ──► any OpenAI-compatible LLM (tool calling)
                                                   │
                                                   └── MCP stdio client ──► fda_mcp_server.py ──► api.fda.gov
                                     traces ──► outputs/traces.jsonl
```

## MCP tools

| Tool | openFDA endpoint | Citation id it returns |
|---|---|---|
| `search_recalls_by_drug` | drug/enforcement | `recall_number` (e.g. `D-0657-2026`) |
| `search_recalls_by_firm` | drug/enforcement | `recall_number` |
| `recent_class1_recalls` | drug/enforcement | `recall_number` |
| `recall_counts_by_class` | drug/enforcement (count) | `recall-counts:<drug>` |
| `top_adverse_reactions` | drug/event (FAERS, count) | `faers-reactions:<drug>` |
| `label_warnings` | drug/label | `label:<set_id>` |

Inputs are validated, responses are cached, and `OPENFDA_API_KEY` is used if set.

## Safety controls

The agent is a plain tool loop with no framework, so every decision is visible and unit-tested.

1. **Policy pre-check:** questions asking for personal medical advice ("should I stop taking...") get an abstention before the model is called.
2. **Untrusted tool output:** tool results go back to the model wrapped as data, with an instruction never to follow text inside them (indirect prompt injection via a recall reason field is a real path).
3. **Structured answer:** the model must return JSON `{answer, citations, abstain, abstain_reason}`, validated with Pydantic.
4. **Grounding guard:** every citation must be an id a tool actually returned in *this* run. A made-up recall number turns the answer into an abstention and is counted as a caught hallucination.
5. **Traces:** each run (question, tool calls + args, timings, raw model output, guard decisions) is appended to `outputs/traces.jsonl`.

## Evaluation

`eval/golden.jsonl` has 30 questions: 22 answerable (ground truth computed **live** from openFDA at eval time, since the data changes weekly) and 8 that should be refused or abstained (medical advice, out of scope, drugs with no data).

| Metric | Meaning |
|---|---|
| `answer_rate` | answerable questions that got an answer |
| `grounded_answer_rate` | answers with at least one verified citation |
| `fact_accuracy` | answers containing the live ground-truth value (e.g. latest recall number, total count) |
| `tool_accuracy` | answerable questions where the expected tool was called |
| `abstain_accuracy` | should-abstain questions where it abstained |
| `hallucinated_citation_rate` | answers where the model invented an id (caught by the guard) |
| p50 / p95 latency | end to end |

```bash
LLM_BASE_URL=https://api.openai.com/v1 LLM_MODEL=<tool-calling model> OPENAI_API_KEY=... \
  python eval/run_eval.py --out eval/results-<model>.json
```

Results: _pending a run with a tool-calling model and API key; the numbers will go here exactly as `run_eval.py` reports them._

## Run it

```bash
pip install -r requirements.txt
pytest -q                                   # 12 tests, incl. a real MCP stdio round trip (no LLM needed)

# MCP server alone, e.g. with the MCP Inspector or Claude Desktop
npx @modelcontextprotocol/inspector python src/fda_mcp_server.py

# agent from the CLI
export LLM_BASE_URL=... LLM_MODEL=... OPENAI_API_KEY=...
python src/agent.py "Have there been any recalls of metformin recently?"

# web UI + API
uvicorn src.api:app --port 8080            # open http://localhost:8080
```

Docker and cloud deploy (Azure Container Apps / AWS App Runner): see [docs/deploy.md](docs/deploy.md).

## Next steps

- Export traces to Langfuse or OpenTelemetry instead of JSONL.
- Point the [ai-redteam-toolkit](https://github.com/travellingcoderr/ai-redteam-toolkit) at `/api/ask`, including injection planted in tool results.
- Run the eval against a self-hosted model on [llm-platform-k8s](https://github.com/travellingcoderr/llm-platform-k8s) and compare with a hosted one.
