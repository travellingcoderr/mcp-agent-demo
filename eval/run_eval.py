"""Golden-set evaluation for the drug-safety agent.

Ground truth for fact questions is computed live from openFDA at eval time (the data changes weekly),
so the eval checks correctness, not memorized answers.

Metrics
  answer_rate          answerable questions that got a grounded (cited) answer
  abstain_accuracy     out-of-scope / advice / no-data questions where the agent abstained
  tool_accuracy        answerable questions where the expected tool was called
  fact_accuracy        fact-checked questions whose answer contains the live ground-truth value
  hallucinated_citation_rate  answers where the model cited an id no tool returned (caught by the guard)
  p50 / p95 latency

  LLM_BASE_URL=... LLM_MODEL=... OPENAI_API_KEY=... python eval/run_eval.py --out eval/results-<model>.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import openfda  # noqa: E402
from agent import DrugSafetyAgent, MCPToolbox, make_client  # noqa: E402


def truth(fact: dict) -> str | None:
    k = fact["kind"]
    if k == "latest_recall":
        r = openfda.search_recalls_by_drug(fact["drug"], 1)["results"]
        return r[0]["recall_number"] if r else None
    if k == "latest_class1":
        r = openfda.recent_class1_recalls(1)["results"]
        return r[0]["recall_number"] if r else None
    if k == "class_count":
        r = {x["term"]: x["count"] for x in openfda.recall_counts_by_class(fact["drug"])["results"]}
        return str(r.get(fact["class"], 0))
    if k == "top_reaction":
        r = openfda.top_adverse_reactions(fact["drug"], 1)["results"]
        return r[0]["term"] if r else None
    raise ValueError(k)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else 0


async def main(a):
    rows = [json.loads(line) for line in open(ROOT / "eval" / "golden.jsonl")]
    if a.limit:
        rows = rows[: a.limit]
    out = []
    async with MCPToolbox() as tb:
        agent = DrugSafetyAgent(make_client(), a.model, tb)
        for r in rows:
            res = await agent.ask(r["question"])
            ans = res.answer
            row = {"id": r["id"], "expect": r["expect"], "abstained": ans.abstain, "citations": ans.citations,
                   "ungrounded": res.ungrounded_citations, "tools": res.tools_called, "latency_s": res.latency_s,
                   "answer": ans.answer[:400], "abstain_reason": ans.abstain_reason}
            if r["expect"] == "answer":
                row["tool_ok"] = r["expect_tool"] in res.tools_called
                if "fact" in r:
                    t = truth(r["fact"])
                    row["truth"] = t
                    hay = (ans.answer + " " + " ".join(ans.citations)).lower()
                    row["fact_ok"] = bool(t) and not ans.abstain and t.lower() in hay
            row["correct"] = (not ans.abstain) if r["expect"] == "answer" else ans.abstain
            print(f"{'OK ' if row['correct'] else 'BAD'} {r['id']} abstain={ans.abstain} tools={res.tools_called} cites={ans.citations[:3]}", flush=True)
            out.append(row)

    ans_rows = [x for x in out if x["expect"] == "answer"]
    abs_rows = [x for x in out if x["expect"] == "abstain"]
    fact_rows = [x for x in ans_rows if "fact_ok" in x]
    answered = [x for x in out if not x["abstained"]]
    summary = {
        "model": a.model, "n": len(out),
        "answer_rate": round(sum(not x["abstained"] for x in ans_rows) / max(1, len(ans_rows)), 3),
        "abstain_accuracy": round(sum(x["abstained"] for x in abs_rows) / max(1, len(abs_rows)), 3),
        "tool_accuracy": round(sum(x["tool_ok"] for x in ans_rows) / max(1, len(ans_rows)), 3),
        "fact_accuracy": round(sum(x["fact_ok"] for x in fact_rows) / max(1, len(fact_rows)), 3),
        "grounded_answer_rate": round(sum(bool(x["citations"]) for x in answered) / max(1, len(answered)), 3),
        "hallucinated_citation_rate": round(sum(bool(x["ungrounded"]) for x in out) / max(1, len(out)), 3),
        "p50_latency_s": round(pct([x["latency_s"] for x in out], 0.5), 2),
        "p95_latency_s": round(pct([x["latency_s"] for x in out], 0.95), 2),
    }
    print(json.dumps(summary, indent=2))
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": out}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4.1-mini"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    asyncio.run(main(p.parse_args()))
