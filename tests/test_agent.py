"""Offline tests: a scripted fake LLM + fake toolbox, plus a real MCP round trip with openFDA stubbed."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import openfda  # noqa: E402
from agent import ADVICE, Answer, DrugSafetyAgent, ground, parse_answer  # noqa: E402

RECALL = {"total": 1, "results": [{"recall_number": "D-0001-2026", "product": "Metformin ER 500mg", "reason": "NDMA",
                                   "classification": "Class II", "firm": "Acme", "recall_initiation_date": "20260101",
                                   "status": "Ongoing"}],
          "sources": [{"id": "D-0001-2026", "url": "https://api.fda.gov/drug/enforcement.json?x"}]}


class FakeToolbox:
    tool_specs = [{"type": "function", "function": {"name": "search_recalls_by_drug", "description": "",
                                                    "parameters": {"type": "object", "properties": {"drug_name": {"type": "string"}}}}}]

    def __init__(self):
        self.calls = []

    async def call(self, name, args):
        self.calls.append((name, args))
        return RECALL


class ScriptedLLM:
    """Returns the queued messages in order, like chat.completions.create would."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.chat = NS(completions=NS(create=self.create))
        self.seen = []

    def create(self, **kw):
        self.seen.append(kw["messages"])
        return NS(choices=[NS(message=self.replies.pop(0))], usage=None)


def tool_call(name, args):
    return NS(content=None, tool_calls=[NS(id="c1", function=NS(name=name, arguments=json.dumps(args)))])


def final(obj):
    return NS(content=json.dumps(obj), tool_calls=None)


def run(coro):
    return asyncio.run(coro)


def test_grounded_answer_keeps_valid_citations():
    llm = ScriptedLLM(tool_call("search_recalls_by_drug", {"drug_name": "metformin"}),
                      final({"answer": "One recall, D-0001-2026, for NDMA.", "citations": ["D-0001-2026"], "abstain": False}))
    tb = FakeToolbox()
    res = run(DrugSafetyAgent(llm, "m", tb).ask("Any metformin recalls?"))
    assert not res.answer.abstain and res.answer.citations == ["D-0001-2026"]
    assert tb.calls == [("search_recalls_by_drug", {"drug_name": "metformin"})]
    # tool output is passed back to the model marked as untrusted
    assert "<untrusted_tool_result>" in llm.seen[1][-1]["content"]


def test_invented_citation_forces_abstain():
    llm = ScriptedLLM(tool_call("search_recalls_by_drug", {"drug_name": "metformin"}),
                      final({"answer": "Recall D-9999-2026.", "citations": ["D-9999-2026"], "abstain": False}))
    res = run(DrugSafetyAgent(llm, "m", FakeToolbox()).ask("Any metformin recalls?"))
    assert res.answer.abstain and res.ungrounded_citations == ["D-9999-2026"]


def test_answer_without_tool_call_is_not_trusted():
    llm = ScriptedLLM(final({"answer": "Metformin was recalled in 2020.", "citations": ["D-0001-2026"], "abstain": False}))
    res = run(DrugSafetyAgent(llm, "m", FakeToolbox()).ask("Any metformin recalls?"))
    assert res.answer.abstain  # cited id was never returned by a tool in this run


def test_medical_advice_short_circuits_without_llm_call():
    llm = ScriptedLLM()
    res = run(DrugSafetyAgent(llm, "m", FakeToolbox()).ask("Should I stop taking my metformin?"))
    assert res.answer.abstain and llm.seen == []


def test_unparseable_reply_abstains():
    llm = ScriptedLLM(NS(content="I think it's fine.", tool_calls=None))
    assert run(DrugSafetyAgent(llm, "m", FakeToolbox()).ask("Any recalls?")).answer.abstain


@pytest.mark.parametrize("q,hit", [("How much ibuprofen should I take?", True), ("Is it safe for me to take warfarin?", True),
                                   ("Any recalls of metformin?", False), ("What warnings are on the warfarin label?", False)])
def test_advice_policy(q, hit):
    assert bool(ADVICE.search(q)) == hit


def test_parse_and_ground_helpers():
    a = parse_answer('Here you go: {"answer": "x", "citations": ["A", "B"], "abstain": false}')
    g, bad = ground(a, {"A": "u"})
    assert g.citations == ["A"] and bad == ["B"]
    assert ground(Answer(abstain=True), {})[0].abstain


def test_input_validation_blocks_query_injection():
    with pytest.raises(openfda.InvalidInput):
        openfda.search_recalls_by_drug('metformin" OR recalling_firm:"*')


def test_mcp_server_round_trip(monkeypatch, tmp_path):
    """Start the real MCP server over stdio and call a tool. openFDA is stubbed via sitecustomize."""
    stub = tmp_path / "sitecustomize.py"
    stub.write_text(
        "import openfda, json\n"
        f"openfda._get = lambda ep, fr: {{'meta': {{'results': {{'total': 1}}}}, 'results': [{{'recall_number': 'D-0001-2026', 'product_description': 'Metformin'}}]}}\n"
    )
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}:{ROOT / 'src'}")
    from agent import MCPToolbox

    async def go():
        async with MCPToolbox() as tb:
            names = {t["function"]["name"] for t in tb.tool_specs}
            assert {"search_recalls_by_drug", "top_adverse_reactions", "label_warnings"} <= names
            return await tb.call("search_recalls_by_drug", {"drug_name": "metformin", "limit": 1})

    out = run(go())
    assert out["sources"][0]["id"] == "D-0001-2026"
