"""FastAPI app: POST /api/ask plus a one-page chat UI at /.

  LLM_BASE_URL=https://api.openai.com/v1 LLM_MODEL=gpt-4.1-mini OPENAI_API_KEY=... uvicorn src.api:app --port 8080
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agent import DrugSafetyAgent, MCPToolbox, make_client, write_trace  # noqa: E402

STATIC = Path(__file__).resolve().parent.parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with MCPToolbox() as tb:
        state["agent"] = DrugSafetyAgent(make_client(), os.getenv("LLM_MODEL", "gpt-4.1-mini"), tb)
        yield


app = FastAPI(title="Drug Safety Agent (openFDA + MCP)", lifespan=lifespan)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/healthz")
def healthz():
    return {"ok": "agent" in state}


@app.post("/api/ask")
async def ask(body: Ask):
    try:
        res = await state["agent"].ask(body.question)
    except Exception as e:
        raise HTTPException(502, f"agent error: {type(e).__name__}") from e
    write_trace(res)
    return {"answer": res.answer.model_dump(), "sources": {c: res.sources[c] for c in res.answer.citations},
            "tools_called": res.tools_called, "latency_s": res.latency_s, "trace_id": res.trace_id}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
