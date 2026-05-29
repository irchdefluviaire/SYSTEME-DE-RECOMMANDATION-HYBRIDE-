"""Agentic GraphRAG chat endpoints."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parents[3]
AGENT_DIR = ROOT / "src" / "08_agentic_graphrag"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from schemas import ChatRequest, ChatResponse  # noqa: E402

router = APIRouter()


def _invoke_agent(request: ChatRequest) -> dict:
    from graph import graph  # noqa: PLC0415

    state = {
        "messages": [("user", request.query)],
        "top_k": request.top_k,
        "backend": "openrouter",
    }
    if request.candidat_id:
        state["candidat_id"] = request.candidat_id

    result = graph.invoke(state)
    answer = ""
    for message in reversed(result.get("messages", [])):
        if hasattr(message, "content") and getattr(message, "content", None):
            answer = str(message.content)
            break
    return {
        "answer": answer,
        "use_case": result.get("use_case"),
        "candidat_id": result.get("candidat_id"),
        "top_k": result.get("top_k", request.top_k),
        "traces": [str(t) for t in result.get("traces", [])],
        "critic": result.get("critic", {}) or {},
    }


@router.post("", response_model=ChatResponse, summary="Agent GraphRAG non streame")
async def chat(request: ChatRequest):
    t0 = time.time()
    try:
        payload = _invoke_agent(request)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not request.include_traces:
        payload["traces"] = []
        payload["critic"] = {}
    return ChatResponse(**payload, latence_ms=round((time.time() - t0) * 1000, 1))


@router.post("/stream", summary="Agent GraphRAG en Server-Sent Events")
async def chat_stream(request: ChatRequest):
    async def event_stream():
        t0 = time.time()
        yield "event: status\ndata: " + json.dumps({"step": "start"}, ensure_ascii=False) + "\n\n"
        try:
            payload = _invoke_agent(request)
            payload["latence_ms"] = round((time.time() - t0) * 1000, 1)
            if not request.include_traces:
                payload["traces"] = []
                payload["critic"] = {}
            yield "event: answer\ndata: " + json.dumps(payload, ensure_ascii=False, default=str) + "\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            err = {"error": str(exc), "latence_ms": round((time.time() - t0) * 1000, 1)}
            yield "event: error\ndata: " + json.dumps(err, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
