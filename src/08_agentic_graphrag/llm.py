"""Small Ollama client used by the LangGraph final generation node."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from settings import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE


def ollama_generate(
    prompt: str,
    *,
    system: str = "",
    model: str = OLLAMA_MODEL,
    timeout_s: int = 300,
) -> str:
    """Call Ollama through LangChain when possible, with HTTP fallback.

    The LangChain path is preferable because LangSmith can attach the model call
    to the LangGraph trace when tracing is enabled.
    """

    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_TEMPERATURE,
            num_predict=220,
        )
        messages = [HumanMessage(content=prompt)]
        if system:
            messages.insert(0, SystemMessage(content=system))
        response = llm.invoke(messages)
        return str(response.content).strip()
    except Exception:
        pass

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": OLLAMA_TEMPERATURE, "num_predict": 220},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
            return str(body.get("response", "")).strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return (
            "Le modele Ollama local n'a pas repondu. "
            f"Backend={model}, erreur={exc}. "
            "La recommandation structuree reste disponible dans les traces."
        )
