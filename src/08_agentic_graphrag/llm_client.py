"""Strict OpenRouter client for the Agentic GraphRAG workflow.

The project policy is explicit: use OPENROUTER_MODEL from .env, normally
openai/gpt-oss-20b:free. The client does not silently switch models.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_STRICT_MODEL = "openai/gpt-oss-20b:free"


class OpenRouterClient:
    """Call OpenRouter with the single configured model."""

    def __init__(self) -> None:
        load_dotenv(ROOT / ".env", override=True)
        self.api_key: str = (
            os.getenv("API_KEY_OPEN_ROUTEUR")
            or os.getenv("OPENROUTER_API_KEY")
            or ""
        )
        self.model: str = os.getenv("OPENROUTER_MODEL", _STRICT_MODEL)
        self.base_url: str = os.getenv("OPENROUTER_BASE_URL", _OPENROUTER_BASE).rstrip("/")
        self.timeout_s: int = int(os.getenv("LLM_TIMEOUT_S", "60"))
        self.max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1200"))

    def _call_model(self, system: str, user: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/irchdefluviaire",
                "X-Title": "Conseiller Emploi-Competences Cameroun",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter inaccessible: {exc.reason}") from exc

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        """Generate with OPENROUTER_MODEL only; retry once on 429."""

        if not self.api_key:
            raise RuntimeError("Cle API OpenRouter manquante: ajoute API_KEY_OPEN_ROUTEUR dans .env")

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                log.info(
                    "OpenRouter -> model=%s attempt=%d max_tokens=%d",
                    self.model,
                    attempt + 1,
                    self.max_tokens,
                )
                return self._call_model(system, user, temperature)
            except RuntimeError as exc:
                last_error = exc
                if "429" in str(exc) and attempt == 0:
                    log.warning("429 sur %s - attente 8s avant retry", self.model)
                    time.sleep(8)
                    continue
                break

        raise RuntimeError(
            f"Le modele OpenRouter configure ({self.model}) a echoue. Derniere erreur: {last_error}"
        )


def get_llm_client() -> OpenRouterClient:
    """Return a fresh client so .env edits are picked up."""

    return OpenRouterClient()
