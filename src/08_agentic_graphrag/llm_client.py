"""
Client LLM OpenRouter pour le workflow Agentic GraphRAG.

Compatible API OpenAI — aucune dépendance supplémentaire requise.
Lit la clé depuis API_KEY_OPEN_ROUTEUR (ou OPENROUTER_API_KEY) dans .env.
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

# Modèles free essayés dans l'ordre si le modèle principal est rate-limité
_FALLBACK_MODELS = [
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-v4-flash:free",
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

_SYSTEM_CONSEILLER = """Tu es un conseiller emploi-compétences expert pour le marché du travail camerounais.
Tu reçois des données extraites de bases de données (graphe Neo4j, recherche sémantique pgvector, référentiels NCF/MEPC).
Réponds en français, de façon claire, structurée et utile.
Base-toi uniquement sur les données fournies — n'invente aucune information.
Si les données sont insuffisantes, dis-le explicitement."""


class OpenRouterClient:
    """Appelle un LLM via l'API OpenRouter avec retry et fallback de modèles."""

    def __init__(self) -> None:
        # Relit le .env à chaque instanciation pour prendre en compte les changements
        load_dotenv(ROOT / ".env", override=True)
        self.api_key: str = (
            os.getenv("API_KEY_OPEN_ROUTEUR")
            or os.getenv("OPENROUTER_API_KEY")
            or ""
        )
        self.model: str = os.getenv("OPENROUTER_MODEL", _FALLBACK_MODELS[0])
        self.base_url: str = os.getenv("OPENROUTER_BASE_URL", _OPENROUTER_BASE).rstrip("/")
        self.timeout_s: int = int(os.getenv("LLM_TIMEOUT_S", "60"))
        self.max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1200"))

    def _call_model(self, model: str, system: str, user: str, temperature: float) -> str:
        """Envoie une requête pour un modèle donné. Lève RuntimeError si erreur HTTP."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
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
                body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter inaccessible: {exc.reason}") from exc

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
    ) -> str:
        """Génère une réponse via OpenRouter avec retry sur 429 et fallback de modèles."""
        if not self.api_key:
            raise RuntimeError(
                "Clé API OpenRouter manquante — ajoute API_KEY_OPEN_ROUTEUR dans .env"
            )

        # Construit la liste : modèle configuré en premier, puis les fallbacks
        candidates = [self.model] + [m for m in _FALLBACK_MODELS if m != self.model]

        last_error: Exception | None = None
        for model in candidates:
            for attempt in range(2):  # 1 retry par modèle sur 429
                try:
                    log.info(
                        "OpenRouter → model=%s attempt=%d max_tokens=%d",
                        model, attempt + 1, self.max_tokens,
                    )
                    return self._call_model(model, system, user, temperature)
                except RuntimeError as exc:
                    last_error = exc
                    if "429" in str(exc):
                        if attempt == 0:
                            log.warning("429 sur %s — attente 8s avant retry", model)
                            time.sleep(8)
                            continue
                        log.warning("429 persistant sur %s — modèle suivant", model)
                        break  # essaie le modèle suivant
                    # Erreur non-429 (404, 401, réseau) : ne pas réessayer ce modèle
                    log.warning("Erreur %s sur %s — modèle suivant", exc, model)
                    break

        raise RuntimeError(
            f"Tous les modèles OpenRouter ont échoué. Dernière erreur: {last_error}"
        )


def get_llm_client() -> OpenRouterClient:
    """Retourne un OpenRouterClient frais (relit le .env à chaque appel)."""
    return OpenRouterClient()
