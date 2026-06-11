"""
answer_critic.py
===========================================================================
Critique legere de fidelite au contexte pour les reponses GraphRAG.

Ce module ne remplace pas une evaluation humaine ou un LLM-as-judge. Il fournit
un garde-fou local, deterministe, utile pour detecter une reponse sans preuve
ou trop deconnectee du contexte recupere.
===========================================================================
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

TOKEN_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
STOPWORDS = {
    "avec", "dans", "des", "les", "une", "pour", "sur", "aux", "du", "de",
    "la", "le", "et", "ou", "est", "sont", "que", "qui", "par", "plus",
    "moins", "cette", "cela", "ce", "ces", "un", "en", "au", "a",
}


@dataclass
class CriticResult:
    faithfulness: float
    answer_correctness_proxy: float
    evidence_coverage: float
    unsupported_terms: list[str]
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in TOKEN_RE.findall(str(text))
        if len(t) >= 4 and t.lower() not in STOPWORDS
    }


def _flatten_context(context: Any) -> str:
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        parts = []
        for value in context.values():
            parts.append(_flatten_context(value))
        return "\n".join(parts)
    if isinstance(context, list):
        return "\n".join(_flatten_context(x) for x in context)
    return str(context)


def critique_answer(answer: str, context: Any, *, min_faithfulness: float = 0.46) -> dict[str, Any]:
    """Retourne une critique locale de la reponse par recouvrement lexical."""

    answer_tokens = _tokens(answer)
    context_tokens = _tokens(_flatten_context(context))
    if not answer_tokens:
        return CriticResult(0.0, 0.0, 0.0, [], "revise").to_dict()
    if not context_tokens:
        return CriticResult(0.0, 0.0, 0.0, sorted(answer_tokens)[:12], "revise").to_dict()

    supported = answer_tokens & context_tokens
    faithfulness = len(supported) / max(len(answer_tokens), 1)
    evidence_coverage = len(supported) / max(len(context_tokens), 1)
    unsupported = sorted(answer_tokens - context_tokens)[:15]
    decision = "accept" if faithfulness >= min_faithfulness else "revise"

    return CriticResult(
        faithfulness=round(faithfulness, 4),
        answer_correctness_proxy=round(faithfulness, 4),
        evidence_coverage=round(evidence_coverage, 4),
        unsupported_terms=unsupported,
        decision=decision,
    ).to_dict()
