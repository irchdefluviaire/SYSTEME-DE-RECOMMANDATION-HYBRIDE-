"""
evaluate_graphrag.py
===========================================================================
Evaluation end-to-end legere pour GraphRAG.

Ce module fournit des metriques calculables sans juge externe:
- context_recall: part des preuves attendues recuperees;
- faithfulness_proxy: recouvrement lexical reponse/contexte;
- answer_correctness_proxy: approximation locale, a remplacer par annotation
  humaine ou LLM-as-judge lorsque disponible.
===========================================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "05_graphrag"))

from answer_critic import critique_answer  # noqa: E402


@dataclass
class GraphRAGEvalResult:
    query_id: str
    context_recall: float
    faithfulness_proxy: float
    answer_correctness_proxy: float
    n_expected: int
    n_retrieved: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def context_recall(expected_ids: set[str], retrieved_ids: list[str]) -> float:
    if not expected_ids:
        return 0.0
    return round(len(expected_ids & set(retrieved_ids)) / len(expected_ids), 4)


def evaluate_one(
    *,
    query_id: str,
    answer: str,
    context: Any,
    expected_evidence_ids: set[str],
    retrieved_evidence_ids: list[str],
) -> dict[str, Any]:
    critic = critique_answer(answer, context)
    return GraphRAGEvalResult(
        query_id=query_id,
        context_recall=context_recall(expected_evidence_ids, retrieved_evidence_ids),
        faithfulness_proxy=float(critic["faithfulness"]),
        answer_correctness_proxy=float(critic["answer_correctness_proxy"]),
        n_expected=len(expected_evidence_ids),
        n_retrieved=len(retrieved_evidence_ids),
    ).to_dict()


def evaluate_batch(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [evaluate_one(**case) for case in cases]
    if not rows:
        return {"n_cases": 0}
    return {
        "n_cases": len(rows),
        "context_recall": round(sum(r["context_recall"] for r in rows) / len(rows), 4),
        "faithfulness_proxy": round(sum(r["faithfulness_proxy"] for r in rows) / len(rows), 4),
        "answer_correctness_proxy": round(
            sum(r["answer_correctness_proxy"] for r in rows) / len(rows),
            4,
        ),
        "details": rows,
    }
