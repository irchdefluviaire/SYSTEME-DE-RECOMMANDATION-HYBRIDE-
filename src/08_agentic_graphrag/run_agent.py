"""
CLI de test du workflow Agentic GraphRAG.

Usage:
  poetry run python src/08_agentic_graphrag/run_agent.py --candidat PPKOU2501080016340 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("AGENT_LANGSMITH_TRACING", "false")

from graph import graph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Agentic GraphRAG")
    parser.add_argument("--candidat", required=True, help="Identifiant candidat")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre d'offres")
    parser.add_argument(
        "--backend",
        choices=["simulation", "llama"],
        default="simulation",
        help="Backend LLM du moteur GraphRAG",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Afficher l'etat final complet en JSON compact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = {
        "messages": [
            ("user", f"Analyse le candidat {args.candidat} et propose les meilleures offres.")
        ],
        "candidat_id": args.candidat,
        "top_k": args.top_k,
        "backend": args.backend,
    }
    result = graph.invoke(state)

    if args.json:
        printable = {
            "candidat_id": result.get("candidat_id"),
            "top_k": result.get("top_k"),
            "backend": result.get("backend"),
            "traces": result.get("traces", []),
            "result": result.get("result", {}),
            "answer": result["messages"][-1].content if result.get("messages") else "",
        }
        print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
        return

    print(result["messages"][-1].content)
    print()
    print("Traces:", " -> ".join(result.get("traces", [])))


if __name__ == "__main__":
    main()
