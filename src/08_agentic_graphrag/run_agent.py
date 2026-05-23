"""CLI helper to test the Agentic GraphRAG workflow outside Studio."""

from __future__ import annotations

import argparse
import json

from langchain_core.messages import HumanMessage

from graph import graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Agentic GraphRAG LangGraph")
    parser.add_argument("--candidat", required=True, help="Identifiant candidat")
    parser.add_argument("--query", default="Recommande les meilleures offres et une roadmap.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["real"], default="real")
    args = parser.parse_args()

    result = graph.invoke(
        {
            "user_query": args.query,
            "messages": [HumanMessage(content=args.query)],
            "candidat_id": args.candidat,
            "top_k": args.top_k,
            "mode": "real",
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
