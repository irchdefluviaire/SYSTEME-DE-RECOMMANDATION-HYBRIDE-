"""Runtime settings for Agentic GraphRAG."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_TOP_K = int(os.getenv("AGENT_TOP_K", "5"))
MAX_REPLAN = int(os.getenv("AGENT_MAX_REPLAN", "1"))

USE_REAL_DBS = True

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.1:latest"
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
USE_OLLAMA = True

LANGSMITH_TRACING = os.getenv(
    "LANGSMITH_TRACING",
    os.getenv("LANGCHAIN_TRACING_V2", "false"),
).strip().lower() in {"1", "true", "yes"}
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agentic-graphrag-recommandation")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

MIN_SCORE_FINAL = float(os.getenv("AGENT_MIN_SCORE_FINAL", "0.45"))
MAX_ESSENTIAL_GAPS = int(os.getenv("AGENT_MAX_ESSENTIAL_GAPS", "3"))
