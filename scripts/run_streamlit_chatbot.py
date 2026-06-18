from __future__ import annotations

import asyncio
from pathlib import Path
import sys

LOG = Path(__file__).resolve().parents[1] / "streamlit-chatbot.launcher.log"

def _log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(message + "\n")

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from streamlit.web import cli as stcli


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        "chatbot_app.py",
        "--server.port=8501",
        "--server.address=127.0.0.1",
        "--server.headless=true",
    ]
    _log("starting streamlit chatbot on http://127.0.0.1:8501")
    try:
        sys.exit(stcli.main())
    except Exception as exc:
        _log(f"streamlit failed: {exc!r}")
        raise
