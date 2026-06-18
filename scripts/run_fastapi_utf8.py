from __future__ import annotations

import asyncio
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "src.06_api.main:app",
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
        reload=False,
    )
