FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /app

COPY requirements.runtime.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.runtime.txt

COPY --chown=app:app . .

USER app

EXPOSE 8000 8501

CMD ["streamlit", "run", "chatbot_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.enableCORS=false"]
