FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/cache/huggingface
ENV DOCLING_CACHE_DIR=/app/cache/docling

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY titan ./titan
COPY baml_src ./baml_src
COPY data ./data
COPY main.py ./main.py

RUN mkdir -p /app/cache/huggingface /app/cache/docling

ENTRYPOINT ["python", "-m", "titan.cli"]
CMD ["index-query", "--query", "Who is the vested owner?", "--top-k", "5", "--qdrant-url", "http://qdrant:6333"]
