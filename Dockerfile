FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/cache/huggingface \
    DOCLING_CACHE_DIR=/app/cache/docling

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
COPY rules ./rules
COPY main.py streamlit_app.py ./

RUN mkdir -p /app/cache/huggingface /app/cache/docling /app/data/out /app/eval \
    && addgroup --system titan \
    && adduser --system --ingroup titan --home /app --no-create-home titan \
    && chown -R titan:titan /app

USER titan

EXPOSE 8501

ENTRYPOINT ["python", "-m", "titan.cli"]
CMD ["--help"]
