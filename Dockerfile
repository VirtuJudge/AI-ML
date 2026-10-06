FROM python:3.11-slim AS base

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN python -m pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY app ./app
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

FROM base AS fake
ENV AI_PROVIDER_MODE=fake
CMD ["python", "-m", "app"]

FROM base AS live
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ffmpeg \
        libegl1 \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
        libgles2 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN uv sync --frozen --no-dev \
    --extra speech \
    --extra vision \
    --extra audio \
    --extra documents \
    --extra embeddings \
    --extra storage \
    --extra groq
ENV AI_PROVIDER_MODE=live
CMD ["python", "-m", "app"]
