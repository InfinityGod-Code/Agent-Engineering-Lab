FROM python:3.14-slim AS builder

RUN pip install uv --quiet
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --quiet

FROM python:3.14-slim
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY . .
EXPOSE 8000
CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
