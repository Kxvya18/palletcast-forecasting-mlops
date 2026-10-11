# Forecast API (S6). Contains the code, the configs and one exported production model.
# The validated data is mounted at run time (it changes every month; the image does not).
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-default-groups      # core libraries only (no training stack)

COPY serving_model ./model
ENV MODEL_URI=/app/model \
    DATA_DIR=/app/data/validated
EXPOSE 8000
CMD ["uv", "run", "--no-default-groups", "uvicorn", "palletcast.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
