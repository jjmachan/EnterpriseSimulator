FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
RUN pip install uv

# Install git for building from git repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/env

WORKDIR /app/env

# Install dependencies
RUN uv sync --no-install-project --no-editable && \
    uv sync --no-editable

# Final runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

# Run the FastAPI server on port 7860 (HF Spaces default)
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]
