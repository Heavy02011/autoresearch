FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /autoresearch

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
RUN pip install uv

# Copy dependency files first (layer caching)
COPY pyproject.toml ./
COPY uv.lock* ./

# Install Python dependencies
RUN uv venv /opt/venv --python python3.10 && \
    . /opt/venv/bin/activate && \
    uv sync --no-dev --index-url https://pypi.org/simple \
    || pip install --no-cache-dir pydantic typer structlog omegaconf wandb onnx onnxruntime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY docker/entrypoint.sh ./docker/entrypoint.sh

RUN chmod +x /autoresearch/docker/entrypoint.sh

# Create run output dirs
RUN mkdir -p /autoresearch/runs /autoresearch/logs

# Non-root user for security
RUN useradd --create-home --shell /bin/bash autoresearch && \
    chown -R autoresearch:autoresearch /autoresearch
USER autoresearch

ENV PYTHONPATH="/autoresearch/src"
ENV AUTORESEARCH_RUNS_DIR="/autoresearch/runs"

ENTRYPOINT ["/autoresearch/docker/entrypoint.sh"]
CMD ["run", "--dry-run", "--max-iterations", "3"]
