FROM python:3.11-slim

WORKDIR /app

# TRANSITION (#48): pyproject's [tool.uv.sources] pulls hip-cargo from git until
# 0.5.0 is on PyPI. Remove this layer together with that table.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /usr/local/bin/uv

# Copy package files (LICENSE is required by pyproject's license-files glob)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install package with full dependencies
RUN uv pip install --system --no-cache ".[full]"

# Make CLI available
CMD ["pfbspec", "--help"]
