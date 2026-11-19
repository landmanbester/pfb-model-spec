# uv is a named stage, not `COPY --from=<image>`, because Dependabot only reads
# FROM lines; .github/dependabot.yml bumps it. Keep it no older than the uv that
# writes uv.lock, since `uv export --locked` below re-checks the lock with it.
FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM python:3.11-slim

WORKDIR /app

# TRANSITION (#48): pyproject's [tool.uv.sources] pulls hip-cargo from git until
# 0.5.0 is on PyPI. Remove this layer together with that table.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# Install the locked dependencies first, so every architecture gets the versions
# CI tested (and this layer is cached until uv.lock changes), then the package
# itself without re-resolving. --locked fails the build if uv.lock is stale.
COPY pyproject.toml uv.lock ./
RUN uv export --locked --no-dev --extra full --no-emit-project --no-header -o /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# LICENSE is required by pyproject's license-files glob.
COPY README.md LICENSE ./
COPY src/ src/
RUN uv pip install --system --no-cache --no-deps .

# Make CLI available
CMD ["pfbspec", "--help"]
