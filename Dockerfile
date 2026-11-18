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
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/

# Install package with full dependencies.
# TRANSITION (#48): constrain hip-cargo to the commit uv.lock pins, so every
# architecture gets the revision CI tested rather than whatever main is at build
# time. Drop uv.lock, the export and the -c with the git source.
RUN uv export --frozen --no-hashes --no-emit-project --no-default-groups --extra full \
        | grep '^hip-cargo @ ' > /tmp/hip-cargo-pin.txt \
    && cat /tmp/hip-cargo-pin.txt \
    && uv pip install --system --no-cache ".[full]" -c /tmp/hip-cargo-pin.txt

# Make CLI available
CMD ["pfbspec", "--help"]
