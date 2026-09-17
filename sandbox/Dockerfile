# Reproducible, network-isolated environment for the TRIBUNE appeals-workflow eval.
#
# Build:
#   docker build -f sandbox/Dockerfile -t tribune-appeals-eval .
# Run (egress denied at container boundary + in-process netguard):
#   docker run --rm --network=none -v $(pwd)/artifacts:/app/artifacts:rw tribune-appeals-eval
#
# Security profile:
# - Unprivileged user (uid 10001)
# - Deny-by-default egress (--network=none)
# - Pinned Python 3.12-slim base
# - Read/write isolated to /app/artifacts volume mount

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Deterministic seeds for the eval run.
    TRIBUNE_SEED=7 \
    PYTHONHASHSEED=0 \
    # Belt-and-suspenders: keep offline defaults explicit.
    TRIBUNE_PROVIDER=local_rules \
    TRIBUNE_VERIFIER_PROVIDER=local_rules \
    TRIBUNE_RULE_STORE=local \
    TRIBUNE_DOC_INGEST=structured \
    TRIBUNE_SANDBOX_MODE=container \
    TRIBUNE_NETGUARD=deny_all \
    TRIBUNE_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

# Create unprivileged execution user
RUN useradd --create-home --uid 10001 evaluser && \
    mkdir -p /app/artifacts && \
    chown -R evaluser:evaluser /app

# Install dependencies first (layer cached)
COPY sandbox/requirements.lock /app/sandbox/requirements.lock
RUN pip install -r /app/sandbox/requirements.lock

COPY pyproject.toml README.md /app/
COPY tribune /app/tribune
RUN pip install --no-deps -e .

USER evaluser

ENTRYPOINT ["/app/sandbox/run_appeals_eval.sh"]
