# Reproducible, network-isolated environment for the TRIBUNE appeals-workflow eval.
#
# Build:
#   docker build -f sandbox/Dockerfile -t tribune-appeals-eval .
# Run (egress denied at the container boundary — see the deny-by-default note):
#   docker run --rm --network=none tribune-appeals-eval
#
# The image pins its base and dependencies so the eval is reproducible from a
# clean checkout. The eval itself is fixtures-only and offline; --network=none
# enforces that at the container boundary, and the in-process netguard enforces
# it again inside the process (belt and suspenders).

FROM python:3.12-slim@sha256:pinnedbypolicy AS base
# NOTE: replace the digest above with the one your org pins (kept as a placeholder
# so this file documents intent without hard-coding a digest that may rotate).

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Deterministic seeds for the eval run.
    TRIBUNE_SEED=7 \
    PYTHONHASHSEED=0 \
    # Belt-and-suspenders: keep the offline defaults explicit.
    TRIBUNE_PROVIDER=local_rules \
    TRIBUNE_VERIFIER_PROVIDER=local_rules \
    TRIBUNE_RULE_STORE=local \
    TRIBUNE_DOC_INGEST=structured

WORKDIR /app

# Install pinned deps first (layer-cached), then the package itself.
COPY sandbox/requirements.lock /app/sandbox/requirements.lock
RUN pip install -r /app/sandbox/requirements.lock

COPY pyproject.toml README.md /app/
COPY tribune /app/tribune
RUN pip install --no-deps -e .

# Drop privileges.
RUN useradd --create-home --uid 10001 evaluser
USER evaluser

ENTRYPOINT ["/app/sandbox/run_appeals_eval.sh"]
