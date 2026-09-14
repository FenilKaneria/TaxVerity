# Step 15.1 (ADR-076). One image, two uses: this is the container ECR/Lambda
# runs (via Lambda Web Adapter, RESPONSE_STREAM mode) *and* the `api` service
# `compose.yaml` runs locally (Step 15.3) — the adapter binary at
# /opt/extensions/lambda-adapter is a Lambda extension, inert outside a Lambda
# execution environment, so no second Dockerfile is needed.
#
# No torch, no baked embedding weights: ADR-075 moved embeddings and
# reranking to the Jina hosted API, which is what makes this fit inside
# Lambda's 10 s INIT budget at all (ADR-076). `uv sync --no-dev` installs
# neither the `dev` dependency group (pytest, ruff, hypothesis, deepeval) nor
# the `embed` optional extra (torch, sentence-transformers) — that extra
# exists only for the offline local fidelity reference, never for serving.

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies before source, for layer caching: an unrelated source edit
# should not force a full dependency re-resolve.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY README.md ./
RUN uv sync --locked --no-dev

# Lambda Web Adapter proxies HTTP requests from the Lambda runtime (or a
# function URL) to the uvicorn server below on $PORT. response_stream mode is
# what makes claim-level SSE (rule 04) reach the client incrementally instead
# of buffering the whole response — required, not an optimisation, since ALB
# cannot stream at all and is why Lambda has no ALB in front of it (ADR-076).
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
ENV PORT=8080 \
    AWS_LWA_INVOKE_MODE=response_stream \
    AWS_LWA_READINESS_CHECK_PATH=/health

EXPOSE 8080

CMD ["uvicorn", "taxverity.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
