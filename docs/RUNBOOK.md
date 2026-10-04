# TaxVerity runbook

How to deploy TaxVerity, roll it back, rotate its secrets and keep it running.
Production runs in AWS `ap-south-1`:

- **API:** a Lambda container image from ECR repository `taxverity`, served
  through a function URL in `RESPONSE_STREAM` mode.
- **Frontend:** Vercel.
- **Database:** Supabase Postgres with pgvector.

Resource definitions are in [`infra/main.tf`](../infra/main.tf).

## 1. Deploy the API

Run these from the repository root. You need the AWS CLI logged in to the
account and Docker running.

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY=$ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com
TAG=r25    # one tag per release; never reuse a tag

aws ecr get-login-password --region ap-south-1 \
  | docker login --username AWS --password-stdin $REGISTRY

# --provenance/--sbom off: Lambda rejects the multi-manifest index buildx
# produces by default.
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -t $REGISTRY/taxverity:$TAG --push .

aws lambda update-function-code --function-name taxverity \
  --image-uri $REGISTRY/taxverity:$TAG --region ap-south-1
aws lambda wait function-updated --function-name taxverity --region ap-south-1
```

Then run the checks in section 2.

**Keep the previous tag in ECR.** Rollback depends on it (section 3).

## 2. Check a deploy

1. **Health.** `curl <function-url>/health` should return 200. A cold start
   takes about 4 s.
2. **Serving guard.** `uv run python scripts/check_serving.py` runs with the
   production `TAXVERITY_DATABASE_URL` and the two `SERVING_*` variables. It
   runs the same check the API makes at startup and exits 1 if the corpus
   version or embedding set does not match the database.
3. **Live smoke.** This registers a throwaway account, runs real turns over
   SSE and deletes the account afterwards:

   ```bash
   uv run python scripts/smoke_deploy.py --base-url <function-url>
   ```

   For every turn, check that it ends with a `final` event. A stream cut off
   before `final` means the turn hit the 180 s Lambda timeout.
4. **Answer quality.** This runs multi-turn conversations against the local
   database. Compare against the previous release's run:

   ```bash
   uv run python scripts/advisor_smoke.py --label <tag> --compare <previous-tag>
   ```

   The guard checks must say **PASS**.

## 3. Roll back

Point the function at the previous image tag:

```bash
aws lambda update-function-code --function-name taxverity \
  --image-uri $REGISTRY/taxverity:<previous-tag> --region ap-south-1
```

Rollback needs no database change, because migrations only add to the
schema. If a release changed `TAXVERITY_SERVING_CORPUS_VERSION` or
`TAXVERITY_SERVING_EMBEDDING_SET_ID`, restore the previous values in the
function's environment too. If you skip this, the serving guard refuses to
start.

## 4. Frontend

Vercel deploys from the `frontend/` directory. It needs one environment
variable, `NEXT_PUBLIC_API_BASE_URL`, set to the Lambda function URL. Next.js
builds the value into the client bundle, so changing it requires a redeploy.

The API's CORS allow-list must name the Vercel origin. Two places hold it:

- `TAXVERITY_APP_BASE_URL` on the Lambda.
- The function URL's CORS configuration, `cors_allow_origins` in
  `infra/main.tf`.

Never set either to `*`. The refresh cookie is sent with credentials.

## 5. Secrets and rotation

Secrets live in AWS Secrets Manager under `taxverity/*`. They reach the
function as environment variables, so a new value takes effect on the next
`update-function-configuration`.

| Secret | Rotation effect |
|---|---|
| `TAXVERITY_JWT_SECRET` | Every access token stops working within 15 minutes. Refresh tokens are stored hashed in the database, so they are unaffected and users are not logged out. |
| `TAXVERITY_JINA_API_KEY` | None if the new key is set before the old key is revoked. The vector store's probe fingerprint checks that the embeddings match, not which key produced them. |
| `TAXVERITY_GROQ_API_KEY` (and the optional `_2`) | None. A missing key fails over to Gemini. |
| `TAXVERITY_OPENAI_API_KEY` | None. Generation falls back to Groq gpt-oss-120b. |
| `TAXVERITY_GEMINI_API_KEY` | None. Gemini is only the fallback. |
| `TAXVERITY_DATABASE_URL` | Requires a redeploy. Use Supabase's session pooler on port 5432. |
| Gmail OAuth refresh token | Run `scripts/gmail_consent.py` again. Until then, mail sending fails, but login keeps working. |

Rotate any key that has appeared in a chat, log or screenshot.

## 6. When a vendor is down

Every vendor on the request path has a fallback built in. An outage lowers
answer quality, never correctness: the verifier and evidence gate run the
same way whichever path produced the answer.

| Vendor down | What happens |
|---|---|
| Jina embeddings | `FallbackRetriever` answers from BM25 plus the citation shortcut. Retrieval recall drops (lenient R@10 0.797 → 0.633). |
| Jina reranker | Results keep the fusion order. The timeout is 1.5 s with a single attempt. |
| OpenAI (generation) | Groq gpt-oss-120b generates. |
| Groq | Gemini runs every node Groq would have run. A long 429 spills over after 5 s instead of waiting. |
| Langfuse | Tracing failures are logged and dropped. They never fail a turn. |
| Gmail | Registration still succeeds. The verification email can be re-sent later. |
| Supabase | Nothing can be served. `/health` fails and the API returns 5xx. |

Look up a failing turn's per-node timings in the trace panel in the UI, or
in Langfuse by thread id.

## 7. Database housekeeping

The test suite creates and drops its own `taxverity_test_<hex>` databases.
A test run that is killed can leave one behind. To list them:

```sql
SELECT datname FROM pg_database WHERE datname LIKE 'taxverity_test_%';
```

Drop each one with `DROP DATABASE <name>;` after checking that no test run
is in progress. Point local test runs at the local `compose.yaml` database,
not at production.

## 8. Dependency updates

- CI's `audit` job runs `pip-audit` on the locked Python dependencies that
  ship, and `npm audit --omit=dev` on the frontend. A finding fails the build.
- Dependabot opens weekly update PRs for uv, npm and GitHub Actions, and
  monthly ones for the Docker base image.
- To bump one Python package:

  ```bash
  uv lock --upgrade-package <name>
  uv run pytest
  ```

## 9. Measure answer quality

`scripts/measure_generation.py` scores answers over
`evals/datasets/answer_gold_v1.jsonl`, which has 81 items:

- 49 answerable questions, each with gold citations and key points;
- 16 out-of-scope negatives;
- 8 calculator cases with hand-worked tax figures;
- 8 safety cases.

Each step stores its output, so a stopped run can be resumed.

1. Start the local database with `docker compose up -d --wait`.
2. Run the items through the graph:

   ```bash
   uv run python scripts/measure_generation.py run --label v1
   ```

   The script refuses to run against a non-local database. Expect about
   5 LLM calls per item. Add `--kind answerable` or `--only g001 g002` to
   run a subset.
3. Label the answers with the LLM judge:

   ```bash
   uv run python scripts/measure_generation.py judge --label v1
   ```

   The judge defaults to Gemini, a different vendor from the gpt-5-mini
   generator. Its replies are cached, so judging the same run again is free.
4. Check the judge against a human:

   ```bash
   uv run python scripts/measure_generation.py export-labels --label v1
   ```

   Label the 30 sampled claims in the CSV by hand without looking at the
   judge's labels. Then compute Cohen's kappa:

   ```bash
   uv run python scripts/measure_generation.py agreement --label v1
   ```

5. Write the report:

   ```bash
   uv run python scripts/measure_generation.py report --label v1
   ```

   It goes to `reports/generation_eval.md`.

**Floors**, fixed before the first run:

- served unsupported-claim rate at most 2%;
- abstention recall at least 0.90;
- calculator exact match 100%.

**Judged numbers** (supported rate, unsupported rate, key-point recall) can
be quoted only once kappa is at least 0.60.

**The ablation:** the "first pass (ungated)" row scores the model's lines
as the model wrote them, before verification. Comparing it with the
"served (gated)" row shows what the verifier prevents.
