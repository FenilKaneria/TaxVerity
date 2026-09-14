# Step 17.8 — deploy smoke test + latency pass

Target: `https://5iiiww6f4ghk7lo6mrzkw7cdzm0sxlze.lambda-url.ap-south-1.on.aws`
Questions: 15
Contract failures: 1 (q003)

## Latency (warm, sequential over one thread)

- total turn: p50 23.5s, p95 120.4s, max 120.4s
- time to first claim/withheld: p50 36.5s, p95 90.5s

## Per-question

| query | claims | withheld | total_s | first_claim_s |
|---|---|---|---|---|
| q001 | 3 | 1 | 17.07 | 3.91 |
| q002 | 1 | 0 | 28.62 | 28.61 |
| q003 | 0 | 1 | 120.36 | 90.46 |
| q009 | 1 | 0 | 12.23 | 12.23 |
| q010 | 1 | 1 | 83.08 | 42.81 |
| q011 | 2 | 0 | 57.51 | 57.21 |
| q012 | 1 | 0 | 30.36 | 30.28 |
| q019 | 12 | 0 | 77.79 | 77.57 |
| q020 | 1 | 0 | 23.53 | 23.53 |
| q021 | 1 | 2 | 109.36 | 48.97 |
| q025 | 0 | 0 | 0.63 | None |
| q026 | 0 | 0 | 0.61 | None |
| q027 | 0 | 0 | 0.48 | None |
| q028 | 0 | 0 | 0.28 | None |
| q029 | 0 | 0 | 5.06 | None |
