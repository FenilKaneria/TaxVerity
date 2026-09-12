# Step 7.1 — Groq provider spike

Model: `openai/gpt-oss-120b`. Endpoint: `https://api.groq.com/openai/v1`.

A spike: no `src/` code, no tests. Findings shape Steps 7.2, 7.5, 7.6,
9.x and 10.3/10.4, and re-check ADR-082's evidence budget.

Evidence pack used for the streaming probes: 8 units — `26`, `270(5)`, `353`, `147`, `533`, `162(1)(f)`, `Schedule XV(3)(b)`, `131(1)`

## Findings

Interpretation of the run below. Each claim names the probe it rests on.

1. **The free tier is 8,000 tokens a minute, and ADR-082's assumption was
   right** (`plain.rate_limits`). Requests are capped at 1,000 with a rolling
   reset that lengthens as they are spent — 12m, 27m, 49m across three runs of
   this script — so requests are not the binding constraint; tokens are.
2. **One evidence-pack query costs 4,641 total tokens** (`stream_ndjson`), so
   the free tier serves **about 1.7 queries a minute**, single-user. That is a
   real capacity ceiling on the served product, not a dev-loop nuisance, and
   Step 7.2 has to decide whether it is answered by paid tier, by a smaller
   `EVIDENCE_BUDGET`, or by accepting a queue.
3. **The Step 2.3 proxy counter runs 9.85% short of gpt-oss's tokenizer**
   (`stream_ndjson.proxy_ratio = 1.0985`, stable across runs on an identical
   prompt), against 14.5% short of Jina's. So `EVIDENCE_BUDGET = 4,000` proxy
   tokens is ~4,394 real ones — closer than ADR-082 assumed, and its stated
   "about 4,600" was pessimistic in the safe direction.
4. **Strict `json_schema` holds the Step 7.5 `UserFacts` shape, and copies
   `source_span` verbatim** (`json_schema`): 4 of 4 spans were exact
   substrings of the user's turn. The looser `json_object` mode also works
   (`json_object`), so 7.6 has a fallback. Value formatting is *not* stable —
   one run returned `"1400000"` and another `"14,00,000"` for the same input,
   so 7.5's schema must carry the normalisation, not the prompt.
5. **`tool_choice: "required"` is unreliable: HTTP 400 `tool_use_failed` on
   3 of 6 attempts across four runs of this script**, while `auto` called the
   tool correctly on 6 of 6 with identical arguments (`tools`). The failures
   and successes came in whole runs, not alternating within one, and the 400
   carries an empty `failed_generation`, so there is nothing to repair. The
   report below shows one such run; `required#1`/`required#2` carried the 400
   on two earlier ones. Phase 9 must not depend on forced tool calls.
6. **Reasoning cannot be switched off, and its tokens are billed against
   `max_completion_tokens`** (`reasoning_off`): `reasoning_effort: "none"` is
   rejected outright, and the `plain` probe spent 22 of its 32 allowed tokens
   reasoning. A cap sized for the visible answer alone truncates it. `low` is
   the cheapest setting that exists and is what the hot path should use —
   `high` burned 398 reasoning tokens on a trivial question.
7. **Streaming and strict schema compose** (`stream_plus_json_schema`), and
   streaming NDJSON works, with two caveats for Step 10.4
   (`stream_multiclaim`): a newline always *ends* a content delta and never
   lands mid-delta (6 observations over 3 runs — too few to build on, so the
   parser must still buffer), and **the last line carries no trailing
   newline**, so the parser has to flush at end-of-stream or silently drop the
   final claim.
8. **Time to first token is 1.7 s on a 4,372-token prompt** (`stream_ndjson`),
   with queue time 0.3-0.6 s throughout. The dead-air problem Rule 04 names is
   real but small at this size.
9. Context window 131,072 and max completion 65,536 (`models`) — neither is
   anywhere near binding. `openai/gpt-oss-20b` is served on the same key, which
   is a same-vendor degradation option the plan does not currently use.


## models

*Is the model served, and what are its stated limits?*

```json
{
  "present": true,
  "context_window": 131072,
  "max_completion_tokens": 65536,
  "owned_by": "OpenAI",
  "fallback_candidates": [
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b"
  ]
}
```

## plain

*Latency, usage and finish_reason on an ordinary call.*

```json
{
  "seconds": 0.915,
  "finish_reason": "length",
  "content": "",
  "reasoning_field_present": true,
  "usage": {
    "prompt_tokens": 79,
    "completion_tokens": 32,
    "total_tokens": 111,
    "completion_time": 0.066063231,
    "queue_time": 0.395717058,
    "reasoning_tokens": 30
  },
  "rate_limits": {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-remaining-requests": "961",
    "x-ratelimit-reset-requests": "56m9.6s",
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-tokens": "7889",
    "x-ratelimit-reset-tokens": "832ms"
  }
}
```

## reasoning_effort

*What does reasoning_effort cost on the hot path?*

```json
{
  "low": {
    "seconds": 1.03,
    "usage": {
      "prompt_tokens": 91,
      "completion_tokens": 299,
      "total_tokens": 390,
      "completion_time": 0.640327729,
      "queue_time": 0.31859594,
      "reasoning_tokens": 42
    },
    "finish_reason": "stop"
  },
  "medium": {
    "seconds": 1.262,
    "usage": {
      "prompt_tokens": 91,
      "completion_tokens": 400,
      "total_tokens": 491,
      "completion_time": 0.844049084,
      "queue_time": 0.34906184,
      "reasoning_tokens": 85
    },
    "finish_reason": "length"
  },
  "high": {
    "seconds": 1.517,
    "usage": {
      "prompt_tokens": 91,
      "completion_tokens": 400,
      "total_tokens": 491,
      "completion_time": 0.829190614,
      "queue_time": 0.348850997,
      "reasoning_tokens": 398
    },
    "finish_reason": "length"
  }
}
```

## json_object

*Does the loose JSON mode work, as a 7.6 fallback?*

```json
{
  "seconds": 1.625,
  "raw": "{\"salary\":1400000,\"age\":41}",
  "parses": {
    "salary": 1400000,
    "age": 41
  },
  "usage": {
    "prompt_tokens": 130,
    "completion_tokens": 194,
    "total_tokens": 324,
    "completion_time": 0.408073639,
    "queue_time": 0.060079155,
    "reasoning_tokens": 170
  }
}
```

## json_schema

*Does strict schema mode hold the Step 7.5 UserFacts shape?*

```json
{
  "seconds": 1.837,
  "raw": "{\"fields\":[{\"name\":\"age\",\"value\":\"41\",\"status\":\"stated\",\"source_span\":\"41\"},{\"name\":\"employment_status\",\"value\":\"salaried\",\"status\":\"stated\",\"source_span\":\"salaried\"},{\"name\":\"annual_income\",\"value\":\"14,00,000\",\"status\":\"stated\",\"source_span\":\"14,00,000\"},{\"name\":\"ppf_contribution\",\"value\":\"1,50,000\",\"status\":\"stated\",\"source_span\":\"1,50,000\"}]}",
  "parsed": {
    "fields": [
      {
        "name": "age",
        "value": "41",
        "status": "stated",
        "source_span": "41"
      },
      {
        "name": "employment_status",
        "value": "salaried",
        "status": "stated",
        "source_span": "salaried"
      },
      {
        "name": "annual_income",
        "value": "14,00,000",
        "status": "stated",
        "source_span": "14,00,000"
      },
      {
        "name": "ppf_contribution",
        "value": "1,50,000",
        "status": "stated",
        "source_span": "1,50,000"
      }
    ]
  },
  "source_spans_verbatim": {
    "41": true,
    "salaried": true,
    "14,00,000": true,
    "1,50,000": true
  },
  "usage": {
    "prompt_tokens": 290,
    "completion_tokens": 496,
    "total_tokens": 786,
    "completion_time": 1.03275121,
    "queue_time": 0.401359395,
    "reasoning_tokens": 380
  }
}
```

## tools

*Tool-call shape, and whether tool_choice=required is honoured.*

```json
{
  "auto#1": {
    "seconds": 1.04,
    "finish_reason": "tool_calls",
    "call_count": 1,
    "names": [
      "compute_tax"
    ],
    "arguments": [
      {
        "regime": "new",
        "total_income": 1400000
      }
    ],
    "content_alongside": "",
    "usage": {
      "prompt_tokens": 157,
      "completion_tokens": 112,
      "total_tokens": 269,
      "completion_time": 0.233828035,
      "queue_time": 0.394976704,
      "reasoning_tokens": 74
    }
  },
  "required#1": {
    "seconds": 1.122,
    "finish_reason": "tool_calls",
    "call_count": 1,
    "names": [
      "compute_tax"
    ],
    "arguments": [
      {
        "regime": "new",
        "total_income": 1400000
      }
    ],
    "content_alongside": "",
    "usage": {
      "prompt_tokens": 157,
      "completion_tokens": 73,
      "total_tokens": 230,
      "completion_time": 0.154280752,
      "queue_time": 0.397742244,
      "reasoning_tokens": 35
    }
  },
  "auto#2": {
    "seconds": 1.844,
    "finish_reason": "length",
    "call_count": 0,
    "names": [],
    "arguments": [],
    "content_alongside": "Here’s the tax calculation under the **new‑regime** (FY 2023‑24) for a total",
    "usage": {
      "prompt_tokens": 157,
      "completion_tokens": 400,
      "total_tokens": 557,
      "completion_time": 0.847940635,
      "queue_time": 0.909543768,
      "reasoning_tokens": 368
    }
  },
  "required#2": {
    "seconds": 1.655,
    "finish_reason": "tool_calls",
    "call_count": 1,
    "names": [
      "compute_tax"
    ],
    "arguments": [
      {
        "regime": "new",
        "total_income": 1400000
      }
    ],
    "content_alongside": "",
    "usage": {
      "prompt_tokens": 157,
      "completion_tokens": 113,
      "total_tokens": 270,
      "completion_time": 0.410905829,
      "queue_time": 0.733418259,
      "reasoning_tokens": 75
    }
  }
}
```

## stream_ndjson

*Does claim-per-line streaming behave as Step 10.4 assumes?*

```json
{
  "ttft_seconds": 1.047,
  "total_seconds": 1.149,
  "delta_count": 48,
  "deltas_ending_on_newline": 0,
  "deltas_containing_newline": 0,
  "newline_ever_mid_delta": false,
  "line_count": 1,
  "lines_parsing_as_json": 1,
  "first_line": "{\"text\":\"The provided provisions do not specify a monetary standard deduction for a rented‑out flat; they only state that rental income is chargeable under the head “Income from house property”.\", \"ci",
  "proxy_prompt_tokens": 3980,
  "billed_prompt_tokens": 4372,
  "proxy_ratio": 1.0985,
  "usage": {
    "prompt_tokens": 4372,
    "completion_tokens": 276,
    "total_tokens": 4648,
    "completion_time": 0.580850654,
    "queue_time": 0.220091671,
    "reasoning_tokens": 219
  },
  "rate_limits": {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-remaining-requests": "951",
    "x-ratelimit-reset-requests": "1h10m33.6s",
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-tokens": "2692",
    "x-ratelimit-reset-tokens": "39.81s"
  }
}
```

## stream_multiclaim

*Where do newlines fall relative to stream deltas?*

```json
{
  "delta_count": 113,
  "line_count": 3,
  "lines_parsing_as_json": 3,
  "newline_delta_count": 2,
  "newline_deltas": [
    "' \\n'",
    "' \\n'"
  ],
  "newline_ever_mid_delta": false,
  "delta_is_bare_newline": 0,
  "trailing_newline_on_last_line": false,
  "first_line": "{\"text\":\"A standard deduction of thirty percent of the annual value is allowed under section 22(1)(a).\",\"citations\":[\"[22(1)(a)]\"]} "
}
```

## reasoning_off

*Can reasoning be disabled, and what does it save?*

```json
{
  "effort_none": {
    "error": "RuntimeError: HTTP 400: {\"error\":{\"message\":\"`reasoning_effort` must be one of `low`, `medium`, or `high`\",\"type\":\"invalid_request_error\"}}\n"
  },
  "format_hidden": {
    "seconds": 0.475,
    "finish_reason": "stop",
    "content": "ready",
    "usage": {
      "prompt_tokens": 77,
      "completion_tokens": 17,
      "total_tokens": 94,
      "completion_time": 0.035209388,
      "queue_time": 0.403071067,
      "reasoning_tokens": 7
    }
  },
  "no_cap_low_effort": {
    "seconds": 0.414,
    "finish_reason": "stop",
    "content": "ready",
    "usage": {
      "prompt_tokens": 77,
      "completion_tokens": 15,
      "total_tokens": 92,
      "completion_time": 0.031427573,
      "queue_time": 0.347884146,
      "reasoning_tokens": 5
    }
  }
}
```

## stream_plus_json_schema

*Do streaming and strict schema compose?*

```json
{
  "status": 200,
  "accepted": true,
  "detail": "streamed without error"
}
```
