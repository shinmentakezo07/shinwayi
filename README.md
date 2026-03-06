# Shin Proxy

Shin Proxy is a FastAPI-based Cursor gateway that exposes OpenAI-compatible and Anthropic-compatible APIs while forwarding requests to Cursor upstream.

It is designed for local and self-hosted use cases where you want:

- OpenAI-style `/v1/chat/completions`
- Anthropic-style `/v1/messages`
- streaming support
- tool-call compatibility
- retry and failover behavior
- caching
- rate limiting
- Prometheus metrics
- internal usage stats

---

## Directory

This README documents the Python proxy implementation in:

- `/teamspace/studios/this_studio/shin/shin`

Main files:

- `app.py` — FastAPI app factory and middleware
- `run.py` — local entry point
- `config.py` — environment configuration
- `routers/openai.py` — OpenAI-compatible endpoint
- `routers/anthropic.py` — Anthropic-compatible endpoint
- `pipeline.py` — request/response transformation and streaming pipeline
- `cursor/client.py` — upstream Cursor HTTP/SSE client
- `analytics.py` — internal stats and Prometheus-style metrics

---

## Features

### API compatibility

- `POST /v1/chat/completions`
- `POST /v1/messages`
- `GET /v1/models`
- `GET /health`
- `GET /metrics`
- `GET /v1/internal/stats`

### Runtime behavior

- streaming SSE responses
- OpenAI tool-call output shaping
- Anthropic tool-use output shaping
- optional reasoning visibility
- upstream retry/backoff
- adaptive concurrency controls
- cache support
- Redis-backed cache support
- request metrics and stream-stage latency metrics

---

## Requirements

You need:

- Python 3.12+ if running directly
- or Docker + Docker Compose if running with containers
- valid Cursor authentication

Install dependencies from:

- `requirements.txt`

Current Python dependencies are declared in `requirements.txt` and include FastAPI, Uvicorn, httpx, redis, OpenTelemetry packages, and token counting support.

---

## Configuration

Configuration is loaded through `config.py` using environment variables and `.env`.

The proxy expects `.env` in the working directory. A sample file exists at:

- `.env.example`

### Core variables

#### Server

- `HOST` — bind host, default `0.0.0.0`
- `PORT` — bind port, default `4000`

#### Client auth to this proxy

- `LITELLM_MASTER_KEY` — bearer token required by clients calling this proxy

#### Cursor upstream

- `CURSOR_BASE_URL` — default `https://cursor.com`
- `CURSOR_MODEL` — default `anthropic/claude-sonnet-4.6`
- `CURSOR_COOKIE` — single Cursor session cookie
- `CURSOR_COOKIES` — multiple cookies for rotation
- `CURSOR_AUTH_HEADER` — optional direct auth header
- `CURSOR_CONTEXT_FILE_PATH` — context file path sent upstream, default `/docs/`

#### Retry and timeout

- `GATEWAY_UPSTREAM_READ_TIMEOUT_SECONDS`
- `GATEWAY_RETRY_ATTEMPTS`
- `GATEWAY_RETRY_BACKOFF_SECONDS`

#### Adaptive concurrency

- `GATEWAY_ADAPTIVE_CONCURRENCY_ENABLED`
- `GATEWAY_ADAPTIVE_CONCURRENCY_INITIAL`
- `GATEWAY_ADAPTIVE_CONCURRENCY_MIN`
- `GATEWAY_ADAPTIVE_CONCURRENCY_MAX`
- `GATEWAY_ADAPTIVE_CONCURRENCY_SUCCESS_WINDOW`
- `GATEWAY_ADAPTIVE_CONCURRENCY_RATE_LIMIT_COOLDOWN_SECONDS`

#### Cache

- `GATEWAY_CACHE_ENABLED`
- `GATEWAY_CACHE_TTL_SECONDS`
- `GATEWAY_CACHE_MAX_ENTRIES`
- `GATEWAY_CACHE_BACKEND`
- `REDIS_URL`
- `REDIS_CACHE_PREFIX`

#### Observability

- `GATEWAY_METRICS_ENABLED`
- `GATEWAY_OTEL_ENABLED`
- `OTEL_SERVICE_NAME`
- `OTEL_EXPORTER_OTLP_ENDPOINT`

#### Prompt / identity

- `GATEWAY_SYSTEM_PROMPT_FILE`
- `GATEWAY_SYSTEM_PROMPT`

#### Pricing used by internal stats

- `GATEWAY_PRICE_ANTHROPIC_PER_1K`
- `GATEWAY_PRICE_OPENAI_PER_1K`

---

## Example `.env`

```env
HOST=0.0.0.0
PORT=4000
LITELLM_MASTER_KEY=sk-local-dev

CURSOR_BASE_URL=https://cursor.com
CURSOR_MODEL=anthropic/claude-sonnet-4.6
CURSOR_COOKIE=WorkosCursorSessionToken=your_real_cookie_here
CURSOR_AUTH_HEADER=

GATEWAY_UPSTREAM_READ_TIMEOUT_SECONDS=300
GATEWAY_RETRY_ATTEMPTS=2
GATEWAY_RETRY_BACKOFF_SECONDS=0.6

GATEWAY_CACHE_ENABLED=true
GATEWAY_CACHE_BACKEND=redis
REDIS_URL=redis://redis:6379/0

GATEWAY_METRICS_ENABLED=true
GATEWAY_OTEL_ENABLED=false

GATEWAY_SYSTEM_PROMPT_FILE=/app/sysprompt.txt
```

If you use multiple Cursor cookies, put them in `CURSOR_COOKIES` instead of `CURSOR_COOKIE`.

---

## Run locally with Python

From the repository root:

```bash
pip install -r shin/requirements.txt
python -m shin.run
```

Default local bind:

- `http://127.0.0.1:4000`

Entry point:

- `run.py`

Uvicorn is launched through `shin.run:main`.

---

## Run with Docker Compose

The proxy has a dedicated compose file at:

- `docker-compose.yml`

From the repository root:

```bash
docker compose -f shin/docker-compose.yml up -d --build
```

This starts:

- `shin-proxy`
- `redis`
- `otel-collector`
- `prometheus`
- `grafana`

Default ports:

- Proxy: `4000`
- Redis: `6379`
- OTEL HTTP: `4318`
- OTEL gRPC: `4317`
- Prometheus: `19090`
- Grafana: `13000`

Restart only the proxy:

```bash
docker compose -f shin/docker-compose.yml restart shin-proxy
```

Rebuild after code changes:

```bash
docker compose -f shin/docker-compose.yml up -d --build shin-proxy
```

---

## Health check

```bash
curl -s -H "Authorization: Bearer sk-local-dev" \
  http://127.0.0.1:4000/health
```

Expected result: JSON health response from the proxy.

---

## OpenAI-compatible usage

Endpoint:

- `POST /v1/chat/completions`

### Non-streaming example

```bash
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4.6",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ]
  }'
```

### Streaming example

```bash
curl -N -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4.6",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Say hello briefly."}
    ]
  }'
```

### Streaming with tools available

```bash
curl -N -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4.6",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Reply with plain text only."}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "dummy_tool",
        "description": "Example tool",
        "parameters": {
          "type": "object",
          "properties": {
            "x": {"type": "string"}
          },
          "required": ["x"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

### Forced tool call example

```bash
curl -N -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4.6",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Call dummy_tool with x set to hello."}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "dummy_tool",
        "description": "Example tool",
        "parameters": {
          "type": "object",
          "properties": {
            "x": {"type": "string"}
          },
          "required": ["x"]
        }
      }
    }],
    "tool_choice": {
      "type": "function",
      "function": {"name": "dummy_tool"}
    }
  }'
```

---

## Anthropic-compatible usage

Endpoint:

- `POST /v1/messages`

### Non-streaming example

```bash
curl -s http://127.0.0.1:4000/v1/messages \
  -H "Authorization: Bearer sk-local-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4.6",
    "max_tokens": 256,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Say hello briefly."}
        ]
      }
    ]
  }'
```

### Streaming example

```bash
curl -N -s http://127.0.0.1:4000/v1/messages \
  -H "Authorization: Bearer sk-local-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4.6",
    "stream": true,
    "max_tokens": 256,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Think briefly and answer."}
        ]
      }
    ]
  }'
```

### Anthropic with thinking enabled

```bash
curl -N -s http://127.0.0.1:4000/v1/messages \
  -H "Authorization: Bearer sk-local-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4.6",
    "stream": true,
    "max_tokens": 256,
    "thinking": {
      "type": "enabled",
      "budget_tokens": 128
    },
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Solve this step by step, then answer briefly."}
        ]
      }
    ]
  }'
```

---

## Models endpoint

```bash
curl -s -H "Authorization: Bearer sk-local-dev" \
  http://127.0.0.1:4000/v1/models
```

---

## Metrics and stats

### Prometheus metrics

```bash
curl -s http://127.0.0.1:4000/metrics
```

The proxy exposes request and stream metrics, including stage latency metrics used for streaming analysis.

Examples you may see:

- `shin_http_requests_total`
- `shin_http_request_duration_seconds`
- `shin_upstream_attempts_total`
- `shin_stream_ttft_seconds`
- `shin_stream_stage_latency_seconds`

Useful stream stage labels include:

- `request_to_stream_open`
- `stream_open_to_first_upstream_sse_line`
- `first_upstream_sse_line_to_first_upstream_delta`
- `stream_open_to_first_upstream_delta`
- `first_upstream_delta_to_first_visible`
- `request_to_first_visible`
- `retry_sleep_before_first_delta`

### Internal stats

```bash
curl -s -H "Authorization: Bearer sk-local-dev" \
  http://127.0.0.1:4000/v1/internal/stats | jq
```

This endpoint is useful for:

- request counts
- usage snapshots
- cost estimation views

---

## Authentication

Client requests to Shin Proxy must include:

```http
Authorization: Bearer <LITELLM_MASTER_KEY>
```

Example:

```bash
-H "Authorization: Bearer sk-local-dev"
```

If this key does not match the configured proxy key, requests will be rejected.

---

## Tool calling behavior

The proxy supports tool compatibility for both OpenAI and Anthropic-style clients.

High-level behavior:

- plain text can stream immediately even when tools are available
- real tool calls are converted to protocol-specific output
- raw internal tool payloads are suppressed from visible text
- reasoning and tool output are kept separate when supported

OpenAI-style clients receive `tool_calls` deltas.

Anthropic-style clients receive `tool_use` blocks.

---

## How streaming works

The streaming pipeline is implemented primarily in:

- `pipeline.py`
- `cursor/client.py`
- `tools/parse.py`
- `converters/from_cursor.py`

The proxy tracks important stream timings such as:

- request start to stream open
- first upstream SSE line
- first upstream delta
- first visible content emitted

This helps distinguish:

- upstream think time
- retry delays before first token
- local buffering or suppression effects

---

## Troubleshooting

### 1. Server starts but requests fail with auth error

Check:

- `LITELLM_MASTER_KEY`
- client `Authorization` header

### 2. Upstream requests fail

Check:

- `CURSOR_COOKIE`
- `CURSOR_COOKIES`
- `CURSOR_AUTH_HEADER`
- upstream account/session validity

### 3. Streaming connects but output seems delayed

Check:

- `/metrics`
- `shin_stream_ttft_seconds`
- `shin_stream_stage_latency_seconds`

This tells you whether the delay is:

- before first upstream delta
- between upstream delta and first visible output

### 4. No visible text with tools enabled

Check logs for warnings like:

- `suppressed_raw_tool_payload_openai_stream`
- `suppressed_raw_tool_payload_anthropic_stream`

These indicate the proxy detected raw tool payload output and intentionally hid it from visible text.

### 5. Docker restart does not pick up code changes

Use rebuild, not only restart:

```bash
docker compose -f shin/docker-compose.yml up -d --build shin-proxy
```

### 6. Redis issues

If cache backend is set to Redis, verify Redis is running and `REDIS_URL` is correct.

---

## Development workflow

### Compile check

```bash
python -m compileall /teamspace/studios/this_studio/shin/shin
```

### Restart container

```bash
docker compose -f /teamspace/studios/this_studio/shin/shin/docker-compose.yml restart shin-proxy
```

### Rebuild container

```bash
docker compose -f /teamspace/studios/this_studio/shin/shin/docker-compose.yml up -d --build shin-proxy
```

### View logs

```bash
docker compose -f /teamspace/studios/this_studio/shin/shin/docker-compose.yml logs -f shin-proxy
```

---

## Notes

- The Docker image copies `sysprompt.txt` into `/app/sysprompt.txt`.
- The default prompt file path is controlled by `GATEWAY_SYSTEM_PROMPT_FILE`.
- The app factory is `shin.app:create_app`.
- The local CLI entry point is `python -m shin.run`.

---

## Quick test sequence

If you want a short smoke test after startup:

```bash
curl -s -H "Authorization: Bearer sk-local-dev" http://127.0.0.1:4000/health
curl -s -H "Authorization: Bearer sk-local-dev" http://127.0.0.1:4000/v1/models
curl -s http://127.0.0.1:4000/metrics > /tmp/shin-metrics.txt
```

Then run one streaming request:

```bash
curl -N -s http://127.0.0.1:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4.6",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Say hello briefly."}
    ]
  }'
```

That is the fastest way to confirm the proxy is alive and streaming correctly.
