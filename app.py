"""
Shin Proxy — FastAPI application factory.

Creates the app, registers routers, middleware, exception handlers,
and lifespan hooks (httpx client init/teardown).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError as FastAPIValidationError
from fastapi.responses import JSONResponse

from shin.analytics import prom_metrics
from shin.config import settings
from shin.handlers import ProxyError, StreamAbortError

log = structlog.get_logger()


# ── Shared httpx client (created in lifespan) ──────────────────────────────
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return the shared httpx AsyncClient. Must be called after app startup."""
    assert _http_client is not None, "httpx client not initialised — app not started?"
    return _http_client


def init_observability() -> None:
    """Initialize optional OpenTelemetry instrumentation."""
    if not settings.otel_enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)

        endpoint = settings.otel_exporter_otlp_endpoint.strip()
        if endpoint:
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        HTTPXClientInstrumentor().instrument()
        # FastAPI app instrumentation is attached in create_app() below.
        log.info("otel_initialized", endpoint=endpoint or "none")
    except Exception as exc:
        log.warning("otel_init_failed", reason=str(exc)[:200])


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _http_client
    read_timeout = (
        None
        if settings.upstream_read_timeout_seconds <= 0
        else settings.upstream_read_timeout_seconds
    )

    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=read_timeout, write=10.0, pool=10.0),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        ),
        follow_redirects=True,
    )
    log.info("httpx_client_started")
    yield
    await _http_client.aclose()
    _http_client = None
    log.info("httpx_client_closed")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _is_anthropic(request: Request) -> bool:
    return request.url.path.startswith("/v1/messages")


def _api_style_from_path(path: str) -> str:
    if path.startswith("/v1/messages"):
        return "anthropic"
    if path.startswith("/v1/chat/completions"):
        return "openai"
    return "internal"


# ── App factory ─────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    init_observability()

    app = FastAPI(
        title="Shin Proxy",
        version="1.0.0",
        description="Clean-architecture Cursor reverse proxy",
        lifespan=_lifespan,
    )

    # ── Exception handlers ──────────────────────────────────────────────

    @app.exception_handler(ProxyError)
    async def proxy_error_handler(request: Request, exc: ProxyError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        if not isinstance(exc, StreamAbortError):
            log.error(
                "proxy_error",
                type=exc.error_type,
                message=exc.message,
                path=str(request.url.path),
                request_id=request_id,
            )
        body = exc.to_anthropic() if _is_anthropic(request) else exc.to_openai()
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(FastAPIValidationError)
    async def validation_error_handler(
        request: Request, exc: FastAPIValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = first.get("loc", ["unknown"])
        msg = f"Invalid request: {loc[-1]} — {first.get('msg', 'validation error')}"
        log.warning("validation_error", errors=errors, path=str(request.url.path))
        body = {
            "error": {
                "message": msg,
                "type": "invalid_request_error",
                "code": "400",
            }
        }
        return JSONResponse(status_code=400, content=body)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        log.exception(
            "unhandled_error",
            path=str(request.url.path),
            request_id=request_id,
        )
        body = {
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": "500",
            }
        }
        return JSONResponse(status_code=500, content=body)

    # ── Middleware: inject request_id ────────────────────────────────────

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)

        path = request.url.path
        api_style = _api_style_from_path(path)
        stream = response.headers.get("content-type", "").startswith("text/event-stream")

        prom_metrics.observe_http_request(
            method=request.method,
            path=path,
            api_style=api_style,
            stream=stream,
            status_code=response.status_code,
            duration_seconds=(time.perf_counter() - started),
        )
        return response

    # ── Register routers ────────────────────────────────────────────────
    from shin.routers.openai import router as openai_router
    from shin.routers.anthropic import router as anthropic_router

    app.include_router(openai_router)
    app.include_router(anthropic_router)

    if settings.otel_enabled:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except Exception as exc:
            log.warning("otel_fastapi_instrumentation_failed", reason=str(exc)[:200])

    return app
