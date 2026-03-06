"""
Shin Proxy — analytics + Prometheus-style metrics.

Includes:
- Existing per-api-key usage accounting snapshot
- In-process Prometheus metrics registry + exposition
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from shin.config import settings


@dataclass
class RequestLog:
    """A single request record."""

    api_key: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    fallback: bool = False
    cache_hit: bool = False
    ts: float = field(default_factory=time.time)


class AnalyticsStore:
    """Per-key usage tracker with snapshot support."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: dict[str, dict] = {}

    def _ensure(self, api_key: str) -> dict:
        if api_key not in self._by_key:
            self._by_key[api_key] = {
                "requests": 0,
                "cache_hits": 0,
                "fallbacks": 0,
                "estimated_input_tokens": 0,
                "estimated_output_tokens": 0,
                "estimated_cost_usd": 0.0,
                "latency_ms_total": 0.0,
                "last_request_ts": 0,
                "providers": {},
            }
        return self._by_key[api_key]

    def record(self, log: RequestLog) -> None:
        with self._lock:
            rec = self._ensure(log.api_key or "anonymous")
            rec["requests"] += 1
            if log.cache_hit:
                rec["cache_hits"] += 1
            if log.fallback:
                rec["fallbacks"] += 1
            rec["estimated_input_tokens"] += log.input_tokens
            rec["estimated_output_tokens"] += log.output_tokens
            rec["estimated_cost_usd"] += log.cost_usd
            rec["latency_ms_total"] += log.latency_ms
            rec["last_request_ts"] = int(time.time())
            rec["providers"][log.provider] = (
                rec["providers"].get(log.provider, 0) + 1
            )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "ts": int(time.time()),
                "keys": json.loads(json.dumps(self._by_key)),
            }


@dataclass
class _HistogramState:
    """Internal histogram bins for one metric+labelset."""

    buckets: tuple[float, ...]
    bins: list[int]
    count: int = 0
    total: float = 0.0


class PrometheusStore:
    """Small in-process Prometheus-compatible metrics store."""

    REQUEST_DURATION_BUCKETS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
    UPSTREAM_DURATION_BUCKETS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
    STREAM_TTFT_BUCKETS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
    STREAM_STAGE_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
        self._histograms: dict[str, dict[tuple[tuple[str, str], ...], _HistogramState]] = {}
        self._meta: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _labels_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        if not labels:
            return ()
        return tuple(sorted((str(k), str(v)) for k, v in labels.items()))

    @staticmethod
    def _escape_label(v: str) -> str:
        return (
            v.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )

    @classmethod
    def _format_labels(cls, labels_key: tuple[tuple[str, str], ...]) -> str:
        if not labels_key:
            return ""
        bits = [f'{k}="{cls._escape_label(v)}"' for k, v in labels_key]
        return "{" + ",".join(bits) + "}"

    def _set_meta(self, name: str, metric_type: str, help_text: str) -> None:
        if name not in self._meta:
            self._meta[name] = (metric_type, help_text)

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        *,
        labels: dict[str, str] | None = None,
        help_text: str,
    ) -> None:
        if not settings.metrics_enabled:
            return
        key = self._labels_key(labels)
        with self._lock:
            self._set_meta(name, "counter", help_text)
            series = self._counters.setdefault(name, {})
            series[key] = series.get(key, 0.0) + float(value)

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, str] | None = None,
        help_text: str,
    ) -> None:
        if not settings.metrics_enabled:
            return
        key = self._labels_key(labels)
        with self._lock:
            self._set_meta(name, "gauge", help_text)
            series = self._gauges.setdefault(name, {})
            series[key] = float(value)

    def observe_histogram(
        self,
        name: str,
        value: float,
        *,
        buckets: tuple[float, ...],
        labels: dict[str, str] | None = None,
        help_text: str,
    ) -> None:
        if not settings.metrics_enabled:
            return
        key = self._labels_key(labels)
        with self._lock:
            self._set_meta(name, "histogram", help_text)
            by_label = self._histograms.setdefault(name, {})
            state = by_label.get(key)
            if state is None:
                state = _HistogramState(
                    buckets=tuple(sorted(buckets)),
                    bins=[0] * (len(buckets) + 1),
                )
                by_label[key] = state

            idx = len(state.buckets)
            for i, bound in enumerate(state.buckets):
                if value <= bound:
                    idx = i
                    break

            state.bins[idx] += 1
            state.count += 1
            state.total += float(value)

    def observe_http_request(
        self,
        *,
        method: str,
        path: str,
        api_style: str,
        stream: bool,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        labels = {
            "method": method,
            "path": path,
            "api_style": api_style,
            "stream": "true" if stream else "false",
            "status_code": str(status_code),
        }
        self.inc_counter(
            "shin_http_requests_total",
            labels=labels,
            help_text="Total HTTP requests handled by Shin.",
        )
        self.observe_histogram(
            "shin_http_request_duration_seconds",
            duration_seconds,
            buckets=self.REQUEST_DURATION_BUCKETS,
            labels=labels,
            help_text="HTTP request duration in seconds.",
        )

    def observe_upstream_attempt(
        self,
        *,
        provider: str,
        outcome: str,
        status_code: int | None,
        duration_seconds: float,
    ) -> None:
        labels = {
            "provider": provider,
            "outcome": outcome,
            "status_code": str(status_code) if status_code is not None else "none",
        }
        self.inc_counter(
            "shin_upstream_attempts_total",
            labels=labels,
            help_text="Total upstream Cursor attempts.",
        )
        self.observe_histogram(
            "shin_upstream_attempt_duration_seconds",
            duration_seconds,
            buckets=self.UPSTREAM_DURATION_BUCKETS,
            labels=labels,
            help_text="Upstream attempt duration in seconds.",
        )

    def observe_cache(self, *, api_style: str, backend: str, result: str) -> None:
        self.inc_counter(
            "shin_cache_operations_total",
            labels={"api_style": api_style, "backend": backend, "result": result},
            help_text="Cache operations by result (hit/miss/set/error).",
        )

    def observe_stream_event(self, *, api_style: str, event: str) -> None:
        self.inc_counter(
            "shin_stream_events_total",
            labels={"api_style": api_style, "event": event},
            help_text="Stream lifecycle events.",
        )

    def observe_stream_ttft(self, *, api_style: str, seconds: float) -> None:
        self.observe_histogram(
            "shin_stream_ttft_seconds",
            seconds,
            buckets=self.STREAM_TTFT_BUCKETS,
            labels={"api_style": api_style},
            help_text="Time-to-first-token for streaming responses.",
        )

    def observe_stream_stage_latency(
        self,
        *,
        api_style: str,
        stage: str,
        seconds: float,
    ) -> None:
        self.observe_histogram(
            "shin_stream_stage_latency_seconds",
            seconds,
            buckets=self.STREAM_STAGE_BUCKETS,
            labels={"api_style": api_style, "stage": stage},
            help_text="Latency between key streaming pipeline stages in seconds.",
        )

    def set_adaptive_concurrency(
        self,
        *,
        limit: int,
        in_flight: int,
        cooldown_remaining_seconds: float,
    ) -> None:
        self.set_gauge(
            "shin_adaptive_concurrency_limit",
            float(limit),
            help_text="Current adaptive concurrency limit.",
        )
        self.set_gauge(
            "shin_adaptive_concurrency_in_flight",
            float(in_flight),
            help_text="Current in-flight requests admitted by adaptive controller.",
        )
        self.set_gauge(
            "shin_adaptive_concurrency_cooldown_remaining_seconds",
            float(max(0.0, cooldown_remaining_seconds)),
            help_text="Remaining cooldown after last upstream rate-limit signal.",
        )

    def observe_circuit_breaker_transition(self, *, state: str) -> None:
        self.inc_counter(
            "shin_circuit_breaker_transitions_total",
            labels={"state": state},
            help_text="Circuit-breaker state transitions.",
        )

    def observe_circuit_breaker_fast_fail(self) -> None:
        self.inc_counter(
            "shin_circuit_breaker_fast_fail_total",
            help_text="Requests rejected because the circuit breaker was open.",
        )

    def observe_circuit_breaker_probe(self, *, outcome: str) -> None:
        self.inc_counter(
            "shin_circuit_breaker_half_open_probes_total",
            labels={"outcome": outcome},
            help_text="Half-open circuit-breaker probe outcomes.",
        )

    def set_circuit_breaker(
        self,
        *,
        state: str,
        open_remaining_seconds: float,
        half_open_probes_in_flight: int,
    ) -> None:
        state_value = {
            "closed": 0.0,
            "half_open": 1.0,
            "open": 2.0,
        }.get(state, -1.0)
        self.set_gauge(
            "shin_circuit_breaker_state",
            state_value,
            help_text="Circuit-breaker state encoded as closed=0, half_open=1, open=2.",
        )
        self.set_gauge(
            "shin_circuit_breaker_open_remaining_seconds",
            float(max(0.0, open_remaining_seconds)),
            help_text="Seconds remaining until the circuit breaker can probe again.",
        )
        self.set_gauge(
            "shin_circuit_breaker_half_open_probes_in_flight",
            float(max(0, half_open_probes_in_flight)),
            help_text="Current half-open probe attempts in flight.",
        )

    def render(self) -> str:
        """Render Prometheus exposition format text."""
        if not settings.metrics_enabled:
            return "# metrics disabled\n"

        lines: list[str] = []
        with self._lock:
            names = sorted(self._meta.keys())
            for name in names:
                metric_type, help_text = self._meta[name]
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {metric_type}")

                if metric_type == "counter":
                    series = self._counters.get(name, {})
                    for labels_key, value in sorted(series.items()):
                        lines.append(f"{name}{self._format_labels(labels_key)} {value}")
                    continue

                if metric_type == "gauge":
                    series = self._gauges.get(name, {})
                    for labels_key, value in sorted(series.items()):
                        lines.append(f"{name}{self._format_labels(labels_key)} {value}")
                    continue

                if metric_type == "histogram":
                    series = self._histograms.get(name, {})
                    for labels_key, state in sorted(series.items()):
                        cumulative = 0
                        for idx, upper in enumerate(state.buckets):
                            cumulative += state.bins[idx]
                            with_le = tuple(sorted((*labels_key, ("le", str(upper)))))
                            lines.append(
                                f"{name}_bucket{self._format_labels(with_le)} {cumulative}"
                            )

                        cumulative += state.bins[-1]
                        with_inf = tuple(sorted((*labels_key, ("le", "+Inf"))))
                        lines.append(
                            f"{name}_bucket{self._format_labels(with_inf)} {cumulative}"
                        )
                        lines.append(
                            f"{name}_sum{self._format_labels(labels_key)} {state.total}"
                        )
                        lines.append(
                            f"{name}_count{self._format_labels(labels_key)} {state.count}"
                        )

        return "\n".join(lines) + "\n"



def estimate_cost(
    provider: str, input_tokens: int, output_tokens: int
) -> float:
    """Estimate cost in USD for a request."""
    prices = {
        "anthropic": settings.price_anthropic_per_1k,
        "openai": settings.price_openai_per_1k,
    }
    unit = prices.get(provider, prices["anthropic"])
    return ((input_tokens + output_tokens) / 1000.0) * unit


# Module-level singletons
analytics = AnalyticsStore()
prom_metrics = PrometheusStore()
