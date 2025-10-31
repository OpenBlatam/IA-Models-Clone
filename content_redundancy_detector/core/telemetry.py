"""
OpenTelemetry setup with graceful fallbacks.

Configures OTLP exporter, FastAPI and httpx instrumentation when available.
Safe to import in environments without OpenTelemetry installed.
"""

from typing import Optional

import logging

logger = logging.getLogger(__name__)


def setup_tracing(
	enabled: bool = True,
	service_name: str = "content-redundancy-detector",
	otlp_endpoint: Optional[str] = None,
	sample_ratio: float = 0.1,
) -> None:
	"""Initialize OpenTelemetry tracing if libraries are available."""
	if not enabled:
		logger.debug("Tracing disabled by configuration")
		return
	try:
		from opentelemetry import trace
		from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # type: ignore
		from opentelemetry.sdk.trace import TracerProvider  # type: ignore
		from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
		from opentelemetry.sdk.trace.sampling import TraceIdRatioBased  # type: ignore
		# Exporter: prefer OTLP, fallback to console
		exporter = None
		if otlp_endpoint:
			try:
				from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
				exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
			except Exception as e:  # pragma: no cover
				logger.warning(f"OTLP exporter not available: {e}")
		if exporter is None:
			try:
				from opentelemetry.sdk.trace.export import ConsoleSpanExporter  # type: ignore
				exporter = ConsoleSpanExporter()
			except Exception:
				exporter = None
		provider = TracerProvider(
			resource=Resource.create({SERVICE_NAME: service_name}),
			sampler=TraceIdRatioBased(max(0.0, min(1.0, sample_ratio)))
		)
		if exporter is not None:
			processor = BatchSpanProcessor(exporter)
			provider.add_span_processor(processor)
		trace.set_tracer_provider(provider)
		logger.info("OpenTelemetry tracing initialized")
	except Exception as e:  # pragma: no cover
		logger.debug(f"Tracing not initialized (libraries missing?): {e}")


def instrument_fastapi(app) -> None:
	"""Instrument FastAPI app if instrumentation is available."""
	try:
		from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
		FastAPIInstrumentor.instrument_app(app)
		logger.info("FastAPI instrumentation enabled")
	except Exception:
		logger.debug("FastAPI instrumentation not available")


def instrument_httpx() -> None:
	"""Instrument httpx if instrumentation is available."""
	try:
		from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # type: ignore
		HTTPXClientInstrumentor().instrument()
		logger.info("httpx instrumentation enabled")
	except Exception:
		logger.debug("httpx instrumentation not available")


def set_request_context_attributes(span, request_id: Optional[str] = None) -> None:
	"""Attach useful attributes to current span if possible."""
	try:
		if span is None:
			return
		if request_id:
			span.set_attribute("request.id", request_id)
	except Exception:
		pass







