"""Base class for OTLP-compatible tracers with context propagation support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lfx.log.logger import logger
from typing_extensions import override

from langflow.services.tracing.base import BaseTracer

if TYPE_CHECKING:
    from opentelemetry.context import Context


class OTLPTracerBase(BaseTracer):
    """Base class for tracers that support OTLP trace context propagation.

    This intermediate class provides default implementations of inject_context
    and extract_context for tracers that use OpenTelemetry's standard context
    propagation mechanisms.

    Subclasses that use OpenTelemetry should inherit from this class to gain
    automatic support for W3C Trace Context propagation.
    """

    @override
    def inject_context(self, carrier: dict[str, str], trace_id: str | None = None) -> None:
        """Inject trace context into a carrier (e.g., HTTP headers) for distributed tracing.

        Uses OpenTelemetry's standard propagation to inject W3C Trace Context headers
        (traceparent, tracestate) into the carrier.

        Args:
            carrier: Dictionary to inject context into (typically HTTP headers)
            trace_id: Optional specific trace ID to inject context from, defaults to root context
        """
        if not getattr(self, "_ready", False):
            return

        try:
            from opentelemetry.propagate import inject

            # Get the appropriate context to inject
            context = self._get_context_for_injection(trace_id)
            if context:
                inject(carrier, context=context)
                logger.debug("Injected OTLP trace context into carrier")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to inject trace context: {e}")

    @override
    def extract_context(self, carrier: dict[str, str]) -> Context | None:
        """Extract trace context from a carrier (e.g., HTTP headers) for distributed tracing.

        Uses OpenTelemetry's standard propagation to extract W3C Trace Context headers
        from the carrier.

        Args:
            carrier: Dictionary to extract context from (typically HTTP headers)

        Returns:
            Extracted OpenTelemetry context or None if extraction fails
        """
        if not getattr(self, "_ready", False):
            return None

        try:
            from opentelemetry.propagate import extract

            context = extract(carrier)
            logger.debug("Extracted OTLP trace context from carrier")
            return context
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to extract trace context: {e}")
            return None

    def _get_context_for_injection(self, trace_id: str | None = None) -> Context | None:
        """Get the appropriate context for injection.

        Subclasses may override this to customise the selection of the trace
        context to inject into outbound request and responses. The default implementation
        will use the top of the contexts stack if available, falling back to the root_context
        if set, then to None.

        Args:
            trace_id: Optional specific trace ID to get context for

        Returns:
            Context to inject or None to use current context
        """
        # Default: try to get from contexts dict if available
        if hasattr(self, "contexts") and trace_id and trace_id in self.contexts:
            return self.contexts[trace_id]

        # Try root_context if available
        if hasattr(self, "root_context"):
            return self.root_context

        # Fall back to current context (None means use current)
        return None
