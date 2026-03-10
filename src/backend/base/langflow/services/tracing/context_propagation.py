"""Central helper for trace context propagation across HTTP and gRPC boundaries.

This module provides utilities for finding and caching active tracers that support
context propagation, making it easy to inject and extract trace context in
middlewares and interceptors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lfx.log.logger import logger

if TYPE_CHECKING:
    from langflow.services.tracing.base import BaseTracer


_cached_propagation_tracer: BaseTracer | None = None
_cache_generation: int = 0


def get_propagation_tracer(force_refresh: bool = False) -> BaseTracer | None:
    """Get an active tracer that supports trace context propagation.

    This function finds and caches the first ready tracer that implements
    both inject_context and extract_context methods. The result is cached
    for performance, but can be refreshed if needed.

    Args:
        force_refresh: If True, ignore the cache and search for a tracer again

    Returns:
        A tracer instance that supports context propagation, or None if none found

    Example:
        >>> from langflow.services.tracing.context_propagation import get_propagation_tracer
        >>>
        >>> tracer = get_propagation_tracer()
        >>> if tracer:
        ...     headers = {}
        ...     tracer.inject_context(headers)
    """
    global _cached_propagation_tracer, _cache_generation

    # Return cached tracer if available and not forcing refresh
    if not force_refresh and _cached_propagation_tracer is not None:
        # Verify cached tracer is still ready
        if hasattr(_cached_propagation_tracer, "ready") and _cached_propagation_tracer.ready:
            return _cached_propagation_tracer
        # Cached tracer is no longer ready, clear cache
        _cached_propagation_tracer = None

    # Search for an active tracer with context propagation support
    try:
        from langflow.services.tracing.service import trace_context_var

        trace_context = trace_context_var.get()
        if trace_context and trace_context.tracers:
            for tracer_name, tracer in trace_context.tracers.items():
                # Check if tracer supports context propagation
                if (
                    hasattr(tracer, "inject_context")
                    and hasattr(tracer, "extract_context")
                    and hasattr(tracer, "ready")
                    and tracer.ready
                ):
                    logger.debug(f"Found active tracer with context propagation: {tracer_name}")
                    _cached_propagation_tracer = tracer
                    _cache_generation += 1
                    return tracer

        logger.debug("No active tracer with context propagation support found")
        return None

    except Exception as e:  # noqa: BLE001
        logger.debug(f"Error finding propagation tracer: {e}")
        return None


def inject_trace_context(carrier: dict[str, str], trace_id: str | None = None) -> bool:
    """Inject trace context into a carrier (typically HTTP headers or gRPC metadata).

    This is a convenience function that finds an active tracer and uses it to
    inject trace context. Returns True if successful, False otherwise.

    Args:
        carrier: Dictionary to inject context into (will be modified in-place)
        trace_id: Optional specific trace ID to inject context from

    Returns:
        True if context was injected successfully, False otherwise

    Example:
        >>> from langflow.services.tracing.context_propagation import inject_trace_context
        >>>
        >>> headers = {"Authorization": "Bearer token"}
        >>> if inject_trace_context(headers):
        ...     response = httpx.get("https://api.example.com", headers=headers)
    """
    tracer = get_propagation_tracer()
    if tracer and hasattr(tracer, "inject_context"):
        try:
            tracer.inject_context(carrier, trace_id=trace_id)
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to inject trace context: {e}")
    return False


def extract_trace_context(carrier: dict[str, str]) -> tuple[Any, BaseTracer | None]:
    """Extract trace context from a carrier (typically HTTP headers or gRPC metadata).

    This is a convenience function that finds an active tracer and uses it to
    extract trace context. Returns both the extracted context and the tracer
    that extracted it.

    Args:
        carrier: Dictionary to extract context from

    Returns:
        Tuple of (extracted_context, tracer) or (None, None) if extraction failed

    Example:
        >>> from langflow.services.tracing.context_propagation import extract_trace_context
        >>>
        >>> context, tracer = extract_trace_context(request.headers)
        >>> if context:
        ...     # Store for later use
        ...     request.state.trace_context = context
        ...     request.state.tracer = tracer
    """
    tracer = get_propagation_tracer()
    if tracer and hasattr(tracer, "extract_context"):
        try:
            context = tracer.extract_context(carrier)
            if context:
                return context, tracer
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to extract trace context: {e}")
    return None, None


def clear_propagation_cache() -> None:
    """Clear the cached propagation tracer.

    This can be useful when tracers are reconfigured or during testing.
    The next call to get_propagation_tracer() will search for a tracer again.
    """
    global _cached_propagation_tracer, _cache_generation
    _cached_propagation_tracer = None
    _cache_generation += 1
    logger.debug("Cleared propagation tracer cache")


def get_cache_generation() -> int:
    """Get the current cache generation number.

    This increments each time the cache is cleared or refreshed, and can be
    useful for debugging or monitoring cache behavior.

    Returns:
        Current cache generation number
    """
    return _cache_generation
