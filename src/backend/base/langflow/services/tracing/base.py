from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from langchain.callbacks.base import BaseCallbackHandler
    from lfx.graph.vertex.base import Vertex

    from langflow.services.tracing.schema import Log


class BaseTracer(ABC):
    trace_id: UUID

    @abstractmethod
    def __init__(
        self,
        trace_name: str,
        trace_type: str,
        project_name: str,
        trace_id: UUID,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def add_trace(
        self,
        trace_id: str,
        trace_name: str,
        trace_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        vertex: Vertex | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: Sequence[Log | dict] = (),
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def end(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_langchain_callback(self) -> BaseCallbackHandler | None:
        raise NotImplementedError

    def inject_context(self, carrier: dict[str, str], trace_id: str | None = None) -> None:
        """Inject trace context into a carrier (e.g., HTTP headers) for distributed tracing.

        Default no-op implementation. Override in subclasses that support context propagation.

        Callers should use the helper in src/backend/base/langflow/services/tracing/context_propagation.py
        rather than directly calling this method on the tracer.

        Args:
            carrier: Dictionary to inject context into (typically HTTP headers)
            trace_id: Optional specific trace ID to inject context from, defaults to root context
        """
        pass

    def extract_context(self, carrier: dict[str, str]) -> Any:
        """Extract trace context from a carrier (e.g., HTTP headers) for distributed tracing.

        Default no-op implementation. Override in subclasses that support context propagation.

        Callers should use the helper in src/backend/base/langflow/services/tracing/context_propagation.py
        rather than directly calling this method on the tracer.

        Args:
            carrier: Dictionary to extract context from (typically HTTP headers)

        Returns:
            Extracted context or None if extraction fails or not supported
        """
        return None
