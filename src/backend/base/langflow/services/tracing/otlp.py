from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

from lfx.log.logger import logger
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from typing_extensions import override

from langflow.schema.data import Data
from langflow.services.tracing.otlp_base import OTLPTracerBase

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from langchain.callbacks.base import BaseCallbackHandler
    from opentelemetry.trace import Span
    from lfx.graph.vertex.base import Vertex

    from langflow.services.tracing.schema import Log


class OTLPTracer(OTLPTracerBase):
    flow_id: str
    tracer_provider: TracerProvider | None = None

    def __init__(self, trace_name: str, trace_type: str, project_name: str, trace_id: UUID):
        self.trace_name = trace_name
        self.trace_type = trace_type
        self.project_name = project_name
        self.trace_id = trace_id
        self.flow_id = trace_name.split(" - ")[-1]

        try:
            self._ready: bool = self.setup_otlp()
            if not self._ready:
                return

            # Get tracer from the provider
            self.tracer = self.tracer_provider.get_tracer(__name__)
            self.spans: dict[str, Span] = {}
            self.contexts: dict[str, Context] = {}

            name_without_id = " - ".join(trace_name.split(" - ")[0:-1])
            name_without_id = project_name if name_without_id == "None" else name_without_id

            # Create root span and establish its context for propagation
            self.root_span = self.tracer.start_span(
                name=name_without_id,
                attributes={
                    "trace.id": str(self.trace_id),
                    "trace.type": "workflow",
                    "flow.id": self.flow_id,
                }
            )
            # Store the root context with the span active for propagation
            self.root_context = trace.set_span_in_context(self.root_span)
        except Exception:  # noqa: BLE001
            logger.debug("Error setting up OTLP tracer")
            self._ready = False

    @property
    def ready(self):
        return self._ready

    def setup_otlp(self) -> bool:
        # Check for standard OTLP endpoint configuration
        # SDK auto-detects from OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT
        traces_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        general_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

        if not traces_endpoint and not general_endpoint:
            return False

        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            # Initialize the shared provider if it doesn't exist
            if self.tracer_provider is None:
                # Resource.create() automatically reads standard OTEL environment variables:
                # - OTEL_SERVICE_NAME (defaults to "unknown_service")
                # - OTEL_RESOURCE_ATTRIBUTES (key=value pairs)
                resource = Resource.create()

                # OTLPSpanExporter() automatically reads configuration from environment:
                # - OTEL_EXPORTER_OTLP_TRACES_ENDPOINT (or OTEL_EXPORTER_OTLP_ENDPOINT + /v1/traces)
                # - OTEL_EXPORTER_OTLP_TRACES_HEADERS (or OTEL_EXPORTER_OTLP_HEADERS)
                # - OTEL_EXPORTER_OTLP_TRACES_TIMEOUT (or OTEL_EXPORTER_OTLP_TIMEOUT)
                # - OTEL_EXPORTER_OTLP_TRACES_COMPRESSION (or OTEL_EXPORTER_OTLP_COMPRESSION)
                # - OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE (or OTEL_EXPORTER_OTLP_CERTIFICATE)
                exporter = OTLPSpanExporter()

                provider = TracerProvider(resource=resource)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                OTLPTracer.tracer_provider = provider

        except ImportError as e:
            logger.exception(f"Failed to import OTLP exporter: {e}")
            return False
        except Exception as e:
            logger.exception(f"Error setting up OTLP tracer: {e}")
            return False
        return True

    @override
    def add_trace(
        self,
        trace_id: str,
        trace_name: str,
        trace_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        vertex: Vertex | None = None,
    ) -> None:
        if not self._ready:
            return

        # Store session information as span attributes
        attributes = {
            "trace.id": trace_id,
            "trace.type": trace_type,
        }

        if "session_id" in inputs and inputs["session_id"] != self.flow_id:
            attributes["session.id"] = inputs["session_id"]

        name_without_id = " (".join(trace_name.split(" (")[0:-1])

        # Determine parent context for proper trace propagation
        parent_context = self.root_context
        if vertex and len(vertex.incoming_edges) > 0:
            for edge in vertex.incoming_edges:
                if edge.source_id in self.contexts:
                    parent_context = self.contexts[edge.source_id]
                    break

        # Create span with parent context for proper trace propagation
        span = self.tracer.start_span(
            name=name_without_id,
            context=parent_context,
            attributes=attributes
        )

        # Store converted inputs as span events
        converted_inputs = self._convert_to_trace_format(inputs)
        if converted_inputs:
            span.add_event("inputs", attributes={"data": str(converted_inputs)})

        # Store span and its context for propagation to child spans
        self.spans[trace_id] = span
        self.contexts[trace_id] = trace.set_span_in_context(span, parent_context)

    @override
    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: Sequence[Log | dict] = (),
    ) -> None:
        if not self._ready:
            return

        span = self.spans.get(trace_id)
        if span:
            # Add outputs as event
            if outputs:
                converted_outputs = self._convert_to_trace_format(outputs)
                span.add_event("outputs", attributes={"data": str(converted_outputs)})

            # Record error if present
            if error:
                span.record_exception(error)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))

            span.end()

    def end(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._ready:
            return

        # Add metadata to root span
        if metadata:
            if "flow_name" in metadata:
                self.root_span.set_attribute("flow.name", metadata["flow_name"])

        # Add inputs/outputs to root span
        if inputs:
            converted_inputs = self._convert_to_trace_format(inputs)
            self.root_span.add_event("inputs", attributes={"data": str(converted_inputs)})

        if outputs:
            converted_outputs = self._convert_to_trace_format(outputs)
            self.root_span.add_event("outputs", attributes={"data": str(converted_outputs)})

        # Record error if present
        if error:
            self.root_span.record_exception(error)
            self.root_span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))

        # End root span
        self.root_span.end()

    def _convert_to_trace_format(self, io_dict: dict[str, Any] | None):
        """Convert inputs/outputs to a format suitable for tracing."""
        if io_dict is None:
            return None

        converted = {}
        for key, value in io_dict.items():
            converted[key] = self._convert_value(value)
        return converted

    def _convert_value(self, value):
        """Convert individual values to trace-compatible formats."""
        from langchain_core.messages import BaseMessage
        from lfx.schema.message import Message

        if isinstance(value, dict):
            return {key: self._convert_value(val) for key, val in value.items()}
        elif isinstance(value, list):
            return [self._convert_value(v) for v in value]
        elif isinstance(value, Message):
            if "prompt" in value:
                prompt = value.load_lc_prompt()
                if len(prompt.input_variables) == 0 and all(isinstance(m, BaseMessage) for m in prompt.messages):
                    return {"messages": [str(m) for m in prompt.messages]}
                else:
                    return cast("dict", value.load_lc_prompt())
            elif value.sender:
                return {"message": str(value.to_lc_message())}
            else:
                return cast("dict", value.to_lc_document())
        elif isinstance(value, Data):
            return cast("dict", value.to_lc_document())
        return value

    def get_langchain_callback(self) -> BaseCallbackHandler | None:
        # Standard OTLP tracing doesn't provide LangChain callbacks
        # This would need to be implemented separately if needed
        return None
