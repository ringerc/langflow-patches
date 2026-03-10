"""gRPC interceptors for trace context propagation.

This module provides client and server interceptors for extracting and injecting
trace context in gRPC calls to enable distributed tracing across gRPC services.
Works with any tracer that supports context propagation (OTLP, LangWatch, Traceloop, etc).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from lfx.log.logger import logger

if TYPE_CHECKING:
    import grpc


class OTLPClientInterceptor:
    """gRPC client interceptor that injects trace context into outgoing calls.

    Works with any tracer that implements the inject_context method.
    """

    def __init__(self, tracer):
        """Initialize the client interceptor.

        Args:
            tracer: Tracer instance with inject_context method (OTLP, LangWatch, Traceloop, etc)
        """
        self.tracer = tracer

    def intercept_unary_unary(
        self,
        continuation: Callable,
        client_call_details: Any,
        request: Any,
    ) -> Any:
        """Intercept unary-unary RPC calls to inject trace context.

        Args:
            continuation: Function to continue the RPC
            client_call_details: Details of the RPC call
            request: Request message

        Returns:
            Response from the RPC call
        """
        try:
            # Create metadata dict from existing metadata
            metadata = dict(client_call_details.metadata or [])

            # Inject trace context into metadata
            if self.tracer and hasattr(self.tracer, "inject_context"):
                self.tracer.inject_context(metadata)
                logger.debug("Injected OTLP trace context into gRPC call")

            # Update client call details with new metadata
            new_details = client_call_details._replace(
                metadata=list(metadata.items())
            )
            return continuation(new_details, request)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to inject trace context into gRPC call: {e}")
            return continuation(client_call_details, request)

    def intercept_unary_stream(
        self,
        continuation: Callable,
        client_call_details: Any,
        request: Any,
    ) -> Any:
        """Intercept unary-stream RPC calls to inject trace context."""
        return self.intercept_unary_unary(continuation, client_call_details, request)

    def intercept_stream_unary(
        self,
        continuation: Callable,
        client_call_details: Any,
        request_iterator: Any,
    ) -> Any:
        """Intercept stream-unary RPC calls to inject trace context."""
        try:
            metadata = dict(client_call_details.metadata or [])
            if self.tracer and hasattr(self.tracer, "inject_context"):
                self.tracer.inject_context(metadata)
                logger.debug("Injected OTLP trace context into gRPC stream call")
            new_details = client_call_details._replace(
                metadata=list(metadata.items())
            )
            return continuation(new_details, request_iterator)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to inject trace context into gRPC stream call: {e}")
            return continuation(client_call_details, request_iterator)

    def intercept_stream_stream(
        self,
        continuation: Callable,
        client_call_details: Any,
        request_iterator: Any,
    ) -> Any:
        """Intercept stream-stream RPC calls to inject trace context."""
        return self.intercept_stream_unary(continuation, client_call_details, request_iterator)


class OTLPServerInterceptor:
    """gRPC server interceptor that extracts trace context from incoming calls.

    Works with any tracer that implements the extract_context method.
    """

    def __init__(self, tracer):
        """Initialize the server interceptor.

        Args:
            tracer: Tracer instance with extract_context method (OTLP, LangWatch, Traceloop, etc)
        """
        self.tracer = tracer

    def intercept_service(self, continuation: Callable, handler_call_details: Any) -> Any:
        """Intercept incoming gRPC calls to extract trace context.

        Args:
            continuation: Function to continue handling the RPC
            handler_call_details: Details of the handler call

        Returns:
            Handler for the RPC call
        """
        try:
            # Extract metadata from the call
            metadata = dict(handler_call_details.invocation_metadata or [])

            # Extract trace context from metadata
            if self.tracer and hasattr(self.tracer, "extract_context"):
                context = self.tracer.extract_context(metadata)
                if context:
                    # Store context for use in handlers
                    # Note: In production, you'd want to use gRPC context to pass this along
                    logger.debug("Extracted OTLP trace context from gRPC call")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to extract trace context from gRPC call: {e}")

        return continuation(handler_call_details)


def create_otlp_grpc_client_interceptors(tracer) -> list:
    """Create gRPC client interceptors for trace context propagation.

    Works with any tracer that implements inject_context (OTLP, LangWatch, Traceloop, etc).

    Args:
        tracer: Tracer instance that implements inject_context method

    Returns:
        List of gRPC interceptors to use with grpc.intercept_channel

    Example:
        >>> import grpc
        >>> from langflow.services.tracing.context_propagation import get_propagation_tracer
        >>>
        >>> # Get an active tracer that supports context propagation
        >>> tracer = get_propagation_tracer()
        >>> if tracer:
        ...     interceptors = create_otlp_grpc_client_interceptors(tracer)
        ...     channel = grpc.insecure_channel('localhost:50051')
        ...     channel = grpc.intercept_channel(channel, *interceptors)
    """
    try:
        import grpc

        interceptor = OTLPClientInterceptor(tracer)
        return [
            grpc.UnaryUnaryClientInterceptor(
                intercept_unary_unary=interceptor.intercept_unary_unary
            ),
            grpc.UnaryStreamClientInterceptor(
                intercept_unary_stream=interceptor.intercept_unary_stream
            ),
            grpc.StreamUnaryClientInterceptor(
                intercept_stream_unary=interceptor.intercept_stream_unary
            ),
            grpc.StreamStreamClientInterceptor(
                intercept_stream_stream=interceptor.intercept_stream_stream
            ),
        ]
    except ImportError:
        logger.warning("grpcio not installed, gRPC trace context propagation unavailable")
        return []


def create_otlp_grpc_server_interceptor(tracer) -> grpc.ServerInterceptor | None:
    """Create gRPC server interceptor for trace context extraction.

    Works with any tracer that implements extract_context (OTLP, LangWatch, Traceloop, etc).

    Args:
        tracer: Tracer instance that implements extract_context method

    Returns:
        gRPC server interceptor or None if grpc is not available

    Example:
        >>> import grpc
        >>> from concurrent import futures
        >>> from langflow.services.tracing.context_propagation import get_propagation_tracer
        >>>
        >>> # Get an active tracer that supports context propagation
        >>> tracer = get_propagation_tracer()
        >>> if tracer:
        ...     interceptor = create_otlp_grpc_server_interceptor(tracer)
        ...     server = grpc.server(
        ...         futures.ThreadPoolExecutor(max_workers=10),
        ...         interceptors=[interceptor] if interceptor else []
        ...     )
    """
    try:
        import grpc

        return OTLPServerInterceptor(tracer)
    except ImportError:
        logger.warning("grpcio not installed, gRPC trace context extraction unavailable")
        return None
