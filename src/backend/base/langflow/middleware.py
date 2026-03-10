from fastapi import HTTPException, Request, Response
from lfx.log.logger import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from langflow.services.deps import get_settings_service


class MaxFileSizeException(HTTPException):
    def __init__(self, detail: str = "File size is larger than the maximum file size {}MB"):
        super().__init__(status_code=413, detail=detail)


class OTLPTraceContextMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and propagate trace context from incoming requests.

    This middleware extracts W3C Trace Context headers (traceparent, tracestate) from
    incoming HTTP requests and makes them available to active tracers for distributed tracing.
    Works with any tracer that supports context propagation (OTLP, LangWatch, Traceloop, etc).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Extract trace context from request headers and process the request."""
        try:
            from langflow.services.tracing.context_propagation import extract_trace_context

            # Extract trace context from incoming headers
            headers_dict = dict(request.headers)
            extracted_context, active_tracer = extract_trace_context(headers_dict)

            # Store the extracted context in request state for use during processing
            if extracted_context and active_tracer:
                request.state.trace_context = extracted_context
                request.state.tracer = active_tracer
                logger.debug("Extracted trace context from incoming request")

        except Exception as e:  # noqa: BLE001
            # Don't fail requests if trace context extraction fails
            logger.debug(f"Failed to extract trace context: {e}")

        # Process the request
        response = await call_next(request)

        # Optionally inject trace context into response headers for correlation
        try:
            if hasattr(request.state, "trace_context") and hasattr(request.state, "tracer"):
                response_headers = {}
                request.state.tracer.inject_context(response_headers)
                # Update response headers with trace context
                for key, value in response_headers.items():
                    if key.lower() in ("traceparent", "tracestate"):
                        response.headers[key] = value
                logger.debug("Injected trace context into response headers")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to inject trace context into response: {e}")

        return response


# Adapted from https://github.com/steinnes/content-size-limit-asgi/blob/master/content_size_limit_asgi/middleware.py#L26
class ContentSizeLimitMiddleware:
    """Content size limiting middleware for ASGI applications.

    Args:
      app (ASGI application): ASGI application
      max_content_size (optional): the maximum content size allowed in bytes, None for no limit
      exception_cls (optional): the class of exception to raise (ContentSizeExceeded is the default)
    """

    def __init__(
        self,
        app,
    ):
        self.app = app
        self.logger = logger

    @staticmethod
    def receive_wrapper(receive):
        received = 0

        async def inner():
            max_file_size_upload = get_settings_service().settings.max_file_size_upload
            nonlocal received
            message = await receive()
            if message["type"] != "http.request" or max_file_size_upload is None:
                return message
            body_len = len(message.get("body", b""))
            received += body_len
            if received > max_file_size_upload * 1024 * 1024:
                # max_content_size is in bytes, convert to MB
                received_in_mb = round(received / (1024 * 1024), 3)
                msg = (
                    f"Content size limit exceeded. Maximum allowed is {max_file_size_upload}MB"
                    f" and got {received_in_mb}MB."
                )
                raise MaxFileSizeException(msg)
            return message

        return inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        wrapper = self.receive_wrapper(receive)
        await self.app(scope, wrapper, send)


def inject_trace_context_into_headers(headers: dict[str, str], trace_id: str | None = None) -> dict[str, str]:
    """Inject trace context into HTTP headers for outgoing requests.

    This function should be used when making outgoing HTTP requests to propagate
    trace context to downstream services. Works with any active tracer that supports
    context propagation (OTLP, LangWatch, Traceloop, ArizePhoenix, etc).

    Args:
        headers: Dictionary of HTTP headers to inject trace context into
        trace_id: Optional specific trace ID to inject context from, defaults to root context

    Returns:
        Updated headers dictionary with trace context injected

    Example:
        >>> import httpx
        >>> from langflow.middleware import inject_trace_context_into_headers
        >>> headers = {"Authorization": "Bearer token"}
        >>> headers = inject_trace_context_into_headers(headers)
        >>> response = httpx.get("https://api.example.com", headers=headers)
    """
    from langflow.services.tracing.context_propagation import inject_trace_context

    # Make a copy to avoid modifying the original
    headers_copy = headers.copy()

    # Inject trace context into the headers copy
    if inject_trace_context(headers_copy, trace_id=trace_id):
        logger.debug("Injected trace context into outgoing request headers")
        return headers_copy

    return headers
