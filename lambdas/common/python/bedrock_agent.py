"""Shared Bedrock Agent action-group helper layer.

This module is packaged as the ``python`` directory of an AWS Lambda layer so
that every action-group handler can import it directly as ``bedrock_agent``.

It implements the Bedrock Agents *function schema* invocation contract:

INPUT EVENT shape (delivered by Bedrock to the action-group Lambda)::

    {
      "messageVersion": "1.0",
      "agent": {"name": str, "id": str, "alias": str, "version": str},
      "actionGroup": str,
      "function": str,
      "parameters": [{"name": str, "type": str, "value": str}, ...],
      "sessionId": str,
      "sessionAttributes": {},
      "promptSessionAttributes": {},
      "inputText": str
    }

OUTPUT response shape (expected back by Bedrock)::

    {
      "messageVersion": "1.0",
      "response": {
        "actionGroup": <event.actionGroup>,
        "function": <event.function>,
        "functionResponse": {"responseBody": {"TEXT": {"body": <string>}}}
      },
      "sessionAttributes": <event.sessionAttributes or {}>,
      "promptSessionAttributes": <event.promptSessionAttributes or {}>
    }

Handlers use this layer like so::

    from bedrock_agent import get_parameters, build_response, dispatch
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

# Module-level logger. Lambda's root logger is already configured, so we just
# attach to the hierarchy and let the runtime forward records to CloudWatch.
logger: logging.Logger = logging.getLogger(__name__)
if not logger.handlers:
    # Defensive default so the module is usable outside the Lambda runtime
    # (e.g. local unit tests) without emitting "No handlers" warnings.
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# Type alias for a handler function: it receives the resolved parameter map and
# returns a plain string body (which may itself be a JSON string).
FunctionHandler = Callable[[Dict[str, str]], str]


def get_named_parameter(
    event: Dict[str, Any], name: str, default: Optional[str] = None
) -> Optional[str]:
    """Return a single named parameter value from a Bedrock action-group event.

    Args:
        event: The raw Bedrock action-group invocation event.
        name: The parameter name to look up (matches the function schema).
        default: Value returned when the parameter is absent or has no value.

    Returns:
        The parameter's string value, or ``default`` when not present.
    """
    try:
        parameters = event.get("parameters") or []
        for parameter in parameters:
            if parameter.get("name") == name:
                value = parameter.get("value")
                # Bedrock always sends values as strings, but guard against None.
                return value if value is not None else default
        return default
    except (AttributeError, TypeError) as error:
        logger.error(
            "Failed to extract parameter '%s' from event: %s", name, error
        )
        return default


def get_parameters(event: Dict[str, Any]) -> Dict[str, str]:
    """Return all action-group parameters as a ``name -> value`` mapping.

    Args:
        event: The raw Bedrock action-group invocation event.

    Returns:
        A dictionary mapping each parameter name to its string value. Parameters
        with a ``None`` value are skipped so callers can rely on ``in`` checks.
    """
    result: Dict[str, str] = {}
    try:
        parameters = event.get("parameters") or []
        for parameter in parameters:
            name = parameter.get("name")
            value = parameter.get("value")
            if name is not None and value is not None:
                result[str(name)] = str(value)
    except (AttributeError, TypeError) as error:
        logger.error("Failed to build parameter map from event: %s", error)
    return result


def build_response(
    event: Dict[str, Any],
    body: str,
    *,
    session_attributes: Optional[Dict[str, Any]] = None,
    prompt_session_attributes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a Bedrock action-group function response envelope.

    Args:
        event: The originating Bedrock invocation event (used to echo back the
            ``actionGroup`` and ``function`` identifiers and the session state).
        body: The human-and-LLM friendly text body. Structured data should be
            passed as a compact JSON string.
        session_attributes: Optional override for the returned session
            attributes. Defaults to the event's session attributes.
        prompt_session_attributes: Optional override for the returned prompt
            session attributes. Defaults to the event's prompt session
            attributes.

    Returns:
        A dictionary matching the Bedrock function-schema response contract.
    """
    resolved_session = (
        session_attributes
        if session_attributes is not None
        else event.get("sessionAttributes") or {}
    )
    resolved_prompt = (
        prompt_session_attributes
        if prompt_session_attributes is not None
        else event.get("promptSessionAttributes") or {}
    )

    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event.get("actionGroup", ""),
            "function": event.get("function", ""),
            "functionResponse": {
                "responseBody": {
                    "TEXT": {
                        "body": body,
                    }
                }
            },
        },
        "sessionAttributes": resolved_session,
        "promptSessionAttributes": resolved_prompt,
    }


def dispatch(
    event: Dict[str, Any], handlers: Dict[str, FunctionHandler]
) -> Dict[str, Any]:
    """Route a Bedrock action-group event to the matching function handler.

    Looks up ``event["function"]`` in ``handlers``, invokes it with the resolved
    parameter map, and wraps the returned body in the Bedrock response envelope.
    Unknown functions and handler exceptions are logged and surfaced as a safe
    error body so the agent still receives a valid (well-formed) response.

    Args:
        event: The raw Bedrock action-group invocation event.
        handlers: Mapping of function name to a callable that accepts the
            resolved parameter dictionary and returns a string body.

    Returns:
        A valid Bedrock action-group response envelope in all cases.
    """
    function_name = event.get("function", "")
    logger.info(
        "Dispatching action group '%s' function '%s'",
        event.get("actionGroup", ""),
        function_name,
    )

    handler = handlers.get(function_name)
    if handler is None:
        logger.error(
            "Unknown function '%s'. Available functions: %s",
            function_name,
            sorted(handlers.keys()),
        )
        error_body = json.dumps(
            {
                "error": "UNKNOWN_FUNCTION",
                "message": (
                    f"The function '{function_name}' is not implemented by "
                    "this action group."
                ),
                "available_functions": sorted(handlers.keys()),
            }
        )
        return build_response(event, error_body)

    try:
        parameters = get_parameters(event)
        body = handler(parameters)
        logger.info("Function '%s' completed successfully", function_name)
        return build_response(event, body)
    except Exception as error:  # noqa: BLE001 - we must never crash the agent.
        logger.exception(
            "Handler for function '%s' raised an exception: %s",
            function_name,
            error,
        )
        error_body = json.dumps(
            {
                "error": "TOOL_EXECUTION_ERROR",
                "message": (
                    "An internal error occurred while processing this request. "
                    "Please try again or escalate to a human agent."
                ),
                "function": function_name,
            }
        )
        return build_response(event, error_body)
