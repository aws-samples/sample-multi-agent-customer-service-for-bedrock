"""Escalation tools action-group Lambda for the AnyCompany escalation agent.

Implements the ``escalation-tools`` action group with two functions:

    create_ticket(account_number, issue_summary, category, priority?)
        -> ticket JSON
    transfer_to_agent(ticket_id, reason, account_number?)
        -> Amazon Connect handoff JSON

Both operations are backed by the shared in-memory mock datastore
(:mod:`mock_data`) in the common Lambda layer. The transfer function simulates
an Amazon Connect warm handoff, inferring the destination queue/team from the
free-text reason.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from bedrock_agent import build_response, dispatch
from mock_data import (
    VALID_PRIORITIES,
    VALID_TICKET_CATEGORIES,
    open_ticket,
    simulate_transfer,
)

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_ticket(parameters: Dict[str, str]) -> str:
    """Open a customer-service escalation ticket.

    Args:
        parameters: Resolved action-group parameters. Requires
            ``account_number``, ``issue_summary`` and ``category``; ``priority``
            is optional and defaults to ``MEDIUM``.

    Returns:
        A compact JSON string with ``ticket_id``, ``status`` (``OPEN``),
        ``category``, ``priority``, ``sla_hours`` and ``created_at``; or an
        error body for invalid/missing input.
    """
    account_number = parameters.get("account_number", "").strip()
    issue_summary = parameters.get("issue_summary", "").strip()
    category = parameters.get("category", "").strip()
    priority = parameters.get("priority", "").strip() or "MEDIUM"

    missing = [
        name
        for name, value in (
            ("account_number", account_number),
            ("issue_summary", issue_summary),
            ("category", category),
        )
        if not value
    ]
    if missing:
        logger.warning("create_ticket missing parameters: %s", missing)
        return json.dumps(
            {
                "error": "MISSING_PARAMETER",
                "message": (
                    "account_number, issue_summary and category are all "
                    "required for create_ticket."
                ),
                "missing": missing,
                "valid_categories": VALID_TICKET_CATEGORIES,
                "valid_priorities": VALID_PRIORITIES,
            }
        )

    try:
        ticket = open_ticket(
            account_number=account_number,
            issue_summary=issue_summary,
            category=category,
            priority=priority,
        )
    except ValueError as error:
        logger.info("create_ticket rejected input: %s", error)
        return json.dumps(
            {
                "error": "INVALID_PARAMETER",
                "message": str(error),
                "valid_categories": VALID_TICKET_CATEGORIES,
                "valid_priorities": VALID_PRIORITIES,
            }
        )

    result = {
        "ticket_id": ticket["ticket_id"],
        "status": ticket["status"],
        "category": ticket["category"],
        "priority": ticket["priority"],
        "sla_hours": ticket["sla_hours"],
        "created_at": ticket["created_at"],
    }
    logger.info(
        "create_ticket opened %s (category=%s, priority=%s, sla=%dh)",
        ticket["ticket_id"],
        ticket["category"],
        ticket["priority"],
        ticket["sla_hours"],
    )
    return json.dumps(result)


def transfer_to_agent(parameters: Dict[str, str]) -> str:
    """Simulate an Amazon Connect warm transfer to a human agent.

    Args:
        parameters: Resolved action-group parameters. Requires ``ticket_id``
            and ``reason``; ``account_number`` is optional.

    Returns:
        A compact JSON string with ``transfer_id``, ``ticket_id``, ``queue``,
        ``assigned_team``, ``estimated_wait_minutes``, ``connect_contact_id``
        and ``status`` (``TRANSFERRED``); or an error body for missing input.
    """
    ticket_id = parameters.get("ticket_id", "").strip()
    reason = parameters.get("reason", "").strip()
    account_number = parameters.get("account_number", "").strip() or None

    missing = [
        name
        for name, value in (
            ("ticket_id", ticket_id),
            ("reason", reason),
        )
        if not value
    ]
    if missing:
        logger.warning("transfer_to_agent missing parameters: %s", missing)
        return json.dumps(
            {
                "error": "MISSING_PARAMETER",
                "message": (
                    "ticket_id and reason are both required for "
                    "transfer_to_agent."
                ),
                "missing": missing,
            }
        )

    transfer = simulate_transfer(
        ticket_id=ticket_id,
        reason=reason,
        account_number=account_number,
    )
    result = {
        "transfer_id": transfer["transfer_id"],
        "ticket_id": transfer["ticket_id"],
        "queue": transfer["queue"],
        "assigned_team": transfer["assigned_team"],
        "estimated_wait_minutes": transfer["estimated_wait_minutes"],
        "connect_contact_id": transfer["connect_contact_id"],
        "status": transfer["status"],
    }
    logger.info(
        "transfer_to_agent routed ticket %s to %s (queue=%s, contact=%s)",
        transfer["ticket_id"],
        transfer["assigned_team"],
        transfer["queue"],
        transfer["connect_contact_id"],
    )
    return json.dumps(result)


# Map of function name -> handler callable for this action group.
HANDLERS: Dict[str, Any] = {
    "create_ticket": create_ticket,
    "transfer_to_agent": transfer_to_agent,
}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point for the escalation-tools action group.

    Args:
        event: The Bedrock action-group invocation event.
        context: The Lambda runtime context (unused).

    Returns:
        A valid Bedrock action-group response envelope.
    """
    logger.info("escalation_tools received function '%s'", event.get("function"))
    try:
        return dispatch(event, HANDLERS)
    except Exception as error:  # noqa: BLE001 - never crash the agent.
        logger.exception("Unhandled error in escalation_tools: %s", error)
        return build_response(
            event,
            json.dumps(
                {
                    "error": "ESCALATION_TOOL_ERROR",
                    "message": "Escalation operation failed unexpectedly.",
                }
            ),
        )
