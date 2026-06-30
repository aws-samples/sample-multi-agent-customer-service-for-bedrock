"""Strands ``@tool`` functions for the customer-service specialists.

These replace the original Bedrock action-group Lambda handlers. The business
logic is unchanged — each tool delegates to the deterministic in-memory
datastore in :mod:`agents.mock_data` (and the keyword classifier in
:mod:`agents.intent`) — but the interface is now a decorated Python function
that Strands exposes to the model via its docstring and type hints, and the
tools run **in process** rather than as separate Lambdas.

Tools return plain dictionaries; Strands serialises them into the tool-result
content automatically. Validation/error payloads mirror the original contract so
the agents behave identically.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from strands import tool

from agents import mock_data
from agents.intent import classify_intent_payload

logger: logging.Logger = logging.getLogger(__name__)

_KNOWN_SAMPLE_ACCOUNTS = ["ACC-100123", "ACC-100456", "ACC-100789"]


def _account_not_found(account_number: str) -> Dict[str, Any]:
    """Build a consistent payload for an unknown account."""
    return {
        "error": "ACCOUNT_NOT_FOUND",
        "account_number": account_number,
        "message": (
            f"No account was found matching '{account_number}'. Please confirm "
            "the account number, which has the format ACC-NNNNNN."
        ),
        "known_sample_accounts": _KNOWN_SAMPLE_ACCOUNTS,
    }


# --------------------------------------------------------------------------- #
# Supervisor tool: intent classification
# --------------------------------------------------------------------------- #
@tool
def classify_intent(query: str) -> Dict[str, Any]:
    """Classify a customer's free-text query into one of BILLING, SCHEDULING,
    ESCALATION or UNKNOWN using a keyword/heuristic scorer.

    Returns the intent, a 0..1 confidence, the recommended specialist and a short
    reasoning string. A normalised top score below 0.34 yields UNKNOWN.

    Args:
        query: The customer's raw question or request to classify.

    Returns:
        A dict with ``intent``, ``confidence``, ``recommended_agent`` and
        ``reasoning``.
    """
    return classify_intent_payload(query)


# --------------------------------------------------------------------------- #
# Billing tools
# --------------------------------------------------------------------------- #
@tool
def lookup_account(account_number: str) -> Dict[str, Any]:
    """Look up a customer account profile by account number.

    Returns name, tariff, status, postcode and whether the account is on a
    payment plan.

    Args:
        account_number: The AnyCompany account number, e.g. ACC-100123.

    Returns:
        The account profile, or an error payload when the account is unknown.
    """
    account_number = (account_number or "").strip()
    if not account_number:
        return {"error": "MISSING_PARAMETER", "message": "account_number is required."}

    account = mock_data.get_account(account_number)
    if account is None:
        logger.info("lookup_account miss for '%s'", account_number)
        return _account_not_found(account_number)

    return {
        "account_number": account["account_number"],
        "name": account["name"],
        "tariff": account["tariff"],
        "status": account["status"],
        "postcode": account["postcode"],
        "on_payment_plan": account["on_payment_plan"],
    }


@tool
def check_balance(account_number: str) -> Dict[str, Any]:
    """Retrieve the current balance and payment status for an account.

    Returns balance_gbp, due_date, last_payment and status.

    Args:
        account_number: The AnyCompany account number, e.g. ACC-100123.

    Returns:
        The balance payload, or an error payload when the account is unknown.
    """
    account_number = (account_number or "").strip()
    if not account_number:
        return {"error": "MISSING_PARAMETER", "message": "account_number is required."}

    account = mock_data.get_account(account_number)
    if account is None:
        logger.info("check_balance miss for '%s'", account_number)
        return _account_not_found(account_number)

    return {
        "account_number": account["account_number"],
        "balance_gbp": account["balance_gbp"],
        "due_date": account["due_date"],
        "last_payment": account["last_payment"],
        "status": account["status"],
    }


# --------------------------------------------------------------------------- #
# Scheduling tools
# --------------------------------------------------------------------------- #
@tool
def check_availability(
    appointment_type: str, postcode: str, preferred_date: str = ""
) -> Dict[str, Any]:
    """List available appointment slots for an appointment type and postcode,
    optionally near a preferred ISO date.

    Returns a list of slots with slot_id, date, time_window and engineer_type.

    Args:
        appointment_type: One of METER_READING, ENGINEER_VISIT,
            LEAK_INVESTIGATION or EV_CHARGER_INSTALL.
        postcode: The customer's UK postcode, e.g. M14 5TG.
        preferred_date: Optional preferred date in ISO format (YYYY-MM-DD).

    Returns:
        A payload with ``appointment_type``, ``postcode`` and a ``slots`` list,
        or an error payload for invalid input.
    """
    appointment_type = (appointment_type or "").strip()
    postcode = (postcode or "").strip()
    preferred = (preferred_date or "").strip() or None

    if not appointment_type or not postcode:
        return {
            "error": "MISSING_PARAMETER",
            "message": "appointment_type and postcode are both required.",
            "valid_appointment_types": mock_data.VALID_APPOINTMENT_TYPES,
        }

    try:
        slots = mock_data.list_appointment_slots(appointment_type, postcode, preferred)
    except ValueError as error:
        logger.info("check_availability rejected input: %s", error)
        return {
            "error": "INVALID_APPOINTMENT_TYPE",
            "message": str(error),
            "valid_appointment_types": mock_data.VALID_APPOINTMENT_TYPES,
        }

    return {
        "appointment_type": appointment_type.upper(),
        "postcode": postcode.upper(),
        "slots": slots,
    }


@tool
def book_appointment(
    account_number: str, slot_id: str, contact_phone: str
) -> Dict[str, Any]:
    """Book a previously listed slot for an account.

    Returns the booking_reference, confirmed_date, confirmed_time_window,
    appointment_type and a CONFIRMED status.

    Args:
        account_number: The AnyCompany account number the booking is for.
        slot_id: The slot_id returned by check_availability.
        contact_phone: A contact phone number for the engineer to confirm.

    Returns:
        The booking confirmation, or an error payload for invalid input.
    """
    account_number = (account_number or "").strip()
    slot_id = (slot_id or "").strip()
    contact_phone = (contact_phone or "").strip()

    missing = [
        name
        for name, value in (
            ("account_number", account_number),
            ("slot_id", slot_id),
            ("contact_phone", contact_phone),
        )
        if not value
    ]
    if missing:
        return {
            "error": "MISSING_PARAMETER",
            "message": "account_number, slot_id and contact_phone are all required.",
            "missing": missing,
        }

    # Confirm the account exists before booking (CRM pre-flight check).
    account = mock_data.get_account(account_number)
    if account is None:
        logger.info("book_appointment rejected unknown account '%s'", account_number)
        return {
            "error": "ACCOUNT_NOT_FOUND",
            "account_number": account_number,
            "message": (
                f"Cannot book an appointment: account '{account_number}' was not "
                "found. Please confirm the account number."
            ),
        }

    booking = mock_data.create_booking(
        account_number=account_number,
        slot_id=slot_id,
        contact_phone=contact_phone,
    )
    return {
        "booking_reference": booking["booking_reference"],
        "confirmed_date": booking["confirmed_date"],
        "confirmed_time_window": booking["confirmed_time_window"],
        "appointment_type": booking["appointment_type"],
        "status": booking["status"],
    }


@tool
def cancel_appointment(booking_reference: str) -> Dict[str, Any]:
    """Cancel an existing appointment by its booking reference.

    Returns the booking_reference and a CANCELLED status.

    Args:
        booking_reference: The booking reference to cancel, e.g. APPT-100001.

    Returns:
        The cancellation acknowledgement, or an error payload for missing input.
    """
    booking_reference = (booking_reference or "").strip()
    if not booking_reference:
        return {
            "error": "MISSING_PARAMETER",
            "message": "booking_reference is required.",
        }
    return mock_data.cancel_booking(booking_reference)


# --------------------------------------------------------------------------- #
# Escalation tools
# --------------------------------------------------------------------------- #
@tool
def create_ticket(
    account_number: str, issue_summary: str, category: str, priority: str = "MEDIUM"
) -> Dict[str, Any]:
    """Open a case/ticket for an issue that cannot be resolved by automation.

    Returns ticket_id, status OPEN, category, priority, sla_hours and created_at.

    Args:
        account_number: The AnyCompany account number the ticket relates to.
        issue_summary: A concise summary of the customer's issue.
        category: One of BILLING_DISPUTE, SUPPLY_ISSUE, COMPLAINT, VULNERABILITY
            or OTHER.
        priority: Optional priority: LOW, MEDIUM (default), HIGH or CRITICAL.

    Returns:
        The ticket payload, or an error payload for invalid/missing input.
    """
    account_number = (account_number or "").strip()
    issue_summary = (issue_summary or "").strip()
    category = (category or "").strip()
    priority = (priority or "").strip() or "MEDIUM"

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
        return {
            "error": "MISSING_PARAMETER",
            "message": "account_number, issue_summary and category are all required.",
            "missing": missing,
            "valid_categories": mock_data.VALID_TICKET_CATEGORIES,
            "valid_priorities": mock_data.VALID_PRIORITIES,
        }

    try:
        ticket = mock_data.open_ticket(
            account_number=account_number,
            issue_summary=issue_summary,
            category=category,
            priority=priority,
        )
    except ValueError as error:
        logger.info("create_ticket rejected input: %s", error)
        return {
            "error": "INVALID_PARAMETER",
            "message": str(error),
            "valid_categories": mock_data.VALID_TICKET_CATEGORIES,
            "valid_priorities": mock_data.VALID_PRIORITIES,
        }

    return {
        "ticket_id": ticket["ticket_id"],
        "status": ticket["status"],
        "category": ticket["category"],
        "priority": ticket["priority"],
        "sla_hours": ticket["sla_hours"],
        "created_at": ticket["created_at"],
    }


@tool
def transfer_to_agent(
    ticket_id: str, reason: str, account_number: str = ""
) -> Dict[str, Any]:
    """Hand the conversation off to a human agent via Amazon Connect (simulated).

    Returns transfer_id, ticket_id, queue, assigned_team, estimated_wait_minutes,
    connect_contact_id and a TRANSFERRED status.

    Args:
        ticket_id: The ticket_id created for this issue.
        reason: Why the transfer is needed; used to infer the target queue/team.
        account_number: Optional customer account number for context.

    Returns:
        The transfer payload, or an error payload for missing input.
    """
    ticket_id = (ticket_id or "").strip()
    reason = (reason or "").strip()
    account = (account_number or "").strip() or None

    missing = [
        name
        for name, value in (("ticket_id", ticket_id), ("reason", reason))
        if not value
    ]
    if missing:
        return {
            "error": "MISSING_PARAMETER",
            "message": "ticket_id and reason are both required.",
            "missing": missing,
        }

    transfer = mock_data.simulate_transfer(
        ticket_id=ticket_id, reason=reason, account_number=account
    )
    return {
        "transfer_id": transfer["transfer_id"],
        "ticket_id": transfer["ticket_id"],
        "queue": transfer["queue"],
        "assigned_team": transfer["assigned_team"],
        "estimated_wait_minutes": transfer["estimated_wait_minutes"],
        "connect_contact_id": transfer["connect_contact_id"],
        "status": transfer["status"],
    }
