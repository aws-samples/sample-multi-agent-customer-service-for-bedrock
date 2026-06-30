"""Scheduling tools action-group Lambda for the AnyCompany scheduling agent.

Implements the ``scheduling-tools`` action group with three functions:

    check_availability(appointment_type, postcode, preferred_date?) -> slots JSON
    book_appointment(account_number, slot_id, contact_phone)        -> booking JSON
    cancel_appointment(booking_reference)                           -> cancel JSON

All operations are backed by the shared in-memory mock datastore
(:mod:`mock_data`) packaged in the common Lambda layer. Slot ids, booking
references, and cancellation outcomes are deterministic.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from bedrock_agent import build_response, dispatch
from mock_data import (
    VALID_APPOINTMENT_TYPES,
    cancel_booking,
    create_booking,
    get_account,
    list_appointment_slots,
)

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_availability(parameters: Dict[str, str]) -> str:
    """Return available appointment slots for a type/postcode/date.

    Args:
        parameters: Resolved action-group parameters. Requires
            ``appointment_type`` and ``postcode``; ``preferred_date`` is
            optional (ISO ``YYYY-MM-DD``).

    Returns:
        A compact JSON string containing ``appointment_type``, ``postcode`` and
        a ``slots`` list (each slot has ``slot_id``, ``date``, ``time_window``
        and ``engineer_type``); or an error body for invalid input.
    """
    appointment_type = parameters.get("appointment_type", "").strip()
    postcode = parameters.get("postcode", "").strip()
    preferred_date = parameters.get("preferred_date", "").strip() or None

    if not appointment_type or not postcode:
        logger.warning("check_availability missing required parameters")
        return json.dumps(
            {
                "error": "MISSING_PARAMETER",
                "message": (
                    "appointment_type and postcode are both required for "
                    "check_availability."
                ),
                "valid_appointment_types": VALID_APPOINTMENT_TYPES,
            }
        )

    try:
        slots = list_appointment_slots(appointment_type, postcode, preferred_date)
    except ValueError as error:
        logger.info("check_availability rejected input: %s", error)
        return json.dumps(
            {
                "error": "INVALID_APPOINTMENT_TYPE",
                "message": str(error),
                "valid_appointment_types": VALID_APPOINTMENT_TYPES,
            }
        )

    logger.info(
        "check_availability returned %d slots for %s in %s",
        len(slots),
        appointment_type.upper(),
        postcode.upper(),
    )
    return json.dumps(
        {
            "appointment_type": appointment_type.upper(),
            "postcode": postcode.upper(),
            "slots": slots,
        }
    )


def book_appointment(parameters: Dict[str, str]) -> str:
    """Book a confirmed appointment against a chosen slot.

    Args:
        parameters: Resolved action-group parameters. Requires
            ``account_number``, ``slot_id`` and ``contact_phone``.

    Returns:
        A compact JSON string with ``booking_reference``, ``confirmed_date``,
        ``confirmed_time_window``, ``appointment_type`` and ``status``
        (``CONFIRMED``); or an error body for invalid input.
    """
    account_number = parameters.get("account_number", "").strip()
    slot_id = parameters.get("slot_id", "").strip()
    contact_phone = parameters.get("contact_phone", "").strip()

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
        logger.warning("book_appointment missing parameters: %s", missing)
        return json.dumps(
            {
                "error": "MISSING_PARAMETER",
                "message": (
                    "account_number, slot_id and contact_phone are all "
                    "required for book_appointment."
                ),
                "missing": missing,
            }
        )

    # Confirm the account exists before booking so we don't create orphan
    # appointments. This mirrors a real CRM pre-flight check.
    account = get_account(account_number)
    if account is None:
        logger.info("book_appointment rejected unknown account '%s'", account_number)
        return json.dumps(
            {
                "error": "ACCOUNT_NOT_FOUND",
                "account_number": account_number,
                "message": (
                    f"Cannot book an appointment: account '{account_number}' "
                    "was not found. Please confirm the account number."
                ),
            }
        )

    booking = create_booking(
        account_number=account_number,
        slot_id=slot_id,
        contact_phone=contact_phone,
    )
    result = {
        "booking_reference": booking["booking_reference"],
        "confirmed_date": booking["confirmed_date"],
        "confirmed_time_window": booking["confirmed_time_window"],
        "appointment_type": booking["appointment_type"],
        "status": booking["status"],
    }
    logger.info(
        "book_appointment confirmed %s for account %s",
        booking["booking_reference"],
        account["account_number"],
    )
    return json.dumps(result)


def cancel_appointment(parameters: Dict[str, str]) -> str:
    """Cancel an existing appointment by its booking reference.

    Args:
        parameters: Resolved action-group parameters. Requires
            ``booking_reference``.

    Returns:
        A compact JSON string with ``booking_reference`` and ``status``
        (``CANCELLED``); or an error body when the reference is missing.
    """
    booking_reference = parameters.get("booking_reference", "").strip()
    if not booking_reference:
        logger.warning("cancel_appointment called without booking_reference")
        return json.dumps(
            {
                "error": "MISSING_PARAMETER",
                "message": (
                    "booking_reference is required for cancel_appointment."
                ),
            }
        )

    result = cancel_booking(booking_reference)
    logger.info(
        "cancel_appointment processed %s -> %s",
        result["booking_reference"],
        result["status"],
    )
    return json.dumps(result)


# Map of function name -> handler callable for this action group.
HANDLERS: Dict[str, Any] = {
    "check_availability": check_availability,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point for the scheduling-tools action group.

    Args:
        event: The Bedrock action-group invocation event.
        context: The Lambda runtime context (unused).

    Returns:
        A valid Bedrock action-group response envelope.
    """
    logger.info("scheduling_tools received function '%s'", event.get("function"))
    try:
        return dispatch(event, HANDLERS)
    except Exception as error:  # noqa: BLE001 - never crash the agent.
        logger.exception("Unhandled error in scheduling_tools: %s", error)
        return build_response(
            event,
            json.dumps(
                {
                    "error": "SCHEDULING_TOOL_ERROR",
                    "message": "Scheduling operation failed unexpectedly.",
                }
            ),
        )
