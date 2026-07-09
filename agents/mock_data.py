"""In-memory mock datastore for the AnyCompany Energy customer-service demo.

This module holds the deterministic business logic (accounts, appointment slot
ids, tickets, simulated Amazon Connect transfers) imported by the Strands
``@tool`` functions in :mod:`agents.tools`. The logic is unchanged from the
original sample.

Everything here is deliberately *deterministic*: booking references, ticket
ids, and transfer/contact ids are derived from the function inputs plus a
monotonic module-level counter, so the demo behaves predictably and the
``testing-guide.md`` scenarios always reproduce the same results.

There is intentionally no persistence and no randomness. The "datastore" is a
set of module-level dictionaries seeded with three fixed sample accounts that
the rest of the project (CDK outputs, demo script, knowledge base) depends on.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Fixed sample accounts. The demo script depends on these EXACT values.
# ---------------------------------------------------------------------------
ACCOUNTS: Dict[str, Dict[str, Any]] = {
    "ACC-100123": {
        "account_number": "ACC-100123",
        "name": "Sarah Thompson",
        "tariff": "Economy 7",
        "balance_gbp": 142.50,
        "status": "ACTIVE",
        "due_date": "2026-07-01",
        "last_payment": {"amount_gbp": 120.00, "date": "2026-05-15"},
        "postcode": "M14 5TG",
        "on_payment_plan": False,
        "vulnerable": False,
    },
    "ACC-100456": {
        "account_number": "ACC-100456",
        "name": "James Okafor",
        "tariff": "Standard Variable",
        "balance_gbp": 0.00,
        "status": "ACTIVE",
        "due_date": "2026-07-03",
        "last_payment": {"amount_gbp": 89.40, "date": "2026-06-01"},
        "postcode": "LS6 2AB",
        "on_payment_plan": True,
        "vulnerable": False,
    },
    "ACC-100789": {
        "account_number": "ACC-100789",
        "name": "Priya Patel",
        "tariff": "EV Smart",
        "balance_gbp": 318.20,
        "status": "ARREARS",
        "due_date": "2026-06-10",
        "last_payment": {"amount_gbp": 45.00, "date": "2026-04-20"},
        "postcode": "BS3 4XY",
        "on_payment_plan": False,
        "vulnerable": True,
    },
}

# Valid appointment types accepted by the scheduling action group.
VALID_APPOINTMENT_TYPES: List[str] = [
    "METER_READING",
    "ENGINEER_VISIT",
    "LEAK_INVESTIGATION",
    "EV_CHARGER_INSTALL",
]

# Maps each appointment type to the engineer skill that services it.
ENGINEER_TYPE_BY_APPOINTMENT: Dict[str, str] = {
    "METER_READING": "Meter Technician",
    "ENGINEER_VISIT": "Field Engineer",
    "LEAK_INVESTIGATION": "Leak Detection Specialist",
    "EV_CHARGER_INSTALL": "EV Installation Engineer",
}

# Short, collision-free codes for each appointment type. These are embedded in
# the slot id so a booking can recover the appointment type (and therefore the
# correct engineer) without any shared state between the (stateless) availability
# and booking Lambda invocations.
APPOINTMENT_TYPE_CODES: Dict[str, str] = {
    "METER_READING": "MR",
    "ENGINEER_VISIT": "EV",
    "LEAK_INVESTIGATION": "LI",
    "EV_CHARGER_INSTALL": "EC",
}
CODE_TO_APPOINTMENT_TYPE: Dict[str, str] = {
    code: appt for appt, code in APPOINTMENT_TYPE_CODES.items()
}

# Standard daily appointment windows offered to customers.
TIME_WINDOWS: List[str] = [
    "08:00-10:00",
    "10:00-12:00",
    "13:00-15:00",
    "15:00-17:00",
]

# Valid escalation ticket categories and their default priorities.
VALID_TICKET_CATEGORIES: List[str] = [
    "BILLING_DISPUTE",
    "SUPPLY_ISSUE",
    "COMPLAINT",
    "VULNERABILITY",
    "OTHER",
]

VALID_PRIORITIES: List[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Service-level-agreement response targets, in hours, derived from the
# category/priority combination. CRITICAL or VULNERABILITY escalations are the
# most urgent (2 hours), matching the policies documented in the knowledge base.
SLA_HOURS_BY_PRIORITY: Dict[str, int] = {
    "CRITICAL": 2,
    "HIGH": 8,
    "MEDIUM": 24,
    "LOW": 72,
}

# In-memory "tables" that mutate over the life of a warm Lambda container.
BOOKINGS: Dict[str, Dict[str, Any]] = {}
TICKETS: Dict[str, Dict[str, Any]] = {}
TRANSFERS: Dict[str, Dict[str, Any]] = {}

# Monotonic counter that guarantees deterministic-yet-unique reference numbers
# within a single warm container while keeping derivation reproducible.
_id_counter = itertools.count(1)


# ---------------------------------------------------------------------------
# Internal deterministic id helpers.
# ---------------------------------------------------------------------------
def _stable_digits(seed: str, length: int) -> str:
    """Derive a fixed-length numeric string deterministically from a seed.

    Args:
        seed: Arbitrary input text to hash.
        length: Number of decimal digits to produce.

    Returns:
        A zero-padded numeric string of exactly ``length`` digits.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    numeric = int(digest, 16) % (10**length)
    return str(numeric).zfill(length)


def _stable_hex(seed: str, length: int) -> str:
    """Derive a fixed-length lowercase hex string deterministically from a seed.

    Args:
        seed: Arbitrary input text to hash.
        length: Number of hex characters to produce.

    Returns:
        A hex string of exactly ``length`` characters.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return digest[:length]


def _next_sequence() -> int:
    """Return the next value from the monotonic module counter.

    Returns:
        A strictly increasing integer for the life of the container.
    """
    return next(_id_counter)


def _parse_iso_date(value: Optional[str]) -> date:
    """Parse an ISO ``YYYY-MM-DD`` date, falling back to a safe default.

    Args:
        value: Optional ISO date string supplied by the caller.

    Returns:
        The parsed ``date``. When ``value`` is missing or invalid, the next
        business-friendly date (two days from the demo's "today") is returned so
        availability lookups always succeed.
    """
    default = date(2026, 6, 18) + timedelta(days=2)
    if not value:
        return default
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        logger.warning("Could not parse preferred_date '%s'; using default", value)
        return default


# ---------------------------------------------------------------------------
# Account helpers (used by billing_tools).
# ---------------------------------------------------------------------------
def get_account(account_number: str) -> Optional[Dict[str, Any]]:
    """Return a deep-ish copy of the account profile, or ``None`` if unknown.

    Args:
        account_number: The customer account identifier (e.g. ``ACC-100123``).

    Returns:
        A copy of the account record, or ``None`` when the account does not
        exist. A copy is returned so callers cannot mutate the seed data.
    """
    normalized = (account_number or "").strip().upper()
    account = ACCOUNTS.get(normalized)
    if account is None:
        # Do not log the account identifier (PII) in clear text.
        logger.info("Account lookup miss (account not found)")
        return None
    # Shallow copy plus a copy of the nested last_payment dict is sufficient
    # because the record otherwise contains only immutable scalars.
    record = dict(account)
    record["last_payment"] = dict(account["last_payment"])
    return record


# ---------------------------------------------------------------------------
# Scheduling helpers (used by scheduling_tools).
# ---------------------------------------------------------------------------
def list_appointment_slots(
    appointment_type: str,
    postcode: str,
    preferred_date: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Generate a deterministic list of appointment slots.

    Slots are derived from the appointment type, postcode, and preferred date so
    repeated calls with the same inputs always yield the same slot ids. Three
    consecutive days (starting from the preferred date) are offered, each with a
    small selection of time windows.

    Args:
        appointment_type: One of :data:`VALID_APPOINTMENT_TYPES`.
        postcode: The customer's postcode (used in slot id derivation).
        preferred_date: Optional ISO date the customer would prefer.

    Returns:
        A list of slot dictionaries, each containing ``slot_id``, ``date``,
        ``time_window`` and ``engineer_type``.

    Raises:
        ValueError: If ``appointment_type`` is not a recognised type.
    """
    appt = (appointment_type or "").strip().upper()
    if appt not in VALID_APPOINTMENT_TYPES:
        raise ValueError(
            f"Unsupported appointment_type '{appointment_type}'. "
            f"Valid types: {', '.join(VALID_APPOINTMENT_TYPES)}"
        )

    clean_postcode = (postcode or "").strip().upper()
    engineer_type = ENGINEER_TYPE_BY_APPOINTMENT[appt]
    start = _parse_iso_date(preferred_date)

    slots: List[Dict[str, str]] = []
    # Offer two windows per day across three consecutive days.
    for day_offset in range(3):
        slot_date = start + timedelta(days=day_offset)
        # Stagger which two windows appear so the schedule looks realistic.
        window_indices = [
            day_offset % len(TIME_WINDOWS),
            (day_offset + 2) % len(TIME_WINDOWS),
        ]
        for window_index in window_indices:
            window = TIME_WINDOWS[window_index]
            slot_id = _encode_slot(appt, clean_postcode, slot_date, window_index)
            slots.append(
                {
                    "slot_id": slot_id,
                    "date": slot_date.isoformat(),
                    "time_window": window,
                    "engineer_type": engineer_type,
                }
            )

    # Do not log location/PII data (postcodes) even in a sample — demonstrates the clear-text-logging control.
    logger.info(
        "Generated %d slots for %s starting %s",
        len(slots),
        appt,
        start.isoformat(),
    )
    return slots


def _encode_slot(
    appointment_type: str,
    postcode: str,
    slot_date: date,
    window_index: int,
) -> str:
    """Build a self-describing slot id that encodes the offered slot's details.

    The slot id carries the appointment-type code, the date and the time-window
    index directly, with a short hash tail (over the postcode + fields) for
    realism and uniqueness. Encoding the booking details *into* the id means a
    later :func:`create_booking` call can recover the exact date, window and
    appointment type the customer selected without any shared state between the
    (stateless, possibly cold-started) availability and booking invocations.

    Args:
        appointment_type: A validated appointment type (key of
            :data:`APPOINTMENT_TYPE_CODES`).
        postcode: The cleaned postcode (folded into the hash tail only).
        slot_date: The date offered for this slot.
        window_index: Index into :data:`TIME_WINDOWS` for this slot's window.

    Returns:
        A slot id of the form ``SLOT-<TYPECODE>-<YYYYMMDD>-<windowIdx>-<hash6>``,
        e.g. ``SLOT-MR-20260620-1-3a9f0c``.
    """
    type_code = APPOINTMENT_TYPE_CODES[appointment_type]
    date_token = slot_date.strftime("%Y%m%d")
    seed = f"{type_code}|{postcode}|{date_token}|{window_index}"
    tail = _stable_hex(seed, 6)
    return f"SLOT-{type_code}-{date_token}-{window_index}-{tail}"


def _decode_slot(slot_id: str) -> Dict[str, str]:
    """Recover the date, time window and appointment type from a slot id.

    Parses the self-describing slot id produced by :func:`_encode_slot`. If the
    id does not match the expected structure (e.g. an old-format or
    customer-mangled value), the function falls back to a deterministic hash so a
    booking confirmation can still be produced rather than failing outright.

    Args:
        slot_id: The slot identifier returned by :func:`list_appointment_slots`.

    Returns:
        A dictionary with ``confirmed_date``, ``confirmed_time_window`` and
        ``appointment_type`` (the latter ``None`` if it could not be decoded).
    """
    parts = (slot_id or "").strip().upper().split("-")
    # Expected: ["SLOT", <TYPECODE>, <YYYYMMDD>, <windowIdx>, <hash6>]
    if len(parts) == 5 and parts[0] == "SLOT":
        type_code, date_token, window_token = parts[1], parts[2], parts[3]
        try:
            parsed_date = datetime.strptime(date_token, "%Y%m%d").date()
            window_index = int(window_token)
            if 0 <= window_index < len(TIME_WINDOWS):
                return {
                    "confirmed_date": parsed_date.isoformat(),
                    "confirmed_time_window": TIME_WINDOWS[window_index],
                    "appointment_type": CODE_TO_APPOINTMENT_TYPE.get(type_code),
                }
        except (ValueError, KeyError):
            logger.warning("Slot id '%s' did not decode cleanly; using fallback", slot_id)

    # Fallback for unrecognised ids: derive stable-but-synthetic details so the
    # booking flow degrades gracefully instead of raising.
    base = date(2026, 6, 18)
    day_offset = int(_stable_digits(slot_id + "|day", 2)) % 14
    window_index = int(_stable_digits(slot_id + "|win", 2)) % len(TIME_WINDOWS)
    return {
        "confirmed_date": (base + timedelta(days=day_offset)).isoformat(),
        "confirmed_time_window": TIME_WINDOWS[window_index],
        "appointment_type": None,
    }


def create_booking(
    account_number: str,
    slot_id: str,
    contact_phone: str,
    appointment_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a confirmed appointment booking.

    Args:
        account_number: The customer account the booking is for.
        slot_id: The chosen slot identifier.
        contact_phone: A contact phone number for day-of-visit coordination.
        appointment_type: Optional appointment type for the confirmation record.
            When omitted, it is inferred as a generic ``ENGINEER_VISIT``.

    Returns:
        A booking confirmation dictionary containing ``booking_reference``,
        ``confirmed_date``, ``confirmed_time_window``, ``appointment_type`` and
        ``status`` (always ``CONFIRMED``).
    """
    sequence = _next_sequence()
    seed = f"{account_number}|{slot_id}|{contact_phone}|{sequence}"
    booking_reference = f"APPT-{_stable_digits(seed, 6)}"
    decoded = _decode_slot(slot_id)

    # Prefer an explicitly supplied appointment_type; otherwise use the type
    # encoded in the slot id; only then fall back to a generic ENGINEER_VISIT.
    appt = (appointment_type or decoded.get("appointment_type") or "ENGINEER_VISIT").strip().upper()
    if appt not in VALID_APPOINTMENT_TYPES:
        appt = "ENGINEER_VISIT"

    booking = {
        "booking_reference": booking_reference,
        "account_number": (account_number or "").strip().upper(),
        "slot_id": slot_id,
        "contact_phone": contact_phone,
        "confirmed_date": decoded["confirmed_date"],
        "confirmed_time_window": decoded["confirmed_time_window"],
        "appointment_type": appt,
        "status": "CONFIRMED",
    }
    BOOKINGS[booking_reference] = booking
    # Do not log the account identifier (PII); the booking reference is sufficient.
    logger.info(
        "Created booking %s (slot %s)",
        booking_reference,
        slot_id,
    )
    return dict(booking)


def cancel_booking(booking_reference: str) -> Dict[str, Any]:
    """Cancel an appointment booking.

    Args:
        booking_reference: The booking reference to cancel.

    Returns:
        A dictionary with ``booking_reference``, ``status`` (always
        ``CANCELLED``) and a human-readable ``message``. The operation is
        idempotent: cancelling an unknown reference still returns a valid
        cancellation acknowledgement so the agent can respond gracefully.
    """
    ref = (booking_reference or "").strip().upper()
    existing = BOOKINGS.get(ref)
    if existing is not None:
        existing["status"] = "CANCELLED"
        message = f"Appointment {ref} has been cancelled."
    else:
        # We never created a real record for this reference (e.g. cold start),
        # but the customer-facing outcome is the same: it is not booked.
        message = (
            f"Appointment {ref} has been cancelled. If you did not have an "
            "active booking under this reference, no action was required."
        )
    logger.info("Cancelled booking %s", ref)
    return {
        "booking_reference": ref,
        "status": "CANCELLED",
        "message": message,
    }


# ---------------------------------------------------------------------------
# Escalation helpers (used by escalation_tools).
# ---------------------------------------------------------------------------
def open_ticket(
    account_number: str,
    issue_summary: str,
    category: str,
    priority: str = "MEDIUM",
) -> Dict[str, Any]:
    """Open a customer-service escalation ticket.

    Args:
        account_number: The customer account the ticket relates to.
        issue_summary: A short free-text description of the issue.
        category: One of :data:`VALID_TICKET_CATEGORIES`.
        priority: One of :data:`VALID_PRIORITIES`. Defaults to ``MEDIUM``.

    Returns:
        A ticket record dictionary with ``ticket_id``, ``status`` (``OPEN``),
        ``category``, ``priority``, ``sla_hours``, ``created_at`` and the
        captured ``account_number`` and ``issue_summary``.

    Raises:
        ValueError: If ``category`` or ``priority`` is not recognised.
    """
    normalized_category = (category or "").strip().upper()
    if normalized_category not in VALID_TICKET_CATEGORIES:
        raise ValueError(
            f"Unsupported category '{category}'. "
            f"Valid categories: {', '.join(VALID_TICKET_CATEGORIES)}"
        )

    normalized_priority = (priority or "MEDIUM").strip().upper()
    if normalized_priority not in VALID_PRIORITIES:
        raise ValueError(
            f"Unsupported priority '{priority}'. "
            f"Valid priorities: {', '.join(VALID_PRIORITIES)}"
        )

    # VULNERABILITY cases are always treated as the most urgent (2-hour SLA),
    # in line with AnyCompany's vulnerable-customer policy, regardless of the
    # priority the caller supplied.
    if normalized_category == "VULNERABILITY":
        sla_hours = SLA_HOURS_BY_PRIORITY["CRITICAL"]
    else:
        sla_hours = SLA_HOURS_BY_PRIORITY[normalized_priority]

    sequence = _next_sequence()
    seed = f"{account_number}|{normalized_category}|{issue_summary}|{sequence}"
    ticket_id = f"TKT-{_stable_digits(seed, 6)}"
    created_at = datetime(2026, 6, 18, 9, 0, 0).isoformat() + "Z"

    ticket = {
        "ticket_id": ticket_id,
        "account_number": (account_number or "").strip().upper(),
        "issue_summary": issue_summary,
        "category": normalized_category,
        "priority": normalized_priority,
        "sla_hours": sla_hours,
        "status": "OPEN",
        "created_at": created_at,
    }
    TICKETS[ticket_id] = ticket
    logger.info(
        "Opened ticket %s (category=%s, priority=%s, sla=%dh)",
        ticket_id,
        normalized_category,
        normalized_priority,
        sla_hours,
    )
    return dict(ticket)


# Routing table used to infer an Amazon Connect queue/team from the free-text
# transfer reason. The first matching keyword wins; ordering matters.
_TRANSFER_ROUTING: List[Dict[str, Any]] = [
    {
        "keywords": ["vulnerable", "vulnerability", "priority service", "elderly", "medical"],
        "queue": "PrioritySupportQueue",
        "assigned_team": "Priority Services Team",
        "estimated_wait_minutes": 2,
    },
    {
        "keywords": ["leak", "no water", "supply", "outage", "emergency", "flood", "burst"],
        "queue": "EmergencySupplyQueue",
        "assigned_team": "Emergency Field Operations",
        "estimated_wait_minutes": 1,
    },
    {
        "keywords": ["legal", "ombudsman", "solicitor", "regulator", "complaint", "escalate", "manager"],
        "queue": "ComplaintsQueue",
        "assigned_team": "Customer Relations Team",
        "estimated_wait_minutes": 6,
    },
    {
        "keywords": ["bill", "billing", "charge", "refund", "dispute", "payment", "tariff", "direct debit"],
        "queue": "BillingQueue",
        "assigned_team": "Billing Specialists Team",
        "estimated_wait_minutes": 4,
    },
]

# Default queue when no routing keyword matches the reason text.
_DEFAULT_TRANSFER_ROUTE: Dict[str, Any] = {
    "queue": "GeneralSupportQueue",
    "assigned_team": "General Support Team",
    "estimated_wait_minutes": 8,
}


def simulate_transfer(
    ticket_id: str,
    reason: str,
    account_number: Optional[str] = None,
) -> Dict[str, Any]:
    """Simulate an Amazon Connect warm transfer to a human agent.

    The destination queue and team are inferred from keywords in ``reason``,
    mimicking a contact-flow routing decision. A deterministic
    ``connect_contact_id`` is derived so the simulated handoff is reproducible.

    Args:
        ticket_id: The escalation ticket id associated with this transfer.
        reason: Free-text reason for the transfer (drives queue selection).
        account_number: Optional customer account number for context.

    Returns:
        A transfer record dictionary with ``transfer_id``, ``ticket_id``,
        ``queue``, ``assigned_team``, ``estimated_wait_minutes``,
        ``connect_contact_id`` and ``status`` (``TRANSFERRED``).
    """
    reason_lower = (reason or "").lower()
    route = _DEFAULT_TRANSFER_ROUTE
    for candidate in _TRANSFER_ROUTING:
        if any(keyword in reason_lower for keyword in candidate["keywords"]):
            route = candidate
            break

    sequence = _next_sequence()
    seed = f"{ticket_id}|{reason}|{account_number or ''}|{sequence}"
    transfer_id = f"XFER-{_stable_digits(seed, 6)}"
    # Amazon Connect contact ids are typically UUID-like; we produce a
    # deterministic 8-char hex token as required by the contract.
    connect_contact_id = f"CT-{_stable_hex(seed, 8)}"

    transfer = {
        "transfer_id": transfer_id,
        "ticket_id": (ticket_id or "").strip().upper(),
        "account_number": (account_number or "").strip().upper() or None,
        "reason": reason,
        "queue": route["queue"],
        "assigned_team": route["assigned_team"],
        "estimated_wait_minutes": route["estimated_wait_minutes"],
        "connect_contact_id": connect_contact_id,
        "status": "TRANSFERRED",
    }
    TRANSFERS[transfer_id] = transfer
    logger.info(
        "Simulated Connect transfer %s for ticket %s -> %s (queue=%s)",
        transfer_id,
        transfer["ticket_id"],
        route["assigned_team"],
        route["queue"],
    )
    return dict(transfer)
