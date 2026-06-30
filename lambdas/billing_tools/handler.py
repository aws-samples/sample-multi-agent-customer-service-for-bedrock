"""Billing tools action-group Lambda for the AnyCompany billing agent.

Implements the ``billing-tools`` action group with two functions:

    lookup_account(account_number)  -> account profile JSON
    check_balance(account_number)   -> balance / due date / last payment JSON

Both read from the shared in-memory mock datastore (:mod:`mock_data`) packaged
in the common Lambda layer. The billing agent is also attached to the Bedrock
Knowledge Base, so these tools provide the structured account facts that
complement RAG answers about tariffs and policies.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from bedrock_agent import build_response, dispatch
from mock_data import get_account

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _account_not_found_body(account_number: str) -> str:
    """Build a consistent JSON body for an unknown account.

    Args:
        account_number: The account number that could not be found.

    Returns:
        A compact JSON string describing the miss, including the known sample
        account numbers so the agent can guide the customer.
    """
    return json.dumps(
        {
            "error": "ACCOUNT_NOT_FOUND",
            "account_number": account_number,
            "message": (
                f"No account was found matching '{account_number}'. Please "
                "confirm the account number, which has the format ACC-NNNNNN."
            ),
            "known_sample_accounts": ["ACC-100123", "ACC-100456", "ACC-100789"],
        }
    )


def lookup_account(parameters: Dict[str, str]) -> str:
    """Return a customer's account profile.

    Args:
        parameters: Resolved action-group parameters. Requires
            ``account_number``.

    Returns:
        A compact JSON string with ``account_number``, ``name``, ``tariff``,
        ``status``, ``postcode`` and ``on_payment_plan``; or an error body when
        the account does not exist.
    """
    account_number = parameters.get("account_number", "").strip()
    if not account_number:
        logger.warning("lookup_account called without account_number")
        return json.dumps(
            {
                "error": "MISSING_PARAMETER",
                "message": "account_number is required for lookup_account.",
            }
        )

    account = get_account(account_number)
    if account is None:
        logger.info("lookup_account miss for '%s'", account_number)
        return _account_not_found_body(account_number)

    profile = {
        "account_number": account["account_number"],
        "name": account["name"],
        "tariff": account["tariff"],
        "status": account["status"],
        "postcode": account["postcode"],
        "on_payment_plan": account["on_payment_plan"],
    }
    logger.info("lookup_account success for '%s'", account["account_number"])
    return json.dumps(profile)


def check_balance(parameters: Dict[str, str]) -> str:
    """Return a customer's current balance, due date and last payment.

    Args:
        parameters: Resolved action-group parameters. Requires
            ``account_number``.

    Returns:
        A compact JSON string with ``account_number``, ``balance_gbp``,
        ``due_date``, ``last_payment`` and ``status``; or an error body when the
        account does not exist.
    """
    account_number = parameters.get("account_number", "").strip()
    if not account_number:
        logger.warning("check_balance called without account_number")
        return json.dumps(
            {
                "error": "MISSING_PARAMETER",
                "message": "account_number is required for check_balance.",
            }
        )

    account = get_account(account_number)
    if account is None:
        logger.info("check_balance miss for '%s'", account_number)
        return _account_not_found_body(account_number)

    balance = {
        "account_number": account["account_number"],
        "balance_gbp": account["balance_gbp"],
        "due_date": account["due_date"],
        "last_payment": account["last_payment"],
        "status": account["status"],
    }
    logger.info(
        "check_balance success for '%s' (balance=%.2f, status=%s)",
        account["account_number"],
        account["balance_gbp"],
        account["status"],
    )
    return json.dumps(balance)


# Map of function name -> handler callable for this action group.
HANDLERS: Dict[str, Any] = {
    "lookup_account": lookup_account,
    "check_balance": check_balance,
}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point for the billing-tools action group.

    Args:
        event: The Bedrock action-group invocation event.
        context: The Lambda runtime context (unused).

    Returns:
        A valid Bedrock action-group response envelope.
    """
    logger.info("billing_tools received function '%s'", event.get("function"))
    try:
        return dispatch(event, HANDLERS)
    except Exception as error:  # noqa: BLE001 - never crash the agent.
        logger.exception("Unhandled error in billing_tools: %s", error)
        return build_response(
            event,
            json.dumps(
                {
                    "error": "BILLING_TOOL_ERROR",
                    "message": "Billing lookup failed unexpectedly.",
                }
            ),
        )
