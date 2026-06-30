"""Supervisor (intent-router) orchestrator agent.

Implements the **agents-as-tools** multi-agent pattern: a single supervisor
agent owns a deterministic ``classify_intent`` tool plus three specialist agents
exposed as callable tools (``billing_specialist``, ``scheduling_specialist``,
``escalation_specialist``). The supervisor classifies each query, routes it to
the right specialist tool, and synthesises a single coherent reply.

This mirrors the original supervisor/router topology, but instead of the
deprecated Bedrock Agents multi-agent collaboration API, the orchestration is
plain Python on the Strands Agents SDK — so it runs in process (Streamlit demo)
and can be deployed to Amazon Bedrock AgentCore Runtime unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

from strands import Agent, tool

from agents.config import RuntimeConfig, build_model, load_config
from agents.specialists import (
    _brandize,
    build_billing_agent,
    build_escalation_agent,
    build_scheduling_agent,
)
from agents.tools import classify_intent

logger: logging.Logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = (
    "# Role\n"
    "You are the AnyCompany Energy customer-service Intent Router and Supervisor "
    "for a UK water/energy supplier. You are the single front door for customer "
    "conversations. Your job is to understand each request, classify its intent, "
    "route it to the correct specialist tool, and stitch the specialists' work "
    "into one coherent, friendly reply.\n\n"
    "# Routing rules\n"
    "Classify every customer turn into BILLING, SCHEDULING, ESCALATION or UNKNOWN, "
    "then route accordingly:\n"
    "- BILLING (balances, payments, tariffs, refunds, direct debit, arrears, "
    "statements) -> call the `billing_specialist` tool.\n"
    "- SCHEDULING (appointments, meter readings, engineer visits, EV charger "
    "installs, reschedule/cancel) -> call the `scheduling_specialist` tool.\n"
    "- ESCALATION (complaints, manager/ombudsman/legal requests, vulnerability, "
    "emergencies such as a major leak or no water, or any explicit request for a "
    "human) -> call the `escalation_specialist` tool.\n"
    "Use the `classify_intent` tool to produce an explicit intent, confidence and "
    "reasoning for the current request; rely on it (together with your own "
    "judgement) to decide routing. If the intent is UNKNOWN or low-confidence, ask "
    "one concise clarifying question before routing.\n\n"
    "# Multi-intent and safety\n"
    "If a request spans multiple intents (e.g. 'my bill is wrong AND I want a "
    "meter reading'), prioritise safety and vulnerability first, then handle the "
    "remaining intents in turn, calling each specialist tool as needed and "
    "combining their results. Any sign of an emergency, vulnerability or a formal "
    "complaint must be routed to the `escalation_specialist` promptly, even if "
    "other intents are present.\n\n"
    "# Tone and synthesis\n"
    "Greet warmly, be concise and reassuring, and never expose internal routing "
    "mechanics, tool names or agent names to the customer. When a specialist "
    "returns information, present it as a single, seamless AnyCompany Energy "
    "response. Always confirm next steps.\n\n"
    "# Compliance and what NOT to do\n"
    "This is a regulated UK utility. Be accurate, do not invent account data, "
    "policies, prices or outcomes, and only act on the account the customer is "
    "authenticated against. Never invent specific contact details (phone numbers, "
    "email addresses, URLs or postal addresses) when relaying a specialist's "
    "answer; only pass on contact specifics that the specialist actually provided "
    "from a tool or the knowledge base. Do not give legal, tax or financial "
    "advice. Do not attempt to perform billing, scheduling or escalation actions "
    "yourself - always delegate to the appropriate specialist tool."
)


def build_supervisor(
    config: Optional[RuntimeConfig] = None, callback_handler: object = None
) -> Agent:
    """Build the supervisor orchestrator agent.

    Args:
        config: Optional runtime config. When omitted, it is loaded from the
            environment via :func:`agents.config.load_config`.
        callback_handler: Optional Strands callback handler. When provided, it is
            wired into the supervisor **and** every specialist so a single
            collector can observe the full nested tool/routing timeline (used by
            the Streamlit demo). ``None`` (default) suppresses console output.

    Returns:
        The supervisor :class:`strands.Agent`, with ``classify_intent`` and the
        three specialist agents wired in as tools.
    """
    config = config or load_config()

    # Build the specialist agents once and close over them in the tool wrappers
    # so each supervisor turn reuses the same configured specialists. The shared
    # callback handler lets the demo capture their internal tool calls too.
    billing_agent = build_billing_agent(config, callback_handler)
    scheduling_agent = build_scheduling_agent(config, callback_handler)
    escalation_agent = build_escalation_agent(config, callback_handler)

    @tool
    def billing_specialist(query: str) -> str:
        """Handle billing queries: account balances, payments, tariffs, refunds,
        direct debit, arrears and billing-policy questions (RAG-grounded).

        Args:
            query: The customer's billing-related request, with any account
                number (ACC-NNNNNN) they provided.

        Returns:
            The billing specialist's answer.
        """
        return str(billing_agent(query))

    @tool
    def scheduling_specialist(query: str) -> str:
        """Handle scheduling queries: check appointment availability, book and
        cancel meter readings, engineer visits, leak investigations and EV
        charger installs.

        Args:
            query: The customer's scheduling-related request (include postcode
                and account number where relevant).

        Returns:
            The scheduling specialist's answer.
        """
        return str(scheduling_agent(query))

    @tool
    def escalation_specialist(query: str) -> str:
        """Handle complaints, disputes, supply emergencies, vulnerable-customer
        situations and human-handoff requests: opens a case and can transfer to a
        human via Amazon Connect.

        Args:
            query: The customer's escalation-related request, with account number
                and any vulnerability/emergency context.

        Returns:
            The escalation specialist's answer.
        """
        return str(escalation_agent(query))

    return Agent(
        model=build_model(config.supervisor_model_id, config, temperature=0.2),
        system_prompt=_brandize(SUPERVISOR_PROMPT),
        tools=[
            classify_intent,
            billing_specialist,
            scheduling_specialist,
            escalation_specialist,
        ],
        callback_handler=callback_handler,
    )


_SUPERVISOR: Optional[Agent] = None


def get_supervisor() -> Agent:
    """Return a lazily-built, process-wide supervisor singleton."""
    global _SUPERVISOR
    if _SUPERVISOR is None:
        _SUPERVISOR = build_supervisor()
    return _SUPERVISOR


def run(query: str) -> str:
    """Convenience: run a single query through the supervisor and return text."""
    return str(get_supervisor()(query))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    prompt = " ".join(sys.argv[1:]) or "What's the balance on ACC-100123?"
    print(run(prompt))
