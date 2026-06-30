"""Intent Router action-group Lambda for the AnyCompany supervisor agent.

Implements the ``intent-classification`` action group with a single function,
``classify_intent``. It uses a transparent keyword/heuristic scoring model to
classify an inbound customer query into one of four intents and recommends the
specialist collaborator agent that should handle it.

Intents:
    BILLING     -> anycompany-billing-agent
    SCHEDULING  -> anycompany-scheduling-agent
    ESCALATION  -> anycompany-escalation-agent
    UNKNOWN     -> (no confident routing; ask the customer to clarify)

The supervisor (multi-agent collaboration SUPERVISOR_ROUTER) can use this as an
explicit pre-routing signal; the function is also useful for analytics and for
deterministic behaviour in the demo.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from bedrock_agent import build_response, dispatch

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Minimum normalized score required to commit to an intent. Anything below this
# threshold is treated as UNKNOWN so the supervisor can ask for clarification.
CONFIDENCE_THRESHOLD: float = 0.34

# Maps each intent to the collaborator agent name configured in the CDK stack.
RECOMMENDED_AGENT_BY_INTENT: Dict[str, str] = {
    "BILLING": "anycompany-billing-agent",
    "SCHEDULING": "anycompany-scheduling-agent",
    "ESCALATION": "anycompany-escalation-agent",
    "UNKNOWN": "anycompany-intent-router",
}

# Weighted keyword vocabularies per intent. Multi-word phrases are checked as
# substrings; single tokens are matched on word boundaries. Higher weights mark
# strongly indicative terms (e.g. "ombudsman" is a near-certain escalation).
KEYWORD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "BILLING": {
        "bill": 1.0,
        "billing": 1.0,
        "balance": 1.0,
        "payment": 1.0,
        "pay": 0.8,
        "tariff": 1.2,
        "refund": 1.2,
        "direct debit": 1.5,
        "charge": 1.0,
        "charged": 1.0,
        "invoice": 1.0,
        "statement": 0.8,
        "overcharged": 1.3,
        "credit": 0.8,
        "arrears": 1.0,
        "owe": 1.0,
        "owed": 1.0,
        "account balance": 1.4,
        "how much": 0.7,
        "economy 7": 1.2,
        "standard variable": 1.2,
        "ev smart": 1.2,
    },
    "SCHEDULING": {
        "appointment": 1.3,
        "book": 1.2,
        "booking": 1.2,
        "schedule": 1.2,
        "reschedule": 1.4,
        "cancel": 1.1,
        "meter reading": 1.5,
        "meter": 0.9,
        "reading": 0.7,
        "engineer": 1.3,
        "visit": 1.0,
        "slot": 1.0,
        "availability": 1.1,
        "available": 0.8,
        "install": 1.0,
        "installation": 1.0,
        "ev charger": 1.4,
        "charger": 1.0,
        "leak investigation": 1.4,
        "come round": 1.0,
        "send someone": 1.1,
    },
    "ESCALATION": {
        "complaint": 1.4,
        "complain": 1.3,
        "escalate": 1.5,
        "escalation": 1.5,
        "manager": 1.3,
        "supervisor": 1.2,
        "legal": 1.5,
        "ombudsman": 1.8,
        "solicitor": 1.5,
        "regulator": 1.4,
        "vulnerable": 1.6,
        "vulnerability": 1.6,
        "emergency": 1.6,
        "leak": 1.3,
        "no water": 1.7,
        "burst": 1.4,
        "flood": 1.4,
        "flooding": 1.4,
        "dangerous": 1.4,
        "unacceptable": 1.2,
        "furious": 1.2,
        "disgusted": 1.2,
        "medical": 1.3,
        "elderly": 1.2,
    },
}


def _score_text(text: str) -> Dict[str, float]:
    """Compute a raw keyword-match score for each intent category.

    Multi-word keywords are matched as case-insensitive substrings; single
    tokens are matched on word boundaries to avoid spurious partial matches
    (e.g. "pay" should not fire inside "payment" twice, and "leak" should not
    fire inside unrelated words).

    Args:
        text: The customer query text.

    Returns:
        A mapping of intent name to its accumulated weighted score.
    """
    lowered = (text or "").lower()
    scores: Dict[str, float] = {intent: 0.0 for intent in KEYWORD_WEIGHTS}

    for intent, vocabulary in KEYWORD_WEIGHTS.items():
        for keyword, weight in vocabulary.items():
            if " " in keyword:
                if keyword in lowered:
                    scores[intent] += weight
            else:
                # Word-boundary match for single tokens.
                pattern = r"\b" + re.escape(keyword) + r"\b"
                if re.search(pattern, lowered):
                    scores[intent] += weight
    return scores


def _build_reasoning(
    intent: str, scores: Dict[str, float], matched: Dict[str, List[str]]
) -> str:
    """Produce a short, human-readable explanation of the routing decision.

    Args:
        intent: The selected intent.
        scores: The raw per-intent scores.
        matched: The matched keywords per intent.

    Returns:
        A concise reasoning string.
    """
    if intent == "UNKNOWN":
        return (
            "No intent category accumulated enough keyword evidence to exceed "
            "the confidence threshold. The query is ambiguous and the customer "
            "should be asked to clarify what they need help with."
        )
    hits = matched.get(intent, [])
    hit_text = ", ".join(f"'{h}'" for h in hits) if hits else "general phrasing"
    competitors = sorted(
        ((s, name) for name, s in scores.items() if name != intent),
        reverse=True,
    )
    runner_up = competitors[0] if competitors else (0.0, "none")
    return (
        f"Matched {intent} indicators ({hit_text}) with a raw score of "
        f"{scores.get(intent, 0.0):.2f}, ahead of the next category "
        f"'{runner_up[1]}' ({runner_up[0]:.2f}). Routing to the "
        f"{RECOMMENDED_AGENT_BY_INTENT[intent]} specialist."
    )


def _matched_keywords(text: str) -> Dict[str, List[str]]:
    """Return, per intent, the list of keywords that matched the text.

    Args:
        text: The customer query text.

    Returns:
        A mapping of intent name to the list of matched keywords.
    """
    lowered = (text or "").lower()
    matched: Dict[str, List[str]] = {intent: [] for intent in KEYWORD_WEIGHTS}
    for intent, vocabulary in KEYWORD_WEIGHTS.items():
        for keyword in vocabulary:
            if " " in keyword:
                if keyword in lowered:
                    matched[intent].append(keyword)
            else:
                if re.search(r"\b" + re.escape(keyword) + r"\b", lowered):
                    matched[intent].append(keyword)
    return matched


def _classify(query: str) -> Tuple[str, float, str]:
    """Classify a query into an intent with a normalized confidence score.

    Confidence is the winning intent's share of the total score across all
    categories. When the total evidence is zero, or the winning confidence is
    below :data:`CONFIDENCE_THRESHOLD`, the result is ``UNKNOWN``.

    Args:
        query: The customer query text.

    Returns:
        A tuple of ``(intent, confidence, reasoning)``.
    """
    scores = _score_text(query)
    matched = _matched_keywords(query)
    total = sum(scores.values())

    if total <= 0.0:
        reasoning = _build_reasoning("UNKNOWN", scores, matched)
        return "UNKNOWN", 0.0, reasoning

    # `max` over the scores dict returns the FIRST key with the highest score,
    # and the dict preserves KEYWORD_WEIGHTS declaration order (BILLING,
    # SCHEDULING, ESCALATION). So ties break deterministically in that order —
    # BILLING wins a tie because it is declared first.
    top_intent = max(scores, key=lambda name: scores[name])
    confidence = round(scores[top_intent] / total, 4)

    if confidence < CONFIDENCE_THRESHOLD:
        reasoning = _build_reasoning("UNKNOWN", scores, matched)
        return "UNKNOWN", confidence, reasoning

    reasoning = _build_reasoning(top_intent, scores, matched)
    return top_intent, confidence, reasoning


def classify_intent(parameters: Dict[str, str]) -> str:
    """Classify a customer query and recommend the specialist agent.

    Args:
        parameters: Resolved action-group parameters. Requires ``query``.

    Returns:
        A compact JSON string with ``intent``, ``confidence``,
        ``recommended_agent`` and ``reasoning``.
    """
    query = parameters.get("query", "").strip()
    if not query:
        logger.warning("classify_intent invoked without a 'query' parameter")
        return json.dumps(
            {
                "intent": "UNKNOWN",
                "confidence": 0.0,
                "recommended_agent": RECOMMENDED_AGENT_BY_INTENT["UNKNOWN"],
                "reasoning": (
                    "No query text was supplied, so the intent cannot be "
                    "determined. Ask the customer what they need help with."
                ),
            }
        )

    intent, confidence, reasoning = _classify(query)
    result = {
        "intent": intent,
        "confidence": confidence,
        "recommended_agent": RECOMMENDED_AGENT_BY_INTENT[intent],
        "reasoning": reasoning,
    }
    logger.info(
        "Classified query as %s (confidence=%.2f)", intent, confidence
    )
    return json.dumps(result)


# Map of function name -> handler callable for this action group.
HANDLERS: Dict[str, Any] = {
    "classify_intent": classify_intent,
}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point for the intent-classification action group.

    Args:
        event: The Bedrock action-group invocation event.
        context: The Lambda runtime context (unused).

    Returns:
        A valid Bedrock action-group response envelope.
    """
    logger.info("intent_router received function '%s'", event.get("function"))
    try:
        return dispatch(event, HANDLERS)
    except Exception as error:  # noqa: BLE001 - never crash the agent.
        logger.exception("Unhandled error in intent_router: %s", error)
        return build_response(
            event,
            json.dumps(
                {
                    "error": "ROUTER_ERROR",
                    "message": "Intent classification failed unexpectedly.",
                }
            ),
        )
