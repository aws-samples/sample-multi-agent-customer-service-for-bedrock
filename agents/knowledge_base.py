"""Knowledge Base retrieval for the billing and escalation specialists.

The specialists search the Bedrock Knowledge Base with a ``retrieve`` tool built
on the Strands SDK's :class:`BedrockKnowledgeBaseStore`, which replaces the
deprecated ``retrieve`` tool from ``strands-agents-tools``. The tool keeps that
tool's name, score floor (0.4) and result count (10), and adds two things:

* **Errors reach the model.** A failed search (wrong id, Region or permission)
  comes back as a tool error rather than an empty result, so "the Knowledge Base
  has nothing on this" and "the Knowledge Base call failed" stay distinct.
* **Retrieved text is screened.** Guardrails applied to Converse don't evaluate
  tool results, so when a guardrail is configured each retrieved passage, with
  its source, goes through ``ApplyGuardrail`` before the model sees it. A passage
  the guardrail blocks (for example an injected instruction) is withheld, masked
  text replaces the original, and if ``ApplyGuardrail`` itself fails the results
  are withheld.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from agents.config import RuntimeConfig

logger: logging.Logger = logging.getLogger(__name__)

# The deprecated strands-agents-tools ``retrieve`` defaults, kept so answers are
# grounded in the same passages as before.
MIN_SCORE: float = 0.4
MAX_RESULTS: int = 10

TOOL_DESCRIPTION = (
    "Search the company Knowledge Base (pricing and tariff guides, billing FAQs, "
    "policies, service procedures and published contact details) and return the "
    "most relevant passages with their scores and sources. Use it to ground any "
    "answer about pricing, billing policy, procedures or contact details."
)
WITHHELD_NOTICE = (
    "{count} retrieved passage(s) were withheld by the content guardrail. Do not "
    "answer from memory what those passages would have covered; offer to escalate "
    "instead."
)


def _policy_actions(node: Any) -> set:
    """Return every policy ``action`` in an ApplyGuardrail assessment (BLOCKED, ANONYMIZED, ...)."""
    if isinstance(node, dict):
        found = {node["action"]} if isinstance(node.get("action"), str) else set()
        return found.union(*(_policy_actions(value) for value in node.values()))
    if isinstance(node, list):
        return set().union(*(_policy_actions(value) for value in node))
    return set()


class PassageScreener:
    """Screen retrieved passages with ``ApplyGuardrail`` (``source="INPUT"``).

    Each passage is checked in its own call, so one blocked passage doesn't
    withhold the others and a passage is never judged as part of a batch.
    """

    def __init__(self, client: Any, guardrail_id: str, guardrail_version: str) -> None:
        self._client = client
        self._guardrail_id = guardrail_id
        self._guardrail_version = guardrail_version

    def screen(self, text: str) -> Optional[str]:
        """Return the passage as the model may see it: as is, masked, or None if withheld.

        Raises:
            botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError:
                ApplyGuardrail failed; the caller withholds the results.
        """
        response = self._client.apply_guardrail(
            guardrailIdentifier=self._guardrail_id,
            guardrailVersion=self._guardrail_version,
            source="INPUT",
            content=[{"text": {"text": text}}],
        )
        if response.get("action") == "NONE":
            return text
        # Anything else is an intervention. Masked text replaces the original only
        # when the guardrail says it masked and blocked nothing; otherwise the
        # output may be the guardrail's blocked message, so the passage is withheld.
        actions = _policy_actions(response.get("assessments"))
        outputs = response.get("outputs") or []
        masked = outputs[0].get("text") if outputs else None
        if "ANONYMIZED" in actions and "BLOCKED" not in actions and masked:
            return masked
        return None


def _source_uri(metadata: dict) -> str:
    """Return the passage's source document URI, if the Knowledge Base gave one."""
    location = metadata.get("_source_location") or {}
    uri = (location.get("s3Location") or {}).get("uri")
    return uri or metadata.get("x-amz-bedrock-kb-source-uri") or "unknown"


def _passage_text(source: str, content: str) -> str:
    """Return the passage text the model sees, and that the guardrail screens."""
    return f"Document ID: {source}\nContent: {content}"


def format_passages(passages: List[tuple], withheld: int) -> str:
    """Format (score, passage text) pairs as the tool result."""
    if not passages:
        text = f"No results found with score >= {MIN_SCORE}."
    else:
        lines = [f"Retrieved {len(passages)} results with score >= {MIN_SCORE}:", ""]
        for score, passage in passages:
            lines += [f"Score: {score:.4f}", passage, ""]
        text = "\n".join(lines).rstrip()
    if withheld:
        text += "\n\n" + WITHHELD_NOTICE.format(count=withheld)
    return text


def _error_code(error: Exception) -> str:
    """Return a short error name to show the model, without account ids or ARNs."""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
        if code:
            return str(code)
    return type(error).__name__


def build_retrieve_tool(config: RuntimeConfig) -> Optional[object]:
    """Return the ``retrieve`` tool for the configured Knowledge Base, or None if none is set."""
    if not config.knowledge_base_enabled:
        logger.warning(
            "No Knowledge Base id configured; billing/escalation RAG retrieval is "
            "disabled. Set STRANDS_KNOWLEDGE_BASE_ID (or KNOWLEDGE_BASE_ID)."
        )
        return None

    # Imported lazily so the package imports without the SDK (e.g. for unit tests).
    import boto3
    from strands import tool
    from strands.vended_memory_stores import BedrockKnowledgeBaseStore

    # A session of its own: boto3's default session isn't safe to create clients
    # from on several threads, and the demo builds agents per browser session.
    session = boto3.Session(region_name=config.region)
    store = BedrockKnowledgeBaseStore(
        name="knowledge_base",
        config={
            "knowledge_base_id": config.knowledge_base_id,
            # The CDK stack creates a vector Knowledge Base; naming the type skips a
            # GetKnowledgeBase call, so only bedrock:Retrieve is needed.
            "knowledge_base_type": "VECTOR",
            "runtime_client": session.client("bedrock-agent-runtime"),
        },
        writable=False,
        min_score=MIN_SCORE,
        max_search_results=MAX_RESULTS,
    )
    screener = (
        PassageScreener(
            session.client("bedrock-runtime"),
            config.guardrail_id,
            config.guardrail_version,
        )
        if config.guardrail_enabled
        else None
    )

    @tool(name="retrieve", description=TOOL_DESCRIPTION)
    async def retrieve(text: str) -> str:
        """Search the Knowledge Base.

        Args:
            text: What to search for, in natural language.
        """
        try:
            entries = await store.search(text)
        except Exception as error:
            logger.exception("retrieve: Knowledge Base search failed")
            raise RuntimeError(f"Knowledge Base search failed ({_error_code(error)}).") from None

        # The old tool dropped results without a score; so does this one, so every
        # result shown meets the floor. Blank passages are dropped too.
        passages = [
            ((entry.metadata or {}).get("_relevance_score"),
             _passage_text(_source_uri(entry.metadata or {}), entry.content))
            for entry in entries
            if (entry.content or "").strip()
            and (entry.metadata or {}).get("_relevance_score") is not None
        ]
        withheld = 0
        if screener is not None and passages:
            try:
                # boto3 is blocking; run the calls off the event loop, side by side.
                # If one fails, the others still finish (and are billed) but are
                # discarded.
                screened = await asyncio.gather(
                    *(asyncio.to_thread(screener.screen, passage) for _, passage in passages)
                )
            except Exception as error:
                logger.exception("retrieve: ApplyGuardrail failed; results withheld")
                raise RuntimeError(
                    f"Content screening failed ({_error_code(error)}); results withheld."
                ) from None
            withheld = screened.count(None)
            if withheld:
                logger.warning("retrieve: the guardrail withheld %d of %d passages", withheld, len(passages))
            passages = [(score, kept) for (score, _), kept in zip(passages, screened) if kept is not None]
        return format_passages(passages, withheld)

    return retrieve
