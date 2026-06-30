"""Streamlit demo frontend for the AnyCompany Energy agentic customer service system.

This app runs the **Strands Agents** multi-agent team *in process*: it imports the
supervisor (intent-router) orchestrator from the :mod:`agents` package and calls
it directly — no remote agent runtime to invoke. A shared Strands **callback
handler** captures the full nested timeline of each turn:

  * the supervisor's ``classify_intent`` decision,
  * which specialist the supervisor *routed* to (a specialist-as-tool call), and
  * every tool the specialist ran (e.g. ``check_balance``, ``create_ticket``),
    with parameters and results.

Configuration is read from environment variables (wire the CDK stack outputs via
``demo/.env``) and can be overridden live from the sidebar.

Run with::

    streamlit run app.py

See ``demo/README.md`` for the full setup walkthrough.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

# Make the sibling `agents` package importable when running `streamlit run app.py`
# from the demo/ directory (the package lives at the repository root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"

INTERACTIONS_PER_MONTH = 12_000
COST_PER_INTERACTION_GBP = 8.50

SAMPLE_ACCOUNTS = (
    ("ACC-100123", "Sarah Thompson", "Economy 7 - GBP 142.50 owed, ACTIVE"),
    ("ACC-100456", "James Okafor", "Standard Variable - GBP 0.00, on payment plan"),
    ("ACC-100789", "Priya Patel", "EV Smart - GBP 318.20 arrears, ARREARS, vulnerable"),
)

# Friendly labels for the specialist-as-tool routing calls.
_ROUTING_TOOLS = {
    "billing_specialist": "Billing Agent",
    "scheduling_specialist": "Scheduling Agent",
    "escalation_specialist": "Escalation Agent",
}


# --------------------------------------------------------------------------- #
# Timeline model + Strands callback collector
# --------------------------------------------------------------------------- #
@dataclass
class TimelineEvent:
    """A single observable step (a tool call / routing decision) in a turn."""

    kind: str  # "routing" | "tool_call"
    title: str
    tool: str
    params: str = ""
    result: str = ""


class TraceCollector:
    """Strands callback handler that records the tool/routing timeline of a turn.

    The same collector is wired into the supervisor and every specialist, so it
    observes the full nested hierarchy: the supervisor's ``classify_intent`` and
    specialist routing calls, plus the tools each specialist runs.
    """

    def __init__(self) -> None:
        self.events: list[TimelineEvent] = []
        self._by_id: dict[str, TimelineEvent] = {}

    def reset(self) -> None:
        self.events = []
        self._by_id = {}

    def __call__(self, **kwargs: Any) -> None:
        tool_use = kwargs.get("current_tool_use") or {}
        if isinstance(tool_use, dict) and tool_use.get("name"):
            self._record_tool_use(tool_use)

        message = kwargs.get("message")
        if isinstance(message, dict):
            self._record_tool_results(message)

    def _record_tool_use(self, tool_use: dict[str, Any]) -> None:
        name = tool_use["name"]
        tid = tool_use.get("toolUseId") or name
        event = self._by_id.get(tid)
        if event is None:
            if name in _ROUTING_TOOLS:
                event = TimelineEvent(
                    kind="routing", title=f"Routed to: {_ROUTING_TOOLS[name]}", tool=name
                )
            else:
                event = TimelineEvent(
                    kind="tool_call", title=f"Tool call: {name}", tool=name
                )
            self._by_id[tid] = event
            self.events.append(event)
        raw_input = tool_use.get("input")
        if raw_input:
            event.params = raw_input if isinstance(raw_input, str) else json.dumps(raw_input)

    def _record_tool_results(self, message: dict[str, Any]) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            tool_result = block.get("toolResult")
            if not isinstance(tool_result, dict):
                continue
            tid = tool_result.get("toolUseId")
            event = self._by_id.get(tid) if tid else None
            if event is None:
                continue
            event.result = _extract_result_text(tool_result.get("content"))


def _extract_result_text(content: Any) -> str:
    """Flatten a toolResult content list into a display string."""
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if "json" in block:
            parts.append(json.dumps(block["json"]))
        elif "text" in block:
            parts.append(str(block["text"]))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Turn result
# --------------------------------------------------------------------------- #
@dataclass
class TurnResult:
    answer: str = ""
    events: list[TimelineEvent] = field(default_factory=list)
    error: str | None = None


def _answer_text(result: Any) -> str:
    """Extract the final assistant text from a Strands AgentResult."""
    try:
        message = getattr(result, "message", None)
        if isinstance(message, dict):
            texts = [
                block["text"]
                for block in message.get("content", [])
                if isinstance(block, dict) and "text" in block
            ]
            if texts:
                return "\n".join(texts).strip()
    except Exception:  # noqa: BLE001 - fall back to str()
        logger.debug("Could not parse AgentResult.message; using str()")
    return str(result).strip()


# --------------------------------------------------------------------------- #
# Configuration / agent construction
# --------------------------------------------------------------------------- #
def apply_sidebar_config_to_env(region: str, kb_id: str, guardrail_id: str) -> None:
    """Push the sidebar values into the environment for ``load_config``."""
    os.environ["AWS_REGION"] = region.strip() or DEFAULT_REGION
    if kb_id.strip():
        os.environ["STRANDS_KNOWLEDGE_BASE_ID"] = kb_id.strip()
    if guardrail_id.strip():
        os.environ["GUARDRAIL_ID"] = guardrail_id.strip()


def get_collector() -> TraceCollector:
    if "collector" not in st.session_state:
        st.session_state["collector"] = TraceCollector()
    return st.session_state["collector"]


def build_or_get_supervisor(config_signature: str):
    """Build (and cache per session) the supervisor for the current config.

    Stored in ``st.session_state`` rather than a global cache so each browser
    session keeps its own conversation history and the shared trace collector.
    """
    cached = st.session_state.get("supervisor_bundle")
    if cached and cached["signature"] == config_signature:
        return cached["supervisor"]

    from agents.config import load_config
    from agents.supervisor import build_supervisor

    collector = get_collector()
    config = load_config()
    supervisor = build_supervisor(config, callback_handler=collector)
    st.session_state["supervisor_bundle"] = {
        "signature": config_signature,
        "supervisor": supervisor,
        "config": config,
    }
    return supervisor


def run_turn(supervisor: Any, collector: TraceCollector, user_text: str) -> TurnResult:
    """Run one supervisor turn, collecting the timeline via the callback."""
    collector.reset()
    result = TurnResult()
    try:
        agent_result = supervisor(user_text)
        result.answer = _answer_text(agent_result)
        result.events = list(collector.events)
        if not result.answer:
            result.answer = "The agent completed without a textual answer. Review the timeline."
    except Exception as exc:  # noqa: BLE001 - surface anything to the operator
        result.error = _friendly_error(exc)
        result.events = list(collector.events)
        logger.exception("Error running supervisor turn")
    return result


def _friendly_error(exc: Exception) -> str:
    """Translate common failures into operator-friendly guidance."""
    text = str(exc)
    lowered = text.lower()
    if "accessdenied" in lowered or "not authorized" in lowered:
        return (
            "Access denied calling Amazon Bedrock. Ensure your AWS credentials can "
            "invoke the configured models (and the guardrail), and that the region "
            f"is correct. Details: {text}"
        )
    if "could not connect" in lowered or "endpoint" in lowered or "credential" in lowered:
        return (
            "Could not reach AWS / no credentials. Check your AWS credentials and "
            f"region. Details: {text}"
        )
    if "throttl" in lowered or "too many requests" in lowered:
        return f"Bedrock throttled the request. Wait a moment and retry. Details: {text}"
    if "validation" in lowered and "inference profile" in lowered:
        return (
            "Bedrock rejected the model id — it must be a cross-region inference "
            "profile for your region's geo. Check AWS_REGION / model access. "
            f"Details: {text}"
        )
    return f"Unexpected error while running the agents: {text}"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_header() -> None:
    st.set_page_config(
        page_title="AnyCompany Energy — Agentic Customer Service", page_icon="💧", layout="wide"
    )
    st.title("AnyCompany Energy — Agentic Customer Service Demo")

    annual_baseline = INTERACTIONS_PER_MONTH * COST_PER_INTERACTION_GBP * 12
    low_saving = annual_baseline * 0.40
    high_saving = annual_baseline * 0.60

    st.markdown(
        f"""
        A **Strands Agents** multi-agent assistant that automates **service ->
        billing -> scheduling -> escalation** for a UK utility. A supervisor
        (Intent Router) classifies each query and delegates to a specialist agent
        using the *agents-as-tools* pattern, with full tool-call transparency.

        **Business case:** {INTERACTIONS_PER_MONTH:,} interactions/month at
        GBP {COST_PER_INTERACTION_GBP:.2f} each
        (~GBP {annual_baseline/1_000:.0f}K/year baseline). Targeting **40-60% cost
        reduction** -> approx. **GBP {low_saving/1_000:.0f}K-{high_saving/1_000:.0f}K/year saved.**
        """
    )

    with st.expander("Sample accounts you can use in the demo", expanded=False):
        for account_number, name, summary in SAMPLE_ACCOUNTS:
            st.markdown(f"- **{account_number}** — {name} ({summary})")
        st.caption(
            "Try: *'What's the balance on ACC-100123?'*, "
            "*'Book a meter reading for ACC-100456 in LS6 2AB'*, "
            "*'I have no water and want to make a complaint about ACC-100789'*."
        )


def render_sidebar() -> tuple[str, str, str]:
    """Render configuration controls; return (region, kb_id, guardrail_id)."""
    st.sidebar.header("Configuration")
    st.sidebar.caption("Defaults are read from the environment (demo/.env); override here.")

    region = st.sidebar.text_input(
        "AWS Region", value=os.environ.get("AWS_REGION", DEFAULT_REGION),
        help="Region where your Knowledge Base / Guardrail live and models are invoked.",
    )
    kb_id = st.sidebar.text_input(
        "Knowledge Base ID",
        value=os.environ.get("STRANDS_KNOWLEDGE_BASE_ID", os.environ.get("KNOWLEDGE_BASE_ID", "")),
        help="CloudFormation output: KnowledgeBaseId. Enables billing/escalation RAG.",
    )
    guardrail_id = st.sidebar.text_input(
        "Guardrail ID",
        value=os.environ.get("GUARDRAIL_ID", ""),
        help="CloudFormation output: GuardrailId. Applied to every model call.",
    )

    if st.sidebar.button("Reset conversation"):
        st.session_state.pop("supervisor_bundle", None)
        st.session_state.pop("messages", None)
        get_collector().reset()
        st.rerun()

    return region, kb_id, guardrail_id


def render_timeline(events: list[TimelineEvent]) -> None:
    """Render a turn's routing/tool timeline into the sidebar."""
    st.sidebar.subheader("Latest turn — agent timeline")
    if not events:
        st.sidebar.caption("No tool activity captured for the latest turn.")
        return
    for event in events:
        icon = "🧭" if event.kind == "routing" else "🛠️"
        st.sidebar.markdown(f"**{icon} {event.title}**")
        if event.params:
            try:
                st.sidebar.json(json.loads(event.params), expanded=False)
            except (ValueError, TypeError):
                st.sidebar.caption(event.params)
        if event.result:
            with st.sidebar.expander("result", expanded=False):
                try:
                    st.json(json.loads(event.result), expanded=False)
                except (ValueError, TypeError):
                    st.write(event.result)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    render_header()
    region, kb_id, guardrail_id = render_sidebar()
    apply_sidebar_config_to_env(region, kb_id, guardrail_id)

    if not kb_id.strip():
        st.sidebar.warning(
            "No Knowledge Base ID set — billing/escalation RAG answers (tariffs, "
            "policies, Ombudsman contacts) will be unavailable until you set it."
        )
    if not guardrail_id.strip():
        st.sidebar.info("No Guardrail ID set — running without the Bedrock guardrail.")

    config_signature = json.dumps(
        {
            "region": os.environ.get("AWS_REGION", ""),
            "kb": os.environ.get("STRANDS_KNOWLEDGE_BASE_ID", ""),
            "guardrail": os.environ.get("GUARDRAIL_ID", ""),
        },
        sort_keys=True,
    )

    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_text = st.chat_input("Ask about billing, appointments, or raise a complaint…")
    if user_text:
        messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            with st.spinner("The agents are working…"):
                try:
                    supervisor = build_or_get_supervisor(config_signature)
                    collector = get_collector()
                    turn = run_turn(supervisor, collector, user_text)
                except Exception as exc:  # noqa: BLE001 - setup/import failure
                    turn = TurnResult(error=_friendly_error(exc))

            if turn.error:
                st.error(turn.error)
                messages.append({"role": "assistant", "content": f"⚠️ {turn.error}"})
            else:
                st.markdown(turn.answer)
                messages.append({"role": "assistant", "content": turn.answer})
            st.session_state["last_events"] = turn.events

    if st.session_state.get("last_events") is not None:
        render_timeline(st.session_state["last_events"])


if __name__ == "__main__":
    main()
