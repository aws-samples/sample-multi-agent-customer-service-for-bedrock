"""Streamlit demo frontend for the AnyCompany Energy agentic customer service system.

This application provides a chat interface that invokes the Bedrock multi-agent
"supervisor" (the Intent Router) via the ``bedrock-agent-runtime`` API. It streams
the agent's final answer to the user while parsing the Bedrock trace stream to build
a transparent, real-time timeline of:

  * which specialist sub-agent (collaborator) the supervisor routed the query to,
  * every action-group tool invocation, including its parameters and result, and
  * the supervisor's own reasoning (rationale).

Configuration is read from environment variables (so the operator can wire the CDK
stack outputs straight into the process environment) and can be overridden live from
the Streamlit sidebar.

Run with::

    streamlit run demo/app.py

See ``demo/README.md`` for the full setup walkthrough, including how to obtain the
agent identifiers from the CDK stack outputs.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

import boto3
import streamlit as st
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_REGION: str = "us-east-1"
"""Default AWS region used when neither the environment nor sidebar provides one."""

#: Bedrock Agents runtime client name (data plane: invoke_agent).
BEDROCK_AGENT_RUNTIME_SERVICE: str = "bedrock-agent-runtime"

#: Bedrock Agents control-plane client name (used for get_agent readiness checks).
BEDROCK_AGENT_SERVICE: str = "bedrock-agent"

#: Bedrock agent lifecycle states that mean the agent will never become ready
#: without operator intervention, so we stop waiting and surface the failure.
#: (NOT_PREPARED is deliberately excluded: an agent can pass through it
#: transiently just after creation, so we keep polling and let the timeout
#: handle a genuinely-stuck agent gracefully rather than raising a false error.)
AGENT_TERMINAL_STATES: frozenset[str] = frozenset({"FAILED", "DELETING"})

#: Defaults for the post-deploy agent-readiness wait. After ``cdk deploy`` an
#: agent can briefly remain in PREPARING/VERSIONING before reaching PREPARED.
AGENT_READY_MAX_WAIT_SECONDS: int = 120
AGENT_READY_POLL_INTERVAL_SECONDS: int = 5

#: Cost / ROI framing for the demo header (AnyCompany Energy business case).
INTERACTIONS_PER_MONTH: int = 12_000
COST_PER_INTERACTION_GBP: float = 8.50

#: Sample account numbers seeded in the shared mock datastore. The demo script and
#: the Lambda mock data depend on these EXACT values, so they are surfaced in the UI
#: to make a live demonstration reproducible.
SAMPLE_ACCOUNTS: tuple[tuple[str, str, str], ...] = (
    ("ACC-100123", "Sarah Thompson", "Economy 7 - GBP 142.50 owed, ACTIVE"),
    ("ACC-100456", "James Okafor", "Standard Variable - GBP 0.00, on payment plan"),
    ("ACC-100789", "Priya Patel", "EV Smart - GBP 318.20 arrears, ARREARS, vulnerable"),
)

#: Maps a friendly collaborator agent name fragment to a short routing label/emoji
#: so the timeline reads cleanly even before the agent names are fully known.
AGENT_LABELS: dict[str, str] = {
    "billing": "Billing Agent",
    "scheduling": "Scheduling Agent",
    "escalation": "Escalation Agent",
    "intent": "Intent Router (Supervisor)",
}


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DemoConfig:
    """Immutable runtime configuration for a single agent invocation.

    Attributes:
        agent_id: The Bedrock supervisor agent id (CDK output ``SupervisorAgentId``).
        agent_alias_id: The supervisor agent alias id (CDK output
            ``SupervisorAgentAliasId``).
        region: The AWS region in which the agent is deployed.
    """

    agent_id: str
    agent_alias_id: str
    region: str

    def is_complete(self) -> bool:
        """Return ``True`` when all required identifiers are present.

        Returns:
            ``True`` if ``agent_id``, ``agent_alias_id`` and ``region`` are all
            non-empty after stripping whitespace; otherwise ``False``.
        """
        return bool(self.agent_id.strip() and self.agent_alias_id.strip() and self.region.strip())


# ---------------------------------------------------------------------------
# Timeline data model
# ---------------------------------------------------------------------------
@dataclass
class TimelineEvent:
    """A single observable step taken by the agent while answering a query.

    Attributes:
        kind: The event category. One of ``"routing"``, ``"rationale"``,
            ``"tool_call"``, ``"tool_result"`` or ``"sub_agent_result"``.
        title: A short human-readable headline for the event.
        detail: A free-form body (rationale text, tool result string, etc.).
        agent: The collaborator/sub-agent associated with the event, if any.
        tool: The action-group function name for tool events, if any.
        parameters: Tool-call parameters as a name -> value mapping.
    """

    kind: str
    title: str
    detail: str = ""
    agent: str | None = None
    tool: str | None = None
    parameters: dict[str, str] = field(default_factory=dict)


@dataclass
class TurnResult:
    """The full outcome of one supervisor invocation.

    Attributes:
        answer: The accumulated final answer text shown to the user.
        events: The ordered list of timeline events parsed from the trace stream.
        routed_to: The set of collaborator agents the supervisor delegated to.
        error: An optional friendly error message if the invocation failed.
    """

    answer: str = ""
    events: list[TimelineEvent] = field(default_factory=list)
    routed_to: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
def load_config_from_env() -> DemoConfig:
    """Build a :class:`DemoConfig` from process environment variables.

    Reads ``SUPERVISOR_AGENT_ID``, ``SUPERVISOR_AGENT_ALIAS_ID`` and ``AWS_REGION``.
    Missing values default to empty strings (region defaults to
    :data:`DEFAULT_REGION`) so the sidebar can prompt the operator to fill them in.

    Returns:
        A :class:`DemoConfig` populated from the environment.
    """
    return DemoConfig(
        agent_id=os.environ.get("SUPERVISOR_AGENT_ID", "").strip(),
        agent_alias_id=os.environ.get("SUPERVISOR_AGENT_ALIAS_ID", "").strip(),
        region=os.environ.get("AWS_REGION", DEFAULT_REGION).strip() or DEFAULT_REGION,
    )


def get_session_id() -> str:
    """Return a stable per-browser-session conversation id.

    The id is created once and persisted in ``st.session_state`` so that all turns
    in a single Streamlit session share the same Bedrock conversation context
    (memory, slots, etc.). A fresh UUID is generated on first access.

    Returns:
        The stable session id string.
    """
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        logger.info("Initialised new Bedrock session id: %s", st.session_state["session_id"])
    return st.session_state["session_id"]


# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bedrock_runtime_client(region: str) -> Any:
    """Create (and cache) a ``bedrock-agent-runtime`` boto3 client for a region.

    The client is cached per-region by Streamlit's resource cache so we do not pay
    the connection-setup cost on every rerun. A bounded retry policy is configured
    to smooth over transient throttling.

    Args:
        region: The AWS region in which the agent is deployed.

    Returns:
        A configured boto3 ``bedrock-agent-runtime`` client.
    """
    logger.info("Creating bedrock-agent-runtime client in region %s", region)
    boto_config = BotoConfig(
        region_name=region,
        retries={"max_attempts": 3, "mode": "adaptive"},
        read_timeout=120,
        connect_timeout=10,
    )
    return boto3.client(BEDROCK_AGENT_RUNTIME_SERVICE, config=boto_config)


@st.cache_resource(show_spinner=False)
def get_bedrock_control_client(region: str) -> Any:
    """Create (and cache) a ``bedrock-agent`` control-plane boto3 client.

    This client is distinct from the runtime client: it talks to the Bedrock
    Agents *control plane* (``get_agent``) and requires the ``bedrock:GetAgent``
    IAM permission rather than ``bedrock:InvokeAgent``. It is used only for the
    post-deploy readiness check.

    Args:
        region: The AWS region in which the agent is deployed.

    Returns:
        A configured boto3 ``bedrock-agent`` client.
    """
    logger.info("Creating bedrock-agent control-plane client in region %s", region)
    boto_config = BotoConfig(
        region_name=region,
        retries={"max_attempts": 3, "mode": "adaptive"},
        read_timeout=30,
        connect_timeout=10,
    )
    return boto3.client(BEDROCK_AGENT_SERVICE, config=boto_config)


# ---------------------------------------------------------------------------
# Agent readiness
# ---------------------------------------------------------------------------
def wait_for_agent_ready(
    agent_id: str,
    region: str,
    max_wait: int = AGENT_READY_MAX_WAIT_SECONDS,
    interval: int = AGENT_READY_POLL_INTERVAL_SECONDS,
) -> bool:
    """Poll the Bedrock control plane until the supervisor agent is ``PREPARED``.

    Immediately after ``cdk deploy`` (or a fresh agent preparation) the agent can
    sit in ``PREPARING``/``VERSIONING`` for a short while before it can serve
    ``invoke_agent`` calls. This blocks until the agent reports ``PREPARED`` or the
    ``max_wait`` budget is exhausted.

    Args:
        agent_id: The supervisor agent id to check.
        region: The AWS region the agent is deployed in.
        max_wait: Maximum number of seconds to wait before giving up.
        interval: Seconds to sleep between successive ``get_agent`` polls.

    Returns:
        ``True`` if the agent reached ``PREPARED`` within ``max_wait`` seconds;
        ``False`` if the wait timed out.

    Raises:
        RuntimeError: If the agent enters a terminal state (``FAILED`` /
            ``DELETING``) from which it cannot become ready.
    """
    client = get_bedrock_control_client(region)
    elapsed = 0
    last_status = "UNKNOWN"
    while elapsed < max_wait:
        response = client.get_agent(agentId=agent_id)
        last_status = response.get("agent", {}).get("agentStatus", "UNKNOWN")
        logger.info("Agent %s status=%s (elapsed %ss)", agent_id, last_status, elapsed)
        if last_status == "PREPARED":
            return True
        if last_status in AGENT_TERMINAL_STATES:
            raise RuntimeError(
                f"Agent {agent_id} is in terminal state '{last_status}' and cannot "
                "serve requests. Re-check the deployment / agent preparation."
            )
        time.sleep(interval)  # nosemgrep: arbitrary-sleep — intentional polling delay for Bedrock agent response streaming
        elapsed += interval
    logger.warning(
        "Agent %s did not reach PREPARED within %ss (last status=%s)",
        agent_id,
        max_wait,
        last_status,
    )
    return False


def ensure_agent_ready(config: DemoConfig) -> bool:
    """Ensure the supervisor agent is ``PREPARED`` before chat is enabled.

    Streamlit reruns the whole script on every interaction, so the readiness
    result is memoised in ``st.session_state`` keyed by agent id: the (possibly
    slow) control-plane wait runs at most once per agent per browser session.
    A spinner is shown while waiting, and any failure is surfaced to the operator.

    Args:
        config: The active demo configuration (must already be complete).

    Returns:
        ``True`` if the agent is ready and chat should be enabled; ``False`` if
        the agent is not ready (timed out or in a terminal/error state).
    """
    ready_cache: dict[str, bool] = st.session_state.setdefault("agent_ready", {})
    if ready_cache.get(config.agent_id):
        return True

    try:
        with st.spinner("Waiting for the supervisor agent to be ready..."):
            is_ready = wait_for_agent_ready(config.agent_id, config.region)
    except RuntimeError as exc:
        st.error(str(exc))
        logger.exception("Agent %s reached a terminal state", config.agent_id)
        return False
    except ClientError as exc:
        st.error(_friendly_client_error(exc))
        logger.exception("Control-plane error checking agent %s readiness", config.agent_id)
        return False
    except BotoCoreError as exc:
        st.error(
            "Could not reach the Bedrock control plane to check agent readiness. "
            f"Check your network, region and credentials. Details: {exc}"
        )
        logger.exception("BotoCoreError checking agent %s readiness", config.agent_id)
        return False

    if is_ready:
        ready_cache[config.agent_id] = True
        return True

    st.warning(
        "The supervisor agent is not in the PREPARED state yet. It may still be "
        "preparing after deployment — wait a moment and use **Re-check agent** in "
        "the sidebar, or confirm the agent prepared successfully in the console."
    )
    return False


# ---------------------------------------------------------------------------
# Trace parsing helpers
# ---------------------------------------------------------------------------
def _label_for_agent(agent_name: str | None) -> str:
    """Return a friendly display label for a collaborator agent name.

    Args:
        agent_name: The raw collaborator name reported by Bedrock (may be ``None``).

    Returns:
        A human-friendly label. Falls back to the raw name (or ``"Supervisor"``).
    """
    if not agent_name:
        return "Supervisor"
    lowered = agent_name.lower()
    for fragment, label in AGENT_LABELS.items():
        if fragment in lowered:
            return label
    return agent_name


def _stringify_parameters(parameters: Iterable[dict[str, Any]] | None) -> dict[str, str]:
    """Convert Bedrock action-group parameter objects into a name -> value map.

    Bedrock reports tool-call parameters as a list of objects shaped like
    ``{"name": ..., "type": ..., "value": ...}``. This flattens them into a simple
    dictionary for display, coercing values to strings defensively.

    Args:
        parameters: The raw parameter list from the trace (may be ``None``).

    Returns:
        A mapping of parameter name to its string value.
    """
    result: dict[str, str] = {}
    if not parameters:
        return result
    for param in parameters:
        if not isinstance(param, dict):
            continue
        name = str(param.get("name", "")).strip()
        if not name:
            continue
        result[name] = str(param.get("value", ""))
    return result


def _parse_orchestration_trace(orchestration: dict[str, Any], result: TurnResult) -> None:
    """Extract timeline events from a single ``orchestrationTrace`` payload.

    Handles the invocation-input shapes (tool calls and sub-agent routing),
    observation shapes (tool results and sub-agent results) and the agent's
    rationale. All lookups are defensive via ``.get()`` because Bedrock only
    populates the subset of fields relevant to the current step.

    Args:
        orchestration: The ``orchestrationTrace`` dictionary from a trace event.
        result: The :class:`TurnResult` accumulator that events are appended to.
    """
    # --- Agent reasoning -------------------------------------------------
    rationale = orchestration.get("rationale", {})
    rationale_text = rationale.get("text", "").strip() if isinstance(rationale, dict) else ""
    if rationale_text:
        result.events.append(
            TimelineEvent(kind="rationale", title="Supervisor reasoning", detail=rationale_text)
        )

    # --- Invocation inputs (the agent is about to do something) ----------
    invocation_input = orchestration.get("invocationInput", {})
    if isinstance(invocation_input, dict):
        # A tool / action-group call.
        action_input = invocation_input.get("actionGroupInvocationInput", {})
        if isinstance(action_input, dict) and action_input:
            action_group = action_input.get("actionGroupName", "unknown-action-group")
            function_name = action_input.get("function", "unknown-function")
            params = _stringify_parameters(action_input.get("parameters"))
            result.events.append(
                TimelineEvent(
                    kind="tool_call",
                    title=f"Tool call: {function_name}",
                    detail=f"Action group: {action_group}",
                    tool=function_name,
                    parameters=params,
                )
            )

        # Routing to a specialist collaborator sub-agent.
        collaborator_input = invocation_input.get("agentCollaboratorInvocationInput", {})
        if isinstance(collaborator_input, dict) and collaborator_input:
            collaborator_name = collaborator_input.get("agentCollaboratorName")
            sub_input = collaborator_input.get("input", {})
            sub_text = sub_input.get("text", "").strip() if isinstance(sub_input, dict) else ""
            label = _label_for_agent(collaborator_name)
            if label not in result.routed_to:
                result.routed_to.append(label)
            result.events.append(
                TimelineEvent(
                    kind="routing",
                    title=f"Routed to: {label}",
                    detail=sub_text,
                    agent=label,
                )
            )

    # --- Observations (results coming back) ------------------------------
    observation = orchestration.get("observation", {})
    if isinstance(observation, dict):
        action_output = observation.get("actionGroupInvocationOutput", {})
        action_text = action_output.get("text", "").strip() if isinstance(action_output, dict) else ""
        if action_text:
            result.events.append(
                TimelineEvent(
                    kind="tool_result",
                    title="Tool result",
                    detail=action_text,
                )
            )

        collaborator_output = observation.get("agentCollaboratorInvocationOutput", {})
        if isinstance(collaborator_output, dict) and collaborator_output:
            collaborator_name = collaborator_output.get("agentCollaboratorName")
            output = collaborator_output.get("output", {})
            output_text = output.get("text", "").strip() if isinstance(output, dict) else ""
            if output_text:
                label = _label_for_agent(collaborator_name)
                result.events.append(
                    TimelineEvent(
                        kind="sub_agent_result",
                        title=f"{label} responded",
                        detail=output_text,
                        agent=label,
                    )
                )


def _parse_routing_classifier_trace(routing: dict[str, Any], result: TurnResult) -> None:
    """Extract a routing decision from a ``routingClassifierTrace`` payload.

    The multi-agent supervisor emits a routing-classifier trace when it decides
    which collaborator should handle the request. The exact sub-shape varies, so we
    inspect both the ``modelInvocationOutput`` (raw classifier text) and the nested
    ``observation`` for a collaborator name.

    Args:
        routing: The ``routingClassifierTrace`` dictionary from a trace event.
        result: The :class:`TurnResult` accumulator that events are appended to.
    """
    if not isinstance(routing, dict):
        return

    observation = routing.get("observation", {})
    if isinstance(observation, dict):
        collaborator_output = observation.get("agentCollaboratorInvocationOutput", {})
        if isinstance(collaborator_output, dict) and collaborator_output:
            label = _label_for_agent(collaborator_output.get("agentCollaboratorName"))
            if label not in result.routed_to:
                result.routed_to.append(label)
            result.events.append(
                TimelineEvent(kind="routing", title=f"Classifier routed to: {label}", agent=label)
            )

    invocation_input = routing.get("invocationInput", {})
    if isinstance(invocation_input, dict):
        collaborator_input = invocation_input.get("agentCollaboratorInvocationInput", {})
        if isinstance(collaborator_input, dict) and collaborator_input:
            label = _label_for_agent(collaborator_input.get("agentCollaboratorName"))
            if label not in result.routed_to:
                result.routed_to.append(label)
            result.events.append(
                TimelineEvent(kind="routing", title=f"Routed to: {label}", agent=label)
            )


def _handle_trace_event(trace_event: dict[str, Any], result: TurnResult) -> None:
    """Dispatch a single ``trace`` stream event to the right parser.

    Args:
        trace_event: The value of ``event["trace"]["trace"]`` from the completion
            EventStream.
        result: The :class:`TurnResult` accumulator that events are appended to.
    """
    if not isinstance(trace_event, dict):
        return

    routing = trace_event.get("routingClassifierTrace")
    if isinstance(routing, dict) and routing:
        _parse_routing_classifier_trace(routing, result)

    orchestration = trace_event.get("orchestrationTrace")
    if isinstance(orchestration, dict) and orchestration:
        _parse_orchestration_trace(orchestration, result)


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------
def invoke_supervisor(config: DemoConfig, session_id: str, user_text: str) -> TurnResult:
    """Invoke the supervisor agent and parse its streamed response and trace.

    Calls ``invoke_agent`` with ``enableTrace=True`` and iterates the returned
    ``completion`` EventStream. ``chunk`` events are decoded and concatenated into
    the final answer; ``trace`` events are parsed into the timeline. All AWS errors
    are caught and converted into friendly messages on :attr:`TurnResult.error`.

    Args:
        config: The validated runtime configuration.
        session_id: The stable conversation/session id for this browser session.
        user_text: The end-user's message.

    Returns:
        A :class:`TurnResult` with the answer, timeline events and any error.
    """
    result = TurnResult()
    try:
        client = get_bedrock_runtime_client(config.region)
        logger.info(
            "Invoking agent id=%s alias=%s session=%s", config.agent_id, config.agent_alias_id, session_id
        )
        response = client.invoke_agent(
            agentId=config.agent_id,
            agentAliasId=config.agent_alias_id,
            sessionId=session_id,
            inputText=user_text,
            enableTrace=True,
        )

        completion = response.get("completion")
        if completion is None:
            result.error = "The agent returned no completion stream. Check the agent alias is prepared."
            logger.error("invoke_agent returned no completion for session %s", session_id)
            return result

        for event in completion:
            if not isinstance(event, dict):
                continue

            chunk = event.get("chunk")
            if isinstance(chunk, dict):
                raw_bytes = chunk.get("bytes")
                if isinstance(raw_bytes, (bytes, bytearray)):
                    result.answer += raw_bytes.decode("utf-8", errors="replace")

            trace = event.get("trace")
            if isinstance(trace, dict):
                inner_trace = trace.get("trace")
                if isinstance(inner_trace, dict):
                    _handle_trace_event(inner_trace, result)

        if not result.answer.strip() and not result.error:
            result.answer = (
                "The agent completed without producing a textual answer. "
                "Review the timeline for tool activity."
            )
        return result

    except ClientError as exc:
        result.error = _friendly_client_error(exc)
        logger.exception("ClientError invoking agent for session %s", session_id)
        return result
    except BotoCoreError as exc:
        result.error = (
            "Could not reach AWS Bedrock. Check your network connection, region and "
            f"credentials. Details: {exc}"
        )
        logger.exception("BotoCoreError invoking agent for session %s", session_id)
        return result
    except Exception as exc:  # noqa: BLE001 - surface anything unexpected to the operator
        result.error = f"Unexpected error while invoking the agent: {exc}"
        logger.exception("Unexpected error invoking agent for session %s", session_id)
        return result


def _friendly_client_error(exc: ClientError) -> str:
    """Translate a boto3 :class:`ClientError` into an operator-friendly message.

    Args:
        exc: The raised client error.

    Returns:
        A human-readable explanation with a suggested remediation.
    """
    error = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
    code = error.get("Code", "Unknown")
    message = error.get("Message", str(exc))

    if code in {"AccessDeniedException", "AccessDenied"}:
        return (
            "Access denied invoking the agent. Ensure your IAM principal has "
            "`bedrock:InvokeAgent` on this agent alias and that the region is correct. "
            f"({message})"
        )
    if code in {"ResourceNotFoundException", "ResourceNotFound"}:
        return (
            "Agent or alias not found. Double-check SUPERVISOR_AGENT_ID and "
            "SUPERVISOR_AGENT_ALIAS_ID against the CDK stack outputs, and confirm the "
            f"region. ({message})"
        )
    if code in {"ThrottlingException", "ThrottledException", "TooManyRequestsException"}:
        return (
            "The request was throttled by Bedrock. Please wait a moment and try again. "
            f"({message})"
        )
    if code == "ValidationException":
        return f"Bedrock rejected the request as invalid: {message}"
    if code in {"ServiceQuotaExceededException", "ServiceUnavailableException", "InternalServerException"}:
        return f"Bedrock is temporarily unavailable ({code}). Please retry shortly. ({message})"
    return f"Bedrock returned an error ({code}): {message}"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_header() -> None:
    """Render the page title and the cost/ROI business framing for the demo."""
    st.set_page_config(page_title="AnyCompany Energy — Agentic Customer Service", page_icon="💧", layout="wide")
    st.title("AnyCompany Energy — Agentic Customer Service Demo")

    annual_baseline = INTERACTIONS_PER_MONTH * COST_PER_INTERACTION_GBP * 12
    low_saving = annual_baseline * 0.40
    high_saving = annual_baseline * 0.60

    st.markdown(
        f"""
        A Bedrock multi-agent assistant that automates **service -> billing -> scheduling
        -> escalation** for a UK utility. The supervisor (Intent Router) classifies each
        query and delegates to a specialist sub-agent, with full tool-call transparency.

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


def render_sidebar(env_config: DemoConfig) -> DemoConfig:
    """Render the sidebar configuration controls and return the active config.

    Environment-variable values are used as the defaults for each input so an
    operator can simply set them before launch, yet still override any value live.

    Args:
        env_config: The configuration loaded from environment variables.

    Returns:
        The :class:`DemoConfig` reflecting the current sidebar values.
    """
    st.sidebar.header("Configuration")
    st.sidebar.caption("Defaults are read from environment variables; override them here.")

    agent_id = st.sidebar.text_input(
        "Supervisor Agent ID",
        value=env_config.agent_id,
        help="CDK output `SupervisorAgentId`.",
    )
    agent_alias_id = st.sidebar.text_input(
        "Supervisor Agent Alias ID",
        value=env_config.agent_alias_id,
        help="CDK output `SupervisorAgentAliasId`.",
    )
    region = st.sidebar.text_input(
        "AWS Region",
        value=env_config.region or DEFAULT_REGION,
        help="The region the AnyCompany stack is deployed to (CDK output `Region`).",
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Session id: `{get_session_id()}`")
    if st.sidebar.button("Reset conversation", use_container_width=True):
        _reset_conversation()
    if st.sidebar.button("Re-check agent", use_container_width=True):
        # Drop the cached readiness verdict so the next run re-polls the
        # control plane (useful if the agent was still preparing earlier).
        st.session_state.pop("agent_ready", None)
        st.rerun()

    return DemoConfig(
        agent_id=agent_id.strip(),
        agent_alias_id=agent_alias_id.strip(),
        region=region.strip() or DEFAULT_REGION,
    )


def _reset_conversation() -> None:
    """Clear the chat history and start a brand-new Bedrock session.

    Removes stored messages and the cached session id from ``st.session_state`` so
    the next turn opens a fresh conversation context, then triggers a rerun.
    """
    st.session_state.pop("messages", None)
    st.session_state.pop("session_id", None)
    logger.info("Conversation reset by operator.")
    st.rerun()


def render_timeline(events: list[TimelineEvent], routed_to: list[str]) -> None:
    """Render the agent activity timeline into the sidebar.

    Shows the routing summary first ("Routed to: ..."), then each step in order:
    supervisor rationale, tool calls (with parameters), tool results and sub-agent
    responses.

    Args:
        events: The ordered timeline events from the latest turn.
        routed_to: The collaborator agents the supervisor delegated to.
    """
    st.sidebar.divider()
    st.sidebar.subheader("Agent activity (last turn)")

    if routed_to:
        st.sidebar.success("Routed to: " + ", ".join(routed_to))
    else:
        st.sidebar.info("Routing handled directly by the supervisor (no sub-agent delegation seen).")

    if not events:
        st.sidebar.caption("No trace activity captured for this turn.")
        return

    icon_by_kind = {
        "routing": "➡️",
        "rationale": "🧠",
        "tool_call": "🔧",
        "tool_result": "📥",
        "sub_agent_result": "🤖",
    }

    for index, event in enumerate(events, start=1):
        icon = icon_by_kind.get(event.kind, "•")
        with st.sidebar.expander(f"{icon} {index}. {event.title}", expanded=False):
            if event.agent:
                st.markdown(f"**Agent:** {event.agent}")
            if event.tool:
                st.markdown(f"**Function:** `{event.tool}`")
            if event.parameters:
                st.markdown("**Parameters:**")
                st.json(event.parameters)
            if event.detail:
                rendered = _maybe_pretty_json(event.detail)
                if rendered is not None:
                    st.json(rendered)
                else:
                    st.markdown(event.detail)


def _maybe_pretty_json(text: str) -> Any | None:
    """Return parsed JSON if ``text`` is a JSON object/array, else ``None``.

    Tool results are frequently compact JSON strings; rendering them with
    ``st.json`` is far more readable than raw text. Non-JSON strings return
    ``None`` so the caller falls back to markdown.

    Args:
        text: The candidate string.

    Returns:
        The parsed object/list if ``text`` is JSON; otherwise ``None``.
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point: render the UI, manage chat state and drive agent invocations."""
    render_header()

    env_config = load_config_from_env()
    config = render_sidebar(env_config)

    if "messages" not in st.session_state:
        # Each message is {"role": "user"|"assistant", "content": str,
        # "events": list[TimelineEvent], "routed_to": list[str]}.
        st.session_state["messages"] = []

    # Replay the conversation so far.
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    session_id = get_session_id()

    # Gate the chat on (1) complete configuration and (2) the supervisor agent
    # having reached the PREPARED state. The chat input stays disabled until both
    # hold, so a freshly-deployed (still-preparing) agent cannot be invoked early.
    if config.is_complete():
        agent_ready = ensure_agent_ready(config)
    else:
        agent_ready = False
        st.info(
            "Set the Supervisor Agent ID, Alias ID and Region in the sidebar (or via "
            "the SUPERVISOR_AGENT_ID / SUPERVISOR_AGENT_ALIAS_ID / AWS_REGION "
            "environment variables) to begin."
        )

    prompt = st.chat_input(
        "Ask about billing, appointments, or raise a complaint...",
        disabled=not agent_ready,
    )

    if prompt:
        if not config.is_complete():
            st.error(
                "Missing configuration. Set the Supervisor Agent ID, Alias ID and Region "
                "in the sidebar (or via the SUPERVISOR_AGENT_ID / SUPERVISOR_AGENT_ALIAS_ID "
                "/ AWS_REGION environment variables) before chatting."
            )
            return

        st.session_state["messages"].append(
            {"role": "user", "content": prompt, "events": [], "routed_to": []}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Routing your request through the agent team..."):
                result = invoke_supervisor(config, session_id, prompt)

            if result.error:
                st.error(result.error)
                answer_to_store = f"_Error: {result.error}_"
            else:
                st.markdown(result.answer)
                answer_to_store = result.answer

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer_to_store,
                "events": result.events,
                "routed_to": result.routed_to,
            }
        )

        # Render the timeline for the turn we just completed.
        render_timeline(result.events, result.routed_to)
        return

    # On reruns without a new prompt, show the timeline of the most recent turn.
    last_assistant = next(
        (m for m in reversed(st.session_state["messages"]) if m["role"] == "assistant"),
        None,
    )
    if last_assistant is not None:
        render_timeline(last_assistant.get("events", []), last_assistant.get("routed_to", []))


if __name__ == "__main__":
    main()
