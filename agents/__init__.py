"""AnyCompany Energy multi-agent customer-service application (Strands Agents SDK).

This package implements the agentic customer-service platform using the
`Strands Agents SDK <https://strandsagents.com>`_ with the **agents-as-tools**
multi-agent pattern:

* :mod:`agents.supervisor` builds the orchestrator (intent-router) agent. It
  owns a deterministic ``classify_intent`` tool plus three *specialist agents
  wrapped as tools*.
* :mod:`agents.specialists` builds the billing, scheduling and escalation
  specialist agents and exposes each as a callable ``@tool``.
* :mod:`agents.tools` holds the ``@tool`` functions (formerly Bedrock
  action-group Lambdas) that the specialists call.
* :mod:`agents.knowledge_base` builds the billing and escalation specialists'
  ``retrieve`` tool, which searches the Knowledge Base and screens each passage
  with the guardrail.
* :mod:`agents.intent` holds the keyword-scoring intent classifier.
* :mod:`agents.mock_data` is the deterministic in-memory datastore (unchanged
  business logic carried over from the original sample).
* :mod:`agents.config` resolves runtime configuration (region, model ids,
  Knowledge Base id, Guardrail id) from the environment.

The agents run **in process** (Strands is "just Python"), so the same code powers
the local Streamlit demo and can be deployed to Amazon Bedrock AgentCore Runtime
for production scale (see the project README "Production deployment").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["build_supervisor", "RuntimeConfig", "load_config"]

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from agents.config import RuntimeConfig, load_config
    from agents.supervisor import build_supervisor


def __getattr__(name: str) -> Any:
    """Lazily resolve top-level exports.

    Keeps ``agents.intent`` / ``agents.mock_data`` / ``agents.config`` importable
    without the Strands SDK installed (e.g. for unit tests), while still exposing
    ``agents.build_supervisor`` etc. at runtime.
    """
    if name in ("RuntimeConfig", "load_config"):
        from agents import config

        return getattr(config, name)
    if name == "build_supervisor":
        from agents.supervisor import build_supervisor

        return build_supervisor
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
