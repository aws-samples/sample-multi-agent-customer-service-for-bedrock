"""Runtime configuration for the Strands customer-service agents.

Configuration is resolved from environment variables so the same code runs
locally (Streamlit demo), in CI, or on Amazon Bedrock AgentCore Runtime. Wire
the values from the CDK stack outputs (Knowledge Base id, Guardrail id, region)
via ``demo/.env`` — see ``demo/.env.example``.

Region -> inference-profile geo is auto-detected (``us-*`` -> ``us.``,
``eu-*``/``me-*``/``af-*`` -> ``eu.``, ``ap-*`` -> ``apac.``) so the correct
cross-region inference profile is selected automatically; override with
``INFERENCE_PROFILE_GEO`` for edge cases. See the README "Supported regions".
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_REGION: Final[str] = "us-east-1"

# Base foundation models, tiered by role. The supervisor (router) uses Claude
# Opus for the strongest routing/multi-intent reasoning; the specialists use
# Claude Sonnet for fast, cost-efficient tool execution. Both are invoked via a
# cross-region inference profile (the ``<geo>.`` prefix is added below) because
# Claude 4.x cannot be invoked on-demand by its bare model id.
# Opus 4.6 requires the -v1 suffix; Sonnet 4.6 uses the bare alias. Verified against bedrock list-inference-profiles.
DEFAULT_SUPERVISOR_BASE_MODEL: Final[str] = "anthropic.claude-opus-4-6-v1"
DEFAULT_SPECIALIST_BASE_MODEL: Final[str] = "anthropic.claude-sonnet-4-6"

# Valid inference-profile geo prefixes.
_VALID_GEOS: Final[dict[str, str]] = {"US": "us", "EU": "eu", "APAC": "apac"}


def detect_geo(region: str) -> str:
    """Map an AWS region to its cross-region inference-profile geography.

    Args:
        region: The deployment region (e.g. ``"us-west-2"``).

    Returns:
        One of ``"US"`` / ``"EU"`` / ``"APAC"`` (falls back to ``"US"`` for any
        unrecognised region prefix).

    Note:
        US and EU profiles for Claude Opus/Sonnet 4.6 are available today; an
        ``apac.`` profile for these models is not yet published. Override with
        ``INFERENCE_PROFILE_GEO`` if your region/model combination needs it.
    """
    r = (region or "").lower()
    if r.startswith("us-"):
        return "US"
    if r.startswith(("eu-", "me-", "af-")):
        return "EU"
    if r.startswith("ap-"):
        return "APAC"
    return "US"


def _profile_id(base_model_id: str, geo_key: str) -> str:
    """Prefix a base model id with the inference-profile geo (e.g. ``us.``)."""
    prefix = _VALID_GEOS[geo_key]
    # Already a profile id (starts with a known geo prefix)? Leave it as-is.
    if base_model_id.split(".", 1)[0] in _VALID_GEOS.values():
        return base_model_id
    return f"{prefix}.{base_model_id}"


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable runtime configuration for the agent application.

    Attributes:
        region: AWS region the agents call Bedrock in.
        supervisor_model_id: Cross-region inference-profile id for the supervisor.
        specialist_model_id: Cross-region inference-profile id for specialists.
        knowledge_base_id: Bedrock Knowledge Base id used by the RAG retrieve
            tool (empty string disables RAG gracefully).
        guardrail_id: Bedrock Guardrail id applied to every model invocation
            (empty string disables the guardrail wiring).
        guardrail_version: Guardrail version to apply (default ``DRAFT``).
    """

    region: str
    supervisor_model_id: str
    specialist_model_id: str
    knowledge_base_id: str
    guardrail_id: str
    guardrail_version: str

    @property
    def guardrail_enabled(self) -> bool:
        """Whether a guardrail id is configured."""
        return bool(self.guardrail_id.strip())

    @property
    def knowledge_base_enabled(self) -> bool:
        """Whether a Knowledge Base id is configured."""
        return bool(self.knowledge_base_id.strip())


def load_config() -> RuntimeConfig:
    """Build a :class:`RuntimeConfig` from environment variables.

    Reads (all optional except where noted):
      * ``AWS_REGION`` — deploy region (default ``us-east-1``).
      * ``STRANDS_KNOWLEDGE_BASE_ID`` (or ``KNOWLEDGE_BASE_ID``) — KB id for RAG.
      * ``GUARDRAIL_ID`` / ``GUARDRAIL_VERSION`` — Bedrock guardrail (version
        defaults to ``DRAFT``).
      * ``SUPERVISOR_MODEL_ID`` / ``SPECIALIST_MODEL_ID`` — base model overrides.
      * ``INFERENCE_PROFILE_GEO`` — override the region-detected geo
        (``US``/``EU``/``APAC``).

    Returns:
        The resolved :class:`RuntimeConfig`. As a convenience, the resolved KB id
        is also exported to ``STRANDS_KNOWLEDGE_BASE_ID`` so the community
        ``retrieve`` tool picks it up.
    """
    region = (os.environ.get("AWS_REGION") or DEFAULT_REGION).strip()

    geo_override = (os.environ.get("INFERENCE_PROFILE_GEO") or "").strip().upper()
    geo_key = geo_override if geo_override in _VALID_GEOS else detect_geo(region)
    if geo_override and geo_override not in _VALID_GEOS:
        logger.warning(
            "Ignoring invalid INFERENCE_PROFILE_GEO '%s'; auto-detected '%s' instead.",
            geo_override,
            geo_key,
        )
    if geo_key == "APAC":
        # APAC profiles for Claude Opus/Sonnet 4.6 are not published yet, so the
        # apac.anthropic.* model ids built below will fail with a Bedrock
        # validationException at invocation time. Warn loudly so a user whose
        # credentials default to an ap-* region understands why calls fail.
        logger.warning(
            "Region '%s' resolves to the APAC inference-profile geo, but Claude "
            "Opus/Sonnet 4.6 have no apac.anthropic.* profile yet, so model calls "
            "will fail. Set AWS_REGION to a US or EU region (or override "
            "INFERENCE_PROFILE_GEO) — see the README 'Supported regions'.",
            region,
        )

    supervisor_base = (
        os.environ.get("SUPERVISOR_MODEL_ID") or DEFAULT_SUPERVISOR_BASE_MODEL
    ).strip()
    specialist_base = (
        os.environ.get("SPECIALIST_MODEL_ID") or DEFAULT_SPECIALIST_BASE_MODEL
    ).strip()

    kb_id = (
        os.environ.get("STRANDS_KNOWLEDGE_BASE_ID")
        or os.environ.get("KNOWLEDGE_BASE_ID")
        or ""
    ).strip()
    # The community `retrieve` tool reads STRANDS_KNOWLEDGE_BASE_ID at call time.
    if kb_id:
        os.environ["STRANDS_KNOWLEDGE_BASE_ID"] = kb_id

    config = RuntimeConfig(
        region=region,
        supervisor_model_id=_profile_id(supervisor_base, geo_key),
        specialist_model_id=_profile_id(specialist_base, geo_key),
        knowledge_base_id=kb_id,
        guardrail_id=(os.environ.get("GUARDRAIL_ID") or "").strip(),
        guardrail_version=(os.environ.get("GUARDRAIL_VERSION") or "DRAFT").strip(),
    )
    logger.info(
        "Resolved runtime config: region=%s geo=%s supervisor=%s specialist=%s "
        "kb=%s guardrail=%s",
        config.region,
        geo_key,
        config.supervisor_model_id,
        config.specialist_model_id,
        "set" if config.knowledge_base_enabled else "unset",
        "set" if config.guardrail_enabled else "unset",
    )
    return config


def build_model(model_id: str, config: RuntimeConfig, *, temperature: float = 0.3):
    """Create a Strands :class:`BedrockModel` with the project's guardrail wired.

    The Bedrock Guardrail (content filters, denied topics, PII ANONYMIZE,
    prompt-attack detection, contextual grounding) is applied natively at the
    model layer via the ``guardrail_*`` parameters, so every supervisor and
    specialist invocation is guarded with no extra wrapper code.

    Args:
        model_id: The cross-region inference-profile model id to invoke.
        config: The resolved runtime configuration.
        temperature: Sampling temperature (low by default for consistent routing
            and tool use).

    Returns:
        A configured :class:`strands.models.BedrockModel`.
    """
    # Imported lazily so the package can be imported (e.g. for unit tests of the
    # intent classifier or mock data) without the SDK installed.
    from strands.models import BedrockModel

    kwargs: dict[str, object] = {
        "model_id": model_id,
        "region_name": config.region,
        "temperature": temperature,
    }
    if config.guardrail_enabled:
        kwargs.update(
            guardrail_id=config.guardrail_id,
            guardrail_version=config.guardrail_version,
            guardrail_trace="enabled",
        )
    return BedrockModel(**kwargs)
