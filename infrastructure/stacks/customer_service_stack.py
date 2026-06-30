"""CDK stack for the AnyCompany Energy agentic customer-service platform.

Overview
--------
This stack provisions an Amazon Bedrock *multi-agent collaboration* topology for
"AnyCompany Energy" (the water/energy arm of AnyCompany Utilities). A single supervisor
agent (``anycompany-intent-router``) classifies each customer query and routes it
to one of three specialist agents:

* ``anycompany-billing-agent``     - accounts, balances, tariffs (also RAG-enabled
                                   against the policy/FAQ Knowledge Base).
* ``anycompany-scheduling-agent``  - appointment availability and bookings.
* ``anycompany-escalation-agent``  - complaints, vulnerability and human handoff.

The supervisor uses ``AgentCollaboratorType.SUPERVISOR_ROUTER`` so that Bedrock's
built-in routing classifier performs low-latency intent routing to the most
appropriate collaborator, while our own ``intent-classification`` action group
gives the supervisor a deterministic, auditable classification tool.

Construct library
-----------------
The higher-level Bedrock constructs come from
``cdklabs.generative-ai-cdk-constructs`` (imported below as ``bedrock``). That
library (pre-1.0) builds, among other things, the OpenSearch Serverless vector
collection + index for the Knowledge Base automatically, which is otherwise a
large amount of boilerplate. The L1 ``aws_cdk.aws_bedrock.CfnAgent`` types are
used only for the function-schema shape that the action groups require.

Key API notes (validated against cdklabs.generative-ai-cdk-constructs 0.1.318)
------------------------------------------------------------------------------
* Foundation models are tiered by role and built from raw Bedrock model-id
  strings via ``bedrock.BedrockFoundationModel(<model_id>, supports_agents=True)``:
  the supervisor/router uses ``anthropic.claude-opus-4-6-v1`` (routing
  intelligence) and the specialists use ``anthropic.claude-sonnet-4-6`` (fast,
  cost-efficient tool execution). NOTE: confirm these exact model ids are
  enabled in your account/region and whether they must be invoked via a
  cross-region inference profile (a ``us.anthropic.*`` style id); adjust the
  strings above if Bedrock requires the profile form.
* ``bedrock.ActionGroupExecutor.fromlambda_function(fn)`` (note: the factory is
  spelled ``fromlambda_function`` in the Python/jsii bindings, not
  ``from_lambda_function``).
* ``bedrock.AgentActionGroup(function_schema=...)`` expects an
  ``aws_cdk.aws_bedrock.CfnAgent.FunctionSchemaProperty``.
* ``bedrock.AgentCollaborator(agent_alias=..., collaboration_instruction=...,
  collaborator_name=..., relay_conversation_history=bool)`` requires an
  ``AgentAlias`` for each collaborator, so we create one alias per specialist.
"""

from __future__ import annotations

import logging
import os
from typing import Final

from aws_cdk import (
    Aws,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_bedrock as cfn_bedrock
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3_deployment
from cdklabs.generative_ai_cdk_constructs import bedrock
from cdklabs.generative_ai_cdk_constructs.bedrock import pii_type
from constructs import Construct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve filesystem paths to the sibling project directories. This stack lives
# in ``infrastructure/stacks/``; the knowledge base, Lambda handlers and shared
# layer live one level above the ``infrastructure`` directory.
# ---------------------------------------------------------------------------
_STACK_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
_INFRA_DIR: Final[str] = os.path.dirname(_STACK_DIR)
_PROJECT_ROOT: Final[str] = os.path.dirname(_INFRA_DIR)

_KNOWLEDGE_BASE_DIR: Final[str] = os.path.join(_PROJECT_ROOT, "knowledge-base")
_LAMBDAS_DIR: Final[str] = os.path.join(_PROJECT_ROOT, "lambdas")
_COMMON_LAYER_DIR: Final[str] = os.path.join(_LAMBDAS_DIR, "common")

# ---------------------------------------------------------------------------
# Configurable defaults (overridable via cdk.json `context` — see cdk.json).
# Keeping these as defaults means a default deploy reproduces the original
# AnyCompany sample exactly (no resource replacement), while a published copy can
# be rebranded for any company/industry purely through context.
# ---------------------------------------------------------------------------
_DEFAULT_COMPANY_NAME: Final[str] = "AnyCompany Energy"
_DEFAULT_COMPANY_SHORT_NAME: Final[str] = "AnyCompany"
# Resource-name prefix (lowercase, used in agent / KB / data-source names).
# Changing this REPLACES the Bedrock agents and Knowledge Base (new ids).
_DEFAULT_AGENT_PREFIX: Final[str] = "anycompany"
# Foundation models, tiered by role:
#   * Supervisor / intent router -> Claude Opus 4.6, for the strongest routing.
#   * Specialist agents -> Claude Sonnet 4.6, for fast tool execution.
# Both are invoked via US cross-region inference profiles (Claude 4.x cannot be
# called on-demand by bare model id).
_DEFAULT_SUPERVISOR_MODEL_ID: Final[str] = "anthropic.claude-opus-4-6-v1"
_DEFAULT_SPECIALIST_MODEL_ID: Final[str] = "anthropic.claude-sonnet-4-6"


# Supported cross-region inference-profile geographies (cdklabs enum). Claude
# Opus/Sonnet inference profiles are US-geo only today; EU/APAC are provided for
# forward-compatibility but are not currently usable for these models.
_INFERENCE_PROFILE_GEOS: Final = {
    "US": bedrock.CrossRegionInferenceProfileRegion.US,
    "EU": bedrock.CrossRegionInferenceProfileRegion.EU,
    "APAC": bedrock.CrossRegionInferenceProfileRegion.APAC,
}


def _detect_inference_geo(region: str) -> str:
    """Map an AWS region to its cross-region inference-profile geography.

    Used as the default when ``inferenceProfileGeo`` context is not set, so the
    correct ``us.`` / ``eu.`` / ``apac.`` profile prefix is chosen automatically
    from the deployment region.

    Args:
        region: The resolved deployment region (e.g. ``"us-west-2"``).

    Returns:
        One of ``"US"`` / ``"EU"`` / ``"APAC"`` (falls back to ``"US"`` for any
        unrecognised region prefix).

    Note:
        US and EU profiles for Claude Opus/Sonnet 4.6 are available today; an
        ``apac.`` profile for these models is not yet published (APAC regions
        expose ``global.`` / country-specific profiles instead). Override with
        ``inferenceProfileGeo`` if your region/model combination needs it.
    """
    region = (region or "").lower()
    if region.startswith("us-"):
        return "US"
    if region.startswith(("eu-", "me-", "af-")):
        return "EU"
    if region.startswith("ap-"):
        return "APAC"
    return "US"  # safe fallback


def _cross_region_profile(
    model_id: str,
    geo: "bedrock.CrossRegionInferenceProfileRegion",
) -> bedrock.CrossRegionInferenceProfile:
    """Wrap a base Bedrock model id in a cross-region inference profile.

    Claude 4.x models do not support on-demand invocation by their bare id; the
    profile (built from a ``supports_cross_region=True`` base model) is what the
    agents invoke, and it also provisions the matching cross-region IAM grants.

    Args:
        model_id: The base Bedrock foundation-model id (e.g.
            ``"anthropic.claude-opus-4-6-v1"``).
        geo: The geography for the cross-region inference profile (US/EU/APAC).
            Must match the AWS region group the stack deploys into.

    Returns:
        A :class:`bedrock.CrossRegionInferenceProfile` usable as an agent
        ``foundation_model``.
    """
    return bedrock.CrossRegionInferenceProfile.from_config(
        geo_region=geo,
        model=bedrock.BedrockFoundationModel(
            model_id,
            supports_agents=True,
            supports_cross_region=True,
        ),
    )


# Embeddings model used by the Knowledge Base vector index.
_EMBEDDINGS_MODEL: Final = bedrock.BedrockFoundationModel.TITAN_EMBED_TEXT_V2_1024


class MultiAgentCustomerServiceStack(Stack):
    """Provision the multi-agent customer-service platform (AnyCompany sample).

    The stack creates, in dependency order:

    1. An S3 bucket holding the markdown knowledge-base documents, populated via
       :class:`aws_cdk.aws_s3_deployment.BucketDeployment`.
    2. A :class:`bedrock.VectorKnowledgeBase` (Titan V2 embeddings) with an
       :class:`bedrock.S3DataSource` pointing at that bucket.
    3. A shared Lambda layer built from ``../lambdas/common``.
    4. Four Python 3.12 tool Lambdas (intent router + billing/scheduling/
       escalation), each attached to the shared layer, with X-Ray active tracing.
    5. A Bedrock Guardrail (content filters, denied topics, sensitive-info
       filters, contextual grounding) attached to all four agents.
    6. Four Bedrock agents: a ``SUPERVISOR_ROUTER`` supervisor plus three
       specialist collaborators; the billing and escalation specialists are also
       attached to the Knowledge Base for RAG.
    7. A CloudWatch dashboard + alarms for observability.
    8. ``CfnOutput`` values exposing the agent / KB / guardrail / bucket ids
       needed by the Streamlit demo client.

    Company branding, agent-name prefix, and foundation-model ids are all
    configurable via ``cdk.json`` context so the sample can be rebranded for any
    company/industry without code changes (defaults reproduce the AnyCompany demo).
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        """Initialise the stack and build all resources.

        Args:
            scope: The parent construct (the CDK ``App``).
            construct_id: The logical stack id (CloudFormation stack name).
            **kwargs: Standard :class:`aws_cdk.StackProps` (e.g. ``env``,
                ``description``), forwarded to the base :class:`Stack`.
        """
        super().__init__(scope, construct_id, **kwargs)

        logger.info("Synthesising %s", construct_id)

        # --- Configurable branding / models (cdk.json context) ----------- #
        # Defaults reproduce the AnyCompany sample exactly, so a default deploy
        # changes no resource identities. Override via `cdk deploy -c key=value`
        # or the `context` block in cdk.json.
        self.company_name: str = (
            self.node.try_get_context("companyName") or _DEFAULT_COMPANY_NAME
        )
        self.company_short_name: str = (
            self.node.try_get_context("companyShortName") or _DEFAULT_COMPANY_SHORT_NAME
        )
        self.agent_prefix: str = (
            self.node.try_get_context("agentPrefix") or _DEFAULT_AGENT_PREFIX
        )
        supervisor_model_id: str = (
            self.node.try_get_context("supervisorModelId") or _DEFAULT_SUPERVISOR_MODEL_ID
        )
        specialist_model_id: str = (
            self.node.try_get_context("specialistModelId") or _DEFAULT_SPECIALIST_MODEL_ID
        )
        # Cross-region inference-profile geography (US/EU/APAC). Auto-detected
        # from the deploy region by default (us- -> US, eu-/me-/af- -> EU,
        # ap- -> APAC); an explicit ``inferenceProfileGeo`` context value
        # overrides the auto-detection for edge cases. Must match the geography
        # of the deploy region (see README "Supported regions").
        geo_override = self.node.try_get_context("inferenceProfileGeo")
        geo_key = (
            str(geo_override).upper()
            if geo_override
            else _detect_inference_geo(self.region)
        )
        if geo_key not in _INFERENCE_PROFILE_GEOS:
            raise ValueError(
                f"Unsupported inferenceProfileGeo '{geo_key}'. "
                f"Valid values: {', '.join(_INFERENCE_PROFILE_GEOS)}."
            )
        inference_geo = _INFERENCE_PROFILE_GEOS[geo_key]
        # Optional ADOT (OpenTelemetry) Lambda layer ARN. When provided, the tool
        # Lambdas are auto-instrumented for OTel; when absent, X-Ray active
        # tracing still emits traces. Left empty by default so a deploy never
        # fails on a stale/region-mismatched layer ARN.
        self.otel_layer_arn: str | None = self.node.try_get_context("otelLayerArn") or None
        # Daily estimated-cost alarm threshold (USD).
        self.daily_cost_alarm_usd: float = float(
            self.node.try_get_context("dailyCostAlarmUsd") or 10
        )

        self._supervisor_model = _cross_region_profile(supervisor_model_id, inference_geo)
        self._specialist_model = _cross_region_profile(specialist_model_id, inference_geo)

        # Build the resource graph. Each helper returns the constructs that later
        # stages depend on, keeping construction order explicit and readable.
        # Knowledge Base documents bucket. Default: the stack creates its own
        # bucket and uploads the sample knowledge-base/*.md docs. If the
        # `existingKnowledgeBaseBucketArn` context key is set, reference that
        # bucket instead and skip the sample upload — the user manages their own
        # docs and runs the KB ingestion job themselves.
        existing_kb_bucket_arn = (
            self.node.try_get_context("existingKnowledgeBaseBucketArn") or None
        )
        if existing_kb_bucket_arn:
            docs_bucket = s3.Bucket.from_bucket_arn(
                self, "ImportedKnowledgeBaseDocsBucket", existing_kb_bucket_arn
            )
            # The user's bucket layout is unknown, so index the whole bucket
            # rather than forcing the sample's "knowledge-base/" prefix.
            kb_inclusion_prefixes = None
        else:
            docs_bucket = self._create_docs_bucket()
            self._deploy_knowledge_base_documents(docs_bucket)
            kb_inclusion_prefixes = ["knowledge-base/"]
        knowledge_base = self._create_knowledge_base(docs_bucket, kb_inclusion_prefixes)

        # Guardrail protecting every agent (content, topics, PII, grounding).
        guardrail = self._create_guardrail()
        self._guardrail = guardrail

        common_layer = self._create_common_layer()
        intent_fn = self._create_tool_lambda("IntentRouter", "intent_router", common_layer)
        billing_fn = self._create_tool_lambda("BillingTools", "billing_tools", common_layer)
        scheduling_fn = self._create_tool_lambda("SchedulingTools", "scheduling_tools", common_layer)
        escalation_fn = self._create_tool_lambda("EscalationTools", "escalation_tools", common_layer)

        # Specialist agents (the collaborators routed to by the supervisor).
        billing_agent = self._create_billing_agent(billing_fn, knowledge_base)
        scheduling_agent = self._create_scheduling_agent(scheduling_fn)
        escalation_agent = self._create_escalation_agent(escalation_fn, knowledge_base)

        # Each collaborator must be exposed to the supervisor via an alias.
        billing_alias = self._create_alias(billing_agent, "BillingAgentAlias", "live")
        scheduling_alias = self._create_alias(scheduling_agent, "SchedulingAgentAlias", "live")
        escalation_alias = self._create_alias(escalation_agent, "EscalationAgentAlias", "live")

        # The supervisor (intent router) collaborating over the three specialists.
        supervisor_agent = self._create_supervisor_agent(
            intent_fn,
            billing_alias,
            scheduling_alias,
            escalation_alias,
        )
        supervisor_alias = self._create_alias(
            supervisor_agent, "SupervisorAgentAlias", "live"
        )

        # Grant every agent permission to apply the guardrail at invocation time.
        for agent in (supervisor_agent, billing_agent, scheduling_agent, escalation_agent):
            guardrail.grant_apply(agent.role)

        # Wire Bedrock -> Lambda invoke permissions defensively. The construct's
        # ActionGroupExecutor.fromlambda_function should add a resource policy, but
        # we add explicit, narrowly-scoped permissions as belt-and-braces so the
        # demo never fails on AccessDenied during tool invocation.
        self._grant_bedrock_invoke(intent_fn, supervisor_agent.agent_arn)
        self._grant_bedrock_invoke(billing_fn, billing_agent.agent_arn)
        self._grant_bedrock_invoke(scheduling_fn, scheduling_agent.agent_arn)
        self._grant_bedrock_invoke(escalation_fn, escalation_agent.agent_arn)

        # Observability: CloudWatch dashboard + alarms over the tool Lambdas.
        self._create_observability(
            lambdas={
                "IntentRouter": intent_fn,
                "Billing": billing_fn,
                "Scheduling": scheduling_fn,
                "Escalation": escalation_fn,
            },
            guardrail=guardrail,
        )

        self._emit_outputs(
            supervisor_agent=supervisor_agent,
            supervisor_alias=supervisor_alias,
            billing_agent=billing_agent,
            scheduling_agent=scheduling_agent,
            escalation_agent=escalation_agent,
            knowledge_base=knowledge_base,
            docs_bucket=docs_bucket,
            guardrail=guardrail,
        )

    # ------------------------------------------------------------------ #
    # Branding helper
    # ------------------------------------------------------------------ #
    def _brandize(self, text: str) -> str:
        """Replace the sample's brand tokens with the configured company name.

        Replaces ``"AnyCompany Energy"`` then standalone ``"AnyCompany"`` so agent
        instructions and other brand-bearing strings reflect the configured
        company. With the default context this is a no-op.

        Args:
            text: Source text containing the sample brand.

        Returns:
            The rebranded text.
        """
        return text.replace(_DEFAULT_COMPANY_NAME, self.company_name).replace(
            _DEFAULT_COMPANY_SHORT_NAME, self.company_short_name
        )

    # ------------------------------------------------------------------ #
    # S3 + Knowledge Base
    # ------------------------------------------------------------------ #
    def _create_docs_bucket(self) -> s3.Bucket:
        """Create the S3 bucket that stores the knowledge-base markdown documents.

        The bucket blocks all public access and uses S3-managed encryption. For
        this demo it is configured for easy teardown: ``RemovalPolicy.DESTROY``
        plus ``auto_delete_objects`` so ``cdk destroy`` removes it cleanly. For a
        true production deployment you would switch to ``RETAIN`` and enable
        versioning + access logging.

        Returns:
            The created :class:`aws_cdk.aws_s3.Bucket`.
        """
        return s3.Bucket(
            self,
            "KnowledgeBaseDocsBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

    def _deploy_knowledge_base_documents(self, bucket: s3.Bucket) -> None:
        """Upload the local ``../knowledge-base`` markdown files into the bucket.

        Uses :class:`aws_cdk.aws_s3_deployment.BucketDeployment`, which packages
        the local directory and copies it to the bucket on each deploy. We confine
        the documents to a ``knowledge-base/`` prefix so the Knowledge Base data
        source can target it precisely.

        Args:
            bucket: The destination documents bucket.
        """
        s3_deployment.BucketDeployment(
            self,
            "DeployKnowledgeBaseDocs",
            sources=[s3_deployment.Source.asset(_KNOWLEDGE_BASE_DIR)],
            destination_bucket=bucket,
            destination_key_prefix="knowledge-base/",
            # Keep teardown clean for a demo: prune objects that disappear from the
            # local source rather than leaving orphans behind.
            prune=True,
            retain_on_delete=False,
        )

    def _create_knowledge_base(
        self,
        bucket: s3.IBucket,
        inclusion_prefixes: "list[str] | None",
    ) -> bedrock.VectorKnowledgeBase:
        """Create the Bedrock Knowledge Base with a Titan-V2-embedded vector index.

        ``bedrock.VectorKnowledgeBase`` (from generative-ai-cdk-constructs) creates
        and manages the Amazon OpenSearch Serverless vector collection, the vector
        index, and the data-access/encryption/network policies automatically when
        no explicit ``vector_store`` is supplied. We then attach an
        :class:`bedrock.S3DataSource` pointing at the docs bucket using a
        fixed-size chunking strategy suitable for short policy/FAQ documents.

        Args:
            bucket: The S3 bucket containing the documents (the sample's own
                bucket, or a user-provided existing bucket).
            inclusion_prefixes: S3 key prefixes the data source should index.
                ``["knowledge-base/"]`` for the sample's own bucket; ``None`` to
                index the whole bucket (used for a user-provided existing bucket
                whose layout is unknown).

        Returns:
            The created :class:`bedrock.VectorKnowledgeBase`.
        """
        knowledge_base = bedrock.VectorKnowledgeBase(
            self,
            "AnyCompanyPolicyKnowledgeBase",
            name=f"{self.agent_prefix}-policy-knowledge-base",
            embeddings_model=_EMBEDDINGS_MODEL,
            # Instruction tells the agent WHEN to query this KB during orchestration.
            # NOTE: when a KB is attached to an agent this string becomes the
            # association description, which the construct validates at <= 200 chars.
            instruction=self._brandize(
                "AnyCompany Energy policies: tariffs, payment plans, arrears and "
                "disconnection, vulnerable customers, complaints, refunds, direct "
                "debit, meter readings and appointments."
            ),
        )

        # The S3 data source feeds the markdown documents into the vector index.
        # Fixed-size chunking with overlap works well for the short, sectioned
        # policy/FAQ markdown used in this project.
        bedrock.S3DataSource(
            self,
            "PolicyDocsDataSource",
            knowledge_base=knowledge_base,
            bucket=bucket,
            data_source_name=f"{self.agent_prefix}-policy-docs",
            inclusion_prefixes=inclusion_prefixes,
            chunking_strategy=bedrock.ChunkingStrategy.fixed_size(
                max_tokens=500,
                overlap_percentage=20,
            ),
            # Remove ingested vectors when the data source is deleted so teardown
            # leaves no orphaned index entries.
            data_deletion_policy=bedrock.DataDeletionPolicy.DELETE,
        )

        return knowledge_base

    # ------------------------------------------------------------------ #
    # Guardrail
    # ------------------------------------------------------------------ #
    def _create_guardrail(self) -> bedrock.Guardrail:
        """Create a Bedrock Guardrail applied to every agent.

        Protects all four agents with:
          * Content filters (hate, insults, sexual, violence, misconduct,
            prompt-attack) at HIGH strength.
          * One denied topic: ``FINANCIAL_ADVICE`` (managed). Competitor and
            "other customers' data" policies are intentionally enforced via agent
            instructions instead of denied topics — see the README "Guardrail
            design notes" for the rationale.
          * Sensitive-information filters that REDACT (ANONYMIZE) sort codes,
            phone numbers, emails and card numbers. UK postcodes and ``ACC-NNNNNN``
            account numbers are intentionally NOT filtered (no detect-only mode
            exists for regex filters and the demo must surface them) — the
            commented-out filters can be enabled for production. See the README
            "Guardrail PII tradeoff".
          * Contextual grounding + relevance scored at threshold 0.7 in
            detect-only (NONE) mode by default; flip to BLOCK to hard-enforce
            grounding (see "Production hardening" in the README).

        Returns:
            The created :class:`bedrock.Guardrail`.
        """
        guardrail = bedrock.Guardrail(
            self,
            "CustomerServiceGuardrail",
            name=f"{self.agent_prefix}-cs-guardrail",
            description=(
                f"Safety guardrail for the {self.company_name} multi-agent "
                "customer-service platform."
            ),
            blocked_input_messaging=(
                "I'm sorry, but I can't help with that request. I can help with "
                "your account, billing, appointments, or a complaint."
            ),
            blocked_outputs_messaging=(
                "I'm sorry, but I can't share that. I can help with your account, "
                "billing, appointments, or a complaint."
            ),
        )

        # --- Content filters (HIGH on input + output) -------------------- #
        for ftype in (
            bedrock.ContentFilterType.HATE,
            bedrock.ContentFilterType.INSULTS,
            bedrock.ContentFilterType.SEXUAL,
            bedrock.ContentFilterType.VIOLENCE,
            bedrock.ContentFilterType.MISCONDUCT,
        ):
            guardrail.add_content_filter(
                type=ftype,
                input_strength=bedrock.ContentFilterStrength.HIGH,
                output_strength=bedrock.ContentFilterStrength.HIGH,
            )
        # Prompt-attack protection is input-only (Bedrock requires output NONE).
        guardrail.add_content_filter(
            type=bedrock.ContentFilterType.PROMPT_ATTACK,
            input_strength=bedrock.ContentFilterStrength.HIGH,
            output_strength=bedrock.ContentFilterStrength.NONE,
        )

        # --- Denied topics ----------------------------------------------- #
        # FINANCIAL_ADVICE (managed) only. A custom "CompetitorDiscussion" topic
        # was evaluated and deliberately REMOVED for this published sample: the
        # ~200-char topic-definition limit makes it impossible to be specific
        # enough to exclude legitimate references (e.g. the Energy Ombudsman,
        # regulators, the customer's own switching/complaints rights) without
        # over-matching, and topic classification is probabilistic — a fuzzy deny
        # rule intermittently blocked the core escalation/RAG flow. Design rule:
        # use Guardrails for HARD safety (PII, prompt-injection, harmful content)
        # and agent INSTRUCTIONS for soft, context-dependent business policy. The
        # agents are already instructed not to recommend competitors or discuss
        # other customers' accounts. See the README "Guardrail design notes" and
        # "Production hardening" sections.
        guardrail.add_denied_topic_filter(bedrock.Topic.FINANCIAL_ADVICE)

        # --- Sensitive information: REDACT (anonymise) ------------------- #
        for ptype in (
            pii_type.General.PHONE,
            pii_type.General.EMAIL,
            pii_type.Finance.CREDIT_DEBIT_CARD_NUMBER,
            pii_type.Finance.CREDIT_DEBIT_CARD_CVV,
            pii_type.Finance.CREDIT_DEBIT_CARD_EXPIRY,
        ):
            guardrail.add_pii_filter(type=ptype, action=bedrock.GuardrailAction.ANONYMIZE)
        # UK bank sort code (NN-NN-NN) — redact.
        guardrail.add_regex_filter(
            name="UKSortCode",
            description="UK bank sort code in NN-NN-NN format.",
            pattern=r"\b\d{2}-\d{2}-\d{2}\b",
            action=bedrock.GuardrailAction.ANONYMIZE,
        )

        # --- Account number / postcode: intentionally NOT filtered -------- #
        # Decision: keep these DETECT-ONLY / opt-in. Bedrock sensitive-info regex
        # filters support only ANONYMIZE or BLOCK (there is no detect-only mode),
        # and this demo legitimately echoes ACC-NNNNNN account numbers and UK
        # postcodes (billing/scheduling depend on the agent surfacing them), so
        # adding either filter would break the happy-path demo. To ENFORCE
        # redaction in a real deployment, uncomment the filters below (ANONYMIZE)
        # — see the README "Guardrail PII tradeoff".
        #
        # guardrail.add_regex_filter(
        #     name="AccountNumber",
        #     description="Internal account number in ACC-NNNNNN format.",
        #     pattern=r"\bACC-\d{6}\b",
        #     action=bedrock.GuardrailAction.ANONYMIZE,
        # )
        # guardrail.add_regex_filter(
        #     name="UKPostcode",
        #     description="UK postcode.",
        #     pattern=r"\b[A-Za-z]{1,2}\d[A-Za-z\d]?\s*\d[A-Za-z]{2}\b",
        #     action=bedrock.GuardrailAction.ANONYMIZE,
        # )

        # --- Contextual grounding (RAG answers must be grounded) --------- #
        # Threshold 0.7 as specified. Action is NONE (detect/measure) by DEFAULT
        # so the platform's conversational, empathy-framed RAG answers (e.g. the
        # escalation agent's complaints replies) are scored for grounding/relevance
        # — visible in traces and CloudWatch — without being blocked, which would
        # break the demo. Set action=GuardrailAction.BLOCK to hard-enforce grounding
        # in a production deployment. See README "Guardrail design notes".
        guardrail.add_contextual_grounding_filter(
            type=bedrock.ContextualGroundingFilterType.GROUNDING,
            threshold=0.7,
            action=bedrock.GuardrailAction.NONE,
        )
        guardrail.add_contextual_grounding_filter(
            type=bedrock.ContextualGroundingFilterType.RELEVANCE,
            threshold=0.7,
            action=bedrock.GuardrailAction.NONE,
        )

        return guardrail

    # ------------------------------------------------------------------ #
    # Lambda layer + tool Lambdas
    # ------------------------------------------------------------------ #
    def _create_common_layer(self) -> lambda_.LayerVersion:
        """Build the shared Lambda layer from ``../lambdas/common``.

        The ``lambdas/common`` directory contains a ``python/`` folder (the
        layout the Lambda Python runtime expects for layers) holding
        ``bedrock_agent.py`` (the action-group request/response helpers) and
        ``mock_data.py`` (the in-memory mock datastore). Each tool Lambda imports
        these directly, e.g. ``from bedrock_agent import dispatch``.

        Returns:
            The created :class:`aws_cdk.aws_lambda.LayerVersion`.
        """
        return lambda_.LayerVersion(
            self,
            "CommonHelpersLayer",
            code=lambda_.Code.from_asset(
                _COMMON_LAYER_DIR,
                exclude=["__pycache__", "**/__pycache__", "*.pyc"],
            ),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description=(
                "Shared Bedrock action-group helpers (bedrock_agent) and the "
                "in-memory mock customer datastore (mock_data) for AnyCompany tools."
            ),
        )

    def _create_tool_lambda(
        self,
        construct_id: str,
        handler_dir: str,
        common_layer: lambda_.LayerVersion,
    ) -> lambda_.Function:
        """Create a Python 3.12 tool Lambda for a Bedrock action group.

        All tool Lambdas share the same shape: a ``handler.py`` exposing
        ``lambda_handler``, the common helper layer attached, a 30 second timeout
        and 256 MB of memory. Mock data is held in process memory inside the layer,
        so no extra IAM/VPC wiring is required. CloudWatch log groups are capped at
        a two-week retention (this is a sandbox/demo; there is no need to keep logs
        indefinitely, which would otherwise accrue storage cost).

        Args:
            construct_id: The CDK logical id for the function.
            handler_dir: The directory name under ``../lambdas`` holding
                ``handler.py`` (e.g. ``"billing_tools"``).
            common_layer: The shared helper layer to attach.

        Returns:
            The created :class:`aws_cdk.aws_lambda.Function`.
        """
        layers: list[lambda_.ILayerVersion] = [common_layer]
        env: dict[str, str] = {
            # Surface useful runtime context; the handlers default to INFO.
            "LOG_LEVEL": "INFO",
            "POWERTOOLS_SERVICE_NAME": f"{self.agent_prefix}-{handler_dir}",
        }

        # Optional OpenTelemetry (ADOT) auto-instrumentation. When an ADOT layer
        # ARN is supplied via context, attach it and set the exec wrapper so the
        # handler is auto-instrumented for OTel traces (exported alongside X-Ray).
        # Left off by default so a deploy never fails on a stale layer ARN; X-Ray
        # active tracing (below) still emits distributed traces regardless.
        if self.otel_layer_arn:
            layers.append(
                lambda_.LayerVersion.from_layer_version_arn(
                    self, f"{construct_id}OtelLayer", self.otel_layer_arn
                )
            )
            env["AWS_LAMBDA_EXEC_WRAPPER"] = "/opt/otel-instrument"

        return lambda_.Function(
            self,
            construct_id,
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(
                os.path.join(_LAMBDAS_DIR, handler_dir),
                exclude=["__pycache__", "**/__pycache__", "*.pyc"],
            ),
            layers=layers,
            timeout=Duration.seconds(30),
            memory_size=256,
            # X-Ray active tracing: emits a distributed trace per invocation so
            # agent -> action-group-Lambda calls are observable end to end.
            tracing=lambda_.Tracing.ACTIVE,
            # Explicit log group with 14-day retention (sandbox/demo) using the
            # current API — avoids the deprecated `log_retention` custom resource.
            log_group=logs.LogGroup(
                self,
                f"{construct_id}LogGroup",
                retention=logs.RetentionDays.TWO_WEEKS,
                removal_policy=RemovalPolicy.DESTROY,
            ),
            description=(
                f"{self.company_short_name} Bedrock action-group tool Lambda "
                f"({handler_dir}). Invoked by Amazon Bedrock agents using the "
                "function-schema request/response contract."
            ),
            environment=env,
        )

    def _grant_bedrock_invoke(self, fn: lambda_.Function, agent_arn: str) -> None:
        """Allow the Amazon Bedrock service to invoke a tool Lambda.

        ``ActionGroupExecutor.fromlambda_function`` is expected to attach a
        resource policy automatically, but we add an explicit, source-scoped
        permission as a defensive measure so action-group invocation cannot fail on
        AccessDenied. The permission is scoped to the ``bedrock.amazonaws.com``
        principal, this account, and the specific agent ARN via the
        ``aws:SourceArn`` condition (applied through ``source_arn``).

        Args:
            fn: The tool Lambda to grant invoke on.
            agent_arn: The ARN of the agent that owns the action group.
        """
        fn.add_permission(
            f"AllowBedrockInvoke-{fn.node.id}",
            principal=iam.ServicePrincipal("bedrock.amazonaws.com"),
            action="lambda:InvokeFunction",
            source_account=Aws.ACCOUNT_ID,
            source_arn=agent_arn,
        )

    # ------------------------------------------------------------------ #
    # Action-group function schemas
    # ------------------------------------------------------------------ #
    @staticmethod
    def _param(type_: str, description: str, *, required: bool) -> cfn_bedrock.CfnAgent.ParameterDetailProperty:
        """Build a single action-group function parameter detail.

        Args:
            type_: The Bedrock parameter primitive type (``"string"``, ``"number"``,
                ``"integer"``, ``"boolean"`` or ``"array"``).
            description: Human/LLM-facing description guiding when and how to fill
                the parameter.
            required: Whether the agent must supply the parameter.

        Returns:
            A :class:`aws_cdk.aws_bedrock.CfnAgent.ParameterDetailProperty`.
        """
        return cfn_bedrock.CfnAgent.ParameterDetailProperty(
            type=type_,
            description=description,
            required=required,
        )

    def _intent_function_schema(self) -> cfn_bedrock.CfnAgent.FunctionSchemaProperty:
        """Function schema for the supervisor's ``intent-classification`` action group.

        Returns:
            The function schema describing ``classify_intent``.
        """
        return cfn_bedrock.CfnAgent.FunctionSchemaProperty(
            functions=[
                cfn_bedrock.CfnAgent.FunctionProperty(
                    name="classify_intent",
                    description=(
                        "Classify a customer's free-text query into one of " +
                        "BILLING, SCHEDULING, ESCALATION or UNKNOWN using a " +
                        "keyword/heuristic scorer. Returns JSON with the intent, a " +
                        "0..1 confidence, the recommended specialist agent and a " +
                        "short reasoning string."
                    ),
                    parameters={
                        "query": self._param(
                            "string",
                            "The customer's raw question or request to classify.",
                            required=True,
                        ),
                    },
                ),
            ]
        )

    def _billing_function_schema(self) -> cfn_bedrock.CfnAgent.FunctionSchemaProperty:
        """Function schema for the ``billing-tools`` action group.

        Returns:
            The function schema describing ``lookup_account`` and ``check_balance``.
        """
        return cfn_bedrock.CfnAgent.FunctionSchemaProperty(
            functions=[
                cfn_bedrock.CfnAgent.FunctionProperty(
                    name="lookup_account",
                    description=(
                        "Look up a customer account profile by account number. " +
                        "Returns name, tariff, status, postcode and whether the " +
                        "account is on a payment plan."
                    ),
                    parameters={
                        "account_number": self._param(
                            "string",
                            "The AnyCompany account number, e.g. ACC-100123.",
                            required=True,
                        ),
                    },
                ),
                cfn_bedrock.CfnAgent.FunctionProperty(
                    name="check_balance",
                    description=(
                        "Retrieve the current balance and payment status for an " +
                        "account. Returns balance_gbp, due_date, last_payment and " +
                        "status."
                    ),
                    parameters={
                        "account_number": self._param(
                            "string",
                            "The AnyCompany account number, e.g. ACC-100123.",
                            required=True,
                        ),
                    },
                ),
            ]
        )

    def _scheduling_function_schema(self) -> cfn_bedrock.CfnAgent.FunctionSchemaProperty:
        """Function schema for the ``scheduling-tools`` action group.

        Returns:
            The function schema describing ``check_availability``,
            ``book_appointment`` and ``cancel_appointment``.
        """
        return cfn_bedrock.CfnAgent.FunctionSchemaProperty(
            functions=[
                cfn_bedrock.CfnAgent.FunctionProperty(
                    name="check_availability",
                    description=(
                        "List available appointment slots for an appointment type " +
                        "and postcode, optionally near a preferred ISO date. " +
                        "Returns a JSON list of slots with slot_id, date, " +
                        "time_window and engineer_type."
                    ),
                    parameters={
                        "appointment_type": self._param(
                            "string",
                            "One of METER_READING, ENGINEER_VISIT, "
                            "LEAK_INVESTIGATION or EV_CHARGER_INSTALL.",
                            required=True,
                        ),
                        "postcode": self._param(
                            "string",
                            "The customer's UK postcode, e.g. M14 5TG.",
                            required=True,
                        ),
                        "preferred_date": self._param(
                            "string",
                            "Optional preferred date in ISO format (YYYY-MM-DD).",
                            required=False,
                        ),
                    },
                ),
                cfn_bedrock.CfnAgent.FunctionProperty(
                    name="book_appointment",
                    description=(
                        "Book a previously listed slot for an account. Returns the " +
                        "booking_reference, confirmed_date, confirmed_time_window, " +
                        "appointment_type and a CONFIRMED status."
                    ),
                    parameters={
                        "account_number": self._param(
                            "string",
                            "The AnyCompany account number the booking is for.",
                            required=True,
                        ),
                        "slot_id": self._param(
                            "string",
                            "The slot_id returned by check_availability.",
                            required=True,
                        ),
                        "contact_phone": self._param(
                            "string",
                            "A contact phone number for the engineer to confirm "
                            "the visit.",
                            required=True,
                        ),
                    },
                ),
                cfn_bedrock.CfnAgent.FunctionProperty(
                    name="cancel_appointment",
                    description=(
                        "Cancel an existing appointment by its booking reference. " +
                        "Returns the booking_reference and a CANCELLED status."
                    ),
                    parameters={
                        "booking_reference": self._param(
                            "string",
                            "The booking reference to cancel, e.g. APPT-100001.",
                            required=True,
                        ),
                    },
                ),
            ]
        )

    def _escalation_function_schema(self) -> cfn_bedrock.CfnAgent.FunctionSchemaProperty:
        """Function schema for the ``escalation-tools`` action group.

        Returns:
            The function schema describing ``create_ticket`` and
            ``transfer_to_agent``.
        """
        return cfn_bedrock.CfnAgent.FunctionSchemaProperty(
            functions=[
                cfn_bedrock.CfnAgent.FunctionProperty(
                    name="create_ticket",
                    description=(
                        "Open a case/ticket for an issue that cannot be resolved by " +
                        "automation. Returns ticket_id, status OPEN, category, " +
                        "priority, sla_hours and created_at."
                    ),
                    parameters={
                        "account_number": self._param(
                            "string",
                            "The AnyCompany account number the ticket relates to.",
                            required=True,
                        ),
                        "issue_summary": self._param(
                            "string",
                            "A concise summary of the customer's issue.",
                            required=True,
                        ),
                        "category": self._param(
                            "string",
                            "One of BILLING_DISPUTE, SUPPLY_ISSUE, COMPLAINT, "
                            "VULNERABILITY or OTHER.",
                            required=True,
                        ),
                        "priority": self._param(
                            "string",
                            "Optional priority: LOW, MEDIUM (default), HIGH or "
                            "CRITICAL.",
                            required=False,
                        ),
                    },
                ),
                cfn_bedrock.CfnAgent.FunctionProperty(
                    name="transfer_to_agent",
                    description=(
                        "Hand the conversation off to a human agent via Amazon " +
                        "Connect. Returns transfer_id, ticket_id, queue, " +
                        "assigned_team, estimated_wait_minutes, connect_contact_id " +
                        "and a TRANSFERRED status."
                    ),
                    parameters={
                        "ticket_id": self._param(
                            "string",
                            "The ticket_id created for this issue.",
                            required=True,
                        ),
                        "reason": self._param(
                            "string",
                            "Why the transfer is needed; used to infer the target "
                            "queue and team.",
                            required=True,
                        ),
                        "account_number": self._param(
                            "string",
                            "Optional account number to attach to the handoff.",
                            required=False,
                        ),
                    },
                ),
            ]
        )

    # ------------------------------------------------------------------ #
    # Agents
    # ------------------------------------------------------------------ #
    def _create_billing_agent(
        self,
        billing_fn: lambda_.Function,
        knowledge_base: bedrock.VectorKnowledgeBase,
    ) -> bedrock.Agent:
        """Create the billing specialist agent (action group + Knowledge Base).

        This agent owns the ``billing-tools`` action group and is also attached to
        the policy Knowledge Base so it can ground tariff/policy explanations in
        AnyCompany's published documents (RAG).

        Args:
            billing_fn: The billing tool Lambda.
            knowledge_base: The policy Knowledge Base to attach for RAG.

        Returns:
            The created :class:`bedrock.Agent`.
        """
        instruction = (
            "# Role\n"
            "You are the AnyCompany Energy Billing Specialist, a careful and "
            "empathetic UK utility billing assistant for a water/energy supplier. "
            "You operate as a collaborator under an intent-router supervisor: you "
            "receive queries that have already been classified as billing-related "
            "(balances, payments, tariffs, refunds, direct debit, arrears).\n\n"
            "# Scope\n"
            "Help customers understand their account: current and historic "
            "balances, due dates, last payment, account status, their tariff and "
            "how it works, payment plans, direct debit, and general billing policy. "
            "You may also answer policy/eligibility questions using the attached "
            "knowledge base.\n\n"
            "# Tool-use rules\n"
            "1. ALWAYS identify the account first. If the customer has not provided "
            "an account number in the form ACC-NNNNNN, ask for it before calling "
            "any tool. Never invent or guess an account number.\n"
            "2. Use `lookup_account` to retrieve the customer's profile (name, "
            "tariff, status, postcode, payment-plan flag) when you need account "
            "context.\n"
            "3. Use `check_balance` to retrieve the live balance, due date, last "
            "payment and status before quoting any figures. Never state a balance "
            "from memory.\n"
            "4. Quote money in GBP to two decimal places. State whether an amount is "
            "owed, in credit, or in arrears.\n\n"
            "# When to defer to the knowledge base\n"
            "For questions about HOW a policy works (tariff definitions, payment-plan "
            "eligibility, arrears/disconnection process, refund timelines, direct "
            "debit rules, meter reading cadence), consult the knowledge base and cite "
            "the relevant policy. Do not fabricate policy details that are not in the "
            "documents. Never invent specific contact details (phone numbers, email "
            "addresses, URLs/web links, postal addresses) or named bodies: only state "
            "such specifics if they appear verbatim in the knowledge base. If a "
            "customer needs a contact channel that the documents do not provide, say "
            "you don't have that specific detail and offer to escalate rather than "
            "guessing a number or address.\n\n"
            "# Escalation triggers (hand back to the supervisor)\n"
            "If the customer disputes a charge and is unhappy, mentions financial "
            "hardship or vulnerability, threatens a complaint/ombudsman/legal action, "
            "reports a supply emergency (no water, major leak), or asks for a human, "
            "explain that you will route them to the right team and defer back to the "
            "supervisor so the Escalation Specialist can take over. Do not attempt to "
            "open tickets or transfer to humans yourself.\n\n"
            "# Tone and compliance\n"
            "Be warm, plain-spoken and concise; avoid jargon. This is a regulated UK "
            "utility, so be accurate and never make promises about waivers, refunds "
            "or disconnections that policy does not support. Only reference the "
            "account the customer has authenticated against; never disclose data "
            "about other accounts. Do not collect full card numbers or passwords.\n\n"
            "# What NOT to do\n"
            "Do not provide legal, tax or financial advice; do not process payments "
            "(this demo cannot take card details); do not modify tariffs or balances; "
            "do not speculate about figures you have not retrieved with a tool."
        )

        instruction = self._brandize(instruction)
        agent = bedrock.Agent(
            self,
            "BillingAgent",
            name=f"{self.agent_prefix}-billing-agent",
            foundation_model=self._specialist_model,
            instruction=instruction,
            guardrail=self._guardrail,
            knowledge_bases=[knowledge_base],
            should_prepare_agent=True,
            idle_session_ttl=Duration.minutes(15),
            description=f"{self.company_short_name} billing specialist (accounts, balances, tariffs, RAG).",
        )

        agent.add_action_group(
            bedrock.AgentActionGroup(
                name="billing-tools",
                description=(
                    "Account lookup and balance retrieval tools backed by the "
                    "billing_tools Lambda."
                ),
                executor=bedrock.ActionGroupExecutor.fromlambda_function(billing_fn),
                function_schema=self._billing_function_schema(),
                enabled=True,
            )
        )
        return agent

    def _create_scheduling_agent(self, scheduling_fn: lambda_.Function) -> bedrock.Agent:
        """Create the scheduling specialist agent (``scheduling-tools`` action group).

        Args:
            scheduling_fn: The scheduling tool Lambda.

        Returns:
            The created :class:`bedrock.Agent`.
        """
        instruction = (
            "# Role\n"
            "You are the AnyCompany Energy Scheduling Specialist, responsible for "
            "booking and managing field appointments for a UK water/energy supplier. "
            "You operate as a collaborator under an intent-router supervisor and "
            "receive queries already classified as scheduling-related (appointments, "
            "meter readings, engineer visits, EV charger installs, reschedules and "
            "cancellations).\n\n"
            "# Scope\n"
            "Find appointment availability, book confirmed slots and cancel existing "
            "bookings. Supported appointment types are exactly: METER_READING, "
            "ENGINEER_VISIT, LEAK_INVESTIGATION and EV_CHARGER_INSTALL.\n\n"
            "# Tool-use rules\n"
            "1. Determine the correct appointment_type from the customer's words: a "
            "meter reading -> METER_READING; a general fault/visit -> ENGINEER_VISIT; "
            "a suspected leak/damp/no-water investigation -> LEAK_INVESTIGATION; an EV "
            "charge point -> EV_CHARGER_INSTALL. If genuinely ambiguous, ask one "
            "clarifying question.\n"
            "2. You need a postcode to check availability. If it is missing, ask for "
            "it (a UK postcode such as LS6 2AB).\n"
            "3. Call `check_availability` and present 2-3 clear options (date and "
            "time window). Quote the exact slot_id internally; show friendly "
            "date/time to the customer.\n"
            "4. To book, call `book_appointment` with the account_number, the chosen "
            "slot_id and a contact_phone. Confirm the booking_reference back to the "
            "customer and the date/time window.\n"
            "5. To cancel, call `cancel_appointment` with the booking_reference. "
            "Confirm the cancellation.\n"
            "6. Never invent slot ids, references, dates or engineer availability. "
            "Only state what the tools return.\n\n"
            "# Escalation triggers (hand back to the supervisor)\n"
            "If the customer describes an emergency (active leak causing damage, no "
            "water supply, gas-like smell), treat it as urgent: gather the postcode, "
            "and defer back to the supervisor so the Escalation Specialist can raise "
            "a priority case and arrange a human/engineer response. Do not downplay "
            "safety issues by offering a routine appointment alone.\n\n"
            "# Tone and compliance\n"
            "Be efficient, friendly and precise about dates and times. Use clear "
            "UK-style dates. Confirm details before booking. Only act on the "
            "account the customer is authenticated against.\n\n"
            "# What NOT to do\n"
            "Do not promise specific engineers or exact arrival times beyond the "
            "stated time window; do not double-book; do not handle billing disputes "
            "or open complaint tickets (defer those back to the supervisor)."
        )

        instruction = self._brandize(instruction)
        agent = bedrock.Agent(
            self,
            "SchedulingAgent",
            name=f"{self.agent_prefix}-scheduling-agent",
            foundation_model=self._specialist_model,
            instruction=instruction,
            guardrail=self._guardrail,
            should_prepare_agent=True,
            idle_session_ttl=Duration.minutes(15),
            description=f"{self.company_short_name} scheduling specialist (availability, bookings, cancellations).",
        )

        agent.add_action_group(
            bedrock.AgentActionGroup(
                name="scheduling-tools",
                description=(
                    "Appointment availability, booking and cancellation tools backed "
                    "by the scheduling_tools Lambda."
                ),
                executor=bedrock.ActionGroupExecutor.fromlambda_function(scheduling_fn),
                function_schema=self._scheduling_function_schema(),
                enabled=True,
            )
        )
        return agent

    def _create_escalation_agent(
        self,
        escalation_fn: lambda_.Function,
        knowledge_base: bedrock.VectorKnowledgeBase,
    ) -> bedrock.Agent:
        """Create the escalation specialist agent (``escalation-tools`` action group).

        Args:
            escalation_fn: The escalation tool Lambda.
            knowledge_base: The policy Knowledge Base, attached for RAG so the
                agent can cite authoritative contact details (e.g. the Energy
                Ombudsman / CCW) and complaints-policy specifics rather than
                inventing them.

        Returns:
            The created :class:`bedrock.Agent`.
        """
        instruction = (
            "# Role\n"
            "You are the AnyCompany Energy Escalation Specialist for a regulated UK "
            "water/energy supplier. You handle complaints, billing disputes, supply "
            "emergencies, vulnerable-customer situations and any case that needs a "
            "human. You operate as a collaborator under an intent-router supervisor "
            "and receive queries already classified as escalation-related.\n\n"
            "# Scope\n"
            "Acknowledge the customer's concern with genuine empathy, capture the "
            "issue accurately, open a tracked case, and where appropriate hand off to "
            "the right human team via Amazon Connect.\n\n"
            "# Tool-use rules\n"
            "1. First, briefly acknowledge and summarise the issue back to the "
            "customer so they feel heard.\n"
            "2. Use `create_ticket` to open a case. Choose the category carefully: "
            "BILLING_DISPUTE (charges/refunds the customer disputes), SUPPLY_ISSUE "
            "(no water, leaks, outages, supply quality), COMPLAINT (service/conduct "
            "complaints), VULNERABILITY (health, disability, financial hardship, "
            "safeguarding, priority-services-register situations) or OTHER. Set "
            "priority by severity: CRITICAL for safety/supply emergencies or acute "
            "vulnerability, HIGH for significant detriment, MEDIUM by default, LOW "
            "for minor matters. Always treat genuine vulnerability as at least HIGH "
            "and emergencies as CRITICAL.\n"
            "3. The tool returns an SLA in hours; tell the customer when they can "
            "expect a response and give them the ticket_id for reference.\n"
            "4. When the situation needs a person (emergency, vulnerability, an "
            "unresolved or sensitive complaint, or an explicit request for a human), "
            "use `transfer_to_agent` with the ticket_id and a clear reason. Relay the "
            "queue/team and the estimated wait, and reassure the customer they will "
            "not have to repeat everything.\n"
            "5. If you do not have an account number for create_ticket, ask for it; "
            "for an active emergency, prioritise safety guidance and still raise the "
            "case even with limited detail.\n\n"
            "# Vulnerability and safety handling\n"
            "Take any mention of health risk, disability, mental health, "
            "safeguarding, age-related need, or being unable to afford essential "
            "supply extremely seriously. Be calm, supportive and never dismissive. "
            "Default such cases to high priority and prefer a human handoff. For "
            "imminent safety risks (gas smell, flooding, electrical hazard) advise "
            "the appropriate emergency action and escalate immediately.\n\n"
            "# Tone and compliance\n"
            "Lead with empathy; be calm, respectful and solution-focused. As a "
            "regulated utility, never dismiss a complaint, never discourage a "
            "customer from escalating to the Ombudsman, and be transparent about next "
            "steps and timeframes. Handle personal data carefully and only for the "
            "authenticated account.\n\n"
            "# What NOT to do\n"
            "Do not promise specific compensation, refunds or outcomes; do not close "
            "or resolve the underlying complaint yourself; do not argue with the "
            "customer; do not delay raising a case for a genuine emergency or "
            "vulnerability. Never invent specific contact details. You ARE attached "
            "to the AnyCompany knowledge base, which holds the authoritative contact "
            "details for the independent redress schemes (the Energy Ombudsman and "
            "the Consumer Council for Water). When a customer asks how to reach the "
            "Ombudsman or CCW, consult the knowledge base and quote the phone number, "
            "email, website or postal address exactly as written there. If a specific "
            "contact detail is not present in the knowledge base, say you don't have "
            "that exact detail and that the team you are transferring them to (or "
            "their deadlock letter) will provide it — never guess or fabricate it."
        )

        instruction = self._brandize(instruction)
        agent = bedrock.Agent(
            self,
            "EscalationAgent",
            name=f"{self.agent_prefix}-escalation-agent",
            foundation_model=self._specialist_model,
            instruction=instruction,
            guardrail=self._guardrail,
            knowledge_bases=[knowledge_base],
            should_prepare_agent=True,
            idle_session_ttl=Duration.minutes(15),
            description=f"{self.company_short_name} escalation specialist (tickets, vulnerability, human handoff).",
        )

        agent.add_action_group(
            bedrock.AgentActionGroup(
                name="escalation-tools",
                description=(
                    "Case/ticket creation and Amazon Connect human-handoff tools "
                    "backed by the escalation_tools Lambda."
                ),
                executor=bedrock.ActionGroupExecutor.fromlambda_function(escalation_fn),
                function_schema=self._escalation_function_schema(),
                enabled=True,
            )
        )
        return agent

    def _create_supervisor_agent(
        self,
        intent_fn: lambda_.Function,
        billing_alias: bedrock.AgentAlias,
        scheduling_alias: bedrock.AgentAlias,
        escalation_alias: bedrock.AgentAlias,
    ) -> bedrock.Agent:
        """Create the supervisor (intent-router) agent with multi-agent collaboration.

        The supervisor uses ``AgentCollaboratorType.SUPERVISOR_ROUTER`` so Bedrock's
        routing classifier dispatches each turn to the best-fit collaborator. We
        additionally give the supervisor an ``intent-classification`` action group so
        it can produce an explicit, auditable classification (intent + confidence +
        reasoning) which is invaluable for the demo's routing timeline.

        Each collaborator is referenced via its alias and given a precise
        ``collaboration_instruction`` describing exactly when it should be selected.

        Args:
            intent_fn: The intent-router tool Lambda.
            billing_alias: Alias of the billing specialist agent.
            scheduling_alias: Alias of the scheduling specialist agent.
            escalation_alias: Alias of the escalation specialist agent.

        Returns:
            The created supervisor :class:`bedrock.Agent`.
        """
        instruction = (
            "# Role\n"
            "You are the AnyCompany Energy customer-service Intent Router and "
            "Supervisor for a UK water/energy supplier. You are the single front "
            "door for customer conversations. Your job is to understand each request, "
            "classify its intent, route it to the correct specialist, and stitch the "
            "specialists' work into one coherent, friendly reply.\n\n"
            "# Routing rules\n"
            "Classify every customer turn into BILLING, SCHEDULING, ESCALATION or "
            "UNKNOWN, then route accordingly:\n"
            "- BILLING (balances, payments, tariffs, refunds, direct debit, arrears, "
            "statements) -> the BillingSpecialist.\n"
            "- SCHEDULING (appointments, meter readings, engineer visits, EV charger "
            "installs, reschedule/cancel) -> the SchedulingSpecialist.\n"
            "- ESCALATION (complaints, manager/ombudsman/legal requests, "
            "vulnerability, emergencies such as a major leak or no water, or any "
            "explicit request for a human) -> the EscalationSpecialist.\n"
            "Use the `classify_intent` tool to produce an explicit intent, "
            "confidence and reasoning for the current request; rely on it (together "
            "with your own judgement) to decide routing. If the intent is UNKNOWN or "
            "low-confidence, ask one concise clarifying question before routing.\n\n"
            "# Multi-intent and safety\n"
            "If a request spans multiple intents (e.g. 'my bill is wrong AND I want a "
            "meter reading'), prioritise safety and vulnerability first, then handle "
            "the remaining intents in turn, routing to each specialist as needed and "
            "combining their results. Any sign of an emergency, vulnerability or a "
            "formal complaint must be routed to the EscalationSpecialist promptly, "
            "even if other intents are present.\n\n"
            "# Tone and synthesis\n"
            "Greet warmly, be concise and reassuring, and never expose internal "
            "routing mechanics, tool names or agent names to the customer. When a "
            "specialist returns information, present it as a single, seamless "
            "AnyCompany Energy response. Always confirm next steps.\n\n"
            "# Compliance and what NOT to do\n"
            "This is a regulated UK utility. Be accurate, do not invent account data, "
            "policies, prices or outcomes, and only act on the account the customer "
            "is authenticated against. Never invent specific contact details (phone "
            "numbers, email addresses, URLs or postal addresses) when relaying a "
            "specialist's answer; only pass on contact specifics that the specialist "
            "actually provided from a tool or the knowledge base. Do not give legal, "
            "tax or financial advice. Do "
            "not attempt to perform billing, scheduling or escalation actions "
            "yourself - always delegate to the appropriate specialist."
        )

        instruction = self._brandize(instruction)
        agent = bedrock.Agent(
            self,
            "SupervisorAgent",
            name=f"{self.agent_prefix}-intent-router",
            foundation_model=self._supervisor_model,
            instruction=instruction,
            guardrail=self._guardrail,
            # SUPERVISOR_ROUTER enables Bedrock's built-in low-latency routing
            # classifier to dispatch turns to the most appropriate collaborator.
            agent_collaboration=bedrock.AgentCollaboratorType.SUPERVISOR_ROUTER,
            agent_collaborators=[
                bedrock.AgentCollaborator(
                    agent_alias=billing_alias,
                    collaborator_name="BillingSpecialist",
                    collaboration_instruction=(
                        "Route here for anything about accounts, balances, payments, " +
                        "tariffs, refunds, direct debit, statements or arrears. This " +
                        "specialist can look up the account and check the balance, and " +
                        "can ground policy answers in the AnyCompany knowledge base."
                    ),
                    # Relay history so the specialist has full context of the turn.
                    relay_conversation_history=True,
                ),
                bedrock.AgentCollaborator(
                    agent_alias=scheduling_alias,
                    collaborator_name="SchedulingSpecialist",
                    collaboration_instruction=(
                        "Route here to check appointment availability, book or cancel " +
                        "appointments for METER_READING, ENGINEER_VISIT, " +
                        "LEAK_INVESTIGATION or EV_CHARGER_INSTALL visits."
                    ),
                    relay_conversation_history=True,
                ),
                bedrock.AgentCollaborator(
                    agent_alias=escalation_alias,
                    collaborator_name="EscalationSpecialist",
                    collaboration_instruction=(
                        "Route here for complaints, billing disputes the customer is " +
                        "unhappy about, supply emergencies (no water, leaks), " +
                        "vulnerable-customer situations, requests for a manager / " +
                        "Ombudsman / legal, or any explicit request to speak to a " +
                        "human. This specialist opens cases and can transfer to a " +
                        "human via Amazon Connect."
                    ),
                    relay_conversation_history=True,
                ),
            ],
            should_prepare_agent=True,
            idle_session_ttl=Duration.minutes(15),
            description="AnyCompany supervisor / intent router (SUPERVISOR_ROUTER multi-agent collaboration).",
        )

        # The supervisor's own deterministic classification tool.
        agent.add_action_group(
            bedrock.AgentActionGroup(
                name="intent-classification",
                description=(
                    "Deterministic keyword/heuristic intent classifier backed by the "
                    "intent_router Lambda."
                ),
                executor=bedrock.ActionGroupExecutor.fromlambda_function(intent_fn),
                function_schema=self._intent_function_schema(),
                enabled=True,
            )
        )
        return agent

    def _create_alias(
        self,
        agent: bedrock.Agent,
        construct_id: str,
        alias_name: str,
    ) -> bedrock.AgentAlias:
        """Create a deployable alias for an agent.

        Aliases are required for two reasons: (1) the supervisor's
        ``AgentCollaborator`` entries must each reference a collaborator *alias*,
        and (2) the demo client invokes the supervisor through an alias id. Creating
        an explicit alias also pins a versioned snapshot of the prepared agent.

        Args:
            agent: The agent to alias.
            construct_id: The CDK logical id for the alias construct.
            alias_name: A human-friendly label used only in the alias description.
                It is deliberately NOT passed as the alias name: omitting the name
                lets the cdklabs construct use its default ``latest-{hash}`` name,
                whose hash changes whenever the agent definition changes. That is
                what drives the alias to roll a fresh agent version on every deploy
                that alters the agent (e.g. a foundation-model change). Passing a
                static name freezes the alias on the version cut at first creation.

        Returns:
            The created :class:`bedrock.AgentAlias`.
        """
        return bedrock.AgentAlias(
            self,
            construct_id,
            agent=agent,
            description=f"Deployable '{alias_name}' alias for {agent.node.id}.",
        )

    # ------------------------------------------------------------------ #
    # Observability (CloudWatch dashboard + alarms)
    # ------------------------------------------------------------------ #
    def _create_observability(
        self,
        *,
        lambdas: dict[str, lambda_.Function],
        guardrail: bedrock.Guardrail,
    ) -> None:
        """Create a CloudWatch dashboard and alarms over the platform.

        The tool Lambdas are the observable proxy for agent activity (each agent's
        action-group calls land on its Lambda; supervisor routing lands on the
        intent-router Lambda). The dashboard surfaces invocations/agent, latency,
        escalation rate, and guardrail activity; alarms cover per-agent error rate
        and an estimated-cost guard.

        Args:
            lambdas: Mapping of display label -> tool Lambda (IntentRouter,
                Billing, Scheduling, Escalation).
            guardrail: The Bedrock guardrail (for intervention metrics).
        """
        intent_fn = lambdas["IntentRouter"]
        billing_fn = lambdas["Billing"]
        scheduling_fn = lambdas["Scheduling"]
        escalation_fn = lambdas["Escalation"]

        # Invocations per agent per hour (Sum, 1h period).
        invocation_metrics = [
            fn.metric_invocations(
                period=Duration.hours(1), statistic="Sum", label=label
            )
            for label, fn in lambdas.items()
        ]

        # Average latency per specialist (Average, 5m period).
        latency_metrics = [
            lambdas[label].metric_duration(
                period=Duration.minutes(5), statistic="Average", label=f"{label} avg ms"
            )
            for label in ("Billing", "Scheduling", "Escalation")
        ]

        # Escalation rate = escalation invocations / total invocations (%).
        escalation_rate = cloudwatch.MathExpression(
            expression="100 * esc / (intent + bill + sched + esc)",
            using_metrics={
                "intent": intent_fn.metric_invocations(period=Duration.hours(1), statistic="Sum"),
                "bill": billing_fn.metric_invocations(period=Duration.hours(1), statistic="Sum"),
                "sched": scheduling_fn.metric_invocations(period=Duration.hours(1), statistic="Sum"),
                "esc": escalation_fn.metric_invocations(period=Duration.hours(1), statistic="Sum"),
            },
            label="Escalation rate (%)",
            period=Duration.hours(1),
        )

        dashboard = cloudwatch.Dashboard(
            self,
            "ObservabilityDashboard",
            dashboard_name=f"{self.agent_prefix}-customer-service",
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Invocations per agent (hourly)",
                left=invocation_metrics,
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Average latency per specialist (ms)",
                left=latency_metrics,
                width=12,
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Escalation rate (%)",
                left=[escalation_rate],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Guardrail invocations vs interventions",
                left=[
                    guardrail.metric_invocations(period=Duration.hours(1), label="Guardrail invocations"),
                    guardrail.metric_invocations_intervened(period=Duration.hours(1), label="Interventions"),
                ],
                width=12,
            ),
        )
        # RAG retrieval count / empty-result rate is not exposed as a built-in
        # CloudWatch metric (retrieval happens inside the Bedrock-managed agent,
        # not our Lambdas). Document how to surface it rather than show an empty
        # graph: enable Bedrock model-invocation logging to CloudWatch and add a
        # metric filter on retrievalResults length == 0 for the empty-result rate.
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown=(
                    "### RAG retrieval count & empty-result rate\n"
                    "RAG retrieval is performed inside the Bedrock-managed agents, "
                    "so it is not emitted as a default CloudWatch metric. To populate "
                    "this panel, enable **Bedrock model-invocation logging** to a "
                    "CloudWatch log group and add metric filters: a `Retrieve` count "
                    "and an empty-result count (where `retrievalResults` is empty). "
                    "Then add those metrics here. The **Guardrail interventions** "
                    "panel above is a useful proxy for blocked/low-grounding answers."
                ),
                width=24,
                height=4,
            ),
        )

        # --- Alarms ------------------------------------------------------ #
        # Per-agent error rate > 5% over 5 minutes.
        for label, fn in lambdas.items():
            error_rate = cloudwatch.MathExpression(
                expression="100 * (errors / invocations)",
                using_metrics={
                    "errors": fn.metric_errors(period=Duration.minutes(5), statistic="Sum"),
                    "invocations": fn.metric_invocations(period=Duration.minutes(5), statistic="Sum"),
                },
                label=f"{label} error rate (%)",
                period=Duration.minutes(5),
            )
            error_rate.create_alarm(
                self,
                f"{label}ErrorRateAlarm",
                threshold=5,
                evaluation_periods=1,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                # No traffic should not fire the alarm.
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                alarm_description=f"{self.company_short_name} {label} agent error rate exceeded 5% over 5 minutes.",
            )

        # Estimated-cost guard. AWS/Billing EstimatedCharges is account-level and
        # only published to us-east-1; it requires "Receive Billing Alerts" to be
        # enabled. This is a pragmatic spend tripwire — for precise per-day cost
        # tracking use AWS Budgets / Cost Explorer (see README).
        cost_alarm_metric = cloudwatch.Metric(
            namespace="AWS/Billing",
            metric_name="EstimatedCharges",
            dimensions_map={"Currency": "USD"},
            statistic="Maximum",
            period=Duration.hours(6),
        )
        cost_alarm_metric.create_alarm(
            self,
            "EstimatedCostAlarm",
            threshold=self.daily_cost_alarm_usd,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.MISSING,
            alarm_description=(
                f"Estimated AWS charges exceeded ${self.daily_cost_alarm_usd:.0f} "
                "(account-level EstimatedCharges; enable billing alerts in us-east-1)."
            ),
        )

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def _emit_outputs(
        self,
        *,
        supervisor_agent: bedrock.Agent,
        supervisor_alias: bedrock.AgentAlias,
        billing_agent: bedrock.Agent,
        scheduling_agent: bedrock.Agent,
        escalation_agent: bedrock.Agent,
        knowledge_base: bedrock.VectorKnowledgeBase,
        docs_bucket: s3.Bucket,
        guardrail: bedrock.Guardrail,
    ) -> None:
        """Emit the CloudFormation outputs consumed by the Streamlit demo client.

        The output *names* are part of the project contract and must match exactly,
        because the demo and docs reference them. The supervisor agent id + alias id
        are the two values the demo strictly requires to call
        ``invoke_agent``; the remaining ids aid debugging and console navigation.

        Args:
            supervisor_agent: The intent-router supervisor agent.
            supervisor_alias: The supervisor's deployable alias.
            billing_agent: The billing specialist agent.
            scheduling_agent: The scheduling specialist agent.
            escalation_agent: The escalation specialist agent.
            knowledge_base: The policy Knowledge Base.
            docs_bucket: The S3 documents bucket.
        """
        CfnOutput(
            self,
            "SupervisorAgentId",
            value=supervisor_agent.agent_id,
            description="Agent id of the supervisor (anycompany-intent-router). Set as SUPERVISOR_AGENT_ID in the demo.",
        )
        CfnOutput(
            self,
            "SupervisorAgentAliasId",
            value=supervisor_alias.alias_id,
            description="Alias id of the supervisor agent. Set as SUPERVISOR_AGENT_ALIAS_ID in the demo.",
        )
        CfnOutput(
            self,
            "BillingAgentId",
            value=billing_agent.agent_id,
            description="Agent id of the billing specialist (anycompany-billing-agent).",
        )
        CfnOutput(
            self,
            "SchedulingAgentId",
            value=scheduling_agent.agent_id,
            description="Agent id of the scheduling specialist (anycompany-scheduling-agent).",
        )
        CfnOutput(
            self,
            "EscalationAgentId",
            value=escalation_agent.agent_id,
            description="Agent id of the escalation specialist (anycompany-escalation-agent).",
        )
        CfnOutput(
            self,
            "KnowledgeBaseId",
            value=knowledge_base.knowledge_base_id,
            description="Id of the AnyCompany policy knowledge base (Titan V2 embeddings).",
        )
        CfnOutput(
            self,
            "DocsBucketName",
            value=docs_bucket.bucket_name,
            description="Name of the S3 bucket holding the knowledge-base markdown documents.",
        )
        CfnOutput(
            self,
            "Region",
            value=self.region,
            description="AWS region the stack is deployed to. Set as AWS_REGION in the demo.",
        )
        CfnOutput(
            self,
            "GuardrailId",
            value=guardrail.guardrail_id,
            description="Id of the Bedrock guardrail applied to all agents. Document as GUARDRAIL_ID.",
        )
        CfnOutput(
            self,
            "GuardrailArn",
            value=guardrail.guardrail_arn,
            description="ARN of the Bedrock guardrail applied to all agents.",
        )
