"""CDK stack for the AnyCompany Energy agentic customer-service platform.

Overview
--------
This stack provisions the **supporting AWS infrastructure** for the agentic
customer-service platform whose agents run on the **Strands Agents SDK** (see the
``agents/`` package). The multi-agent orchestration itself is plain Python — a
supervisor (intent router) that routes to billing / scheduling / escalation
specialists via the *agents-as-tools* pattern — so it runs in process for the
demo and can be deployed to **Amazon Bedrock AgentCore Runtime** for production
(see the README "Production deployment"). It is no longer a set of
Bedrock-managed agents, so this stack provisions only the durable, shared
resources the agents depend on:

* A **Bedrock Knowledge Base** (Amazon OpenSearch Serverless vector index +
  Titan Text Embeddings V2) over the policy/FAQ markdown documents in S3 — the
  RAG source for the billing and escalation specialists.
* A **Bedrock Guardrail** (content filters, denied topics, PII ANONYMIZE,
  prompt-attack detection; contextual grounding configured but not evaluated)
  that the Strands agents apply on every model invocation via the model layer.
* **CloudWatch** observability — a dashboard (Bedrock invocation/latency/error
  metrics + guardrail activity) and alarms.

Construct library
-----------------
The Knowledge Base and Guardrail constructs come from
``cdklabs.generative-ai-cdk-constructs`` (imported below as ``bedrock``), which
builds the OpenSearch Serverless vector collection + index and the guardrail
policies with far less boilerplate than the raw L1s.

Branding, the resource-name prefix, the (default) foundation-model ids and the
stack name are configurable via ``cdk.json`` context, so the sample can be
rebranded for any company/industry without code changes.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Final

from aws_cdk import (
    CfnOutput,
    Duration,
    Fn,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_bedrock as bedrock_l1
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_opensearchserverless as aoss
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3_deployment
from cdklabs.generative_ai_cdk_constructs import bedrock
from cdklabs.generative_ai_cdk_constructs.bedrock import pii_type
from constructs import Construct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve filesystem paths to the sibling project directories.
# ---------------------------------------------------------------------------
_STACK_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
_INFRA_DIR: Final[str] = os.path.dirname(_STACK_DIR)
_PROJECT_ROOT: Final[str] = os.path.dirname(_INFRA_DIR)

_KNOWLEDGE_BASE_DIR: Final[str] = os.path.join(_PROJECT_ROOT, "knowledge-base")

# ---------------------------------------------------------------------------
# Configurable defaults (overridable via cdk.json `context`).
# Defaults reproduce the original "AnyCompany Energy" sample exactly.
# ---------------------------------------------------------------------------
_DEFAULT_COMPANY_NAME: Final[str] = "AnyCompany Energy"
_DEFAULT_COMPANY_SHORT_NAME: Final[str] = "AnyCompany"
# Resource-name prefix (lowercase, used in KB / guardrail / dashboard names).
_DEFAULT_AGENT_PREFIX: Final[str] = "anycompany"

# Default foundation models the Strands agents use (the runtime reads its own
# copies from demo/.env; these are kept here so the CloudWatch dashboard can
# graph the matching Bedrock model metrics). Supervisor -> Claude Opus 4.6,
# specialists -> Claude Sonnet 4.6, invoked via cross-region inference profiles.
# Opus 4.6 requires the -v1 suffix; Sonnet 4.6 uses the bare alias. Verified against bedrock list-inference-profiles.
_DEFAULT_SUPERVISOR_MODEL_ID: Final[str] = "anthropic.claude-opus-4-6-v1"
_DEFAULT_SPECIALIST_MODEL_ID: Final[str] = "anthropic.claude-sonnet-4-6"

# Embeddings model used by the Knowledge Base vector index.
_EMBEDDINGS_MODEL: Final = bedrock.BedrockFoundationModel.TITAN_EMBED_TEXT_V2_1024

# Valid inference-profile geo prefixes (string form, used for dashboard metrics).
_GEO_PREFIXES: Final[dict[str, str]] = {"US": "us", "EU": "eu", "APAC": "apac"}


def _detect_geo(region: str) -> str:
    """Map an AWS region to its cross-region inference-profile geography.

    ``us-*`` -> US, ``eu-*``/``me-*``/``af-*`` -> EU, ``ap-*`` -> APAC, with a US
    fallback. See the README "Supported regions".
    """
    r = (region or "").lower()
    if r.startswith("us-"):
        return "US"
    if r.startswith(("eu-", "me-", "af-")):
        return "EU"
    if r.startswith("ap-"):
        return "APAC"
    return "US"


def _profile_model_id(base_model_id: str, geo_key: str) -> str:
    """Prefix a base model id with the inference-profile geo (e.g. ``us.``)."""
    if base_model_id.split(".", 1)[0] in _GEO_PREFIXES.values():
        return base_model_id
    return f"{_GEO_PREFIXES[geo_key]}.{base_model_id}"



def _construct_child(parent: Construct, child_id: str, expected: type) -> object:
    """Return a construct's child, failing synth if the construct library changed.

    Used for overrides of generative-ai-cdk-constructs internals, which are
    validated against the pinned version (0.1.318).
    """
    child = parent.node.try_find_child(child_id)
    if not isinstance(child, expected):
        raise RuntimeError(
            f"Expected a {expected.__name__} '{child_id}' under {parent.node.path}; "
            "generative-ai-cdk-constructs may have changed. Re-check the overrides in "
            "customer_service_stack.py."
        )
    return child


def _vpc_endpoint_name(prefix: str) -> str:
    """Return a valid OpenSearch Serverless VPC endpoint name for the resource prefix.

    Endpoint names allow only 3-32 lowercase letters, digits and hyphens, starting
    with a letter, which is stricter than the other resources the prefix names.
    """
    name = re.sub(r"[^a-z0-9-]", "-", f"{prefix}-kb-index".lower())
    if not name[0].isalpha():
        name = f"kb-{name}"
    return name[:32]

class MultiAgentCustomerServiceStack(Stack):
    """Provision the supporting infrastructure (KB + Guardrail + observability).

    The stack creates, in dependency order:

    1. An S3 bucket holding the ``knowledge-base/*.md`` documents (or references a
       user-provided existing bucket) and a ``BucketDeployment`` that uploads
       them.
    2. A Bedrock Knowledge Base (OpenSearch Serverless + Titan V2) with an S3
       data source (fixed-size chunking, 500 tokens / 20% overlap). The vector
       collection has no public network access.
    3. A Bedrock Guardrail (content filters, denied topics, PII ANONYMIZE,
       prompt-attack detection; contextual grounding configured but not
       evaluated).
    4. A CloudWatch dashboard + alarms.
    5. ``CfnOutput`` values (Knowledge Base id, Guardrail id/arn, bucket, region)
       consumed by the Strands agent runtime / Streamlit demo via ``demo/.env``.

    Company branding, resource-name prefix and the default model ids are all
    configurable via ``cdk.json`` context.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        """Initialise the stack and build all resources.

        Args:
            scope: The parent construct (the CDK ``App``).
            construct_id: The logical stack id (CloudFormation stack name).
            **kwargs: Standard :class:`aws_cdk.StackProps`, forwarded to ``Stack``.
        """
        super().__init__(scope, construct_id, **kwargs)

        logger.info("Synthesising %s", construct_id)

        # --- Configurable branding / models (cdk.json context) ----------- #
        self.company_name: str = (
            self.node.try_get_context("companyName") or _DEFAULT_COMPANY_NAME
        )
        self.company_short_name: str = (
            self.node.try_get_context("companyShortName") or _DEFAULT_COMPANY_SHORT_NAME
        )
        self.agent_prefix: str = (
            self.node.try_get_context("agentPrefix") or _DEFAULT_AGENT_PREFIX
        )
        # Default models the runtime uses (also graphed on the dashboard). The geo
        # prefix is auto-detected from the deploy region (override via context).
        supervisor_model_id: str = (
            self.node.try_get_context("supervisorModelId") or _DEFAULT_SUPERVISOR_MODEL_ID
        )
        specialist_model_id: str = (
            self.node.try_get_context("specialistModelId") or _DEFAULT_SPECIALIST_MODEL_ID
        )
        geo_override = self.node.try_get_context("inferenceProfileGeo")
        geo_key = (
            str(geo_override).upper()
            if geo_override
            else _detect_geo(self.region)
        )
        if geo_key not in _GEO_PREFIXES:
            raise ValueError(
                f"Unsupported inferenceProfileGeo '{geo_key}'. "
                f"Valid values: {', '.join(_GEO_PREFIXES)}."
            )
        self._supervisor_profile_id = _profile_model_id(supervisor_model_id, geo_key)
        self._specialist_profile_id = _profile_model_id(specialist_model_id, geo_key)

        # Daily estimated-cost alarm threshold (USD).
        self.daily_cost_alarm_usd: float = float(
            self.node.try_get_context("dailyCostAlarmUsd") or 10
        )

        # --- Knowledge Base documents bucket ----------------------------- #
        # Default: the stack creates its own bucket and uploads the sample
        # knowledge-base/*.md docs. If `existingKnowledgeBaseBucketArn` is set,
        # reference that bucket instead and skip the sample upload — the user
        # manages their own docs and runs the KB ingestion job themselves.
        existing_kb_bucket_arn = (
            self.node.try_get_context("existingKnowledgeBaseBucketArn") or None
        )
        if existing_kb_bucket_arn:
            docs_bucket = s3.Bucket.from_bucket_arn(
                self, "ImportedKnowledgeBaseDocsBucket", existing_kb_bucket_arn
            )
            kb_inclusion_prefixes = None
        else:
            docs_bucket = self._create_docs_bucket()
            self._deploy_knowledge_base_documents(docs_bucket)
            kb_inclusion_prefixes = ["knowledge-base/"]
        knowledge_base = self._create_knowledge_base(docs_bucket, kb_inclusion_prefixes)
        self._make_vector_collection_private(knowledge_base)

        # Guardrail applied by the Strands agents on every model invocation.
        guardrail = self._create_guardrail()

        # Observability: dashboard + alarms over Bedrock + guardrail activity.
        self._create_observability(guardrail=guardrail)

        self._emit_outputs(
            knowledge_base=knowledge_base,
            docs_bucket=docs_bucket,
            guardrail=guardrail,
        )

    # ------------------------------------------------------------------ #
    # Branding helper
    # ------------------------------------------------------------------ #
    def _brandize(self, text: str) -> str:
        """Replace the sample brand tokens with the configured company name."""
        return text.replace(_DEFAULT_COMPANY_NAME, self.company_name).replace(
            _DEFAULT_COMPANY_SHORT_NAME, self.company_short_name
        )

    # ------------------------------------------------------------------ #
    # S3 + Knowledge Base
    # ------------------------------------------------------------------ #
    def _create_docs_bucket(self) -> s3.Bucket:
        """Create the S3 bucket that stores the knowledge-base markdown documents.

        Blocks all public access and uses S3-managed encryption. For this demo it
        is configured for easy teardown (``RemovalPolicy.DESTROY`` +
        ``auto_delete_objects``). For production switch to ``RETAIN`` and enable
        versioning + access logging.
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
        """Upload the local ``../knowledge-base`` markdown files into the bucket."""
        s3_deployment.BucketDeployment(
            self,
            "DeployKnowledgeBaseDocs",
            sources=[s3_deployment.Source.asset(_KNOWLEDGE_BASE_DIR)],
            destination_bucket=bucket,
            destination_key_prefix="knowledge-base/",
            prune=True,
            retain_on_delete=False,
        )

    def _create_knowledge_base(
        self,
        bucket: s3.IBucket,
        inclusion_prefixes: "list[str] | None",
    ) -> bedrock.VectorKnowledgeBase:
        """Create the Bedrock Knowledge Base with a Titan-V2-embedded vector index.

        ``bedrock.VectorKnowledgeBase`` creates and manages the OpenSearch
        Serverless vector collection, the vector index, and the
        access/encryption/network policies. An :class:`bedrock.S3DataSource`
        points at the docs bucket with fixed-size chunking (500 tokens, 20%
        overlap) suitable for the short policy/FAQ documents.

        Args:
            bucket: The S3 bucket containing the documents.
            inclusion_prefixes: S3 key prefixes to index (``["knowledge-base/"]``
                for the sample bucket; ``None`` to index a whole existing bucket).
        """
        knowledge_base = bedrock.VectorKnowledgeBase(
            self,
            "AnyCompanyPolicyKnowledgeBase",
            name=f"{self.agent_prefix}-policy-knowledge-base",
            embeddings_model=_EMBEDDINGS_MODEL,
            instruction=self._brandize(
                "AnyCompany Energy policies: tariffs, payment plans, arrears and "
                "disconnection, vulnerable customers, complaints, refunds, direct "
                "debit, meter readings and appointments."
            ),
        )

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
            data_deletion_policy=bedrock.DataDeletionPolicy.DELETE,
        )

        return knowledge_base

    def _make_vector_collection_private(
        self, knowledge_base: bedrock.VectorKnowledgeBase
    ) -> None:
        """Remove public network access from the Knowledge Base's vector collection.

        generative-ai-cdk-constructs 0.1.318 creates the OpenSearch Serverless
        collection with a network policy that allows public access (to the
        collection and to OpenSearch Dashboards), and has no option to change it.
        This replaces the policy with private access only:

        * Bedrock reaches the collection through AWS service private access
          (``SourceServices: bedrock.amazonaws.com``).
        * The construct's index-creation Lambda reaches it through an OpenSearch
          Serverless VPC endpoint, in a VPC with no internet access.
        * No OpenSearch Dashboards access.
        """
        collection = knowledge_base.vector_store
        collection_name = collection.collection_name

        # A VPC with one subnet, no internet gateway and no NAT, used only by the
        # index-creation Lambda. Raw resources avoid a synth-time AZ lookup.
        vpc = ec2.CfnVPC(
            self,
            "VectorIndexVpc",
            cidr_block="10.0.0.0/24",
            enable_dns_support=True,
            enable_dns_hostnames=True,
        )
        subnet = ec2.CfnSubnet(
            self,
            "VectorIndexSubnet",
            vpc_id=vpc.ref,
            cidr_block="10.0.0.0/26",
            availability_zone=Fn.select(0, Fn.get_azs()),
        )
        # Inline egress rules replace a security group's default allow-all egress.
        # The endpoint's rule matches no traffic (the pattern CDK uses), so it has
        # no egress at all.
        endpoint_sg = ec2.CfnSecurityGroup(
            self,
            "VectorIndexEndpointSg",
            vpc_id=vpc.ref,
            group_description="OpenSearch Serverless VPC endpoint: HTTPS from the index-creation Lambda only.",
            security_group_egress=[
                ec2.CfnSecurityGroup.EgressProperty(
                    ip_protocol="icmp",
                    cidr_ip="255.255.255.255/32",
                    from_port=252,
                    to_port=86,
                    description="Disallow all traffic",
                )
            ],
        )
        function_sg = ec2.CfnSecurityGroup(
            self,
            "VectorIndexFunctionSg",
            vpc_id=vpc.ref,
            group_description="Index-creation Lambda: HTTPS to the OpenSearch Serverless VPC endpoint only.",
            security_group_egress=[
                ec2.CfnSecurityGroup.EgressProperty(
                    ip_protocol="tcp",
                    from_port=443,
                    to_port=443,
                    destination_security_group_id=endpoint_sg.attr_group_id,
                    description="HTTPS to the OpenSearch Serverless VPC endpoint",
                )
            ],
        )
        endpoint_ingress = ec2.CfnSecurityGroupIngress(
            self,
            "VectorIndexEndpointFromFunction",
            group_id=endpoint_sg.attr_group_id,
            ip_protocol="tcp",
            from_port=443,
            to_port=443,
            source_security_group_id=function_sg.attr_group_id,
        )
        endpoint = aoss.CfnVpcEndpoint(
            self,
            "VectorIndexEndpoint",
            name=_vpc_endpoint_name(self.agent_prefix),
            vpc_id=vpc.ref,
            subnet_ids=[subnet.ref],
            security_group_ids=[endpoint_sg.attr_group_id],
        )

        network_policy = _construct_child(collection, "NetworkPolicy", aoss.CfnSecurityPolicy)
        network_policy.policy = self.to_json_string(
            [
                {
                    "Description": "Private: Bedrock and the index-creation VPC endpoint",
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{collection_name}"],
                        }
                    ],
                    "AllowFromPublic": False,
                    "SourceServices": ["bedrock.amazonaws.com"],
                    "SourceVPCEs": [endpoint.attr_id],
                }
            ]
        )

        # Run the construct's index custom-resource Lambda (one per stack) in the VPC.
        provider = _construct_child(self, "OpenSearchIndexCRProvider", Construct)
        role = _construct_child(provider, "CRRole", iam.Role)
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaVPCAccessExecutionRole"
            )
        )
        function = _construct_child(
            _construct_child(provider, "CustomResourcesFunction", Construct),
            "Resource",
            lambda_.CfnFunction,
        )
        function.add_property_override(
            "VpcConfig",
            {"SubnetIds": [subnet.ref], "SecurityGroupIds": [function_sg.attr_group_id]},
        )
        # The index is created and deleted through the endpoint, so the function
        # must not run before the endpoint and its ingress rule exist, or after
        # either is deleted. Nothing else references the ingress rule, so without
        # this CloudFormation would delete it first on stack delete.
        function.node.add_dependency(endpoint, endpoint_ingress)

    # ------------------------------------------------------------------ #
    # Guardrail
    # ------------------------------------------------------------------ #
    def _create_guardrail(self) -> bedrock.Guardrail:
        """Create a Bedrock Guardrail applied by every Strands agent.

        The Strands ``BedrockModel`` wires this guardrail into every supervisor
        and specialist invocation (see ``agents/config.py``). It enforces:
          * Content filters (hate, insults, sexual, violence, misconduct,
            prompt-attack) at HIGH strength.
          * One denied topic: ``FINANCIAL_ADVICE`` (managed). Competitor and
            "other customers' data" policies are enforced via agent instructions
            instead — see the README "Guardrail design notes".
          * Sensitive-information filters that REDACT (ANONYMIZE) sort codes,
            phone numbers, emails and card details. UK postcodes and
            ``ACC-NNNNNN`` account numbers are intentionally NOT filtered (no
            detect-only mode exists for regex filters and the demo must surface
            them) — the commented-out filters can be enabled for production.
          * Contextual grounding + relevance filters at threshold 0.7. They're
            not evaluated in this sample: the agents' requests carry no tagged
            grounding source, guardrails on Converse don't evaluate tool
            results, and the retrieve tool screens passages as input, while
            grounding is checked only on output. See the README "Guardrail
            design notes".
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
        # FINANCIAL_ADVICE (managed) only. See the README "Guardrail design notes"
        # for why competitor / other-customer policies are enforced via agent
        # instructions rather than (probabilistic, ~200-char-limited) denied topics.
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
        # Bedrock sensitive-info regex filters support only ANONYMIZE or BLOCK
        # (no detect-only mode), and this demo legitimately echoes ACC-NNNNNN
        # account numbers and UK postcodes. To ENFORCE redaction in production,
        # uncomment the filters below — see the README "Guardrail PII tradeoff".
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

        # --- Contextual grounding --------------------------------------- #
        # Threshold 0.7, action NONE (measure, don't block). Grounding only runs
        # when a request tags a grounding source and a query, which the agents'
        # Converse calls don't. See README "Guardrail design notes".
        grounding_filters = (
            (bedrock.ContextualGroundingFilterType.GROUNDING, 0.7),
            (bedrock.ContextualGroundingFilterType.RELEVANCE, 0.7),
        )
        for ftype, threshold in grounding_filters:
            guardrail.add_contextual_grounding_filter(
                type=ftype, threshold=threshold, action=bedrock.GuardrailAction.NONE
            )
        # generative-ai-cdk-constructs 0.1.318 drops each filter's `action`, so set
        # it on the underlying CfnGuardrail.
        cfn_guardrail = _construct_child(guardrail, "MyGuardrail", bedrock_l1.CfnGuardrail)
        cfn_guardrail.add_property_override(
            "ContextualGroundingPolicyConfig.FiltersConfig",
            [
                {"Type": ftype.value, "Threshold": threshold, "Action": "NONE"}
                for ftype, threshold in grounding_filters
            ],
        )

        return guardrail

    # ------------------------------------------------------------------ #
    # Observability (CloudWatch dashboard + alarms)
    # ------------------------------------------------------------------ #
    def _bedrock_metric(
        self, metric_name: str, model_id: str, *, statistic: str, label: str
    ) -> cloudwatch.Metric:
        """Build an ``AWS/Bedrock`` metric scoped to a specific model id."""
        return cloudwatch.Metric(
            namespace="AWS/Bedrock",
            metric_name=metric_name,
            dimensions_map={"ModelId": model_id},
            statistic=statistic,
            period=Duration.minutes(5),
            label=label,
        )

    def _create_observability(self, *, guardrail: bedrock.Guardrail) -> None:
        """Create a CloudWatch dashboard and alarms over Bedrock + guardrail.

        Because the agents run on Bedrock (in process or on AgentCore Runtime)
        rather than as Bedrock-managed agents, the observable signals are the
        **Bedrock model metrics** (per inference profile) and the **guardrail**
        metrics this stack owns. The dashboard graphs invocations, latency and
        errors for the supervisor/specialist models plus guardrail interventions;
        alarms cover Bedrock server errors and an estimated-cost guard.

        Args:
            guardrail: The Bedrock guardrail (for intervention metrics).
        """
        models = {
            "Supervisor (Opus)": self._supervisor_profile_id,
            "Specialists (Sonnet)": self._specialist_profile_id,
        }

        invocation_metrics = [
            self._bedrock_metric("Invocations", model_id, statistic="Sum", label=label)
            for label, model_id in models.items()
        ]
        latency_metrics = [
            self._bedrock_metric(
                "InvocationLatency", model_id, statistic="Average", label=f"{label} avg ms"
            )
            for label, model_id in models.items()
        ]
        error_metrics = [
            self._bedrock_metric(
                "InvocationClientErrors", self._specialist_profile_id,
                statistic="Sum", label="Client errors (Sonnet)",
            ),
            self._bedrock_metric(
                "InvocationServerErrors", self._specialist_profile_id,
                statistic="Sum", label="Server errors (Sonnet)",
            ),
            self._bedrock_metric(
                "InvocationThrottles", self._specialist_profile_id,
                statistic="Sum", label="Throttles (Sonnet)",
            ),
        ]

        dashboard = cloudwatch.Dashboard(
            self,
            "ObservabilityDashboard",
            dashboard_name=f"{self.agent_prefix}-customer-service",
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Bedrock invocations per model (5m)", left=invocation_metrics, width=12
            ),
            cloudwatch.GraphWidget(
                title="Bedrock invocation latency (avg ms)", left=latency_metrics, width=12
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Bedrock errors & throttles (5m)", left=error_metrics, width=12
            ),
            cloudwatch.GraphWidget(
                title="Guardrail invocations vs interventions",
                left=[
                    guardrail.metric_invocations(
                        period=Duration.hours(1), label="Guardrail invocations"
                    ),
                    guardrail.metric_invocations_intervened(
                        period=Duration.hours(1), label="Interventions"
                    ),
                ],
                width=12,
            ),
        )
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown=(
                    "### Agent-level tracing & RAG observability\n"
                    "The supervisor and specialists run on the **Strands Agents "
                    "SDK**. Per-turn routing and tool timelines are emitted via the "
                    "Strands **callback/event system** (shown live in the Streamlit "
                    "demo sidebar) and as **OpenTelemetry** traces. When deployed to "
                    "**Amazon Bedrock AgentCore Runtime**, enable CloudWatch "
                    "Transaction Search + ADOT for end-to-end traces (see the README "
                    "'Production deployment'). RAG retrieval is not a default "
                    "CloudWatch metric. Bedrock model invocation logs hold full, "
                    "unmasked prompts and tool results, so protect the log group "
                    "first (README 'What the guardrail doesn't cover'). The "
                    "guardrail panel above counts model calls and the retrieve tool's "
                    "passage checks, so it's a rough proxy for blocked or masked "
                    "content."
                ),
                width=24,
                height=4,
            ),
        )

        # --- Alarms ------------------------------------------------------ #
        # Bedrock server errors on the specialist model over 5 minutes.
        self._bedrock_metric(
            "InvocationServerErrors", self._specialist_profile_id,
            statistic="Sum", label="Server errors",
        ).create_alarm(
            self,
            "BedrockServerErrorAlarm",
            threshold=5,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description=(
                f"{self.company_short_name} Bedrock specialist-model server errors "
                "exceeded 5 in 5 minutes."
            ),
        )

        # Estimated-cost guard (account-level AWS/Billing EstimatedCharges; only
        # published to us-east-1 and requires "Receive Billing Alerts" enabled).
        cloudwatch.Metric(
            namespace="AWS/Billing",
            metric_name="EstimatedCharges",
            dimensions_map={"Currency": "USD"},
            statistic="Maximum",
            period=Duration.hours(6),
        ).create_alarm(
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
        knowledge_base: bedrock.VectorKnowledgeBase,
        docs_bucket: s3.IBucket,
        guardrail: bedrock.Guardrail,
    ) -> None:
        """Emit the CloudFormation outputs consumed by the Strands agent runtime.

        The Strands demo reads these via ``demo/.env``: the Knowledge Base id
        (RAG), the guardrail id (applied on every model call) and the region.
        """
        CfnOutput(
            self,
            "KnowledgeBaseId",
            value=knowledge_base.knowledge_base_id,
            description=(
                "Id of the policy knowledge base (Titan V2 embeddings). Set as "
                "STRANDS_KNOWLEDGE_BASE_ID in the demo."
            ),
        )
        CfnOutput(
            self,
            "GuardrailId",
            value=guardrail.guardrail_id,
            description="Id of the Bedrock guardrail. Set as GUARDRAIL_ID in the demo.",
        )
        CfnOutput(
            self,
            "GuardrailArn",
            value=guardrail.guardrail_arn,
            description="ARN of the Bedrock guardrail applied by the agents.",
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
