#!/usr/bin/env python3
"""CDK application entry point for the multi-agent customer-service platform.

This module instantiates a single :class:`MultiAgentCustomerServiceStack`, which
provisions the full Amazon Bedrock multi-agent collaboration topology (a
``SUPERVISOR_ROUTER`` intent-router agent fronting three specialist agents),
an Amazon Bedrock Knowledge Base backed by Amazon OpenSearch Serverless, a
Bedrock Guardrail applied to every agent, the action-group tool Lambdas
(with X-Ray tracing), a CloudWatch observability dashboard + alarms, and a
shared Lambda layer.

Branding, agent-name prefix, foundation-model ids, and the stack name are all
configurable via ``cdk.json`` context (or ``-c key=value`` on the CLI), so the
sample can be rebranded for any company/industry. Defaults reproduce the
original "AnyCompany Energy" demo exactly. See ``cdk.json`` for the full list.

The stack is bound to an explicit :class:`~aws_cdk.Environment` derived from the
standard CDK bootstrap environment variables (``CDK_DEFAULT_ACCOUNT`` and
``CDK_DEFAULT_REGION``). Binding an environment is required because the
generative-ai-cdk-constructs Knowledge Base provisions region-specific
resources (OpenSearch Serverless collections, Bedrock model ARNs) that cannot be
created in an environment-agnostic stack.
"""

from __future__ import annotations

import logging
import os

import aws_cdk as cdk

from stacks.customer_service_stack import MultiAgentCustomerServiceStack

# Module-level logger. CDK synthesis is a normal Python process, so standard
# logging is the simplest way to surface synthesis-time diagnostics.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default CloudFormation stack name. Override via ``-c stackName=...`` or the
# ``stackName`` key in cdk.json to deploy under a different name.
_DEFAULT_STACK_NAME = "MultiAgentCustomerServiceStack"


def _resolve_environment(region_context: str | None = None) -> cdk.Environment:
    """Resolve the AWS environment (account + region) for the stack.

    Region precedence (highest first):
      1. The ``region`` CDK context key (``-c region=...`` or cdk.json), so a
         published sample respects user-provided configuration.
      2. ``CDK_DEFAULT_REGION`` (populated by the CDK CLI from the active
         credentials/profile).
      3. ``us-east-1`` fallback (the demo default).

    Args:
        region_context: Value of the ``region`` context key, if provided.

    Returns:
        A :class:`aws_cdk.Environment`. Note: this sample uses **US** cross-region
        inference profiles, so the region must be a US region unless
        ``inferenceProfileGeo`` is also overridden (see README "Supported regions").
    """
    account = os.environ.get("CDK_DEFAULT_ACCOUNT")
    region = region_context or os.environ.get("CDK_DEFAULT_REGION") or "us-east-1"

    if not account:
        # We do not hard-fail here: ``cdk synth`` can run without credentials for
        # template inspection. The CLI will require a real account at deploy time.
        logger.warning(
            "CDK_DEFAULT_ACCOUNT is not set; synthesising an account-agnostic "
            "template. Deployment will require valid AWS credentials."
        )

    return cdk.Environment(account=account, region=region)


def main() -> None:
    """Synthesise the CDK application."""
    app = cdk.App()

    # Stack name, branding, and region are context-configurable (see cdk.json).
    stack_name = app.node.try_get_context("stackName") or _DEFAULT_STACK_NAME
    company_name = app.node.try_get_context("companyName") or "AnyCompany Energy"
    region_context = app.node.try_get_context("region")

    MultiAgentCustomerServiceStack(
        app,
        stack_name,
        env=_resolve_environment(region_context),
        description=(
            f"{company_name} multi-agent customer-service platform: Amazon Bedrock "
            "multi-agent collaboration (intent router + billing/scheduling/"
            "escalation specialists), a Titan-V2-embedded Knowledge Base, a Bedrock "
            "guardrail, and the supporting Lambda action groups."
        ),
    )

    # Apply project-wide tags so every resource is attributable in billing and
    # discoverable in the console. Tag values are context-configurable; defaults
    # match the original deployment so existing stacks are not re-tagged (note:
    # changing tags on an OpenSearch Serverless collection forces replacement).
    cdk.Tags.of(app).add(
        "project", app.node.try_get_context("projectTag") or "multi-agent-customer-service"
    )
    cdk.Tags.of(app).add(
        "owner", app.node.try_get_context("ownerTag") or "customer-service-platform"
    )
    cdk.Tags.of(app).add("environment", "demo")

    app.synth()


if __name__ == "__main__":
    main()
