"""CDK stack package for the multi-agent customer-service platform.

This package contains the single stack,
:class:`stacks.customer_service_stack.MultiAgentCustomerServiceStack`, which
provisions the supporting infrastructure (a Titan-V2 Bedrock Knowledge Base on
OpenSearch Serverless, an S3 document store, a Bedrock Guardrail, and CloudWatch
observability) for the Strands Agents application in the ``agents/`` package.
"""

from __future__ import annotations

from stacks.customer_service_stack import MultiAgentCustomerServiceStack

__all__ = ["MultiAgentCustomerServiceStack"]
