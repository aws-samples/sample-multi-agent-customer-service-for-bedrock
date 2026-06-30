"""CDK stack package for the multi-agent customer-service platform.

This package contains the single production stack,
:class:`stacks.customer_service_stack.MultiAgentCustomerServiceStack`, which
wires together the Amazon Bedrock agents, a guardrail, the knowledge base, S3
document store, the action-group tool Lambdas, and CloudWatch observability.
"""

from __future__ import annotations

from stacks.customer_service_stack import MultiAgentCustomerServiceStack

__all__ = ["MultiAgentCustomerServiceStack"]
