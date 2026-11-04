# Setup & Deployment Guide — Multi-Agent Customer Service Platform

This is the main setup guide for the **multi-agent customer-service platform**
sample (fictional scenario: **AnyCompany Energy**, a UK utility), built with the
**Strands Agents SDK** on **Amazon Bedrock**.

It covers prerequisites, deploying the supporting CDK stack (Knowledge Base +
Guardrail + CloudWatch), syncing the Knowledge Base, wiring the agent runtime
(demo) to the stack outputs, deploying to AgentCore Runtime, and troubleshooting.

For the architecture and business case see [architecture.md](./architecture.md).
For test scenarios see [testing-guide.md](./testing-guide.md).

---

## 1. Prerequisites

### 1.1 AWS account & Bedrock model access
- An AWS account where you can deploy CDK (CloudFormation, IAM, S3, Bedrock,
  OpenSearch Serverless, CloudWatch) and invoke Bedrock models/guardrails.
- **Amazon Bedrock model access enabled** in your target region for:
  - **Anthropic Claude Opus 4.6** — `anthropic.claude-opus-4-6-v1` (supervisor).
  - **Anthropic Claude Sonnet 4.6** — `anthropic.claude-sonnet-4-6` (specialists).
  - **Amazon Titan Text Embeddings V2** — `amazon.titan-embed-text-v2:0` (KB).

  Enable access in the Bedrock console under **Model access** (models can take a
  few minutes to activate).

  **Cross-region inference profiles:** the Claude 4.6 models do not support
  on-demand invocation by their bare id. The runtime invokes them through a
  cross-region inference profile (e.g. `us.anthropic.claude-opus-4-6-v1`); the geo
  prefix (`us.`/`eu.`/`apac.`) is auto-detected from your region by
  `agents/config.py`. Granting model access to the two Anthropic base models is
  sufficient. Confirm profiles with
  `aws bedrock list-inference-profiles --region <your-region>`.

- **Recommended region: `us-east-1`** (broadest model availability). US and EU
  regions are supported today; APAC is not yet supported for Claude 4.6 (see the
  top-level README "Supported regions").

### 1.2 Local tooling
- **Python 3.10+** (Strands requires 3.10+; 3.11/3.12 are fine).
- **Node.js 22 LTS** or later (the AWS CDK CLI runs via `npx aws-cdk` — no global install).
  ```bash
  npx aws-cdk --version   # expect 2.x
  ```
- **AWS credentials** configured (`aws configure`, SSO, or a role session):
  ```bash
  aws sts get-caller-identity
  ```

---

## 2. Deploy the supporting infrastructure

All commands assume you start from the repository root. The stack name is
configurable (`-c stackName=...`, default `MultiAgentCustomerServiceStack`):

```bash
STACK=MultiAgentCustomerServiceStack   # change if you override -c stackName
```

### 2.1 Create a virtual environment and install
```bash
cd infrastructure
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional dev tooling
```

### 2.2 Bootstrap the account/region (first time only)
```bash
export AWS_REGION=us-east-1
npx aws-cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/us-east-1
```

### 2.3 Synthesize and deploy
```bash
npx aws-cdk synth
npx aws-cdk deploy
```
The stack provisions:
- the S3 docs bucket and deploys the four `knowledge-base/*.md` files,
- the Bedrock Knowledge Base + OpenSearch Serverless vector index (Titan V2),
- a Bedrock **Guardrail** (applied by the agents on every model call),
- a CloudWatch dashboard + alarms.

The OpenSearch Serverless collection can take **10–15 minutes** on first deploy.

> **Cost note — OpenSearch Serverless.** The Knowledge Base vector store bills a
> **continuous minimum-OCU charge** while the collection exists (the dominant
> standing cost), even when idle. The collection's VPC endpoint adds a smaller
> hourly AWS PrivateLink charge, and a Route 53 private hosted zone for its DNS. Run `npx aws-cdk destroy` (section 8) when you
> are not actively using the sample.

---

## 3. Sync the Knowledge Base

Deploying uploads the documents to S3 and creates the empty vector index, **but
it does not embed the documents automatically**. Run a one-time **Sync** after
deploy (and again whenever the `knowledge-base/*.md` files change):

1. Open the **Amazon Bedrock console** → **Knowledge Bases** and select the
   knowledge base named `<agentPrefix>-policy-knowledge-base` (its id is the
   `KnowledgeBaseId` stack output).
2. Under **Data source**, select the data source and choose **Sync**.
3. Wait until the sync status is **Completed**.

Only after the sync completes can the billing/escalation specialists answer
tariff/policy questions via RAG.

For headless / CI use, start the ingestion job from the CLI instead (the
data-source id is not a stack output, so look it up first):

```bash
KB=$(aws cloudformation describe-stacks --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='KnowledgeBaseId'].OutputValue" --output text)
DS=$(aws bedrock-agent list-data-sources --knowledge-base-id "$KB" \
  --query "dataSourceSummaries[0].dataSourceId" --output text)
aws bedrock-agent start-ingestion-job --knowledge-base-id "$KB" --data-source-id "$DS"
# poll until COMPLETE:
aws bedrock-agent list-ingestion-jobs --knowledge-base-id "$KB" --data-source-id "$DS" \
  --query "ingestionJobSummaries[0].status"
```

---

## 4. Collect the stack outputs

The stack emits these CloudFormation outputs (exact names): `KnowledgeBaseId`,
`GuardrailId`, `GuardrailArn`, `DocsBucketName`, `Region`.

Print them all:
```bash
aws cloudformation describe-stacks --stack-name "$STACK" \
  --query "Stacks[0].Outputs" --output table
```

---

## 5. Run the agents (Streamlit demo)

```bash
cd demo
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # edit .env with your stack outputs
set -a && source .env && set +a
streamlit run app.py
```

Set these in `demo/.env` (see `demo/.env.example`):

| Variable | From stack output | Notes |
|---|---|---|
| `AWS_REGION` | `Region` | Inference-profile geo is auto-detected from it |
| `STRANDS_KNOWLEDGE_BASE_ID` | `KnowledgeBaseId` | Enables billing/escalation RAG |
| `GUARDRAIL_ID` | `GuardrailId` | Applied on every model call |
| `GUARDRAIL_VERSION` | _(default `DRAFT`)_ | Guardrail version |

The Streamlit app imports the supervisor from the `agents/` package and runs it
**in process**; you can also override the values from the sidebar. The sidebar
shows a live per-turn timeline (classification → routing → specialist tool calls).

Headless (no UI) — install the package once so it runs from any directory:
```bash
pip install -e .          # from the repo root; installs the `agents` package
python -m agents.supervisor "What's the balance on ACC-100123?"
```
(Without `pip install -e .`, run it **from the repo root** with
`agents/requirements.txt` installed.)

Use the sample accounts (see [testing-guide.md](./testing-guide.md)):
- **ACC-100123** — Sarah Thompson, Economy 7, balance owed.
- **ACC-100456** — James Okafor, Standard Variable, on a payment plan.
- **ACC-100789** — Priya Patel, EV Smart, in arrears, vulnerable.

---

## 6. Deploy to AgentCore Runtime (production)

For production scale, deploy the same agents to **Amazon Bedrock AgentCore
Runtime**. Wrap the supervisor in a thin entrypoint and deploy with the AgentCore
CLI — see the top-level README "Production deployment" and the
[Strands AgentCore deployment guide](https://strandsagents.com/docs/user-guide/deploy/deploy_to_bedrock_agentcore/python/).
The Knowledge Base, Guardrail and CloudWatch resources from this stack are reused
as-is.

---

## 7. Troubleshooting

**Model not available / access denied on invoke**
- Confirm **Model access** for Claude Opus 4.6, Claude Sonnet 4.6 and Titan V2 is
  enabled in the **same region** you set in `AWS_REGION`. Newly enabled models can
  take a few minutes to activate.

**`validationException ... inference profile` errors**
- The model id must be a cross-region inference profile for your region's geo
  (e.g. `us.anthropic.claude-opus-4-6-v1`). `agents/config.py` builds this
  automatically from `AWS_REGION`; if you set a custom `SUPERVISOR_MODEL_ID` /
  `SPECIALIST_MODEL_ID`, pass the base id (e.g. `anthropic.claude-sonnet-4-6`) and
  let the geo prefix be added, or set `INFERENCE_PROFILE_GEO` explicitly.
- APAC regions are not yet supported for Claude 4.6 — use US or EU.

**RAG returns nothing / "I don't have that information"**
- The Knowledge Base **Sync** has not been run or has not finished (section 3).
- `STRANDS_KNOWLEDGE_BASE_ID` is unset or wrong. If unset, the billing/escalation
  specialists warn at startup that RAG is disabled; if wrong, the `retrieve` tool
  returns an error such as "Knowledge Base search failed
  (ResourceNotFoundException)". Confirm the docs landed in the bucket:
  ```bash
  aws s3 ls "s3://$(aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='DocsBucketName'].OutputValue" --output text)/"
  ```

**Guardrail errors**
- Ensure `GUARDRAIL_ID` matches the `GuardrailId` output and your principal has
  `bedrock:ApplyGuardrail`. Leaving `GUARDRAIL_ID` unset runs the agents without
  the guardrail (the demo sidebar flags this).

**Throttling**
- Bedrock applied rate limiting. Retry after a short pause; request a model quota
  increase for sustained load.

**Stack stuck deleting the S3 bucket**
- The bucket uses `auto_delete_objects = True` + `removal_policy = DESTROY`. If a
  delete hangs, ensure the auto-delete custom resource ran; otherwise empty the
  bucket and retry `npx aws-cdk destroy`.

---

## 8. Teardown
```bash
cd infrastructure
source .venv/bin/activate
npx aws-cdk destroy
```
The S3 bucket auto-deletes its objects, so the stack tears down without manual
steps. Expect it to take about 25 minutes: AWS releases the index-creation
Lambda's VPC network interfaces slowly (about 20 minutes in testing), and the
OpenSearch Serverless VPC endpoint takes a few more. Because
OpenSearch Serverless bills a continuous minimum-OCU charge while it exists,
destroying the stack when you are done is the main way to avoid standing cost.
