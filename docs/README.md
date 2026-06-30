# Setup & Deployment Guide — Multi-Agent Customer Service Platform

This is the main setup guide for the **multi-agent customer-service platform**
sample (fictional scenario: **AnyCompany Energy**, a UK utility).

It covers prerequisites, deploying the CDK stack, starting the Knowledge Base
ingestion job, wiring the demo to the stack outputs, and troubleshooting.

For the architecture and business case see
[architecture.md](./architecture.md). For the demo walkthrough see
[demo-script.md](./demo-script.md).

---

## 1. Prerequisites

### 1.1 AWS account & Bedrock model access
- An AWS account where you have permissions to deploy CDK (CloudFormation,
  IAM, Lambda, S3, Bedrock, OpenSearch Serverless, CloudWatch).
- **Amazon Bedrock model access enabled** in your target region for **all**
  models used by this project (a tiered model strategy — Opus for routing
  intelligence, Sonnet for fast specialist tool execution):
  - **Anthropic Claude Opus 4.6** — `anthropic.claude-opus-4-6-v1`
    (used by the supervisor / intent-router agent).
  - **Anthropic Claude Sonnet 4.6** — `anthropic.claude-sonnet-4-6`
    (used by the three specialist agents: billing, scheduling, escalation).
  - **Amazon Titan Text Embeddings V2** — `amazon.titan-embed-text-v2:0`
    (used by the Knowledge Base).

  Enable access in the Bedrock console under **Model access**. Models can take a
  few minutes to become available after enabling.

  **Important — cross-region inference profiles:** the Claude 4.6 models do **not**
  support on-demand invocation by their bare model id. The stack therefore invokes
  them through the **US cross-region inference profiles**
  `us.anthropic.claude-opus-4-6-v1` and `us.anthropic.claude-sonnet-4-6` (built in
  code via `CrossRegionInferenceProfile`). Granting **Model access** to the two
  Anthropic base models above is sufficient — the inference profile fans out to the
  base model across the US regions, and the CDK construct provisions the matching
  IAM permissions automatically. You can confirm the profiles are `ACTIVE` with
  `aws bedrock list-inference-profiles --region us-east-1`.

- **Recommended region: `us-east-1`** — it has the broadest Bedrock Agents,
  multi-agent collaboration, and model availability, and is the demo default.

### 1.2 Local tooling
- **Python 3.11+** for the CDK app and the demo. (The deployed Lambda **runtime**
  is Python 3.12, set in the stack; your local Python only needs to be 3.11+.)
- **Node.js 18+**. The AWS CDK v2 CLI is invoked via `npx aws-cdk` — no global
  install is required:
  ```bash
  npx aws-cdk --version   # expect 2.x
  ```
- **AWS credentials** configured (e.g. `aws configure`, SSO, or a role session).
  Verify with:
  ```bash
  aws sts get-caller-identity
  ```
- A **CDK-bootstrapped** account/region (see step 2.3).

---

## 2. Deploy the infrastructure

All commands below assume you start from the repository root:
`capstone-agentic-customer-service/`.

The stack name is configurable via CDK context (`-c stackName=...`, default
`MultiAgentCustomerServiceStack`). The commands below set a `STACK` shell
variable once so nothing hard-codes the name:

```bash
STACK=MultiAgentCustomerServiceStack   # change if you override -c stackName
```

### 2.1 Create and activate a virtual environment
```bash
cd infrastructure
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

### 2.2 Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional: dev/test tooling
```
`infrastructure/requirements.txt` pins `aws-cdk-lib`, `constructs`, and
`cdklabs.generative-ai-cdk-constructs`.

### 2.3 Bootstrap the account/region (first time only)
```bash
export AWS_REGION=us-east-1
npx aws-cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/us-east-1
```

### 2.4 Synthesize and deploy
```bash
npx aws-cdk synth
npx aws-cdk deploy            # deploys the single stack; name set via -c stackName
```
The stack provisions:
- the S3 docs bucket and deploys the four `knowledge-base/*.md` files,
- the Bedrock Knowledge Base + OpenSearch Serverless vector index (Titan V2),
  attached to **both** the billing and escalation agents for RAG,
- a Bedrock **Guardrail** applied to all four agents,
- the four action-group Lambdas (Python 3.12 runtime, X-Ray tracing) and the
  shared layer,
- the four Bedrock agents (supervisor + 3 collaborators) and their aliases. The
  collaborator/supervisor aliases use the construct's default hash-based names, so
  they **automatically roll a new agent version** whenever an agent's definition
  changes (model, instruction, KB attachment); the alias **ids** stay stable across
  in-place updates,
- a CloudWatch dashboard + alarms.

Deployment of the OpenSearch Serverless collection and Bedrock agents can take
**10–20 minutes**.

> **Cost note — OpenSearch Serverless.** The Knowledge Base vector store is
> Amazon OpenSearch Serverless, which bills a **continuous minimum-OCU charge**
> while the collection exists (the dominant standing cost of this stack), even
> when idle. Run `npx aws-cdk destroy` (section 7) when you are not actively
> using the sample.

---

## 3. Start the Knowledge Base ingestion job

Deploying the stack uploads the documents to S3 and creates the empty vector
index, **but it does not embed the documents automatically**. You must run an
**ingestion job** once after deploy (and again whenever the
`knowledge-base/*.md` files change).

1. Grab the Knowledge Base ID from the stack outputs:
   ```bash
   KB_ID=$(aws cloudformation describe-stacks \
     --stack-name "$STACK" \
     --query "Stacks[0].Outputs[?OutputKey=='KnowledgeBaseId'].OutputValue" \
     --output text)
   echo "Knowledge Base: $KB_ID"
   ```
2. Find the data source ID for that Knowledge Base:
   ```bash
   DS_ID=$(aws bedrock-agent list-data-sources \
     --knowledge-base-id "$KB_ID" \
     --query "dataSourceSummaries[0].dataSourceId" \
     --output text)
   echo "Data source: $DS_ID"
   ```
3. Start the ingestion job:
   ```bash
   aws bedrock-agent start-ingestion-job \
     --knowledge-base-id "$KB_ID" \
     --data-source-id "$DS_ID"
   ```
4. Wait for it to complete (`COMPLETE`):
   ```bash
   aws bedrock-agent list-ingestion-jobs \
     --knowledge-base-id "$KB_ID" \
     --data-source-id "$DS_ID" \
     --query "ingestionJobSummaries[0].status"
   ```
Only after the job reports `COMPLETE` can the billing/escalation agents answer
tariff/policy questions via RAG.

---

## 4. Collect the stack outputs

The stack emits these CloudFormation outputs (exact names):
`SupervisorAgentId`, `SupervisorAgentAliasId`, `BillingAgentId`,
`SchedulingAgentId`, `EscalationAgentId`, `KnowledgeBaseId`, `DocsBucketName`,
`Region`, `GuardrailId`, `GuardrailArn`.

Print them all:
```bash
aws cloudformation describe-stacks \
  --stack-name "$STACK" \
  --query "Stacks[0].Outputs" --output table
```

Export the two the demo needs:
```bash
export SUPERVISOR_AGENT_ID=$(aws cloudformation describe-stacks \
  --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='SupervisorAgentId'].OutputValue" \
  --output text)

export SUPERVISOR_AGENT_ALIAS_ID=$(aws cloudformation describe-stacks \
  --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='SupervisorAgentAliasId'].OutputValue" \
  --output text)

export AWS_REGION=us-east-1
```

---

## 5. Run the demo

```bash
cd demo
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The Streamlit app reads `SUPERVISOR_AGENT_ID`, `SUPERVISOR_AGENT_ALIAS_ID`, and
`AWS_REGION` from the environment, with **sidebar overrides** if you prefer to
paste them in. It maintains a stable `sessionId`, calls
`invoke_agent(..., enableTrace=True)`, and renders a live routing/tool timeline.
Copy `demo/.env.example` to `demo/.env` and fill in your own stack outputs (see
`demo/README.md`).

Use the sample accounts in your queries (see
[demo-script.md](./demo-script.md)):
- **ACC-100123** — Sarah Thompson, Economy 7, balance owed.
- **ACC-100456** — James Okafor, Standard Variable, on a payment plan.
- **ACC-100789** — Priya Patel, EV Smart, in arrears, vulnerable.

---

## 6. Troubleshooting

**Model not available / `AccessDeniedException` on invoke**
- Confirm **Model access** for *all* of Claude Opus 4.6, Claude Sonnet 4.6 and
  Titan V2 is enabled in the **same region** you deployed to. Newly enabled
  models can take a few minutes to activate.

**Region / availability mismatch**
- Bedrock Agents, multi-agent collaboration, and specific models are not in
  every region. If deployment or invocation fails with availability errors,
  redeploy in **`us-east-1`** and set `AWS_REGION=us-east-1` for the demo.

**Inference profile note**
- The newer Claude 4.x models (`anthropic.claude-opus-4-6-v1`,
  `anthropic.claude-sonnet-4-6`) **require a cross-region inference profile** — they
  cannot be invoked on-demand by their bare model id. The stack already handles this:
  in `infrastructure/stacks/customer_service_stack.py` each base model is wrapped in a
  `CrossRegionInferenceProfile.from_config(geo_region=US, model=BedrockFoundationModel(..., supports_cross_region=True))`,
  which the agents use as their `foundation_model`.
- **Do not** just change the `foundation_model` string to a bare `us.anthropic.*`
  value: that makes invocation work but generates an incorrect
  `foundation-model/us.anthropic...` IAM ARN, so the agent then fails with
  `accessDeniedException`. The `CrossRegionInferenceProfile` construct is required
  because it also emits the correct IAM (the `inference-profile/*` ARN plus the
  underlying regional `foundation-model/*` ARNs).
- Two failure signatures and what they mean:
  - `validationException: ... model ID ... isn't supported. Retry ... with an inference profile`
    → the agent is pointing at a bare foundation-model id; wrap it in the profile.
  - `accessDeniedException` after switching to a profile → the IAM grant is on the
    wrong ARN; use the `CrossRegionInferenceProfile` construct rather than a raw string.
- Confirm the profiles are available with
  `aws bedrock list-inference-profiles --region us-east-1` and ensure **Model access**
  is enabled for the two Anthropic base models in that region.

**RAG returns nothing / "I don't have that information"**
- The **ingestion job was not run** or has not finished. Re-run step 3 and wait
  for status `COMPLETE`. Confirm the four `knowledge-base/*.md` files landed in
  `DocsBucketName`:
  ```bash
  aws s3 ls "s3://$(aws cloudformation describe-stacks \
    --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='DocsBucketName'].OutputValue" \
    --output text)/"
  ```

**`ResourceNotFoundException` in the demo**
- The `SUPERVISOR_AGENT_ID` / `SUPERVISOR_AGENT_ALIAS_ID` are wrong or from a
  different region. Re-export from the stack outputs (step 4) and confirm the
  sidebar region matches the deploy region.

**Throttling / `ThrottlingException`**
- Bedrock applied rate limiting. Retry after a short pause; the demo surfaces a
  friendly error. Request a model quota increase for sustained load.

**IAM / `AccessDenied` invoking a tool Lambda**
- Verify Bedrock has permission to invoke the action-group Lambdas. The
  `ActionGroupExecutor.from_lambda_function` construct normally wires the
  resource policy; if not, the stack adds an explicit `lambda.add_permission`
  for `bedrock.amazonaws.com`. Check the Lambda's resource-based policy and the
  agent service role's `bedrock:InvokeModel` / `bedrock:Retrieve` grants.

**Stack stuck deleting the S3 bucket**
- The bucket uses `auto_delete_objects = True` and `removal_policy = DESTROY`
  for easy teardown. If a delete hangs, ensure the auto-delete custom resource
  Lambda ran; otherwise empty the bucket manually and retry `npx aws-cdk destroy`.

---

## 7. Teardown
```bash
cd infrastructure
source .venv/bin/activate
npx aws-cdk destroy
```
The S3 bucket auto-deletes its objects, so the stack tears down cleanly. Because
OpenSearch Serverless bills a continuous minimum-OCU charge while it exists,
destroying the stack when you are done is the main way to avoid standing cost.
