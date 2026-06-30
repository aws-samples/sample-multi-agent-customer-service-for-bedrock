# AnyCompany Energy — Streamlit Demo

An interactive chat UI that invokes the Bedrock **supervisor** agent (the AnyCompany
Intent Router) and visualises, in real time, how the multi-agent team handles each
request: which specialist sub-agent the query was **routed to**, every tool
invocation with its parameters and result, and the supervisor's own reasoning.

This is the human-facing front door for the agentic customer service system. It talks
to AWS via `boto3`'s `bedrock-agent-runtime` client and streams the agent's response.

---

## What it shows

- **Chat interface** (`st.chat_message` / `st.chat_input`) with a stable conversation
  session so multi-turn context within the session works (follow-up questions
  retain prior turns). This uses a stable `sessionId` only — no Bedrock Agent
  long-term memory (`memoryId`) is configured.
- **Sidebar timeline** for the latest turn:
  - `Routed to: <agent>` — the collaborator(s) the supervisor delegated to.
  - Supervisor **rationale** (its reasoning text).
  - Every **tool call**: action group, function name, and parameters.
  - Every **tool result** and **sub-agent response** (rendered as JSON when applicable).
- **Cost / ROI framing** at the top: 12,000 interactions/month at GBP 8.50 each, with
  the 40-60% automation savings target.
- **Sample account numbers** you can use live in the demo.

---

## Prerequisites

1. **Deploy the CDK stack first.** The demo needs a *prepared* supervisor agent alias.
   From `infrastructure/`:

   ```bash
   npx aws-cdk deploy
   ```

2. **AWS credentials** available to `boto3` (any standard mechanism):
   - `aws configure` / a named profile (`export AWS_PROFILE=...`), or
   - environment variables `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`, or
   - an instance/role credential provider.

   The principal needs `bedrock:InvokeAgent` on the supervisor agent alias.

3. **Python 3.11+** and the demo dependencies:

   ```bash
   python3.11 -m venv .venv   # any Python 3.11+ works (3.12/3.13 are fine too)
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Configure the agent identifiers (from the CDK outputs)

After `cdk deploy`, the stack prints CloudFormation outputs. The two the demo needs are
`SupervisorAgentId` and `SupervisorAgentAliasId`, plus the `Region`. Capture them with:

```bash
aws cloudformation describe-stacks \
  --stack-name MultiAgentCustomerServiceStack \
  --query "Stacks[0].Outputs[?OutputKey=='SupervisorAgentId' || OutputKey=='SupervisorAgentAliasId' || OutputKey=='Region'].[OutputKey,OutputValue]" \
  --output table
```

Then export them as environment variables (these are the defaults the demo reads):

```bash
export SUPERVISOR_AGENT_ID="<SupervisorAgentId from outputs>"
export SUPERVISOR_AGENT_ALIAS_ID="<SupervisorAgentAliasId from outputs>"
export AWS_REGION="us-east-1"   # or the Region output value
```

> **Tip:** copy the template and fill in **your own** stack outputs, then load it:
>
> ```bash
> cp demo/.env.example demo/.env
> # edit demo/.env, replacing each REPLACE_WITH_... value with your stack outputs
> set -a && source demo/.env && set +a
> ```
>
> `demo/.env` is gitignored (only `demo/.env.example` is committed), so your
> populated values are never published.

> You can skip the environment variables entirely and just paste the IDs into the
> **sidebar** once the app is running — the sidebar fields override the environment.

| Variable | CDK output | Default |
|---|---|---|
| `SUPERVISOR_AGENT_ID` | `SupervisorAgentId` | *(empty — required)* |
| `SUPERVISOR_AGENT_ALIAS_ID` | `SupervisorAgentAliasId` | *(empty — required)* |
| `AWS_REGION` | `Region` | `us-east-1` |

---

## Run

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (default <http://localhost:8501>). Confirm the
three configuration fields in the sidebar are populated, then start chatting.

---

## Try these (sample accounts)

The shared mock datastore is seeded with three fixed accounts:

| Account | Customer | Tariff | State |
|---|---|---|---|
| `ACC-100123` | Sarah Thompson | Economy 7 | GBP 142.50 owed, ACTIVE |
| `ACC-100456` | James Okafor | Standard Variable | GBP 0.00, on a payment plan |
| `ACC-100789` | Priya Patel | EV Smart | GBP 318.20 arrears, ARREARS, vulnerable |

Example prompts (and the expected routing):

- **Billing:** "What's the balance on `ACC-100123`?" -> routed to **Billing Agent**
  (`lookup_account` / `check_balance`).
- **Scheduling:** "Book a meter reading for `ACC-100456` in LS6 2AB next week." ->
  routed to **Scheduling Agent** (`check_availability` then `book_appointment`).
- **Escalation:** "I have no water at `ACC-100789` and want to make a complaint." ->
  routed to **Escalation Agent** (`create_ticket`, then `transfer_to_agent`).
  Note: the Connect transfer queue is inferred from the *reason text*, not the
  account's vulnerable flag — a reason mentioning "no water"/"leak" routes to the
  **Emergency Supply** queue, whereas a reason led by "vulnerable"/"medical" routes
  to the **Priority Services** queue (as in demo-script Conversation 4).

Watch the **sidebar timeline** to see the routing decision, the tool calls with their
parameters, and the results flowing back.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Access denied invoking the agent` | The IAM principal lacks `bedrock:InvokeAgent` on the alias, or the wrong region. |
| `Agent or alias not found` | `SUPERVISOR_AGENT_ID` / `SUPERVISOR_AGENT_ALIAS_ID` don't match the CDK outputs, or wrong region. |
| `The request was throttled` | Transient Bedrock throttling — the client retries automatically; wait and retry. |
| No textual answer, but tools ran | Expand the timeline; the agent may have produced only tool activity. |
| "Missing configuration" | Fill the three sidebar fields (or set the env vars) before sending a message. |

Use **Reset conversation** in the sidebar to clear history and start a fresh Bedrock
session id.
