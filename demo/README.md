# AnyCompany Energy — Streamlit Demo

An interactive chat UI that runs the **Strands Agents** multi-agent team **in
process** and visualises, in real time, how the team handles each request: the
supervisor's intent classification, which specialist it **routed** to, and every
tool the specialist ran — with parameters and results.

The app imports the supervisor from the sibling `agents/` package and calls it
directly (no remote agent runtime to invoke). A shared Strands **callback
handler** captures the full nested timeline of each turn.

---

## What it shows

- **Chat interface** (`st.chat_message` / `st.chat_input`) with per-session
  conversation history (the supervisor `Agent` retains context across turns
  within a browser session).
- **Sidebar timeline** for the latest turn, built from the Strands callback/event
  stream:
  - `Routed to: <specialist>` — the specialist-as-tool the supervisor called.
  - Every **tool call** (e.g. `classify_intent`, `check_balance`, `create_ticket`)
    with its parameters, and the tool **result** (rendered as JSON when applicable).
- **Cost / ROI framing** at the top: 12,000 interactions/month at GBP 8.50 each,
  with the 40-60% automation savings target.
- **Sample account numbers** you can use live in the demo.

---

## Prerequisites

1. **Deploy the CDK stack first** (Knowledge Base + Guardrail + CloudWatch) and
   **sync the Knowledge Base** — see [../docs/README.md](../docs/README.md).

2. **AWS credentials** available to `boto3` (any standard mechanism: `aws
   configure` / a named profile / environment variables / an instance role). The
   principal needs to invoke the Claude models, apply the guardrail
   (`bedrock:ApplyGuardrail`) and query the Knowledge Base (`bedrock:Retrieve`).

3. **Python 3.10+** and the demo dependencies:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Configure (from the CDK outputs)

The demo reads its configuration from environment variables. Copy the template
and fill in **your own** stack outputs:

```bash
cp .env.example .env
# edit .env:
#   AWS_REGION                 = the Region output
#   STRANDS_KNOWLEDGE_BASE_ID  = the KnowledgeBaseId output
#   GUARDRAIL_ID               = the GuardrailId output
set -a && source .env && set +a
```

`demo/.env` is gitignored (only `demo/.env.example` is committed), so your values
are never published.

| Variable | CDK output | Default |
|---|---|---|
| `AWS_REGION` | `Region` | `us-east-1` |
| `STRANDS_KNOWLEDGE_BASE_ID` | `KnowledgeBaseId` | _(empty — RAG disabled if unset)_ |
| `GUARDRAIL_ID` | `GuardrailId` | _(empty — guardrail off if unset)_ |
| `GUARDRAIL_VERSION` | — | `DRAFT` |

You can also skip the env vars and paste the Region / Knowledge Base ID /
Guardrail ID into the **sidebar** once the app is running — the sidebar overrides
the environment.

---

## Run

```bash
streamlit run app.py
```

Streamlit opens the app (default <http://localhost:8501>). Confirm the sidebar
configuration is populated, then start chatting.

---

## Try these (sample accounts)

The shared mock datastore is seeded with three fixed accounts:

| Account | Customer | Tariff | State |
|---|---|---|---|
| `ACC-100123` | Sarah Thompson | Economy 7 | GBP 142.50 owed, ACTIVE |
| `ACC-100456` | James Okafor | Standard Variable | GBP 0.00, on a payment plan |
| `ACC-100789` | Priya Patel | EV Smart | GBP 318.20 arrears, ARREARS, vulnerable |

Example prompts (and the expected routing):

- **Billing:** "What's the balance on `ACC-100123`?" -> **Billing Agent**
  (`check_balance`).
- **Scheduling:** "Book a meter reading for `ACC-100456` in LS6 2AB next week." ->
  **Scheduling Agent** (`check_availability` then `book_appointment`).
- **Escalation:** "I have no water at `ACC-100789` and want to make a complaint." ->
  **Escalation Agent** (`create_ticket`, then `transfer_to_agent`). The Connect
  transfer queue is inferred from the *reason text*: "no water"/"leak" routes to
  the **Emergency Supply** queue, while "vulnerable"/"medical" routes to the
  **Priority Services** queue.

Watch the **sidebar timeline** to see the routing decision, the tool calls with
their parameters, and the results flowing back.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Access-denied / credentials error | Your principal can't invoke the models, guardrail or Knowledge Base; or the region is wrong. |
| `validationException ... inference profile` | The region's geo has no profile for Claude 4.6 (APAC) — use US/EU; or set `INFERENCE_PROFILE_GEO`. |
| "RAG disabled" warning / no policy answers | `STRANDS_KNOWLEDGE_BASE_ID` unset, or the Knowledge Base **Sync** has not completed. |
| Throttling | Transient Bedrock rate limiting — wait and retry. |
| No textual answer, but tools ran | Expand the timeline; the agents may have produced only tool activity. |

Use **Reset conversation** in the sidebar to clear history and rebuild the agents
with the current configuration.
