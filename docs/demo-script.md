# Demo Script — AnyCompany Energy Agentic Customer Service

A walkthrough of **five sample conversations** for developers exploring the
sample. Each one
lists the customer line(s), the **expected routing** (which collaborator
handles it), the **tool calls** that fire and their parameters, and the
**expected response**.

These conversations depend on the fixed sample accounts and the exact function
names defined in the contract. The **stable** values — account numbers, names,
balances, tariffs, statuses, SLA hours, queues/teams — reproduce exactly. The
**generated identifiers** (slot ids, booking references, ticket ids, transfer
ids, Connect contact ids) are produced by a deterministic hash that also folds
in a per-container request counter, so their *shape* is fixed but the exact
digits/characters will differ run-to-run. They are shown below as illustrative
examples (e.g. `APPT-######`); read them as "a value of this form", not a literal
you should expect to see.

### Sample accounts (usable in the demo)

| Account | Name | Tariff | Status | Balance (GBP) | Postcode | Notes |
|---|---|---|---|---|---|---|
| **ACC-100123** | Sarah Thompson | Economy 7 | ACTIVE | 142.50 owed, due 2026-07-01 | M14 5TG | last payment 120.00 on 2026-05-15 |
| **ACC-100456** | James Okafor | Standard Variable | ACTIVE | 0.00, due 2026-07-03 | LS6 2AB | on payment plan; last payment 89.40 on 2026-06-01 |
| **ACC-100789** | Priya Patel | EV Smart | ARREARS | 318.20 arrears, due 2026-06-10 | BS3 4XY | vulnerable; last payment 45.00 on 2026-04-20 |

### How routing works
The supervisor (`anycompany-intent-router`) calls
`classify_intent(query)` which scores the query against weighted keyword
vocabularies (see `lambdas/intent_router/handler.py`):
- **BILLING** — bill, billing, balance, payment, pay, tariff, refund,
  `direct debit`, charge, invoice, statement, credit, arrears, owe/owed,
  `account balance`, `economy 7`, `standard variable`, `ev smart`.
- **SCHEDULING** — appointment, book, booking, schedule, reschedule, cancel,
  `meter reading`, meter, engineer, visit, slot, availability, install,
  `ev charger`, `leak investigation`.
- **ESCALATION** — complaint, complain, escalate, manager, legal, ombudsman,
  solicitor, regulator, vulnerable, emergency, leak, `no water`, burst, flood,
  medical, elderly.

The winning category's **share of the total score** becomes the confidence;
**below 0.34 → UNKNOWN** and the supervisor gives a graceful clarification
fallback. (Note: the BILLING vocabulary matches the phrase `account balance`,
not the bare word *account* — so "balance" is what fires on a query like
"balance on ACC-100123".) You can watch the classification, the
*Routed to: `<agent>`* label, every tool call with its parameters and result,
and the supervisor rationale in the Streamlit sidebar timeline.

---

## Conversation 1 — Billing / tariff RAG question

**Customer:**
> "Can you explain how the Economy 7 tariff works and when the cheap hours are?"

**Expected routing:**
- `classify_intent` → **BILLING** (keywords: *tariff*). High confidence.
- *Routed to:* **anycompany-billing-agent**.

**Tools / retrieval that fire:**
- No structured tool call is required. The Billing agent performs a **Knowledge
  Base retrieval (RAG)** against `tariff-guide.md` for the *Economy 7* tariff.

**Expected response:**
A grounded explanation of the **Economy 7** tariff — a dual-rate (day/night)
tariff with cheaper electricity during a 7-hour off-peak overnight window —
sourced from `tariff-guide.md`. The answer should match AnyCompany's published
tariff documentation and may cite the source.

**What this demonstrates:** RAG grounding — the answer comes from the
ingested policy documents, not the model's parametric memory. (Requires the
ingestion job from `docs/README.md` step 3 to have completed.)

---

## Conversation 2 — Balance lookup

**Customer:**
> "What is the outstanding balance on my account ACC-100123, and when is it due?"

**Expected routing:**
- `classify_intent` → **BILLING** (keyword: *balance*). High confidence.
- *Routed to:* **anycompany-billing-agent**.

**Tools that fire:**
- `check_balance(account_number="ACC-100123")`

**Expected tool result (JSON in the TEXT body):**
```json
{
  "account_number": "ACC-100123",
  "balance_gbp": 142.50,
  "due_date": "2026-07-01",
  "last_payment": { "amount_gbp": 120.00, "date": "2026-05-15" },
  "status": "ACTIVE"
}
```

**Expected response:**
"Sarah, your account **ACC-100123** has an outstanding balance of **GBP 142.50**,
due on **1 July 2026**. Your last payment of GBP 120.00 was received on
15 May 2026. The account is **ACTIVE**." (The agent may optionally call
`lookup_account` first to greet by name.)

---

## Conversation 3 — Booking a meter-reading / engineer appointment

**Customer (turn 1):**
> "I need to book a meter reading at my property in M14 5TG, ideally on
> 25 June 2026."

**Expected routing:**
- `classify_intent` → **SCHEDULING** (keywords: *book*, *meter reading*). High
  confidence.
- *Routed to:* **anycompany-scheduling-agent**.

**Tools that fire (turn 1):**
- `check_availability(appointment_type="METER_READING", postcode="M14 5TG", preferred_date="2026-06-25")`

**Expected tool result:** a deterministic JSON object with a `slots` list — two
two-hour windows per day across three consecutive days from the preferred date.
The `slot_id` is self-describing (`SLOT-<TYPECODE>-<YYYYMMDD>-<windowIndex>-<hash>`),
so booking it later confirms exactly this date and window. Illustrative shape:
```json
{
  "appointment_type": "METER_READING",
  "postcode": "M14 5TG",
  "slots": [
    { "slot_id": "SLOT-MR-20260625-0-08af5e", "date": "2026-06-25", "time_window": "08:00-10:00", "engineer_type": "Meter Technician" },
    { "slot_id": "SLOT-MR-20260625-2-721c9b", "date": "2026-06-25", "time_window": "13:00-15:00", "engineer_type": "Meter Technician" },
    { "slot_id": "SLOT-MR-20260626-1-b4b250", "date": "2026-06-26", "time_window": "10:00-12:00", "engineer_type": "Meter Technician" }
  ]
}
```
*(The four possible windows are `08:00-10:00`, `10:00-12:00`, `13:00-15:00` and
`15:00-17:00`; the engineer type for METER_READING is `Meter Technician`. The
six-character hash tail in each `slot_id` will differ at runtime.)*

**Customer (turn 2):**
> "Book me the 08:00 slot. My account is ACC-100123 and my number is
> 07700 900123."

**Tools that fire (turn 2):**
- `book_appointment(account_number="ACC-100123", slot_id="SLOT-MR-20260625-0-08af5e", contact_phone="07700900123")`

**Expected tool result:**
```json
{
  "booking_reference": "APPT-143177",
  "confirmed_date": "2026-06-25",
  "confirmed_time_window": "08:00-10:00",
  "appointment_type": "METER_READING",
  "status": "CONFIRMED"
}
```
*(`booking_reference` is of the form `APPT-######` and will differ at runtime.
Crucially, `confirmed_date`/`confirmed_time_window` always match the slot that
was offered — they are decoded from the `slot_id`, never re-randomised.)*

**Expected response:**
"Your **meter reading** is confirmed for **25 June 2026, 08:00–10:00**. Your
booking reference is **APPT-143177**. We'll text the confirmation to
07700 900123." Mention the booking can be cancelled with `cancel_appointment`
using that reference.

> Variant: the same flow with
> `appointment_type="ENGINEER_VISIT"` (keyword *engineer*) demonstrates the
> other appointment types — `ENGINEER_VISIT`, `LEAK_INVESTIGATION`,
> `EV_CHARGER_INSTALL`.

---

## Conversation 4 — Escalation / complaint → ticket + transfer

**Customer:**
> "I want to make a complaint and speak to a manager. My account ACC-100789 has
> been wrongly put into arrears and I'm a vulnerable customer — this needs to be
> escalated."

**Expected routing:**
- `classify_intent` → **ESCALATION** (keywords: *complaint*, *manager*,
  *escalate*, *vulnerable*). High confidence.
- *Routed to:* **anycompany-escalation-agent**.

**Tools that fire (in order):**
1. `create_ticket(account_number="ACC-100789", issue_summary="Customer disputes arrears balance; reports being wrongly placed into arrears. Vulnerable customer requesting escalation to a manager.", category="VULNERABILITY", priority="HIGH")`
2. `transfer_to_agent(ticket_id="TKT-938209", reason="Vulnerable customer complaint regarding disputed arrears; requests manager escalation.", account_number="ACC-100789")`

**Expected tool results:**
```json
{
  "ticket_id": "TKT-938209",
  "status": "OPEN",
  "category": "VULNERABILITY",
  "priority": "HIGH",
  "sla_hours": 2,
  "created_at": "2026-06-18T09:00:00Z"
}
```
*(`ticket_id` is of the form `TKT-######` and will differ at runtime. The
VULNERABILITY category **pins the SLA to 2 hours regardless of the priority
passed** — even though `priority="HIGH"` (which alone maps to 8h), the category
override wins. See `open_ticket` in `lambdas/common/python/mock_data.py`.)*
```json
{
  "transfer_id": "XFER-227380",
  "ticket_id": "TKT-938209",
  "queue": "PrioritySupportQueue",
  "assigned_team": "Priority Services Team",
  "estimated_wait_minutes": 2,
  "connect_contact_id": "CT-ef6a8a2f",
  "status": "TRANSFERRED"
}
```
*(`transfer_id` (`XFER-######`) and `connect_contact_id` (`CT-<8 hex>`) are
distinct values and both differ at runtime. The queue/team are inferred from the
reason text: "vulnerable"/"medical" → `PrioritySupportQueue` / `Priority Services
Team`, ~2 min wait.)*

**Expected response:**
"I'm sorry to hear that, Priya. I've raised a priority case — ticket
**TKT-938209** (VULNERABILITY, **2-hour SLA**) — and I'm transferring you now to
our **Priority Services Team** (estimated wait ~2 minutes). They'll review the
disputed arrears with you. Your reference is TKT-938209."

**What this demonstrates:** the full escalation chain — a vulnerable
flag drives a high-priority ticket *and* an Amazon Connect handoff to the right
queue/team inferred from the reason text. Note the reason text drives queue
selection: a "vulnerable"/"medical" reason routes to the Priority Services
queue, whereas a reason mentioning "no water"/"leak" would route to the
**Emergency Supply** queue instead.

---

## Conversation 5 — Ambiguous / unknown intent (graceful fallback)

**Customer:**
> "Hi there, I just had a quick question about something."

**Expected routing:**
- `classify_intent` → no category keywords match, top normalised score
  **below 0.34** → **UNKNOWN**.
- The supervisor does **not** route to a specialist.

**Tools that fire:** none.

**Expected response:**
A graceful clarification, e.g. "Happy to help! I can assist with three things:
**billing** (balances, tariffs, payments), **appointments** (meter readings,
engineer visits, leak investigations, EV charger installs), and **complaints or
escalations**. Which of these is it about — or could you tell me a bit more?"

**What this demonstrates:** the classifier refuses to guess on low-confidence input
and falls back to a helpful menu instead of mis-routing, which protects the
customer experience and the deflection metrics.

---

## Quick reference — what to type in the demo

| # | Type into chat | Expect routed to | Tools |
|---|---|---|---|
| 1 | "Explain how the Economy 7 tariff works" | billing-agent | KB retrieval (RAG) |
| 2 | "Balance on ACC-100123 and due date?" | billing-agent | `check_balance` |
| 3 | "Book a meter reading in M14 5TG on 2026-06-25" then "book the 08:00 slot, ACC-100123, 07700900123" | scheduling-agent | `check_availability` → `book_appointment` |
| 4 | "Complaint, manager, vulnerable, ACC-100789 wrongly in arrears, escalate" | escalation-agent | `create_ticket` → `transfer_to_agent` |
| 5 | "I had a quick question about something" | (none — UNKNOWN) | none |

See [docs/README.md](./README.md) for deploy + ingestion steps and
[docs/architecture.md](./architecture.md) for the routing internals.
