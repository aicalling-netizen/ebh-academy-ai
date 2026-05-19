# Shakira Realtime Voice Stack — Ultravox Native API

**Status:** Approved design, ready for planning
**Date:** 2026-05-20
**Project:** EBH Academy AI (`ebh-academy-ai/`)
**Author:** Claude (Opus 4.6) + user brainstorm

---

## 1. Goals

Beat the current Shakira stack on four dimensions simultaneously:

1. **Latency** — drop end-to-end from ~900-2200ms to ~150-300ms
2. **Tool calling** — keep native, typed function calling (4 tools)
3. **Realtime / full-duplex** — true interruptible conversation
4. **Quality** — equal-or-better voice quality, equal-or-better conversation flow

Constraints (from brainstorm):

- **Transport:** open to switching off LiveKit
- **Arabic:** English primary, Arabic functional (not flagship)
- **Budget:** under $0.10 per minute of conversation
- **Approach:** parallel stack, A/B test, then cut over (current Shakira stays live)

---

## 2. Chosen Platform: Ultravox Native API

Ultravox (Fixie) is a hosted speech-to-speech model with native tool calling, webhook-based external tools, voice overrides (ElevenLabs/Inworld/Google/etc.), and its own WebRTC transport.

**Why not the LiveKit Ultravox plugin?**
- Known bug: tool parameters coerced to strings ([livekit/agents#3713](https://github.com/livekit/agents/issues/3713))
- Doesn't expose voice overrides, SIP, lifecycle callbacks
- Wrapping it adds a layer that hides Ultravox's real capabilities

Since the user is open to switching transport, **Path A (Ultravox Native API)** is the chosen path.

---

## 3. Architecture

```
┌─────────────────────┐     WebRTC      ┌──────────────────────┐
│   Web Client        │◄───────────────►│   Ultravox Cloud     │
│ (Ultravox JS SDK)   │                 │ (STT + LLM + TTS)    │
└─────────┬───────────┘                 └──────────┬───────────┘
          │                                        │ webhooks (HMAC-signed)
          │ joinUrl                                ▼
          │                              ┌──────────────────────┐
          │                              │   Tool Server (EC2)  │
          │                              │  /ultravox/tools/*   │
          │                              │  (existing logic)    │
          │                              └──────────┬───────────┘
          ▼                                         │
┌─────────────────────┐                             ▼
│  FastAPI Gateway    │                  ┌──────────────────────┐
│  (existing main.py) │─── existing ────►│      Supabase        │
│  + new endpoints    │   persistence    │  (inquiries, logs)   │
└─────────────────────┘                  └──────────────────────┘
```

**Key flow:**
1. Web client hits `POST /api/public/ultravox-call` on the gateway
2. Gateway calls Ultravox API → creates a call → gets a `joinUrl`
3. Client opens WebRTC connection to Ultravox using the `joinUrl`
4. Speech-to-speech runs in Ultravox cloud
5. When Ultravox decides to call a tool, it POSTs to the EC2 tool server (HMAC-signed)
6. Tool server returns JSON; Ultravox speaks the response

---

## 4. Components

### 4.1 Ultravox Agent (configured once on ultravox.ai)

A persistent Agent record on Ultravox holding the default config:

- **System prompt:** `data/system_prompt_ultravox.txt` (Layer 1 — see §5)
- **Default voice:** Ultravox-hosted female voice (TBD during voice selection in implementation)
- **Tools (4):** webhook URLs pointing to the EC2 tool server
- **VAD settings:**
  - `turnEndpointDelay`: 384ms
  - `minimumInterruptionDuration`: 90ms
  - `frameActivationThreshold`: 0.1
- **Language hint:** set per-call (`en-US` or `ar`)
- **Max duration:** 1800s (30 min) — covers any realistic academy inquiry

Created via Ultravox console or API once at setup; `ULTRAVOX_AGENT_ID` stored in `.env`.

### 4.2 Tool Server (EC2 — alongside current Shakira)

Four new HTTP endpoints, mounted on the existing FastAPI gateway:

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/ultravox/tools/search-courses` | `{ query: str }` | `{ status, count, courses: [...] }` |
| `POST /api/ultravox/tools/search-faq` | `{ query: str }` | `{ status, count, results: [...] }` |
| `POST /api/ultravox/tools/capture-lead` | `{ name, phone, email?, course_interest?, notes? }` | `{ status, message, inquiry_id? }` |
| `POST /api/ultravox/tools/datetime` | `{ timezone?: str }` | `{ date, time, day_of_week, formatted }` |

Each endpoint:
1. Validates `X-Ultravox-Signature` header (HMAC-SHA256, shared secret from `ULTRAVOX_TOOL_SECRET`)
2. Delegates to the existing tool handler in `tools/*.py` (zero logic duplication)
3. Returns JSON shaped for Ultravox to speak

Live in a new module: `routers/ultravox_tools.py`. Mounted in `main.py`.

### 4.3 Web Client (`demo/shakira_realtime.html`)

New page modeled on the current `demo/index.html`:
- Same orb UI, language picker, transcript panel
- Fetches `joinUrl` from `/api/public/ultravox-call`
- Uses Ultravox JS client SDK (loaded from CDN) to open WebRTC connection
- Receives transcripts via Ultravox data messages, renders to transcript panel
- Hides "Talk to Shakira (Classic)" button vs "Talk to Shakira (Realtime)" — A/B routing

Current `demo/index.html` stays untouched and reachable.

### 4.4 Gateway endpoints (additions to `main.py`)

```
POST /api/public/ultravox-call
  body: { lang: "en"|"ar", source?: str }
  returns: { joinUrl, callId, agentId }
  rate-limited: 20/min/IP (same as existing)
```

Internal flow:
1. Picks language → builds Ultravox call body
2. Selects voice override (Inworld Arabic for `ar`, default Ultravox voice for `en`)
3. Sets `templateContext` with session metadata (campaign source, time of day, caller hints)
4. Calls Ultravox API → returns `joinUrl` to client

### 4.5 Session State (`core/conversation_state.py`)

The current Shakira prompt tracks state in-LLM (ANONYMOUS → IDENTIFIED → QUALIFIED → BOOKED). For Ultravox, this moves to **app code**:

- Per-call session record in Redis (or in-memory if Redis unavailable)
- Updated whenever a tool fires (e.g. `capture-lead` → mark IDENTIFIED)
- Optional dynamic prompt injection via Ultravox's `templateContext` if a state hint helps the model

This is a deliberate trade: the 8B Ultravox model is worse at long-running in-context state tracking; the app is much better at it.

---

## 5. System Prompt Strategy

Current prompt: 501 lines, Claude-optimized, embeds courses + FAQ + state machine + safety rules + bilingual templates.

New prompt structure (three layers):

### Layer 1 — Persona Core (in `system_prompt_ultravox.txt`, ~80 lines)
- Who Shakira is, EBH Academy context
- Voice style (max 25 words/reply, natural speech, AI disclosure)
- Safety: no job promises, no visa claims, parental consent for minors, suicidal ideation hotline
- Tool-use guidance (when to call which tool)
- Bilingual: brief Arabic phrasing notes (still Shakira the persona; Arabic TTS voice is the existing Inworld "Aisha")

### Layer 2 — Conversation State (in app code, `core/conversation_state.py`)
- Turn counter, identified/qualified/booked flags, CARE mode trigger detection
- Injected into the call as `templateContext` mustache variables when relevant
- Example: when `turn_count >= 4 && !phone_captured`, inject `{{contact_capture_hint}}` with "This is a good moment to ask if they'd like a free counseling call — get name and phone."

### Layer 3 — Reference Material (in tool responses)
- Course details, FAQ answers, DHA/KHDA/CIDESCO accuracy rules: returned by `search_courses` and `search_academy_faq` tools
- Existing tool implementations already return this; payloads may be enriched slightly (e.g. add `accreditation_note` field)

**Why this matters:** 8B models follow short, behavioral prompts well but lose accuracy when stuffed with 500 lines of conditional logic. Moving state to code and details to tools plays to Ultravox's strengths.

---

## 6. Voice Quality

### English
Ultravox-hosted female voice. Selection done during implementation (try 2-3 candidates, A/B internally).

### Arabic
**Inworld voice override** with the existing `Aisha` voice (same voice used in current Shakira's Faseeh fallback). Reuses existing Inworld account and tuning.

Example call body fragment for Arabic:
```json
{
  "languageHint": "ar",
  "voiceOverrides": {
    "inworld": { "voiceId": "Aisha" }
  }
}
```

Rationale: meets the "Arabic functional" bar without introducing a new TTS provider.

---

## 7. Cost Model

Ultravox base: ~$0.05/min (covers STT+LLM+TTS in one).
With Inworld voice override (Arabic only): adds Inworld TTS billing, already in budget.

Estimated total: **$0.05-0.08/min** depending on language mix.
Below the $0.10/min cap.

---

## 8. Migration & A/B Strategy

### Phase 1 — Build (single PR)
- Add `routers/ultravox_tools.py`, `core/ultravox_client.py`, `core/conversation_state.py`, `data/system_prompt_ultravox.txt`, `demo/shakira_realtime.html`
- Add `/api/public/ultravox-call` to `main.py`
- Add `ULTRAVOX_API_KEY`, `ULTRAVOX_AGENT_ID`, `ULTRAVOX_TOOL_SECRET` to `.env.example`
- Tests: signature validation, tool endpoint logic, call creation request shape
- **Do not touch** `agent.py`, current `tools/*.py` logic, current `demo/index.html`

### Phase 2 — Internal testing
- Team uses `shakira_realtime.html` end-to-end
- Measure:
  - TTFT (time to first audio token)
  - Full-turn latency (user stop → agent first word)
  - Tool-call success rate (HTTP 200 + parseable JSON)
  - Subjective voice quality (1-5 scoring rubric)
- Tune: VAD settings, voice selection, prompt phrasing

### Phase 3 — Public A/B (separate PR)
- Add `?engine=realtime|classic` query param on landing page
- 50/50 split, or by campaign source
- Both engines write to same `inquiries` table with new `engine` column
- Track via dashboard: completion rate, lead capture rate, avg call duration

### Phase 4 — Cut over (separate PR, only after data wins)
- Default landing → realtime engine
- Keep `?engine=classic` available for fallback
- Deprecate classic after ~4 weeks of stable realtime performance

---

## 9. File Plan

**New:**
```
ebh-academy-ai/
├── routers/
│   └── ultravox_tools.py        # 4 webhook endpoints
├── core/
│   ├── ultravox_client.py       # Ultravox REST API wrapper
│   └── conversation_state.py    # Session state tracking
├── data/
│   └── system_prompt_ultravox.txt
├── demo/
│   └── shakira_realtime.html
└── tests/
    ├── test_ultravox_tools.py
    └── test_ultravox_client.py
```

**Modified:**
```
main.py            # Mount ultravox_tools router, add /api/public/ultravox-call
.env.example       # New ULTRAVOX_* vars
requirements.txt   # Add httpx if not already pinned (Ultravox API client)
```

**Untouched:**
```
agent.py           # Current LiveKit Shakira (classic engine)
tools/*.py         # Logic reused via imports
core/db.py         # Same Supabase persistence
demo/index.html    # Classic web client
```

---

## 10. Open Questions (resolve during planning)

1. **Voice selection** — which Ultravox-hosted female voice fits Shakira's persona? Test 2-3 candidates in Phase 2.
2. **Redis vs in-memory state** — current `ebh-academy-ai/core/` doesn't appear to use Redis. Start with in-memory dict (single-process gateway); upgrade to Redis if we ever run multi-instance.
3. **Ultravox JS SDK loading** — official CDN? Self-hosted bundle? Resolve during web client build.
4. **Initial message** — should Shakira speak first (`firstSpeakerSettings.agent`) or wait for caller (`firstSpeakerSettings.user`)? Default: agent speaks first with "Hi, this is Shakira from EBH Academy — how can I help?"

---

## 11. Non-goals (explicit YAGNI)

- **No SIP/phone integration** — web only for now. Ultravox supports SIP but it's out of scope.
- **No recording-and-replay** — Ultravox supports `recordingEnabled: true` but enabling and storing recordings is a separate decision.
- **No retraining** — using Ultravox's hosted model as-is. No fine-tuning.
- **No prompt-as-source-of-truth for courses/FAQ** — moved to tool responses.
- **No swapping LiveKit Shakira to use Ultravox-via-LiveKit-plugin** — that's Path B and is explicitly rejected.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| 8B model struggles with Shakira's nuance | Layered prompt strategy (§5); fall back to classic via A/B |
| Arabic voice quality dips vs current Faseeh | Voice override → Inworld Aisha (same as current) |
| Webhook latency adds to perceived TTFT | Tool endpoints on same EC2 as gateway, low intra-region latency |
| Ultravox tool param string coercion bug | Bug is in LiveKit plugin (Path B); native API (Path A) has typed params |
| 30-min max call duration insufficient | Realistic academy calls are 3-8 min; if needed, raise to 60min |
| Ultravox vendor lock-in | A/B parallel keeps classic ready as failover; can swap to OpenAI Realtime later if needed |

---

## 13. Success Criteria

Realtime cuts over to default if **all** are true after Phase 3:

- TTFT p50 < 400ms (vs current ~1500ms)
- Full-turn latency p50 < 700ms (vs current ~2000ms)
- Tool-call success rate ≥ 98%
- Lead capture rate per call ≥ current classic baseline
- No regression in completion rate vs classic
- Subjective voice quality ≥ 4/5 average in team testing
