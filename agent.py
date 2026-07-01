"""LiveKit voice agent worker for EBH Academy.

Web-only voice agent — no SIP/telephony. Handles inbound web calls
from prospective students asking about courses, enrollment, and certifications.

Runtime:
- LLM: Anthropic Claude
- STT: Deepgram
- TTS: Inworld
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent))

import logging

from livekit.agents import (
    APIConnectOptions,
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    llm,
    tts,
)
from livekit import rtc
try:
    # Only needed for the cascaded stack's VAD; the realtime stack uses
    # OpenAI's server-side turn detection, so silero (and its torch dep)
    # is optional.
    from livekit.plugins import silero
    _SILERO_AVAILABLE = True
except ImportError:
    _SILERO_AVAILABLE = False

try:
    from livekit.plugins import deepgram as lk_deepgram
    _DEEPGRAM_AVAILABLE = True
except ImportError:
    _DEEPGRAM_AVAILABLE = False

try:
    from livekit.plugins import inworld as lk_inworld
    _INWORLD_AVAILABLE = True
except ImportError:
    _INWORLD_AVAILABLE = False

try:
    from livekit.plugins import anthropic as lk_anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    from livekit.plugins import munsit as lk_munsit
    _MUNSIT_AVAILABLE = True
except ImportError:
    _MUNSIT_AVAILABLE = False

try:
    from livekit.plugins import faseeh as lk_faseeh
    _FASEEH_AVAILABLE = True
except ImportError:
    _FASEEH_AVAILABLE = False

try:
    from livekit.plugins import groq as lk_groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

try:
    from livekit.plugins import elevenlabs as lk_elevenlabs
    _ELEVENLABS_AVAILABLE = True
except ImportError:
    _ELEVENLABS_AVAILABLE = False

try:
    # Bundled via livekit-agents[openai]; used for the speech-to-speech realtime stack.
    from livekit.plugins import openai as lk_openai
    _OPENAI_REALTIME_AVAILABLE = True
except ImportError:
    _OPENAI_REALTIME_AVAILABLE = False

from livekit.agents.voice.agent_session import SessionConnectOptions
from core.time_context import build_uae_time_context, now_uae

logger = logging.getLogger("livekit-shakira")

# ── Startup validation ───────────────────────────────────────────────────

def _validate_env() -> None:
    missing: list[str] = []
    for var, desc in {
        "LIVEKIT_URL": "LiveKit server URL",
        "LIVEKIT_API_KEY": "LiveKit API key",
        "LIVEKIT_API_SECRET": "LiveKit API secret",
    }.items():
        if not os.getenv(var, "").strip():
            missing.append(f"  {var} — {desc}")

    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        missing.append("  ANTHROPIC_API_KEY — required for Claude LLM")
    if not os.getenv("DEEPGRAM_API_KEY", "").strip():
        missing.append("  DEEPGRAM_API_KEY — required for speech-to-text")
    if not os.getenv("INWORLD_API_KEY", "").strip():
        missing.append("  INWORLD_API_KEY — required for text-to-speech")

    if missing:
        msg = (
            "\n╔══════════════════════════════════════════════════════════╗\n"
            "║  FATAL: Required environment variables are missing      ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
            + "\n".join(missing)
            + "\n\nSet these in .env or your environment and restart."
        )
        logger.critical(msg)
        raise SystemExit(msg)

_validate_env()

# ── System prompt ────────────────────────────────────────────────────────

_PROMPT_FILE = Path(__file__).parent / "data" / "system_prompt.txt"


def _load_prompt() -> str:
    if not _PROMPT_FILE.exists():
        logger.warning("System prompt file not found: %s", _PROMPT_FILE)
        return (
            "You are Shakira, a warm and knowledgeable AI advisor at EBH Academy in Dubai. "
            "Help callers with course information, enrollment, and certifications. "
            "Keep responses short and conversational."
        )
    text = _PROMPT_FILE.read_text(encoding="utf-8")
    text = text.replace("{ACADEMY_NAME}", os.getenv("ACADEMY_NAME", "EBH Academy"))
    logger.info("Loaded system prompt (%d chars)", len(text))
    return text


_SYSTEM_PROMPT_BASE = _load_prompt()


def _build_session_prompt(lang: str = "en") -> str:
    """Compose full prompt with real-time context and language directive."""
    time_ctx = build_uae_time_context()
    if lang == "ar":
        lang_directive = (
            "--- LANGUAGE ---\n"
            "The caller chose Arabic. You MUST reply in Modern Standard Arabic only. "
            "Do not switch to English even if the caller mixes English words. "
            "Keep numbers, prices (AED), and proper names readable; the rest must be Arabic."
        )
    elif lang == "hi":
        lang_directive = (
            "--- LANGUAGE ---\n"
            "The caller chose Hindi (हिन्दी). You MUST reply in conversational Hindi (Devanagari script). "
            "Use natural spoken Hindi; English loanwords for technical terms (course, certificate, AED) are fine. "
            "Do not switch to English, Arabic, or any other language."
        )
    elif lang == "ur":
        lang_directive = (
            "--- LANGUAGE ---\n"
            "The caller chose Urdu (اردو). You MUST reply in conversational Urdu (Nastaliq script). "
            "Use natural spoken Urdu; English loanwords for technical terms (course, certificate, AED) are fine. "
            "Do not switch to English, Hindi, Arabic, or any other language."
        )
    else:
        lang_directive = (
            "--- LANGUAGE ---\n"
            "The caller chose English. Reply in English only."
        )
    return f"{_SYSTEM_PROMPT_BASE}\n\n{lang_directive}\n\n--- LIVE CONTEXT ---\n{time_ctx}"


# ── Production-parity realtime prompt: guardrails + inbound/outbound channel ──
_RT_GUARDRAILS_FILE = Path(__file__).parent / "data" / "system_prompt_realtime_guardrails.txt"
_RT_INBOUND_FILE = Path(__file__).parent / "data" / "system_prompt_realtime_inbound.txt"
_RT_OUTBOUND_FILE = Path(__file__).parent / "data" / "system_prompt_realtime_outbound.txt"


def _build_realtime_prompt(lang: str = "en", outbound: bool = False) -> str:
    """Realtime instructions = CORE GUARDRAILS + the inbound/outbound channel
    prompt, read FRESH from disk each call (no restart to change), with the live
    Dubai datetime substituted. Falls back to the base prompt if files are missing.
    Mirrors PAM's realtime_probe_agent.py prompt structure."""
    try:
        guardrails = _RT_GUARDRAILS_FILE.read_text(encoding="utf-8").strip()
        channel = (_RT_OUTBOUND_FILE if outbound else _RT_INBOUND_FILE).read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning("Realtime prompt files missing (%r) — falling back to base prompt", e)
        return _build_session_prompt(lang)
    channel = channel.replace("{CURRENT_UAE_DATETIME}", build_uae_time_context())
    lang_line = "\n\nThe caller chose Arabic — reply in Arabic only." if lang == "ar" else ""
    return f"{guardrails}\n\n{channel}{lang_line}"


# ── STT / TTS builders ──────────────────────────────────────────────────

# Nova-3 "multi" mode does NOT include Arabic — must use language=ar explicitly.
_STT_KEYTERMS_EN = [
    "CIDESCO", "KHDA", "DHA", "IAO",
    "dermaplaning", "maderotherapy", "Madero",
    "Tabby", "EBH", "Shakira",
]
_STT_KEYTERMS_AR = [
    "شاكيرا", "أكاديمية", "دبي", "دورة", "شهادة",
]
# Shared by Hindi + Urdu — academy/industry acronyms keep English form in both languages
_STT_KEYTERMS_HI = [
    "CIDESCO", "KHDA", "DHA", "IAO", "EBH", "Shakira",
    "dermaplaning", "Madero", "Tabby",
    "अकादमी", "कोर्स", "दुबई", "शकीरा",
]


def _build_stt(lang: str = "en"):
    # Arabic → Munsit (much better Arabic accuracy than Deepgram)
    if lang == "ar":
        munsit_key = os.getenv("MUNSIT_API_KEY", "").strip()
        if not munsit_key:
            raise RuntimeError("MUNSIT_API_KEY not set — required for Arabic STT")
        model = os.getenv("MUNSIT_STT_MODEL", "munsit-en-ar")  # code-switching

        # Default to our custom streaming adapter (true streaming, ~500ms interim updates).
        # Set MUNSIT_MODE=batch to fall back to the upstream plugin's batch mode
        # (slower per-utterance latency but well-tested).
        munsit_mode = os.getenv("MUNSIT_MODE", "streaming").strip().lower()
        if munsit_mode == "streaming":
            from core.munsit_streaming_stt import MunsitStreamingSTT
            return MunsitStreamingSTT(
                api_key=munsit_key,
                model=model,
                language="ar",
            )

        if not _MUNSIT_AVAILABLE:
            raise RuntimeError("livekit-plugins-munsit not installed")
        return lk_munsit.STT(
            model=model,
            mode="batch",
            api_key=munsit_key,
            language="ar",
        )

    # English / Hindi / Urdu → Deepgram
    if not _DEEPGRAM_AVAILABLE:
        raise RuntimeError("livekit-plugins-deepgram not installed")
    model = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
    if lang == "hi":
        language = "hi"
        keyterm = _STT_KEYTERMS_HI
    elif lang == "ur":
        # Nova-3 supports Urdu (ur). Falls back to nova-2 if Deepgram rejects.
        language = "ur"
        keyterm = _STT_KEYTERMS_HI  # share — most Hindi keyterms apply to Urdu too
    else:
        language = os.getenv("DEEPGRAM_LANGUAGE", "multi")
        keyterm = _STT_KEYTERMS_EN
    return lk_deepgram.STT(
        model=model,
        language=language,
        interim_results=True,
        punctuate=True,
        smart_format=True,
        no_delay=True,
        endpointing_ms=200,
        keyterm=keyterm,
    )


def _build_tts(lang: str = "en", stack: str = "cascaded"):
    # English + cascaded stack (A/B test) → ElevenLabs
    if lang == "en" and stack == "cascaded":
        if not _ELEVENLABS_AVAILABLE:
            raise RuntimeError("livekit-plugins-elevenlabs not installed")
        el_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if not el_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set — required for cascaded stack TTS")
        voice = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Bella, warm female
        model = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
        return lk_elevenlabs.TTS(voice_id=voice, model=model, api_key=el_key, language="en")

    # Arabic → Faseeh (Munsit's TTS, dedicated Arabic voices)
    if lang == "ar":
        if not _FASEEH_AVAILABLE:
            raise RuntimeError("livekit-plugins-faseeh not installed")
        munsit_key = os.getenv("MUNSIT_API_KEY", "").strip()
        if not munsit_key:
            raise RuntimeError("MUNSIT_API_KEY not set — required for Arabic TTS")
        voice = os.getenv("FASEEH_VOICE", "WcxyRPjVQcpVYmceBQO4Helb")  # Aisha (Emirati female)
        model = os.getenv("FASEEH_MODEL", "faseeh-v1-preview")
        speed = float(os.getenv("FASEEH_SPEED", "1.04"))
        stability = float(os.getenv("FASEEH_STABILITY", "1.0"))
        return lk_faseeh.TTS(
            voice_id=voice,
            model=model,
            api_key=munsit_key,
            stability=stability,
            speed=speed,
        )

    # Hindi / Urdu / English → Inworld
    if not _INWORLD_AVAILABLE:
        raise RuntimeError("livekit-plugins-inworld not installed")
    if lang == "hi":
        voice = os.getenv("INWORLD_VOICE_HI", "Riya")  # Hindi female, professional & clean
        return lk_inworld.TTS(voice=voice, language="hi-IN")
    if lang == "ur":
        # Inworld has no native Urdu voice; Aanya (Hindi) speaks Urdu acceptably
        # since spoken Urdu and Hindi are >90% mutually intelligible.
        voice = os.getenv("INWORLD_VOICE_UR", "Aanya")
        return lk_inworld.TTS(voice=voice, language="hi-IN")
    voice = os.getenv("INWORLD_VOICE", "Abby")
    return lk_inworld.TTS(voice=voice, language="en-US")


# ── Agent class ──────────────────────────────────────────────────────────

class ShakiraAgent(Agent):
    """Shakira — EBH Academy AI Advisor (LiveKit transport)."""

    def __init__(self, lang: str = "en", instructions: "str | None" = None) -> None:
        super().__init__(instructions=instructions or _build_session_prompt(lang))
        self._call_start = time.monotonic()
        self._tool_calls: list[dict] = []
        self._lang = lang
        logger.info("ShakiraAgent initialized (lang=%s)", lang)

    # ── Tools ────────────────────────────────────────────────────────────

    @function_tool()
    async def get_current_datetime(self, timezone: str = "UAE +04:00") -> dict:
        """Get current date and time in UAE."""
        from tools.current_datetime import _handle
        result = await _handle({"timezone": timezone})
        logger.info("Tool get_current_datetime: %s", result.get("formatted", ""))
        return result

    @function_tool()
    async def search_courses(self, query: str) -> dict:
        """Search EBH Academy courses by keyword or area of interest.

        Use this when a caller asks about courses, pricing, duration,
        certifications, or what programs are available.
        """
        from tools.course_inquiry import _handle
        result = await _handle({"query": query})
        logger.info("Tool search_courses(%s): %d results", query, result.get("count", 0))
        return result

    @function_tool()
    async def search_academy_faq(self, query: str) -> dict:
        """Search academy FAQ for accreditation, policies, payment, location questions.

        Use for questions about KHDA, CIDESCO, DHA licensing, Tabby payments,
        prerequisites, refund policy, location, or general academy information.
        """
        from tools.academy_faq import _handle
        result = await _handle({"query": query})
        logger.info("Tool search_academy_faq(%s): %d results", query, result.get("count", 0))
        return result

    @function_tool()
    async def enroll(
        self,
        context: RunContext,
        name: str,
        phone: str,
        email: str = "",
        course_interest: str = "",
        notes: str = "",
    ) -> dict:
        """Reserve a place / trial class / advisory session for a prospective student.

        Call this once you have the caller's name and phone number. Admissions
        follows up to confirm. Blocked if a voicemail/machine was detected.
        """
        st = getattr(context, "userdata", None) or {}
        if isinstance(st, dict) and st.get("voicemail"):
            logger.warning("enroll BLOCKED: voicemail/machine detected (phone=%s)", phone)
            return {"status": "error", "message": "No live caller (voicemail detected) — no enrolment created."}
        from tools.enrollment_lead import _handle
        result = await _handle({
            "name": name, "phone": phone, "email": email,
            "course_interest": course_interest, "notes": notes,
        })
        logger.info("Tool enroll: %s — %s", name, result.get("status"))
        return result

    @function_tool()
    async def check_class_availability(self, date: str = "", course: str = "") -> dict:
        """Check upcoming intakes / trial-class or advisory-session availability before reserving a place."""
        logger.info("Tool check_class_availability(date=%s course=%s)", date, course)
        return {
            "status": "ok",
            "message": "We have flexible intakes and trial sessions most weekdays, roughly ten to seven. "
                       "Take the caller's preferred day and reserve a place with enroll — admissions confirms the exact time.",
        }

    @function_tool()
    async def transfer_to_human(self, context: RunContext, reason: str = "", category: str = "OPS-ESC") -> dict:
        """Escalate to the admissions team (caller asks for a person/manager, complaint, refund,
        or an unresolved issue). Notifies the team to call back — never promise a live person now."""
        st = getattr(context, "userdata", None) or {}
        phone = str((st or {}).get("phone") or "").strip()
        name = str((st or {}).get("caller_name") or "").strip()
        url = os.getenv("N8N_ESCALATION_WEBHOOK_URL", "").strip()
        try:
            if url:
                import httpx
                async with httpx.AsyncClient(timeout=6.0) as c:
                    await c.post(url, json={"reason": str(reason or "")[:200], "category": (category or "OPS-ESC").strip(),
                                            "phone": phone, "name": name, "source": "realtime", "channel": "web"})
                logger.info("transfer_to_human: escalation sent (phone=%s cat=%s)", phone, category)
            else:
                logger.warning("transfer_to_human: N8N_ESCALATION_WEBHOOK_URL not set — logged only")
            if isinstance(st, dict):
                st["escalated"] = True
        except Exception as e:
            logger.warning("transfer_to_human failed: %r", e)
        return {"status": "callback_requested",
                "message": "Our admissions team has been notified and will reach out to you very shortly."}

    @function_tool()
    async def end_call(self, context: RunContext, reason: str = "assistant_requested_end") -> dict:
        """End the call after a brief closing line. Use when the conversation is complete
        (a place is reserved, or the caller said goodbye). Say the closing sentence FIRST, then call this."""
        st = getattr(context, "userdata", None) or {}
        if isinstance(st, dict):
            st["end_requested"] = True
        room_name = str((st or {}).get("room_name") or "").strip()
        if not room_name:
            return {"status": "unavailable", "message": ""}

        async def _delayed_hangup():
            await asyncio.sleep(6)  # let the closing line finish playing
            await _hangup_room(room_name, f"end_call:{reason}")

        asyncio.create_task(_delayed_hangup())
        return {"status": "scheduled", "message": "Call will end right after the closing sentence."}


# ── Entrypoint ───────────────────────────────────────────────────────────

# ── Silence watchdog: auto-hang-up on dead air / voicemail (ported from PAM realtime) ──
def _wd_env_f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except Exception:
        return default


_SILENCE_WARN_S = _wd_env_f("REALTIME_SILENCE_WARNING_S", 12.0)   # dead air after caller spoke -> "still there?"
_SILENCE_GRACE_S = _wd_env_f("REALTIME_SILENCE_GRACE_S", 10.0)    # more dead air after warning -> hang up
_NOSPEAK_HANGUP_S = _wd_env_f("REALTIME_NOSPEAK_HANGUP_S", 30.0)  # caller never speaks -> hang up (0 = off)


def _watchdog_action(*, now: float, mono_start: float, last_caller_ts: float, caller_turns: int,
                     agent_speaking: bool, idle_warned_at: float) -> "str | None":
    """Pure decision: None | 'warn' | 'reset' | 'hangup_silence' | 'hangup_nospeak'."""
    # Caller never spoke (voicemail / dead line) -> hang up to stop burning realtime cost.
    if _NOSPEAK_HANGUP_S > 0 and caller_turns == 0 and (now - mono_start) >= _NOSPEAK_HANGUP_S:
        return "hangup_nospeak"
    # Never act while Shakira is talking.
    if agent_speaking:
        return None
    silent = now - last_caller_ts
    if idle_warned_at <= 0:
        if caller_turns > 0 and silent >= _SILENCE_WARN_S:
            return "warn"
        return None
    if last_caller_ts > idle_warned_at:   # caller resumed after the warning
        return "reset"
    if (now - idle_warned_at) >= _SILENCE_GRACE_S:
        return "hangup_silence"
    return None


async def _hangup_room(room_name: str, reason: str) -> None:
    """Delete the LiveKit room to end the call. Used by the watchdog. Never raises."""
    if not room_name:
        return
    try:
        from livekit import api as _lkapi
        lk = _lkapi.LiveKitAPI(
            url=os.environ["LIVEKIT_URL"],
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
        )
        try:
            await lk.room.delete_room(_lkapi.DeleteRoomRequest(room=room_name))
        finally:
            await lk.aclose()
        logger.info("watchdog hangup: room closed (room=%s reason=%s)", room_name, reason)
    except Exception as e:
        logger.warning("watchdog hangup failed (room=%s reason=%s): %r", room_name, reason, e)


def _install_silence_watchdog(session, ctx):
    """Wire the dead-air auto-hangup watchdog onto a realtime session.

    Tracks caller speech via user_state_changed (VAD-driven — fires without input
    transcription or a local silero VAD). Returns a coroutine to spawn as a task.
    """
    wd = {
        "mono_start": time.monotonic(), "last_caller_ts": time.monotonic(),
        "caller_turns": 0, "agent_speaking": False, "idle_warned_at": 0.0, "ended": False,
    }

    @session.on("user_state_changed")
    def _on_user_state(ev):
        try:
            if getattr(ev, "new_state", "") == "speaking":
                wd["last_caller_ts"] = time.monotonic()
                wd["caller_turns"] += 1
                wd["idle_warned_at"] = 0.0
        except Exception:
            pass

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        try:
            speaking = getattr(ev, "new_state", "") == "speaking"
            wd["agent_speaking"] = speaking
            if speaking:
                wd["idle_warned_at"] = 0.0
        except Exception:
            pass

    async def _loop():
        room_name = ctx.room.name
        while not wd["ended"]:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            action = _watchdog_action(
                now=now, mono_start=wd["mono_start"], last_caller_ts=wd["last_caller_ts"],
                caller_turns=wd["caller_turns"], agent_speaking=wd["agent_speaking"],
                idle_warned_at=wd["idle_warned_at"],
            )
            if action == "warn":
                wd["idle_warned_at"] = now
                try:
                    await session.generate_reply(
                        instructions="Gently check if the caller is still there, in one short sentence."
                    )
                except Exception:
                    pass
            elif action in ("hangup_silence", "hangup_nospeak"):
                wd["ended"] = True
                logger.info("watchdog: ending call (%s) room=%s", action, room_name)
                await _hangup_room(room_name, action)
                return

    return _loop


async def entrypoint(ctx: JobContext) -> None:
    """Called by LiveKit worker for each inbound web call."""
    await ctx.connect()
    logger.info("Room connected: %s", ctx.room.name)

    # Read language + stack from job dispatch metadata (set by gateway)
    import json as _json
    lang = "en"
    stack = "cascaded"  # default for non-Ultravox path (Ultravox calls don't dispatch this agent)
    try:
        raw_md = getattr(ctx.job, "metadata", "") or ""
        if raw_md:
            md = _json.loads(raw_md)
            cand_lang = (md.get("lang") or "").strip().lower()
            if cand_lang in {"en", "ar", "hi", "ur"}:
                lang = cand_lang
            cand_stack = (md.get("stack") or "").strip().lower()
            if cand_stack in {"cascaded", "ultravox", "realtime"}:
                stack = cand_stack
    except Exception as me:
        logger.warning("Failed to parse job metadata, defaulting: %s", me)
    logger.info("Session lang=%s stack=%s", lang, stack)

    # Realtime uses the production-parity guardrails+inbound prompt structure
    # (read fresh per call); other stacks use the base session prompt.
    if stack == "realtime":
        agent = ShakiraAgent(lang=lang, instructions=_build_realtime_prompt(lang, outbound=False))
    else:
        agent = ShakiraAgent(lang=lang)

    if stack == "realtime" and lang == "en":
        # ── Speech-to-speech: OpenAI Realtime (best phone-call experience) ──
        # The model does STT + LLM + TTS in one connection with native
        # turn-taking and barge-in — no separate engines, no Silero VAD.
        # That fluid turn-taking is the whole reason to use this for phone.
        if not _OPENAI_REALTIME_AVAILABLE:
            raise RuntimeError("livekit-plugins-openai not installed — required for realtime stack")
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY not set — required for realtime stack")
        rt_model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-mini").strip()
        rt_voice = os.getenv("OPENAI_REALTIME_VOICE", "marin").strip()
        # OpenAI Realtime requires temperature >= 0.6; clamp so a stray 0.3 won't 400.
        rt_temp = max(0.6, float(os.getenv("OPENAI_REALTIME_TEMPERATURE", "0.7")))
        realtime_model = lk_openai.realtime.RealtimeModel(
            model=rt_model,
            voice=rt_voice,
            temperature=rt_temp,
            api_key=openai_key,
            # Disable the input-audio transcription sidecar. The S2S model
            # understands caller audio natively; the sidecar only produces a
            # text copy for the transcript panel, and on gpt-realtime-mini it
            # currently errors with request_headers_too_large. Re-enable with a
            # working transcription model once that's sorted.
            input_audio_transcription=None,
        )
        session = AgentSession(
            llm=realtime_model,
            userdata={"room_name": ctx.room.name, "phone": "", "caller_name": "",
                      "voicemail": False, "end_requested": False, "escalated": False},
        )
        logger.info("Stack: OpenAI Realtime (model=%s voice=%s temp=%.2f)", rt_model, rt_voice, rt_temp)
    else:
        # ── Cascaded: STT + LLM + TTS ──
        # English → Groq Llama-4-Scout + ElevenLabs; Arabic/Hindi/Urdu → Claude Haiku + Inworld/Faseeh.
        stt_engine = _build_stt(lang)
        tts_engine = _build_tts(lang, stack)

        llm_temperature = float(os.getenv("LIVEKIT_TEMPERATURE", "0.3"))
        llm_max_tokens = int(os.getenv("LIVEKIT_MAX_COMPLETION_TOKENS", "400"))
        llm_max_tokens = max(64, min(llm_max_tokens, 2048))
        llm_timeout_s = float(os.getenv("LIVEKIT_LLM_TIMEOUT_S", "25"))

        if lang == "en" and stack == "cascaded":
            if not _GROQ_AVAILABLE:
                raise RuntimeError("livekit-plugins-groq not installed")
            groq_key = os.getenv("GROQ_API_KEY", "").strip()
            if not groq_key:
                raise RuntimeError("GROQ_API_KEY not set — required for cascaded stack LLM")
            groq_model = os.getenv("GROQ_LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct").strip()
            llm_engine = lk_groq.LLM(
                model=groq_model,
                api_key=groq_key,
                temperature=llm_temperature,
                max_completion_tokens=llm_max_tokens,
            )
            logger.info("LLM: Groq %s", groq_model)
        else:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            anthropic_model = os.getenv("ANTHROPIC_LLM_MODEL", "claude-haiku-4-5-20251001").strip()
            llm_kwargs: dict[str, Any] = dict(
                model=anthropic_model,
                api_key=anthropic_key,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
            )
            caching = os.getenv("ANTHROPIC_PROMPT_CACHING", "true").strip().lower() in ("1", "true", "yes")
            if caching:
                llm_kwargs["caching"] = "ephemeral"
            llm_engine = lk_anthropic.LLM(**llm_kwargs)
            logger.info("LLM: Anthropic %s", anthropic_model)

        session_conn_opts = SessionConnectOptions(
            llm_conn_options=APIConnectOptions(
                max_retry=1,
                retry_interval=0.5,
                timeout=max(5.0, llm_timeout_s),
            ),
            stt_conn_options=APIConnectOptions(max_retry=0, timeout=20.0),
            tts_conn_options=APIConnectOptions(max_retry=0, timeout=30.0),
        )

        # Voice activity detection (matches PAM — defaults only)
        if not _SILERO_AVAILABLE:
            raise RuntimeError("livekit-plugins-silero not installed — required for the cascaded stack VAD")
        vad = silero.VAD.load()

        session = AgentSession(
            stt=stt_engine,
            llm=llm_engine,
            tts=tts_engine,
            vad=vad,
            conn_options=session_conn_opts,
        )

    # Greeting (language-specific)
    if lang == "ar":
        greeting = os.getenv(
            "ACADEMY_GREETING_AR",
            "مرحباً بك في أكاديمية إي بي إتش! أنا شاكيرا، مستشارتك الأكاديمية. كيف يمكنني مساعدتك اليوم؟",
        )
    elif lang == "hi":
        greeting = os.getenv(
            "ACADEMY_GREETING_HI",
            "नमस्ते! EBH अकादमी में आपका स्वागत है। मैं शकीरा हूँ, आपकी अकादमिक सलाहकार। मैं आज आपकी कैसे मदद कर सकती हूँ?",
        )
    elif lang == "ur":
        greeting = os.getenv(
            "ACADEMY_GREETING_UR",
            "السلام علیکم! EBH اکیڈمی میں خوش آمدید۔ میں شکیرا ہوں، آپ کی اکیڈمک ایڈوائزر۔ میں آج آپ کی کیسے مدد کر سکتی ہوں؟",
        )
    else:
        greeting = os.getenv(
            "ACADEMY_GREETING",
            "Hello and welcome to EBH Academy! I'm Shakira, your academy advisor. How can I help you today?",
        )

    # Wire the dead-air auto-hangup watchdog (realtime only) before the session starts,
    # so its event handlers are registered in time.
    _watchdog_loop = _install_silence_watchdog(session, ctx) if stack == "realtime" else None

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=True),
    )
    if stack == "realtime":
        # A RealtimeSession has no separate TTS, so session.say() isn't
        # available — have the model speak the greeting instead.
        await session.generate_reply(
            instructions=f"Greet the caller warmly as Shakira. Say exactly: {greeting}"
        )
    else:
        await session.say(greeting)

    if _watchdog_loop is not None:
        asyncio.create_task(_watchdog_loop())
        logger.info("Silence watchdog armed (warn=%.0fs grace=%.0fs nospeak=%.0fs)",
                    _SILENCE_WARN_S, _SILENCE_GRACE_S, _NOSPEAK_HANGUP_S)

    logger.info("Shakira agent started in room %s", ctx.room.name)


# ── Worker CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("LIVEKIT_AGENT_NAME", "academy-agent"),
        ),
    )
