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
    WorkerOptions,
    cli,
    function_tool,
    llm,
    tts,
)
from livekit import rtc
from livekit.plugins import silero

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
    else:
        lang_directive = (
            "--- LANGUAGE ---\n"
            "The caller chose English. Reply in English only."
        )
    return f"{_SYSTEM_PROMPT_BASE}\n\n{lang_directive}\n\n--- LIVE CONTEXT ---\n{time_ctx}"


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

    # English (and default) → Deepgram
    if not _DEEPGRAM_AVAILABLE:
        raise RuntimeError("livekit-plugins-deepgram not installed")
    model = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
    language = os.getenv("DEEPGRAM_LANGUAGE", "multi")
    return lk_deepgram.STT(
        model=model,
        language=language,
        interim_results=True,
        punctuate=True,
        smart_format=True,
        no_delay=True,
        endpointing_ms=200,
        keyterm=_STT_KEYTERMS_EN,
    )


def _build_tts(lang: str = "en"):
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

    # English (and default) → Inworld
    if not _INWORLD_AVAILABLE:
        raise RuntimeError("livekit-plugins-inworld not installed")
    voice = os.getenv("INWORLD_VOICE", "Abby")
    return lk_inworld.TTS(voice=voice, language="en-US")


# ── Agent class ──────────────────────────────────────────────────────────

class ShakiraAgent(Agent):
    """Shakira — EBH Academy AI Advisor (LiveKit transport)."""

    def __init__(self, lang: str = "en") -> None:
        super().__init__(instructions=_build_session_prompt(lang))
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
    async def capture_enrollment_lead(
        self,
        name: str,
        phone: str,
        email: str = "",
        course_interest: str = "",
        notes: str = "",
    ) -> dict:
        """Capture a prospective student's details for admissions follow-up.

        Call this after collecting the caller's name and phone number.
        The admissions team will reach out to them.
        """
        from tools.enrollment_lead import _handle
        result = await _handle({
            "name": name,
            "phone": phone,
            "email": email,
            "course_interest": course_interest,
            "notes": notes,
        })
        logger.info("Tool capture_enrollment_lead: %s — %s", name, result.get("status"))
        return result


# ── Entrypoint ───────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    """Called by LiveKit worker for each inbound web call."""
    await ctx.connect()
    logger.info("Room connected: %s", ctx.room.name)

    # Read language from job dispatch metadata (set by gateway)
    import json as _json
    lang = "en"
    try:
        raw_md = getattr(ctx.job, "metadata", "") or ""
        if raw_md:
            md = _json.loads(raw_md)
            cand = (md.get("lang") or "").strip().lower()
            if cand in {"en", "ar"}:
                lang = cand
    except Exception as me:
        logger.warning("Failed to parse job metadata, defaulting to English: %s", me)
    logger.info("Session language: %s", lang)

    stt_engine = _build_stt(lang)
    tts_engine = _build_tts(lang)

    # LLM setup
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    anthropic_model = os.getenv("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-5").strip()
    llm_temperature = float(os.getenv("LIVEKIT_TEMPERATURE", "0.3"))
    llm_max_tokens = int(os.getenv("LIVEKIT_MAX_COMPLETION_TOKENS", "400"))
    llm_max_tokens = max(64, min(llm_max_tokens, 2048))
    llm_timeout_s = float(os.getenv("LIVEKIT_LLM_TIMEOUT_S", "25"))

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

    # Session connection options
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
    vad = silero.VAD.load()

    allow_interruptions = os.getenv("LIVEKIT_ALLOW_INTERRUPTIONS", "true").strip().lower() in ("1", "true", "yes")

    agent = ShakiraAgent(lang=lang)
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
    else:
        greeting = os.getenv(
            "ACADEMY_GREETING",
            "Hello and welcome to EBH Academy! I'm Shakira, your academy advisor. How can I help you today?",
        )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=True),
    )
    await session.say(greeting)
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
