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

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
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


def _build_session_prompt() -> str:
    """Compose full prompt with real-time context."""
    time_ctx = build_uae_time_context()
    return f"{_SYSTEM_PROMPT_BASE}\n\n--- LIVE CONTEXT ---\n{time_ctx}"


# ── STT / TTS builders ──────────────────────────────────────────────────

def _build_stt():
    if not _DEEPGRAM_AVAILABLE:
        raise RuntimeError("livekit-plugins-deepgram not installed")
    model = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
    language = os.getenv("DEEPGRAM_LANGUAGE", "multi")
    return lk_deepgram.STT(
        model=model,
        language=language,
        detect_language=True,
        interim_results=True,
        punctuate=True,
        smart_format=True,
        no_delay=True,
        endpointing_ms=200,
        keywords=[
            ("CIDESCO", 5.0),
            ("KHDA", 5.0),
            ("DHA", 5.0),
            ("IAO", 3.0),
            ("dermaplaning", 5.0),
            ("maderotherapy", 5.0),
            ("Madero", 4.0),
            ("Tabby", 3.0),
            ("EBH", 5.0),
            ("Shakira", 4.0),
        ],
    )


def _build_tts():
    if not _INWORLD_AVAILABLE:
        raise RuntimeError("livekit-plugins-inworld not installed")
    voice = os.getenv("INWORLD_VOICE", "Elara")
    return lk_inworld.TTS(voice=voice)


# ── Agent class ──────────────────────────────────────────────────────────

class ShakiraAgent(Agent):
    """Shakira — EBH Academy AI Advisor (LiveKit transport)."""

    def __init__(self) -> None:
        super().__init__(instructions=_build_session_prompt())
        self._call_start = time.monotonic()
        self._tool_calls: list[dict] = []
        logger.info("ShakiraAgent initialized")

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

    stt_engine = _build_stt()
    tts_engine = _build_tts()

    # LLM setup
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    anthropic_model = os.getenv("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-5").strip()
    llm_temperature = float(os.getenv("LIVEKIT_TEMPERATURE", "0.3"))
    llm_max_tokens = int(os.getenv("LIVEKIT_MAX_COMPLETION_TOKENS", "80"))
    llm_max_tokens = max(32, min(llm_max_tokens, 1024))
    llm_timeout_s = float(os.getenv("LIVEKIT_LLM_TIMEOUT_S", "25"))

    llm_kwargs: dict[str, Any] = dict(
        model=anthropic_model,
        api_key=anthropic_key,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )
    caching = os.getenv("ANTHROPIC_PROMPT_CACHING", "true").strip().lower() in ("1", "true", "yes")
    if caching and hasattr(lk_anthropic.LLM, "__init__"):
        llm_kwargs["caching"] = True
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

    # Voice activity detection
    vad = silero.VAD.load(
        min_speech_duration=0.08,
        min_silence_duration=0.35,
    )

    allow_interruptions = os.getenv("LIVEKIT_ALLOW_INTERRUPTIONS", "true").strip().lower() in ("1", "true", "yes")

    agent = ShakiraAgent()
    session = AgentSession(
        stt=stt_engine,
        llm=llm_engine,
        tts=tts_engine,
        vad=vad,
        turn_detection=None,
    )

    # Greeting
    greeting = os.getenv(
        "ACADEMY_GREETING",
        "Hello and welcome to EBH Academy! I'm Shakira, your academy advisor. How can I help you today?",
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=True),
        conn_options=session_conn_opts,
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
