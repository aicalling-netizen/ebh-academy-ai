"""FastAPI gateway for EBH Academy AI voice agent.

Serves:
- Public LiveKit token endpoint (web callers)
- Static files (LiveKit client JS)
- Demo pages (Talk to Shakira, dashboard)
- API endpoints for inquiry management
"""
from __future__ import annotations

import os
import time
import logging
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("academy-gateway")

# ── Startup env-var validation ───────────────────────────────────────────

_REQUIRED_ENV_VARS = {
    "LIVEKIT_URL": "LiveKit server URL (e.g. wss://your-project.livekit.cloud)",
    "LIVEKIT_API_KEY": "LiveKit API key",
    "LIVEKIT_API_SECRET": "LiveKit API secret",
}


def _validate_startup_env() -> None:
    missing: list[str] = []
    for var, description in _REQUIRED_ENV_VARS.items():
        if not os.getenv(var, "").strip():
            missing.append(f"  {var} — {description}")
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

    if not os.getenv("SUPABASE_URL", "").strip():
        logger.warning("SUPABASE_URL not set — inquiry data will not persist")


_validate_startup_env()


# ── Rate limiting ────────────────────────────────────────────────────────

_public_rate: dict[str, list[float]] = defaultdict(list)
_PUBLIC_IP_LIMIT = 20


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _check_public_rate(request: Request) -> None:
    ip = _get_client_ip(request)
    now = time.time()
    window = [t for t in _public_rate[ip] if now - t < 60]
    if len(window) >= _PUBLIC_IP_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    window.append(now)
    _public_rate[ip] = window


# ── Auth helper ──────────────────────────────────────────────────────────

def _verify_api_key(request: Request) -> str:
    expected = os.getenv("GATEWAY_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="GATEWAY_API_KEY not configured")
    token = (
        request.headers.get("x-api-key", "")
        or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        or request.query_params.get("api_key", "")
    )
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token


# ── FastAPI app ──────────────────────────────────────────────────────────

app = FastAPI(
    title="EBH Academy AI",
    description="Voice AI advisor for EBH Academy — course info, enrollment, certifications",
    version="1.0.0",
)

# Static files
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ebh-academy-ai"}


# ── Demo pages ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Redirect to the Talk to Shakira page."""
    page = Path(__file__).parent / "demo" / "index.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>EBH Academy AI</h1><p>Demo page not found.</p>")


@app.get("/talk", response_class=HTMLResponse)
async def serve_talk():
    """Serve the Talk to Shakira web call page."""
    page = Path(__file__).parent / "demo" / "index.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Talk page not found")


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    page = Path(__file__).parent / "demo" / "dashboard.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Dashboard not found")


# ── Public LiveKit token endpoint ────────────────────────────────────────

@app.get("/api/public/livekit-token")
async def api_public_livekit_token(
    request: Request,
    _rate_limit: None = Depends(_check_public_rate),
):
    """Generate a LiveKit token for public web callers (no API key needed)."""
    import json as _json

    lang = (request.query_params.get("lang") or "en").strip().lower()
    if lang not in {"en", "ar", "hi", "ur"}:
        lang = "en"
    stack = (request.query_params.get("stack") or "cascaded").strip().lower()
    if stack not in {"cascaded", "ultravox", "realtime"}:
        stack = "cascaded"

    try:
        from livekit.api import AccessToken, VideoGrants, LiveKitAPI, CreateAgentDispatchRequest

        lk_url_internal = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
        lk_url_public = os.getenv("LIVEKIT_PUBLIC_URL", lk_url_internal)
        lk_key = os.getenv("LIVEKIT_API_KEY", "devkey")
        lk_secret = os.getenv("LIVEKIT_API_SECRET", "secret")
        lk_agent_name = os.getenv("LIVEKIT_AGENT_NAME", "academy-agent").strip() or "academy-agent"

        room = f"academy-web-{int(time.time() * 1000)}"
        identity = f"web-caller-{int(time.time() * 1000)}"

        token = (
            AccessToken(api_key=lk_key, api_secret=lk_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(VideoGrants(room_join=True, room=room))
            .to_jwt()
        )

        force_dispatch = os.getenv("LIVEKIT_FORCE_DISPATCH", "true").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if force_dispatch:
            try:
                dispatch_metadata = _json.dumps({"lang": lang, "stack": stack})
                async with LiveKitAPI(url=lk_url_internal, api_key=lk_key, api_secret=lk_secret) as lk_api:
                    await lk_api.agent_dispatch.create_dispatch(
                        CreateAgentDispatchRequest(
                            room=room,
                            agent_name=lk_agent_name,
                            metadata=dispatch_metadata,
                        )
                    )
                logger.info("Dispatched agent '%s' to room '%s' (lang=%s)", lk_agent_name, room, lang)
            except Exception as de:
                logger.warning("Agent dispatch failed: %s", de)

        return {"token": token, "url": lk_url_public, "room": room, "identity": identity, "lang": lang}

    except ImportError as ie:
        raise HTTPException(status_code=500, detail=f"LiveKit not configured: {ie}")
    except Exception:
        logger.exception("Token generation failed")
        raise HTTPException(status_code=500, detail="Failed to generate token")


# ── Inquiry API (authenticated) ──────────────────────────────────────────

@app.get("/api/inquiries")
async def list_inquiries(
    limit: int = 50,
    offset: int = 0,
    _token: str = Depends(_verify_api_key),
):
    """List academy inquiries (enrollment leads)."""
    from core.db import get_inquiries
    rows = await get_inquiries(limit=limit, offset=offset)
    return {"items": rows, "count": len(rows)}


@app.get("/api/call-logs")
async def list_call_logs(
    limit: int = 50,
    offset: int = 0,
    _token: str = Depends(_verify_api_key),
):
    """List call logs."""
    from core.db import get_call_logs
    rows = await get_call_logs(limit=limit, offset=offset)
    return {"items": rows, "count": len(rows)}


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
