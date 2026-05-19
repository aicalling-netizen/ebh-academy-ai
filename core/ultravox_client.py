"""Thin async wrapper around the Ultravox REST API.

We only use the create-call endpoint:
  POST /api/agents/{agent_id}/calls

Auth: X-API-Key header. See https://docs.ultravox.ai/.
"""
from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger("academy.ultravox.client")

_DEFAULT_INWORLD_AR_VOICE = "WcxyRPjVQcpVYmceBQO4Helb"  # Aisha — reused from current Faseeh setup


@dataclass
class UltravoxConfig:
    api_key: str
    agent_id: str
    base_url: str = "https://api.ultravox.ai"
    max_duration: str = "1800s"
    tool_secret: str = ""  # passed to Ultravox so signed callbacks can be verified

    @classmethod
    def from_env(cls) -> "UltravoxConfig":
        return cls(
            api_key=os.getenv("ULTRAVOX_API_KEY", "").strip(),
            agent_id=os.getenv("ULTRAVOX_AGENT_ID", "").strip(),
            base_url=os.getenv("ULTRAVOX_BASE_URL", "https://api.ultravox.ai").strip(),
            max_duration=os.getenv("ULTRAVOX_MAX_DURATION", "1800s").strip(),
            tool_secret=os.getenv("ULTRAVOX_TOOL_SECRET", "").strip(),
        )


class UltravoxClient:
    """Async client for creating Ultravox agent calls."""

    def __init__(self, config: UltravoxConfig) -> None:
        if not config.api_key:
            raise ValueError("UltravoxConfig.api_key is required")
        if not config.agent_id:
            raise ValueError("UltravoxConfig.agent_id is required")
        self._config = config

    async def create_call(
        self,
        *,
        lang: str,
        source: str,
    ) -> dict:
        """Create an Ultravox call and return the response JSON.

        The returned dict contains at least: callId, joinUrl, created.
        Raises httpx.HTTPStatusError on non-2xx responses.
        """
        url = f"{self._config.base_url}/api/agents/{self._config.agent_id}/calls"
        body = self._build_request_body(lang=lang, source=source)
        headers = {"X-API-Key": self._config.api_key, "Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            _result = client.post(url, headers=headers, json=body, timeout=15.0)
            resp = await _result if inspect.isawaitable(_result) else _result
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "Ultravox call created: callId=%s lang=%s source=%s",
            data.get("callId"),
            lang,
            source,
        )
        return data

    def _build_request_body(self, *, lang: str, source: str) -> dict:
        lang_hint = "ar" if lang == "ar" else "en-US"
        body: dict = {
            "languageHint": lang_hint,
            "initialOutputMedium": "MESSAGE_MEDIUM_VOICE",
            "maxDuration": self._config.max_duration,
            "firstSpeakerSettings": {
                "agent": {
                    "text": (
                        "مرحباً، أنا شاكيرا من أكاديمية إي بي إتش. كيف يمكنني مساعدتك؟"
                        if lang == "ar"
                        else "Hi, this is Shakira from EBH Academy — how can I help?"
                    ),
                    "uninterruptible": False,
                }
            },
            "medium": {
                "webRtc": {
                    "dataMessages": {
                        "transcript": True,
                        "state": True,
                        "clientToolInvocation": True,
                    }
                }
            },
            "metadata": {"source": source},
        }
        if self._config.tool_secret:
            body["sharedSecrets"] = [self._config.tool_secret]
        if lang == "ar":
            voice_id = os.getenv("ULTRAVOX_AR_INWORLD_VOICE_ID", _DEFAULT_INWORLD_AR_VOICE)
            body["voiceOverrides"] = {"inworld": {"voiceId": voice_id}}
        return body
