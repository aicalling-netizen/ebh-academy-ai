"""Direct WebSocket STT adapter for Munsit, working around the upstream plugin bug.

Upstream `livekit-plugins-munsit` 0.3.0 has `mode="streaming"` broken: it sends
raw PCM after the first chunk, but Munsit's server validates every message as a
WAV file and rejects continuation chunks with
`Not a valid WAV file (missing RIFF header)`. See:
  https://github.com/CNTXTFZCO0/livekit-plugins-munsit/issues/1

Workaround: buffer ~500 ms of audio, then send each window as a complete WAV
(44-byte header + window PCM). Every chunk is a self-contained valid WAV.
Munsit accumulates the audio server-side and returns cumulative transcripts.

This delivers true streaming-style behaviour (interim transcripts every ~500 ms)
without the ~0.5–1 s utterance-end latency of `mode="batch"`.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)

from livekit.plugins.munsit._utils import build_wav_header

_DEFAULT_WS_URL = "wss://api.munsit.com/api/v1/websocket/speech-to-text"
_DEFAULT_SAMPLE_RATE = 16000
_DEFAULT_FLUSH_INTERVAL_MS = 500


@dataclass
class _Options:
    api_key: str
    model: str
    language: str
    sample_rate: int
    num_channels: int
    flush_interval_ms: int
    base_url: str


class MunsitStreamingSTT(stt.STT):
    """Streaming Munsit STT via direct WebSocket (bypasses broken plugin path)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "munsit-en-ar",
        language: str = "ar",
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        num_channels: int = 1,
        flush_interval_ms: int = _DEFAULT_FLUSH_INTERVAL_MS,
        base_url: str = _DEFAULT_WS_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True)
        )
        if not api_key:
            raise ValueError("api_key is required")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if flush_interval_ms < 100:
            raise ValueError("flush_interval_ms must be >= 100")
        self._opts = _Options(
            api_key=api_key,
            model=model,
            language=language,
            sample_rate=sample_rate,
            num_channels=num_channels,
            flush_interval_ms=flush_interval_ms,
            base_url=base_url,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _recognize_impl(self, *args: Any, **kwargs: Any) -> stt.SpeechEvent:
        # One-shot batch recognition is not the use case for this adapter.
        raise NotImplementedError("MunsitStreamingSTT is streaming-only")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "MunsitStreamingSpeechStream":
        opts = self._opts
        if language:
            opts = _Options(
                api_key=opts.api_key,
                model=opts.model,
                language=language,
                sample_rate=opts.sample_rate,
                num_channels=opts.num_channels,
                flush_interval_ms=opts.flush_interval_ms,
                base_url=opts.base_url,
            )
        return MunsitStreamingSpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            http_session=self._ensure_session(),
        )


class MunsitStreamingSpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: MunsitStreamingSTT,
        opts: _Options,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = http_session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._request_id: str = ""
        self._buffer = bytearray()
        self._last_flush = time.monotonic()
        self._last_cumulative: str = ""
        self._utterance_active: bool = False

    def _build_url(self) -> str:
        params = {
            "x-api-key": self._opts.api_key,
            "model": self._opts.model,
        }
        if self._opts.language:
            params["language"] = self._opts.language
        return f"{self._opts.base_url}?{urlencode(params)}"

    async def _run(self) -> None:
        url = self._build_url()
        try:
            async with self._session.ws_connect(
                url,
                timeout=aiohttp.ClientWSTimeout(ws_receive=60.0, ws_close=10),
            ) as ws:
                self._ws = ws
                self._request_id = utils.shortuuid()
                self._buffer.clear()
                self._last_flush = time.monotonic()
                self._last_cumulative = ""
                self._utterance_active = False

                send_task = asyncio.create_task(self._send_loop())
                recv_task = asyncio.create_task(self._recv_loop(ws))
                try:
                    done, pending = await asyncio.wait(
                        [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        if not task.cancelled():
                            task.result()
                except APIStatusError:
                    raise
                finally:
                    self._ws = None
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Munsit WebSocket connection error: {e}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError(f"Munsit WebSocket timeout: {e}") from e

    async def _send_loop(self) -> None:
        ws = self._ws
        assert ws is not None
        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                self._buffer.extend(bytes(data.data))
                elapsed_ms = (time.monotonic() - self._last_flush) * 1000
                if elapsed_ms >= self._opts.flush_interval_ms:
                    await self._flush_chunk()
            else:  # _FlushSentinel — utterance finalize requested
                if self._buffer:
                    await self._flush_chunk()
                # Brief wait for the server to send the last transcript, then finalize.
                await asyncio.sleep(0.25)
                self._emit_final()

    async def _flush_chunk(self) -> None:
        ws = self._ws
        if ws is None or ws.closed or not self._buffer:
            return
        header = build_wav_header(
            sample_rate=self._opts.sample_rate, num_channels=self._opts.num_channels
        )
        payload = header + bytes(self._buffer)
        self._buffer.clear()
        self._last_flush = time.monotonic()
        msg = {"event": "audio_chunk", "data": {"audioBuffer": list(payload)}}
        try:
            await ws.send_str(json.dumps(msg))
        except ConnectionResetError as e:
            raise APIConnectionError(f"Munsit WebSocket closed unexpectedly: {e}") from e

    def _emit_interim(self, text: str) -> None:
        self._utterance_active = True
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[
                    stt.SpeechData(language=self._opts.language, text=text)
                ],
            )
        )

    def _emit_final(self) -> None:
        if not self._utterance_active or not self._last_cumulative:
            self._last_cumulative = ""
            return
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[
                    stt.SpeechData(
                        language=self._opts.language, text=self._last_cumulative
                    )
                ],
            )
        )
        self._utterance_active = False
        self._last_cumulative = ""

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                event = data.get("event") or data.get("type")
                if event == "transcription":
                    payload = data.get("data", "")
                    text = payload if isinstance(payload, str) else ""
                    text = text.strip()
                    if text and text != self._last_cumulative:
                        self._last_cumulative = text
                        self._emit_interim(text)
                elif event == "transcription_error":
                    err = data.get("data") or data.get("message", "unknown error")
                    raise APIStatusError(
                        message=f"Munsit transcription_error: {err}",
                        status_code=500,
                        request_id=self._request_id,
                        body=None,
                    )
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
