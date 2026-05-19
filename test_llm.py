"""Quick smoke test: hit Anthropic with the exact same config the agent uses."""
import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).parent))

from livekit.plugins import anthropic as lk_anthropic
from livekit.agents import llm


async def main():
    prompt_file = Path(__file__).parent / "data" / "system_prompt.txt"
    sys_prompt = prompt_file.read_text(encoding="utf-8")
    sys_prompt = sys_prompt.replace("{ACADEMY_NAME}", os.getenv("ACADEMY_NAME", "EBH Academy"))
    print(f"System prompt: {len(sys_prompt)} chars")

    eng = lk_anthropic.LLM(
        model=os.getenv("ANTHROPIC_LLM_MODEL", "claude-haiku-4-5-20251001"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.3,
        max_tokens=400,
        caching="ephemeral",
    )

    ctx = llm.ChatContext()
    ctx.add_message(role="system", content=sys_prompt)
    ctx.add_message(role="user", content="Hello Shakira, how are you?")

    print("Calling Anthropic...")
    t0 = time.monotonic()
    stream = eng.chat(chat_ctx=ctx)
    out = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            out.append(chunk.delta.content)
    elapsed = time.monotonic() - t0
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Response: {''.join(out)}")


if __name__ == "__main__":
    asyncio.run(main())
