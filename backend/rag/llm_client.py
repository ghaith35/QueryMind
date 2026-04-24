"""
Gemini LLM client.

Uses google.genai (new SDK — google.generativeai is deprecated).
Default model: gemini-2.5-flash-lite, overridable via LLM_MODEL.

Provides both:
    - generate(): blocking, returns full text
    - stream_generate(): async generator, yields text tokens for WebSocket streaming
"""

import os
import logging
from typing import AsyncGenerator, Optional

log = logging.getLogger(__name__)

_client = None


def _get_model_name() -> str:
    """Read the current model name from the environment at call time."""
    return os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")


def _get_client():
    global _client
    if _client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=api_key)
        log.info("Gemini client initialised (model=%s)", _get_model_name())
    return _client


def _make_config(
    system_prompt: str,
    temperature: float,
    genai_types,
) -> "genai_types.GenerateContentConfig":
    """
    Build the GenerateContentConfig.

    Gemini 2.5 Flash (and Flash Lite) support thinking_config which enables
    an internal chain-of-thought pass before the visible response. This
    significantly improves reasoning quality on multi-hop and synthesis questions.
    A budget of 8192 tokens is generous without adding noticeable latency.

    Falls back gracefully if the SDK version does not expose ThinkingConfig.
    """
    thinking_budget = int(os.environ.get("LLM_THINKING_BUDGET", "8192"))
    try:
        thinking_cfg = genai_types.ThinkingConfig(thinking_budget=thinking_budget)
        return genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            thinking_config=thinking_cfg,
        )
    except Exception:
        return genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        )


def generate_fast(
    user_message: str,
    system_prompt: str,
    temperature: float = 0.0,
) -> str:
    """
    Fast, no-thinking LLM call for internal pipeline tasks (HyDE generation,
    query expansion). Thinking is explicitly disabled (budget=0) to minimize
    latency. Not for user-facing answers.
    """
    from google.genai import types as genai_types

    client = _get_client()
    model_name = _get_model_name()
    try:
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        )
    except Exception:
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        )
    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config=config,
    )
    return response.text


def generate(
    user_message: str,
    system_prompt: str,
    temperature: float = 0.2,
) -> str:
    """
    Blocking LLM call. Returns full response text.
    temperature=0.2: enough randomness for natural prose while keeping
    citation format stable (citations are enforced structurally, not by temperature).
    """
    from google.genai import types as genai_types

    client = _get_client()
    model_name = _get_model_name()
    config = _make_config(system_prompt, temperature, genai_types)
    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config=config,
    )
    return response.text


async def stream_generate(
    user_message: str,
    system_prompt: str,
    temperature: float = 0.2,
) -> AsyncGenerator[str, None]:
    """
    Async generator — yields text tokens as they arrive from Gemini.
    Use in FastAPI WebSocket handlers to push answer_stream messages.
    """
    import asyncio
    from google.genai import types as genai_types

    client = _get_client()
    loop = asyncio.get_event_loop()
    model_name = _get_model_name()
    config = _make_config(system_prompt, temperature, genai_types)

    # google.genai streaming is sync; run in thread pool to not block the event loop
    def _sync_stream():
        return client.models.generate_content_stream(
            model=model_name,
            contents=user_message,
            config=config,
        )

    stream = await loop.run_in_executor(None, _sync_stream)
    for chunk in stream:
        if chunk.text:
            yield chunk.text
