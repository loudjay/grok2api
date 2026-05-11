"""OpenAI-compatible chat completion helpers."""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncGenerator

import orjson

from app.core.config import get_config
from app.services.compat.common import (
    ChatArtifacts,
    finalize_chat_request,
    iterate_chat_events,
    prepare_chat_request,
)
from app.services.compat.media import render_generated_image
from app.services.compat.tooling import ParsedToolCall, ToolSieve
from app.services.compat.usage import (
    estimate_prompt_tokens,
    estimate_tokens,
    estimate_tool_call_tokens,
)


def make_chat_response_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def build_chat_usage(prompt: str, *, text: str = "", thinking: str = "", tool_calls: list[Any] | None = None) -> dict:
    prompt_tokens = estimate_prompt_tokens(prompt)
    if tool_calls:
        completion_tokens = estimate_tool_call_tokens(tool_calls)
        reasoning_tokens = 0
    else:
        reasoning_tokens = estimate_tokens(thinking)
        completion_tokens = estimate_tokens(text) + reasoning_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "text_tokens": prompt_tokens,
            "audio_tokens": 0,
            "image_tokens": 0,
        },
        "completion_tokens_details": {
            "text_tokens": max(0, completion_tokens - reasoning_tokens),
            "audio_tokens": 0,
            "reasoning_tokens": reasoning_tokens,
        },
    }


def make_chat_response(model: str, artifacts: ChatArtifacts) -> dict:
    response_id = make_chat_response_id()
    usage = build_chat_usage(artifacts.prompt, text=artifacts.text, thinking=artifacts.thinking)
    message: dict[str, Any] = {"role": "assistant", "content": artifacts.text}
    if artifacts.thinking:
        message["reasoning_content"] = artifacts.thinking
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage,
    }


def make_tool_response(model: str, artifacts: ChatArtifacts) -> dict:
    response_id = make_chat_response_id()
    usage = build_chat_usage(artifacts.prompt, tool_calls=artifacts.tool_calls)
    tool_calls = [
        {
            "id": call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
        for call in artifacts.tool_calls
    ]
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": usage,
    }


def _chunk_payload(response_id: str, model: str, delta: dict[str, Any], finish_reason: str | None = None) -> str:
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


async def chat_completions(
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool | None = None,
    thinking: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> dict | AsyncGenerator[str, None]:
    emit_think = _resolve_emit_think(thinking)
    prepared = await prepare_chat_request(
        model=model,
        messages=messages,
        emit_think=emit_think,
        tools=tools,
        tool_choice=tool_choice,
    )
    if not stream:
        artifacts = await _collect_completion(prepared, emit_think=emit_think)
        return make_tool_response(model, artifacts) if artifacts.tool_calls else make_chat_response(model, artifacts)
    return _stream_completion(prepared, emit_think=emit_think)


async def _collect_completion(prepared, *, emit_think: bool) -> ChatArtifacts:
    from app.services.compat.common import collect_chat_artifacts

    success = False
    try:
        artifacts = await collect_chat_artifacts(prepared, emit_think=emit_think)
        success = bool(artifacts.tool_calls or artifacts.text)
        return artifacts
    finally:
        await finalize_chat_request(prepared, success=success)


async def _stream_completion(prepared, *, emit_think: bool) -> AsyncGenerator[str, None]:
    response_id = make_chat_response_id()
    sieve = ToolSieve(prepared.tool_names) if prepared.tool_names else None
    role_sent = False
    success = False
    tool_done = False
    try:
        async for event in iterate_chat_events(prepared.raw_stream):
            if not role_sent:
                role_sent = True
                yield _chunk_payload(response_id, prepared.model, {"role": "assistant", "content": ""})

            if event.kind == "thinking":
                if emit_think and event.content:
                    yield _chunk_payload(
                        response_id,
                        prepared.model,
                        {"role": "assistant", "reasoning_content": event.content},
                    )
                continue

            if event.kind == "text":
                if sieve:
                    safe_text, calls = sieve.feed(event.content)
                    if safe_text:
                        yield _chunk_payload(response_id, prepared.model, {"content": safe_text})
                    if calls is not None:
                        async for chunk in _stream_tool_calls(response_id, prepared.model, calls):
                            yield chunk
                        tool_done = True
                        success = True
                        break
                    continue
                yield _chunk_payload(response_id, prepared.model, {"content": event.content})
                success = True
                continue

            if event.kind == "image":
                markup = await render_generated_image(prepared.token, event.image_url)
                if markup:
                    yield _chunk_payload(response_id, prepared.model, {"content": f"{markup}\n"})
                    success = True
                continue

            if event.kind == "soft_stop":
                break

        if sieve and not tool_done:
            flushed = sieve.flush() or []
            if flushed:
                async for chunk in _stream_tool_calls(response_id, prepared.model, flushed):
                    yield chunk
                tool_done = True
                success = True

        if not tool_done:
            yield _chunk_payload(response_id, prepared.model, {}, "stop")
            yield "data: [DONE]\n\n"
    finally:
        await finalize_chat_request(prepared, success=success)


async def _stream_tool_calls(response_id: str, model: str, calls: list[ParsedToolCall]) -> AsyncGenerator[str, None]:
    for index, call in enumerate(calls):
        delta = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "index": index,
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
            ],
        }
        yield _chunk_payload(response_id, model, delta)
    yield _chunk_payload(response_id, model, {}, "tool_calls")
    yield "data: [DONE]\n\n"


def _resolve_emit_think(thinking: str | None) -> bool:
    if thinking == "enabled":
        return True
    if thinking == "disabled":
        return False
    return bool(get_config("grok.thinking", True))


__all__ = ["chat_completions"]
