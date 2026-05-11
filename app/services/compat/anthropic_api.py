"""Anthropic Messages API compatibility helpers."""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncGenerator

import orjson

from app.services.compat.common import ChatArtifacts, finalize_chat_request, iterate_chat_events, prepare_chat_request
from app.services.compat.media import render_generated_image
from app.services.compat.tooling import ToolSieve
from app.services.compat.usage import estimate_prompt_tokens, estimate_tokens, estimate_tool_call_tokens


def make_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def parse_anthropic_messages(messages: list[dict[str, Any]], system: str | list[Any] | None) -> list[dict[str, Any]]:
    internal: list[dict[str, Any]] = []
    system_text = _coerce_system(system)
    if system_text:
        internal.append({"role": "system", "content": system_text})
    for message in messages:
        role = str(message.get("role") or "user").strip() or "user"
        internal.extend(_normalize_anthropic_content(role, message.get("content")))
    return internal


def normalize_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema"),
            },
        }
        for tool in tools or []
    ]


def normalize_anthropic_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return "auto"
    choice_type = str(tool_choice.get("type") or "auto").strip()
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return choice_type or "auto"


def make_messages_response(model: str, artifacts: ChatArtifacts) -> dict:
    input_tokens = estimate_prompt_tokens(artifacts.prompt)
    if artifacts.tool_calls:
        content = [
            {
                "type": "tool_use",
                "id": call.call_id,
                "name": call.name,
                "input": _safe_json(call.arguments),
            }
            for call in artifacts.tool_calls
        ]
        output_tokens = estimate_tool_call_tokens(artifacts.tool_calls)
        stop_reason = "tool_use"
    else:
        content = [{"type": "text", "text": artifacts.text}]
        output_tokens = estimate_tokens(artifacts.text) + estimate_tokens(artifacts.thinking)
        stop_reason = "end_turn"
    return {
        "id": make_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


async def messages_create(
    *,
    model: str,
    messages: list[dict[str, Any]],
    system: str | list[Any] | None,
    stream: bool,
    emit_think: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> dict | AsyncGenerator[str, None]:
    prepared = await prepare_chat_request(
        model=model,
        messages=parse_anthropic_messages(messages, system),
        emit_think=emit_think,
        tools=normalize_anthropic_tools(tools),
        tool_choice=normalize_anthropic_tool_choice(tool_choice),
    )
    if not stream:
        artifacts = await _collect_message(prepared, emit_think=emit_think)
        return make_messages_response(model, artifacts)
    return _stream_message(prepared, emit_think=emit_think)


async def _collect_message(prepared, *, emit_think: bool) -> ChatArtifacts:
    from app.services.compat.common import collect_chat_artifacts

    success = False
    try:
        artifacts = await collect_chat_artifacts(prepared, emit_think=emit_think)
        success = bool(artifacts.tool_calls or artifacts.text)
        return artifacts
    finally:
        await finalize_chat_request(prepared, success=success)


async def _stream_message(prepared, *, emit_think: bool) -> AsyncGenerator[str, None]:
    message_id = make_message_id()
    sieve = ToolSieve(prepared.tool_names) if prepared.tool_names else None
    text_index = 0
    output_text = ""
    thinking_text = ""
    tool_calls = []
    success = False
    try:
        yield _anthropic_sse("message_start", {"type": "message_start", "message": {"id": message_id, "type": "message", "role": "assistant", "model": prepared.model, "content": [], "stop_reason": None, "usage": {"input_tokens": estimate_prompt_tokens(prepared.prompt), "output_tokens": 0}}})
        async for event in iterate_chat_events(prepared.raw_stream):
            if event.kind == "thinking":
                if emit_think and event.content:
                    thinking_text += event.content
                    yield _anthropic_sse("content_block_delta", {"type": "content_block_delta", "index": text_index, "delta": {"type": "thinking_delta", "thinking": event.content}})
                continue
            if event.kind == "text":
                if sieve:
                    safe_text, calls = sieve.feed(event.content)
                    if safe_text:
                        output_text += safe_text
                        yield _anthropic_sse("content_block_delta", {"type": "content_block_delta", "index": text_index, "delta": {"type": "text_delta", "text": safe_text}})
                    if calls is not None:
                        tool_calls = calls
                        break
                    continue
                output_text += event.content
                yield _anthropic_sse("content_block_delta", {"type": "content_block_delta", "index": text_index, "delta": {"type": "text_delta", "text": event.content}})
                continue
            if event.kind == "image":
                markup = await render_generated_image(prepared.token, event.image_url)
                if markup:
                    output_text += f"{markup}\n"
                    yield _anthropic_sse("content_block_delta", {"type": "content_block_delta", "index": text_index, "delta": {"type": "text_delta", "text": f"{markup}\n"}})
                continue
            if event.kind == "soft_stop":
                break

        if sieve and not tool_calls:
            tool_calls = sieve.flush() or []

        stop_reason = "tool_use" if tool_calls else "end_turn"
        output_tokens = estimate_tool_call_tokens(tool_calls) if tool_calls else estimate_tokens(output_text) + estimate_tokens(thinking_text)
        yield _anthropic_sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": output_tokens}})
        yield _anthropic_sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"
        success = bool(tool_calls or output_text.strip())
    finally:
        await finalize_chat_request(prepared, success=success)


def _normalize_anthropic_content(role: str, content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return []
    tool_results = [item for item in content if isinstance(item, dict) and item.get("type") == "tool_result"]
    if tool_results:
        return [
            {
                "role": "tool",
                "tool_call_id": item.get("tool_use_id", ""),
                "content": _tool_result_text(item.get("content")),
            }
            for item in tool_results
        ]
    text_parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "text":
            text_parts.append({"type": "text", "text": item.get("text", "")})
        elif item_type == "tool_use":
            tool_calls.append(
                {
                    "id": item.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": orjson.dumps(item.get("input") or {}).decode(),
                    },
                }
            )
        elif item_type == "image":
            source = item.get("source") or {}
            if isinstance(source, dict) and source.get("type") == "base64":
                url = f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"
                text_parts.append({"type": "image_url", "image_url": {"url": url}})
    if tool_calls:
        return [{"role": "assistant", "content": None, "tool_calls": tool_calls}]
    if text_parts:
        return [{"role": role, "content": text_parts}]
    return []


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = [str(item.get("text") or "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
    return "\n".join(texts)


def _coerce_system(system: str | list[Any] | None) -> str:
    if isinstance(system, str):
        return system.strip()
    if not isinstance(system, list):
        return ""
    texts = [str(item.get("text") or "") for item in system if isinstance(item, dict) and item.get("type") == "text"]
    return "\n".join(texts).strip()


def _safe_json(raw: str) -> Any:
    try:
        return orjson.loads(raw)
    except Exception:
        return {}


def _anthropic_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"


__all__ = ["messages_create"]
