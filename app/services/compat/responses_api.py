"""OpenAI Responses API compatibility helpers."""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncGenerator

import orjson

from app.services.compat.common import ChatArtifacts, finalize_chat_request, iterate_chat_events, prepare_chat_request
from app.services.compat.media import render_generated_image
from app.services.compat.tooling import ToolSieve
from app.services.compat.usage import estimate_prompt_tokens, estimate_tokens, estimate_tool_call_tokens


def make_response_id(prefix: str = "resp") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def normalize_response_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools or []:
        if tool.get("type") == "function" and "function" not in tool and "name" in tool:
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters"),
                    },
                }
            )
        else:
            normalized.append(tool)
    return normalized


def parse_responses_input(input_value: str | list[Any], instructions: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages
    for item in input_value:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "message" if "role" in item else None)
        if item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
            continue
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                }
            )
            continue
        if item_type != "message":
            continue
        messages.append({"role": item.get("role", "user"), "content": _normalize_response_content(item.get("content"))})
    return messages


def make_response_usage(prompt: str, artifacts: ChatArtifacts) -> dict:
    prompt_tokens = estimate_prompt_tokens(prompt)
    if artifacts.tool_calls:
        output_tokens = estimate_tool_call_tokens(artifacts.tool_calls)
        reasoning_tokens = 0
    else:
        reasoning_tokens = estimate_tokens(artifacts.thinking)
        output_tokens = estimate_tokens(artifacts.text) + reasoning_tokens
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


def make_response_object(model: str, response_id: str, artifacts: ChatArtifacts) -> dict:
    output: list[dict[str, Any]] = []
    if artifacts.thinking:
        output.append(
            {
                "id": make_response_id("rs"),
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": artifacts.thinking}],
                "status": "completed",
            }
        )
    if artifacts.tool_calls:
        output.extend(
            {
                "id": make_response_id("fc"),
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
                "status": "completed",
            }
            for call in artifacts.tool_calls
        )
    else:
        output.append(
            {
                "id": make_response_id("msg"),
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": artifacts.text, "annotations": []}],
                "status": "completed",
            }
        )
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": make_response_usage(artifacts.prompt, artifacts),
    }


async def responses_create(
    *,
    model: str,
    input_value: str | list[Any],
    instructions: str | None,
    stream: bool,
    emit_think: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> dict | AsyncGenerator[str, None]:
    messages = parse_responses_input(input_value, instructions)
    prepared = await prepare_chat_request(
        model=model,
        messages=messages,
        emit_think=emit_think,
        tools=normalize_response_tools(tools),
        tool_choice=tool_choice,
    )
    if not stream:
        artifacts = await _collect_response(prepared, emit_think=emit_think)
        return make_response_object(model, make_response_id(), artifacts)
    return _stream_response(prepared, emit_think=emit_think)


async def _collect_response(prepared, *, emit_think: bool) -> ChatArtifacts:
    from app.services.compat.common import collect_chat_artifacts

    success = False
    try:
        artifacts = await collect_chat_artifacts(prepared, emit_think=emit_think)
        success = bool(artifacts.tool_calls or artifacts.text)
        return artifacts
    finally:
        await finalize_chat_request(prepared, success=success)


async def _stream_response(prepared, *, emit_think: bool) -> AsyncGenerator[str, None]:
    response_id = make_response_id()
    reasoning_id = make_response_id("rs")
    message_id = make_response_id("msg")
    sieve = ToolSieve(prepared.tool_names) if prepared.tool_names else None
    output_text = ""
    thinking_text = ""
    tool_items: list[dict[str, Any]] = []
    success = False
    try:
        yield _sse("response.created", {"type": "response.created", "response": {"id": response_id, "object": "response", "created_at": int(time.time()), "status": "in_progress", "model": prepared.model, "output": []}})
        async for event in iterate_chat_events(prepared.raw_stream):
            if event.kind == "thinking":
                if emit_think and event.content:
                    thinking_text += event.content
                    yield _sse("response.reasoning_summary_text.delta", {"type": "response.reasoning_summary_text.delta", "item_id": reasoning_id, "output_index": 0, "summary_index": 0, "delta": event.content})
                continue
            if event.kind == "text":
                if sieve:
                    safe_text, calls = sieve.feed(event.content)
                    if safe_text:
                        output_text += safe_text
                        yield _sse("response.output_text.delta", {"type": "response.output_text.delta", "item_id": message_id, "output_index": 1 if thinking_text else 0, "content_index": 0, "delta": safe_text})
                    if calls is not None:
                        tool_items = _build_response_tool_items(calls)
                        break
                    continue
                output_text += event.content
                yield _sse("response.output_text.delta", {"type": "response.output_text.delta", "item_id": message_id, "output_index": 1 if thinking_text else 0, "content_index": 0, "delta": event.content})
                continue
            if event.kind == "image":
                markup = await render_generated_image(prepared.token, event.image_url)
                if markup:
                    output_text += f"{markup}\n"
                    yield _sse("response.output_text.delta", {"type": "response.output_text.delta", "item_id": message_id, "output_index": 1 if thinking_text else 0, "content_index": 0, "delta": f"{markup}\n"})
                continue
            if event.kind == "soft_stop":
                break

        if sieve and not tool_items:
            tool_items = _build_response_tool_items(sieve.flush() or [])

        artifacts = ChatArtifacts(prepared.prompt, output_text.strip(), thinking_text.strip(), [item["_parsed"] for item in tool_items])
        payload = make_response_object(prepared.model, response_id, artifacts)
        if tool_items:
            for index, item in enumerate(tool_items):
                yield _sse("response.output_item.done", {"type": "response.output_item.done", "output_index": index + (1 if thinking_text else 0), "item": {k: v for k, v in item.items() if k != "_parsed"}})
        yield _sse("response.completed", {"type": "response.completed", "response": payload})
        yield "data: [DONE]\n\n"
        success = bool(tool_items or output_text.strip())
    finally:
        await finalize_chat_request(prepared, success=success)


def _normalize_response_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    normalized: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "").strip()
        if part_type in {"input_text", "output_text"}:
            normalized.append({"type": "text", "text": part.get("text", "")})
            continue
        if part_type in {"image", "input_image"}:
            source = part.get("image_url") or part.get("source") or {}
            if isinstance(source, dict):
                url = str(source.get("url") or "").strip()
            else:
                url = str(source or "").strip()
            if url:
                normalized.append({"type": "image_url", "image_url": {"url": url}})
    return normalized


def _build_response_tool_items(calls) -> list[dict[str, Any]]:
    return [
        {
            "_parsed": call,
            "id": make_response_id("fc"),
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
            "status": "completed",
        }
        for call in calls
    ]


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"


__all__ = ["responses_create"]
