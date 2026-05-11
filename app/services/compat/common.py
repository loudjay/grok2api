"""Shared helpers for OpenAI/Responses/Anthropic compatibility layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.exceptions import AppException, ErrorType, UpstreamException, ValidationException
from app.core.logger import logger
from app.services.compat.media import render_generated_image
from app.services.compat.stream_adapter import GrokStreamAdapter, StreamEvent, classify_line
from app.services.compat.tooling import (
    ParsedToolCall,
    ToolSieve,
    build_tool_system_prompt,
    extract_tool_names,
    inject_into_message,
    tool_calls_to_xml,
)
from app.services.grok.chat import GrokChatService
from app.services.grok.model import ModelService
from app.services.grok.assets import UploadService
from app.services.request_stats import request_stats
from app.services.token import get_token_manager


@dataclass(slots=True)
class PreparedChatRequest:
    model: str
    token: str
    token_manager: Any
    prompt: str
    tool_names: list[str]
    raw_stream: Any


@dataclass(slots=True)
class ChatArtifacts:
    prompt: str
    text: str
    thinking: str
    tool_calls: list[ParsedToolCall]


def require_chat_model(model: str):
    model_info = ModelService.get(model)
    if not model_info:
        raise ValidationException(
            message=f"The model `{model}` does not exist or you do not have access to it.",
            param="model",
            code="model_not_found",
        )
    if model_info.is_image or model_info.is_video:
        raise ValidationException(
            message=f"The model `{model}` is not supported on this endpoint.",
            param="model",
            code="model_not_supported",
        )
    return model_info


async def prepare_chat_request(
    *,
    model: str,
    messages: list[dict[str, Any]],
    emit_think: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> PreparedChatRequest:
    model_info = require_chat_model(model)
    prompt, attachments = flatten_messages(messages)
    if not prompt.strip():
        raise ValidationException(
            message="Message content cannot be empty",
            param="messages",
            code="empty_content",
        )

    tool_names: list[str] = []
    if tools:
        tool_names = extract_tool_names(tools)
        tool_prompt = build_tool_system_prompt(tools, tool_choice)
        prompt = inject_into_message(prompt, tool_prompt)

    token_manager, token = await _get_runtime_token(model)
    file_ids = await _upload_attachments(token, attachments)
    service = GrokChatService()
    raw_stream = await service.chat(
        token=token,
        message=prompt,
        model=model_info.grok_model,
        mode=model_info.model_mode,
        think=emit_think,
        stream=True,
        file_attachments=file_ids,
    )
    return PreparedChatRequest(
        model=model,
        token=token,
        token_manager=token_manager,
        prompt=prompt,
        tool_names=tool_names,
        raw_stream=raw_stream,
    )


async def collect_chat_artifacts(prepared: PreparedChatRequest, *, emit_think: bool) -> ChatArtifacts:
    adapter = GrokStreamAdapter()
    sieve = ToolSieve(prepared.tool_names) if prepared.tool_names else None
    text_parts: list[str] = []
    tool_calls: list[ParsedToolCall] = []
    async for raw_line in prepared.raw_stream:
        event_type, data = classify_line(raw_line)
        if event_type == "done":
            break
        if event_type != "data" or not data:
            continue
        for event in adapter.feed(data):
            if event.kind == "text":
                if sieve:
                    safe_text, calls = sieve.feed(event.content)
                    if safe_text:
                        text_parts.append(safe_text)
                    if calls is not None:
                        tool_calls = calls
                        break
                    continue
                text_parts.append(event.content)
                continue
            if event.kind == "image":
                text_parts.append(await render_generated_image(prepared.token, event.image_url))
                continue
            if event.kind == "soft_stop":
                break
        if tool_calls:
            break

    if sieve and not tool_calls:
        tool_calls = sieve.flush() or []

    if not tool_calls and not text_parts and adapter.final_text:
        text_parts.append(adapter.final_text)
    thinking = adapter.final_thinking if emit_think and not tool_calls else ""

    return ChatArtifacts(
        prompt=prepared.prompt,
        text="".join(part for part in text_parts if part).strip(),
        thinking=thinking.strip(),
        tool_calls=tool_calls,
    )


async def iterate_chat_events(raw_stream) -> Any:
    adapter = GrokStreamAdapter()
    async for raw_line in raw_stream:
        event_type, data = classify_line(raw_line)
        if event_type == "done":
            yield StreamEvent("soft_stop")
            return
        if event_type != "data" or not data:
            continue
        for event in adapter.feed(data):
            yield event


def flatten_messages(messages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    parts: list[str] = []
    attachments: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip() or "user"
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if role == "tool":
            tool_result = _coerce_tool_result(message)
            if tool_result:
                parts.append(tool_result)
            continue
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            xml = tool_calls_to_xml(tool_calls)
            text = content.strip() if isinstance(content, str) else ""
            parts.append(f"[assistant]: {text}\n{xml}".strip())
            continue
        extracted_text, extracted_attachments = _extract_content(role, content)
        attachments.extend(extracted_attachments)
        if extracted_text:
            parts.append(extracted_text)
    return "\n\n".join(part for part in parts if part).strip(), attachments


async def finalize_chat_request(prepared: PreparedChatRequest, *, success: bool) -> None:
    try:
        if success:
            await prepared.token_manager.sync_usage(
                prepared.token,
                prepared.model,
                consume_on_fail=True,
                is_usage=True,
            )
        await request_stats.record_request(prepared.model, success=success)
    except Exception as exc:
        logger.warning("Compatibility request finalization failed: {}", exc)


def _extract_content(role: str, content: Any) -> tuple[str, list[str]]:
    parts: list[str] = []
    attachments: list[str] = []
    if isinstance(content, str):
        value = content.strip()
        if value:
            return f"[{role}]: {value}", []
        return "", []
    if not isinstance(content, list):
        return "", []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "text":
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
        elif item_type == "image_url":
            url = item.get("image_url")
            if isinstance(url, dict):
                value = str(url.get("url") or "").strip()
            else:
                value = str(url or "").strip()
            if value:
                attachments.append(value)
        elif item_type in {"input_audio", "file"}:
            inner = item.get(item_type)
            if isinstance(inner, dict):
                value = str(inner.get("data") or inner.get("file_data") or inner.get("url") or "").strip()
            else:
                value = str(inner or "").strip()
            if value:
                attachments.append(value)
    joined = "\n".join(part for part in parts if part).strip()
    if not joined:
        return "", attachments
    return f"[{role}]: {joined}", attachments


def _coerce_tool_result(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    else:
        text, _ = _extract_content("tool", content)
        text = text.removeprefix("[tool]: ").strip()
    if not text:
        return ""
    call_id = str(message.get("tool_call_id") or "").strip()
    if call_id:
        return f"[tool result for {call_id}]:\n{text}"
    return f"[tool result]:\n{text}"


async def _get_runtime_token(model: str) -> tuple[Any, str]:
    try:
        token_manager = await get_token_manager()
        await token_manager.reload_if_stale()
        token = token_manager.get_token_for_model(model)
    except Exception as exc:
        logger.error("Failed to get token for {}: {}", model, exc)
        raise AppException(
            message="Internal service error obtaining token",
            error_type=ErrorType.SERVER.value,
            code="internal_error",
        ) from exc

    if token:
        return token_manager, token
    raise AppException(
        message="No available tokens. Please try again later.",
        error_type=ErrorType.RATE_LIMIT.value,
        code="rate_limit_exceeded",
        status_code=429,
    )


async def _upload_attachments(token: str, attachments: list[str]) -> list[str]:
    if not attachments:
        return []
    upload_service = UploadService()
    file_ids: list[str] = []
    try:
        for attachment in attachments:
            file_id, _ = await upload_service.upload(attachment, token)
            if file_id:
                file_ids.append(file_id)
    except ValidationException:
        raise
    except Exception as exc:
        raise UpstreamException(
            message=f"Attachment upload failed: {exc}",
            details={"error": str(exc)},
        ) from exc
    finally:
        await upload_service.close()
    return file_ids


__all__ = [
    "ChatArtifacts",
    "PreparedChatRequest",
    "collect_chat_artifacts",
    "finalize_chat_request",
    "flatten_messages",
    "iterate_chat_events",
    "prepare_chat_request",
    "require_chat_model",
]
