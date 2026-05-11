"""Anthropic-compatible messages routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.auth import verify_api_key
from app.core.exceptions import ValidationException
from app.services.compat.anthropic_api import messages_create


router = APIRouter(tags=["Messages"])


class AnthropicMessage(BaseModel):
    role: str = Field(...)
    content: Any = Field(default="")

    model_config = {"extra": "allow"}


class MessagesCreateRequest(BaseModel):
    model: str = Field(...)
    messages: list[AnthropicMessage] = Field(...)
    system: str | list[Any] | None = None
    stream: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    thinking: dict[str, Any] | None = None

    model_config = {"extra": "ignore"}


@router.post("/messages")
async def create_message(
    request: MessagesCreateRequest,
    _api_key: str | None = Depends(verify_api_key),
):
    if not request.messages:
        raise ValidationException("messages cannot be empty", param="messages", code="empty_messages")

    thinking_cfg = request.thinking or {}
    emit_think = str(thinking_cfg.get("type") or "").strip().lower() != "disabled"
    result = await messages_create(
        model=request.model,
        messages=[item.model_dump(exclude_none=True) for item in request.messages],
        system=request.system,
        stream=bool(request.stream),
        emit_think=emit_think,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )
    if isinstance(result, dict):
        return JSONResponse(result)
    return StreamingResponse(
        result,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


__all__ = ["router"]
