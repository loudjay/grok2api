"""OpenAI Responses API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.auth import verify_api_key
from app.core.exceptions import ValidationException
from app.services.compat.responses_api import responses_create


router = APIRouter(tags=["Responses"])


class ResponsesReasoning(BaseModel):
    effort: str | None = None


class ResponsesCreateRequest(BaseModel):
    model: str = Field(...)
    input: str | list[Any] = Field(...)
    instructions: str | None = None
    stream: bool | None = None
    reasoning: ResponsesReasoning | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None

    model_config = {"extra": "ignore"}


@router.post("/responses")
async def create_response(
    request: ResponsesCreateRequest,
    _api_key: str | None = Depends(verify_api_key),
):
    if not request.input:
        raise ValidationException("input cannot be empty", param="input", code="empty_input")

    reasoning = request.reasoning
    if isinstance(reasoning, ResponsesReasoning):
        effort = reasoning.effort
    elif isinstance(reasoning, dict):
        effort = str(reasoning.get("effort") or "").strip() or None
    else:
        effort = None
    emit_think = effort != "none"

    result = await responses_create(
        model=request.model,
        input_value=request.input,
        instructions=request.instructions,
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
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


__all__ = ["router"]

