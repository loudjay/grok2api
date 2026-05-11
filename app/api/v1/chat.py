"""OpenAI-compatible chat completions routes."""

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.core.auth import verify_api_key
from app.core.exceptions import ValidationException
from app.services.compat.openai_chat import chat_completions as compat_chat_completions
from app.services.grok.model import ModelService
from app.services.quota import enforce_daily_quota


router = APIRouter(tags=["Chat"])

VALID_ROLES = ["developer", "system", "user", "assistant", "tool"]
USER_CONTENT_TYPES = ["text", "image_url", "input_audio", "file"]


class MessageItem(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}")
        return value


class VideoConfig(BaseModel):
    aspect_ratio: Optional[str] = Field("3:2", description="Video aspect ratio")
    video_length: Optional[int] = Field(6, description="Video length in seconds")
    resolution: Optional[str] = Field("SD", description="Video resolution")
    preset: Optional[str] = Field("custom", description="Video preset")

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, value: Optional[str]) -> Optional[str]:
        allowed = ["2:3", "3:2", "1:1", "9:16", "16:9"]
        if value and value not in allowed:
            raise ValidationException(
                message=f"aspect_ratio must be one of {allowed}",
                param="video_config.aspect_ratio",
                code="invalid_aspect_ratio",
            )
        return value

    @field_validator("video_length")
    @classmethod
    def validate_video_length(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and (value < 5 or value > 15):
            raise ValidationException(
                message="video_length must be between 5 and 15 seconds",
                param="video_config.video_length",
                code="invalid_video_length",
            )
        return value

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: Optional[str]) -> Optional[str]:
        allowed = ["SD", "HD"]
        if value and value not in allowed:
            raise ValidationException(
                message=f"resolution must be one of {allowed}",
                param="video_config.resolution",
                code="invalid_resolution",
            )
        return value

    @field_validator("preset")
    @classmethod
    def validate_preset(cls, value: Optional[str]) -> str:
        if not value:
            return "custom"
        allowed = ["fun", "normal", "spicy", "custom"]
        if value not in allowed:
            raise ValidationException(
                message=f"preset must be one of {allowed}",
                param="video_config.preset",
                code="invalid_preset",
            )
        return value


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model name")
    messages: List[MessageItem] = Field(..., description="Message list")
    stream: Optional[bool] = Field(None, description="Whether to stream")
    thinking: Optional[str] = Field(None, description="Thinking mode")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="OpenAI tool definitions")
    tool_choice: Optional[Any] = Field(None, description="Tool selection mode")
    video_config: Optional[VideoConfig] = Field(None, description="Video generation config")

    model_config = {"extra": "ignore"}


def validate_request(request: ChatCompletionRequest) -> None:
    if not ModelService.valid(request.model):
        raise ValidationException(
            message=f"The model `{request.model}` does not exist or you do not have access to it.",
            param="model",
            code="model_not_found",
        )

    for idx, message in enumerate(request.messages):
        if message.role == "tool" and not str(message.tool_call_id or "").strip():
            raise ValidationException(
                message="The `tool` role requires `tool_call_id`",
                param=f"messages.{idx}.tool_call_id",
                code="missing_tool_call_id",
            )

        content = message.content
        if isinstance(content, str):
            if not content.strip():
                raise ValidationException(
                    message="Message content cannot be empty",
                    param=f"messages.{idx}.content",
                    code="empty_content",
                )
            continue

        if isinstance(content, list):
            if not content:
                raise ValidationException(
                    message="Message content cannot be an empty array",
                    param=f"messages.{idx}.content",
                    code="empty_content",
                )

            for block_idx, block in enumerate(content):
                if not isinstance(block, dict) or not block:
                    raise ValidationException(
                        message="Content block cannot be empty",
                        param=f"messages.{idx}.content.{block_idx}",
                        code="empty_block",
                    )
                block_type = block.get("type")
                if not isinstance(block_type, str) or not block_type.strip():
                    raise ValidationException(
                        message="Content block must have a non-empty `type`",
                        param=f"messages.{idx}.content.{block_idx}.type",
                        code="missing_type",
                    )

                if message.role == "user":
                    if block_type not in USER_CONTENT_TYPES:
                        raise ValidationException(
                            message=f"Invalid content block type: '{block_type}'",
                            param=f"messages.{idx}.content.{block_idx}.type",
                            code="invalid_type",
                        )
                elif message.role == "tool" or block_type != "text":
                    raise ValidationException(
                        message=f"The `{message.role}` role only supports 'text' type, got '{block_type}'",
                        param=f"messages.{idx}.content.{block_idx}.type",
                        code="invalid_type",
                    )

                if block_type == "text":
                    text = block.get("text", "")
                    if not isinstance(text, str) or not text.strip():
                        raise ValidationException(
                            message="Text content cannot be empty",
                            param=f"messages.{idx}.content.{block_idx}.text",
                            code="empty_text",
                        )
                    continue

                if block_type == "image_url":
                    image_url = block.get("image_url")
                    if not isinstance(image_url, dict) or not str(image_url.get("url") or "").strip():
                        raise ValidationException(
                            message="image_url must have a `url` field",
                            param=f"messages.{idx}.content.{block_idx}.image_url",
                            code="missing_url",
                        )
                    continue

                if block_type in {"input_audio", "file"} and not block.get(block_type):
                    raise ValidationException(
                        message=f"{block_type} content cannot be empty",
                        param=f"messages.{idx}.content.{block_idx}.{block_type}",
                        code="empty_content",
                    )
            continue

        if message.role != "assistant":
            raise ValidationException(
                message="Message content cannot be empty",
                param=f"messages.{idx}.content",
                code="empty_content",
            )


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    validate_request(request)
    await enforce_daily_quota(api_key, request.model)

    model_info = ModelService.get(request.model)
    if model_info and model_info.is_video:
        from app.services.grok.media import VideoService

        video_config = request.video_config or VideoConfig()
        result = await VideoService.completions(
            model=request.model,
            messages=[message.model_dump() for message in request.messages],
            stream=request.stream,
            thinking=request.thinking,
            aspect_ratio=video_config.aspect_ratio,
            video_length=video_config.video_length,
            resolution=video_config.resolution,
            preset=video_config.preset,
        )
    else:
        result = await compat_chat_completions(
            model=request.model,
            messages=[message.model_dump() for message in request.messages],
            stream=request.stream,
            thinking=request.thinking,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )

    if isinstance(result, dict):
        return JSONResponse(content=result)
    return StreamingResponse(
        result,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


__all__ = ["router"]
