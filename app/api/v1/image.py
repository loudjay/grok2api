"""
Image Generation API 路由
"""

import asyncio
import base64
import random
from pathlib import Path
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from app.core.auth import verify_api_key
from app.core.config import get_config
from app.core.exceptions import AppException, ErrorType, ValidationException
from app.core.logger import logger
from app.services.grok.assets import UploadService
from app.services.grok.chat import GrokChatService
from app.services.grok.model import ModelService
from app.services.grok.processor import ImageCollectProcessor, ImageStreamProcessor
from app.services.quota import enforce_daily_quota
from app.services.request_stats import request_stats
from app.services.token import get_token_manager


router = APIRouter(tags=["Images"])
ALLOWED_RESPONSE_FORMATS = {"b64_json", "base64", "url"}


class ImageGenerationRequest(BaseModel):
    """图片生成请求 - OpenAI 兼容"""

    prompt: str = Field(..., description="图片描述")
    model: Optional[str] = Field("grok-imagine-1.0", description="模型名称")
    n: Optional[int] = Field(1, ge=1, le=10, description="生成数量 (1-10)")
    size: Optional[str] = Field("1024x1024", description="图片尺寸 (暂不支持)")
    quality: Optional[str] = Field("standard", description="图片质量 (暂不支持)")
    response_format: Optional[str] = Field(None, description="响应格式")
    style: Optional[str] = Field(None, description="风格 (暂不支持)")
    stream: Optional[bool] = Field(False, description="是否流式输出")


class ImageEditRequest(BaseModel):
    """图片编辑请求 - OpenAI 兼容"""

    prompt: str = Field(..., description="编辑描述")
    model: Optional[str] = Field("grok-imagine-1.0", description="模型名称")
    image: Optional[Union[str, List[str]]] = Field(None, description="待编辑图片文件")
    n: Optional[int] = Field(1, ge=1, le=10, description="生成数量 (1-10)")
    size: Optional[str] = Field("1024x1024", description="图片尺寸 (暂不支持)")
    quality: Optional[str] = Field("standard", description="图片质量 (暂不支持)")
    response_format: Optional[str] = Field(None, description="响应格式")
    style: Optional[str] = Field(None, description="风格 (暂不支持)")
    stream: Optional[bool] = Field(False, description="是否流式输出")


def validate_generation_request(request: ImageGenerationRequest):
    """验证图片生成请求参数"""
    model_id = request.model or "grok-imagine-1.0"
    model_info = ModelService.get(model_id)
    if not model_info or not model_info.is_image:
        image_models = [m.model_id for m in ModelService.MODELS if m.is_image]
        raise ValidationException(
            message=f"The model `{model_id}` is not supported for image generation. Supported: {image_models}",
            param="model",
            code="model_not_supported",
        )

    if not request.prompt or not request.prompt.strip():
        raise ValidationException(
            message="Prompt cannot be empty",
            param="prompt",
            code="empty_prompt",
        )

    if request.n is None:
        request.n = 1
    if request.n < 1 or request.n > 10:
        raise ValidationException(
            message="n must be between 1 and 10",
            param="n",
            code="invalid_n",
        )

    if request.stream and request.n not in [1, 2]:
        raise ValidationException(
            message="Streaming is only supported when n=1 or n=2",
            param="stream",
            code="invalid_stream_n",
        )

    if request.response_format:
        candidate = request.response_format.lower()
        if candidate not in ALLOWED_RESPONSE_FORMATS:
            raise ValidationException(
                message=f"response_format must be one of {sorted(ALLOWED_RESPONSE_FORMATS)}",
                param="response_format",
                code="invalid_response_format",
            )


def validate_edit_request(request: ImageEditRequest, images: List[UploadFile]):
    """验证图片编辑请求参数"""
    validate_generation_request(
        ImageGenerationRequest(
            prompt=request.prompt,
            model=request.model,
            n=request.n,
            size=request.size,
            quality=request.quality,
            response_format=request.response_format,
            style=request.style,
            stream=request.stream,
        )
    )
    if not images:
        raise ValidationException(
            message="Image is required",
            param="image",
            code="missing_image",
        )
    if len(images) > 16:
        raise ValidationException(
            message="Too many images. Maximum is 16.",
            param="image",
            code="invalid_image_count",
        )


def resolve_response_format(response_format: Optional[str]) -> str:
    candidate = response_format
    if not candidate:
        candidate = get_config("app.image_format", "url")
    if isinstance(candidate, str):
        candidate = candidate.lower()
    if candidate in ALLOWED_RESPONSE_FORMATS:
        return candidate
    raise ValidationException(
        message=f"response_format must be one of {sorted(ALLOWED_RESPONSE_FORMATS)}",
        param="response_format",
        code="invalid_response_format",
    )


def response_field_name(response_format: str) -> str:
    if response_format == "url":
        return "url"
    if response_format == "base64":
        return "base64"
    return "b64_json"


async def call_grok(
    token: str,
    prompt: str,
    model_info,
    file_attachments: Optional[List[str]] = None,
    response_format: str = "b64_json",
) -> List[str]:
    """
    调用 Grok 获取图片，返回图片列表
    """
    chat_service = GrokChatService()

    try:
        response = await chat_service.chat(
            token=token,
            message=prompt,
            model=model_info.grok_model,
            mode=model_info.model_mode,
            think=False,
            stream=True,
            file_attachments=file_attachments,
        )

        processor = ImageCollectProcessor(
            model_info.model_id,
            token,
            response_format=response_format,
        )
        return await processor.process(response)
    except Exception as e:
        logger.error(f"Grok image call failed: {e}")
        return []


async def _record_request(model_id: str, success: bool):
    try:
        await request_stats.record_request(model_id, success=success)
    except Exception:
        pass


async def _get_token_for_model(model_id: str):
    """获取指定模型可用 token，失败时抛出统一异常"""
    try:
        token_mgr = await get_token_manager()
        await token_mgr.reload_if_stale()
        token = token_mgr.get_token_for_model(model_id)
    except Exception as e:
        logger.error(f"Failed to get token: {e}")
        await _record_request(model_id or "image", False)
        raise AppException(
            message="Internal service error obtaining token",
            error_type=ErrorType.SERVER.value,
            code="internal_error",
        )

    if not token:
        await _record_request(model_id or "image", False)
        raise AppException(
            message="No available tokens. Please try again later.",
            error_type=ErrorType.RATE_LIMIT.value,
            code="rate_limit_exceeded",
            status_code=429,
        )
    return token_mgr, token


def _pick_images(all_images: List[str], n: int) -> List[str]:
    if len(all_images) >= n:
        return random.sample(all_images, n)
    selected = all_images.copy()
    while len(selected) < n:
        selected.append("error")
    return selected


def _build_image_response(selected_images: List[str], response_field: str) -> JSONResponse:
    import time

    return JSONResponse(
        content={
            "created": int(time.time()),
            "data": [{response_field: img} for img in selected_images],
            "usage": {
                "total_tokens": 0 * len([img for img in selected_images if img != "error"]),
                "input_tokens": 0,
                "output_tokens": 0 * len([img for img in selected_images if img != "error"]),
                "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
            },
        }
    )


@router.post("/images/generations")
async def create_image(
    request: ImageGenerationRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Image Generation API

    流式响应格式:
    - event: image_generation.partial_image
    - event: image_generation.completed
    """
    if request.stream is None:
        request.stream = False

    validate_generation_request(request)
    model_id = request.model or "grok-imagine-1.0"
    n = int(request.n or 1)
    response_format = resolve_response_format(request.response_format)
    request.response_format = response_format
    response_field = response_field_name(response_format)

    await enforce_daily_quota(api_key, model_id, image_count=n)
    token_mgr, token = await _get_token_for_model(model_id)
    model_info = ModelService.get(model_id)

    if request.stream:
        chat_service = GrokChatService()
        try:
            response = await chat_service.chat(
                token=token,
                message=f"Image Generation: {request.prompt}",
                model=model_info.grok_model,
                mode=model_info.model_mode,
                think=False,
                stream=True,
            )
        except Exception:
            await _record_request(model_info.model_id, False)
            raise

        processor = ImageStreamProcessor(
            model_info.model_id,
            token,
            n=n,
            response_format=response_format,
        )

        async def _wrapped_stream():
            completed = False
            try:
                async for chunk in processor.process(response):
                    yield chunk
                completed = True
            finally:
                try:
                    if completed:
                        await token_mgr.sync_usage(
                            token,
                            model_info.model_id,
                            consume_on_fail=True,
                            is_usage=True,
                        )
                        await _record_request(model_info.model_id, True)
                    else:
                        await _record_request(model_info.model_id, False)
                except Exception:
                    pass

        return StreamingResponse(
            _wrapped_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    calls_needed = (n + 1) // 2
    if calls_needed == 1:
        all_images = await call_grok(
            token,
            f"Image Generation: {request.prompt}",
            model_info,
            response_format=response_format,
        )
    else:
        tasks = [
            call_grok(
                token,
                f"Image Generation: {request.prompt}",
                model_info,
                response_format=response_format,
            )
            for _ in range(calls_needed)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_images = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Concurrent call failed: {result}")
            elif isinstance(result, list):
                all_images.extend(result)

    selected_images = _pick_images(all_images, n)
    success = any(isinstance(img, str) and img and img != "error" for img in selected_images)
    try:
        if success:
            await token_mgr.sync_usage(
                token,
                model_info.model_id,
                consume_on_fail=True,
                is_usage=True,
            )
        await _record_request(model_info.model_id, bool(success))
    except Exception:
        pass

    return _build_image_response(selected_images, response_field)


@router.post("/images/edits")
async def edit_image(
    prompt: str = Form(...),
    image: Optional[List[UploadFile]] = File(None),
    image_alias: Optional[List[UploadFile]] = File(None, alias="image[]"),
    model: Optional[str] = Form("grok-imagine-1.0"),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    quality: str = Form("standard"),
    response_format: Optional[str] = Form(None),
    style: Optional[str] = Form(None),
    stream: Optional[bool] = Form(False),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Image Edits API

    同官方 API 格式，仅支持 multipart/form-data 文件上传
    """
    try:
        edit_request = ImageEditRequest(
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            quality=quality,
            response_format=response_format,
            style=style,
            stream=stream,
        )
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = first.get("loc", [])
            msg = first.get("msg", "Invalid request")
            code = first.get("type", "invalid_value")
            param_parts = [str(x) for x in loc if not (isinstance(x, int) or str(x).isdigit())]
            param = ".".join(param_parts) if param_parts else None
            raise ValidationException(message=msg, param=param, code=code)
        raise ValidationException(message="Invalid request", code="invalid_value")

    if edit_request.stream is None:
        edit_request.stream = False
    if edit_request.n is None:
        edit_request.n = 1

    response_format = resolve_response_format(edit_request.response_format)
    edit_request.response_format = response_format
    response_field = response_field_name(response_format)
    images = (image or []) + (image_alias or [])
    validate_edit_request(edit_request, images)

    model_id = edit_request.model or "grok-imagine-1.0"
    n = int(edit_request.n or 1)

    await enforce_daily_quota(api_key, model_id, image_count=n)

    max_image_bytes = 50 * 1024 * 1024
    allowed_types = {"image/png", "image/jpeg", "image/webp", "image/jpg"}
    image_payloads: List[str] = []

    for item in images:
        content = await item.read()
        await item.close()
        if not content:
            raise ValidationException(
                message="File content is empty",
                param="image",
                code="empty_file",
            )
        if len(content) > max_image_bytes:
            raise ValidationException(
                message="Image file too large. Maximum is 50MB.",
                param="image",
                code="file_too_large",
            )

        mime = (item.content_type or "").lower()
        if mime == "image/jpg":
            mime = "image/jpeg"
        ext = Path(item.filename or "").suffix.lower()
        if mime not in allowed_types:
            if ext in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif ext == ".png":
                mime = "image/png"
            elif ext == ".webp":
                mime = "image/webp"
            else:
                raise ValidationException(
                    message="Unsupported image type. Supported: png, jpg, webp.",
                    param="image",
                    code="invalid_image_type",
                )

        image_payloads.append(f"data:{mime};base64,{base64.b64encode(content).decode()}")

    token_mgr, token = await _get_token_for_model(model_id)
    model_info = ModelService.get(model_id)

    file_ids: List[str] = []
    upload_service = UploadService()
    try:
        for payload in image_payloads:
            file_id, _ = await upload_service.upload(payload, token)
            if file_id:
                file_ids.append(file_id)
    finally:
        await upload_service.close()

    if edit_request.stream:
        chat_service = GrokChatService()
        try:
            response = await chat_service.chat(
                token=token,
                message=f"Image Edit: {edit_request.prompt}",
                model=model_info.grok_model,
                mode=model_info.model_mode,
                think=False,
                stream=True,
                file_attachments=file_ids,
            )
        except Exception:
            await _record_request(model_info.model_id, False)
            raise

        processor = ImageStreamProcessor(
            model_info.model_id,
            token,
            n=n,
            response_format=response_format,
        )

        async def _wrapped_stream():
            completed = False
            try:
                async for chunk in processor.process(response):
                    yield chunk
                completed = True
            finally:
                try:
                    if completed:
                        await token_mgr.sync_usage(
                            token,
                            model_info.model_id,
                            consume_on_fail=True,
                            is_usage=True,
                        )
                        await _record_request(model_info.model_id, True)
                    else:
                        await _record_request(model_info.model_id, False)
                except Exception:
                    pass

        return StreamingResponse(
            _wrapped_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    calls_needed = (n + 1) // 2
    if calls_needed == 1:
        all_images = await call_grok(
            token,
            f"Image Edit: {edit_request.prompt}",
            model_info,
            file_attachments=file_ids,
            response_format=response_format,
        )
    else:
        tasks = [
            call_grok(
                token,
                f"Image Edit: {edit_request.prompt}",
                model_info,
                file_attachments=file_ids,
                response_format=response_format,
            )
            for _ in range(calls_needed)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_images = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Concurrent call failed: {result}")
            elif isinstance(result, list):
                all_images.extend(result)

    selected_images = _pick_images(all_images, n)
    success = any(isinstance(img, str) and img and img != "error" for img in selected_images)
    try:
        if success:
            await token_mgr.sync_usage(
                token,
                model_info.model_id,
                consume_on_fail=True,
                is_usage=True,
            )
        await _record_request(model_info.model_id, bool(success))
    except Exception:
        pass

    return _build_image_response(selected_images, response_field)


__all__ = ["router"]
