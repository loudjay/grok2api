"""Helpers for rendering generated Grok assets in compatible responses."""

from __future__ import annotations

from app.core.config import get_config
from app.services.grok.assets import DownloadService


async def render_generated_image(token: str, url: str) -> str:
    image_format = str(get_config("app.image_format", "url") or "url").strip().lower()
    service = DownloadService()
    try:
        if image_format == "base64":
            base64_data = await service.to_base64(url, token, "image")
            if base64_data:
                return f"![image]({base64_data})"
        local_url = await _cache_local_url(service, token, url)
        if local_url:
            return f"![image]({local_url})"
        return f"![image]({url})"
    finally:
        await service.close()


async def _cache_local_url(service: DownloadService, token: str, url: str) -> str:
    await service.download(url, token, "image")
    path = _normalize_asset_path(url)
    local_path = f"/v1/files/image{path}"
    app_url = str(get_config("app.app_url", "") or "").strip().rstrip("/")
    if app_url:
        return f"{app_url}{local_path}"
    return local_path


def _normalize_asset_path(raw: str) -> str:
    value = str(raw or "").strip()
    if value.startswith("http"):
        from urllib.parse import urlparse

        value = urlparse(value).path
    if not value.startswith("/"):
        value = f"/{value}"
    return value


__all__ = ["render_generated_image"]
