"""Parse Grok NDJSON chat responses into structured events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import orjson


def classify_line(line: str | bytes) -> tuple[str, str]:
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    text = line.strip()
    if not text:
        return "skip", ""
    if text.startswith("data:"):
        data = text[5:].strip()
        if data == "[DONE]":
            return "done", ""
        return "data", data
    if text.startswith("event:"):
        return "skip", ""
    if text.startswith("{"):
        return "data", text
    return "skip", ""


@dataclass(slots=True)
class StreamEvent:
    kind: str
    content: str = ""
    image_url: str = ""


class GrokStreamAdapter:
    __slots__ = ("text_parts", "thinking_parts", "image_urls", "_final_message")

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.image_urls: list[str] = []
        self._final_message = ""

    def feed(self, data: str) -> list[StreamEvent]:
        try:
            obj = orjson.loads(data)
        except Exception:
            return []

        result = obj.get("result")
        if not isinstance(result, dict):
            return []
        response = result.get("response")
        if not isinstance(response, dict):
            return []

        events: list[StreamEvent] = []
        token = response.get("token")
        if isinstance(token, str) and token:
            if bool(response.get("isThinking")):
                self.thinking_parts.append(token)
                events.append(StreamEvent("thinking", content=token))
            else:
                self.text_parts.append(token)
                events.append(StreamEvent("text", content=token))

        model_response = response.get("modelResponse")
        if isinstance(model_response, dict):
            message = model_response.get("message")
            if isinstance(message, str) and message.strip():
                self._final_message = message
            generated = model_response.get("generatedImageUrls")
            if isinstance(generated, list):
                for raw in generated:
                    if not isinstance(raw, str):
                        continue
                    image_url = raw.strip()
                    if not image_url or image_url in self.image_urls:
                        continue
                    self.image_urls.append(image_url)
                    events.append(StreamEvent("image", image_url=image_url))

        if response.get("finalMetadata") or response.get("isSoftStop"):
            events.append(StreamEvent("soft_stop"))
        return events

    @property
    def final_text(self) -> str:
        text = "".join(self.text_parts).strip()
        if text:
            return text
        return self._final_message.strip()

    @property
    def final_thinking(self) -> str:
        return "".join(self.thinking_parts).strip()


__all__ = ["GrokStreamAdapter", "StreamEvent", "classify_line"]

