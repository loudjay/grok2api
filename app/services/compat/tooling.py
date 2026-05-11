"""Tool prompt injection and tool-call parsing helpers."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any


_TOOL_SYSTEM_HEADER = """\
You have access to the following tools.

AVAILABLE TOOLS:
{tool_definitions}

TOOL CALL FORMAT — follow these rules exactly:
- When calling a tool, output ONLY the XML block below. No text before or after it.
- <parameters> must be a single-line valid JSON object.
- Place multiple tool calls inside ONE <tool_calls> element.
- Do NOT use markdown code fences around the XML.

<tool_calls>
  <tool_call>
    <tool_name>TOOL_NAME</tool_name>
    <parameters>{{"key":"value"}}</parameters>
  </tool_call>
</tool_calls>

{tool_choice_instruction}\
"""

_CHOICE_AUTO = "WHEN TO CALL: Call a tool only when it is clearly needed. Otherwise respond in plain text."
_CHOICE_NONE = "WHEN TO CALL: Do NOT call any tools. Respond in plain text only."
_CHOICE_REQUIRED = "WHEN TO CALL: You MUST output a <tool_calls> XML block. Do NOT write a plain-text reply."
_CHOICE_FORCED = 'WHEN TO CALL: You MUST call the tool named "{name}" and output ONLY a <tool_calls> XML block.'

_TOOL_SYNTAX_PATTERNS = re.compile(
    r"<tool_calls|<tool_call|<function_call|<invoke\s|"
    r'"tool_calls"\s*:|\btool_calls\b',
    re.IGNORECASE,
)
_XML_ROOT_RE = re.compile(r"<tool_calls\s*>(.*?)</tool_calls\s*>", re.DOTALL | re.IGNORECASE)
_XML_CALL_RE = re.compile(r"<tool_call\s*>(.*?)</tool_call\s*>", re.DOTALL | re.IGNORECASE)
_XML_NAME_RE = re.compile(r"<tool_name\s*>(.*?)</tool_name\s*>", re.DOTALL | re.IGNORECASE)
_XML_PARAMS_RE = re.compile(r"<parameters\s*>(.*?)</parameters\s*>", re.DOTALL | re.IGNORECASE)
_FC_RE = re.compile(r"<function_call\s*>(.*?)</function_call\s*>", re.DOTALL | re.IGNORECASE)
_INVOKE_RE = re.compile(r'<invoke\s+name=["\']?(\w+)["\']?\s*>(.*?)</invoke\s*>', re.DOTALL | re.IGNORECASE)
_FC_NAME_RE = re.compile(r"<name\s*>(.*?)</name\s*>", re.DOTALL | re.IGNORECASE)
_FC_ARGS_RE = re.compile(r"<arguments\s*>(.*?)</arguments\s*>", re.DOTALL | re.IGNORECASE)
_JSON_ARR_RE = re.compile(r"\[[\s\S]+\]", re.DOTALL)
_JSON_DECODER = json.JSONDecoder()
_OPEN_TAG_RE = re.compile(r"<tool_calls[\s>]?", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"</tool_calls\s*>", re.IGNORECASE)


@dataclass
class ParsedToolCall:
    call_id: str
    name: str
    arguments: str

    @staticmethod
    def make(name: str, arguments: Any) -> "ParsedToolCall":
        call_id = f"call_{int(time.time() * 1000)}{os.urandom(3).hex()}"
        if isinstance(arguments, str):
            args_str = arguments
        else:
            try:
                args_str = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                args_str = "{}"
        return ParsedToolCall(call_id=call_id, name=name, arguments=args_str)


@dataclass
class ParseResult:
    calls: list[ParsedToolCall] = field(default_factory=list)
    saw_tool_syntax: bool = False


def build_tool_system_prompt(tools: list[dict[str, Any]], tool_choice: Any = None) -> str:
    return _TOOL_SYSTEM_HEADER.format(
        tool_definitions=_format_tool_definitions(tools),
        tool_choice_instruction=_build_choice_instruction(tool_choice),
    )


def extract_tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        func = tool.get("function") or {}
        name = str(func.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def inject_into_message(message: str, system_prompt: str) -> str:
    return f"[system]: {system_prompt}\n\n{message}"


def tool_calls_to_xml(tool_calls: list[dict[str, Any]]) -> str:
    lines = ["<tool_calls>"]
    for tool_call in tool_calls:
        func = tool_call.get("function") or {}
        name = str(func.get("name") or "").strip()
        args = str(func.get("arguments") or "{}").strip() or "{}"
        try:
            args = json.dumps(json.loads(args), ensure_ascii=False, separators=(",", ":"))
        except Exception:
            pass
        lines.append("  <tool_call>")
        lines.append(f"    <tool_name>{name}</tool_name>")
        lines.append(f"    <parameters>{args}</parameters>")
        lines.append("  </tool_call>")
    lines.append("</tool_calls>")
    return "\n".join(lines)


def parse_tool_calls(text: str, available_tools: list[str] | None = None) -> ParseResult:
    result = ParseResult()
    if not text or not text.strip():
        return result
    if not _TOOL_SYNTAX_PATTERNS.search(text):
        return result
    result.saw_tool_syntax = True
    calls = (
        _parse_xml_tool_calls(text)
        or _parse_json_envelope(text)
        or _parse_json_array(text)
        or _parse_alt_xml(text)
    )
    if calls and available_tools:
        calls = [call for call in calls if call.name in available_tools]
    result.calls = calls or []
    return result


class ToolSieve:
    __slots__ = ("_tool_names", "_buf", "_capturing", "_done")

    def __init__(self, tool_names: list[str]) -> None:
        self._tool_names = tool_names
        self._buf = ""
        self._capturing = False
        self._done = False

    def feed(self, chunk: str) -> tuple[str, list[ParsedToolCall] | None]:
        if self._done or not chunk:
            return (chunk if not self._capturing else ""), None
        if self._capturing:
            return self._feed_capturing(chunk)
        return self._feed_scanning(chunk)

    def flush(self) -> list[ParsedToolCall] | None:
        if self._done or not self._buf:
            return None
        self._done = True
        result = parse_tool_calls(self._buf, self._tool_names)
        self._buf = ""
        if result.saw_tool_syntax:
            return result.calls
        return None

    def _feed_scanning(self, chunk: str) -> tuple[str, list[ParsedToolCall] | None]:
        combined = self._buf + chunk
        self._buf = ""
        match = _OPEN_TAG_RE.search(combined)
        if match is None:
            safe, leftover = _split_at_boundary(combined, "<tool_calls")
            self._buf = leftover
            return safe, None
        safe_part = combined[: match.start()]
        self._buf = combined[match.start() :]
        self._capturing = True
        _, calls = self._feed_capturing("")
        return safe_part, calls

    def _feed_capturing(self, chunk: str) -> tuple[str, list[ParsedToolCall] | None]:
        self._buf += chunk
        match = _CLOSE_TAG_RE.search(self._buf)
        if match is None:
            return "", None
        xml_block = self._buf[: match.end()]
        self._buf = ""
        self._capturing = False
        self._done = True
        result = parse_tool_calls(xml_block, self._tool_names)
        return "", (result.calls if result.saw_tool_syntax else None)


def _format_tool_definitions(tools: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for tool in tools:
        func = tool.get("function") or {}
        name = str(func.get("name") or "").strip()
        desc = str(func.get("description") or "").strip()
        params = func.get("parameters")
        lines = [f"Tool: {name}"]
        if desc:
            lines.append(f"Description: {desc}")
        if params:
            try:
                lines.append(f"Parameters: {json.dumps(params, ensure_ascii=False)}")
            except Exception:
                lines.append(f"Parameters: {params}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _build_choice_instruction(tool_choice: Any) -> str:
    if tool_choice is None or tool_choice == "auto":
        return _CHOICE_AUTO
    if tool_choice == "none":
        return _CHOICE_NONE
    if tool_choice == "required":
        return _CHOICE_REQUIRED
    if isinstance(tool_choice, dict):
        tc_type = str(tool_choice.get("type") or "").strip()
        if tc_type == "none":
            return _CHOICE_NONE
        if tc_type == "required":
            return _CHOICE_REQUIRED
        if tc_type == "function":
            name = str((tool_choice.get("function") or {}).get("name") or "").strip()
            if name:
                return _CHOICE_FORCED.format(name=name)
    return _CHOICE_AUTO


def _parse_xml_tool_calls(text: str) -> list[ParsedToolCall]:
    match = _XML_ROOT_RE.search(text)
    if not match:
        return []
    calls: list[ParsedToolCall] = []
    for call_match in _XML_CALL_RE.finditer(match.group(1)):
        inner = call_match.group(1)
        name_match = _XML_NAME_RE.search(inner)
        params_match = _XML_PARAMS_RE.search(inner)
        if not name_match:
            continue
        parsed_args = _parse_json_tolerant(params_match.group(1).strip() if params_match else "{}")
        if parsed_args is None:
            continue
        calls.append(ParsedToolCall.make(name_match.group(1).strip(), parsed_args))
    return calls


def _parse_json_envelope(text: str) -> list[ParsedToolCall]:
    if '"tool_calls"' not in text:
        return []
    obj = _extract_outermost_json_obj(text)
    if not isinstance(obj, dict):
        return []
    raw_calls = obj.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    return _extract_from_call_list(raw_calls)


def _parse_json_array(text: str) -> list[ParsedToolCall]:
    match = _JSON_ARR_RE.search(text)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
    except Exception:
        return []
    if not isinstance(arr, list):
        return []
    return _extract_from_call_list(arr)


def _parse_alt_xml(text: str) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for match in _FC_RE.finditer(text):
        inner = match.group(1)
        name_match = _FC_NAME_RE.search(inner)
        args_match = _FC_ARGS_RE.search(inner)
        if not name_match:
            continue
        args = _parse_json_tolerant(args_match.group(1).strip() if args_match else "{}")
        if args is None:
            continue
        calls.append(ParsedToolCall.make(name_match.group(1).strip(), args))
    for match in _INVOKE_RE.finditer(text):
        args = _parse_json_tolerant(match.group(2).strip())
        calls.append(ParsedToolCall.make(match.group(1).strip(), args if args is not None else {}))
    return calls


def _extract_outermost_json_obj(text: str) -> Any:
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = _JSON_DECODER.raw_decode(text, start)
        return obj
    except Exception:
        end = text.rfind("}") + 1
        return _try_repair_json(text[start:end]) if end > start else None


def _extract_from_call_list(items: list[Any]) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("tool_name") or "").strip()
        args = item.get("input") or item.get("arguments") or item.get("parameters") or {}
        if name:
            calls.append(ParsedToolCall.make(name, args))
    return calls


def _parse_json_tolerant(raw: str) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return _try_repair_json(raw)


def _try_repair_json(raw: str) -> Any:
    try:
        return json.loads(re.sub(r"(?<!\\)\n", r"\\n", raw))
    except Exception:
        return None


def _split_at_boundary(text: str, prefix: str) -> tuple[str, str]:
    limit = min(len(prefix) - 1, len(text))
    for size in range(limit, 0, -1):
        if text.endswith(prefix[:size]):
            return text[:-size], text[-size:]
    return text, ""


__all__ = [
    "ParseResult",
    "ParsedToolCall",
    "ToolSieve",
    "build_tool_system_prompt",
    "extract_tool_names",
    "inject_into_message",
    "parse_tool_calls",
    "tool_calls_to_xml",
]
