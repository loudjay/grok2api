import json

import pytest

from app.services.compat.anthropic_api import messages_create
from app.services.compat.common import PreparedChatRequest
from app.services.compat.responses_api import responses_create


class _DummyTokenManager:
    async def sync_usage(self, *_args, **_kwargs):
        return True


def _tool_xml_stream():
    payload = {
        "result": {
            "response": {
                "token": "<tool_calls><tool_call><tool_name>lookup</tool_name><parameters>{\"id\":1}</parameters></tool_call></tool_calls>",
                "isThinking": False,
            }
        }
    }

    async def _gen():
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return _gen()


def _text_stream():
    frames = [
        {"result": {"response": {"token": "hello ", "isThinking": False}}},
        {"result": {"response": {"token": "world", "isThinking": False}}},
        {"result": {"response": {"finalMetadata": {"done": True}}}},
    ]

    async def _gen():
        for frame in frames:
            yield f"data: {json.dumps(frame)}\n\n"
        yield "data: [DONE]\n\n"

    return _gen()


@pytest.mark.asyncio
async def test_responses_create_maps_tool_calls(monkeypatch):
    prepared = PreparedChatRequest(
        model="grok-4",
        token="token-demo",
        token_manager=_DummyTokenManager(),
        prompt="prompt",
        tool_names=["lookup"],
        raw_stream=_tool_xml_stream(),
    )

    async def _fake_prepare(**_kwargs):
        return prepared

    async def _fake_finalize(_prepared, *, success):
        assert success is True

    monkeypatch.setattr("app.services.compat.responses_api.prepare_chat_request", _fake_prepare)
    monkeypatch.setattr("app.services.compat.responses_api.finalize_chat_request", _fake_finalize)

    result = await responses_create(
        model="grok-4",
        input_value="demo",
        instructions=None,
        stream=False,
        emit_think=True,
        tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
        tool_choice="auto",
    )

    assert result["object"] == "response"
    output = result["output"]
    assert output[0]["type"] == "function_call"
    assert output[0]["name"] == "lookup"
    assert output[0]["arguments"] == '{"id":1}'


@pytest.mark.asyncio
async def test_messages_create_maps_tool_use(monkeypatch):
    prepared = PreparedChatRequest(
        model="grok-4",
        token="token-demo",
        token_manager=_DummyTokenManager(),
        prompt="prompt",
        tool_names=["lookup"],
        raw_stream=_tool_xml_stream(),
    )

    async def _fake_prepare(**_kwargs):
        return prepared

    async def _fake_finalize(_prepared, *, success):
        assert success is True

    monkeypatch.setattr("app.services.compat.anthropic_api.prepare_chat_request", _fake_prepare)
    monkeypatch.setattr("app.services.compat.anthropic_api.finalize_chat_request", _fake_finalize)

    result = await messages_create(
        model="grok-4",
        messages=[{"role": "user", "content": "demo"}],
        system=None,
        stream=False,
        emit_think=True,
        tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "lookup"},
    )

    assert result["type"] == "message"
    assert result["stop_reason"] == "tool_use"
    assert result["content"][0]["type"] == "tool_use"
    assert result["content"][0]["name"] == "lookup"
    assert result["content"][0]["input"] == {"id": 1}


@pytest.mark.asyncio
async def test_responses_create_text_output(monkeypatch):
    prepared = PreparedChatRequest(
        model="grok-4",
        token="token-demo",
        token_manager=_DummyTokenManager(),
        prompt="prompt",
        tool_names=[],
        raw_stream=_text_stream(),
    )

    async def _fake_prepare(**_kwargs):
        return prepared

    async def _fake_finalize(_prepared, *, success):
        assert success is True

    monkeypatch.setattr("app.services.compat.responses_api.prepare_chat_request", _fake_prepare)
    monkeypatch.setattr("app.services.compat.responses_api.finalize_chat_request", _fake_finalize)

    result = await responses_create(
        model="grok-4",
        input_value="demo",
        instructions=None,
        stream=False,
        emit_think=True,
    )

    assert result["output"][0]["type"] == "message"
    assert result["output"][0]["content"][0]["text"] == "hello world"
