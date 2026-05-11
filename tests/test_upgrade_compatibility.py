import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import app.core.config as config_module
import app.core.storage as storage_module
import app.services.token.manager as token_manager_module
from app.core.legacy_migration import migrate_legacy_cache_dirs
from app.services.api_keys import api_key_manager


class _DummyRemoteStorage:
    def __init__(self, config_data=None, token_data=None):
        self._config_data = config_data
        self._token_data = token_data
        self.saved_config = None
        self.saved_tokens = None

    async def load_config(self):
        return self._config_data

    async def save_config(self, data):
        self.saved_config = data
        self._config_data = data

    async def load_tokens(self):
        return self._token_data

    async def save_tokens(self, data):
        self.saved_tokens = data
        self._token_data = data

    @asynccontextmanager
    async def acquire_lock(self, _name: str, timeout: int = 10):
        yield


def test_config_load_merges_legacy_setting_file(monkeypatch, tmp_path):
    defaults_path = tmp_path / "config.defaults.toml"
    defaults_path.write_text(
        "\n".join(
            [
                "[app]",
                'app_url = "http://127.0.0.1:8000"',
                'admin_username = "admin"',
                'app_key = "grok2api"',
                'api_key = ""',
                'image_format = "url"',
                "",
                "[grok]",
                'base_proxy_url = ""',
                'asset_proxy_url = ""',
                'cf_clearance = ""',
                "temporary = false",
                "thinking = true",
                "dynamic_statsig = true",
                'filter_tags = ["xaiartifact","xai:tool_usage_card"]',
                "timeout = 600",
                "retry_status_codes = [401, 429, 403]",
                "",
                "[cache]",
                "limit_mb = 1536",
                "",
            ]
        ),
        encoding="utf-8",
    )
    legacy_path = tmp_path / "data" / "setting.toml"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        "\n".join(
            [
                "[global]",
                'base_url = "https://legacy.example.com"',
                'admin_username = "legacy-admin"',
                'admin_password = "legacy-pass"',
                "image_cache_max_size_mb = 256",
                "video_cache_max_size_mb = 512",
                "",
                "[grok]",
                'api_key = "legacy-api-key"',
                'proxy_url = "https://proxy.example.com"',
                'cache_proxy_url = "https://asset-proxy.example.com"',
                'cf_clearance = "legacy-cf"',
                "temporary = true",
                "show_thinking = false",
                "dynamic_statsig = false",
                'filtered_tags = "foo, bar"',
                "stream_total_timeout = 321",
                "retry_status_codes = [401, 418]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    remote = _DummyRemoteStorage(config_data=None)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_FILE", defaults_path)
    monkeypatch.setattr(config_module, "LEGACY_CONFIG_FILE", legacy_path)
    monkeypatch.setattr(storage_module, "CONFIG_FILE", tmp_path / "data" / "config.toml")
    monkeypatch.setattr(storage_module, "get_storage", lambda: remote)

    cfg = config_module.Config()
    asyncio.run(cfg.load())

    assert cfg.get("app.app_url") == "https://legacy.example.com"
    assert cfg.get("app.admin_username") == "legacy-admin"
    assert cfg.get("app.app_key") == "legacy-pass"
    assert cfg.get("app.api_key") == "legacy-api-key"
    assert cfg.get("grok.base_proxy_url") == "https://proxy.example.com"
    assert cfg.get("grok.asset_proxy_url") == "https://asset-proxy.example.com"
    assert cfg.get("grok.cf_clearance") == "legacy-cf"
    assert cfg.get("grok.temporary") is True
    assert cfg.get("grok.thinking") is False
    assert cfg.get("grok.dynamic_statsig") is False
    assert cfg.get("grok.filter_tags") == ["foo", "bar"]
    assert cfg.get("grok.timeout") == 321
    assert cfg.get("grok.retry_status_codes") == [401, 418]
    assert cfg.get("cache.limit_mb") == 768
    assert remote.saved_config is not None


def test_token_manager_bootstraps_remote_storage_from_local_token_file(monkeypatch, tmp_path):
    token_file = tmp_path / "data" / "token.json"
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(
        """
{
  "ssoBasic": [
    {"token": "sso=token-a", "status": "active", "quota": 80}
  ],
  "ssoSuper": [
    {"token": "token-b", "status": "active", "quota": 70, "heavy_quota": 5}
  ]
}
        """.strip(),
        encoding="utf-8",
    )

    remote = _DummyRemoteStorage(token_data=None)
    monkeypatch.setattr(storage_module, "TOKEN_FILE", token_file)
    monkeypatch.setattr(token_manager_module, "get_storage", lambda: remote)
    token_manager_module.TokenManager._instance = None

    manager = asyncio.run(token_manager_module.get_token_manager())

    assert remote.saved_tokens is not None
    assert remote.saved_tokens["ssoBasic"][0]["token"] == "token-a"
    assert manager.get_pool_tokens("ssoBasic")[0].token == "token-a"
    assert manager.get_pool_tokens("ssoSuper")[0].heavy_quota == 5

    token_manager_module.TokenManager._instance = None


def test_api_key_manager_loads_legacy_files(monkeypatch, tmp_path):
    api_keys_path = tmp_path / "data" / "api_keys.json"
    usage_path = tmp_path / "data" / "api_key_usage.json"
    api_keys_path.parent.mkdir(parents=True, exist_ok=True)
    api_keys_path.write_text(
        """
[
  {"key": "legacy-active", "name": "Legacy Active", "created_at": 1700000000, "is_active": true, "chat_limit": 10},
  {"key": "legacy-disabled", "name": "Legacy Disabled", "created_at": 1700000001, "is_active": false}
]
        """.strip(),
        encoding="utf-8",
    )
    usage_path.write_text(
        """
{
  "2026-04-16": {
    "legacy-active": {"chat_used": 3, "heavy_used": 0, "image_used": 1, "video_used": 0, "updated_at": 1700000000000}
  }
}
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(api_key_manager, "file_path", api_keys_path)
    monkeypatch.setattr(api_key_manager, "usage_path", usage_path)
    monkeypatch.setattr(api_key_manager, "_keys", [])
    monkeypatch.setattr(api_key_manager, "_usage", {})
    monkeypatch.setattr(api_key_manager, "_loaded", False)
    monkeypatch.setattr(api_key_manager, "_usage_loaded", False)

    asyncio.run(api_key_manager.init())
    rows = api_key_manager.get_all_keys()
    usage = asyncio.run(api_key_manager.usage_for_day("2026-04-16"))

    assert [row["key"] for row in rows] == ["legacy-active", "legacy-disabled"]
    assert api_key_manager.validate_key("legacy-active")["name"] == "Legacy Active"
    assert api_key_manager.validate_key("legacy-disabled") is None
    assert usage["legacy-active"]["chat_used"] == 3
    assert usage["legacy-active"]["image_used"] == 1


def test_migrate_legacy_cache_dirs_moves_old_cache_files(tmp_path):
    legacy_image = tmp_path / "temp" / "image"
    legacy_video = tmp_path / "temp" / "video"
    legacy_image.mkdir(parents=True, exist_ok=True)
    legacy_video.mkdir(parents=True, exist_ok=True)
    (legacy_image / "a.png").write_bytes(b"image-data")
    (legacy_video / "b.mp4").write_bytes(b"video-data")

    result = migrate_legacy_cache_dirs(tmp_path)

    assert result["migrated"] is True
    assert result["moved"] == 2
    assert (tmp_path / "tmp" / "image" / "a.png").read_bytes() == b"image-data"
    assert (tmp_path / "tmp" / "video" / "b.mp4").read_bytes() == b"video-data"
    assert not (tmp_path / "temp").exists()
