import base64
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.cloud_tts import frontend_contrib as fc
from plugins.cloud_tts import plugin as plugin_module
from plugins.cloud_tts import state


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    configs = {slug: {} for slug in fc.PROVIDER_SLUGS}
    extras = {slug: {} for slug in fc.PROVIDER_SLUGS}
    saved: list[str] = []

    def fake_load(plugin_root, slug=state.PROVIDER_SLUG):
        return dict(configs[slug])

    def fake_save(plugin_root, data, slug=state.PROVIDER_SLUG):
        configs[slug] = dict(data)
        saved.append(slug)

    monkeypatch.setattr(state, "load_plugin_config", fake_load)
    monkeypatch.setattr(state, "save_plugin_config", fake_save)
    monkeypatch.setattr(state, "get_cloud_extra", lambda: dict(extras[state.PROVIDER_SLUG]))
    monkeypatch.setattr(state, "get_qwen_extra", lambda: dict(extras[state.QWEN_PROVIDER_SLUG]))
    monkeypatch.setattr(
        state, "get_gpt_sovits_extra", lambda: dict(extras[state.GPT_SOVITS_PROVIDER_SLUG])
    )
    monkeypatch.setattr(
        state, "set_cloud_extra", lambda e: extras[state.PROVIDER_SLUG].update(e)
    )
    monkeypatch.setattr(
        state, "set_qwen_extra", lambda e: extras[state.QWEN_PROVIDER_SLUG].update(e)
    )
    monkeypatch.setattr(
        state,
        "set_gpt_sovits_extra",
        lambda e: extras[state.GPT_SOVITS_PROVIDER_SLUG].update(e),
    )
    monkeypatch.setattr(state, "current_tts_provider", lambda: state.PROVIDER_SLUG)
    monkeypatch.setattr(
        state,
        "load_characters",
        lambda: [{"name": "Hanadan", "prompt_text": "卡片文本"}],
    )
    monkeypatch.setattr(
        state,
        "find_character",
        lambda name: {"name": name, "prompt_text": "卡片文本"} if name == "Hanadan" else None,
    )
    monkeypatch.setattr(state, "resolve_reference_audio", lambda char: None)
    monkeypatch.setattr(state, "plugin_data_root", lambda: tmp_path / "plugin_data")
    monkeypatch.setattr(state, "suppress_prompt_constraint", lambda *a, **k: None)
    monkeypatch.setattr(
        state, "migrate_package_config_to_data_root", lambda *a, **k: None
    )
    monkeypatch.setattr(
        state, "migrate_api_extra_to_plugin_state", lambda *a, **k: None
    )
    return SimpleNamespace(
        configs=configs, extras=extras, saved=saved, plugin_root=tmp_path
    )


def _surface(fake_env):
    return fc.build_api_surface(fake_env.plugin_root)


def _run_action(surface, action_id, values):
    action = next(a for a in surface.actions if a.id == action_id)
    return action.run(values)


def test_load_values_snapshot_has_three_providers_without_secrets(fake_env):
    fake_env.extras[state.PROVIDER_SLUG]["api_key"] = "sk-very-secret"
    fake_env.extras[state.PROVIDER_SLUG]["base_api_url"] = "https://secret.example"

    values = _surface(fake_env).load_values()

    assert values["provider"] == state.PROVIDER_SLUG
    slugs = [block["slug"] for block in values["providers"]]
    assert slugs == list(fc.PROVIDER_SLUGS)
    minimax = values["providers"][0]
    assert minimax["key_configured"] is True
    assert values["providers"][1]["key_configured"] is False
    serialized = json.dumps(values, ensure_ascii=False)
    assert "sk-very-secret" not in serialized
    assert "secret.example" not in serialized
    assert values["characters"] == [{"name": "Hanadan", "prompt_text": "卡片文本"}]


def test_save_values_whitelists_plugin_domain_fields(fake_env):
    surface = _surface(fake_env)

    surface.save_values(
        {
            "provider": state.PROVIDER_SLUG,
            "model": "speech-2.8-hd",
            "default_voice_id": "v1",
            "auto_prompt_constraint": True,
            "api_key": "injected",
            "base_api_url": "https://evil.example",
        }
    )

    cfg = fake_env.configs[state.PROVIDER_SLUG]
    assert cfg["model"] == "speech-2.8-hd"
    assert cfg["auto_prompt_constraint"] is True
    assert "api_key" not in cfg
    assert "base_api_url" not in cfg
    extra = fake_env.extras[state.PROVIDER_SLUG]
    assert extra == {"model": "speech-2.8-hd", "default_voice_id": "v1"}


def test_save_values_rejects_unknown_provider(fake_env):
    with pytest.raises(ValueError):
        _surface(fake_env).save_values({"provider": "nope"})


def test_bind_voice_updates_map_and_versions(fake_env):
    surface = _surface(fake_env)

    result = _run_action(
        surface,
        "bind_voice",
        {"provider": state.PROVIDER_SLUG, "character": "Hanadan", "voice_id": "v9"},
    )

    assert result["character"]["voice_id"] == "v9"
    cfg = fake_env.configs[state.PROVIDER_SLUG]
    assert cfg["voice_id_map"] == {"Hanadan": "v9"}
    assert cfg["voice_id_versions"]["Hanadan"][0]["source"] == "manual"

    result = _run_action(
        surface,
        "bind_voice",
        {"provider": state.PROVIDER_SLUG, "character": "Hanadan", "voice_id": ""},
    )

    assert result["character"]["voice_id"] == ""
    assert fake_env.configs[state.PROVIDER_SLUG]["voice_id_map"] == {}


def test_list_characters_reuses_loaded_rows_without_refinding(fake_env, monkeypatch):
    monkeypatch.setattr(
        state,
        "load_characters",
        lambda: [
            {"name": "Hanadan", "prompt_text": "first"},
            {"name": "Akari", "prompt_text": "second"},
        ],
    )
    monkeypatch.setattr(
        state,
        "find_character",
        lambda name: pytest.fail(f"unexpected character reload for {name}"),
    )

    result = _run_action(
        _surface(fake_env),
        "list_characters",
        {"provider": state.PROVIDER_SLUG},
    )

    rows = result["characters"]
    assert [row["name"] for row in rows] == ["Hanadan", "Akari"]
    assert [row["prompt_text"] for row in rows] == ["first", "second"]


def test_upload_reference_writes_file_and_maps(fake_env):
    surface = _surface(fake_env)
    content = b"RIFF-fake-wav-bytes"

    result = _run_action(
        surface,
        "upload_reference",
        {
            "provider": state.PROVIDER_SLUG,
            "character": "Hanadan",
            "filename": "ref.wav",
            "content_base64": base64.b64encode(content).decode("ascii"),
            "text": "试听文本",
            "language": "zh",
        },
    )

    stored = Path(result["path"])
    assert stored.is_file()
    assert stored.read_bytes() == content
    cfg = fake_env.configs[state.PROVIDER_SLUG]
    assert cfg["local_reference_audio_map"]["Hanadan"] == str(stored)
    assert cfg["reference_text_map"]["Hanadan"] == "试听文本"
    assert cfg["reference_audio_language_map"]["Hanadan"] == "zh"


def test_upload_reference_rejects_bad_extension_and_oversize(fake_env, monkeypatch):
    surface = _surface(fake_env)
    base = {
        "provider": state.PROVIDER_SLUG,
        "character": "Hanadan",
        "content_base64": base64.b64encode(b"x").decode("ascii"),
    }

    with pytest.raises(ValueError, match="不支持的音频格式"):
        _run_action(surface, "upload_reference", {**base, "filename": "evil.exe"})

    monkeypatch.setattr(fc, "MAX_REFERENCE_AUDIO_BYTES", 4)
    big = base64.b64encode(b"12345").decode("ascii")
    with pytest.raises(ValueError, match="上限"):
        _run_action(
            surface,
            "upload_reference",
            {**base, "filename": "ref.wav", "content_base64": big},
        )


def test_clear_reference_pops_maps(fake_env):
    fake_env.configs[state.PROVIDER_SLUG] = {
        "local_reference_audio_map": {"Hanadan": "C:/x.wav"},
    }
    surface = _surface(fake_env)

    result = _run_action(
        surface,
        "clear_reference",
        {"provider": state.PROVIDER_SLUG, "character": "Hanadan"},
    )

    assert result["character"]["reference_audio"] == ""
    assert fake_env.configs[state.PROVIDER_SLUG]["local_reference_audio_map"] == {}


def test_clone_voice_binds_and_returns_demo_url(fake_env, monkeypatch, tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"audio")
    fake_env.extras[state.PROVIDER_SLUG]["api_key"] = "sk-ok"
    fake_env.configs[state.PROVIDER_SLUG] = {
        "local_reference_audio_map": {"Hanadan": str(ref)},
        "reference_text_map": {"Hanadan": "自定义文本"},
    }
    demo = tmp_path / "demo.mp3"
    demo.write_bytes(b"demo")
    calls = {}

    class FakeAdapter:
        last_clone_demo_audio_path = str(demo)

        def create_cloned_voice_from_file(self, path, **kwargs):
            calls["path"] = Path(path)
            calls["kwargs"] = kwargs
            return "voice-123"

    monkeypatch.setattr(fc, "_build_adapter", lambda root, slug: FakeAdapter())

    result = _run_action(
        _surface(fake_env),
        "clone_voice",
        {"provider": state.PROVIDER_SLUG, "character": "Hanadan"},
    )

    assert result["voice_id"] == "voice-123"
    assert result["demo_url"].startswith("/api/media?path=")
    assert calls["kwargs"]["prompt_text"] == "自定义文本"
    cfg = fake_env.configs[state.PROVIDER_SLUG]
    assert cfg["voice_id_map"]["Hanadan"] == "voice-123"
    record = cfg["voice_id_versions"]["Hanadan"][0]
    assert record["voice_id"] == "voice-123"
    assert record["source"] == "local_upload"


def test_clone_voice_requires_api_key_and_reference(fake_env):
    surface = _surface(fake_env)

    with pytest.raises(ValueError, match="API KEY"):
        _run_action(
            surface,
            "clone_voice",
            {"provider": state.PROVIDER_SLUG, "character": "Hanadan"},
        )

    fake_env.extras[state.PROVIDER_SLUG]["api_key"] = "sk-ok"
    with pytest.raises(ValueError, match="参考音频"):
        _run_action(
            surface,
            "clone_voice",
            {"provider": state.PROVIDER_SLUG, "character": "Hanadan"},
        )

    with pytest.raises(ValueError, match="GPT SoVITS"):
        _run_action(
            surface,
            "clone_voice",
            {"provider": state.GPT_SOVITS_PROVIDER_SLUG, "character": "Hanadan"},
        )


def test_export_and_import_voice_roundtrip(fake_env):
    surface = _surface(fake_env)
    _run_action(
        surface,
        "bind_voice",
        {"provider": state.PROVIDER_SLUG, "character": "Hanadan", "voice_id": "v7"},
    )

    exported = _run_action(
        surface,
        "export_voice",
        {"provider": state.PROVIDER_SLUG, "character": "Hanadan"},
    )

    assert exported["payload"]["voice_id"] == "v7"
    assert exported["payload"]["type"] == "cloud_tts.voice_id"
    assert exported["filename"].endswith(".json")

    fake_env.configs[state.QWEN_PROVIDER_SLUG] = {}
    imported = _run_action(
        surface,
        "import_voice",
        {
            "provider": state.QWEN_PROVIDER_SLUG,
            "payload": exported["payload"],
            "character": "Hanadan",
        },
    )

    assert imported["imported"] >= 1
    qwen_cfg = fake_env.configs[state.QWEN_PROVIDER_SLUG]
    assert qwen_cfg["voice_id_map"]["Hanadan"] == "v7"


def test_export_voice_without_binding_raises(fake_env):
    with pytest.raises(ValueError, match="没有可导出"):
        _run_action(
            _surface(fake_env),
            "export_voice",
            {"provider": state.PROVIDER_SLUG, "character": "Hanadan"},
        )


def test_save_gpt_sovits_profile_stores_clean_profile(fake_env):
    result = _run_action(
        _surface(fake_env),
        "save_gpt_sovits_profile",
        {
            "character": "Hanadan",
            "profile": {
                "ref_audio_path": "/srv/ref.wav",
                "prompt_text": "  你好  ",
                "unknown_field": "dropped",
            },
        },
    )

    profile = result["character"]["gpt_sovits_profile"]
    assert profile == {"ref_audio_path": "/srv/ref.wav", "prompt_text": "你好"}
    cfg = fake_env.configs[state.GPT_SOVITS_PROVIDER_SLUG]
    assert cfg["gpt_sovits_character_profiles"]["Hanadan"] == profile


def test_constraint_actions_roundtrip(fake_env, monkeypatch):
    zh_vid = state.DEFAULT_PROMPT_VERSION_IDS["zh"]
    store = {
        "character_name": "Hanadan",
        "selected_version": zh_vid,
        "versions": {
            zh_vid: {"name": "中文版", "constraint_text": "旧文本", "source": "default"},
        },
    }
    upserts: list[tuple] = []
    monkeypatch.setattr(state, "load_character_constraints", lambda name: dict(store))
    monkeypatch.setattr(
        state,
        "upsert_constraint_version",
        lambda char, vid, text, **kw: upserts.append((char, vid, text, kw)),
    )
    monkeypatch.setattr(state, "select_constraint_version", lambda name, vid: True)
    propagated: list[str] = []
    monkeypatch.setattr(
        state,
        "propagate_default_template",
        lambda text, language=None: propagated.append(text) or 3,
    )
    surface = _surface(fake_env)

    loaded = _run_action(surface, "get_constraints", {"character": "Hanadan"})
    assert loaded["selected_version"] == zh_vid
    zh_row = next(v for v in loaded["versions"] if v["version_id"] == zh_vid)
    assert zh_row["constraint_text"] == "旧文本"

    saved = _run_action(
        surface,
        "save_constraints",
        {"character": "Hanadan", "version_id": zh_vid, "text": "新文本"},
    )
    assert upserts[-1][2] == "新文本"
    assert upserts[-1][3]["source"] == "manual"
    assert saved["synced_characters"] == 0
    assert not propagated

    _run_action(
        surface,
        "save_constraints",
        {"character": "默认模板", "version_id": zh_vid, "text": "母版文本"},
    )
    assert upserts[-1][3]["source"] == "default"
    assert propagated == ["母版文本"]
