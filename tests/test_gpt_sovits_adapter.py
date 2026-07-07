from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.cloud_tts import state
from plugins.cloud_tts import host_hook
from plugins.cloud_tts.gpt_sovits_adapter import GPTSoVITSApiAdapter


class DummyResponse:
    def __init__(self, content=b"RIFFfake", json_data=None, status_code=200):
        self.content = content
        self._json_data = json_data or {"ok": True}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(response=self)
        return None

    def json(self):
        return self._json_data


def test_state_recognizes_gpt_sovits_provider():
    assert state.GPT_SOVITS_PROVIDER_SLUG == "gpt-sovits-api"
    assert state.is_any_cloud_tts_provider("gpt-sovits-api")
    assert "v2ProPlus" in state.GPT_SOVITS_MODELS
    assert "v2Pro2025" in state.GPT_SOVITS_MODELS


def test_switch_model_uses_remote_profile_paths_without_suffix_filter(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, params, headers, timeout))
        return DummyResponse(json_data={"message": "ok"})

    import plugins.cloud_tts.gpt_sovits_adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "get", fake_get)

    adapter = GPTSoVITSApiAdapter(
        use_runtime_config=False,
        base_api_url="http://gsv.local:9880/",
        api_key="token-1",
        character_profiles={
            "華淡": {
                "gpt_weights_path": "/srv/models/gpt/hanadan-v4.ckpt",
                "sovits_weights_path": "/srv/models/sovits/hanadan-s2v4.ckpt",
            }
        },
    )

    assert adapter.switch_model({"character_name": "華淡"}) is True

    assert calls == [
        (
            "http://gsv.local:9880/set_gpt_weights",
            {"weights_path": "/srv/models/gpt/hanadan-v4.ckpt"},
            {"Authorization": "Bearer token-1"},
            30,
        ),
        (
            "http://gsv.local:9880/set_sovits_weights",
            {"weights_path": "/srv/models/sovits/hanadan-s2v4.ckpt"},
            {"Authorization": "Bearer token-1"},
            30,
        ),
    ]


def test_generate_speech_posts_v4_request_with_server_reference(monkeypatch, tmp_path):
    posts = []

    def fake_post(url, json=None, headers=None, timeout=None):
        posts.append((url, json, headers, timeout))
        return DummyResponse(content=b"WAVEFORM")

    import plugins.cloud_tts.gpt_sovits_adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "post", fake_post)

    adapter = GPTSoVITSApiAdapter(
        use_runtime_config=False,
        base_api_url="http://gsv.local:9880",
        model="v4",
        api_key="token-2",
        media_type="wav",
        text_split_method="cut5",
        sample_steps=24,
        super_sampling=True,
        character_profiles={
            "華淡": {
                "ref_audio_path": "/srv/refs/hanadan.wav",
                "prompt_text": "前辈，观测开始。",
                "prompt_lang": "zh",
                "text_lang": "zh",
            }
        },
    )

    output_path = adapter.generate_speech(
        "测试文本",
        file_path=tmp_path / "speech.wav",
        character_name="華淡",
        ref_audio_path="C:/local/should-not-be-used.wav",
        prompt_text="main prompt should not win",
        prompt_lang="ja",
        speed_factor=1.25,
    )

    assert Path(output_path).read_bytes() == b"WAVEFORM"
    assert posts == [
        (
            "http://gsv.local:9880/tts",
            {
                "text": "测试文本",
                "text_lang": "zh",
                "ref_audio_path": "/srv/refs/hanadan.wav",
                "prompt_text": "前辈，观测开始。",
                "prompt_lang": "zh",
                "media_type": "wav",
                "streaming_mode": False,
                "text_split_method": "cut5",
                "batch_size": 1,
                "batch_threshold": 0.75,
                "split_bucket": True,
                "return_fragment": False,
                "speed_factor": 1.25,
                "fragment_interval": 0.3,
                "seed": -1,
                "parallel_infer": True,
                "repetition_penalty": 1.35,
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "sample_steps": 24,
                "super_sampling": True,
            },
            {"Authorization": "Bearer token-2"},
            120,
        )
    ]


def test_generate_speech_requires_gpt_sovits_profile_reference(monkeypatch, tmp_path):
    posts = []

    def fake_post(url, json=None, headers=None, timeout=None):
        posts.append((url, json, headers, timeout))
        return DummyResponse(content=b"SHOULD_NOT_WRITE")

    import plugins.cloud_tts.gpt_sovits_adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "post", fake_post)

    adapter = GPTSoVITSApiAdapter(
        use_runtime_config=False,
        base_api_url="http://gsv.local:9880",
        local_reference_audio_map={"??": "C:/local/minimax-ref.wav"},
    )

    output_path = adapter.generate_speech(
        "????",
        file_path=tmp_path / "speech.wav",
        character_name="??",
        ref_audio_path="C:/main/form/path.wav",
        prompt_text="main prompt",
        prompt_lang="zh",
    )

    assert output_path is None
    assert posts == []
    assert not (tmp_path / "speech.wav").exists()


def test_generate_speech_stops_after_failed_weight_switch(monkeypatch, tmp_path):
    posts = []

    def fake_get(url, params=None, headers=None, timeout=None):
        raise RuntimeError("weights missing")

    def fake_post(url, json=None, headers=None, timeout=None):
        posts.append((url, json, headers, timeout))
        return DummyResponse(content=b"SHOULD_NOT_WRITE")

    import plugins.cloud_tts.gpt_sovits_adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "get", fake_get)
    monkeypatch.setattr(adapter_module.requests, "post", fake_post)

    adapter = GPTSoVITSApiAdapter(
        use_runtime_config=False,
        base_api_url="http://gsv.local:9880",
        character_profiles={
            "??": {
                "gpt_weights_path": "/missing/gpt.ckpt",
                "sovits_weights_path": "/missing/sovits.ckpt",
                "ref_audio_path": "/srv/refs/hanadan.wav",
                "prompt_text": "???",
                "prompt_lang": "zh",
            }
        },
    )

    assert adapter.switch_model({"character_name": "??"}) is False
    output_path = adapter.generate_speech(
        "????",
        file_path=tmp_path / "speech.wav",
        character_name="??",
    )

    assert output_path is None
    assert posts == []
    assert not (tmp_path / "speech.wav").exists()


def test_switch_model_falls_back_to_legacy_set_model_when_api_v2_weights_missing(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(("GET", url, params, headers, timeout))
        if url.endswith("/set_gpt_weights"):
            return DummyResponse(status_code=404)
        return DummyResponse()

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(("POST", url, json, headers, timeout))
        return DummyResponse()

    import plugins.cloud_tts.gpt_sovits_adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "get", fake_get)
    monkeypatch.setattr(adapter_module.requests, "post", fake_post)

    adapter = GPTSoVITSApiAdapter(
        use_runtime_config=False,
        base_api_url="http://gsv.local:9880",
        api_key="token-3",
        character_profiles={
            "華淡": {
                "gpt_weights_path": "/srv/models/gpt/hanadan-v2pro2025.ckpt",
                "sovits_weights_path": "/srv/models/sovits/hanadan-v2pro2025.pth",
            }
        },
    )

    assert adapter.switch_model({"character_name": "華淡"}) is True

    assert calls == [
        (
            "GET",
            "http://gsv.local:9880/set_gpt_weights",
            {"weights_path": "/srv/models/gpt/hanadan-v2pro2025.ckpt"},
            {"Authorization": "Bearer token-3"},
            30,
        ),
        (
            "POST",
            "http://gsv.local:9880/set_model",
            {
                "gpt_model_path": "/srv/models/gpt/hanadan-v2pro2025.ckpt",
                "sovits_model_path": "/srv/models/sovits/hanadan-v2pro2025.pth",
            },
            {"Authorization": "Bearer token-3"},
            30,
        ),
    ]


def test_generate_speech_falls_back_to_legacy_root_tts_when_api_v2_tts_missing(monkeypatch, tmp_path):
    posts = []

    def fake_post(url, json=None, headers=None, timeout=None):
        posts.append((url, json, headers, timeout))
        if url.endswith("/tts"):
            return DummyResponse(status_code=404)
        return DummyResponse(content=b"LEGACY_WAVE")

    import plugins.cloud_tts.gpt_sovits_adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "post", fake_post)

    adapter = GPTSoVITSApiAdapter(
        use_runtime_config=False,
        base_api_url="http://gsv.local:9880",
        api_key="token-4",
        media_type="wav",
        character_profiles={
            "華淡": {
                "ref_audio_path": "/srv/refs/hanadan.wav",
                "prompt_text": "前辈，观测开始。",
                "prompt_lang": "zh",
                "text_lang": "zh",
            }
        },
    )

    output_path = adapter.generate_speech(
        "测试文本",
        file_path=tmp_path / "speech.wav",
        character_name="華淡",
        speed_factor=1.25,
    )

    assert Path(output_path).read_bytes() == b"LEGACY_WAVE"
    assert posts == [
        (
            "http://gsv.local:9880/tts",
            {
                "text": "测试文本",
                "text_lang": "zh",
                "ref_audio_path": "/srv/refs/hanadan.wav",
                "prompt_text": "前辈，观测开始。",
                "prompt_lang": "zh",
                "media_type": "wav",
                "streaming_mode": False,
                "text_split_method": "cut5",
                "batch_size": 1,
                "batch_threshold": 0.75,
                "split_bucket": True,
                "return_fragment": False,
                "speed_factor": 1.25,
                "fragment_interval": 0.3,
                "seed": -1,
                "parallel_infer": True,
                "repetition_penalty": 1.35,
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "sample_steps": 32,
                "super_sampling": False,
            },
            {"Authorization": "Bearer token-4"},
            120,
        ),
        (
            "http://gsv.local:9880/",
            {
                "refer_wav_path": "/srv/refs/hanadan.wav",
                "prompt_text": "前辈，观测开始。",
                "prompt_language": "zh",
                "text": "测试文本",
                "text_language": "zh",
                "cut_punc": "",
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "speed": 1.25,
                "inp_refs": None,
                "sample_steps": 32,
                "if_sr": False,
            },
            {"Authorization": "Bearer token-4"},
            120,
        ),
    ]


def test_gpt_sovits_display_name_is_cloud():
    plugin_root = Path(__file__).resolve().parents[1]
    checked = [
        plugin_root / "settings.py",
        plugin_root / "README.md",
        plugin_root / "CHANGELOG.md",
        plugin_root / "gpt_sovits_adapter.py",
        plugin_root / "plugin.py",
    ]
    for path in checked:
        content = path.read_text(encoding="utf-8")
        assert "GPT SoVITS Cloud" in content
        assert "GPT-SoVITS API" not in content

def test_plugin_description_mentions_gpt_sovits_cloud():
    plugin_text = (Path(__file__).resolve().parents[1] / "plugin.py").read_text(
        encoding="utf-8"
    )

    assert "GPT SoVITS Cloud" in plugin_text
    assert "per-character" in plugin_text
    assert "server-side reference audio" in plugin_text
    assert "model paths" in plugin_text
    assert "GPT-SoVITS API" not in plugin_text

def test_settings_guide_mentions_gpt_sovits_cloud_usage():
    settings_text = (Path(__file__).resolve().parents[1] / "settings.py").read_text(
        encoding="utf-8"
    )

    assert "<b>4. GPT SoVITS Cloud 模式：</b>" in settings_text
    assert "服务器路径" in settings_text
    assert "GSV 参考音频" in settings_text
    assert "GPT / SoVITS 模型路径" in settings_text
    assert "/data/models/gpt/hanadan-gpt.ckpt" in settings_text
    assert "/data/models/sovits/hanadan-sovits.pth" in settings_text


def test_docs_explain_gpt_sovits_api_py_and_api_v2_compatibility():
    plugin_root = Path(__file__).resolve().parents[1]
    readme = (plugin_root / "README.md").read_text(encoding="utf-8")
    changelog = (plugin_root / "CHANGELOG.md").read_text(encoding="utf-8")

    for content in (readme, changelog):
        assert "api.py" in content
        assert "api_v2.py" in content
        assert "/set_model" in content
        assert "/set_gpt_weights" in content
    assert "主 API 页 Provider" in readme
    assert "服务端只有 `/set_model`" in readme
    assert "三个 TTS 引擎" in readme
    assert "/data/models/gpt/hanadan-gpt.ckpt" in readme
    assert "/data/models/sovits/hanadan-sovits.pth" in readme


def test_host_hook_labels_main_api_tts_provider_as_cloud():
    prefs = (
        ("genie-tts", "Genie TTS"),
        (state.GPT_SOVITS_PROVIDER_SLUG, "Gpt Sovits Api"),
    )

    patched = host_hook._with_gpt_sovits_cloud_label(prefs)
    repatched = host_hook._with_gpt_sovits_cloud_label(patched)

    assert patched[0] == (state.GPT_SOVITS_PROVIDER_SLUG, "GPT SoVITS Cloud")
    assert repatched == patched
    assert [
        label for slug, label in patched if slug == state.GPT_SOVITS_PROVIDER_SLUG
    ] == ["GPT SoVITS Cloud"]



def test_generate_speech_uses_caller_file_path_verbatim(monkeypatch, tmp_path):
    def fake_post(url, json=None, headers=None, timeout=None):
        return DummyResponse(content=b"WAVEFORM")

    import plugins.cloud_tts.gpt_sovits_adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "post", fake_post)

    adapter = GPTSoVITSApiAdapter(
        use_runtime_config=False,
        base_api_url="http://gsv.local:9880",
        model="v4",
        media_type="wav",
        character_profiles={
            "華淡": {
                "ref_audio_path": "/srv/refs/hanadan.wav",
                "prompt_text": "前辈，观测开始。",
                "prompt_lang": "zh",
                "text_lang": "zh",
            }
        },
    )

    # 宿主传入 .part 临时路径：必须原样写入，不得“纠正”扩展名
    part_path = tmp_path / "3.wav.part"

    output_path = adapter.generate_speech(
        "测试文本",
        file_path=part_path,
        character_name="華淡",
    )

    assert Path(output_path) == part_path.resolve()
    assert part_path.read_bytes() == b"WAVEFORM"
    assert not (tmp_path / "3.wav.wav").exists()
