from pathlib import Path
import importlib.util
import sys
import types

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HOST_ROOT = next(
    (item for item in PLUGIN_ROOT.parents if (item / "plugins").is_dir()),
    PLUGIN_ROOT.parent,
)
if str(HOST_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_ROOT))
if "plugins.cloud_tts" not in sys.modules:
    plugins_pkg = sys.modules.setdefault("plugins", types.ModuleType("plugins"))
    plugins_pkg.__path__ = [str(HOST_ROOT / "plugins")]  # type: ignore[attr-defined]
    cloud_pkg = types.ModuleType("plugins.cloud_tts")
    cloud_pkg.__path__ = [str(PLUGIN_ROOT)]  # type: ignore[attr-defined]
    sys.modules["plugins.cloud_tts"] = cloud_pkg
try:
    sdk_spec = importlib.util.find_spec("sdk.adapters")
except ModuleNotFoundError:
    sdk_spec = None
if sdk_spec is None:
    sdk_pkg = sys.modules.setdefault("sdk", types.ModuleType("sdk"))
    sdk_pkg.__path__ = []  # type: ignore[attr-defined]
    adapters_pkg = types.ModuleType("sdk.adapters")

    class TTSAdapter:
        pass

    adapters_pkg.TTSAdapter = TTSAdapter
    sys.modules["sdk.adapters"] = adapters_pkg
from plugins.cloud_tts.adapter import CloudTTSAdapter


class DummyResponse:
    def __init__(self, content=b"", json_data=None, status_code=200):
        self.content = content
        self._json_data = json_data or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(response=self)
        return None

    def json(self):
        return self._json_data


def test_minimax_generate_speech_downloads_url_audio(monkeypatch, tmp_path):
    calls = []
    audio_url = "https://audio.example.test/generated.mp3"
    audio_bytes = b"fake mp3 bytes"

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(("POST", url, headers, json, timeout))
        return DummyResponse(
            json_data={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": audio_url, "status": 2},
                "extra_info": {"usage_characters": 5},
            }
        )

    def fake_get(url, headers=None, timeout=None):
        calls.append(("GET", url, headers, None, timeout))
        return DummyResponse(content=audio_bytes)

    import plugins.cloud_tts.adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "post", fake_post)
    monkeypatch.setattr(adapter_module.requests, "get", fake_get)

    adapter = CloudTTSAdapter(
        use_runtime_config=False,
        api_key="user-platform-key",
        base_api_url="https://api.example.test/v1",
        default_voice_id="voice-1",
        audio_format="mp3",
        request_timeout=17,
    )

    out_path = tmp_path / "speech.mp3"

    result = adapter.generate_speech("hello", file_path=out_path)

    assert result == str(out_path.resolve())
    assert out_path.read_bytes() == audio_bytes
    assert calls[0][0] == "POST"
    assert calls[0][1] == "https://api.example.test/v1/t2a_v2"
    assert calls[0][3]["output_format"] == "url"
    assert calls[1] == ("GET", audio_url, None, None, 17)


def test_minimax_voice_clone_downloads_demo_audio(monkeypatch, tmp_path):
    calls = []
    demo_audio_url = "https://audio.example.test/clone-demo.mp3"
    demo_audio_bytes = b"fake clone demo mp3"
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"fake wav bytes")

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(("POST", url, headers, json, timeout))
        return DummyResponse(
            json_data={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "demo_audio": demo_audio_url,
                "input_sensitive": False,
                "input_sensitive_type": 0,
            }
        )

    def fake_get(url, headers=None, timeout=None):
        calls.append(("GET", url, headers, None, timeout))
        return DummyResponse(content=demo_audio_bytes)

    import plugins.cloud_tts.adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "post", fake_post)
    monkeypatch.setattr(adapter_module.requests, "get", fake_get)

    adapter = CloudTTSAdapter(
        use_runtime_config=False,
        api_key="user-platform-key",
        base_api_url="https://api.example.test/v1",
        model="speech-2.8-hd",
        voice_cache_path=str(tmp_path / "voice_cache.json"),
        clone_demo_audio_dir=str(tmp_path / "clone_demos"),
        request_timeout=17,
    )
    monkeypatch.setattr(adapter, "_prepare_clone_upload_audio", lambda path: path)
    monkeypatch.setattr(adapter, "_upload_file", lambda path, purpose: 123456)

    voice_id = adapter.create_cloned_voice_from_file(
        reference_audio,
        prompt_text="试听文本",
        voice_id="hanadan-demo-voice",
    )

    assert voice_id == "hanadan-demo-voice"
    demo_path = Path(adapter.last_clone_demo_audio_path)
    assert demo_path.is_file()
    assert demo_path.read_bytes() == demo_audio_bytes
    assert demo_path.parent == tmp_path / "clone_demos"
    assert calls[0][1] == "https://api.example.test/v1/voice_clone"
    assert calls[0][3]["text"] == "试听文本"
    assert calls[1] == ("GET", demo_audio_url, None, None, 17)


def test_minimax_voice_clone_http_error_includes_server_detail(monkeypatch, tmp_path):
    reference_audio = tmp_path / "short.wav"
    reference_audio.write_bytes(b"fake short wav bytes")

    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse(
            status_code=400,
            json_data={
                "base_resp": {
                    "status_code": 1002,
                    "status_msg": "reference audio is shorter than 10 seconds",
                }
            },
        )

    import plugins.cloud_tts.adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "post", fake_post)

    adapter = CloudTTSAdapter(
        use_runtime_config=False,
        api_key="user-platform-key",
        base_api_url="https://api.example.test/v1",
        model="speech-2.8-hd",
        voice_cache_path=str(tmp_path / "voice_cache.json"),
        clone_demo_audio_dir=str(tmp_path / "clone_demos"),
        request_timeout=17,
    )
    monkeypatch.setattr(adapter, "_prepare_clone_upload_audio", lambda path: path)
    monkeypatch.setattr(adapter, "_upload_file", lambda path, purpose: 123456)

    with pytest.raises(RuntimeError) as excinfo:
        adapter.create_cloned_voice_from_file(
            reference_audio,
            prompt_text="demo text",
            voice_id="short-demo-voice",
        )

    assert "reference audio is shorter than 10 seconds" in str(excinfo.value)


def test_minimax_generate_speech_uses_caller_file_path_verbatim(monkeypatch, tmp_path):
    audio_url = "https://audio.example.test/generated.wav"
    audio_bytes = b"fake wav bytes"

    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse(
            json_data={
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "data": {"audio": audio_url, "status": 2},
                "extra_info": {"usage_characters": 5},
            }
        )

    def fake_get(url, headers=None, timeout=None):
        return DummyResponse(content=audio_bytes)

    import plugins.cloud_tts.adapter as adapter_module

    monkeypatch.setattr(adapter_module.requests, "post", fake_post)
    monkeypatch.setattr(adapter_module.requests, "get", fake_get)

    adapter = CloudTTSAdapter(
        use_runtime_config=False,
        api_key="user-platform-key",
        base_api_url="https://api.example.test/v1",
        default_voice_id="voice-1",
        audio_format="wav",
        request_timeout=17,
    )

    # 宿主传入 .part 临时路径：必须原样写入，不得“纠正”扩展名
    part_path = tmp_path / "3.wav.part"

    result = adapter.generate_speech("hello", file_path=part_path)

    assert result == str(part_path.resolve())
    assert part_path.read_bytes() == audio_bytes
    assert not (tmp_path / "3.wav.wav").exists()
