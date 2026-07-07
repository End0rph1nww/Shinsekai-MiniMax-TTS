from __future__ import annotations

import ast
import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from sdk.adapters import TTSAdapter

from plugins.cloud_tts import state


VALID_MODELS = (
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
)
VALID_LANGUAGE_BOOSTS = (
    "Chinese",
    "Chinese,Yue",
    "English",
    "Arabic",
    "Russian",
    "Spanish",
    "French",
    "Portuguese",
    "German",
    "Turkish",
    "Dutch",
    "Ukrainian",
    "Vietnamese",
    "Indonesian",
    "Japanese",
    "Italian",
    "Korean",
    "Thai",
    "Polish",
    "Romanian",
    "Greek",
    "Czech",
    "Finnish",
    "Hindi",
    "Bulgarian",
    "Danish",
    "Hebrew",
    "Malay",
    "Persian",
    "Slovak",
    "Swedish",
    "Croatian",
    "Filipino",
    "Hungarian",
    "Norwegian",
    "Slovenian",
    "Catalan",
    "Nynorsk",
    "Tamil",
    "Afrikaans",
    "auto",
)
VALID_AUDIO_FORMATS = ("mp3", "pcm", "flac", "wav")
VALID_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100)
VALID_BITRATES = (32000, 64000, 128000, 256000)
VALID_CHANNELS = (1, 2)
VALID_EMOTIONS = (
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
    "calm",
    "fluent",
)


class CloudTTSAdapter(TTSAdapter):
    """MiniMax synchronous T2A adapter with voice clone cache support."""

    _CLONE_UPLOAD_SUFFIXES = {".mp3", ".m4a", ".wav"}

    def __init__(
        self,
        api_key: str = "",
        base_api_url: str = "https://api.minimaxi.com/v1",
        model: str = "speech-2.8-hd",
        default_voice_id: str = "",
        voice_id_map: dict[str, str] | str | None = None,
        voice_id_versions: dict[str, Any] | str | None = None,
        voice_language_map: dict[str, Any] | str | None = None,
        voice_cache_path: str = "cache/audio/cloud_tts_voice_cache.json",
        clone_demo_audio_dir: str = "cache/audio/cloud_tts_clone_demo",
        local_reference_audio_map: dict[str, str] | str | None = None,
        reference_audio_language_map: dict[str, Any] | str | None = None,
        language_boost: str = "auto",
        audio_format: str = "wav",
        sample_rate: int = 32000,
        bitrate: int = 128000,
        channel: int = 1,
        speed: float = 1.0,
        vol: float = 1.0,
        pitch: int = 0,
        emotion: str = "",
        request_timeout: int = 120,
        auto_clone_from_reference: bool = False,
        need_noise_reduction: bool = False,
        need_volume_normalization: bool = False,
        use_runtime_config: bool = True,
        **_ignored_kwargs: Any,
    ) -> None:
        # 官方链路：adapter 参数由 api.yaml 的 tts_extra_configs 注入。
        # 这里仍读取插件状态，是为了兼容旧版 config.json 与角色 voice_id 文件。
        runtime_cfg: dict[str, Any] = {}
        if use_runtime_config:
            plugin_cfg = state.load_runtime_plugin_config()
            runtime_cfg = dict(plugin_cfg)
            runtime_cfg.update(state.get_cloud_extra())
        if runtime_cfg:
            api_key = runtime_cfg.get("api_key", api_key)
            base_api_url = runtime_cfg.get("base_api_url", base_api_url)
            model = runtime_cfg.get("model", model)
            default_voice_id = runtime_cfg.get(
                "default_voice_id",
                runtime_cfg.get("voice_id", default_voice_id),
            )
            voice_id_map = runtime_cfg.get("voice_id_map", voice_id_map)
            voice_id_versions = runtime_cfg.get("voice_id_versions", voice_id_versions)
            local_reference_audio_map = runtime_cfg.get(
                "local_reference_audio_map",
                local_reference_audio_map,
            )
            reference_audio_language_map = runtime_cfg.get(
                "reference_audio_language_map",
                reference_audio_language_map,
            )
        self.api_key = self._normalize_api_key(api_key)
        self.base_api_url = (base_api_url or "https://api.minimaxi.com/v1").rstrip("/")
        self.model = self._normalize_choice(model, VALID_MODELS, "speech-2.8-hd")
        self.default_voice_id = (default_voice_id or "").strip()
        self.voice_id_map = self._coerce_voice_id_map(voice_id_map)
        self.voice_id_versions = voice_id_versions
        self.voice_language_map = state.coerce_voice_language_map(voice_language_map)
        self.voice_cache_path = state.project_path(voice_cache_path)
        self.clone_demo_audio_dir = state.project_path(clone_demo_audio_dir)
        self.local_reference_audio_map = self._coerce_local_reference_audio_map(
            local_reference_audio_map
        )
        self.reference_audio_language_map = state.coerce_voice_language_map(
            reference_audio_language_map
        )
        self.language_boost = self._normalize_choice(language_boost, VALID_LANGUAGE_BOOSTS, "auto")
        self.audio_format = self._normalize_choice(audio_format, VALID_AUDIO_FORMATS, "wav")
        self.sample_rate = self._normalize_sample_rate(sample_rate)
        self.bitrate = self._normalize_numeric_choice(bitrate, VALID_BITRATES, 128000)
        self.channel = self._normalize_numeric_choice(channel, VALID_CHANNELS, 1)
        self.speed = self._clamp_float(speed, 1.0, 0.5, 2.0)
        self.vol = self._clamp_float(vol, 1.0, 0.0, 10.0)
        self.pitch = self._clamp_int(pitch, 0, -12, 12)
        self.emotion = self._normalize_emotion(emotion)
        self.request_timeout = int(request_timeout or 120)
        self.auto_clone_from_reference = bool(auto_clone_from_reference)
        self.need_noise_reduction = bool(need_noise_reduction)
        self.need_volume_normalization = bool(need_volume_normalization)
        self.last_clone_demo_audio_url = ""
        self.last_clone_demo_audio_path = ""

    @classmethod
    def get_config_schema(cls) -> dict[str, dict]:
        # API 页只放连接凭证；MiniMax 行为参数集中在插件设置页。
        return {
            "api_key": {
                "type": "str",
                "label": "MiniMax API KEY",
                "default": "",
                "secret": True,
            },
            "base_api_url": {
                "type": "str",
                "label": "MiniMax Base URL",
                "default": "https://api.minimaxi.com/v1",
            },
        }

    def switch_model(self, model_info: Any) -> None:
        if isinstance(model_info, dict):
            vid = str(
                model_info.get("cloud_voice_id")
                or model_info.get("minimax_voice_id")
                or ""
            ).strip()
            if vid:
                name = str(model_info.get("character_name") or "").strip()
                if name:
                    self.voice_id_map[name] = vid
                else:
                    self.default_voice_id = vid

    def _log(self, message: str) -> None:
        print(f"Cloud TTS\uff1a{message}", flush=True)

    def generate_speech(self, text, file_path=None, **kwargs):
        api_key_error = self._api_key_error()
        if api_key_error:
            self._log(f"\u5408\u6210\u5931\u8d25\uff1a{api_key_error}")
            return None
        text_value = str(text or "")
        character_name = str(kwargs.get("character_name") or "").strip()
        if not character_name:
            character_name = "\u672a\u6307\u5b9a\u89d2\u8272"
        self._log(
            f"\u5f00\u59cb\u5408\u6210\uff1a\u89d2\u8272={character_name}\uff0c"
            f"\u6587\u672c\u957f\u5ea6={len(text_value)}\uff0c\u6a21\u578b={self.model}"
        )
        # A caller-supplied file_path must be used verbatim: the host hands us a
        # ".part" temp path and renames it after validating, so "fixing" the
        # suffix here makes the host miss the file entirely.
        out_path = Path(file_path or f"cache/audio/cloud_tts_{int(time.time() * 1000)}.{self.audio_format}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        voice_id = self._voice_id_for_request(**kwargs)
        if not voice_id:
            self._log(
                "\u5408\u6210\u5931\u8d25\uff1a\u6ca1\u6709\u53ef\u7528 voice_id\uff0c"
                "\u4e5f\u6ca1\u6709\u53ef\u81ea\u52a8\u514b\u9686\u7684\u53c2\u8003\u97f3\u9891\u3002"
            )
            return None

        speed = kwargs.get("speed_factor")
        speed_value = self._clamp_float(
            speed if speed is not None else self.speed,
            self.speed,
            0.5,
            2.0,
        )

        language_boost = self._language_boost_for(
            kwargs.get("text_lang"),
            character_name=character_name,
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "text": text_value,
            "stream": False,
            "language_boost": language_boost,
            "output_format": "url",
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed_value,
                "vol": self.vol,
                "pitch": self.pitch,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": self.bitrate,
                "format": self.audio_format,
                "channel": self.channel,
            },
            "subtitle_enable": False,
        }
        if self.emotion:
            payload["voice_setting"]["emotion"] = self.emotion

        try:
            self._log(
                "\u5408\u6210\u53c2\u6570\uff1a"
                f"voice_id={voice_id}\uff0c\u8bed\u8a00\u589e\u5f3a={language_boost}\uff0c"
                f"\u683c\u5f0f={self.audio_format}\uff0c"
                f"\u91c7\u6837\u7387={self.sample_rate}\uff0c\u8bed\u901f={speed_value:.2f}"
            )
            self._log(
                "\u6b63\u5728\u8bf7\u6c42 MiniMax \u6587\u751f\u97f3\u63a5\u53e3 /t2a_v2 ..."
            )
            resp = requests.post(
                f"{self.base_api_url}/t2a_v2",
                headers=self._json_headers(),
                json=payload,
                timeout=self.request_timeout,
            )
            data = self._json_response(resp, "/t2a_v2")
            self._raise_for_base_resp(data)
            audio = (data.get("data") or {}).get("audio")
            if not audio:
                raise RuntimeError("MiniMax returned empty audio.")
            self._log(
                "\u63a5\u53e3\u8fd4\u56de\u6210\u529f\uff0c"
                "\u6b63\u5728\u89e3\u7801\u5e76\u5199\u5165\u97f3\u9891\u6587\u4ef6..."
            )
            out_path.write_bytes(self._audio_bytes(str(audio)))
            abs_path = os.path.abspath(out_path)
            self._log(f"\u5408\u6210\u5b8c\u6210\uff1a{abs_path}")
            return abs_path
        except Exception as exc:
            self._log(f"\u5408\u6210\u5931\u8d25\uff1a{exc}")
            return None

    def create_cloned_voice_from_file(
        self,
        audio_path: str | Path,
        *,
        character_name: str = "",
        prompt_text: str = "",
        voice_id: str = "",
        reference_audio_language: str = "auto",
    ) -> str:
        self._ensure_api_key()
        path = state.project_path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        chosen_voice_id = self._normalize_voice_id(voice_id) if voice_id else self._new_voice_id(path, character_name)
        display_name = character_name or "\u672a\u547d\u540d\u89d2\u8272"
        self.last_clone_demo_audio_url = ""
        self.last_clone_demo_audio_path = ""
        self._log(
            f"\u5f00\u59cb\u58f0\u7ebf\u514b\u9686\uff1a\u89d2\u8272={display_name}\uff0c"
            f"voice_id={chosen_voice_id}"
        )
        upload_path = self._prepare_clone_upload_audio(path)
        file_id = self._upload_file(upload_path, "voice_clone")
        payload: dict[str, Any] = {
            "file_id": file_id,
            "voice_id": chosen_voice_id,
            "model": self.model,
            "need_noise_reduction": self.need_noise_reduction,
            "need_volume_normalization": self.need_volume_normalization,
        }
        prompt_text = str(prompt_text or "").strip()
        if prompt_text:
            payload["text"] = prompt_text
        clone_language_boost = self._language_boost_from_code(reference_audio_language)
        if clone_language_boost:
            payload["language_boost"] = clone_language_boost
        self._log(
            "\u6b63\u5728\u521b\u5efa MiniMax \u514b\u9686\u58f0\u7ebf /voice_clone ..."
        )
        resp = requests.post(
            f"{self.base_api_url}/voice_clone",
            headers=self._json_headers(),
            json=payload,
            timeout=self.request_timeout,
        )
        data = self._json_response(resp, "/voice_clone")
        self._raise_for_base_resp(data)
        demo_audio = str(data.get("demo_audio") or "").strip()
        if demo_audio.startswith(("http://", "https://")):
            self.last_clone_demo_audio_url = demo_audio
            try:
                demo_path = self._download_clone_demo_audio(demo_audio, chosen_voice_id)
            except Exception as exc:
                self._log(f"\u58f0\u7ebf\u514b\u9686\u8bd5\u542c\u97f3\u9891\u4e0b\u8f7d\u5931\u8d25\uff1a{exc}")
            else:
                self.last_clone_demo_audio_path = str(demo_path.resolve())
                self._log(f"\u58f0\u7ebf\u514b\u9686\u8bd5\u542c\u97f3\u9891\u5df2\u4fdd\u5b58\uff1a{self.last_clone_demo_audio_path}")
        self._cache_voice(path, character_name, chosen_voice_id)
        if character_name:
            state.upsert_voice_record(
                character_name,
                chosen_voice_id,
                {
                    "source": "auto_clone",
                    "model": self.model,
                    "reference_audio_path": str(path),
                    "reference_audio_language": state.normalize_voice_language_code(
                        reference_audio_language
                    ),
                },
                selected=True,
                provider_slug=state.PROVIDER_SLUG,
            )
        self._log(
            f"\u58f0\u7ebf\u514b\u9686\u5b8c\u6210\uff0c"
            f"\u5e76\u5df2\u5199\u5165\u672c\u5730\u7f13\u5b58\uff1a{chosen_voice_id}"
        )
        return chosen_voice_id

    def _json_headers(self) -> dict[str, str]:
        return {
            **self._auth_headers(),
            "Content-Type": "application/json",
        }

    def _auth_headers(self) -> dict[str, str]:
        self._ensure_api_key()
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _normalize_api_key(value: Any) -> str:
        key = str(value or "").strip()
        if key.lower().startswith("bearer "):
            key = key[7:].strip()
        return key

    def _api_key_error(self) -> str:
        if not self.api_key:
            return "API KEY \u4e3a\u7a7a\uff0c\u8bf7\u5148\u5728\u4e3b\u83dc\u5355 API \u8bbe\u7f6e\u9875\u586b\u5199 MiniMax API KEY \u5e76\u4fdd\u5b58\u3002"
        try:
            self.api_key.encode("ascii")
        except UnicodeEncodeError:
            return (
                "API KEY \u5305\u542b\u975e ASCII \u5b57\u7b26\uff0c\u50cf\u662f\u7c98\u8d34\u4e86\u4e2d\u6587\u63d0\u793a\u6216\u8fd0\u884c\u65e5\u5fd7\uff0c"
                "\u8bf7\u5728\u4e3b\u83dc\u5355 API \u8bbe\u7f6e\u9875\u91cd\u65b0\u586b\u5199 MiniMax API KEY \u5e76\u4fdd\u5b58\u3002"
            )
        return ""

    def _ensure_api_key(self) -> None:
        error = self._api_key_error()
        if error:
            raise RuntimeError(error)

    def _upload_file(self, path: Path, purpose: str) -> int:
        self._log(f"\u6b63\u5728\u4e0a\u4f20\u53c2\u8003\u97f3\u9891\uff1a{path.name}")
        with path.open("rb") as f:
            resp = requests.post(
                f"{self.base_api_url}/files/upload",
                headers=self._auth_headers(),
                data={"purpose": purpose},
                files={"file": (path.name, f)},
                timeout=self.request_timeout,
            )
        data = self._json_response(resp, "/files/upload")
        self._raise_for_base_resp(data)
        file_id = (data.get("file") or {}).get("file_id")
        if file_id is None:
            raise RuntimeError("MiniMax upload response missing file_id.")
        self._log(f"\u53c2\u8003\u97f3\u9891\u4e0a\u4f20\u5b8c\u6210\uff1afile_id={file_id}")
        return int(file_id)

    def _prepare_clone_upload_audio(self, path: Path) -> Path:
        ffmpeg = self._find_ffmpeg()
        if ffmpeg is None:
            if (
                path.suffix.lower() in self._CLONE_UPLOAD_SUFFIXES
                and path.stat().st_size <= 20_000_000
            ):
                self._log(
                    "\u672a\u627e\u5230 ffmpeg\uff0c\u53c2\u8003\u97f3\u9891"
                    "\u683c\u5f0f\u548c\u5927\u5c0f\u5df2\u7b26\u5408\u8981\u6c42\uff0c"
                    "\u76f4\u63a5\u4e0a\u4f20\u3002"
                )
                return path
            raise RuntimeError(
                "Reference audio needs conversion. Install plugin dependencies with: "
                "runtime\\python.exe -m pip install -r plugins\\cloud_tts\\requirements.txt"
            )

        self._log(
            "\u6b63\u5728\u628a\u53c2\u8003\u97f3\u9891\u8f6c\u6362\u4e3a "
            "MiniMax \u53ef\u63a5\u53d7\u7684 wav \u683c\u5f0f..."
        )
        out_dir = state.project_path("cache/audio/cloud_tts_upload")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_{state.short_hash(str(path) + str(path.stat().st_mtime_ns), 10)}.wav"
        cmd = [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-t",
            "300",
            "-ac",
            "1",
            "-ar",
            "32000",
            "-sample_fmt",
            "s16",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"Failed to convert reference audio with ffmpeg: {detail}") from exc
        if not out_path.is_file() or out_path.stat().st_size <= 0:
            raise RuntimeError("Converted reference audio is empty.")
        if out_path.stat().st_size > 20_000_000:
            raise RuntimeError("Converted reference audio is still larger than 20 MB.")
        self._log(f"\u53c2\u8003\u97f3\u9891\u8f6c\u6362\u5b8c\u6210\uff1a{out_path}")
        return out_path

    def _find_ffmpeg(self) -> Path | None:
        try:
            import imageio_ffmpeg
        except ImportError:
            return None
        item = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if item.is_file():
            return item
        return None

    def _raise_for_base_resp(self, data: dict[str, Any]) -> None:
        base = data.get("base_resp") or {}
        code = base.get("status_code", 0)
        if code not in (0, "0", None):
            raise RuntimeError(f"{code}: {base.get('status_msg', 'MiniMax API error')}")

    def _json_response(self, resp: requests.Response, endpoint: str) -> dict[str, Any]:
        data: Any = None
        json_error: Exception | None = None
        try:
            data = resp.json()
        except ValueError as exc:
            json_error = exc
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            detail = self._server_error_detail(data, getattr(resp, "text", ""))
            status = getattr(resp, "status_code", "")
            prefix = f"{endpoint} HTTP {status}" if status else f"{endpoint} HTTP error"
            if detail:
                raise RuntimeError(f"{prefix}: {detail}") from exc
            raise RuntimeError(f"{prefix}: {exc}") from exc
        if json_error is not None:
            raise RuntimeError(f"{endpoint} returned invalid JSON: {json_error}") from json_error
        if not isinstance(data, dict):
            raise RuntimeError(f"{endpoint} returned non-object JSON response.")
        return data

    @staticmethod
    def _server_error_detail(data: Any, raw_text: str = "") -> str:
        if isinstance(data, dict):
            base = data.get("base_resp")
            if isinstance(base, dict):
                msg = str(base.get("status_msg") or "").strip()
                code = base.get("status_code")
                if msg:
                    return f"{code}: {msg}" if code not in (None, "") else msg
            for key in ("detail", "message", "msg", "error_msg", "status_msg"):
                value = data.get(key)
                if isinstance(value, (str, int, float)):
                    text = str(value).strip()
                    if text:
                        return text
            error = data.get("error")
            if isinstance(error, dict):
                for key in ("message", "msg", "detail", "code"):
                    value = error.get(key)
                    if isinstance(value, (str, int, float)):
                        text = str(value).strip()
                        if text:
                            return text
            try:
                return json.dumps(data, ensure_ascii=False)[:1000]
            except TypeError:
                return str(data)[:1000]
        return str(raw_text or "").strip()[:1000]

    def _audio_bytes(self, audio: str) -> bytes:
        s = audio.strip()
        if s.startswith(("http://", "https://")):
            resp = requests.get(s, timeout=self.request_timeout)
            resp.raise_for_status()
            if not resp.content:
                raise RuntimeError("MiniMax returned empty audio file.")
            return resp.content
        return self._decode_audio(s)

    def _download_clone_demo_audio(self, url: str, voice_id: str) -> Path:
        resp = requests.get(url, timeout=self.request_timeout)
        resp.raise_for_status()
        if not resp.content:
            raise RuntimeError("MiniMax returned empty clone demo audio.")
        suffix = Path(urlparse(url).path).suffix.lower()
        allowed_suffixes = {f".{item}" for item in VALID_AUDIO_FORMATS}
        allowed_suffixes.update({".m4a", ".ogg", ".aac"})
        if suffix not in allowed_suffixes:
            suffix = ".mp3"
        stem = self._normalize_voice_id(voice_id or "voice_clone_demo")
        filename = f"{stem}_{int(time.time() * 1000)}{suffix}"
        self.clone_demo_audio_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.clone_demo_audio_dir / filename
        out_path.write_bytes(resp.content)
        return out_path

    def _decode_audio(self, audio: str) -> bytes:
        s = audio.strip()
        if re.fullmatch(r"[0-9a-fA-F]+", s or ""):
            return bytes.fromhex(s)
        return base64.b64decode(s)

    def _voice_language_for_character(self, character_name: str) -> str:
        target = (character_name or "").strip().lower()
        if not target:
            return "auto"
        for key, value in self.voice_language_map.items():
            if key.strip().lower() == target:
                return state.normalize_voice_language_code(value)
        return "auto"

    def _language_boost_for(self, text_lang: Any, *, character_name: str = "") -> str:
        code = self._voice_language_for_character(character_name)
        if code == "auto":
            code = state.normalize_voice_language_code(text_lang)
        boosted = self._language_boost_from_code(code)
        if boosted:
            return boosted
        return self.language_boost or "auto"

    def _language_boost_from_code(self, code: Any) -> str:
        code = state.normalize_voice_language_code(code)
        if code == "ja":
            return "Japanese"
        if code == "zh":
            return "Chinese"
        if code == "yue":
            return "Chinese,Yue"
        if code == "en":
            return "English"
        return "auto" if code == "auto" else ""

    def _voice_id_for_request(self, **kwargs) -> str:
        character_name = str(kwargs.get("character_name") or "").strip()
        ref_audio_path = str(kwargs.get("ref_audio_path") or "").strip()
        local_ref_path = self._local_reference_audio_for_character(character_name)
        if local_ref_path:
            if (
                not ref_audio_path
                or state.project_path(ref_audio_path).resolve() != local_ref_path
            ):
                self._log(
                    f"\u4f7f\u7528\u89d2\u8272\u672c\u5730\u53c2\u8003\u97f3\u9891\uff1a"
                    f"{character_name} -> {local_ref_path}"
                )
            ref_audio_path = str(local_ref_path)
        # 优先级保持不变：角色绑定 > 默认保底 > 自动克隆缓存 > 现场克隆。
        mapped = self._voice_id_for_character(character_name)
        if mapped:
            self._log(f"\u4f7f\u7528\u89d2\u8272\u56fa\u5b9a voice_id\uff1a{character_name} -> {mapped}")
            return mapped
        if self.default_voice_id:
            if character_name:
                self._log(
                    f"\u89d2\u8272 {character_name} \u672a\u7ed1\u5b9a voice_id\uff0c"
                    f"\u4f7f\u7528\u9ed8\u8ba4\u4fdd\u5e95 voice_id\uff1a{self.default_voice_id}"
                )
            return self.default_voice_id
        if character_name and ref_audio_path:
            cached = self._cached_voice(ref_audio_path, character_name)
            if cached:
                self._log(f"\u4f7f\u7528\u7f13\u5b58\u514b\u9686 voice_id\uff1a{character_name} -> {cached}")
                return cached
        if self.auto_clone_from_reference and ref_audio_path:
            prompt_text = str(kwargs.get("prompt_text") or "").strip()
            display_name = character_name or "\u672a\u547d\u540d\u89d2\u8272"
            self._log(
                f"\u672a\u627e\u5230 voice_id\uff0c"
                f"\u51c6\u5907\u4ece\u53c2\u8003\u97f3\u9891\u81ea\u52a8\u514b\u9686\uff1a{display_name}"
            )
            return self.create_cloned_voice_from_file(
                ref_audio_path,
                character_name=character_name,
                prompt_text=prompt_text,
                reference_audio_language=self._reference_audio_language_for_character(
                    character_name
                ),
            )
        return ""

    def _voice_id_for_character(self, character_name: str) -> str:
        if not character_name:
            return ""
        for key, value in self.voice_id_map.items():
            if key.strip().lower() == character_name.strip().lower():
                return str(value or "").strip()
        return ""

    def _local_reference_audio_for_character(self, character_name: str) -> Path | None:
        if not character_name:
            return None
        target = character_name.strip().lower()
        for key, value in self.local_reference_audio_map.items():
            if key.strip().lower() != target:
                continue
            raw = str(value or "").strip()
            if not raw:
                return None
            path = state.project_path(raw).resolve()
            if path.is_file():
                return path
            self._log(f"\u672c\u5730\u53c2\u8003\u97f3\u9891\u4e0d\u5b58\u5728\uff1a{path}")
            return None
        return None

    def _reference_audio_language_for_character(self, character_name: str) -> str:
        if not character_name:
            return "auto"
        target = character_name.strip().lower()
        for key, value in self.reference_audio_language_map.items():
            if key.strip().lower() == target:
                return state.normalize_voice_language_code(value)
        return "auto"

    @staticmethod
    def _normalize_sample_rate(value: Any) -> int:
        try:
            sample_rate = int(value)
        except (TypeError, ValueError):
            return 32000
        if sample_rate in VALID_SAMPLE_RATES:
            return sample_rate
        return min(VALID_SAMPLE_RATES, key=lambda item: abs(item - sample_rate))

    @staticmethod
    def _normalize_numeric_choice(value: Any, valid: tuple[int, ...], default: int) -> int:
        try:
            item = int(value)
        except (TypeError, ValueError):
            return default
        if item in valid:
            return item
        return min(valid, key=lambda candidate: abs(candidate - item))

    @staticmethod
    def _normalize_choice(value: Any, valid: tuple[str, ...], default: str) -> str:
        item = str(value or "").strip()
        if item in valid:
            return item
        lowered = item.lower()
        for candidate in valid:
            if candidate.lower() == lowered:
                return candidate
        return default

    @staticmethod
    def _normalize_emotion(value: Any) -> str:
        item = str(value or "").strip()
        if not item:
            return ""
        if item == "neutral":
            item = "calm"
        return CloudTTSAdapter._normalize_choice(item, VALID_EMOTIONS, "")

    @staticmethod
    def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
        try:
            item = float(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, item))

    @staticmethod
    def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            item = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, item))

    def _coerce_voice_id_map(self, value: dict[str, str] | str | None) -> dict[str, str]:
        if value is None:
            return {}
        raw: Any = value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                try:
                    raw = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, item in raw.items():
            name = str(key or "").strip()
            vid = str(item or "").strip()
            if name and vid:
                out[name] = vid
        return out

    def _coerce_local_reference_audio_map(
        self,
        value: dict[str, str] | str | None,
    ) -> dict[str, str]:
        if value is None:
            return {}
        raw: Any = value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                try:
                    raw = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, item in raw.items():
            name = str(key or "").strip()
            path = str(item or "").strip()
            if name and path:
                out[name] = path
        return out

    def _cache_data(self) -> dict[str, Any]:
        if not self.voice_cache_path.is_file():
            return {"voices": {}}
        try:
            raw = json.loads(self.voice_cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {"voices": {}}
        if not isinstance(raw, dict):
            return {"voices": {}}
        voices = raw.get("voices")
        if not isinstance(voices, dict):
            raw["voices"] = {}
        return raw

    def _save_cache_data(self, data: dict[str, Any]) -> None:
        self.voice_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.voice_cache_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cache_key(self, path: Path, character_name: str) -> str:
        api_hash = state.short_hash(self.api_key)
        file_hash = state.sha256_file(path)
        return state.short_hash(
            f"{self.base_api_url}|{api_hash}|{character_name}|{file_hash}",
            size=32,
        )

    def _cached_voice(self, audio_path: str | Path, character_name: str) -> str:
        path = state.project_path(audio_path).resolve()
        if not path.is_file():
            return ""
        data = self._cache_data()
        rec = (data.get("voices") or {}).get(self._cache_key(path, character_name))
        if isinstance(rec, dict):
            return str(rec.get("voice_id") or "").strip()
        return ""

    def _cache_voice(self, path: Path, character_name: str, voice_id: str) -> None:
        data = self._cache_data()
        voices = data.setdefault("voices", {})
        voices[self._cache_key(path, character_name)] = {
            "voice_id": voice_id,
            "character_name": character_name,
            "reference_audio_path": str(path),
            "reference_audio_sha256": state.sha256_file(path),
            "api_key_sha256": state.short_hash(self.api_key),
            "base_api_url": self.base_api_url,
            "model": self.model,
            "created_at": int(time.time()),
        }
        self._save_cache_data(data)

    def _new_voice_id(self, path: Path, character_name: str) -> str:
        stamp = time.strftime("%Y%m%d%H%M%S")
        seed = f"{character_name}|{path}|{time.time_ns()}"
        return self._normalize_voice_id(f"shinsekai_{stamp}_{state.short_hash(seed, 10)}")

    def _normalize_voice_id(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
        cleaned = cleaned.strip("_-")
        if not cleaned or not cleaned[0].isalpha():
            cleaned = f"voice_{cleaned}"
        if len(cleaned) < 8:
            cleaned = f"{cleaned}_{state.short_hash(cleaned, 8)}"
        return cleaned[:256].rstrip("_-")
