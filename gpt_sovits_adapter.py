from __future__ import annotations

import ast
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from sdk.adapters import TTSAdapter

from plugins.cloud_tts import state


class GPTSoVITSApiAdapter(TTSAdapter):
    """HTTP adapter for a self-hosted GPT-SoVITS api_v2.py server."""

    def __init__(
        self,
        api_key: str = "",
        base_api_url: str = "http://127.0.0.1:9880",
        model: str = state.GPT_SOVITS_DEFAULT_MODEL,
        character_profiles: dict[str, Any] | str | None = None,
        gpt_sovits_character_profiles: dict[str, Any] | str | None = None,
        local_reference_audio_map: dict[str, str] | str | None = None,
        reference_audio_language_map: dict[str, Any] | str | None = None,
        text_split_method: str = "cut5",
        gpt_sovits_text_split_method: str = "",
        media_type: str = "wav",
        gpt_sovits_media_type: str = "",
        streaming_mode: bool = False,
        gpt_sovits_streaming_mode: bool | str | None = None,
        batch_size: int = 1,
        gpt_sovits_batch_size: int | str | None = None,
        batch_threshold: float = 0.75,
        gpt_sovits_batch_threshold: float | str | None = None,
        split_bucket: bool = True,
        gpt_sovits_split_bucket: bool | str | None = None,
        fragment_interval: float = 0.3,
        gpt_sovits_fragment_interval: float | str | None = None,
        seed: int = -1,
        gpt_sovits_seed: int | str | None = None,
        parallel_infer: bool = True,
        gpt_sovits_parallel_infer: bool | str | None = None,
        repetition_penalty: float = 1.35,
        gpt_sovits_repetition_penalty: float | str | None = None,
        top_k: int = 15,
        gpt_sovits_top_k: int | str | None = None,
        top_p: float = 1.0,
        gpt_sovits_top_p: float | str | None = None,
        temperature: float = 1.0,
        gpt_sovits_temperature: float | str | None = None,
        sample_steps: int = 32,
        gpt_sovits_sample_steps: int | str | None = None,
        super_sampling: bool = False,
        gpt_sovits_super_sampling: bool | str | None = None,
        request_timeout: int = 120,
        use_runtime_config: bool = True,
        **_ignored_kwargs: Any,
    ) -> None:
        runtime_cfg: dict[str, Any] = {}
        if use_runtime_config:
            plugin_cfg = state.load_plugin_config(
                state.plugin_data_root(),
                state.GPT_SOVITS_PROVIDER_SLUG,
            )
            runtime_cfg = dict(plugin_cfg)
            runtime_cfg.update(state.get_gpt_sovits_extra())
        if runtime_cfg:
            api_key = runtime_cfg.get("api_key", api_key)
            base_api_url = runtime_cfg.get("base_api_url", base_api_url)
            model = runtime_cfg.get("model", model)
            character_profiles = runtime_cfg.get(
                "gpt_sovits_character_profiles",
                runtime_cfg.get("character_profiles", character_profiles),
            )
            local_reference_audio_map = runtime_cfg.get(
                "local_reference_audio_map",
                local_reference_audio_map,
            )
            reference_audio_language_map = runtime_cfg.get(
                "reference_audio_language_map",
                reference_audio_language_map,
            )
            gpt_sovits_text_split_method = runtime_cfg.get(
                "gpt_sovits_text_split_method",
                gpt_sovits_text_split_method,
            )
            gpt_sovits_media_type = runtime_cfg.get(
                "gpt_sovits_media_type",
                gpt_sovits_media_type,
            )
            gpt_sovits_streaming_mode = runtime_cfg.get(
                "gpt_sovits_streaming_mode",
                gpt_sovits_streaming_mode,
            )
            gpt_sovits_batch_size = runtime_cfg.get(
                "gpt_sovits_batch_size",
                gpt_sovits_batch_size,
            )
            gpt_sovits_batch_threshold = runtime_cfg.get(
                "gpt_sovits_batch_threshold",
                gpt_sovits_batch_threshold,
            )
            gpt_sovits_split_bucket = runtime_cfg.get(
                "gpt_sovits_split_bucket",
                gpt_sovits_split_bucket,
            )
            gpt_sovits_fragment_interval = runtime_cfg.get(
                "gpt_sovits_fragment_interval",
                gpt_sovits_fragment_interval,
            )
            gpt_sovits_seed = runtime_cfg.get("gpt_sovits_seed", gpt_sovits_seed)
            gpt_sovits_parallel_infer = runtime_cfg.get(
                "gpt_sovits_parallel_infer",
                gpt_sovits_parallel_infer,
            )
            gpt_sovits_repetition_penalty = runtime_cfg.get(
                "gpt_sovits_repetition_penalty",
                gpt_sovits_repetition_penalty,
            )
            gpt_sovits_top_k = runtime_cfg.get("gpt_sovits_top_k", gpt_sovits_top_k)
            gpt_sovits_top_p = runtime_cfg.get("gpt_sovits_top_p", gpt_sovits_top_p)
            gpt_sovits_temperature = runtime_cfg.get(
                "gpt_sovits_temperature",
                gpt_sovits_temperature,
            )
            gpt_sovits_sample_steps = runtime_cfg.get(
                "gpt_sovits_sample_steps",
                gpt_sovits_sample_steps,
            )
            gpt_sovits_super_sampling = runtime_cfg.get(
                "gpt_sovits_super_sampling",
                gpt_sovits_super_sampling,
            )
        profiles = gpt_sovits_character_profiles or character_profiles
        self.api_key = self._normalize_api_key(api_key)
        self.base_api_url = (base_api_url or "http://127.0.0.1:9880").rstrip("/")
        self.model = self._normalize_model(model)
        self.character_profiles = self._coerce_character_profiles(profiles)
        self.local_reference_audio_map = self._coerce_path_map(local_reference_audio_map)
        self.reference_audio_language_map = state.coerce_voice_language_map(
            reference_audio_language_map
        )
        self.text_split_method = str(
            gpt_sovits_text_split_method or text_split_method or "cut5"
        ).strip() or "cut5"
        self.media_type = self._normalize_media_type(gpt_sovits_media_type or media_type)
        self.streaming_mode = self._as_bool(gpt_sovits_streaming_mode, streaming_mode)
        self.batch_size = self._as_int(gpt_sovits_batch_size, batch_size, minimum=1)
        self.batch_threshold = self._as_float(gpt_sovits_batch_threshold, batch_threshold)
        self.split_bucket = self._as_bool(gpt_sovits_split_bucket, split_bucket)
        self.fragment_interval = self._as_float(gpt_sovits_fragment_interval, fragment_interval)
        self.seed = self._as_int(gpt_sovits_seed, seed)
        self.parallel_infer = self._as_bool(gpt_sovits_parallel_infer, parallel_infer)
        self.repetition_penalty = self._as_float(
            gpt_sovits_repetition_penalty,
            repetition_penalty,
        )
        self.top_k = self._as_int(gpt_sovits_top_k, top_k, minimum=1)
        self.top_p = self._as_float(gpt_sovits_top_p, top_p)
        self.temperature = self._as_float(gpt_sovits_temperature, temperature)
        self.sample_steps = self._as_int(gpt_sovits_sample_steps, sample_steps, minimum=1)
        self.super_sampling = self._as_bool(gpt_sovits_super_sampling, super_sampling)
        self.request_timeout = int(request_timeout or 120)
        self._api_variant = "api_v2"
        self._current_gpt_weights = ""
        self._current_sovits_weights = ""
        self._last_switch_error = ""

    @classmethod
    def get_config_schema(cls) -> dict[str, dict]:
        return {
            "api_key": {
                "type": "str",
                "label": "GPT-SoVITS Token（可选）",
                "default": "",
                "secret": True,
            },
            "base_api_url": {
                "type": "str",
                "label": "GPT-SoVITS Base URL",
                "default": "http://127.0.0.1:9880",
            },
        }

    def switch_model(self, model_info: Any) -> bool:
        self._last_switch_error = ""
        profile = self._profile_from_model_info(model_info)
        gpt_path = self._clean_text(
            profile.get("gpt_weights_path")
            or self._dict_get(model_info, "gpt_weights_path")
            or self._dict_get(model_info, "gpt_model_path")
        )
        sovits_path = self._clean_text(
            profile.get("sovits_weights_path")
            or self._dict_get(model_info, "sovits_weights_path")
            or self._dict_get(model_info, "sovits_model_path")
        )
        if not gpt_path and not sovits_path:
            self._log("未配置远端模型路径，沿用 GPT-SoVITS 服务器当前加载的模型。")
            return True
        try:
            if self._api_variant == "legacy":
                self._set_legacy_model(gpt_path, sovits_path)
            else:
                try:
                    if gpt_path and gpt_path != self._current_gpt_weights:
                        self._set_weights("set_gpt_weights", gpt_path)
                    if sovits_path and sovits_path != self._current_sovits_weights:
                        self._set_weights("set_sovits_weights", sovits_path)
                except Exception as exc:
                    if not self._is_missing_endpoint_error(exc):
                        raise
                    self._log(
                        "api_v2 model switch endpoint is unavailable; "
                        "falling back to legacy /set_model."
                    )
                    self._set_legacy_model(gpt_path, sovits_path)
                    self._api_variant = "legacy"
            if gpt_path:
                self._current_gpt_weights = gpt_path
            if sovits_path:
                self._current_sovits_weights = sovits_path
            return True
        except Exception as exc:
            self._last_switch_error = str(exc)
            self._log(f"切换模型失败：{exc}")
            return False

    def generate_speech(self, text, file_path=None, **kwargs):
        text_value = str(text or "")
        character_name = str(kwargs.get("character_name") or "").strip()
        if self._last_switch_error:
            self._log(f"Synthesis blocked because the last GPT-SoVITS model switch failed: {self._last_switch_error}")
            return None
        profile = self._profile_for_character(character_name)
        ref_audio_path = self._first_text(
            profile.get("ref_audio_path"),
        )
        if not ref_audio_path:
            self._log("合成失败：未配置 GPT-SoVITS 服务器可访问的参考音频路径。")
            return None
        prompt_text = self._first_text(
            profile.get("prompt_text"),
            kwargs.get("prompt_text"),
        )
        prompt_lang = self._normalize_lang(
            self._first_text(
                profile.get("prompt_lang"),
                self.reference_audio_language_map.get(character_name, ""),
                kwargs.get("prompt_lang"),
                state.DEFAULT_PROMPT_LANGUAGE,
            )
        )
        text_lang = self._normalize_lang(
            self._first_text(
                profile.get("text_lang"),
                kwargs.get("text_lang"),
                state.DEFAULT_PROMPT_LANGUAGE,
            )
        )
        # A caller-supplied file_path must be used verbatim: the host hands us a
        # ".part" temp path and renames it after validating, so "fixing" the
        # suffix here makes the host miss the file entirely.
        out_path = Path(
            file_path
            or f"cache/audio/gpt_sovits_{int(time.time() * 1000)}.{self.media_type}"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        speed = self._as_float(kwargs.get("speed_factor"), 1.0)
        payload = {
            "text": text_value,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "media_type": self.media_type,
            "streaming_mode": self.streaming_mode,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "batch_threshold": self.batch_threshold,
            "split_bucket": self.split_bucket,
            "return_fragment": False,
            "speed_factor": speed,
            "fragment_interval": self.fragment_interval,
            "seed": self.seed,
            "parallel_infer": self.parallel_infer,
            "repetition_penalty": self.repetition_penalty,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "sample_steps": self.sample_steps,
            "super_sampling": self.super_sampling,
        }
        try:
            self._log(
                f"开始合成：角色={character_name or '未指定角色'}，"
                f"文本长度={len(text_value)}，模式={self.model}，参考音频={ref_audio_path}"
            )
            if self._api_variant == "legacy":
                resp = self._post_legacy_tts(payload, speed)
            else:
                try:
                    resp = requests.post(
                        self._url("tts"),
                        headers=self._headers(),
                        json=payload,
                        timeout=self.request_timeout,
                    )
                    resp.raise_for_status()
                except Exception as exc:
                    if not self._is_missing_endpoint_error(exc):
                        raise
                    self._log(
                        "api_v2 /tts endpoint is unavailable; "
                        "falling back to legacy root TTS endpoint."
                    )
                    resp = self._post_legacy_tts(payload, speed)
                    self._api_variant = "legacy"
            out_path.write_bytes(resp.content)
            abs_path = os.path.abspath(out_path)
            self._log(f"合成完成：{abs_path}")
            return abs_path
        except Exception as exc:
            self._log(f"合成失败：{exc}")
            return None

    def _set_weights(self, endpoint: str, weights_path: str) -> None:
        self._log(f"请求 GPT-SoVITS /{endpoint}：{weights_path}")
        resp = requests.get(
            self._url(endpoint),
            params={"weights_path": weights_path},
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()

    def _set_legacy_model(self, gpt_path: str, sovits_path: str) -> None:
        payload = {
            "gpt_model_path": gpt_path or None,
            "sovits_model_path": sovits_path or None,
        }
        self._log(
            "Requesting GPT-SoVITS legacy /set_model: "
            f"gpt={gpt_path or '<unchanged>'}, sovits={sovits_path or '<unchanged>'}"
        )
        resp = requests.post(
            self._url("set_model"),
            json=payload,
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()

    def _post_legacy_tts(self, api_v2_payload: dict[str, Any], speed: float):
        payload = {
            "refer_wav_path": api_v2_payload["ref_audio_path"],
            "prompt_text": api_v2_payload["prompt_text"],
            "prompt_language": api_v2_payload["prompt_lang"],
            "text": api_v2_payload["text"],
            "text_language": api_v2_payload["text_lang"],
            "cut_punc": "",
            "top_k": api_v2_payload["top_k"],
            "top_p": api_v2_payload["top_p"],
            "temperature": api_v2_payload["temperature"],
            "speed": speed,
            "inp_refs": None,
            "sample_steps": api_v2_payload["sample_steps"],
            "if_sr": api_v2_payload["super_sampling"],
        }
        self._log("Requesting GPT-SoVITS legacy root TTS endpoint.")
        resp = requests.post(
            self.base_api_url + "/",
            json=payload,
            headers=self._headers(),
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        return resp

    def _is_missing_endpoint_error(self, exc: Exception) -> bool:
        if not isinstance(exc, requests.HTTPError):
            return False
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code in {404, 405}

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _url(self, endpoint: str) -> str:
        return f"{self.base_api_url}/{endpoint.lstrip('/')}"

    def _log(self, message: str) -> None:
        print(f"GPT SoVITS Cloud：{message}", flush=True)

    def _profile_from_model_info(self, model_info: Any) -> dict[str, str]:
        name = self._dict_get(model_info, "character_name")
        return self._profile_for_character(name)

    def _profile_for_character(self, character_name: str | None) -> dict[str, str]:
        name = str(character_name or "").strip()
        if name and name in self.character_profiles:
            return dict(self.character_profiles[name])
        return {}

    def _coerce_character_profiles(self, value: Any) -> dict[str, dict[str, str]]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    value = {}
        if not isinstance(value, dict):
            return {}
        out: dict[str, dict[str, str]] = {}
        for key, item in value.items():
            name = str(key or "").strip()
            if not name or not isinstance(item, dict):
                continue
            profile = {
                "gpt_weights_path": self._clean_text(
                    item.get("gpt_weights_path") or item.get("gpt_model_path")
                ),
                "sovits_weights_path": self._clean_text(
                    item.get("sovits_weights_path") or item.get("sovits_model_path")
                ),
                "ref_audio_path": self._clean_text(
                    item.get("ref_audio_path") or item.get("reference_audio_path")
                ),
                "prompt_text": self._clean_text(item.get("prompt_text")),
                "prompt_lang": self._normalize_lang(item.get("prompt_lang") or ""),
                "text_lang": self._normalize_lang(item.get("text_lang") or ""),
            }
            clean = {k: v for k, v in profile.items() if v}
            if clean:
                out[name] = clean
        return out

    def _coerce_path_map(self, value: Any) -> dict[str, str]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    value = {}
        if not isinstance(value, dict):
            return {}
        out: dict[str, str] = {}
        for key, item in value.items():
            name = str(key or "").strip()
            path = self._clean_text(item)
            if name and path:
                out[name] = path
        return out

    def _normalize_api_key(self, value: Any) -> str:
        text = str(value or "").strip()
        if text.lower().startswith("bearer "):
            return text[7:].strip()
        return text

    def _normalize_model(self, value: Any) -> str:
        text = str(value or "").strip()
        return text or state.GPT_SOVITS_DEFAULT_MODEL

    def _normalize_media_type(self, value: Any) -> str:
        text = str(value or "wav").strip().lower().lstrip(".")
        return text if text in state.GPT_SOVITS_MEDIA_TYPES else "wav"

    def _normalize_lang(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if text == "auto":
            return state.DEFAULT_PROMPT_LANGUAGE
        if text in state.GPT_SOVITS_LANGUAGE_CODES:
            return text
        return ""

    def _first_text(self, *values: Any) -> str:
        for value in values:
            text = self._clean_text(value)
            if text:
                return text
        return ""

    def _clean_text(self, value: Any) -> str:
        text = str(value or "").strip()
        if text in {".", "./", ".\\"}:
            return ""
        return text

    def _dict_get(self, value: Any, key: str) -> str:
        if isinstance(value, dict):
            return self._clean_text(value.get(key))
        return ""

    def _as_bool(self, value: Any, default: bool) -> bool:
        if value is None or value == "":
            return bool(default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on", "y"}:
            return True
        if text in {"0", "false", "no", "off", "n"}:
            return False
        return bool(default)

    def _as_int(self, value: Any, default: int, minimum: int | None = None) -> int:
        try:
            parsed = int(float(str(value).strip())) if value not in (None, "") else int(default)
        except Exception:
            parsed = int(default)
        if minimum is not None and parsed < minimum:
            return int(default)
        return parsed

    def _as_float(self, value: Any, default: float) -> float:
        try:
            return float(str(value).strip()) if value not in (None, "") else float(default)
        except Exception:
            return float(default)
