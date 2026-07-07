from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml


PROVIDER_SLUG = "minimax-tts"
QWEN_PROVIDER_SLUG = "qwen-tts"
GPT_SOVITS_PROVIDER_SLUG = "gpt-sovits-api"
PLUGIN_ID = "com.shinsekai.cloud_tts"
PLUGIN_ENTRY = "plugins.cloud_tts.plugin:CloudTtsPlugin"
PLUGIN_VERSION = "0.12.2"

LEGACY_PROVIDER_SLUG = "cloud-tts"
LEGACY_PLUGIN_ID = "com.shinsekai.minimax_tts"
LEGACY_PLUGIN_ENTRY = "plugins.minimax_tts.plugin:MinimaxTtsPlugin"
LEGACY_PROVIDER_SLUGS = frozenset({LEGACY_PROVIDER_SLUG})
LEGACY_PLUGIN_ENTRIES = frozenset({LEGACY_PLUGIN_ENTRY})

# API 页面只保留连接凭证，避免和插件设置页重复展示行为参数。
ADAPTER_CONFIG_KEYS = {
    "api_key",
    "base_api_url",
    "model",
    "default_voice_id",
    "language_type",
}

# 这些字段由 Cloud TTS 插件设置页维护，运行时会传给 TTS adapter。
PLUGIN_STATE_KEYS = {
    "model",
    "local_reference_audio_map",
    "reference_audio_language_map",
    "reference_text_map",
    "auto_prompt_constraint",
    "protect_translate_tone_tags",
    "gpt_sovits_character_profiles",
    "gpt_sovits_text_split_method",
    "gpt_sovits_media_type",
    "gpt_sovits_streaming_mode",
    "gpt_sovits_batch_size",
    "gpt_sovits_batch_threshold",
    "gpt_sovits_split_bucket",
    "gpt_sovits_fragment_interval",
    "gpt_sovits_seed",
    "gpt_sovits_parallel_infer",
    "gpt_sovits_repetition_penalty",
    "gpt_sovits_top_k",
    "gpt_sovits_top_p",
    "gpt_sovits_temperature",
    "gpt_sovits_sample_steps",
    "gpt_sovits_super_sampling",
}

# ----------------------------------------------------------------------
# Qwen TTS (DashScope / 百炼) 常量
# ----------------------------------------------------------------------

QWEN_MODELS = (
    "qwen3-tts-vc-2026-01-22",
)
QWEN_DEFAULT_MODEL = "qwen3-tts-vc-2026-01-22"

QWEN_LANGUAGE_TYPES = (
    ("Chinese", "中文（普通话）"),
    ("English", "英语"),
    ("Japanese", "日语"),
    ("Korean", "韩语"),
    ("French", "法语"),
    ("German", "德语"),
    ("Russian", "俄语"),
    ("Italian", "意大利语"),
    ("Spanish", "西班牙语"),
    ("Portuguese", "葡萄牙语"),
)

QWEN_VOICE_ENROLLMENT_MODEL = "qwen-voice-enrollment"
# 声音复刻时 target_model 必须与合成时一致，且必须是 VC 系列模型
QWEN_VC_MODEL = "qwen3-tts-vc-2026-01-22"

# GPT-SoVITS API constants. api_v2.py is the server entrypoint name; it is not a model-version cap.
GPT_SOVITS_MODELS = (
    "auto",
    "v1-v2-v2Pro",
    "v2Pro2025",
    "v2ProPlus",
    "v3",
    "v4",
    "custom",
)
GPT_SOVITS_DEFAULT_MODEL = "auto"
GPT_SOVITS_MEDIA_TYPES = ("wav", "mp3", "ogg", "aac", "raw")
GPT_SOVITS_LANGUAGE_OPTIONS = (
    ("auto", "auto"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("yue", "Cantonese"),
    ("en", "English"),
    ("ko", "Korean"),
)
GPT_SOVITS_LANGUAGE_CODES = tuple(code for code, _label in GPT_SOVITS_LANGUAGE_OPTIONS if code != "auto")

# 所有 Cloud TTS 支持的 provider slug
ALL_CLOUD_TTS_SLUGS = frozenset({PROVIDER_SLUG, QWEN_PROVIDER_SLUG, GPT_SOVITS_PROVIDER_SLUG})

# 提示词约束块标记：注入时包裹在 system prompt 两端，移除时通过正则匹配这两个标记定位
CONSTRAINT_START = "<<<CLOUD_TTS_TONE_CONSTRAINT_START>>>"
CONSTRAINT_END = "<<<CLOUD_TTS_TONE_CONSTRAINT_END>>>"
# 兼容旧版 MiniMax 插件标记，否则升级后旧块删不掉会重复追加
LEGACY_CONSTRAINT_START = "<<<MINIMAX_TTS_TONE_CONSTRAINT_START>>>"
LEGACY_CONSTRAINT_END = "<<<MINIMAX_TTS_TONE_CONSTRAINT_END>>>"
# 抑制注入的时间戳：插件保存配置后短暂抑制注入，避免保存触发连锁同步
_PROMPT_CONSTRAINT_SUPPRESS_UNTIL = 0.0

# MiniMax 支持的 19 种语气标签，受语气标签保护开关管控
CLOUD_TTS_TONE_TAGS = (
    "(laughs)",
    "(chuckle)",
    "(coughs)",
    "(clear-throat)",
    "(groans)",
    "(breath)",
    "(pant)",
    "(inhale)",
    "(exhale)",
    "(gasps)",
    "(sniffs)",
    "(sighs)",
    "(snorts)",
    "(burps)",
    "(lip-smacking)",
    "(humming)",
    "(hissing)",
    "(emm)",
    "(sneezes)",
)

VOICE_LANGUAGE_OPTIONS = (
    ("auto", "跟随主菜单语音语言"),
    ("zh", "中文"),
    ("ja", "日语"),
    ("yue", "粤语"),
    ("en", "英语"),
)
VALID_VOICE_LANGUAGES = tuple(code for code, _label in VOICE_LANGUAGE_OPTIONS)

PROMPT_LANGUAGE_OPTIONS = (
    ("zh", "中文"),
    ("ja", "日语"),
    ("yue", "粤语"),
    ("en", "英语"),
)
PROMPT_LANGUAGE_CODES = tuple(code for code, _label in PROMPT_LANGUAGE_OPTIONS)
DEFAULT_PROMPT_LANGUAGE = "zh"
DEFAULT_PROMPT_VERSION_IDS = {
    "zh": "default_zh",
    "ja": "default_ja",
    "yue": "default_yue",
    "en": "default_en",
}


# ----------------------------------------------------------------------
# Per-Character Prompt Constraint Version System (1.0.0)
# Each character has their own constraint versions stored in:
# {plugin_data_root}/prompt_constraints/{character_name}.json
# ----------------------------------------------------------------------


def _constraint_file_stem(character_name: str) -> str:
    """Generate safe filename stem for character constraint file."""
    name = (character_name or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    safe = safe.strip("._-")[:48]
    if not safe:
        safe = "character"
    return f"{safe}_{short_hash(name, 10)}"


def prompt_constraints_root() -> Path:
    """Root directory for per-character constraint files."""
    return plugin_data_root() / "prompt_constraints"


def constraint_file_path(character_name: str) -> Path:
    """Path to a specific character's constraint file."""
    return prompt_constraints_root() / f"{_constraint_file_stem(character_name)}.json"


def _default_constraint_store(character_name: str) -> dict[str, Any]:
    """Create default empty constraint store for a character."""
    return {
        "character_name": character_name,
        "selected_version": None,
        "versions": {},
        "updated_at": int(time.time()),
    }


def _load_old_prompt_constraints() -> dict[str, str]:
    """Load old global constraints for migration from previous version."""
    old_root = legacy_plugin_data_root()
    old_path = old_root / "prompt_constraints.json"
    if not old_path.is_file():
        return _get_hardcoded_constraints()
    try:
        raw = json.loads(old_path.read_text(encoding="utf-8"))
        return raw.get("constraints", _get_hardcoded_constraints())
    except Exception:
        return _get_hardcoded_constraints()


def normalize_voice_language_code(value: Any) -> str:
    """Normalize UI / config voice language codes to the plugin's short codes."""
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"", "auto", "default", "follow", "main"}:
        return "auto"
    if raw in {"zh", "zh_cn", "zh_hans", "cn", "chinese", "mandarin", "中文"}:
        return "zh"
    if raw in {"ja", "jp", "japanese", "日本語", "日语", "日語"}:
        return "ja"
    if raw in {"yue", "zh_yue", "zh_hk", "cantonese", "粤语", "粵語"}:
        return "yue"
    if raw in {"en", "eng", "english", "英语", "英語"}:
        return "en"
    return "auto"


def voice_language_label(value: Any) -> str:
    code = normalize_voice_language_code(value)
    labels = dict(VOICE_LANGUAGE_OPTIONS)
    return labels.get(code, labels["auto"])


def _tone_tag_lines() -> str:
    return """(laughs)：开心、调侃、得意、明显笑出来时使用。
(chuckle)：克制的轻笑、轻松吐槽、小声笑时使用。
(coughs)：咳嗽、被呛到、掩饰尴尬时使用。
(clear-throat)：准备开口、切换到正式语气、收束气氛时使用。
(groans)：痛苦、费力、困扰、不情愿时使用。
(breath)：柔和停顿、贴近感、轻声说话、自然换气时使用。
(pant)：跑动后、慌乱、急促、体力消耗时使用。
(inhale)：开口前吸气、震惊前、准备说重要内容时使用。
(exhale)：释然、放松、疲惫、压下情绪后使用。
(gasps)：惊讶、震惊、突然发现异常、危险逼近时使用。
(sniffs)：委屈、鼻音、快哭但忍住时使用。
(sighs)：无奈、担心、疲惫、提醒风险、收束语气时使用。
(snorts)：不屑、忍笑、轻蔑、得意反应时使用。
(burps)：打嗝或故意滑稽时使用。
(lip-smacking)：犹豫、思考、略带不满、准备评价时使用。
(humming)：轻声哼、愉快、思考、拖长语气时使用。
(hissing)：压低声音、警告、危险感、阴沉语气时使用。
(emm)：犹豫、思考、短暂停顿、组织语言时使用。
(sneezes)：喷嚏。"""


def _dialog_schema_example(translate_hint: str) -> str:
    return f"""{{
  "dialog": [
    {{
      "character_name": "角色名",
      "sprite": "str, 对应的立绘ID字符串，例如 01, 02",
      "speech": "该角色说的中文台词",
      "effect": "角色的特效名称（可选），选择范围在 LEAVE、SHOCKED、DISAPPOINTED、ATTENTION 内",
      "translate": "{translate_hint}"
    }}
  ]
}}"""


def _language_instruction_block(code: str) -> str:
    if code == "zh":
        return f"""角色语音目标：中文。
translate 字段不是外语翻译，而是 Cloud TTS 的中文合成文本。
中文适配规则：
1. translate 必须使用自然简体中文，可以与 speech 完全相同，也可以在不改变语义的前提下改得更口语、更适合朗读。
2. speech 只负责屏幕显示，保持干净中文；translate 才允许加入语气标签。
3. 角色口癖可以少量保留，例如"りょ""ヤバ""ガチ？""前辈"，但整句主体必须是中文。
4. 标签优先放在句首或自然停顿处，例如"(chuckle)前辈，这个我来观测。"、"前辈，(sighs)这个风险要先压住。"
5. 不要把标签翻译成"笑声""叹气"，也不要写成舞台说明。
中文示例：
{_dialog_schema_example("该句台词的中文 TTS 文本（与系统语音目标语言一致，可加入语气标签）")}"""
    if code == "ja":
        return f"""角色语音目标：日语。
translate 字段是 Cloud TTS 的日语合成文本。
日语适配规则：
1. translate 要把 speech 的中文台词改写为自然日语，不要逐字硬翻。
2. 可以保留角色称呼、口癖和语气，例如"先輩""りょ""ヤバ""ガチ？"；但整体必须像日语台词。
3. 语气标签仍然使用英文括号标签，不能翻译成日语。
4. 标签适合放在句首或日语停顿处，例如"(chuckle)先輩、これは華淡が観測します。"、"えっと……(sighs)先輩、それは少し危ないです。"
5. speech 仍然保持简体中文，不能把日语或标签写进 speech。
日语示例：
{_dialog_schema_example("该句台词的日语译文（与系统语音目标语言一致，可加入语气标签）")}"""
    if code == "yue":
        return f"""角色语音目标：粤语。
translate 字段是 Cloud TTS 的粤语合成文本。
粤语适配规则：
1. translate 要把 speech 的中文台词改写为自然粤语口语，可以使用"啦""喎""啫""咁""唔""冇"等粤语表达。
2. 不要只把普通话词序照搬成粤语，要让句子适合粤语朗读。
3. 语气标签仍然使用英文括号标签，不能翻译成中文或粤语。
4. 标签适合放在句首或自然停顿处，例如"(chuckle)前辈，呢个我嚟睇住，问题唔大。"、"(sighs)前辈，呢度要小心啲。"
5. speech 仍然保持简体中文，不能把粤语写进 speech。
粤语示例：
{_dialog_schema_example("该句台词的粤语译文（与系统语音目标语言一致，可加入语气标签）")}"""
    if code == "en":
        return f"""角色语音目标：英语。
translate 字段是 Cloud TTS 的英语合成文本。
英语适配规则：
1. translate 要把 speech 的中文台词改写为自然口语英语，不要逐字硬翻。
2. 可以使用 contraction，例如 I’ll、don’t、it’s，让语音更自然。
3. 角色称呼可按语境处理，例如"前辈"可写为 Senpai 或 senior；保持角色风格优先。
4. 语气标签仍然使用英文括号标签，不要改写成 stage directions。
5. speech 仍然保持简体中文，不能把英语或标签写进 speech。
英语示例：
{_dialog_schema_example("该句台词的英语译文（与系统语音目标语言一致，可加入语气标签）")}"""
    return """角色语音目标：自动。
请根据插件里为当前角色选择的语音语言生成 translate；如果没有角色语言设置，则跟随主菜单语音语言。
自动适配规则：
1. 中文目标时，translate 使用自然简体中文，不翻译成外语。
2. 日语、粤语、英语目标时，translate 改写为对应语言的自然口语。
3. 无论目标语言是什么，每条角色对白都必须输出 translate 字段。
4. speech 永远保持自然简体中文，不加入语气标签。
5. 语气标签只允许进入 translate 字段。"""


def _language_constraint_body(voice_language: Any) -> str:
    code = normalize_voice_language_code(voice_language)
    target = _language_instruction_block(code)
    return f"""Cloud TTS 启用时，每条角色对白必须输出 translate 字段。
speech 字段用于屏幕显示，必须保持自然简体中文，不要放入语气标签。
translate 字段用于 Cloud TTS 合成，可以根据角色语音目标语言改写，并且允许加入 Cloud TTS 支持的语气标签。

{target}

严格规则：
1. speech 字段必须是自然简体中文，不出现 (laughs)、(sighs)、(breath)、(gasps) 等括号语气标签。
2. 语气标签只能放在 translate 字段，不要放进 speech 字段。
3. 不要加入舞台说明、动作描写、旁白、Markdown 或代码块。
4. 标签不翻译成中文，也不写进旁白；它只给 Cloud TTS 做语音控制提示。
5. 仅当模型选择 speech-2.8-hd 或 speech-2.8-turbo 时，才推荐在 translate 中加入语气标签；其他模型尽量少用或不用。
6. 不限制语气标签数量；可根据台词情绪自然使用多个标签，但不要为了堆叠而无意义添加。
7. 标签要贴近情绪发生的位置，不能把所有句子都机械放同一个标签。

可用语气标签：
{_tone_tag_lines()}

跨语言标签使用建议：
轻快调侃、得意、自信：优先使用 (laughs) 或 (chuckle)
惊讶、发现异常、ヤバ 展开：优先使用 (gasps)
担心、提醒风险、需要刹车：优先使用 (sighs)
温柔陪伴、靠近感、语音助手模式：可使用 (breath)
犹豫、思考、短暂停顿：可使用 (emm)、(inhale) 或 (lip-smacking)
运动、急促、慌乱：可使用 (pant)、(breath)
强烈疲惫或放下情绪：可使用 (exhale)、(sighs)
委屈、鼻音、快哭：可使用 (sniffs)
搞怪或特殊音效：少量使用 (coughs)、(clear-throat)、(sneezes)、(burps)
危险、压低声音：可使用 (hissing)"""


def build_default_constraint_text(voice_language: Any = "auto") -> str:
    """Build the built-in Cloud TTS prompt constraint for a voice language."""
    body = _language_constraint_body(voice_language).strip()
    return f"{CONSTRAINT_START}\n{body}\n{CONSTRAINT_END}"


def _get_hardcoded_constraints() -> dict[str, str]:
    """Return hardcoded default constraints for migration fallback."""
    data = {
        "default": build_default_constraint_text("auto"),
        "auto": build_default_constraint_text("auto"),
    }
    for code in PROMPT_LANGUAGE_CODES:
        data[code] = build_default_constraint_text(code)
    return data


def _prompt_language_label(language: Any) -> str:
    code = normalize_voice_language_code(language)
    labels = dict(PROMPT_LANGUAGE_OPTIONS)
    return labels.get(code, labels[DEFAULT_PROMPT_LANGUAGE])


def _normalize_prompt_language(language: Any) -> str | None:
    code = normalize_voice_language_code(language)
    return code if code in PROMPT_LANGUAGE_CODES else None


def _default_prompt_version_id(language: Any) -> str:
    code = _normalize_prompt_language(language) or DEFAULT_PROMPT_LANGUAGE
    return DEFAULT_PROMPT_VERSION_IDS[code]


def _runtime_prompt_language(language: Any = "auto") -> str:
    """Resolve the prompt language from the main program state."""
    return (
        _normalize_prompt_language(language)
        or _normalize_prompt_language(current_system_voice_language())
        or DEFAULT_PROMPT_LANGUAGE
    )


def _prompt_language_from_version_id(version_id: str | None) -> str | None:
    for code, default_vid in DEFAULT_PROMPT_VERSION_IDS.items():
        if version_id == default_vid:
            return code
    return None


def _preferred_prompt_language(character_name: str) -> str:
    return _runtime_prompt_language()


def _default_prompt_version_record(language: str, *, created_at: int | None = None) -> dict[str, Any]:
    code = _normalize_prompt_language(language) or DEFAULT_PROMPT_LANGUAGE
    order = PROMPT_LANGUAGE_CODES.index(code) + 1
    label = _prompt_language_label(code)
    return {
        "name": f"{label}默认提示词",
        "constraint_text": build_default_constraint_text(code),
        "created_at": created_at if created_at is not None else int(time.time()),
        "source": "default",
        "language": code,
        "sort_order": order * 10,
    }


def _constraint_version_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int, str]:
    vid, vdata = item
    if isinstance(vdata, dict) and vdata.get("sort_order") is not None:
        try:
            return (0, int(vdata.get("sort_order")), vid)
        except (TypeError, ValueError):
            pass
    created_at = 0
    if isinstance(vdata, dict):
        try:
            created_at = int(vdata.get("created_at", 0))
        except (TypeError, ValueError):
            created_at = 0
    return (1, created_at, vid)


def ensure_default_prompt_versions(store: dict[str, Any]) -> bool:
    """
    Ensure each character has four built-in prompt versions: zh/ja/yue/en.
    Returns True when the store was changed and should be saved.
    """
    changed = False
    character_name = str(store.get("character_name") or "")
    versions = store.get("versions")
    if not isinstance(versions, dict):
        versions = {}
        store["versions"] = versions
        changed = True

    # Migrate the old one-version default store into the four-language layout.
    legacy_v1 = versions.get("v1")
    legacy_custom_text = None
    legacy_custom_name = None
    if (
        len(versions) == 1
        and isinstance(legacy_v1, dict)
        and legacy_v1.get("source") == "default"
        and not _normalize_prompt_language(legacy_v1.get("language"))
    ):
        text = str(legacy_v1.get("constraint_text") or "")
        if text and text != build_default_constraint_text("auto"):
            legacy_custom_text = text
            legacy_custom_name = str(legacy_v1.get("name") or "")
        versions.clear()
        store["selected_version"] = None
        changed = True

    now = int(time.time())
    for index, code in enumerate(PROMPT_LANGUAGE_CODES):
        vid = DEFAULT_PROMPT_VERSION_IDS[code]
        record = versions.get(vid)
        if not isinstance(record, dict):
            versions[vid] = _default_prompt_version_record(code, created_at=now + index)
            changed = True
            continue
        default_record = _default_prompt_version_record(code, created_at=record.get("created_at", now + index))
        for key in ("name", "constraint_text", "source", "language", "sort_order"):
            if not record.get(key):
                record[key] = default_record[key]
                changed = True
        if (
            record.get("source") == "default"
            and record.get("constraint_text") != default_record["constraint_text"]
        ):
            record["constraint_text"] = default_record["constraint_text"]
            changed = True
        if _normalize_prompt_language(record.get("language")) != code:
            record["language"] = code
            changed = True

    if legacy_custom_text:
        zh_record = versions.get(DEFAULT_PROMPT_VERSION_IDS[DEFAULT_PROMPT_LANGUAGE])
        if isinstance(zh_record, dict):
            zh_record["constraint_text"] = legacy_custom_text
            if legacy_custom_name:
                zh_record["name"] = legacy_custom_name
            changed = True

    preferred_vid = _default_prompt_version_id(_preferred_prompt_language(character_name))
    selected = store.get("selected_version")
    if not selected or selected not in versions:
        store["selected_version"] = preferred_vid if preferred_vid in versions else next(iter(versions), None)
        changed = True

    # Legacy source="default" versions without language should not stay ambiguous.
    preferred_language = _preferred_prompt_language(character_name)
    for vid, record in versions.items():
        if not isinstance(record, dict):
            continue
        if record.get("source") == "default" and not _normalize_prompt_language(record.get("language")):
            record["language"] = _prompt_language_from_version_id(vid) or preferred_language
            changed = True

    return changed


def get_default_template_text(language: Any = DEFAULT_PROMPT_LANGUAGE) -> str:
    """
    获取当前默认模板文本。
    优先读取「默认模板」角色里对应语言的母版，缺失时回退到内置默认模板。
    """
    store = load_character_constraints("默认模板")
    versions = store.get("versions", {})
    code = _normalize_prompt_language(language) or DEFAULT_PROMPT_LANGUAGE
    vid = DEFAULT_PROMPT_VERSION_IDS[code]
    record = versions.get(vid)
    if isinstance(record, dict) and record.get("constraint_text"):
        return str(record.get("constraint_text"))
    return build_default_constraint_text(code)


def propagate_default_template(new_text: str, language: Any | None = None) -> int:
    """
    将默认模板内容同步到所有标记为 source='default' 的角色版本。
    指定 language 时只同步同语种版本，避免中文母版覆盖日语/粤语/英语母版。
    返回同步的角色数量。
    """
    target_language = _normalize_prompt_language(language) if language is not None else None
    count = 0
    root = prompt_constraints_root()
    if not root.is_dir():
        return 0
    for path in root.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("character_name", ""))
        if name == "默认模板":
            continue
        versions = raw.get("versions") if isinstance(raw.get("versions"), dict) else {}
        updated = False
        for vid, vdata in versions.items():
            if not isinstance(vdata, dict):
                continue
            record_language = _normalize_prompt_language(vdata.get("language"))
            if vdata.get("source") == "default" and (target_language is None or record_language == target_language):
                vdata["constraint_text"] = new_text
                vdata["created_at"] = int(time.time())
                if target_language:
                    vdata["language"] = target_language
                updated = True
        if updated:
            raw["updated_at"] = int(time.time())
            save_character_constraints(raw)
            count += 1
    return count


def load_character_constraints(character_name: str) -> dict[str, Any]:
    """
    Load constraint store for a specific character.
    Auto-creates four language-specific default versions if missing.
    """
    path = constraint_file_path(character_name)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("character_name"):
                if ensure_default_prompt_versions(raw):
                    save_character_constraints(raw)
                return raw
        except Exception:
            pass
    store = _default_constraint_store(character_name)
    ensure_default_prompt_versions(store)
    save_character_constraints(store)
    return store


def save_character_constraints(store: dict[str, Any]) -> None:
    """Save constraint store for a character to file."""
    path = constraint_file_path(store.get("character_name", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    store["updated_at"] = int(time.time())
    _write_json(path, store)


def _next_version_id(existing: set[str]) -> str:
    """Generate next version ID like 'v2' from existing set {'v1'}."""
    nums = []
    for v in existing:
        try:
            nums.append(int(v[1:]))
        except (ValueError, IndexError):
            pass
    next_num = max(nums) + 1 if nums else 1
    return f"v{next_num}"


def upsert_constraint_version(
    character_name: str,
    version_id: str | None,
    constraint_text: str,
    *,
    name: str = "",
    source: str = "manual",
    language: Any | None = None,
) -> tuple[dict[str, Any], str]:
    """
    添加或更新角色的约束版本。

    Args:
        character_name: 角色名
        version_id: 版本ID（如 "v1"），None 时自动生成下一个版本号
        constraint_text: 完整约束文本（含标记）
        name: 版本名称（用户自定义）
        source: 版本来源（"manual", "imported", "migrated"）
        language: 该版本适配的语音语言（zh/ja/yue/en）

    Returns:
        (更新后的 store, 实际使用的 version_id)
    """
    store = load_character_constraints(character_name)

    if version_id is None:
        existing = set(store["versions"].keys())
        version_id = _next_version_id(existing)

    old_record = store["versions"].get(version_id)
    old_language = None
    if isinstance(old_record, dict):
        old_language = _normalize_prompt_language(old_record.get("language"))
    record_language = (
        _normalize_prompt_language(language)
        or old_language
        or _prompt_language_from_version_id(version_id)
    )

    record = {
        "name": name or "",
        "constraint_text": constraint_text,
        "created_at": int(time.time()),
        "source": source,
    }
    if record_language:
        record["language"] = record_language
    if isinstance(old_record, dict) and old_record.get("sort_order") is not None:
        record["sort_order"] = old_record.get("sort_order")
    elif record_language and version_id == DEFAULT_PROMPT_VERSION_IDS.get(record_language):
        record["sort_order"] = (PROMPT_LANGUAGE_CODES.index(record_language) + 1) * 10

    store["versions"][version_id] = record

    if store["selected_version"] is None or store["selected_version"] not in store["versions"]:
        store["selected_version"] = version_id

    save_character_constraints(store)
    return store, version_id


def remove_constraint_version(character_name: str, version_id: str) -> bool:
    """
    Remove a specific constraint version.
    Returns False if version doesn't exist or is the last version.
    """
    store = load_character_constraints(character_name)
    if version_id not in store["versions"]:
        return False
    if len(store["versions"]) <= 1:
        return False

    del store["versions"][version_id]

    if store["selected_version"] == version_id:
        store["selected_version"] = next(iter(store["versions"].keys()))

    save_character_constraints(store)
    return True


def select_constraint_version(character_name: str, version_id: str) -> bool:
    """Select a constraint version for a character. Returns False if version doesn't exist."""
    store = load_character_constraints(character_name)
    if version_id not in store["versions"]:
        return False
    store["selected_version"] = version_id
    save_character_constraints(store)
    return True


def get_character_constraint_record(character_name: str) -> dict[str, Any] | None:
    """
    Get the currently selected constraint record for a character.
    Returns None if no constraint is configured.
    """
    store = load_character_constraints(character_name)
    selected = store.get("selected_version")
    if not selected or selected not in store["versions"]:
        return None
    record = store["versions"][selected]
    return dict(record) if isinstance(record, dict) else None


def get_character_constraint_record_for_language(
    character_name: str,
    voice_language: Any = "auto",
) -> dict[str, Any] | None:
    """
    获取角色在运行时语音语言下的约束记录。

    优先级：手改版本 > 固定槽位默认版本 > None。
    设置页的语言下拉框仅用于编辑，不影响注入语言；注入语言跟随主程序当前语音语言。
    """
    store = load_character_constraints(character_name)
    language = _runtime_prompt_language(voice_language)
    version_id = DEFAULT_PROMPT_VERSION_IDS[language]
    custom_record = _find_custom_constraint_record_for_language(store, language)
    if custom_record:
        return custom_record
    record = store.get("versions", {}).get(version_id)
    return dict(record) if isinstance(record, dict) else None


def _constraint_record_has_custom_text(record: dict[str, Any], language: str) -> bool:
    text = str(record.get("constraint_text") or "").strip()
    if not text:
        return False
    if record.get("source") != "default":
        return True
    return text != build_default_constraint_text(language).strip()


def _find_custom_constraint_record_for_language(
    store: dict[str, Any],
    language: str,
) -> dict[str, Any] | None:
    versions = store.get("versions", {})
    if not isinstance(versions, dict):
        return None

    # 固定四语言槽位里的手改版本优先，确保运行时读到角色自己的模板。
    default_vid = DEFAULT_PROMPT_VERSION_IDS[language]
    fixed_record = versions.get(default_vid)
    if isinstance(fixed_record, dict) and _constraint_record_has_custom_text(fixed_record, language):
        return dict(fixed_record)

    # 兼容旧版隐藏版本：如果同语言旧版本是手改模板，不再回退到硬编码默认模板。
    for vid, record in versions.items():
        if vid == default_vid or not isinstance(record, dict):
            continue
        # 旧版迁移残留的默认版本可能带着当前语言标记，但文本仍是旧的 auto 模板。
        # 这类默认记录不能盖过固定四语言槽位，否则会把粤语等语言误注入为自动。
        if record.get("source") == "default":
            continue
        record_language = _normalize_prompt_language(record.get("language")) or _prompt_language_from_version_id(vid)
        if record_language != language:
            continue
        if _constraint_record_has_custom_text(record, language):
            return dict(record)

    return None


def get_character_constraint_text(
    character_name: str,
    voice_language: Any = "auto",
) -> str | None:
    """
    获取角色在当前主程序语音语言下的提示词约束文本。

    注入语言跟随主程序当前语音语言，不受设置页语言下拉框影响。
    无记录或标记为 default 时回退到硬编码默认模板。
    """
    language = _runtime_prompt_language(voice_language)
    record = get_character_constraint_record_for_language(character_name, language)
    if not record:
        return build_default_constraint_text(language)
    text = record.get("constraint_text")
    if text:
        return str(text)
    if record.get("source") == "default":
        return build_default_constraint_text(language)
    return None


def list_constraint_versions(character_name: str) -> list[tuple[str, dict[str, Any]]]:
    """List all versions for a character with metadata."""
    store = load_character_constraints(character_name)
    return sorted(
        [(vid, dict(vdata)) for vid, vdata in store["versions"].items()],
        key=_constraint_version_sort_key,
    )


def migrate_constraints_from_v0() -> dict[str, int]:
    """
    Migrate old global constraints from v0.x to per-character versioned system.
    Called on first run if old data exists.
    Returns dict mapping character_name -> number of versions migrated.
    """
    migrated = {}
    old_constraints = _load_old_prompt_constraints()

    characters = load_characters()
    for char in characters:
        name = str(char.get("name", "")).strip()
        if not name:
            continue

        old_key = str(char.get("prompt_constraint_key") or "").strip()
        if old_key and old_key in old_constraints:
            constraint_text = old_constraints[old_key]
        elif "default" in old_constraints:
            constraint_text = old_constraints["default"]
        else:
            continue

        upsert_constraint_version(name, "v1", constraint_text, source="migrated")
        migrated[name] = 1

    return migrated


# ----------------------------------------------------------------------
# Legacy compatibility functions (kept for prompt_hook.py)
# ----------------------------------------------------------------------


def remove_prompt_constraint_text(text: str) -> str:
    """从模板文本中移除约束标记及其内容（同时兼容新旧两套标记）。"""
    src = text or ""
    # 兼容旧 MiniMax 插件标记，否则升级到 Cloud TTS 后旧块删不掉会重复追加
    patterns = (
        (CONSTRAINT_START, CONSTRAINT_END),
        (LEGACY_CONSTRAINT_START, LEGACY_CONSTRAINT_END),
    )
    for start, end in patterns:
        # 非贪婪匹配约束块，连同尾随空白一并删除
        pattern = re.compile(
            rf"{re.escape(start)}.*?{re.escape(end)}[ \t]*(?:\r?\n)*",
            re.DOTALL,
        )
        src = pattern.sub("", src)
    return src.lstrip("\r\n")


def unwrap_prompt_constraint_text(text: str) -> str:
    """从约束块中提取内部正文（去掉 <<<MARKER>>> 包裹）。"""
    src = (text or "").strip()
    for start, end in (
        (CONSTRAINT_START, CONSTRAINT_END),
        (LEGACY_CONSTRAINT_START, LEGACY_CONSTRAINT_END),
    ):
        pattern = re.compile(
            rf"{re.escape(start)}\s*(.*?)\s*{re.escape(end)}",
            re.DOTALL,
        )
        match = pattern.search(src)
        if match:
            return match.group(1).strip()
    return src


def wrap_prompt_constraint_text(body: str) -> str:
    """用 Cloud TTS 约束标记包裹正文。"""
    clean = (body or "").strip()
    return f"{CONSTRAINT_START}\n{clean}\n{CONSTRAINT_END}"


def combine_prompt_constraint_texts(texts: list[str]) -> str:
    """合并多个角色的提示词约束为一个约束块。

    - 单角色：直接包裹该角色的约束正文
    - 多角色：加通用 guard（强制 translate 字段规则），再按角色分列各自的语言约束
    - 相同正文去重
    """
    bodies: list[str] = []
    seen: set[str] = set()
    for text in texts:
        body = unwrap_prompt_constraint_text(text)
        key = body.strip()
        if not key or key in seen:
            continue
        bodies.append(body)
        seen.add(key)
    if not bodies:
        return ""
    if len(bodies) == 1:
        return wrap_prompt_constraint_text(bodies[0])
    guard = (
        "Cloud TTS 通用强制规则：每条角色对白必须输出 translate 字段。"
        "speech 字段只用于屏幕显示，必须保持自然简体中文且不能包含语气标签；"
        "translate 字段用于语音合成，允许按角色语音目标语言改写并加入 Cloud TTS 支持的语气标签。"
    )
    combined = "以下约束按不同角色语音语言合并；生成对白时，请按对应角色的语音目标处理 translate 字段。\n\n"
    combined += f"{guard}\n\n"
    combined += "\n\n".join(f"【角色语音约束 {i}】\n{body}" for i, body in enumerate(bodies, start=1))
    return wrap_prompt_constraint_text(combined)


def protect_tone_tags(text: str) -> tuple[str, dict[str, str]]:
    """在主程序 remove_parentheses 执行前，把 19 种 MiniMax 语气标签替换为无括号占位符。

    返回 (受保护文本, 占位符→原始标签映射)。
    占位符不含括号，所以不会在后续的括号清理中被误删。
    """
    protected = str(text or "")
    placeholders: dict[str, str] = {}
    for index, tag in enumerate(CLOUD_TTS_TONE_TAGS):
        if tag not in protected:
            continue
        token = f"__CLOUD_TTS_TONE_TAG_{index}__"
        protected = protected.replace(tag, token)
        placeholders[token] = tag
    return protected, placeholders


def restore_tone_tags(text: str, placeholders: dict[str, str]) -> str:
    """在 remove_parentheses 执行完毕后，把占位符还原为原始 MiniMax 语气标签。"""
    restored = str(text or "")
    for token, tag in placeholders.items():
        restored = restored.replace(token, tag)
    return restored


def add_prompt_constraint_text(text: str, constraint: str | None = None) -> str:
    """把提示词约束块注入到 system prompt 最顶部（先清旧块，再加新块）。"""
    constraint_text = constraint or ""
    base = remove_prompt_constraint_text(text)
    if base:
        base = base.lstrip("\r\n")
        return f"{constraint_text}\n\n{base}"
    return f"{constraint_text}\n"


def sync_prompt_constraint_text(
    text: str, *, active: bool, constraint_key: str | None = None
) -> str:
    """按主 TTS 选择同步 MiniMax 语气约束（legacy兼容版本）."""
    if active:
        return add_prompt_constraint_text(text, constraint_key)
    return remove_prompt_constraint_text(text)


def project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "data" / "config").is_dir():
        return cwd.resolve()
    return Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root() / p


def is_cloud_tts_provider(provider: str | None) -> bool:
    slug = str(provider or "").strip().lower()
    return slug == PROVIDER_SLUG or slug in LEGACY_PROVIDER_SLUGS


def is_qwen_tts_provider(provider: str | None) -> bool:
    slug = str(provider or "").strip().lower()
    return slug == QWEN_PROVIDER_SLUG


def is_gpt_sovits_provider(provider: str | None) -> bool:
    slug = str(provider or "").strip().lower()
    return slug == GPT_SOVITS_PROVIDER_SLUG


def is_any_cloud_tts_provider(provider: str | None) -> bool:
    slug = str(provider or "").strip().lower()
    return slug in ALL_CLOUD_TTS_SLUGS or slug in LEGACY_PROVIDER_SLUGS


def is_cloud_tts_entry(entry: str | None) -> bool:
    value = str(entry or "").strip()
    return value == PLUGIN_ENTRY or value in LEGACY_PLUGIN_ENTRIES


def api_config_path() -> Path:
    return project_root() / "data" / "config" / "api.yaml"


def system_config_path() -> Path:
    return project_root() / "data" / "config" / "system_config.yaml"


def characters_config_path() -> Path:
    return project_root() / "data" / "config" / "characters.yaml"


def plugin_manifest_path() -> Path:
    return project_root() / "data" / "config" / "plugins.yaml"


def plugin_package_root() -> Path:
    return Path(__file__).resolve().parent


def plugin_data_root() -> Path:
    return project_root() / "data" / "plugins" / PLUGIN_ID.replace("/", "_")


def legacy_plugin_data_root() -> Path:
    return project_root() / "data" / "plugins" / LEGACY_PLUGIN_ID.replace("/", "_")


def legacy_package_voice_store_root() -> Path:
    return project_root() / "plugins" / "minimax_tts" / "voices"


def legacy_package_voice_store_roots() -> list[Path]:
    roots = [plugin_package_root() / "voices", legacy_package_voice_store_root()]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            key = root.resolve()
        except OSError:
            key = root
        if key not in seen:
            out.append(root)
            seen.add(key)
    return out


def migrate_legacy_plugin_data_root() -> None:
    legacy = legacy_plugin_data_root()
    target = plugin_data_root()
    try:
        if legacy.resolve() == target.resolve():
            return
    except OSError:
        if legacy == target:
            return
    if not legacy.is_dir():
        return
    for src in legacy.rglob("*"):
        dst = target / src.relative_to(legacy)
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def voice_store_root(provider_slug: str = PROVIDER_SLUG) -> Path:
    return plugin_data_root() / provider_slug / "voices"


def voice_defaults_path(provider_slug: str = PROVIDER_SLUG) -> Path:
    return voice_store_root(provider_slug) / "_defaults.json"


def migrate_voice_store_to_provider(provider_slug: str) -> None:
    """将旧版共享 voice 目录迁移到 MiniMax per-provider 子目录。旧数据都是 MiniMax 的，Qwen 不迁移。"""
    if provider_slug != PROVIDER_SLUG:
        return
    old_root = plugin_data_root() / "voices"
    new_root = voice_store_root(provider_slug)
    if not old_root.is_dir() or new_root.is_dir():
        return
    new_root.mkdir(parents=True, exist_ok=True)
    for src in sorted(old_root.glob("*.json")):
        dst = new_root / src.name
        if not dst.exists():
            try:
                dst.write_bytes(src.read_bytes())
            except OSError:
                pass


def migrate_legacy_voice_store() -> None:
    target = voice_store_root()
    target.mkdir(parents=True, exist_ok=True)
    for legacy in legacy_package_voice_store_roots():
        if not legacy.is_dir():
            continue
        for src in legacy.glob("*.json"):
            dst = target / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
    # 旧文件是用户运行数据，迁移时只复制不删除，避免插件升级过程误伤。


def read_yaml(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default if raw is None else raw


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def load_api_config() -> dict[str, Any]:
    raw = read_yaml(api_config_path(), {})
    return raw if isinstance(raw, dict) else {}


def load_system_config() -> dict[str, Any]:
    raw = read_yaml(system_config_path(), {})
    return raw if isinstance(raw, dict) else {}


def current_system_voice_language() -> str:
    data = load_system_config()
    return normalize_voice_language_code(data.get("voice_language") or "auto")


def save_api_config(data: dict[str, Any]) -> None:
    write_yaml(api_config_path(), data)


def migrate_legacy_api_config() -> None:
    data = load_api_config()
    changed = False
    provider = str(data.get("tts_provider") or "").strip()
    if is_cloud_tts_provider(provider) and provider.lower() != PROVIDER_SLUG:
        data["tts_provider"] = PROVIDER_SLUG
        changed = True

    all_extra = data.get("tts_extra_configs")
    if isinstance(all_extra, dict):
        current = all_extra.get(PROVIDER_SLUG)
        legacy = all_extra.get(LEGACY_PROVIDER_SLUG)
        if not isinstance(current, dict) and isinstance(legacy, dict):
            all_extra[PROVIDER_SLUG] = dict(legacy)
            data["tts_extra_configs"] = all_extra
            changed = True

    if changed:
        save_api_config(data)


def get_cloud_extra() -> dict[str, Any]:
    migrate_legacy_api_config()
    data = load_api_config()
    all_extra = data.get("tts_extra_configs")
    if not isinstance(all_extra, dict):
        return {}
    cur = all_extra.get(PROVIDER_SLUG)
    if not isinstance(cur, dict):
        cur = all_extra.get(LEGACY_PROVIDER_SLUG)
    return dict(cur) if isinstance(cur, dict) else {}


def get_qwen_extra() -> dict[str, Any]:
    data = load_api_config()
    all_extra = data.get("tts_extra_configs")
    if not isinstance(all_extra, dict):
        return {}
    cur = all_extra.get(QWEN_PROVIDER_SLUG)
    return dict(cur) if isinstance(cur, dict) else {}


def get_gpt_sovits_extra() -> dict[str, Any]:
    data = load_api_config()
    all_extra = data.get("tts_extra_configs")
    if not isinstance(all_extra, dict):
        return {}
    cur = all_extra.get(GPT_SOVITS_PROVIDER_SLUG)
    return dict(cur) if isinstance(cur, dict) else {}


def set_cloud_extra(extra: dict[str, Any]) -> None:
    migrate_legacy_api_config()
    data = load_api_config()
    all_extra = data.get("tts_extra_configs")
    if not isinstance(all_extra, dict):
        all_extra = {}
    merged = adapter_config_from_values(dict(all_extra.get(PROVIDER_SLUG) or {}))
    incoming = adapter_config_from_values(extra)
    for key, value in list(incoming.items()):
        if not str(value or "").strip() and str(merged.get(key) or "").strip():
            incoming.pop(key, None)
    merged.update(incoming)
    all_extra[PROVIDER_SLUG] = merged
    data["tts_extra_configs"] = all_extra
    save_api_config(data)


def set_qwen_extra(extra: dict[str, Any]) -> None:
    """保存 Qwen TTS adapter 配置到 api.yaml。"""
    data = load_api_config()
    all_extra = data.get("tts_extra_configs")
    if not isinstance(all_extra, dict):
        all_extra = {}
    merged = adapter_config_from_values(dict(all_extra.get(QWEN_PROVIDER_SLUG) or {}))
    incoming = adapter_config_from_values(extra)
    for key, value in list(incoming.items()):
        if not str(value or "").strip() and str(merged.get(key) or "").strip():
            incoming.pop(key, None)
    merged.update(incoming)
    all_extra[QWEN_PROVIDER_SLUG] = merged
    data["tts_extra_configs"] = all_extra
    save_api_config(data)




def set_gpt_sovits_extra(extra: dict[str, Any]) -> None:
    """Save GPT-SoVITS API adapter connection config to api.yaml."""
    data = load_api_config()
    all_extra = data.get("tts_extra_configs")
    if not isinstance(all_extra, dict):
        all_extra = {}
    merged = adapter_config_from_values(dict(all_extra.get(GPT_SOVITS_PROVIDER_SLUG) or {}))
    incoming = adapter_config_from_values(extra)
    for key, value in list(incoming.items()):
        if not str(value or "").strip() and str(merged.get(key) or "").strip():
            incoming.pop(key, None)
    merged.update(incoming)
    all_extra[GPT_SOVITS_PROVIDER_SLUG] = merged
    data["tts_extra_configs"] = all_extra
    save_api_config(data)

def adapter_config_from_values(values: dict[str, Any]) -> dict[str, Any]:
    """抽取官方 adapter extra 配置，供 api.yaml 持久化。"""
    return {k: values[k] for k in ADAPTER_CONFIG_KEYS if k in values}


def plugin_state_from_values(values: dict[str, Any]) -> dict[str, Any]:
    """只保留插件私有状态，避免把 API key 等 adapter 参数写进插件源码区。"""
    return {k: values[k] for k in PLUGIN_STATE_KEYS if k in values}


def coerce_voice_language_map(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = {}
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, item in value.items():
        name = str(key or "").strip()
        code = normalize_voice_language_code(item)
        if name and code != "auto":
            out[name] = code
    return out


def migrate_api_extra_to_plugin_state(plugin_root: Path | None = None) -> None:
    """把旧版 api.yaml 中的行为参数迁回插件数据目录，只留下 API 凭证。"""
    extra = get_cloud_extra()
    state_values = plugin_state_from_values(extra)
    api_values = adapter_config_from_values(extra)
    if not state_values:
        if extra != api_values:
            set_cloud_extra(api_values)
        return
    root = plugin_root or plugin_data_root()
    current = _read_json(root / "config.json", {})
    current_cfg = dict(current) if isinstance(current, dict) else {}
    merged = dict(state_values)
    merged.update(
        {k: v for k, v in current_cfg.items() if v not in (None, "", {}, [])}
    )
    _write_plugin_config_file(root, plugin_state_from_values(merged))
    set_cloud_extra(api_values)


def set_tts_provider(provider: str) -> None:
    data = load_api_config()
    data["tts_provider"] = provider
    save_api_config(data)


def current_tts_provider() -> str:
    data = load_api_config()
    return str(data.get("tts_provider") or "").strip()


def clear_cloud_tts_provider_if_selected() -> bool:
    """插件被禁用时，避免主菜单保留一个下一次无法注册的 Cloud TTS 引擎。"""
    data = load_api_config()
    provider = str(data.get("tts_provider") or "")
    if is_any_cloud_tts_provider(provider):
        data["tts_provider"] = "none"
        save_api_config(data)
        return True
    return False


def remove_cloud_extra() -> None:
    data = load_api_config()
    all_extra = data.get("tts_extra_configs")
    if isinstance(all_extra, dict):
        all_extra.pop(PROVIDER_SLUG, None)
        all_extra.pop(LEGACY_PROVIDER_SLUG, None)
        all_extra.pop(QWEN_PROVIDER_SLUG, None)
        all_extra.pop(GPT_SOVITS_PROVIDER_SLUG, None)
        data["tts_extra_configs"] = all_extra
        save_api_config(data)


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return raw


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _voice_file_stem(character_name: str) -> str:
    name = (character_name or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    safe = safe.strip("._-")[:48]
    if not safe:
        safe = "character"
    return f"{safe}_{short_hash(name, 10)}"


def voice_file_path(character_name: str, provider_slug: str = PROVIDER_SLUG) -> Path:
    return voice_store_root(provider_slug) / f"{_voice_file_stem(character_name)}.json"


def _normalize_voice_record(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        rec = dict(item)
        voice_id = str(rec.get("voice_id") or rec.get("id") or "").strip()
    else:
        rec = {}
        voice_id = str(item or "").strip()
    if not voice_id:
        return None
    rec["voice_id"] = voice_id
    rec.setdefault("created_at", 0)
    return rec


def _voice_stores_from_config(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stores: dict[str, dict[str, Any]] = {}
    raw_map = cfg.get("voice_id_map")
    raw_versions = cfg.get("voice_id_versions")
    if not isinstance(raw_map, dict):
        raw_map = {}
    if not isinstance(raw_versions, dict):
        raw_versions = {}
    for name, voice_id in raw_map.items():
        character_name = str(name or "").strip()
        vid = str(voice_id or "").strip()
        if not character_name or not vid:
            continue
        stores.setdefault(
            character_name,
            {
                "character_name": character_name,
                "selected_voice_id": vid,
                "voices": [],
            },
        )
    for name, items in raw_versions.items():
        character_name = str(name or "").strip()
        if not character_name:
            continue
        raw_items = items if isinstance(items, list) else [items]
        store = stores.setdefault(
            character_name,
            {
                "character_name": character_name,
                "selected_voice_id": "",
                "voices": [],
            },
        )
        seen = {
            str(rec.get("voice_id") or "").strip()
            for rec in store.get("voices", [])
            if isinstance(rec, dict)
        }
        for item in raw_items:
            rec = _normalize_voice_record(item)
            if not rec:
                continue
            vid = str(rec["voice_id"])
            if vid in seen:
                continue
            store.setdefault("voices", []).append(rec)
            seen.add(vid)
    for store in stores.values():
        selected = str(store.get("selected_voice_id") or "").strip()
        if selected and not any(
            str(rec.get("voice_id") or "").strip() == selected
            for rec in store.get("voices", [])
            if isinstance(rec, dict)
        ):
            store.setdefault("voices", []).append(
                {
                    "voice_id": selected,
                    "source": "selected",
                    "created_at": 0,
                }
            )
    return stores


def load_voice_stores(provider_slug: str = PROVIDER_SLUG) -> dict[str, dict[str, Any]]:
    migrate_legacy_voice_store()
    migrate_voice_store_to_provider(provider_slug)
    root = voice_store_root(provider_slug)
    stores: dict[str, dict[str, Any]] = {}
    file_map: dict[str, list[Path]] = {}
    if not root.is_dir():
        return stores
    for path in sorted(root.glob("*.json")):
        if path.name == "_defaults.json":
            continue
        raw = _read_json(path, {})
        if not isinstance(raw, dict):
            continue
        character_name = str(raw.get("character_name") or "").strip()
        if not character_name:
            continue
        file_map.setdefault(character_name, []).append(path)
        voices: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw.get("voices") or []:
            rec = _normalize_voice_record(item)
            if not rec:
                continue
            vid = str(rec["voice_id"])
            if vid in seen:
                continue
            voices.append(rec)
            seen.add(vid)
        selected = str(raw.get("selected_voice_id") or "").strip()
        if not selected and voices:
            selected = str(voices[-1].get("voice_id") or "").strip()
        this_updated = int(raw.get("updated_at") or 0)
        if character_name in stores:
            existing = stores[character_name]
            existing_vids = {str(v.get("voice_id") or "").strip() for v in existing["voices"] if isinstance(v, dict)}
            for v in voices:
                if str(v.get("voice_id") or "").strip() not in existing_vids:
                    existing["voices"].append(v)
                    existing_vids.add(str(v.get("voice_id") or "").strip())
            if this_updated > existing["updated_at"]:
                if selected:
                    existing["selected_voice_id"] = selected
                existing["updated_at"] = this_updated
        else:
            stores[character_name] = {
                "character_name": character_name,
                "selected_voice_id": selected,
                "voices": voices,
                "updated_at": this_updated,
            }
    # 清理非 canonical 文件名的重复旧文件
    for character_name, paths in file_map.items():
        if len(paths) <= 1:
            continue
        canonical = voice_file_path(character_name, provider_slug)
        for p in paths:
            try:
                if p.resolve() != canonical.resolve() and p.exists():
                    p.unlink()
            except OSError:
                pass
    return stores


def load_voice_defaults(provider_slug: str = PROVIDER_SLUG) -> dict[str, Any]:
    migrate_legacy_voice_store()
    migrate_voice_store_to_provider(provider_slug)
    raw = _read_json(voice_defaults_path(provider_slug), {})
    return dict(raw) if isinstance(raw, dict) else {}


def save_voice_defaults(default_voice_id: str | None, provider_slug: str = PROVIDER_SLUG) -> None:
    voice_id = str(default_voice_id or "").strip()
    path = voice_defaults_path(provider_slug)
    if not voice_id:
        if path.exists():
            path.unlink()
        return
    _write_json(path, {"default_voice_id": voice_id})


def voice_config_from_files(provider_slug: str = PROVIDER_SLUG) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    stores = load_voice_stores(provider_slug)
    voice_map: dict[str, str] = {}
    versions: dict[str, list[dict[str, Any]]] = {}
    for name, store in stores.items():
        selected = str(store.get("selected_voice_id") or "").strip()
        voices = [dict(rec) for rec in store.get("voices", []) if isinstance(rec, dict)]
        if selected:
            voice_map[name] = selected
        if voices:
            versions[name] = voices
    return voice_map, versions


def save_voice_config_to_files(
    voice_id_map: dict[str, str] | None,
    voice_id_versions: dict[str, Any] | None,
    provider_slug: str = PROVIDER_SLUG,
) -> None:
    cfg = {
        "voice_id_map": voice_id_map or {},
        "voice_id_versions": voice_id_versions or {},
    }
    stores = _voice_stores_from_config(cfg)
    for character_name, store in stores.items():
        store["updated_at"] = int(time.time())
        _write_json(voice_file_path(character_name, provider_slug), store)


def upsert_voice_record(
    character_name: str,
    voice_id: str,
    record: dict[str, Any] | None = None,
    *,
    selected: bool = True,
    provider_slug: str = PROVIDER_SLUG,
) -> None:
    name = (character_name or "").strip()
    vid = (voice_id or "").strip()
    if not name or not vid:
        return
    path = voice_file_path(name, provider_slug)
    raw = _read_json(path, {})
    if not isinstance(raw, dict):
        raw = {}
    voices = []
    seen: set[str] = set()
    for item in raw.get("voices") or []:
        rec = _normalize_voice_record(item)
        if not rec:
            continue
        cur = str(rec["voice_id"])
        if cur in seen:
            continue
        voices.append(rec)
        seen.add(cur)
    if vid not in seen:
        rec = dict(record or {})
        rec["voice_id"] = vid
        rec.setdefault("created_at", int(time.time()))
        voices.append(rec)
    raw["character_name"] = name
    raw["voices"] = voices
    if selected:
        raw["selected_voice_id"] = vid
    else:
        raw.setdefault("selected_voice_id", vid)
    raw["updated_at"] = int(time.time())
    _write_json(path, raw)


def _strip_voice_config(data: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(data)
    for key in (
        "voice_id_map",
        "voice_id_versions",
        "voice_map",
        "voice_id",
        "voice_id_prefix",
        "default_voice_id",
    ):
        cleaned.pop(key, None)
    return cleaned


def _write_plugin_config_file(plugin_root: Path, data: dict[str, Any]) -> None:
    plugin_root.mkdir(parents=True, exist_ok=True)
    _write_json(plugin_root / "config.json", data)


def _merge_voice_files_into_config(data: dict[str, Any], provider_slug: str = PROVIDER_SLUG) -> dict[str, Any]:
    merged = dict(data)
    raw_default_voice_id = str(merged.get("default_voice_id") or "").strip()
    voice_map, versions = voice_config_from_files(provider_slug)
    defaults = load_voice_defaults(provider_slug)
    # 旧 config.json 中的 voice 数据均为 MiniMax 时代遗留，只迁入 minimax-tts
    if raw_default_voice_id and "default_voice_id" not in defaults:
        save_voice_defaults(raw_default_voice_id, PROVIDER_SLUG)
        if provider_slug == PROVIDER_SLUG:
            defaults = load_voice_defaults(provider_slug)
    old_stores = _voice_stores_from_config(merged)
    if old_stores:
        old_map: dict[str, str] = {}
        old_versions: dict[str, list[dict[str, Any]]] = {}
        for name, store in old_stores.items():
            selected = str(store.get("selected_voice_id") or "").strip()
            voices = [dict(rec) for rec in store.get("voices", []) if isinstance(rec, dict)]
            if selected and name not in voice_map:
                old_map[name] = selected
            if voices and name not in versions:
                old_versions[name] = voices
        if old_map or old_versions:
            save_voice_config_to_files(
                {**old_map, **voice_map},
                {**old_versions, **versions},
                provider_slug=PROVIDER_SLUG,
            )
            if provider_slug == PROVIDER_SLUG:
                voice_map, versions = voice_config_from_files(provider_slug)
    merged = _strip_voice_config(merged)
    if voice_map:
        merged["voice_id_map"] = voice_map
    if versions:
        merged["voice_id_versions"] = versions
    default_voice_id = str(defaults.get("default_voice_id") or "").strip()
    if default_voice_id:
        merged["default_voice_id"] = default_voice_id
    return merged


def load_plugin_config(plugin_root: Path, provider_slug: str = PROVIDER_SLUG) -> dict[str, Any]:
    roots = [plugin_root, plugin_data_root(), plugin_package_root()]
    seen: set[Path] = set()
    paths: list[Path] = []
    for root in roots:
        try:
            path = (root / "config.json").resolve()
        except OSError:
            path = root / "config.json"
        if path not in seen:
            paths.append(path)
            seen.add(path)
    for path in paths:
        if not path.is_file():
            continue
        raw = _read_json(path, {})
        if isinstance(raw, dict):
            return _merge_voice_files_into_config(raw, provider_slug)
    return {}


def load_plugin_base_config() -> dict[str, Any]:
    for root in (plugin_data_root(), plugin_package_root()):
        raw = _read_json(root / "config.json", {})
        if isinstance(raw, dict):
            return dict(raw)
    return {}


def _bool_config_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def save_plugin_config(plugin_root: Path, data: dict[str, Any], provider_slug: str = PROVIDER_SLUG) -> None:
    voice_map = data.get("voice_id_map")
    voice_versions = data.get("voice_id_versions")
    save_voice_defaults(str(data.get("default_voice_id") or ""), provider_slug)
    if isinstance(voice_map, dict) or isinstance(voice_versions, dict):
        save_voice_config_to_files(
            voice_map if isinstance(voice_map, dict) else {},
            voice_versions if isinstance(voice_versions, dict) else {},
            provider_slug=provider_slug,
        )
    _write_plugin_config_file(plugin_root, plugin_state_from_values(data))


def migrate_package_config_to_data_root() -> None:
    src = plugin_package_root() / "config.json"
    dst_root = plugin_data_root()
    dst = dst_root / "config.json"
    if not src.is_file():
        return
    src_cfg = load_plugin_config(plugin_package_root())
    if not src_cfg:
        return
    dst_cfg: dict[str, Any] = {}
    if dst.is_file():
        try:
            raw = json.loads(dst.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        if isinstance(raw, dict):
            dst_cfg = raw
    merged = dict(src_cfg)
    merged.update({k: v for k, v in dst_cfg.items() if v not in (None, "", {}, [])})
    adapter_extra = adapter_config_from_values(merged)
    adapter_extra = {
        key: value
        for key, value in adapter_extra.items()
        if str(value or "").strip()
    }
    if adapter_extra:
        set_cloud_extra(adapter_extra)
    save_plugin_config(dst_root, merged)


def load_runtime_plugin_config() -> dict[str, Any]:
    migrate_legacy_api_config()
    migrate_legacy_plugin_data_root()
    migrate_package_config_to_data_root()
    migrate_api_extra_to_plugin_state()
    return load_plugin_config(plugin_data_root())


def load_characters() -> list[dict[str, Any]]:
    raw = read_yaml(characters_config_path(), [])
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def find_character(name: str) -> dict[str, Any] | None:
    target = (name or "").strip().lower()
    for item in load_characters():
        if str(item.get("name", "")).strip().lower() == target:
            return item
    return None


def save_characters(characters: list[dict[str, Any]]) -> None:
    """保存角色列表到 characters.yaml."""
    write_yaml(characters_config_path(), characters)


def resolve_reference_audio(character: dict[str, Any]) -> Path | None:
    raw = str(character.get("refer_audio_path") or "").strip()
    if not raw:
        return None
    return project_path(raw).resolve()


def suppress_prompt_constraint(seconds: float = 3.0) -> None:
    """插件设置页保存期间短暂抑制模板 hook，避免 UI 刷新误注入系统提示词。"""
    global _PROMPT_CONSTRAINT_SUPPRESS_UNTIL
    until = time.monotonic() + max(0.1, float(seconds or 0.0))
    _PROMPT_CONSTRAINT_SUPPRESS_UNTIL = max(_PROMPT_CONSTRAINT_SUPPRESS_UNTIL, until)


def prompt_constraint_suppressed() -> bool:
    return time.monotonic() < _PROMPT_CONSTRAINT_SUPPRESS_UNTIL


def prompt_constraint_enabled() -> bool:
    """提示词约束开关是否在设置页被用户开启（读 config.json 的 auto_prompt_constraint 字段）。"""
    cfg = load_plugin_base_config()
    if "auto_prompt_constraint" in cfg:
        return bool(cfg.get("auto_prompt_constraint"))
    return False


def prompt_constraint_active() -> bool:
    """运行时判断是否应该注入提示词约束：插件启用 + 主 TTS 是 MiniMax + 约束开关已开 + 未被抑制。"""
    if prompt_constraint_suppressed():
        return False
    if not plugin_manifest_enabled():
        return False
    if not is_cloud_tts_provider(current_tts_provider()):
        return False
    return prompt_constraint_enabled()


def translate_tone_tag_protection_enabled() -> bool:
    cfg = load_plugin_base_config()
    if "protect_translate_tone_tags" in cfg:
        return bool(cfg.get("protect_translate_tone_tags"))
    # 旧版本没有这个开关时保持原行为：Cloud TTS 会自动保护 MiniMax 语气标签。
    return True


def translate_tone_tag_protection_active() -> bool:
    """Cloud TTS 当前启用时，是否保留 translate 中的 MiniMax 语气标签。"""
    if not plugin_manifest_enabled():
        return False
    if not is_cloud_tts_provider(current_tts_provider()):
        return False
    return translate_tone_tag_protection_enabled()


def plugin_manifest_enabled() -> bool:
    raw = read_yaml(plugin_manifest_path(), [])
    if not isinstance(raw, list):
        return True
    found = False
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry = str(item.get("entry") or "").strip()
        if is_cloud_tts_entry(entry):
            found = True
            return bool(item.get("enabled", True))
    return True if not found else False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(text: str, size: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:size]
