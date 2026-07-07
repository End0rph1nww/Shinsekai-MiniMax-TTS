const params = new URLSearchParams(window.location.search);
const app = {
  pluginId: params.get("pluginId") || "",
  pageId: params.get("pageId") || "cloud_tts",
  detail: null,
  values: {},
  providerSlug: "",
  characters: [],
  selectedCharacter: "",
  constraints: null,
  pendingConfirm: null,
  busy: false,
};

const $ = (selector) => document.querySelector(selector);
const appShell = $("#cloud-tts-app");
const statusLine = $("#status-line");
const toastRegion = $("#toast-region");

const GSV_KEYS = [
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
];

const PROVIDER_LABELS = {
  "minimax-tts": "MiniMax TTS",
  "qwen-tts": "Qwen3 TTS",
  "gpt-sovits-api": "GPT SoVITS Cloud",
};

function apiPath(suffix) {
  if (!app.pluginId) {
    throw new Error("missing pluginId");
  }
  return `/api/plugins/${encodeURIComponent(app.pluginId)}${suffix}`;
}

function bridgeToken() {
  const readToken = (source) => {
    const queries = [];
    try {
      const url = new URL(source || "", window.location.href);
      queries.push(url.search);
      const hashQueryIndex = String(url.hash || "").indexOf("?");
      if (hashQueryIndex >= 0) {
        queries.push(url.hash.slice(hashQueryIndex + 1));
      }
    } catch (error) {
      queries.push(source || "");
    }
    for (const item of queries) {
      const query = new URLSearchParams(String(item || "").replace(/^\?/, ""));
      const token = String(query.get("shinsekai_bridge_token") || query.get("token") || "").trim();
      if (token) return token;
    }
    return "";
  };
  const own = readToken(window.location.href);
  if (own) return own;
  try {
    const parent = window.parent && window.parent !== window ? readToken(window.parent.location.href) : "";
    if (parent) return parent;
  } catch (error) {
    // Cross-origin parents cannot be inspected; fall through to top.
  }
  try {
    const top = window.top && window.top !== window ? readToken(window.top.location.href) : "";
    if (top) return top;
  } catch (error) {
    // Ignore and try document.referrer below.
  }
  return readToken(document.referrer || "");
}

async function requestJson(url, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  const token = bridgeToken();
  if (token && !headers["X-Shinsekai-Bridge-Token"]) {
    headers["X-Shinsekai-Bridge-Token"] = token;
  }
  const response = await fetch(url, {
    ...options,
    headers,
  });
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (error) {
      payload = { message: text };
    }
  }
  if (!response.ok) {
    throw new Error(payload.message || payload.error || response.statusText);
  }
  return payload;
}

function pageFromDetail(detail) {
  return (detail?.pages || []).find((page) => page.id === app.pageId) || detail?.pages?.[0] || null;
}

function updateFromPage(page) {
  if (!page) {
    return;
  }
  app.values = page.values || {};
  const providers = app.values.providers || [];
  if (!app.providerSlug) {
    app.providerSlug = app.values.provider || providers[0]?.slug || "minimax-tts";
  }
  if (!providers.some((provider) => provider.slug === app.providerSlug)) {
    app.providerSlug = providers[0]?.slug || "minimax-tts";
  }
}

function activeProvider() {
  const providers = app.values.providers || [];
  return (
    providers.find((provider) => provider.slug === app.providerSlug) ||
    providers[0] ||
    { slug: app.providerSlug || "minimax-tts", label: "MiniMax TTS", models: [], voice_options: [] }
  );
}

function isGsvProvider() {
  return app.providerSlug === "gpt-sovits-api";
}

function isQwenProvider() {
  return app.providerSlug === "qwen-tts";
}

function baseCharacters() {
  const names = new Map();
  for (const char of app.values.characters || []) {
    if (char?.name) {
      names.set(char.name, { name: char.name, prompt_text: char.prompt_text || "" });
    }
  }
  for (const char of app.characters || []) {
    if (char?.name) {
      names.set(char.name, { ...names.get(char.name), ...char });
    }
  }
  return [...names.values()];
}

function selectedCharacterRow() {
  return baseCharacters().find((char) => char.name === app.selectedCharacter) || null;
}

function setBusy(value) {
  app.busy = value;
  appShell.dataset.state = value ? "busy" : "ready";
  for (const button of document.querySelectorAll("button")) {
    if (!button.id || button.id === "confirm-cancel" || button.id === "confirm-ok") {
      continue;
    }
    if (value) {
      if (!button.disabled) {
        button.dataset.busyDisabled = "true";
        button.disabled = true;
      }
    } else if (button.dataset.busyDisabled === "true") {
      delete button.dataset.busyDisabled;
      button.disabled = false;
    }
  }
  updatePlayState();
}

function showToast(message, kind = "info") {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.dataset.kind = kind;
  toast.textContent = message;
  toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function setSelectOptions(select, options, value, { placeholder = "" } = {}) {
  const current = String(value || "");
  select.innerHTML = "";
  if (placeholder) {
    const item = document.createElement("option");
    item.value = "";
    item.textContent = placeholder;
    select.append(item);
  }
  for (const option of options) {
    const item = document.createElement("option");
    item.value = option.value;
    item.textContent = option.label;
    select.append(item);
  }
  select.value = current;
  if (select.value !== current && options.length) {
    select.value = options[0].value;
  }
}

async function loadDetail() {
  if (!app.pluginId) {
    statusLine.textContent = "缺少 pluginId，无法连接宿主。";
    appShell.dataset.state = "offline";
    return;
  }
  const detail = await requestJson(apiPath("/ui"));
  app.detail = detail;
  updateFromPage(pageFromDetail(detail));
}

async function saveConfig(values) {
  const result = await requestJson(apiPath(`/ui/${encodeURIComponent(app.pageId)}/config`), {
    method: "POST",
    body: JSON.stringify({ values }),
  });
  updateFromPage(result.page);
  return result;
}

async function runAction(actionId, values = {}) {
  const result = await requestJson(
    apiPath(`/ui/${encodeURIComponent(app.pageId)}/actions/${encodeURIComponent(actionId)}`),
    {
      method: "POST",
      body: JSON.stringify({ values }),
    },
  );
  updateFromPage(result.page);
  if (
    actionId === "list_characters" &&
    Array.isArray(result.result?.characters) &&
    (!values.provider || values.provider === app.providerSlug)
  ) {
    app.characters = result.result.characters;
  }
  if (result.result?.character?.name) {
    upsertCharacter(result.result.character);
    app.selectedCharacter = result.result.character.name;
  }
  return result;
}

function upsertCharacter(character) {
  const list = [...app.characters];
  const index = list.findIndex((item) => item.name === character.name);
  if (index >= 0) {
    list[index] = { ...list[index], ...character };
  } else {
    list.push(character);
  }
  app.characters = list;
}

async function refreshCharacters({ loadSelectedConstraints = true } = {}) {
  const providerSlug = app.providerSlug;
  const result = await runAction("list_characters", { provider: providerSlug });
  if (providerSlug !== app.providerSlug) {
    return result;
  }
  const chars = result.result?.characters || [];
  if (!app.selectedCharacter || !chars.some((char) => char.name === app.selectedCharacter)) {
    app.selectedCharacter = chars[0]?.name || "";
  }
  if (loadSelectedConstraints && app.selectedCharacter) {
    await loadConstraints({ silent: true });
  }
  return result;
}

async function refreshAll() {
  setBusy(true);
  try {
    await loadDetail();
    await refreshCharacters();
    render();
    statusLine.textContent = `${providerLabel()} 已连接。`;
  } catch (error) {
    statusLine.textContent = "连接失败。";
    showToast(error.message || String(error), "error");
    render();
  } finally {
    setBusy(false);
  }
}

function providerLabel(slug = app.providerSlug) {
  return activeProvider().label || PROVIDER_LABELS[slug] || slug || "Cloud TTS";
}

function render() {
  renderGlobalSettings();
  renderCharacters();
  renderSelectedCharacter();
  renderConstraints();
  updatePlayState();
}

function renderGlobalSettings() {
  const providers = app.values.providers || [];
  const provider = activeProvider();

  setSelectOptions(
    $("#provider-select"),
    providers.map((item) => ({ value: item.slug, label: item.label || PROVIDER_LABELS[item.slug] || item.slug })),
    app.providerSlug,
  );
  setSelectOptions(
    $("#model-select"),
    (provider.models || []).map((model) => ({ value: model, label: model })),
    provider.model,
    { placeholder: provider.models?.length ? "" : "无模型选项" },
  );

  const voiceOptions = provider.voice_options || [];
  const datalist = $("#default-voice-options");
  datalist.innerHTML = voiceOptions
    .map((item) => `<option value="${escapeHtml(item.voice_id)}">${escapeHtml(item.label)}</option>`)
    .join("");
  $("#default-voice-input").value = provider.default_voice_id || "";
  $("#default-voice-field").classList.toggle("is-hidden", isGsvProvider());

  $("#qwen-language-field").classList.toggle("is-hidden", !isQwenProvider());
  $("#qwen-language-select").value = provider.language_type || "Chinese";

  $("#auto-prompt-constraint").checked = Boolean(provider.auto_prompt_constraint);
  $("#protect-tone-tags").checked = provider.protect_translate_tone_tags !== false;

  const showKeyBanner = !isGsvProvider() && provider.key_configured === false;
  $("#key-banner").hidden = !showKeyBanner;
  $("#gsv-global-panel").classList.toggle("is-hidden", !isGsvProvider());

  const gsv = provider.gpt_sovits || {};
  for (const key of GSV_KEYS) {
    const input = $(`#${key}`);
    if (input) {
      input.value = gsv[key] ?? "";
    }
  }
  $("#gpt_sovits_super_sampling").checked = Boolean(gsv.gpt_sovits_super_sampling);
}

function renderCharacters() {
  const chars = baseCharacters();
  $("#character-count").textContent = chars.length ? `${chars.length} 个角色` : "暂无角色。";
  const grid = $("#character-grid");
  if (!chars.length) {
    grid.innerHTML = `<div class="nb-card tool-panel">暂无角色</div>`;
    return;
  }
  grid.innerHTML = chars
    .map((char) => {
      const selected = char.name === app.selectedCharacter ? " is-selected" : "";
      const voice = char.voice_id || "未绑定";
      const ref = char.reference_audio || char.card_reference_audio || "无参考音频";
      return `
        <button class="character-card${selected}" type="button" data-character="${escapeHtml(char.name)}">
          <span class="character-card__name">${escapeHtml(char.name)}</span>
          <span class="character-card__meta">
            <span>voice: ${escapeHtml(voice)}</span>
            <span>ref: ${escapeHtml(ref)}</span>
          </span>
        </button>
      `;
    })
    .join("");
}

function renderSelectedCharacter() {
  const char = selectedCharacterRow();
  const provider = activeProvider();
  const hasCharacter = Boolean(char);

  $("#selected-character-pill").textContent = char?.name || "未选择";
  $("#voice-id-input").value = char?.voice_id || "";
  $("#reference-text-input").value = char?.reference_text || char?.prompt_text || "";
  $("#reference-language-select").value = char?.reference_language || "auto";
  $("#reference-path").textContent = char?.reference_audio || char?.card_reference_audio || "未绑定参考音频";

  const voiceList = $("#voice-option-list");
  voiceList.innerHTML = (provider.voice_options || [])
    .map((item) => `<option value="${escapeHtml(item.voice_id)}">${escapeHtml(item.label)}</option>`)
    .join("");

  const versions = Array.isArray(char?.versions) ? char.versions : [];
  $("#voice-version-list").innerHTML = versions.length
    ? versions
        .map((item, index) => `
          <div class="version-list__item">
            <code>${escapeHtml(item.voice_id || "")}</code>
            <span>${escapeHtml(item.source || `v${index + 1}`)}</span>
          </div>
        `)
        .join("")
    : `<div class="path-box">没有版本记录</div>`;

  const profile = char?.gpt_sovits_profile || {};
  $("#gsv-ref-audio-path").value = profile.ref_audio_path || "";
  $("#gsv-gpt-weights-path").value = profile.gpt_weights_path || "";
  $("#gsv-sovits-weights-path").value = profile.sovits_weights_path || "";
  $("#gsv-prompt-text").value = profile.prompt_text || "";
  $("#gsv-prompt-lang").value = profile.prompt_lang || "";
  $("#gsv-text-lang").value = profile.text_lang || "";

  for (const selector of [
    "#voice-id-input",
    "#reference-upload",
    "#reference-text-input",
    "#reference-language-select",
    "#upload-reference",
    "#clear-reference",
    "#clone-voice",
    "#export-voice",
    "#import-voice",
    "#save-gsv-profile",
    "#save-constraint",
  ]) {
    const node = $(selector);
    if (node) {
      node.disabled = !hasCharacter;
    }
  }
  $("#clone-voice").disabled = !hasCharacter || isGsvProvider();
}

function renderConstraints() {
  const versions = app.constraints?.versions || [];
  const select = $("#constraint-version-select");
  setSelectOptions(
    select,
    versions.map((item) => ({
      value: item.version_id,
      label: `${item.language_label || item.language || ""} / ${item.name || item.version_id}`,
    })),
    app.constraints?.selected_version || versions[0]?.version_id || "",
    { placeholder: versions.length ? "" : "无版本" },
  );
  const current = versions.find((item) => item.version_id === select.value) || versions[0] || {};
  $("#constraint-name-input").value = current.name || "";
  $("#constraint-textarea").value = current.constraint_text || "";
}

function collectGlobalValues() {
  const payload = {
    provider: app.providerSlug,
    model: $("#model-select").value,
    auto_prompt_constraint: $("#auto-prompt-constraint").checked,
    protect_translate_tone_tags: $("#protect-tone-tags").checked,
  };
  if (!isGsvProvider()) {
    payload.default_voice_id = $("#default-voice-input").value.trim();
  }
  if (isQwenProvider()) {
    payload.qwen_language_type = $("#qwen-language-select").value;
  }
  if (isGsvProvider()) {
    for (const key of GSV_KEYS) {
      payload[key] = $(`#${key}`).value.trim();
    }
    payload.gpt_sovits_super_sampling = $("#gpt_sovits_super_sampling").checked;
  }
  return payload;
}

async function handleSaveGlobal() {
  setBusy(true);
  try {
    const result = await saveConfig(collectGlobalValues());
    await refreshCharacters();
    render();
    showToast(result.message || "全局设置已保存。");
  } catch (error) {
    showToast(error.message || String(error), "error");
  } finally {
    setBusy(false);
  }
}

function selectedNameOrThrow() {
  if (!app.selectedCharacter) {
    throw new Error("请先选择角色。");
  }
  return app.selectedCharacter;
}

async function handleBindVoice(event) {
  event.preventDefault();
  setBusy(true);
  try {
    const result = await runAction("bind_voice", {
      provider: app.providerSlug,
      character: selectedNameOrThrow(),
      voice_id: $("#voice-id-input").value.trim(),
    });
    render();
    showToast(result.message || "音色已绑定。");
  } catch (error) {
    showToast(error.message || String(error), "error");
  } finally {
    setBusy(false);
  }
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error || new Error("文件读取失败"));
    reader.onload = () => {
      const result = String(reader.result || "");
      resolve(result.includes(",") ? result.split(",").pop() : result);
    };
    reader.readAsDataURL(file);
  });
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error || new Error("文件读取失败"));
    reader.onload = () => resolve(String(reader.result || ""));
    reader.readAsText(file, "utf-8");
  });
}

async function handleUploadReference() {
  const file = $("#reference-upload").files?.[0];
  if (!file) {
    showToast("请选择参考音频。", "error");
    return;
  }
  setBusy(true);
  try {
    const content_base64 = await readFileAsBase64(file);
    const result = await runAction("upload_reference", {
      provider: app.providerSlug,
      character: selectedNameOrThrow(),
      filename: file.name,
      content_base64,
      text: $("#reference-text-input").value,
      language: $("#reference-language-select").value,
    });
    render();
    showToast(result.message || "参考音频已上传。");
  } catch (error) {
    showToast(error.message || String(error), "error");
  } finally {
    setBusy(false);
  }
}

async function handleClearReference() {
  confirmAction("确定清除当前角色的参考音频？", async () => {
    setBusy(true);
    try {
      const result = await runAction("clear_reference", {
        provider: app.providerSlug,
        character: selectedNameOrThrow(),
      });
      render();
      showToast(result.message || "参考音频已清除。");
    } catch (error) {
      showToast(error.message || String(error), "error");
    } finally {
      setBusy(false);
    }
  });
}

async function handleCloneVoice() {
  setBusy(true);
  $("#clone-progress").hidden = false;
  try {
    const result = await runAction("clone_voice", {
      provider: app.providerSlug,
      character: selectedNameOrThrow(),
    });
    const demoUrl = result.result?.demo_url || "";
    if (demoUrl) {
      $("#clone-audio").src = demoUrl;
      $("#demo-path").textContent = result.result?.demo_path || demoUrl;
    }
    render();
    showToast(result.message || "克隆完成。");
  } catch (error) {
    showToast(error.message || String(error), "error");
  } finally {
    $("#clone-progress").hidden = true;
    setBusy(false);
  }
}

async function handleExportVoice() {
  setBusy(true);
  try {
    const result = await runAction("export_voice", {
      provider: app.providerSlug,
      character: selectedNameOrThrow(),
    });
    const blob = new Blob([JSON.stringify(result.result?.payload || {}, null, 2)], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = result.result?.filename || "cloud_tts_voice.json";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast(result.message || "导出完成。");
  } catch (error) {
    showToast(error.message || String(error), "error");
  } finally {
    setBusy(false);
  }
}

async function handleImportVoiceFile(event) {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) {
    return;
  }
  confirmAction("导入会合并 voice_id 记录并可能更新当前绑定。", async () => {
    setBusy(true);
    try {
      const text = await readFileAsText(file);
      const payload = JSON.parse(text);
      const result = await runAction("import_voice", {
        provider: app.providerSlug,
        character: app.selectedCharacter,
        source_name: file.name,
        payload,
      });
      $("#import-summary").textContent = `已导入 ${result.result?.imported ?? 0} 条记录`;
      await refreshCharacters();
      render();
      showToast(result.message || "导入完成。");
    } catch (error) {
      showToast(error.message || String(error), "error");
    } finally {
      setBusy(false);
    }
  });
}

async function handleSaveGsvProfile() {
  setBusy(true);
  try {
    const result = await runAction("save_gpt_sovits_profile", {
      character: selectedNameOrThrow(),
      profile: {
        ref_audio_path: $("#gsv-ref-audio-path").value,
        gpt_weights_path: $("#gsv-gpt-weights-path").value,
        sovits_weights_path: $("#gsv-sovits-weights-path").value,
        prompt_text: $("#gsv-prompt-text").value,
        prompt_lang: $("#gsv-prompt-lang").value,
        text_lang: $("#gsv-text-lang").value,
      },
    });
    render();
    showToast(result.message || "GSV profile 已保存。");
  } catch (error) {
    showToast(error.message || String(error), "error");
  } finally {
    setBusy(false);
  }
}

async function loadConstraints({ silent = false } = {}) {
  if (!app.selectedCharacter) {
    app.constraints = null;
    renderConstraints();
    return;
  }
  try {
    const result = await runAction("get_constraints", {
      character: app.selectedCharacter,
    });
    app.constraints = result.result || null;
    renderConstraints();
  } catch (error) {
    if (!silent) {
      showToast(error.message || String(error), "error");
    }
  }
}

async function handleSaveConstraint() {
  setBusy(true);
  try {
    const result = await runAction("save_constraints", {
      character: selectedNameOrThrow(),
      version_id: $("#constraint-version-select").value,
      name: $("#constraint-name-input").value,
      text: $("#constraint-textarea").value,
    });
    app.constraints = result.result || null;
    renderConstraints();
    showToast(result.message || "约束已保存。");
  } catch (error) {
    showToast(error.message || String(error), "error");
  } finally {
    setBusy(false);
  }
}

function updateConstraintFieldsFromSelection() {
  const versions = app.constraints?.versions || [];
  const current = versions.find((item) => item.version_id === $("#constraint-version-select").value) || {};
  $("#constraint-name-input").value = current.name || "";
  $("#constraint-textarea").value = current.constraint_text || "";
}

function confirmAction(message, callback) {
  app.pendingConfirm = callback;
  $("#confirm-message").textContent = message;
  $("#confirm-panel").hidden = false;
}

function closeConfirm() {
  app.pendingConfirm = null;
  $("#confirm-panel").hidden = true;
}

function updatePlayState() {
  const audio = $("#clone-audio");
  const play = $("#play-demo");
  play.disabled = app.busy || !audio.src;
  play.textContent = audio.paused ? "播放" : "暂停";
}

function bindEvents() {
  $("#refresh-page").addEventListener("click", refreshAll);
  $("#save-global").addEventListener("click", handleSaveGlobal);
  $("#provider-select").addEventListener("change", async (event) => {
    app.providerSlug = event.target.value;
    app.characters = [];
    app.constraints = null;
    render();
    setBusy(true);
    try {
      await refreshCharacters({ loadSelectedConstraints: false });
      render();
      statusLine.textContent = `${providerLabel()} 已切换。`;
      if (app.selectedCharacter) {
        loadConstraints({ silent: true });
      }
    } catch (error) {
      showToast(error.message || String(error), "error");
    } finally {
      setBusy(false);
    }
  });

  $("#character-grid").addEventListener("click", async (event) => {
    const card = event.target.closest("[data-character]");
    if (!card) {
      return;
    }
    app.selectedCharacter = card.dataset.character || "";
    render();
    await loadConstraints();
    render();
  });

  $("#voice-bind-form").addEventListener("submit", handleBindVoice);
  $("#upload-reference").addEventListener("click", handleUploadReference);
  $("#clear-reference").addEventListener("click", handleClearReference);
  $("#clone-voice").addEventListener("click", handleCloneVoice);
  $("#export-voice").addEventListener("click", handleExportVoice);
  $("#import-voice").addEventListener("click", () => $("#voice-import-file").click());
  $("#voice-import-file").addEventListener("change", handleImportVoiceFile);
  $("#save-gsv-profile").addEventListener("click", handleSaveGsvProfile);
  $("#constraint-version-select").addEventListener("change", updateConstraintFieldsFromSelection);
  $("#save-constraint").addEventListener("click", handleSaveConstraint);
  $("#confirm-cancel").addEventListener("click", closeConfirm);
  $("#confirm-ok").addEventListener("click", () => {
    const callback = app.pendingConfirm;
    closeConfirm();
    if (callback) {
      void callback();
    }
  });
  $("#clone-audio").addEventListener("play", updatePlayState);
  $("#clone-audio").addEventListener("pause", updatePlayState);
  $("#clone-audio").addEventListener("ended", updatePlayState);
  $("#play-demo").addEventListener("click", async () => {
    const audio = $("#clone-audio");
    if (!audio.src) {
      return;
    }
    if (audio.paused) {
      await audio.play();
    } else {
      audio.pause();
    }
    updatePlayState();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      void refreshAll();
    }
  });
}

bindEvents();
void refreshAll();
