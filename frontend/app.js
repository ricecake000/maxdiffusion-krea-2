/* Krea 2 Turbo console — talks to maxdiffusion.serve_krea2 (docs/krea2_api.md). */

// Denoise is the hot stage and takes most of a request; the rest ride a
// descending cyan ramp so neighbouring cool stages stay tellable apart.
const STAGES = [
  { key: "queue_ms", name: "queue", color: "var(--graphite)" },
  { key: "prompt_encoding_ms", name: "text", color: "#57c7d6" },
  { key: "denoise_ms", name: "denoise", color: "var(--heat)" },
  { key: "vae_decode_ms", name: "decode", color: "#3f97a6" },
  { key: "encode_ms", name: "encode", color: "#2c6d79" },
];

// Turbo runs a fixed schedule, so the shape of a request barely moves within a
// size. The first trace of each size replaces these; they only animate request one.
const FALLBACK_TIMING = {
  "1k": { queue_ms: 0, prompt_encoding_ms: 40, denoise_ms: 950, vae_decode_ms: 130, encode_ms: 60 },
  "2k": { queue_ms: 0, prompt_encoding_ms: 40, denoise_ms: 6300, vae_decode_ms: 600, encode_ms: 250 },
};

const FORMAT_NOTES = {
  avif: "small",
  webp: "fast",
  jpeg: "compat",
  png: "lossless",
};

const STORE = {
  endpoint: "krea2.endpoint",
  token: "krea2.token",
  aspect: "krea2.aspect",
  size: "krea2.size",
  format: "krea2.format",
};
const HISTORY_LIMIT = 24;
const MAX_PROMPT = 4000;
const MAX_SEED = 2147483647;
const ASPECT_BOX_PX = 30;

const el = (id) => document.getElementById(id);

const ui = {
  form: el("console-form"),
  prompt: el("prompt"),
  counter: el("counter"),
  sizes: el("sizes"),
  aspects: el("aspects"),
  presetNote: el("preset-note"),
  formats: el("formats"),
  seedModes: Array.from(document.querySelectorAll('input[name="seed-mode"]')),
  seed: el("seed"),
  generate: el("generate"),
  generateLabel: el("generate-label"),
  notice: el("notice"),
  mat: el("mat"),
  image: el("image"),
  matEmpty: el("mat-empty"),
  trace: el("trace"),
  traceTotal: el("trace-total"),
  bar: el("bar"),
  legend: el("legend"),
  readout: el("readout"),
  factSeed: el("fact-seed"),
  factPreset: el("fact-preset"),
  factSize: el("fact-size"),
  factFormat: el("fact-format"),
  factId: el("fact-id"),
  download: el("download"),
  restore: el("restore"),
  session: el("session"),
  sessionCount: el("session-count"),
  strip: el("strip"),
  status: el("status"),
  statusText: el("status-text"),
  settings: el("settings"),
  settingsForm: el("settings-form"),
  endpoint: el("endpoint"),
  token: el("token"),
  openSettings: el("open-settings"),
  forget: el("forget"),
};

function stored(key, fallback) {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}

function store(key, value) {
  try {
    if (value === null || value === "") localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: settings last for this page only */
  }
}

const state = {
  endpoint: stored(STORE.endpoint, "http://localhost:8000"),
  token: stored(STORE.token, ""),
  aspect: stored(STORE.aspect, "1:1"),
  size: stored(STORE.size, "1k"),
  format: stored(STORE.format, null),
  busy: false,
  estimate: structuredClone(FALLBACK_TIMING),
  items: [],
  currentId: null,
  // null until GET /v1/presets answers; then { presets, formats, defaultFormat, maxQueue, timeoutS }.
  server: null,
  // Bumped whenever the endpoint or token changes. GET /v1/presets and /readyz replies that were requested under
  // an older generation are discarded, so a late answer from the old endpoint cannot overwrite the new one's state.
  generation: 0,
};

function bumpGeneration() {
  state.generation += 1;
  return state.generation;
}

const isStale = (generation) => generation !== state.generation;

/* ── formatting ─────────────────────────────────────────── */

const formatMs = (value) =>
  value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;

const presetLabel = (size, aspect) => `${size}@${aspect}`;

/* ── notices ────────────────────────────────────────────── */

function say(message, tone = "error") {
  ui.notice.textContent = message;
  ui.notice.dataset.tone = tone;
  ui.notice.hidden = false;
}

function clearNotice() {
  ui.notice.hidden = true;
  ui.notice.textContent = "";
}

/* ── settings ───────────────────────────────────────────── */

function normalizeEndpoint(value) {
  return value.trim().replace(/\/+$/, "");
}

ui.openSettings.addEventListener("click", () => {
  ui.endpoint.value = state.endpoint;
  ui.token.value = state.token;
  ui.settings.showModal();
});

ui.settingsForm.addEventListener("submit", () => {
  state.endpoint = normalizeEndpoint(ui.endpoint.value) || "http://localhost:8000";
  state.token = ui.token.value.trim();
  store(STORE.endpoint, state.endpoint);
  store(STORE.token, state.token);
  bumpGeneration();
  state.server = null;
  renderPresets();
  clearNotice();
  checkHealth();
});

ui.forget.addEventListener("click", () => {
  state.token = "";
  store(STORE.token, null);
  ui.token.value = "";
  bumpGeneration();
  state.server = null;
  renderPresets();
  ui.settings.close();
});

/* ── prompt ─────────────────────────────────────────────── */

function syncPrompt() {
  const length = ui.prompt.value.trim().length;
  ui.counter.textContent = length;
  ui.counter.parentElement.dataset.over = String(length > MAX_PROMPT);
  syncGenerate();
}

function syncGenerate() {
  const length = ui.prompt.value.trim().length;
  ui.generate.disabled =
    state.busy || state.server === null || length === 0 || length > MAX_PROMPT;
}

ui.prompt.addEventListener("input", syncPrompt);

ui.prompt.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    ui.form.requestSubmit();
  }
});

/* ── presets and formats (from GET /v1/presets) ─────────── */

function presetsOfSize(size) {
  return state.server ? state.server.presets.filter((preset) => preset.image_size === size) : [];
}

function currentPreset() {
  return presetsOfSize(state.size).find((preset) => preset.aspect_ratio === state.aspect) || null;
}

function segmentButton({ name, note, checked, onPick }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "segment";
  button.setAttribute("role", "radio");
  button.setAttribute("aria-checked", String(checked));

  const label = document.createElement("span");
  label.className = "segment-name";
  label.textContent = name;
  button.append(label);

  if (note) {
    const small = document.createElement("span");
    small.className = "segment-note";
    small.textContent = note;
    button.append(small);
  }
  button.addEventListener("click", onPick);
  return button;
}

function aspectButton(preset) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "aspect";
  button.setAttribute("role", "radio");
  button.setAttribute("aria-checked", String(preset.aspect_ratio === state.aspect));
  button.title = `${presetLabel(preset.image_size, preset.aspect_ratio)} · text ≤ ${Math.max(
    ...preset.text_tokens,
  )} tokens`;

  const scale = ASPECT_BOX_PX / Math.max(preset.width, preset.height);
  const box = document.createElement("span");
  box.className = "aspect-box";
  box.setAttribute("aria-hidden", "true");
  box.style.setProperty("--w", `${Math.round(preset.width * scale)}px`);
  box.style.setProperty("--h", `${Math.round(preset.height * scale)}px`);

  const name = document.createElement("span");
  name.className = "aspect-name";
  name.textContent = preset.aspect_ratio;

  const px = document.createElement("span");
  px.className = "aspect-px";
  px.textContent = `${preset.width}×${preset.height}`;

  button.append(box, name, px);
  button.addEventListener("click", () => setPreset(preset.image_size, preset.aspect_ratio));
  return button;
}

function renderPresets() {
  const server = state.server;
  if (server === null) {
    ui.sizes.replaceChildren();
    ui.aspects.replaceChildren();
    ui.formats.replaceChildren();
    ui.presetNote.hidden = false;
    ui.presetNote.textContent = state.token
      ? "The preset list loads from the server once it is ready."
      : "Set the bearer token under Endpoint to load the server's presets.";
    syncGenerate();
    return;
  }

  // A selection saved from another server (or another serving set) may not exist here.
  const sizes = [...new Set(server.presets.map((preset) => preset.image_size))];
  if (!sizes.includes(state.size)) state.size = sizes[0];
  if (!presetsOfSize(state.size).some((preset) => preset.aspect_ratio === state.aspect)) {
    state.aspect = presetsOfSize(state.size)[0].aspect_ratio;
  }
  if (!server.formats.includes(state.format)) state.format = server.defaultFormat;
  store(STORE.size, state.size);
  store(STORE.aspect, state.aspect);
  store(STORE.format, state.format);

  ui.sizes.replaceChildren(
    ...sizes.map((size) =>
      segmentButton({
        name: size,
        note: `${presetsOfSize(size).length} ratio${presetsOfSize(size).length === 1 ? "" : "s"}`,
        checked: size === state.size,
        onPick: () => {
          const keep = presetsOfSize(size).some((preset) => preset.aspect_ratio === state.aspect);
          setPreset(size, keep ? state.aspect : presetsOfSize(size)[0].aspect_ratio);
        },
      }),
    ),
  );
  ui.aspects.replaceChildren(...presetsOfSize(state.size).map(aspectButton));
  ui.formats.replaceChildren(
    ...server.formats.map((format) =>
      segmentButton({
        name: format,
        note: FORMAT_NOTES[format] || "",
        checked: format === state.format,
        onPick: () => setFormat(format),
      }),
    ),
  );

  const preset = currentPreset();
  ui.presetNote.hidden = false;
  ui.presetNote.textContent =
    `${preset.width} × ${preset.height} · prompt ≤ ${Math.max(...preset.text_tokens)} text tokens · ` +
    `queue ${server.maxQueue}, timeout ${server.timeoutS} s`;
  syncGenerate();
  if (!state.busy && state.currentId === null) showIdleTrace();
}

function setPreset(size, aspect) {
  state.size = size;
  state.aspect = aspect;
  renderPresets();
}

function setFormat(format) {
  state.format = format;
  renderPresets();
}

async function loadPresets() {
  if (!state.token) {
    state.server = null;
    renderPresets();
    return;
  }
  const generation = state.generation;
  try {
    const response = await fetch(`${state.endpoint}/v1/presets`, {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    if (isStale(generation)) return;
    if (!response.ok) {
      if (response.status === 401) {
        say("The endpoint rejected this token. Open Endpoint and check it against KREA2_API_TOKEN.");
      } else if (response.status !== 503) {
        say(`Could not load the preset list (HTTP ${response.status}).`, "info");
      }
      return;
    }
    const body = await response.json();
    if (isStale(generation)) return;
    if (!body || !Array.isArray(body.presets) || body.presets.length === 0 || !Array.isArray(body.formats)) {
      throw new Error("The preset response has an invalid shape.");
    }
    state.server = {
      presets: body.presets,
      formats: body.formats,
      defaultFormat: body.default_format || body.formats[0],
      maxQueue: body.max_queue,
      timeoutS: body.request_timeout_s,
    };
    renderPresets();
  } catch {
    if (isStale(generation)) return;
    say("Could not read the preset list returned by the endpoint.", "info");
  }
}

/* ── seed ───────────────────────────────────────────────── */

function seedMode() {
  return ui.seedModes.find((input) => input.checked).value;
}

function syncSeedInput() {
  ui.seed.disabled = seedMode() !== "fixed";
}

for (const input of ui.seedModes) {
  input.addEventListener("change", syncSeedInput);
}

function readSeed() {
  if (seedMode() !== "fixed") return null;
  const raw = ui.seed.value.trim();
  if (raw === "") return null;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0 || value > MAX_SEED) {
    throw new Error(`Seed must be a whole number between 0 and ${MAX_SEED}.`);
  }
  return value;
}

/* ── health ─────────────────────────────────────────────── */

function setStatus(stateName, text) {
  ui.status.dataset.state = stateName;
  ui.statusText.textContent = text;
}

async function checkHealth() {
  const generation = state.generation;
  try {
    const response = await fetch(`${state.endpoint}/readyz`, { cache: "no-store" });
    if (isStale(generation)) return;
    if (response.ok) {
      let text = "ready";
      try {
        const body = await response.json();
        if (body.queue_depth > 0) text = `busy · ${body.queue_depth} in flight`;
      } catch {
        /* older server without the queue fields */
      }
      if (isStale(generation)) return;
      setStatus("ready", text);
      if (state.server === null) loadPresets();
      return;
    }
    setStatus("warming", response.status === 503 ? "warming up" : `http ${response.status}`);
  } catch {
    if (isStale(generation)) return;
    setStatus("down", "unreachable");
  }
}

/* ── the trace ──────────────────────────────────────────── */

function segmentsFrom(timing) {
  const segments = STAGES.map((stage) => ({ ...stage, ms: Math.max(0, timing[stage.key] || 0) })).filter(
    (segment) => segment.key !== "queue_ms" || segment.ms > 40,
  );
  const span = segments.reduce((acc, segment) => acc + segment.ms, 0) || 1;
  return segments.map((segment) => ({ ...segment, fraction: segment.ms / span }));
}

function paintBar(segments) {
  ui.bar.replaceChildren(
    ...segments.map((segment) => {
      const node = document.createElement("span");
      node.className = "seg";
      node.style.width = `${segment.fraction * 100}%`;
      node.style.setProperty("--seg-color", segment.color);
      node.dataset.stage = segment.name;
      return node;
    }),
  );
}

function paintLegend(segments, { withValues }) {
  ui.legend.replaceChildren(
    ...segments.map((segment) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      item.dataset.stage = segment.name;
      item.dataset.dim = String(!withValues);

      const key = document.createElement("span");
      key.className = "legend-key";
      key.style.setProperty("--seg-color", segment.color);

      const name = document.createElement("span");
      name.className = "legend-name";
      name.textContent = segment.name;

      item.append(key, name);

      if (withValues) {
        const value = document.createElement("span");
        value.className = "legend-value";
        value.textContent = formatMs(segment.ms);
        item.append(value);
      }
      return item;
    }),
  );
}

function estimateFor(size) {
  return state.estimate[size] || state.estimate["1k"];
}

function showIdleTrace() {
  ui.trace.dataset.state = "idle";
  ui.bar.dataset.mode = "idle";
  const segments = segmentsFrom(estimateFor(state.size)).map((segment) => ({
    ...segment,
    color: "var(--rule)",
  }));
  paintBar(segments);
  paintLegend(segments, { withValues: false });
  ui.traceTotal.textContent = "—";
}

let sweep = null;

function startSweep() {
  const segments = segmentsFrom(estimateFor(state.size));
  const estimatedTotal = segments.reduce((acc, segment) => acc + segment.ms, 0) || 1;
  ui.trace.dataset.state = "running";
  ui.bar.dataset.mode = "running";
  paintBar(segments);
  paintLegend(segments, { withValues: false });

  const started = performance.now();
  const nodes = Array.from(ui.bar.children);
  const legendItems = Array.from(ui.legend.children);

  const tick = () => {
    const elapsed = performance.now() - started;
    // Hold just short of the end until the response actually lands.
    const progress = Math.min(elapsed / estimatedTotal, 0.97);
    let start = 0;
    segments.forEach((segment, index) => {
      const end = start + segment.fraction;
      const node = nodes[index];
      const passed = progress >= end;
      const active = !passed && progress >= start;
      node.dataset.passed = String(passed);
      node.dataset.active = String(active);
      if (active) {
        const within = (progress - start) / (segment.fraction || 1);
        node.style.setProperty("--head", `${within * 100}%`);
      }
      legendItems[index].dataset.dim = String(!active && !passed);
      start = end;
    });
    ui.traceTotal.textContent = formatMs(elapsed);
    sweep = requestAnimationFrame(tick);
  };
  sweep = requestAnimationFrame(tick);
}

function stopSweep() {
  if (sweep !== null) cancelAnimationFrame(sweep);
  sweep = null;
}

function showTrace(item) {
  stopSweep();
  ui.trace.dataset.state = "done";
  ui.bar.dataset.mode = "done";
  paintBar(item.segments);
  paintLegend(item.segments, { withValues: true });
  ui.traceTotal.textContent = formatMs(item.timing.total_ms);
}

/* ── results ────────────────────────────────────────────── */

const extensionOf = (mimeType) => (mimeType.split("/").pop() || "img").replace("jpeg", "jpg");

function showItem(item) {
  state.currentId = item.id;
  ui.image.src = item.url;
  ui.image.alt = item.prompt;
  ui.image.hidden = false;
  ui.matEmpty.hidden = true;
  showTrace(item);

  ui.factSeed.textContent = item.seed;
  ui.factPreset.textContent = presetLabel(item.size, item.aspect);
  ui.factSize.textContent = `${item.width} × ${item.height}`;
  ui.factFormat.textContent = item.mimeType;
  ui.factId.textContent = item.id.replace(/^img_/, "").slice(0, 8);
  ui.factId.title = item.id;
  ui.download.textContent = `Download ${extensionOf(item.mimeType).toUpperCase()}`;
  ui.readout.hidden = false;

  for (const shot of ui.strip.children) {
    shot.querySelector(".shot").setAttribute("aria-current", String(shot.dataset.id === item.id));
  }
}

function addItem(item) {
  state.items.unshift(item);
  while (state.items.length > HISTORY_LIMIT) {
    URL.revokeObjectURL(state.items.pop().url);
  }
  renderStrip();
}

function renderStrip() {
  ui.session.hidden = state.items.length === 0;
  ui.sessionCount.textContent = `${state.items.length} image${state.items.length === 1 ? "" : "s"}`;

  ui.strip.replaceChildren(
    ...state.items.map((item) => {
      const li = document.createElement("li");
      li.dataset.id = item.id;

      const button = document.createElement("button");
      button.type = "button";
      button.className = "shot";
      button.setAttribute("aria-current", String(item.id === state.currentId));
      button.title = item.prompt;

      const thumb = document.createElement("img");
      thumb.src = item.url;
      thumb.alt = `${presetLabel(item.size, item.aspect)}, seed ${item.seed}: ${item.prompt}`;
      thumb.loading = "lazy";

      const meter = document.createElement("span");
      meter.className = "shot-meter";
      for (const segment of item.segments) {
        const bit = document.createElement("span");
        bit.style.width = `${segment.fraction * 100}%`;
        bit.style.setProperty("--seg-color", segment.color);
        meter.append(bit);
      }

      // The seed and preset are what a reproducible re-run needs.
      const caption = document.createElement("span");
      caption.className = "shot-caption";
      caption.textContent = `seed ${item.seed}`;

      const preset = document.createElement("span");
      preset.className = "shot-preset";
      preset.textContent = `${presetLabel(item.size, item.aspect)} · ${extensionOf(item.mimeType)}`;
      caption.append(preset);

      button.append(thumb, meter, caption);
      button.addEventListener("click", () => showItem(item));
      li.append(button);
      return li;
    }),
  );
}

ui.download.addEventListener("click", () => {
  const item = state.items.find((entry) => entry.id === state.currentId);
  if (!item) return;
  const anchor = document.createElement("a");
  anchor.href = item.url;
  anchor.download =
    `krea2-${item.size}-${item.aspect.replace(":", "x")}-${item.seed}-` +
    `${item.id.replace(/^img_/, "").slice(0, 8)}.${extensionOf(item.mimeType)}`;
  anchor.click();
});

ui.restore.addEventListener("click", () => {
  const item = state.items.find((entry) => entry.id === state.currentId);
  if (!item) return;
  ui.prompt.value = item.prompt;
  state.size = item.size;
  state.aspect = item.aspect;
  state.format = item.format;
  renderPresets();
  ui.seedModes.find((input) => input.value === "fixed").checked = true;
  ui.seed.value = item.seed;
  syncSeedInput();
  syncPrompt();
  ui.prompt.focus();
});

/* ── errors ─────────────────────────────────────────────── */

async function describeFailure(response) {
  let detail = null;
  try {
    detail = (await response.json()).detail;
  } catch {
    /* no JSON body */
  }

  if (response.status === 401) {
    return "The endpoint rejected this token. Open Endpoint and check it against KREA2_API_TOKEN.";
  }
  if (response.status === 422) {
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0];
      const field = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : "request";
      return `The server rejected ${field}: ${first.msg}`;
    }
    return typeof detail === "string"
      ? detail
      : "The server rejected the request. Check the prompt length, preset, format, and seed.";
  }
  if (response.status === 429) {
    const wait = response.headers.get("Retry-After") || "5";
    return `The server queue is full. Try again in ${wait} s.`;
  }
  if (response.status === 503) {
    return "The model runtime is still warming up. Readiness turns green when every preset is compiled or loaded.";
  }
  if (response.status === 504) {
    return "The generation did not finish within the server's timeout. The server may still be busy with it.";
  }
  if (response.status === 500) {
    return typeof detail === "string"
      ? `${detail}. Check the server log for the traceback.`
      : "Generation failed on the server. Check the server log for the traceback.";
  }
  return typeof detail === "string" ? detail : `Request failed with HTTP ${response.status}.`;
}

/* ── generate ───────────────────────────────────────────── */

ui.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;

  const prompt = ui.prompt.value.trim();
  if (prompt.length === 0 || prompt.length > MAX_PROMPT) return;

  if (!state.token) {
    say("Add the bearer token under Endpoint before generating.", "info");
    ui.settings.showModal();
    return;
  }
  if (state.server === null || currentPreset() === null) {
    say("The server's preset list has not loaded yet.", "info");
    return;
  }

  let seed;
  try {
    seed = readSeed();
  } catch (error) {
    say(error.message);
    ui.seed.focus();
    return;
  }

  const size = state.size;
  const aspect = state.aspect;
  const format = state.format;
  const body = { prompt, aspect_ratio: aspect, image_size: size, response_format: format };
  if (seed !== null) body.seed = seed;

  clearNotice();
  setBusy(true);
  startSweep();

  let response;
  try {
    response = await fetch(`${state.endpoint}/v1/images/generations`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${state.token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
  } catch {
    setBusy(false);
    showIdleTrace();
    say(
      `Could not reach ${state.endpoint}. Check the address, and make sure ` +
        `${location.origin} is listed in KREA2_ALLOWED_ORIGINS on the server.`,
    );
    checkHealth();
    return;
  }

  if (!response.ok) {
    setBusy(false);
    showIdleTrace();
    say(await describeFailure(response), response.status === 429 ? "info" : "error");
    checkHealth();
    // A rejected preset usually means the server's serving set changed under us.
    if (response.status === 422) loadPresets();
    return;
  }

  let url = null;
  try {
    const result = await response.json();
    const imageData = result?.data?.[0];
    if (
      typeof result?.id !== "string" ||
      typeof imageData?.b64_json !== "string" ||
      imageData.b64_json.length === 0 ||
      !result?.timing
    ) {
      throw new Error("The generation response has an invalid shape.");
    }

    const mimeType = imageData.mime_type || `image/${format}`;
    const bytes = Uint8Array.from(atob(imageData.b64_json), (char) => char.charCodeAt(0));
    url = URL.createObjectURL(new Blob([bytes], { type: mimeType }));

    const itemSize = result.image_size || size;
    // Queue time depends on other clients, not on this request's shape.
    state.estimate[itemSize] = {
      queue_ms: 0,
      prompt_encoding_ms: result.timing.prompt_encoding_ms,
      denoise_ms: result.timing.denoise_ms,
      vae_decode_ms: result.timing.vae_decode_ms,
      encode_ms: result.timing.encode_ms,
    };

    const item = {
      id: result.id,
      prompt,
      aspect: result.aspect_ratio || aspect,
      size: itemSize,
      format: result.response_format || format,
      seed: result.seed,
      width: result.width,
      height: result.height,
      timing: result.timing,
      segments: segmentsFrom(result.timing),
      url,
      mimeType,
    };

    addItem(item);
    showItem(item);
    checkHealth();
  } catch {
    if (url !== null) URL.revokeObjectURL(url);
    showIdleTrace();
    say("The endpoint returned an invalid image response. Check the server log and API version.");
  } finally {
    setBusy(false);
  }
});

function setBusy(busy) {
  state.busy = busy;
  ui.mat.dataset.busy = String(busy);
  ui.generateLabel.textContent = busy ? "Generating" : "Generate image";
  syncGenerate();
  if (!busy) stopSweep();
}

/* A browser without a decoder for the chosen format (AVIF on an old browser)
   must be told so rather than left looking at a broken frame. */
ui.image.addEventListener("error", () => {
  if (!ui.image.src) return;
  const item = state.items.find((entry) => entry.id === state.currentId);
  const kind = item ? extensionOf(item.mimeType).toUpperCase() : "this";
  say(`This browser could not decode the ${kind} image. Download it, or pick another format.`, "info");
});

/* ── start ──────────────────────────────────────────────── */

syncSeedInput();
syncPrompt();
renderPresets();
showIdleTrace();
checkHealth();
setInterval(checkHealth, 15000);

if (!state.token) {
  say("Set the server address and bearer token under Endpoint to begin.", "info");
}
