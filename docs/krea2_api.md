# Krea 2 Turbo HTTP API

This document specifies the HTTP API served by `maxdiffusion.serve_krea2`. The server keeps one Krea 2 Turbo
runtime resident on the accelerator, serves the named resolution presets from the AOT executable cache, runs one
generation at a time and returns one image per request as base64 JSON in AVIF, WebP, JPEG or PNG.

## Server

- Start: `krea2-api` (installed entry point) or `python -m maxdiffusion.serve_krea2`
- Default address: `0.0.0.0:8000`, one process, one uvicorn worker
- Content type: `application/json`
- Model: `krea-2-turbo`, one image per request
- Inference steps, guidance and every other model setting come from the YAML config (`KREA2_CONFIG`)
- LoRA: whatever `lora_config` the config carries is applied to every request, exactly as `generate_krea2`
  does on the command line. There is no LoRA field and no LoRA endpoint.

Put the server behind HTTPS before its port is reachable from the internet.

### Startup and readiness

At startup the server:

1. loads the runtime (`generate_krea2.create_runtime`: model, weight cache, mesh, pipeline, AOT cache);
2. builds the serving set from `KREA2_SERVE_PRESETS` x `KREA2_SERVE_TEXT_TOKENS`;
3. compiles, or loads from the AOT cache, every (resolution, text bucket) executable of that set.

`/livez` answers from the first second; `/readyz` answers `503` until step 3 has finished and `200` after. A
failure in any step is logged with its traceback and stops the process with exit code 1 (exit code 3 when the
`KREA2_*` settings themselves are invalid, for example an encoder quality outside 0–100), so a supervisor can
restart it. Stopping the server during startup exits without waiting for the warmup.

When the config enables classifier-free guidance (`do_classifier_free_guidance` and `guidance_scale > 0`; the
Turbo configs have guidance 0 and are unaffected), every request also encodes the config's `negative_prompt`.
Its text bucket must then be in `KREA2_SERVE_TEXT_TOKENS`, or startup fails with exit code 1 naming the bucket,
because an unwarmed bucket would compile in the request path. `negative_prompt` may be a string (empty or unset
means no negative prompt) or a list with exactly one string; a list of any other length fails startup with exit
code 1, with or without guidance, because the pipeline would reject it on every request (one image per request).

### Shutdown

On SIGINT/SIGTERM the server shuts down in this order, so the process is gone after at most about twice
`KREA2_SHUTDOWN_GRACE_S` (default 10 s each):

1. **Drain** (at most `KREA2_SHUTDOWN_GRACE_S`): the server stops accepting connections, closes idle ones and lets
   requests already in flight finish. Requests still open when the grace runs out are cancelled and answered
   `500` by uvicorn; a cancelled request drops its generation if that has not started yet.
2. **Lifespan shutdown**: `/readyz` turns false and every generation, tokenization or image encode still queued
   is cancelled, so no new work starts from here on.
3. **Running work** (at most another `KREA2_SHUTDOWN_GRACE_S`): a generation, tokenization or encode that is
   already running cannot be interrupted (for example one whose request got `504`, or was cancelled in step 1).
   The process waits for it and then exits immediately without it, with the exit code it would have had.

A second SIGINT during the drain skips the rest of step 1 and step 2; step 3 still bounds the exit.

With `aot_cache_dir` pointing at a cache that already holds the serving set (for example built with
`generate_krea2 ... krea2_precompile=all`), startup step 3 only loads executables. With an empty cache it compiles
them, which takes minutes per resolution.

### Serving set: presets and text buckets

A preset is an `aspect_ratio` + `image_size` pair with a fixed width and height (see
`models/krea2/resolution_presets.py`). `KREA2_SERVE_PRESETS` uses the `krea2_precompile` syntax: `all` (all 22),
a size (`1k`, `2k`), or `<size>@<ratio>` (`2k@16:9`), comma separated.

| `aspect_ratio` | 1k width x height | 2k width x height |
|---|---:|---:|
| `1:1` | 1024 x 1024 | 2048 x 2048 |
| `5:4` / `4:5` | 1152 x 896 / 896 x 1152 | 2304 x 1792 / 1792 x 2304 |
| `4:3` / `3:4` | 1152 x 864 / 864 x 1152 | 2304 x 1728 / 1728 x 2304 |
| `3:2` / `2:3` | 1248 x 832 / 832 x 1248 | 2496 x 1664 / 1664 x 2496 |
| `16:9` / `9:16` | 1344 x 768 / 768 x 1344 | 2688 x 1536 / 1536 x 2688 |
| `21:9` / `9:21` | 1536 x 672 / 672 x 1536 | 3072 x 1344 / 1344 x 3072 |

The prompt is tokenized and padded to a text bucket (a multiple of the pipeline's compaction multiple, at most
`max_sequence_length`; 128 / 256 / 384 / 512 with the v6e-1 preset). Each (resolution, text bucket) pair is its
own executable, so only buckets listed in `KREA2_SERVE_TEXT_TOKENS` (default `128,256`) are served. A prompt
whose bucket is not warmed is rejected with `422` instead of triggering a compile in the request path.
`GET /v1/presets` lists exactly what is warmed.

## Authentication

Every `/v1` route requires the static bearer token set in `KREA2_API_TOKEN`:

```http
Authorization: Bearer <KREA2_API_TOKEN>
```

Missing or invalid credentials receive `401 Unauthorized` with `WWW-Authenticate: Bearer`. `/livez` and
`/readyz` need no token.

## Generate an image

```http
POST /v1/images/generations
Content-Type: application/json
Authorization: Bearer <KREA2_API_TOKEN>
```

### Request body

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `prompt` | string | yes | — | Leading and trailing whitespace is removed; 1–4000 characters after trimming. Its token count must fit a warmed text bucket. |
| `aspect_ratio` | string | no | `"1:1"` | One of `1:1`, `5:4`, `4:3`, `3:2`, `16:9`, `21:9`, `4:5`, `3:4`, `2:3`, `9:16`, `9:21`. |
| `image_size` | string | no | `"1k"` | `"1k"` or `"2k"`. Together with `aspect_ratio` it must name a warmed preset. |
| `seed` | integer or null | no | server-drawn | Noise seed in `0`–`2147483647`. When omitted or null the server draws a fresh random seed for this request (the config's `seed` is never used) and echoes it in the response. |
| `response_format` | string or null | no | `"avif"` | `"avif"`, `"webp"`, `"jpeg"` or `"png"`; must be enabled in `KREA2_FORMATS`. When `avif` is disabled the default is the first enabled format. |

Unknown fields (for example the old `lora` field) are rejected with `422`.

```json
{
  "prompt": "a cinematic photograph of a fox walking through fresh snow",
  "aspect_ratio": "16:9",
  "image_size": "1k",
  "seed": 42,
  "response_format": "webp"
}
```

### Successful response

Status `200 OK`:

```json
{
  "id": "img_f90e1a8c47db42f7bf168068f97442b3",
  "created": 1791360000,
  "model": "krea-2-turbo",
  "aspect_ratio": "16:9",
  "image_size": "1k",
  "width": 1344,
  "height": 768,
  "seed": 42,
  "response_format": "webp",
  "data": [
    {
      "mime_type": "image/webp",
      "b64_json": "UklGRl..."
    }
  ],
  "timing": {
    "queue_ms": 3,
    "prompt_encoding_ms": 41,
    "denoise_ms": 940,
    "vae_decode_ms": 120,
    "encode_ms": 38,
    "total_ms": 1149
  }
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique response id with an `img_` prefix. |
| `created` | integer | Unix time in seconds. |
| `model` | string | Always `"krea-2-turbo"`. |
| `aspect_ratio`, `image_size` | string | The preset that was generated. |
| `width`, `height` | integer | Its pixel size. |
| `seed` | integer | The seed used: the request's, or the fresh one the server drew when the request had none. Sending it again with the same prompt and preset reproduces the image on the same runtime. |
| `response_format` | string | The format of `data[0]`. |
| `data` | array | Exactly one image. |
| `data[0].mime_type` | string | `image/avif`, `image/webp`, `image/jpeg` or `image/png`. |
| `data[0].b64_json` | string | Base64 image bytes without a data-URL prefix. |
| `timing` | object | Wall-clock milliseconds, informational. |

`timing` fields:

| Field | Meaning |
|---|---|
| `queue_ms` | From the request's arrival to the start of its generation (waiting behind other generations, plus the text-bucket check). |
| `prompt_encoding_ms` | Text encoder. |
| `denoise_ms` | Transformer denoising loop. |
| `vae_decode_ms` | VAE decode and transfer to the host. |
| `encode_ms` | Image encoding and base64, on the encode pool after the generation slot was released. |
| `total_ms` | From arrival to the finished response body. |

### Image formats

| `response_format` | Pillow settings | When to use it |
|---|---|---|
| `avif` (default) | quality `KREA2_AVIF_QUALITY` (85), speed `KREA2_AVIF_SPEED` (8), 4:4:4, `KREA2_AVIF_THREADS` (4) threads | Smallest files at high quality; encoding costs tens of ms at 1k and more at 2k. |
| `webp` | quality `KREA2_WEBP_QUALITY` (90), method 4 | Fast encode, small files, universally supported by browsers. |
| `jpeg` | quality `KREA2_JPEG_QUALITY` (92), 4:4:4 | Maximum compatibility with non-browser tools. |
| `png` | compress level 1 | Lossless, large (several MB at 2k); for archiving or further editing. |

## Presets

```http
GET /v1/presets
Authorization: Bearer <KREA2_API_TOKEN>
```

Lists the warmed serving set (`503` before the server is ready):

```json
{
  "presets": [
    {"aspect_ratio": "1:1", "image_size": "1k", "width": 1024, "height": 1024, "text_tokens": [128, 256]},
    {"aspect_ratio": "16:9", "image_size": "1k", "width": 1344, "height": 768, "text_tokens": [128, 256]}
  ],
  "formats": ["avif", "webp", "jpeg", "png"],
  "default_format": "avif",
  "max_queue": 4,
  "request_timeout_s": 60.0
}
```

`text_tokens` lists the warmed text buckets of each preset; a prompt longer than the largest one is rejected.

## Concurrency, queueing and timeouts

- Generations run strictly one at a time on one worker thread (JAX owns the accelerator in this one process).
- Tokenization for the bucket check and image encoding run on a separate pool of `KREA2_ENCODE_THREADS`
  threads, so the next generation starts while the previous image is being encoded.
- Admission: at most `KREA2_MAX_QUEUE` requests may be in flight (tokenizing, waiting for or running their
  generation). Admission is decided on arrival, before any tokenization or generation work; beyond the limit the
  server answers `429 Too Many Requests` with `Retry-After: 5` at once and does not queue the request. A request
  gives its slot back when its generation ends, before its image is encoded.
- A request waits at most `KREA2_REQUEST_TIMEOUT_S` seconds from its arrival (tokenization + queue +
  generation + image encoding) and then receives `504 Gateway Timeout`. If its generation had not started yet it
  is dropped and its slot frees at once; if it was already running it still finishes in the background and keeps
  its slot until then (the image is discarded). A tokenization or encode that outlives its request likewise
  finishes in the background.
- `/livez` and `/readyz` answer while a generation runs.

## Health endpoints

No authentication.

`GET /livez` → `200 {"status": "ok"}` as soon as the process serves HTTP.

`GET /readyz` → once the serving set is warmed:

```json
{"status": "ready", "queue_depth": 1, "busy": true}
```

`queue_depth` counts requests in flight (tokenizing, waiting or running; the number compared with
`KREA2_MAX_QUEUE`), `busy` is true while a generation runs. Before that it
answers `503 {"detail": "Model is not ready"}` (`"Model startup failed"` in the moment before a failed startup
stops the process).

## Errors

FastAPI validation errors use its standard `detail` array; the others carry a string `detail`.

| Status | Meaning | Typical cause |
|---:|---|---|
| `401` | Unauthorized | Missing or invalid bearer token. |
| `422` | Unprocessable Entity | Invalid or unknown field, prompt blank or over 4000 characters, seed out of range, unknown ratio/size, a preset that is not warmed (the detail lists the warmed ones), a prompt with too many tokens for the warmed text buckets, a `response_format` that is not enabled. |
| `429` | Too Many Requests | `KREA2_MAX_QUEUE` generations already in flight. Honour `Retry-After`. |
| `500` | Internal Server Error | `{"detail": "Image generation failed"}`: the runtime or the encoder raised; the server log has the exception type and traceback. |
| `503` | Service Unavailable | The runtime is still loading or warming up. |
| `504` | Gateway Timeout | Tokenization, queueing, generation and image encoding did not finish within `KREA2_REQUEST_TIMEOUT_S` of the request's arrival. |

Every response carries `X-Request-ID`: the client's value when it sends a safe one (1–128 characters from
`A-Z a-z 0-9 . _ : -`), otherwise a fresh id. The server logs one line per generation request with this id, the
preset, format, seed, text bucket, timings and status.

## CORS

Set a comma-separated allowlist of browser origins (scheme, host and port; no path, no trailing slash):

```bash
export KREA2_ALLOWED_ORIGINS='https://frontend.example,http://localhost:5173'
```

Allowed methods `GET`, `POST`, `OPTIONS`; request headers `Authorization`, `Content-Type`, `X-Request-ID`;
exposed response headers `X-Request-ID`, `Retry-After`.

## Examples

### cURL

```bash
curl -sS http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer ${KREA2_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a fox walking through fresh snow", "aspect_ratio": "16:9", "response_format": "webp"}' \
  | python3 -c 'import base64, json, sys; r = json.load(sys.stdin); open("fox.webp", "wb").write(base64.b64decode(r["data"][0]["b64_json"])); print(r["seed"], r["timing"])'
```

### Browser

```javascript
const response = await fetch("https://api.example.com/v1/images/generations", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    prompt: "a cinematic photograph of a fox walking through fresh snow",
    aspect_ratio: "16:9",
    image_size: "1k",
    response_format: "avif",
  }),
});

if (response.status === 429) {
  const wait = Number(response.headers.get("Retry-After") || 5);
  // try again after `wait` seconds
}
if (!response.ok) {
  throw new Error(`Krea 2 request failed: ${response.status}`);
}

const result = await response.json();
const { mime_type, b64_json } = result.data[0];
const bytes = Uint8Array.from(atob(b64_json), (char) => char.charCodeAt(0));
const imageUrl = URL.createObjectURL(new Blob([bytes], { type: mime_type }));
document.querySelector("#result").src = imageUrl;
```

Call `URL.revokeObjectURL(imageUrl)` once the image is no longer shown. The static console in `frontend/` is a
complete client.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `KREA2_API_TOKEN` | — (required) | Bearer token for `/v1` routes. |
| `KREA2_HOST` | `0.0.0.0` | Listen address. |
| `KREA2_PORT` | `8000` | Listen port. |
| `KREA2_ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:5173` | CORS origin allowlist. |
| `KREA2_DISABLE_DOCS` | off | `1`, `true` or `yes` disables the `/docs` page. |
| `KREA2_CONFIG` | bundled `configs/base_krea2_turbo.yml` | Krea 2 YAML config (`configs/base_krea2_turbo_v6e1.yml` on a v6e-1). |
| `KREA2_CONFIG_OVERRIDES` | empty | Shell-quoted `key=value` config overrides, passed through verbatim, e.g. `aot_cache_dir=/cache/aot aot_cache_lazy_load=True aot_cache_gcs=gs://bucket/aot krea2_weight_cache_dir=/cache/weights`. Setting `krea2_precompile`, `krea2_weight_cache_build_only` or a `batch_size` other than 1 refuses to start. |
| `KREA2_MODEL_PATH` | config's model | Sets `pretrained_model_name_or_path` (a local directory avoids a Hub lookup). |
| `KREA2_OUTPUT_DIR` | `/tmp/krea2-api` | Run output directory; generated API images are not written there. |
| `KREA2_SERVE_PRESETS` | `all` | Presets to warm and serve (`krea2_precompile` syntax). |
| `KREA2_SERVE_TEXT_TOKENS` | `128,256` | Text buckets to warm per preset. |
| `KREA2_MAX_QUEUE` | `4` | Generations allowed in flight (waiting + running) before `429`. |
| `KREA2_REQUEST_TIMEOUT_S` | `60` | Per-request wait limit before `504`, counted from arrival. |
| `KREA2_SHUTDOWN_GRACE_S` | `10` | At shutdown, the bound on draining in-flight requests and then, again, on waiting for a running generation, tokenization or encode before exiting without it (> 0; see Shutdown). |
| `KREA2_ENCODE_THREADS` | `2` | Threads for tokenization and image encoding. |
| `KREA2_FORMATS` | `avif,webp,jpeg,png` | Enabled response formats. The AVIF encoder is checked at startup only when `avif` is enabled. |
| `KREA2_AVIF_QUALITY` | `85` | AVIF quality 0–100. Out-of-range encoder values refuse to start. |
| `KREA2_AVIF_SPEED` | `8` | AVIF speed 0–10 (higher is faster, slightly larger). |
| `KREA2_AVIF_THREADS` | `4` | AVIF encoder threads. |
| `KREA2_WEBP_QUALITY` | `90` | WebP quality 0–100. |
| `KREA2_JPEG_QUALITY` | `92` | JPEG quality 0–100. |

The server always appends `run_name=krea2_api`, `output_dir=$KREA2_OUTPUT_DIR`, `batch_size=1` and
`prompt=warmup` after the overrides, so those keys cannot be overridden.

A v6e-1 launch with the caches on local disk:

```bash
export KREA2_API_TOKEN='replace-with-a-long-random-token'
export KREA2_CONFIG=src/maxdiffusion/configs/base_krea2_turbo_v6e1.yml
export KREA2_MODEL_PATH=/mnt/krea2/Krea-2-Turbo
export KREA2_CONFIG_OVERRIDES='aot_cache_dir=/mnt/krea2/aot aot_cache_lazy_load=True krea2_weight_cache_dir=/mnt/krea2/weights'
export KREA2_SERVE_PRESETS=all KREA2_SERVE_TEXT_TOKENS=128,256
python -m maxdiffusion.serve_krea2
```

## Docker

`maxdiffusion_krea2_server.Dockerfile` builds on the dependency image, installs the package, fails the build
when Pillow lacks AVIF, and runs `krea2-api` on port 8000. Pass the variables above with `-e` and mount the
model and caches.
