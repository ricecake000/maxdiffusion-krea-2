"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""HTTP API for Krea 2 Turbo (contract: docs/krea2_api.md).

One process keeps one Krea2Runtime (generate_krea2.create_runtime) resident on the accelerator. At startup the
runtime loads the model and compiles or loads every (resolution, text bucket) executable of the serving set into
the AOT cache; readiness stays false until that finishes. Requests then run one generation at a time on a single
worker thread; tokenization (the text-bucket check) and image encoding run on a separate small pool, so the next
generation starts while the previous image is being encoded. A bounded admission counter answers 429 instead of
queueing without limit.

Run it with `krea2-api` or `python -m maxdiffusion.serve_krea2`; configuration comes from KREA2_* environment
variables (see ServerSettings.from_env and _runtime_argv).
"""

import time

# Wall clock when the server process started importing: handed to the runtime's startup timeline
# (KREA2_PROCESS_T0) so its marks count from server start, not from the lazy generate_krea2 import.
_SERVER_T0 = time.time()

# pylint: disable=wrong-import-position
import asyncio
import base64
import concurrent.futures
from contextlib import asynccontextmanager
import dataclasses
import functools
import io
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import sys
import traceback
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.datastructures import MutableHeaders

from maxdiffusion import max_logging
from maxdiffusion.models.krea2.resolution_presets import (
    KREA2_ASPECT_RATIOS,
    KREA2_DEFAULT_ASPECT_RATIO,
    KREA2_DEFAULT_IMAGE_SIZE,
    KREA2_IMAGE_SIZES,
)

MODEL_NAME = "krea-2-turbo"
MAX_SEED = 2**31 - 1
MAX_PROMPT_CHARS = 4000
RETRY_AFTER_S = 5
# Literal status code: starlette renamed the 422 constant (HTTP_422_UNPROCESSABLE_CONTENT) and deprecated the old one.
_HTTP_422 = 422

# response_format -> (Pillow format name, MIME type). Order is the documented order.
IMAGE_FORMATS: Dict[str, Tuple[str, str]] = {
    "avif": ("AVIF", "image/avif"),
    "webp": ("WEBP", "image/webp"),
    "jpeg": ("JPEG", "image/jpeg"),
    "png": ("PNG", "image/png"),
}
ResponseFormat = Literal["avif", "webp", "jpeg", "png"]

_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_REQUEST_ID_SCOPE_KEY = "krea2.request_id"


# ---------------------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------------------


def _env(environ: Mapping[str, str], name: str, default: str) -> str:
  value = environ.get(name, "").strip()
  return value if value else default


def _env_int(environ: Mapping[str, str], name: str, default: int, minimum: int, maximum: Optional[int] = None) -> int:
  raw = _env(environ, name, str(default))
  try:
    value = int(raw)
  except ValueError as e:
    raise RuntimeError(f"{name} must be an integer, got {raw!r}.") from e
  if value < minimum or (maximum is not None and value > maximum):
    bounds = f">= {minimum}" if maximum is None else f"in {minimum}..{maximum}"
    raise RuntimeError(f"{name} must be {bounds}, got {value}.")
  return value


def _env_float(environ: Mapping[str, str], name: str, default: float, minimum: float) -> float:
  raw = _env(environ, name, str(default))
  try:
    value = float(raw)
  except ValueError as e:
    raise RuntimeError(f"{name} must be a number, got {raw!r}.") from e
  if not math.isfinite(value) or not value > minimum:
    raise RuntimeError(f"{name} must be a finite number > {minimum}, got {value}.")
  return value


def _parse_formats(raw: str) -> Tuple[str, ...]:
  formats = []
  for entry in raw.split(","):
    name = entry.strip().lower()
    if not name:
      continue
    if name not in IMAGE_FORMATS:
      raise RuntimeError(f"KREA2_FORMATS entry {name!r} is unknown; valid values: {', '.join(IMAGE_FORMATS)}.")
    if name not in formats:
      formats.append(name)
  if not formats:
    raise RuntimeError("KREA2_FORMATS must name at least one format.")
  return tuple(formats)


def _parse_text_tokens(raw: str) -> Tuple[int, ...]:
  tokens = []
  for entry in raw.split(","):
    entry = entry.strip()
    if not entry:
      continue
    try:
      tokens.append(int(entry))
    except ValueError as e:
      raise RuntimeError(f"KREA2_SERVE_TEXT_TOKENS entries must be integers, got {entry!r}.") from e
  return tuple(tokens)


def _allowed_origins(environ: Mapping[str, str] = os.environ) -> List[str]:
  raw = environ.get("KREA2_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
  return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


@dataclasses.dataclass(frozen=True)
class ServerSettings:
  """Server settings read once from KREA2_* environment variables at startup."""

  api_token: str
  formats: Tuple[str, ...]
  default_format: str
  max_queue: int
  request_timeout_s: float
  encode_threads: int
  shutdown_grace_s: float
  presets_spec: str
  text_tokens: Tuple[int, ...]
  encode_options: Dict[str, Dict[str, Any]]

  @classmethod
  def from_env(cls, environ: Mapping[str, str] = os.environ) -> "ServerSettings":
    """Reads and validates the settings; raises RuntimeError on a missing token or a bad value."""
    token = environ.get("KREA2_API_TOKEN", "")
    if not token:
      raise RuntimeError("KREA2_API_TOKEN must be set before starting the server.")
    formats = _parse_formats(_env(environ, "KREA2_FORMATS", ",".join(IMAGE_FORMATS)))
    encode_options = {
        "avif": {
            "quality": _env_int(environ, "KREA2_AVIF_QUALITY", 85, 0, 100),
            "speed": _env_int(environ, "KREA2_AVIF_SPEED", 8, 0, 10),
            "subsampling": "4:4:4",
            "max_threads": _env_int(environ, "KREA2_AVIF_THREADS", 4, 1),
        },
        "webp": {"quality": _env_int(environ, "KREA2_WEBP_QUALITY", 90, 0, 100), "method": 4},
        "jpeg": {"quality": _env_int(environ, "KREA2_JPEG_QUALITY", 92, 0, 100), "subsampling": 0},
        "png": {"compress_level": 1},
    }
    return cls(
        api_token=token,
        formats=formats,
        default_format="avif" if "avif" in formats else formats[0],
        max_queue=_env_int(environ, "KREA2_MAX_QUEUE", 4, 1),
        request_timeout_s=_env_float(environ, "KREA2_REQUEST_TIMEOUT_S", 60.0, 0.0),
        encode_threads=_env_int(environ, "KREA2_ENCODE_THREADS", 2, 1),
        shutdown_grace_s=_env_float(environ, "KREA2_SHUTDOWN_GRACE_S", 10.0, 0.0),
        presets_spec=_env(environ, "KREA2_SERVE_PRESETS", "all"),
        text_tokens=_parse_text_tokens(_env(environ, "KREA2_SERVE_TEXT_TOKENS", "128,256")),
        encode_options=encode_options,
    )


def _is_false(value: str) -> bool:
  return value.strip().strip("'\"").lower() in ("", "false", "0", "no", "none")


def _runtime_argv(environ: Mapping[str, str] = os.environ) -> List[str]:
  """Builds the generate_krea2 argv from the environment.

  Layout: program name, KREA2_CONFIG, the KREA2_CONFIG_OVERRIDES entries verbatim, the optional
  pretrained_model_name_or_path from KREA2_MODEL_PATH, then the server-owned keys (run_name, output_dir,
  batch_size=1, prompt), which win over any earlier value. Raises RuntimeError for an override the server cannot
  serve with: krea2_precompile (precompile mode exits), krea2_weight_cache_build_only (no device placement) or a
  batch_size other than 1.
  """
  config_path = _env(environ, "KREA2_CONFIG", str(Path(__file__).with_name("configs") / "base_krea2_turbo.yml"))
  output_dir = _env(environ, "KREA2_OUTPUT_DIR", "/tmp/krea2-api")
  try:
    overrides = shlex.split(environ.get("KREA2_CONFIG_OVERRIDES", ""))
  except ValueError as e:
    raise RuntimeError(f"KREA2_CONFIG_OVERRIDES is not a valid shell word list: {e}") from e
  for override in overrides:
    key, sep, value = override.partition("=")
    key = key.strip()
    if not sep or not key:
      raise RuntimeError(f"KREA2_CONFIG_OVERRIDES entries must be key=value, got {override!r}.")
    if key == "krea2_precompile" and value.strip().strip("'\""):
      raise RuntimeError(
          "KREA2_CONFIG_OVERRIDES must not set krea2_precompile: the server compiles its serving set from "
          "KREA2_SERVE_PRESETS / KREA2_SERVE_TEXT_TOKENS (run generate_krea2 for a precompile-only job)."
      )
    if key == "krea2_weight_cache_build_only" and not _is_false(value):
      raise RuntimeError(
          "KREA2_CONFIG_OVERRIDES must not set krea2_weight_cache_build_only (it never places weights)."
      )
    if key == "batch_size" and value.strip().strip("'\"") != "1":
      raise RuntimeError(f"The server generates one image per request; batch_size must be 1, got {value!r}.")
  argv = ["maxdiffusion.serve_krea2", config_path, *overrides]
  model_path = environ.get("KREA2_MODEL_PATH", "").strip()
  if model_path:
    argv.append(f"pretrained_model_name_or_path={model_path}")
  argv += ["run_name=krea2_api", f"output_dir={output_dir}", "batch_size=1", "prompt=warmup"]
  return argv


def load_serving_runtime(argv: Sequence[str]):
  """Creates the long-lived Krea2Runtime (lazy import: jax/model modules load only here)."""
  os.environ.setdefault("KREA2_PROCESS_T0", repr(_SERVER_T0))
  from maxdiffusion.generate_krea2 import create_runtime  # pylint: disable=import-outside-toplevel

  return create_runtime(list(argv))


# ---------------------------------------------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------------------------------------------


def ensure_avif_encoder() -> None:
  """Raises RuntimeError when this Pillow build cannot write AVIF."""
  Image.init()
  if Image.registered_extensions().get(".avif") != "AVIF" or "AVIF" not in Image.SAVE:
    raise RuntimeError(
        "Pillow AVIF encoder is unavailable. Install an AVIF-enabled Pillow wheel (Pillow >= 11.3 bundles libavif), "
        "or drop avif from KREA2_FORMATS."
    )


def encode_image(image: Image.Image, response_format: str, options: Mapping[str, Any]) -> bytes:
  """Encodes one PIL image as `response_format` with the given Pillow save options."""
  pil_format, _ = IMAGE_FORMATS[response_format]
  if response_format != "png" and image.mode != "RGB":
    image = image.convert("RGB")
  buffer = io.BytesIO()
  image.save(buffer, format=pil_format, **options)
  return buffer.getvalue()


def encode_image_base64(image: Image.Image, response_format: str, options: Mapping[str, Any]) -> str:
  """encode_image followed by base64 (both run on the encode pool, never the generation thread)."""
  return base64.b64encode(encode_image(image, response_format, options)).decode("ascii")


# ---------------------------------------------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------------------------------------------


class GenerationRequest(BaseModel):
  """Body of POST /v1/images/generations."""

  model_config = ConfigDict(extra="forbid")

  prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
  aspect_ratio: Literal[KREA2_ASPECT_RATIOS] = KREA2_DEFAULT_ASPECT_RATIO  # type: ignore[valid-type]
  image_size: Literal[KREA2_IMAGE_SIZES] = KREA2_DEFAULT_IMAGE_SIZE  # type: ignore[valid-type]
  seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
  response_format: Optional[ResponseFormat] = None

  @field_validator("prompt", mode="before")
  @classmethod
  def strip_prompt(cls, value):
    # Strip before the length constraints run, so a blank prompt fails min_length.
    return value.strip() if isinstance(value, str) else value


class ImageData(BaseModel):
  mime_type: Literal["image/avif", "image/webp", "image/jpeg", "image/png"]
  b64_json: str


class TimingData(BaseModel):
  queue_ms: int
  prompt_encoding_ms: int
  denoise_ms: int
  vae_decode_ms: int
  encode_ms: int
  total_ms: int


class GenerationResponse(BaseModel):
  id: str
  created: int
  model: Literal["krea-2-turbo"] = MODEL_NAME
  aspect_ratio: str
  image_size: str
  width: int
  height: int
  seed: int
  response_format: ResponseFormat
  data: List[ImageData]
  timing: TimingData


class PresetInfo(BaseModel):
  aspect_ratio: str
  image_size: str
  width: int
  height: int
  text_tokens: List[int]


class PresetsResponse(BaseModel):
  presets: List[PresetInfo]
  formats: List[str]
  default_format: str
  max_queue: int
  request_timeout_s: float


# ---------------------------------------------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class WarmedPreset:
  """One warmed (aspect_ratio, image_size) preset and its warmed text buckets."""

  aspect_ratio: str
  image_size: str
  width: int
  height: int
  buckets: Set[int]

  @property
  def label(self) -> str:
    return f"{self.image_size}@{self.aspect_ratio}"


@dataclasses.dataclass
class ServerState:
  """Mutable per-app state. Counters are touched only on the event loop thread, except `running`."""

  settings: ServerSettings
  gen_pool: concurrent.futures.ThreadPoolExecutor
  encode_pool: concurrent.futures.ThreadPoolExecutor
  runtime: Any = None
  warmed: Dict[Tuple[str, str], WarmedPreset] = dataclasses.field(default_factory=dict)
  ready: bool = False
  startup_finished: bool = False  # loading + warmup ended, successfully or not
  startup_error: Optional[BaseException] = None
  startup_task: Optional[asyncio.Task] = None
  in_flight: int = 0  # admitted requests: tokenizing, waiting for or running their generation
  running: bool = False  # a runtime.generate call is executing (written by the generation thread)
  # Request futures submitted to gen_pool or encode_pool (generation, tokenization, image encoding) and not yet
  # reported done to the event loop. main() waits (bounded) for any still running at shutdown: a 504 leaves its
  # work running, the threads cannot be interrupted, and the interpreter's exit hook would join them.
  pool_futures: Set[concurrent.futures.Future] = dataclasses.field(default_factory=set)

  def release_slot(self) -> None:
    self.in_flight -= 1


def build_warmed_presets(plan) -> Dict[Tuple[str, str], WarmedPreset]:
  """Groups a serving plan [(Krea2Resolution, text_bucket), ...] by (aspect_ratio, image_size), in plan order."""
  warmed: Dict[Tuple[str, str], WarmedPreset] = {}
  for resolution, bucket in plan:
    key = (resolution.aspect_ratio, resolution.image_size)
    entry = warmed.get(key)
    if entry is None:
      entry = warmed[key] = WarmedPreset(
          aspect_ratio=resolution.aspect_ratio,
          image_size=resolution.image_size,
          width=int(resolution.width),
          height=int(resolution.height),
          buckets=set(),
      )
    entry.buckets.add(int(bucket))
  return warmed


def _check_runtime_config(config) -> None:
  batch_size = getattr(config, "batch_size", 1)
  if batch_size != 1:
    raise RuntimeError(f"The serving runtime must have batch_size 1, got {batch_size}.")
  if str(getattr(config, "krea2_precompile", "") or "").strip():
    raise RuntimeError("The serving config must not set krea2_precompile (set KREA2_SERVE_PRESETS instead).")


def _serving_negative_prompts(negative_prompt) -> List[str]:
  """The config's negative_prompt as the pipeline normalizes it for batch_size 1: None -> [""], a string s -> [s],
  a sequence -> list(seq), which must hold exactly one entry (the pipeline would fail every request otherwise)."""
  if negative_prompt is None:
    negative_prompt = ""
  if isinstance(negative_prompt, str):
    return [negative_prompt]
  negative_prompts = list(negative_prompt)
  if len(negative_prompts) != 1:
    raise RuntimeError(
        f"The config's negative_prompt is a list of {len(negative_prompts)} entries; the server generates one image "
        "per request (batch_size 1), so it must be a string or a list with exactly one entry."
    )
  return negative_prompts


def _check_negative_prompt_bucket(runtime, plan) -> None:
  """With classifier-free guidance every request also runs the config's negative prompt through the text encoder;
  its text bucket must be warmed too, or the first request would compile in the request path. The negative prompt
  is passed to every pipeline call, so its shape is validated whether CFG is on or not."""
  from maxdiffusion.pipelines.krea2.krea2_pipeline import (  # pylint: disable=import-outside-toplevel
      is_classifier_free_guidance_enabled,
  )

  config = runtime.config
  negative_prompts = _serving_negative_prompts(getattr(config, "negative_prompt", ""))
  if not is_classifier_free_guidance_enabled(
      float(getattr(config, "guidance_scale", 0.0)), getattr(config, "do_classifier_free_guidance", None)
  ):
    return
  neg_bucket = int(runtime.text_bucket(negative_prompts))
  # Every warmed preset serves it, so every preset needs the bucket (the real plan is presets x buckets).
  missing = [p.label for p in build_warmed_presets(plan).values() if neg_bucket not in p.buckets]
  if missing:
    warmed = sorted({int(bucket) for _, bucket in plan})
    raise RuntimeError(
        f"Classifier-free guidance is on and the configured negative_prompt needs the {neg_bucket}-token text "
        f"bucket, which is not in KREA2_SERVE_TEXT_TOKENS for {', '.join(missing)} (warmed buckets: "
        f"{', '.join(str(b) for b in warmed)}); an unwarmed bucket would compile at request time. Add "
        f"{neg_bucket} to KREA2_SERVE_TEXT_TOKENS or shorten the negative prompt."
    )


def prepare_runtime(loader: Callable[[], Any], settings: ServerSettings):
  """Loads the runtime and warms the serving set (blocking; runs on the generation thread).

  Returns (runtime, warmed presets).
  """
  runtime = loader()
  _check_runtime_config(runtime.config)
  plan = runtime.serving_plan(settings.presets_spec, list(settings.text_tokens))
  if not plan:
    raise RuntimeError(f"KREA2_SERVE_PRESETS={settings.presets_spec!r} selects no preset.")
  _check_negative_prompt_bucket(runtime, plan)
  max_logging.log(f"Krea 2 API: warming {len(plan)} (resolution, text bucket) executables")
  runtime.precompile(plan)
  return runtime, build_warmed_presets(plan)


async def _startup(app: FastAPI, server: ServerState, loader: Callable[[], Any], on_failure) -> None:
  loop = asyncio.get_running_loop()
  started = time.perf_counter()
  max_logging.log("Krea 2 API: loading the long-lived runtime ...")
  try:
    runtime, warmed = await loop.run_in_executor(server.gen_pool, prepare_runtime, loader, server.settings)
  except asyncio.CancelledError:
    raise
  except Exception as exc:  # pylint: disable=broad-exception-caught
    server.startup_error = exc
    server.startup_finished = True
    max_logging.log(f"Krea 2 API startup failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    on_failure(app, exc)
    return
  server.runtime = runtime
  server.warmed = warmed
  server.ready = True
  server.startup_finished = True
  labels = ", ".join(f"{p.label}:{'/'.join(str(b) for b in sorted(p.buckets))}" for p in warmed.values())
  max_logging.log(f"Krea 2 API is ready after {time.perf_counter() - started:.1f} s; warmed presets: {labels}")


def _exit_on_startup_failure(app: FastAPI, exc: BaseException) -> None:
  """Default startup-failure policy: stop uvicorn with exit code 1 (main()), or hard-exit without it."""
  del exc
  server = getattr(app.state, "uvicorn_server", None)
  if server is not None:
    app.state.exit_code = 1
    server.should_exit = True
    return
  # Served by an external uvicorn without main(): never leave a process that can only answer 503.
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(1)  # pylint: disable=protected-access


# ---------------------------------------------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------------------------------------------


class RequestIdMiddleware:
  """Echoes a safe client X-Request-ID (or a fresh one) on every response and exposes it to handlers."""

  def __init__(self, app):
    self.app = app

  async def __call__(self, scope, receive, send):
    if scope["type"] != "http":
      await self.app(scope, receive, send)
      return
    request_id = ""
    for name, value in scope.get("headers", ()):
      if name == b"x-request-id":
        request_id = value.decode("latin-1")
        break
    if not _REQUEST_ID_RE.fullmatch(request_id):
      request_id = uuid.uuid4().hex
    scope[_REQUEST_ID_SCOPE_KEY] = request_id

    async def send_with_id(message):
      if message["type"] == "http.response.start":
        MutableHeaders(scope=message)["X-Request-ID"] = request_id
      await send(message)

    await self.app(scope, receive, send_with_id)


bearer = HTTPBearer(auto_error=False)


async def require_api_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> None:
  expected = getattr(request.app.state, "api_token", None)
  if (
      not expected
      or credentials is None
      or credentials.scheme.lower() != "bearer"
      or not secrets.compare_digest(credentials.credentials.encode("utf-8"), expected.encode("utf-8"))
  ):
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _server(request: Request) -> Optional[ServerState]:
  return getattr(request.app.state, "server", None)


def _ready_server(request: Request) -> ServerState:
  server = _server(request)
  if server is None or not server.ready:
    detail = "Model startup failed" if server is not None and server.startup_error else "Model is not ready"
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
  return server


def _ms(seconds) -> int:
  try:
    return max(0, round(float(seconds) * 1000))
  except (TypeError, ValueError):
    return 0


def _track_pool_future(
    server: ServerState,
    loop: asyncio.AbstractEventLoop,
    future: concurrent.futures.Future,
    on_done: Optional[Callable[[], None]] = None,
) -> concurrent.futures.Future:
  """Keeps a pool future in server.pool_futures until it is done; then, on the event loop, removes it and calls
  on_done. Returns the future."""
  server.pool_futures.add(future)

  def _done_on_loop(done_future):
    server.pool_futures.discard(done_future)
    if on_done is not None:
      on_done()

  def _on_done(done_future):
    try:
      loop.call_soon_threadsafe(_done_on_loop, done_future)
    except RuntimeError:
      pass  # the event loop already closed (server shutdown): nobody is left to count it

  future.add_done_callback(_on_done)
  return future


def _run_generation(server: ServerState, prompt: str, preset: WarmedPreset, seed: int):
  """Generation-thread body: one runtime.generate call. Returns (start time, images, trace)."""
  started = time.perf_counter()
  server.running = True
  try:
    images, trace = server.runtime.generate(
        [prompt], height=preset.height, width=preset.width, seed=seed, output_type="pil"
    )
  finally:
    server.running = False
  return started, images, trace


def create_app(
    runtime_loader: Optional[Callable[[], Any]] = None,
    *,
    on_startup_failure: Optional[Callable[[FastAPI, BaseException], None]] = None,
) -> FastAPI:
  """Builds the FastAPI app.

  Args:
    runtime_loader: zero-argument callable returning a runtime (section "runtime interface" of the docs); None
      builds the argv from the environment (_runtime_argv, validated when the app starts) and calls
      load_serving_runtime.
    on_startup_failure: called as f(app, exc) when loading or warming the runtime fails; the default stops the
      server with exit code 1.
  """
  failure_policy = on_startup_failure or _exit_on_startup_failure

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    settings = ServerSettings.from_env()
    if "avif" in settings.formats:
      ensure_avif_encoder()
    loader = runtime_loader
    if loader is None:
      loader = functools.partial(load_serving_runtime, _runtime_argv())
    server = ServerState(
        settings=settings,
        gen_pool=concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="krea2-generate"),
        encode_pool=concurrent.futures.ThreadPoolExecutor(
            max_workers=settings.encode_threads, thread_name_prefix="krea2-encode"
        ),
    )
    app.state.api_token = settings.api_token
    app.state.server = server
    # Load and warm in the background: /livez answers and /readyz says 503 while this runs.
    server.startup_task = asyncio.create_task(_startup(app, server, loader, failure_policy))
    try:
      yield
    finally:
      # Runs after uvicorn's drain (bounded by timeout_graceful_shutdown, see main()). Cancelling the queued pool
      # work here means no generation, tokenization or encode starts once shutdown has begun; work that is already
      # running cannot be interrupted and is left to main()'s bounded wait (_exit_after_pool_work).
      server.ready = False
      server.startup_task.cancel()
      server.gen_pool.shutdown(wait=False, cancel_futures=True)
      server.encode_pool.shutdown(wait=False, cancel_futures=True)

  app = FastAPI(
      title="Krea 2 Turbo API",
      version="2.0.0",
      lifespan=lifespan,
      docs_url=None if os.environ.get("KREA2_DISABLE_DOCS", "").lower() in ("1", "true", "yes") else "/docs",
  )
  app.add_middleware(
      CORSMiddleware,
      allow_origins=_allowed_origins(),
      allow_credentials=True,
      allow_methods=["GET", "POST", "OPTIONS"],
      allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
      expose_headers=["X-Request-ID", "Retry-After"],
      max_age=3600,
  )
  app.add_middleware(RequestIdMiddleware)

  @app.get("/livez", include_in_schema=False)
  async def livez():
    return {"status": "ok"}

  @app.get("/readyz", include_in_schema=False)
  async def readyz(request: Request):
    server = _ready_server(request)
    return {"status": "ready", "queue_depth": server.in_flight, "busy": server.running}

  @app.get("/v1/presets", response_model=PresetsResponse, dependencies=[Depends(require_api_token)])
  async def list_presets(request: Request):
    server = _ready_server(request)
    settings = server.settings
    return PresetsResponse(
        presets=[
            PresetInfo(
                aspect_ratio=p.aspect_ratio,
                image_size=p.image_size,
                width=p.width,
                height=p.height,
                text_tokens=sorted(p.buckets),
            )
            for p in server.warmed.values()
        ],
        formats=list(settings.formats),
        default_format=settings.default_format,
        max_queue=settings.max_queue,
        request_timeout_s=settings.request_timeout_s,
    )

  @app.post("/v1/images/generations", response_model=GenerationResponse, dependencies=[Depends(require_api_token)])
  async def generate_image(payload: GenerationRequest, request: Request):
    arrived = time.perf_counter()
    log = {
        "request_id": request.scope.get(_REQUEST_ID_SCOPE_KEY, "-"),
        "preset": f"{payload.image_size}@{payload.aspect_ratio}",
        "format": payload.response_format or "-",
        "seed": payload.seed,
    }
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    try:
      response = await _generate(request, payload, arrived, log)
      status_code = status.HTTP_200_OK
      return response
    except HTTPException as e:
      status_code = e.status_code
      raise
    finally:
      log["total_ms"] = _ms(time.perf_counter() - arrived)
      fields = " ".join(f"{key}={value}" for key, value in log.items())
      max_logging.log(f"[krea2-api] status={status_code} {fields}")

  async def _generate(request: Request, payload: GenerationRequest, arrived: float, log: Dict[str, Any]):
    server = _ready_server(request)
    settings = server.settings
    loop = asyncio.get_running_loop()

    response_format = payload.response_format or settings.default_format
    log["format"] = response_format
    if response_format not in settings.formats:
      raise HTTPException(
          status_code=_HTTP_422,
          detail=f"response_format {response_format!r} is not enabled on this server; "
          f"enabled: {', '.join(settings.formats)}.",
      )
    preset = server.warmed.get((payload.aspect_ratio, payload.image_size))
    if preset is None:
      raise HTTPException(
          status_code=_HTTP_422,
          detail=f"Preset {payload.image_size}@{payload.aspect_ratio} is not warmed on this server; "
          f"warmed presets: {', '.join(p.label for p in server.warmed.values())}.",
      )

    # An omitted seed means a fresh one per request (the config's seed is never used by the API); it is echoed so
    # the image can be reproduced. Drawn before admission, so a failure here cannot leak a slot.
    seed = payload.seed if payload.seed is not None else secrets.randbelow(MAX_SEED + 1)
    log["seed"] = seed

    # Admission before any pool work: no await between the check and the increment, so this is atomic on the event
    # loop, and a saturated encode pool cannot make the 429 late or queue unbounded tokenization work.
    if server.in_flight >= settings.max_queue:
      raise HTTPException(
          status_code=status.HTTP_429_TOO_MANY_REQUESTS,
          detail=f"The generation queue is full ({settings.max_queue} in flight); retry shortly.",
          headers={"Retry-After": str(RETRY_AFTER_S)},
      )
    server.in_flight += 1
    deadline = arrived + settings.request_timeout_s
    timeout_detail = f"Image generation did not finish within {settings.request_timeout_s:g} s."

    def remaining_s() -> float:
      return max(0.0, deadline - time.perf_counter())

    # Until the generation future exists this request owns the slot and must release it on every exit path
    # (tokenization failure or timeout, 422 bucket rejection, cancellation); after that the future's done callback
    # releases it, when the generation really ends or is dropped from the queue.
    future = None
    try:
      # The text bucket decides which executable runs; an unwarmed bucket would compile for minutes. Tokenizing is
      # CPU-only but keep it off the event loop, and bound the wait by the request deadline. On timeout wait_for
      # cancels a still-queued tokenization; a running one stays tracked in pool_futures until it ends.
      try:
        tokenize = _track_pool_future(
            server, loop, server.encode_pool.submit(server.runtime.text_bucket, [payload.prompt])
        )
        bucket = await asyncio.wait_for(asyncio.wrap_future(tokenize), remaining_s())
      except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=timeout_detail) from exc
      except Exception as exc:  # pylint: disable=broad-exception-caught
        max_logging.log(f"Krea 2 tokenization failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Image generation failed"
        ) from exc
      log["text_tokens"] = bucket
      if bucket not in preset.buckets:
        warmed_buckets = sorted(preset.buckets)
        if bucket > warmed_buckets[-1]:
          detail = (
              f"The prompt has too many tokens for the warmed text buckets: it needs the {bucket}-token bucket, "
              f"the largest warmed bucket for {preset.label} is {warmed_buckets[-1]}. Shorten the prompt."
          )
        else:
          detail = (
              f"The prompt needs the {bucket}-token text bucket, which is not warmed for {preset.label} "
              f"(warmed: {', '.join(str(b) for b in warmed_buckets)})."
          )
        raise HTTPException(status_code=_HTTP_422, detail=detail)
      if deadline - time.perf_counter() <= 0:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=timeout_detail)
      future = server.gen_pool.submit(_run_generation, server, payload.prompt, preset, seed)
    finally:
      if future is None:
        server.release_slot()
    # No await since the submit: the done callback that releases the slot is attached before anything can end it.
    _track_pool_future(server, loop, future, server.release_slot)

    try:
      # On timeout wait_for cancels the wrapper, which cancels a still-queued generation; a running one finishes in
      # the background and keeps its slot until then.
      started, images, trace = await asyncio.wait_for(asyncio.wrap_future(future), remaining_s())
    except asyncio.TimeoutError as exc:
      raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=timeout_detail) from exc
    except Exception as exc:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Krea 2 generation failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Image generation failed") from exc

    queue_ms = _ms(started - arrived)
    log["queue_ms"] = queue_ms
    trace = trace or {}
    if len(images) != 1:
      max_logging.log(f"Krea 2 generation returned {len(images)} images; expected 1.")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Image generation failed")
    timing = {
        "queue_ms": queue_ms,
        "prompt_encoding_ms": _ms(trace.get("prompt_encoding", 0.0)),
        "denoise_ms": _ms(trace.get("denoise_loop", 0.0)),
        "vae_decode_ms": _ms(trace.get("vae_decode", 0.0)),
    }
    log.update({k: v for k, v in timing.items() if k != "queue_ms"})

    encode_started = time.perf_counter()
    try:
      encode = _track_pool_future(
          server,
          loop,
          server.encode_pool.submit(
              encode_image_base64, images[0], response_format, settings.encode_options[response_format]
          ),
      )
      b64_json = await asyncio.wait_for(asyncio.wrap_future(encode), remaining_s())
    except asyncio.TimeoutError as exc:
      raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=timeout_detail) from exc
    except Exception as exc:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Krea 2 image encoding failed: {type(exc).__name__}: {exc}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Image generation failed") from exc
    timing["encode_ms"] = _ms(time.perf_counter() - encode_started)
    timing["total_ms"] = _ms(time.perf_counter() - arrived)
    log["encode_ms"] = timing["encode_ms"]

    response_id = "img_" + uuid.uuid4().hex
    log["id"] = response_id
    return GenerationResponse(
        id=response_id,
        created=int(time.time()),
        aspect_ratio=preset.aspect_ratio,
        image_size=preset.image_size,
        width=preset.width,
        height=preset.height,
        seed=int(seed),
        response_format=response_format,
        data=[ImageData(mime_type=IMAGE_FORMATS[response_format][1], b64_json=b64_json)],
        timing=TimingData(**timing),
    )

  return app


app = create_app()


def _exit_after_pool_work(
    state: Optional[ServerState], exit_code: int, exit_fn: Optional[Callable[[int], None]] = None
) -> None:
  """After uvicorn stopped: waits at most KREA2_SHUTDOWN_GRACE_S for pool work that is still running (a
  generation, tokenization or image encode whose request got 504 or was cancelled by the drain timeout), then
  hard-exits through `exit_fn` if it has not finished.

  Pool threads cannot be interrupted, and Python's exit hook joins every ThreadPoolExecutor worker even after
  shutdown(wait=False), so without this a stuck generation or tokenization would keep the process alive forever.
  """
  if state is None:
    return
  pending = [future for future in list(state.pool_futures) if not future.done()]
  if not pending:
    return
  grace_s = state.settings.shutdown_grace_s
  max_logging.log(f"Krea 2 API: waiting up to {grace_s:g} s for {len(pending)} running pool task(s) to finish ...")
  _, not_done = concurrent.futures.wait(pending, timeout=grace_s)
  if not not_done:
    return
  max_logging.log(f"Krea 2 API: {len(not_done)} pool task(s) still running after {grace_s:g} s; exiting without them.")
  sys.stdout.flush()
  sys.stderr.flush()
  (exit_fn or os._exit)(exit_code)  # pylint: disable=protected-access


def main() -> None:
  """Runs the server (one process, one worker) and exits non-zero when startup fails."""
  import uvicorn  # pylint: disable=import-outside-toplevel

  # The same parser the app's lifespan uses (one source of truth); read here for the drain bound.
  try:
    settings = ServerSettings.from_env()
  except RuntimeError as e:
    max_logging.log(f"Krea 2 API: invalid settings: {e}")
    sys.exit(3)  # the same exit code uvicorn's lifespan failure gives below
  # Shutdown order: uvicorn stops accepting connections and drains in-flight requests for at most
  # shutdown_grace_s (then cancels them, which drops their queued generations); the lifespan then cancels every
  # queued pool future; finally _exit_after_pool_work waits at most another shutdown_grace_s for running pool work
  # and hard-exits without it.
  config = uvicorn.Config(
      app,
      host=os.environ.get("KREA2_HOST", "0.0.0.0"),
      port=int(os.environ.get("KREA2_PORT", "8000")),
      workers=1,
      lifespan="on",
      timeout_graceful_shutdown=settings.shutdown_grace_s,
  )
  server = uvicorn.Server(config)
  app.state.uvicorn_server = server
  try:
    server.run()
  except KeyboardInterrupt:
    pass
  if not server.started:
    sys.exit(3)  # uvicorn's STARTUP_FAILURE: the lifespan raised (bad KREA2_* settings)
  exit_code = getattr(app.state, "exit_code", 0)
  state = getattr(app.state, "server", None)
  if state is not None and not state.startup_finished:
    # Stopped while the runtime was still loading or warming: the generation thread cannot be interrupted and
    # the interpreter would join it at exit, so leave without waiting for the warmup to finish.
    max_logging.log("Krea 2 API stopped during startup; exiting without waiting for the warmup.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)  # pylint: disable=protected-access
  _exit_after_pool_work(state, exit_code)
  if exit_code:
    sys.exit(exit_code)


if __name__ == "__main__":
  main()
