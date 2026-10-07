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

# CPU tests for the Krea 2 HTTP server (serve_krea2). No model: a FakeRuntime implements the duck-typed runtime
# interface, with threading.Event gates that hold a generation (or the loader) open.

import asyncio
import base64
from contextlib import asynccontextmanager
from io import BytesIO
import threading
import time
import types

import pytest

pytest.importorskip("fastapi")
import httpx  # pylint: disable=wrong-import-position
from PIL import Image  # pylint: disable=wrong-import-position

from maxdiffusion import serve_krea2  # pylint: disable=wrong-import-position
from maxdiffusion.models.krea2.resolution_presets import resolve_krea2_resolution  # pylint: disable=wrong-import-position

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ORIGIN = "https://frontend.example"
GATE_TIMEOUT_S = 10.0


class FakeRuntime:
  """Implements the runtime interface serve_krea2 codes against."""

  def __init__(self):
    self.config = types.SimpleNamespace(
        batch_size=1,
        max_sequence_length=512,
        output_name="krea2.png",
        krea2_precompile="",
        guidance_scale=0.0,
        do_classifier_free_guidance=False,
        negative_prompt="",
    )
    self.gate = threading.Event()  # generate() blocks until this is set
    self.gate.set()
    self.bucket_gate = threading.Event()  # text_bucket() blocks until this is set
    self.bucket_gate.set()
    self.bucket_calls = []
    self.calls = []
    self.generate_started = threading.Event()
    self.buckets = {}  # prompt -> text bucket; default 128
    self.error = None
    self.serving_plan_args = None
    self.precompiled = None
    self.thread_names = set()

  def serving_plan(self, presets_spec, text_tokens):
    self.serving_plan_args = (presets_spec, list(text_tokens))
    self.thread_names.add(threading.current_thread().name)
    plan = []
    for ratio, size, buckets in (("1:1", "1k", (128, 256)), ("16:9", "1k", (128, 256)), ("9:16", "2k", (128,))):
      for bucket in buckets:
        plan.append((resolve_krea2_resolution(ratio, size), bucket))
    return plan

  def precompile(self, plan):
    self.precompiled = list(plan)
    self.thread_names.add(threading.current_thread().name)
    return [{"label": resolution.label, "text_tokens": bucket} for resolution, bucket in plan]

  def text_bucket(self, prompts):
    assert len(prompts) == 1
    self.bucket_calls.append(prompts[0])
    if not self.bucket_gate.wait(GATE_TIMEOUT_S):
      raise RuntimeError("bucket gate was never released")
    return self.buckets.get(prompts[0], 128)

  def generate(self, prompts, *, height, width, seed=None, output_type="pil"):
    self.calls.append(
        {"prompts": list(prompts), "height": height, "width": width, "seed": seed, "output_type": output_type}
    )
    self.thread_names.add(threading.current_thread().name)
    self.generate_started.set()
    if not self.gate.wait(GATE_TIMEOUT_S):
      raise RuntimeError("test gate was never released")
    if self.error is not None:
      raise self.error
    trace = {
        "prompt_encoding": 0.011,
        "denoise_loop": 0.022,
        "vae_decode": 0.033,
        "seed": seed,
    }
    return [Image.new("RGB", (width, height), (200, 30, 30))], trace


@pytest.fixture
def env(monkeypatch):
  for name in (
      "KREA2_CONFIG",
      "KREA2_CONFIG_OVERRIDES",
      "KREA2_MODEL_PATH",
      "KREA2_OUTPUT_DIR",
      "KREA2_FORMATS",
      "KREA2_MAX_QUEUE",
      "KREA2_REQUEST_TIMEOUT_S",
      "KREA2_ENCODE_THREADS",
      "KREA2_SERVE_PRESETS",
      "KREA2_SERVE_TEXT_TOKENS",
      "KREA2_SHUTDOWN_GRACE_S",
      "KREA2_AVIF_QUALITY",
      "KREA2_AVIF_SPEED",
      "KREA2_AVIF_THREADS",
      "KREA2_WEBP_QUALITY",
      "KREA2_JPEG_QUALITY",
  ):
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setenv("KREA2_API_TOKEN", TOKEN)
  monkeypatch.setenv("KREA2_ALLOWED_ORIGINS", ORIGIN)
  return monkeypatch


def _make_app(runtime=None, loader=None):
  runtime = runtime or FakeRuntime()
  failures = []
  app = serve_krea2.create_app(loader or (lambda: runtime), on_startup_failure=lambda app, exc: failures.append(exc))
  return app, runtime, failures


async def _wait_until(predicate, timeout=GATE_TIMEOUT_S):
  deadline = time.monotonic() + timeout
  while not predicate():
    if time.monotonic() > deadline:
      raise AssertionError("condition not reached in time")
    await asyncio.sleep(0.01)


@asynccontextmanager
async def _client(app, *, wait_ready=True, raise_app_exceptions=True):
  async with app.router.lifespan_context(app):
    if wait_ready:
      await asyncio.wait_for(asyncio.shield(app.state.server.startup_task), GATE_TIMEOUT_S)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
      yield client


def _generate(client, **body):
  body.setdefault("prompt", "a fox in the snow")
  return client.post("/v1/images/generations", headers=AUTH, json=body)


def _decode(body):
  return Image.open(BytesIO(base64.b64decode(body["data"][0]["b64_json"])))


def test_every_v1_route_requires_the_bearer_token(env):
  app, runtime, _ = _make_app()

  async def body():
    async with _client(app) as client:
      results = [
          await client.post("/v1/images/generations", json={"prompt": "a fox"}),
          await client.post(
              "/v1/images/generations", json={"prompt": "a fox"}, headers={"Authorization": "Bearer wrong"}
          ),
          await client.post("/v1/images/generations", json={"prompt": "a fox"}, headers={"Authorization": TOKEN}),
          await client.get("/v1/presets"),
          await client.get("/v1/presets", headers={"Authorization": "Bearer wrong"}),
      ]
      ok = await client.get("/v1/presets", headers=AUTH)
    return results, ok

  results, ok = asyncio.run(body())
  assert [r.status_code for r in results] == [401] * 5
  assert all(r.headers["www-authenticate"] == "Bearer" for r in results)
  assert ok.status_code == 200
  assert runtime.calls == []


def test_readyz_is_503_until_startup_completes(env):
  runtime = FakeRuntime()
  loader_gate = threading.Event()

  def loader():
    if not loader_gate.wait(GATE_TIMEOUT_S):
      raise RuntimeError("loader gate never released")
    return runtime

  app, _, failures = _make_app(runtime, loader)

  async def body():
    async with _client(app, wait_ready=False) as client:
      try:
        before = await client.get("/readyz")
        live = await client.get("/livez")
        presets = await client.get("/v1/presets", headers=AUTH)
        gen = await _generate(client)
      finally:
        loader_gate.set()
      await asyncio.wait_for(asyncio.shield(app.state.server.startup_task), GATE_TIMEOUT_S)
      after = await client.get("/readyz")
    return before, live, presets, gen, after

  before, live, presets, gen, after = asyncio.run(body())
  assert before.status_code == 503
  assert before.json() == {"detail": "Model is not ready"}
  assert live.status_code == 200 and live.json() == {"status": "ok"}
  assert presets.status_code == 503
  assert gen.status_code == 503
  assert after.status_code == 200
  assert after.json() == {"status": "ready", "queue_depth": 0, "busy": False}
  assert failures == []
  assert runtime.serving_plan_args == ("all", [128, 256])
  assert [(r.label, b) for r, b in runtime.precompiled] == [
      ("1k@1:1", 128),
      ("1k@1:1", 256),
      ("1k@16:9", 128),
      ("1k@16:9", 256),
      ("2k@9:16", 128),
  ]
  # The loader, plan and precompile run on the generation thread, like every later generate call.
  assert all(name.startswith("krea2-generate") for name in runtime.thread_names)


@pytest.mark.parametrize(
    "response_format, mime_type, pil_format",
    [
        ("avif", "image/avif", "AVIF"),
        ("webp", "image/webp", "WEBP"),
        ("jpeg", "image/jpeg", "JPEG"),
        ("png", "image/png", "PNG"),
    ],
)
def test_generate_returns_each_format(env, response_format, mime_type, pil_format):
  app, runtime, _ = _make_app()

  async def body():
    async with _client(app) as client:
      return await _generate(
          client,
          prompt="  a fox in the snow  ",
          aspect_ratio="16:9",
          image_size="1k",
          seed=42,
          response_format=response_format,
      )

  response = asyncio.run(body())
  assert response.status_code == 200, response.text
  result = response.json()
  assert result["id"].startswith("img_")
  assert result["model"] == "krea-2-turbo"
  assert (result["aspect_ratio"], result["image_size"]) == ("16:9", "1k")
  assert (result["width"], result["height"]) == (1344, 768)
  assert result["seed"] == 42
  assert result["response_format"] == response_format
  assert len(result["data"]) == 1
  assert result["data"][0]["mime_type"] == mime_type
  image = _decode(result)
  assert image.format == pil_format
  assert image.size == (1344, 768)
  timing = result["timing"]
  assert set(timing) == {"queue_ms", "prompt_encoding_ms", "denoise_ms", "vae_decode_ms", "encode_ms", "total_ms"}
  assert (timing["prompt_encoding_ms"], timing["denoise_ms"], timing["vae_decode_ms"]) == (11, 22, 33)
  assert timing["total_ms"] >= timing["encode_ms"]
  assert runtime.calls == [
      {"prompts": ["a fox in the snow"], "height": 768, "width": 1344, "seed": 42, "output_type": "pil"}
  ]


def test_defaults_are_avif_1k_square(env):
  app, runtime, _ = _make_app()

  async def body():
    async with _client(app) as client:
      return await _generate(client)

  response = asyncio.run(body())
  assert response.status_code == 200, response.text
  result = response.json()
  assert result["response_format"] == "avif"
  assert result["data"][0]["mime_type"] == "image/avif"
  assert _decode(result).format == "AVIF"
  assert (result["aspect_ratio"], result["image_size"], result["width"], result["height"]) == ("1:1", "1k", 1024, 1024)
  assert runtime.calls[0]["height"] == 1024 and runtime.calls[0]["width"] == 1024


def test_unknown_and_disabled_formats_are_rejected(env):
  env.setenv("KREA2_FORMATS", "png,jpeg")
  app, runtime, _ = _make_app()

  async def body():
    async with _client(app) as client:
      unknown = await _generate(client, response_format="gif")
      disabled = await _generate(client, response_format="avif")
      default = await _generate(client)
      presets = await client.get("/v1/presets", headers=AUTH)
    return unknown, disabled, default, presets

  unknown, disabled, default, presets = asyncio.run(body())
  assert unknown.status_code == 422
  assert disabled.status_code == 422
  assert "png, jpeg" in disabled.json()["detail"]
  # With avif disabled the default is the first enabled format.
  assert default.status_code == 200
  assert default.json()["response_format"] == "png"
  assert presets.json()["formats"] == ["png", "jpeg"]
  assert presets.json()["default_format"] == "png"
  assert len(runtime.calls) == 1


def test_request_validation(env):
  app, runtime, _ = _make_app()

  async def body():
    async with _client(app) as client:
      return [
          await _generate(client, prompt="   "),
          await _generate(client, prompt="x" * 4001),
          await _generate(client, aspect_ratio="7:4"),
          await _generate(client, image_size="4k"),
          await _generate(client, seed=-1),
          await _generate(client, seed=2**31),
          await _generate(client, lora="s3400"),
      ]

  responses = asyncio.run(body())
  assert [r.status_code for r in responses] == [422] * len(responses)
  assert runtime.calls == []


def test_unwarmed_preset_is_rejected_with_the_warmed_list(env):
  app, runtime, _ = _make_app()

  async def body():
    async with _client(app) as client:
      return await _generate(client, aspect_ratio="1:1", image_size="2k")

  response = asyncio.run(body())
  assert response.status_code == 422
  detail = response.json()["detail"]
  assert "2k@1:1" in detail
  assert "1k@1:1, 1k@16:9, 2k@9:16" in detail
  assert runtime.calls == []


def test_prompt_needing_an_unwarmed_text_bucket_is_rejected(env):
  runtime = FakeRuntime()
  runtime.buckets = {"long prompt": 384, "medium prompt": 256}
  app, _, _ = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      return (
          await _generate(client, prompt="long prompt"),
          await _generate(client, prompt="medium prompt", aspect_ratio="9:16", image_size="2k"),
          await _generate(client, prompt="medium prompt"),
      )

  too_long, too_long_2k, ok = asyncio.run(body())
  assert too_long.status_code == 422
  assert "too many tokens" in too_long.json()["detail"]
  assert "384" in too_long.json()["detail"] and "256" in too_long.json()["detail"]
  assert too_long_2k.status_code == 422
  assert "2k@9:16 is 128" in too_long_2k.json()["detail"]
  assert ok.status_code == 200
  assert [call["prompts"] for call in runtime.calls] == [["medium prompt"]]


def test_seed_is_echoed_or_drawn_fresh_by_the_server(env):
  app, runtime, _ = _make_app()

  async def body():
    async with _client(app) as client:
      fixed = await _generate(client, seed=7, response_format="png")
      drawn = [await _generate(client, response_format="png") for _ in range(3)]
      return fixed, drawn

  fixed, drawn = asyncio.run(body())
  assert fixed.json()["seed"] == 7
  assert all(r.status_code == 200 for r in drawn)
  drawn_seeds = [r.json()["seed"] for r in drawn]
  # An omitted seed is drawn by the server for every request (never the config's seed) and passed to the runtime.
  assert all(isinstance(seed, int) and 0 <= seed <= serve_krea2.MAX_SEED for seed in drawn_seeds)
  assert len(set(drawn_seeds)) > 1
  assert [call["seed"] for call in runtime.calls] == [7, *drawn_seeds]


def test_a_failing_seed_draw_does_not_leak_the_slot(env, monkeypatch):
  env.setenv("KREA2_MAX_QUEUE", "1")
  app, runtime, _ = _make_app()
  original = serve_krea2.secrets.randbelow
  seed_draws = []

  def flaky_randbelow(n):
    if n == serve_krea2.MAX_SEED + 1:
      seed_draws.append(n)
      if len(seed_draws) == 1:
        raise OSError("no entropy")
    return original(n)

  monkeypatch.setattr(serve_krea2.secrets, "randbelow", flaky_randbelow)

  async def body():
    async with _client(app, raise_app_exceptions=False) as client:
      failed = await _generate(client, response_format="png")
      in_flight = app.state.server.in_flight
      # With KREA2_MAX_QUEUE=1 a leaked slot would turn this into a 429.
      ok = await _generate(client, response_format="png")
    return failed, in_flight, ok

  failed, in_flight, ok = asyncio.run(body())
  assert failed.status_code == 500
  assert in_flight == 0
  assert ok.status_code == 200, ok.text
  assert len(seed_draws) == 2
  assert len(runtime.calls) == 1


def test_queue_full_answers_429_without_queueing(env):
  env.setenv("KREA2_MAX_QUEUE", "2")
  runtime = FakeRuntime()
  runtime.gate.clear()
  app, _, _ = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      try:
        first = asyncio.create_task(_generate(client, response_format="png"))
        second = asyncio.create_task(_generate(client, response_format="png"))
        await _wait_until(lambda: app.state.server.in_flight == 2)
        rejected = await _generate(client, response_format="png")
        ready = await client.get("/readyz")
      finally:
        runtime.gate.set()
      done = await asyncio.gather(first, second)
      await _wait_until(lambda: app.state.server.in_flight == 0)
      after = await _generate(client, response_format="png")
    return rejected, ready, done, after

  rejected, ready, done, after = asyncio.run(body())
  assert rejected.status_code == 429
  assert rejected.headers["retry-after"] == "5"
  assert ready.json() == {"status": "ready", "queue_depth": 2, "busy": True}
  assert [r.status_code for r in done] == [200, 200]
  assert after.status_code == 200
  assert len(runtime.calls) == 3


def test_429_is_answered_while_the_encode_pool_is_saturated(env):
  env.setenv("KREA2_MAX_QUEUE", "2")
  env.setenv("KREA2_ENCODE_THREADS", "1")
  runtime = FakeRuntime()
  runtime.bucket_gate.clear()
  app, _, _ = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      try:
        tasks = [asyncio.create_task(_generate(client, prompt=f"p{i}", response_format="png")) for i in range(3)]
        # Admission happens before tokenization, so the extra request is turned away while every tokenization
        # (one running, one queued on the single encode thread) is still blocked.
        done, _ = await asyncio.wait(tasks, timeout=2.0, return_when=asyncio.FIRST_COMPLETED)
        early = [task.result() for task in done]
        in_flight = app.state.server.in_flight
      finally:
        runtime.bucket_gate.set()
      return early, in_flight, await asyncio.gather(*tasks)

  early, in_flight, results = asyncio.run(body())
  assert [r.status_code for r in early] == [429]
  assert early[0].headers["retry-after"] == "5"
  assert in_flight == 2
  assert sorted(r.status_code for r in results) == [200, 200, 429]
  assert len(runtime.calls) == 2
  # The rejected request never reached the tokenizer.
  assert len(runtime.bucket_calls) == 2


def test_tokenization_timeout_answers_504_and_releases_the_slot(env):
  env.setenv("KREA2_REQUEST_TIMEOUT_S", "0.3")
  env.setenv("KREA2_MAX_QUEUE", "1")
  runtime = FakeRuntime()
  runtime.bucket_gate.clear()
  app, _, _ = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      try:
        started = time.monotonic()
        timed_out = await asyncio.wait_for(_generate(client, response_format="png"), GATE_TIMEOUT_S)
        elapsed = time.monotonic() - started
        in_flight = app.state.server.in_flight
      finally:
        runtime.bucket_gate.set()
      return timed_out, elapsed, in_flight, await _generate(client, response_format="png")

  timed_out, elapsed, in_flight, after = asyncio.run(body())
  assert timed_out.status_code == 504
  assert "0.3 s" in timed_out.json()["detail"]
  assert elapsed < 2.0
  assert in_flight == 0
  assert after.status_code == 200
  assert len(runtime.calls) == 1


def test_timeout_answers_504_and_the_slot_frees_when_the_generation_ends(env):
  env.setenv("KREA2_REQUEST_TIMEOUT_S", "0.2")
  runtime = FakeRuntime()
  runtime.gate.clear()
  app, _, _ = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      try:
        running, queued = await asyncio.gather(
            _generate(client, response_format="png"), _generate(client, response_format="png")
        )
        # The running generation keeps its slot; the one still queued was dropped.
        await _wait_until(lambda: app.state.server.in_flight == 1)
        busy = await client.get("/readyz")
      finally:
        runtime.gate.set()
      await _wait_until(lambda: app.state.server.in_flight == 0)
      idle = await client.get("/readyz")
      return running, queued, busy, idle

  running, queued, busy, idle = asyncio.run(body())
  assert running.status_code == 504
  assert queued.status_code == 504
  assert "0.2 s" in running.json()["detail"]
  assert busy.json() == {"status": "ready", "queue_depth": 1, "busy": True}
  assert idle.json() == {"status": "ready", "queue_depth": 0, "busy": False}
  assert len(runtime.calls) == 1


def test_slot_is_reusable_after_a_timeout(env):
  env.setenv("KREA2_REQUEST_TIMEOUT_S", "0.3")
  runtime = FakeRuntime()
  runtime.gate.clear()
  app, _, _ = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      try:
        timed_out = await _generate(client, response_format="png")
      finally:
        runtime.gate.set()
      await _wait_until(lambda: app.state.server.in_flight == 0)
      return timed_out, await _generate(client, response_format="png")

  timed_out, after = asyncio.run(body())
  assert timed_out.status_code == 504
  assert after.status_code == 200
  assert len(runtime.calls) == 2


def test_encode_timeout_answers_504_and_the_encode_stays_tracked(env, monkeypatch):
  env.setenv("KREA2_REQUEST_TIMEOUT_S", "0.5")
  runtime = FakeRuntime()
  app, _, _ = _make_app(runtime)
  encode_gate = threading.Event()
  original = serve_krea2.encode_image_base64

  def gated_encode(image, response_format, options):
    if not encode_gate.wait(GATE_TIMEOUT_S):
      raise RuntimeError("encode gate never released")
    return original(image, response_format, options)

  monkeypatch.setattr(serve_krea2, "encode_image_base64", gated_encode)

  async def body():
    async with _client(app) as client:
      try:
        started = time.monotonic()
        timed_out = await asyncio.wait_for(_generate(client, response_format="png"), GATE_TIMEOUT_S)
        elapsed = time.monotonic() - started
        tracked = [f for f in app.state.server.pool_futures if not f.done()]
        in_flight = app.state.server.in_flight
      finally:
        encode_gate.set()
      await _wait_until(lambda: not app.state.server.pool_futures)
      return timed_out, elapsed, tracked, in_flight

  timed_out, elapsed, tracked, in_flight = asyncio.run(body())
  assert timed_out.status_code == 504
  assert "0.5 s" in timed_out.json()["detail"]
  assert elapsed < 2.0
  # The encode outlived its request; main() would find it here and wait at most KREA2_SHUTDOWN_GRACE_S.
  assert len(tracked) == 1
  assert in_flight == 0
  assert len(runtime.calls) == 1


def test_health_endpoints_answer_while_a_generation_runs(env):
  runtime = FakeRuntime()
  runtime.gate.clear()
  app, _, _ = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      try:
        pending = asyncio.create_task(_generate(client, response_format="png"))
        await _wait_until(runtime.generate_started.is_set)
        live = await asyncio.wait_for(client.get("/livez"), 2)
        ready = await asyncio.wait_for(client.get("/readyz"), 2)
        presets = await asyncio.wait_for(client.get("/v1/presets", headers=AUTH), 2)
      finally:
        runtime.gate.set()
      return live, ready, presets, await pending

  live, ready, presets, generated = asyncio.run(body())
  assert live.status_code == 200
  assert ready.json() == {"status": "ready", "queue_depth": 1, "busy": True}
  assert presets.status_code == 200
  assert generated.status_code == 200


def test_encoding_runs_after_the_generation_slot_is_released(env, monkeypatch):
  env.setenv("KREA2_MAX_QUEUE", "1")
  runtime = FakeRuntime()
  app, _, _ = _make_app(runtime)
  encode_gate = threading.Event()
  encode_entered = threading.Event()
  original = serve_krea2.encode_image_base64
  encode_threads = []

  def gated_encode(image, response_format, options):
    encode_threads.append(threading.current_thread().name)
    if not encode_entered.is_set():
      encode_entered.set()
      if not encode_gate.wait(GATE_TIMEOUT_S):
        raise RuntimeError("encode gate never released")
    return original(image, response_format, options)

  monkeypatch.setattr(serve_krea2, "encode_image_base64", gated_encode)

  async def body():
    async with _client(app) as client:
      try:
        first = asyncio.create_task(_generate(client, prompt="first", response_format="png"))
        await _wait_until(encode_entered.is_set)
        await _wait_until(lambda: app.state.server.in_flight == 0)
        ready_while_encoding = await client.get("/readyz")
        # With KREA2_MAX_QUEUE=1 this is only admitted because the first request gave its slot back.
        second = await asyncio.wait_for(_generate(client, prompt="second", response_format="png"), GATE_TIMEOUT_S)
        first_done_early = first.done()
      finally:
        encode_gate.set()
      return await first, second, first_done_early, ready_while_encoding

  first, second, first_done_early, ready_while_encoding = asyncio.run(body())
  # The second request generated and encoded while the first one's encode was still blocked.
  assert ready_while_encoding.json() == {"status": "ready", "queue_depth": 0, "busy": False}
  assert not first_done_early
  assert second.status_code == 200 and first.status_code == 200
  assert [call["prompts"] for call in runtime.calls] == [["first"], ["second"]]
  assert all(name.startswith("krea2-encode") for name in encode_threads)


def test_cors_preflight_allows_the_configured_origin(env):
  app, _, _ = _make_app()

  async def body():
    async with _client(app) as client:
      return await client.options(
          "/v1/images/generations",
          headers={
              "Origin": ORIGIN,
              "Access-Control-Request-Method": "POST",
              "Access-Control-Request-Headers": "authorization,content-type",
          },
      )

  response = asyncio.run(body())
  assert response.status_code == 200
  assert response.headers["access-control-allow-origin"] == ORIGIN


def test_presets_endpoint_lists_the_warmed_set(env):
  env.setenv("KREA2_MAX_QUEUE", "3")
  env.setenv("KREA2_REQUEST_TIMEOUT_S", "45")
  app, _, _ = _make_app()

  async def body():
    async with _client(app) as client:
      return await client.get("/v1/presets", headers=AUTH)

  response = asyncio.run(body())
  assert response.status_code == 200
  assert response.json() == {
      "presets": [
          {"aspect_ratio": "1:1", "image_size": "1k", "width": 1024, "height": 1024, "text_tokens": [128, 256]},
          {"aspect_ratio": "16:9", "image_size": "1k", "width": 1344, "height": 768, "text_tokens": [128, 256]},
          {"aspect_ratio": "9:16", "image_size": "2k", "width": 1536, "height": 2688, "text_tokens": [128]},
      ],
      "formats": ["avif", "webp", "jpeg", "png"],
      "default_format": "avif",
      "max_queue": 3,
      "request_timeout_s": 45.0,
  }


def test_runtime_exception_answers_500_and_frees_the_slot(env):
  runtime = FakeRuntime()
  runtime.error = ValueError("device on fire")
  app, _, _ = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      failed = await _generate(client)
      await _wait_until(lambda: app.state.server.in_flight == 0)
      runtime.error = None
      return failed, await _generate(client, response_format="png")

  failed, ok = asyncio.run(body())
  assert failed.status_code == 500
  assert failed.json() == {"detail": "Image generation failed"}
  assert ok.status_code == 200


def test_request_id_is_echoed_or_generated(env):
  app, _, _ = _make_app()

  async def body():
    async with _client(app) as client:
      return (
          await client.get("/livez", headers={"X-Request-ID": "client-123"}),
          await client.get("/livez", headers={"X-Request-ID": "bad id with spaces"}),
          await _generate(client, response_format="png", seed=1),
      )

  echoed, replaced, generated = asyncio.run(body())
  assert echoed.headers["x-request-id"] == "client-123"
  assert replaced.headers["x-request-id"] != "bad id with spaces"
  assert len(replaced.headers["x-request-id"]) == 32
  assert generated.headers["x-request-id"]


def test_startup_failure_calls_the_failure_policy(env):
  def loader():
    raise FileNotFoundError("no model here")

  app, _, failures = _make_app(loader=loader)

  async def body():
    async with _client(app, wait_ready=False) as client:
      await asyncio.wait_for(asyncio.shield(app.state.server.startup_task), GATE_TIMEOUT_S)
      return await client.get("/readyz")

  ready = asyncio.run(body())
  assert ready.status_code == 503
  assert ready.json() == {"detail": "Model startup failed"}
  assert len(failures) == 1 and isinstance(failures[0], FileNotFoundError)


def test_startup_rejects_a_runtime_with_batch_size_above_one(env):
  runtime = FakeRuntime()
  runtime.config.batch_size = 2
  app, _, failures = _make_app(runtime)

  async def body():
    async with _client(app, wait_ready=False):
      await asyncio.wait_for(asyncio.shield(app.state.server.startup_task), GATE_TIMEOUT_S)

  asyncio.run(body())
  assert len(failures) == 1 and "batch_size" in str(failures[0])
  assert runtime.precompiled is None


@pytest.mark.parametrize(
    "negative_prompt, bucket, missing",
    [
        ("long negative prompt", 384, "1k@1:1, 1k@16:9, 2k@9:16"),
        ("medium negative prompt", 256, "for 2k@9:16 "),  # warmed for the 1k presets only
    ],
)
def test_startup_requires_the_cfg_negative_prompt_bucket_to_be_warmed(env, negative_prompt, bucket, missing):
  runtime = FakeRuntime()
  runtime.config.guidance_scale = 3.5
  runtime.config.do_classifier_free_guidance = True
  runtime.config.negative_prompt = negative_prompt
  runtime.buckets = {negative_prompt: bucket}
  app, _, failures = _make_app(runtime)

  async def body():
    async with _client(app, wait_ready=False) as client:
      await asyncio.wait_for(asyncio.shield(app.state.server.startup_task), GATE_TIMEOUT_S)
      return await client.get("/readyz")

  ready = asyncio.run(body())
  assert ready.status_code == 503
  assert len(failures) == 1 and isinstance(failures[0], RuntimeError)
  message = str(failures[0])
  assert f"{bucket}-token" in message and "KREA2_SERVE_TEXT_TOKENS" in message and "negative_prompt" in message
  assert missing in message
  assert runtime.precompiled is None
  assert runtime.bucket_calls == [negative_prompt]


@pytest.mark.parametrize(
    "guidance_scale, do_cfg, negative_prompt, expected_bucket_calls",
    [
        (3.5, True, "blurry", ["blurry"]),  # CFG on, bucket 128 is warmed for every preset
        (3.5, True, ["blurry"], ["blurry"]),  # a one-entry list is passed as is, not wrapped in another list
        (3.5, True, None, [""]),  # None means the empty negative prompt, like the pipeline
        (3.5, None, "", [""]),  # CFG on by guidance alone, empty negative prompt -> bucket 128
        (3.5, False, "long negative prompt", []),  # CFG switched off: the negative prompt is never encoded
        (0.0, True, "long negative prompt", []),  # turbo: guidance 0 disables CFG
    ],
)
def test_startup_accepts_a_warmed_or_unused_negative_prompt(
    env, guidance_scale, do_cfg, negative_prompt, expected_bucket_calls
):
  runtime = FakeRuntime()
  runtime.config.guidance_scale = guidance_scale
  runtime.config.do_classifier_free_guidance = do_cfg
  runtime.config.negative_prompt = negative_prompt
  runtime.buckets = {"blurry": 128, "long negative prompt": 384}
  app, _, failures = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      return await client.get("/readyz")

  ready = asyncio.run(body())
  assert failures == []
  assert ready.status_code == 200
  assert runtime.bucket_calls == expected_bucket_calls


@pytest.mark.parametrize("do_cfg", [True, False])
def test_startup_rejects_a_negative_prompt_list_that_is_not_one_entry(env, do_cfg):
  runtime = FakeRuntime()
  runtime.config.guidance_scale = 3.5
  runtime.config.do_classifier_free_guidance = do_cfg
  runtime.config.negative_prompt = ["a", "b"]
  app, _, failures = _make_app(runtime)

  async def body():
    async with _client(app, wait_ready=False) as client:
      await asyncio.wait_for(asyncio.shield(app.state.server.startup_task), GATE_TIMEOUT_S)
      return await client.get("/readyz")

  ready = asyncio.run(body())
  assert ready.status_code == 503
  # The pipeline would reject two negative prompts for batch_size 1 on every request, CFG or not.
  assert len(failures) == 1 and isinstance(failures[0], RuntimeError)
  assert "negative_prompt" in str(failures[0]) and "2 entries" in str(failures[0])
  assert runtime.bucket_calls == []
  assert runtime.precompiled is None


@pytest.mark.parametrize(
    "overrides",
    [
        "krea2_precompile=all",
        "aot_cache_dir=/x krea2_precompile='2k@16:9'",
        "krea2_weight_cache_build_only=True",
        "batch_size=4",
        "not_a_pair",
    ],
)
def test_startup_refuses_unservable_overrides(env, overrides):
  env.setenv("KREA2_CONFIG_OVERRIDES", overrides)
  with pytest.raises(RuntimeError):
    serve_krea2._runtime_argv()  # pylint: disable=protected-access

  # The default loader validates the argv when the app starts, before importing generate_krea2.
  app = serve_krea2.create_app(on_startup_failure=lambda app, exc: None)

  async def body():
    async with app.router.lifespan_context(app):
      pass

  with pytest.raises(RuntimeError):
    asyncio.run(body())


def test_startup_requires_a_token(env):
  env.delenv("KREA2_API_TOKEN")
  app, _, _ = _make_app()

  async def body():
    async with app.router.lifespan_context(app):
      pass

  with pytest.raises(RuntimeError, match="KREA2_API_TOKEN"):
    asyncio.run(body())


def test_runtime_argv_composition_from_env(env):
  env.setenv("KREA2_CONFIG", "/configs/base_krea2_turbo_v6e1.yml")
  env.setenv(
      "KREA2_CONFIG_OVERRIDES",
      "aot_cache_dir=/mnt/aot aot_cache_lazy_load=True aot_cache_gcs=gs://bucket/aot "
      "'krea2_weight_cache_dir=/mnt/weights dir' krea2_precompile= krea2_weight_cache_build_only=False batch_size=1",
  )
  env.setenv("KREA2_MODEL_PATH", "/models/Krea-2-Turbo")
  env.setenv("KREA2_OUTPUT_DIR", "/srv/out")

  argv = serve_krea2._runtime_argv()  # pylint: disable=protected-access
  assert argv == [
      "maxdiffusion.serve_krea2",
      "/configs/base_krea2_turbo_v6e1.yml",
      "aot_cache_dir=/mnt/aot",
      "aot_cache_lazy_load=True",
      "aot_cache_gcs=gs://bucket/aot",
      "krea2_weight_cache_dir=/mnt/weights dir",
      "krea2_precompile=",
      "krea2_weight_cache_build_only=False",
      "batch_size=1",
      "pretrained_model_name_or_path=/models/Krea-2-Turbo",
      "run_name=krea2_api",
      "output_dir=/srv/out",
      "batch_size=1",
      "prompt=warmup",
  ]


def test_runtime_argv_defaults(env):
  argv = serve_krea2._runtime_argv()  # pylint: disable=protected-access
  assert argv[1].endswith("configs/base_krea2_turbo.yml")
  assert argv[2:] == ["run_name=krea2_api", "output_dir=/tmp/krea2-api", "batch_size=1", "prompt=warmup"]


def test_settings_from_env(env):
  env.setenv("KREA2_SERVE_PRESETS", "1k,2k@16:9")
  env.setenv("KREA2_SERVE_TEXT_TOKENS", "128, 256,384")
  env.setenv("KREA2_AVIF_QUALITY", "70")
  settings = serve_krea2.ServerSettings.from_env()
  assert settings.presets_spec == "1k,2k@16:9"
  assert settings.text_tokens == (128, 256, 384)
  assert settings.max_queue == 4
  assert settings.request_timeout_s == 60.0
  assert settings.encode_threads == 2
  assert settings.encode_options["avif"] == {"quality": 70, "speed": 8, "subsampling": "4:4:4", "max_threads": 4}
  assert settings.encode_options["jpeg"] == {"quality": 92, "subsampling": 0}
  assert settings.encode_options["webp"] == {"quality": 90, "method": 4}
  assert settings.encode_options["png"] == {"compress_level": 1}
  assert settings.shutdown_grace_s == 10.0
  for name, value in (
      ("KREA2_FORMATS", "gif"),
      ("KREA2_MAX_QUEUE", "0"),
      ("KREA2_REQUEST_TIMEOUT_S", "x"),
      ("KREA2_SHUTDOWN_GRACE_S", "0"),
      ("KREA2_AVIF_QUALITY", "101"),
      ("KREA2_AVIF_QUALITY", "-1"),
      ("KREA2_AVIF_SPEED", "11"),
      ("KREA2_WEBP_QUALITY", "101"),
      ("KREA2_JPEG_QUALITY", "-1"),
      ("KREA2_JPEG_QUALITY", "101"),
  ):
    env.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
      serve_krea2.ServerSettings.from_env()
    env.delenv(name)
  for name, value in (
      ("KREA2_AVIF_QUALITY", "100"),
      ("KREA2_AVIF_SPEED", "10"),
      ("KREA2_WEBP_QUALITY", "0"),
      ("KREA2_JPEG_QUALITY", "100"),
  ):
    env.setenv(name, value)
    serve_krea2.ServerSettings.from_env()
    env.delenv(name)


@pytest.mark.parametrize("name, value", [("KREA2_AVIF_SPEED", "11"), ("KREA2_JPEG_QUALITY", "101")])
def test_startup_refuses_out_of_range_encoder_options(env, name, value):
  env.setenv(name, value)
  app, runtime, _ = _make_app()

  async def body():
    async with app.router.lifespan_context(app):
      pass

  with pytest.raises(RuntimeError, match=name):
    asyncio.run(body())
  assert runtime.serving_plan_args is None


def _shutdown_state(env, grace_s):
  env.setenv("KREA2_SHUTDOWN_GRACE_S", str(grace_s))
  settings = serve_krea2.ServerSettings.from_env()
  pool = serve_krea2.concurrent.futures.ThreadPoolExecutor(max_workers=1)
  state = serve_krea2.ServerState(settings=settings, gen_pool=pool, encode_pool=pool)
  return state, pool


def test_shutdown_hard_exits_after_the_grace_when_a_generation_still_runs(env):
  state, pool = _shutdown_state(env, 0.2)
  release = threading.Event()
  try:
    state.pool_futures.add(pool.submit(release.wait, GATE_TIMEOUT_S))
    exits = []
    started = time.monotonic()
    serve_krea2._exit_after_pool_work(state, 7, exits.append)  # pylint: disable=protected-access
    elapsed = time.monotonic() - started
  finally:
    release.set()
    pool.shutdown(wait=True)
  assert exits == [7]
  assert 0.15 <= elapsed < 2.0


def test_shutdown_returns_when_the_generation_finishes_within_the_grace(env):
  state, pool = _shutdown_state(env, 5)
  release = threading.Event()
  try:
    state.pool_futures.add(pool.submit(release.wait, GATE_TIMEOUT_S))
    threading.Timer(0.1, release.set).start()
    exits = []
    started = time.monotonic()
    serve_krea2._exit_after_pool_work(state, 0, exits.append)  # pylint: disable=protected-access
    elapsed = time.monotonic() - started
    # Nothing in flight (or no state at all): returns at once.
    serve_krea2._exit_after_pool_work(state, 0, exits.append)  # pylint: disable=protected-access
    serve_krea2._exit_after_pool_work(None, 0, exits.append)  # pylint: disable=protected-access
  finally:
    release.set()
    pool.shutdown(wait=True)
  assert exits == []
  assert elapsed < 2.0


def test_pool_futures_are_tracked_until_they_finish(env):
  env.setenv("KREA2_REQUEST_TIMEOUT_S", "0.2")
  runtime = FakeRuntime()
  runtime.gate.clear()
  app, _, _ = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      try:
        timed_out = await _generate(client, response_format="png")
        tracked = [f for f in app.state.server.pool_futures if not f.done()]
      finally:
        runtime.gate.set()
      await _wait_until(lambda: not app.state.server.pool_futures)
      return timed_out, tracked

  timed_out, tracked = asyncio.run(body())
  # The 504 left its generation running; main() would find it here and wait at most KREA2_SHUTDOWN_GRACE_S.
  assert timed_out.status_code == 504
  assert len(tracked) == 1


def test_a_tokenization_outliving_its_request_triggers_the_exit_guard(env):
  env.setenv("KREA2_REQUEST_TIMEOUT_S", "0.3")
  env.setenv("KREA2_SHUTDOWN_GRACE_S", "0.2")
  runtime = FakeRuntime()
  runtime.bucket_gate.clear()  # the tokenizer never returns while the request and the shutdown run
  app, _, _ = _make_app(runtime)

  async def body():
    async with _client(app) as client:
      return await asyncio.wait_for(_generate(client, response_format="png"), GATE_TIMEOUT_S)

  exits = []
  try:
    # The lifespan has exited (pools shut down without waiting) and the loop is closed, as in main().
    timed_out = asyncio.run(body())
    state = app.state.server
    pending = [f for f in state.pool_futures if not f.done()]
    started = time.monotonic()
    serve_krea2._exit_after_pool_work(state, 5, exits.append)  # pylint: disable=protected-access
    elapsed = time.monotonic() - started
  finally:
    runtime.bucket_gate.set()
  assert timed_out.status_code == 504
  # Only the tokenization is pending (no generation was submitted), and the guard still waits for it, then exits.
  assert len(pending) == 1
  assert runtime.calls == []
  assert exits == [5]
  assert 0.15 <= elapsed < 2.0


def test_main_bounds_the_drain_with_the_shutdown_grace(env, monkeypatch):
  import uvicorn  # pylint: disable=import-outside-toplevel

  configs = []

  class FakeConfig:

    def __init__(self, app, **kwargs):
      self.app = app
      self.kwargs = kwargs
      configs.append(self)

  class FakeServer:

    def __init__(self, config):
      self.config = config
      self.started = True

    def run(self):
      pass

  monkeypatch.setattr(uvicorn, "Config", FakeConfig)
  monkeypatch.setattr(uvicorn, "Server", FakeServer)
  monkeypatch.setattr(serve_krea2, "app", serve_krea2.create_app(lambda: FakeRuntime()))
  env.setenv("KREA2_SHUTDOWN_GRACE_S", "3.5")
  serve_krea2.main()
  assert len(configs) == 1
  assert configs[0].kwargs["timeout_graceful_shutdown"] == 3.5
  assert configs[0].kwargs["lifespan"] == "on"

  # Invalid settings stop main() with uvicorn's startup-failure code before any server is built.
  env.setenv("KREA2_SHUTDOWN_GRACE_S", "0")
  with pytest.raises(SystemExit) as exc_info:
    serve_krea2.main()
  assert exc_info.value.code == 3
  assert len(configs) == 1


def test_avif_encoder_round_trip():
  serve_krea2.ensure_avif_encoder()
  options = {"quality": 85, "speed": 8, "subsampling": "4:4:4", "max_threads": 2}
  encoded = serve_krea2.encode_image_base64(Image.new("RGB", (32, 16), "red"), "avif", options)
  decoded = Image.open(BytesIO(base64.b64decode(encoded)))
  assert decoded.format == "AVIF"
  assert decoded.size == (32, 16)
