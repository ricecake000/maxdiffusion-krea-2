"""Opt-in cold-start shortcuts of the Krea 2 runtime and server (`KREA2_STARTUP_OPTS`).

Nothing here changes a compiled program or an AOT file name: the options only
change how the process gets to the same loaded state. Unset or empty, the
runtime starts exactly as it did before this module existed.

`KREA2_STARTUP_OPTS` is a comma-separated list of:
  np_vae             the VAE weights are converted with numpy (no XLA compiles, no jax.clear_caches()).
  lazy_load_shapes   the load-structure `jax.eval_shape` of a component runs only when the component is built
                     from its checkpoint (weight-cache miss or build-only mode).
  shape_cache        the two runtime abstract param trees are read from `<aot_cache_dir>/krea2-shapes-<key>.pkl`
                     (`shape_cache_key`: every *.py of this package, the library versions, the model configs and
                     the whole pyconfig); implies lazy_load_shapes. No aot_cache_dir or a LoRA: no cache.
  aot_parallel_load  lazy AOT loads lock per signature, so warm-up workers deserialize at once.
  defer_warmup       (serving) ready after the first preset's warm-up; the other executables load on background
                     threads, see `deferred_executables`. Implies aot_parallel_load. What it gives up: only the
                     first preset's resolution runs through the pipeline before ready, so the first request of
                     every other resolution also compiles the small programs jax builds outside the AOT cache
                     for a new shape (image ids, scheduler and latent ops; never a transformer, text encoder or
                     VAE program: those are the manifest's executables). A full warm-up pays that before ready.
  all                every option above.

This module imports no jax at import time (the server module imports it before the runtime loads).
"""

import hashlib
import json
import os
import pickle
import re
import sys
import threading
import time
from typing import List, Optional, Tuple

from maxdiffusion import max_logging

STARTUP_OPTS = ("np_vae", "lazy_load_shapes", "shape_cache", "aot_parallel_load", "defer_warmup")
_IMPLIED = {"shape_cache": ("lazy_load_shapes",), "defer_warmup": ("aot_parallel_load",)}
_SHAPE_CACHE_FORMAT = 1
_MANIFEST_FORMAT = 1


def resolve_startup_opts(value) -> frozenset:
  """The option set of a `KREA2_STARTUP_OPTS` value (None or '' = none); ValueError on an unknown name."""
  names = [name.strip() for name in str(value or "").split(",") if name.strip()]
  unknown = sorted(set(names) - set(STARTUP_OPTS) - {"all"})
  if unknown:
    raise ValueError(
        f"KREA2_STARTUP_OPTS: unknown option(s) {', '.join(unknown)}; known: {', '.join(STARTUP_OPTS)}, all"
    )
  opts = set(STARTUP_OPTS) if "all" in names else set(names)
  for name in list(opts):
    opts.update(_IMPLIED.get(name, ()))
  return frozenset(opts)


def describe_startup_opts(opts) -> str:
  return "[startup] options: " + (", ".join(name for name in STARTUP_OPTS if name in opts) or "none")


# --------------------------------------------------------------------------- shape cache
def _package_source_digest() -> str:
  """sha256 over the relative path and content of every *.py of the maxdiffusion package (tests excluded)."""
  package = os.path.dirname(os.path.abspath(__file__))
  digest = hashlib.sha256()
  for root, dirs, files in os.walk(package):
    dirs[:] = sorted(d for d in dirs if d not in ("tests", "__pycache__"))
    for name in sorted(files):
      if name.endswith(".py"):
        path = os.path.join(root, name)
        with open(path, "rb") as f:
          digest.update(os.path.relpath(path, package).encode() + b"\0" + f.read() + b"\0")
  return digest.hexdigest()


def _library_versions() -> dict:
  import importlib.metadata  # pylint: disable=import-outside-toplevel

  versions = {"python": list(sys.version_info[:3])}
  for name in ("jax", "flax", "qwix", "numpy", "ml_dtypes"):
    try:
      versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      versions[name] = None
  return versions


def shape_cache_key(config_keys: dict, model_configs: dict) -> str:
  """The key of the abstract param trees: everything `jax.eval_shape(model.init)` can depend on.

  The package source (a changed module may change a param tree), the library
  versions (pickled jax / flax objects), jax's x64 switch, the two model
  `config.json` dicts and every pyconfig key (address-stripped reprs, sorted).
  Deliberately too broad: a
  key that changes without need only costs one recomputation.
  """
  import jax  # pylint: disable=import-outside-toplevel

  parts = {
      "format": _SHAPE_CACHE_FORMAT,
      # x64 decides how jax canonicalizes the dtypes of the abstract leaves.
      "jax_x64": bool(jax.config.jax_enable_x64),
      "source": _package_source_digest(),
      "versions": _library_versions(),
      "models": model_configs,
      "config": sorted((str(key), re.sub(r"0x[0-9a-fA-F]+", "@", repr(value))) for key, value in config_keys.items()),
  }
  return hashlib.sha256(json.dumps(parts, sort_keys=True, default=repr).encode()).hexdigest()[:16]


def shape_cache_path(cache_dir: str, key: str) -> str:
  return os.path.join(cache_dir, f"krea2-shapes-{key}.pkl")


def load_shape_cache(path: str, names: Tuple[str, ...]) -> Optional[tuple]:
  """The cached abstract trees for `names` (in that order), or None on a miss. Never raises for a bad file."""
  try:
    with open(path, "rb") as f:
      blob = pickle.load(f)
    if blob["format"] != _SHAPE_CACHE_FORMAT or tuple(blob["names"]) != tuple(names):
      raise ValueError("other format or contents")
    trees = tuple(blob["trees"])
    if len(trees) != len(names):
      raise ValueError("tree count")
  except FileNotFoundError:
    return None
  except Exception as exc:  # pylint: disable=broad-exception-caught
    max_logging.log(f"[startup] shape cache unreadable ({type(exc).__name__}: {exc}): {path}; computing the shapes")
    return None
  max_logging.log(f"[startup] shape cache hit {path}")
  return trees


def save_shape_cache(path: str, names: Tuple[str, ...], trees: tuple) -> bool:
  """Writes the trees atomically; a failure is logged and returns False (the start goes on without the cache)."""
  tmp_path = f"{path}.tmp.{os.getpid()}"
  try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, "wb") as f:
      pickle.dump({"format": _SHAPE_CACHE_FORMAT, "names": list(names), "trees": list(trees)}, f)
    os.replace(tmp_path, path)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    max_logging.log(f"[startup] shape cache not saved ({type(exc).__name__}: {exc}): {path}")
    try:
      os.remove(tmp_path)
    except OSError:
      pass
    return False
  max_logging.log(f"[startup] shape cache saved {path}")
  return True


# --------------------------------------------------------------------------- deferred warm-up
# The manifest this process trusted or wrote: dropped when it turns out wrong (drop_serving_manifest).
_MANIFEST_PATH: List[str] = []


def _plan_rows(plan) -> list:
  return [[r.label, int(r.height), int(r.width), int(bucket)] for r, bucket in plan]


def serving_manifest_path(plan) -> str:
  """The manifest file of `plan` under the installed AOT cache ('' while the cache is off)."""
  from maxdiffusion import aot_cache  # pylint: disable=import-outside-toplevel

  cache_dir, fingerprint, _ = aot_cache.install_state()
  if not cache_dir:
    return ""
  plan_key = hashlib.sha256(json.dumps(_plan_rows(plan)).encode()).hexdigest()[:12]
  return os.path.join(cache_dir, f"krea2-serving-manifest-{fingerprint}-{plan_key}.json")


def deferred_executables(plan) -> Optional[List[Tuple[str, str]]]:
  """The executables a deferred warm-up of `plan` may leave to background loading, or None (warm everything now).

  They are the `(wrapped fn name, signature)` pairs of the manifest that a
  full warm-up of exactly this plan wrote under the installed AOT cache
  (`write_serving_manifest`), and only when every one of their files is
  there: then nothing the plan needs has to be compiled. None, with one
  `[deferred] off: <reason>` line, when the AOT cache is off or not lazy,
  there is no manifest for this plan and fingerprint, it is unreadable or
  lists a missing file.
  """
  from maxdiffusion import aot_cache  # pylint: disable=import-outside-toplevel

  def off(reason):  # returns None
    max_logging.log(f"[deferred] off: {reason}")

  cache_dir, _, lazy_load = aot_cache.install_state()
  if not cache_dir:
    return off("the AOT cache is off (no aot_cache_dir)")
  if not lazy_load:
    return off("aot_cache_lazy_load is off (every executable is loaded at install)")
  path = serving_manifest_path(plan)
  try:
    with open(path, encoding="utf-8") as f:
      manifest = json.load(f)
    if manifest["format"] != _MANIFEST_FORMAT or manifest["plan"] != _plan_rows(plan):
      raise ValueError("other format or plan")
    pairs = [(str(name), str(signature)) for name, signature in manifest["executables"]]
  except FileNotFoundError:
    return off(f"no manifest {path} yet; this start's full warm-up writes it")
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return off(f"unreadable manifest {path} ({type(exc).__name__}: {exc})")
  missing = [pair for pair in pairs if not os.path.isfile(aot_cache.signature_path(*pair))]
  if missing or not pairs:
    return off(f"{len(missing)} of the manifest's {len(pairs)} executables have no file in {cache_dir}")
  _MANIFEST_PATH[:] = [path]
  return pairs


def warmup_compiled(records) -> bool:
  """Whether a warm-up that returned `records` (run_precompile) compiled anything.

  A saved executable, or one that this process compiled and could not save
  (`aot_cache.on_disk_signatures`' unsaved count: a full or read-only cache
  dir reports `saved` 0 although it compiled).
  """
  from maxdiffusion import aot_cache  # pylint: disable=import-outside-toplevel

  return any(record.get("saved") for record in records or ()) or aot_cache.on_disk_signatures()[1] > 0


def write_serving_manifest(plan) -> Optional[str]:
  """After a FULL warm-up of `plan`: records every executable this process loaded or saved. Returns the path.

  Not written (None, one log line) while the cache is off or when a compiled
  executable has no file (a failed save): a later deferred start would miss it.
  """
  from maxdiffusion import aot_cache  # pylint: disable=import-outside-toplevel

  path = serving_manifest_path(plan)
  if not path:
    return None
  pairs, unsaved = aot_cache.on_disk_signatures()
  if unsaved or not pairs:
    max_logging.log(
        f"[deferred] no manifest written: {unsaved} compiled executable(s) have no file, {len(pairs)} have one"
    )
    return None
  tmp_path = f"{path}.tmp.{os.getpid()}"
  try:
    with open(tmp_path, "w", encoding="utf-8") as f:
      json.dump({"format": _MANIFEST_FORMAT, "plan": _plan_rows(plan), "executables": [list(p) for p in pairs]}, f)
    os.replace(tmp_path, path)
  except OSError as exc:
    max_logging.log(f"[deferred] no manifest written ({exc!r}): {path}")
    return None
  _MANIFEST_PATH[:] = [path]
  max_logging.log(f"[deferred] manifest written {path} ({len(pairs)} executables)")
  return path


def drop_serving_manifest(reason: str) -> None:
  """Removes the manifest this process trusted or wrote, so the next start warms (and compiles) everything."""
  while _MANIFEST_PATH:
    path = _MANIFEST_PATH.pop()
    try:
      os.remove(path)
      max_logging.log(f"[deferred] manifest removed ({reason}): {path}")
    except OSError as exc:
      max_logging.log(f"[deferred] manifest not removed ({exc!r}; {reason}): {path}")


def start_deferred_load(pairs, timeline=None, workers: int = 1) -> threading.Thread:
  """Makes `pairs` the AOT cache's strict set and loads them on a background thread (`workers` loader threads).

  Strict first: from here on a call of one of these executables that cannot be
  loaded raises `aot_cache.AotCacheMiss` instead of compiling inside a request
  (anything outside the manifest compiles as it always did). Returns the thread. When
  the loads end, one `[deferred] <n> executable(s) loaded, <f> failed in <s> s;
  t=<seconds since the timeline's start>` line is logged (and the timeline gets
  the mark `deferred_done`); a failed load removes the manifest.
  """
  from maxdiffusion import aot_cache  # pylint: disable=import-outside-toplevel

  aot_cache.set_strict(pairs)

  def run():
    started = time.perf_counter()
    try:
      loaded, failed = aot_cache.load_signatures(pairs, workers=workers)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      loaded, failed = 0, list(pairs)
      max_logging.log(f"[deferred] background load stopped: {type(exc).__name__}: {exc}")
    if failed:
      drop_serving_manifest(f"{len(failed)} executable(s) did not load, e.g. {failed[0][0]} {failed[0][1]}")
    since_start = ""
    if timeline is not None:
      timeline.mark("deferred_done", parallel=True)
      since_start = f"; t={timeline.marks()['deferred_done']:.2f}"
    max_logging.log(
        f"[deferred] {loaded} executable(s) loaded, {len(failed)} failed in "
        f"{time.perf_counter() - started:.1f} s{since_start}"
    )

  thread = threading.Thread(target=run, name="krea2-deferred-load", daemon=True)
  thread.start()
  return thread
