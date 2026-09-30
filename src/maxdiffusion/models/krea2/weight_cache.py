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

# Host cache of the final (quantized) Krea 2 weight trees (krea2_weight_cache_dir).
#
# A start with a cache hit reads the trees generate_krea2 would otherwise build
# from the bf16 checkpoint (read, rotate-half permutation, int8 quantization)
# and never opens the checkpoint's safetensors files. One directory per
# component and fingerprint:
#
#   <cache_dir>/<component>-<fp>/meta.json     layout, fingerprint inputs, leaf index
#   <cache_dir>/<component>-<fp>/weights.bin   raw little-endian leaf bytes, 64-byte aligned
#
# A directory is complete iff meta.json exists: it is written into a temporary
# directory (weights.bin first, meta.json last) that is renamed into place.
# There is no content checksum: hashing ~16 GiB would cost what the cache saves.
# This module must not import `generate_krea2` or the pipeline.

import collections
import hashlib
import importlib.metadata
import json
import math
import os
import secrets
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from maxdiffusion import max_logging
from .text_encoder_quant import KREA2_TEXT_ENCODER_WEIGHT_QUANT_REVISION
from .transformer_quant import KREA2_TRANSFORMER_WEIGHT_QUANT_REVISION
from .util import KREA2_ROPE_PERMUTATION_REVISION

# Bump when the on-disk layout changes; directories of another format then miss.
KREA2_WEIGHT_CACHE_FORMAT = 1
# Bump when anything else that produces the stored transformer values changes: the checkpoint
# conversion (load_and_convert_krea2_weights). The quantization and the rotate-half permutation have
# their own revisions (KREA2_TRANSFORMER_WEIGHT_QUANT_REVISION, KREA2_ROPE_PERMUTATION_REVISION).
KREA2_TRANSFORMER_WEIGHT_BUILD_REVISION = 1
# Bump when anything else that produces the stored text encoder values changes: the checkpoint
# conversion (load_and_convert_qwen3_weights, the bf16 normalization in generate_krea2,
# load_qwen3_embedding_table). The quantization has KREA2_TEXT_ENCODER_WEIGHT_QUANT_REVISION.
KREA2_TEXT_ENCODER_WEIGHT_BUILD_REVISION = 1

_META_FILE = "meta.json"
_WEIGHTS_FILE = "weights.bin"
_ALIGNMENT = 64
# Free space kept beyond the bytes to write, so the cache never fills the disk.
_FREE_SPACE_MARGIN = 64 * 1024**2
_GIB = 1024**3
# list_source_files' entry for a checkpoint directory that exists but cannot be listed.
_UNREADABLE_SOURCE = "<unreadable>"


class WeightCacheSpec(NamedTuple):
  """One component's cache settings as generate_krea2 resolves them."""

  cache_dir: str
  component: str
  meta: dict
  source_files: Optional[list]


class _ShortRead(Exception):
  pass


# Resource problems a cache read turns into a miss: I/O, memory, and a thread
# that cannot start or a broken pool (RuntimeError).
_RESOURCE_ERRORS = (OSError, MemoryError, RuntimeError)


def weight_cache_fingerprint(meta: dict) -> str:
  return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def component_dir(cache_dir: str, component: str, meta: dict) -> str:
  return os.path.join(cache_dir, f"{component}-{weight_cache_fingerprint(meta)}")


def list_source_files(source_dir: str) -> Optional[list]:
  """Sorted `[[basename, size], ...]` of the `*.safetensors` files in `source_dir`.

  Returns None only for a confirmed absence (`source_dir` missing, not a
  directory, or without `*.safetensors` names): the cache is then trusted
  without the checkpoint. A failed scan is not an absence: an unlistable
  directory gives `[["<unreadable>", None]]` and a file whose size cannot be
  read (e.g. a dangling Hugging Face snapshot symlink) is kept with size None,
  so such a list never matches a stored one and `load_component` misses.
  Only lists and stats the files (never opens them), so a changed checkpoint
  is detected without reading it.
  """
  try:
    names = sorted(name for name in os.listdir(source_dir) if name.endswith(".safetensors"))
  except (FileNotFoundError, NotADirectoryError):
    return None
  except OSError:
    return [[_UNREADABLE_SOURCE, None]]
  files = []
  for name in names:
    try:
      size = os.path.getsize(os.path.join(source_dir, name))
    except OSError:
      size = None
    files.append([name, size])
  return files or None


def _source_scan_complete(source_files) -> bool:
  # False when list_source_files could not read the directory or a file's size.
  return all(size is not None for _, size in source_files)


def _snapshot_name(snapshot_dir: str) -> str:
  # The commit hash for a Hugging Face snapshot directory.
  return os.path.basename(os.path.normpath(snapshot_dir))


def transformer_weight_cache_meta(
    config,
    snapshot_dir,
    quantization,
    quant_targets,
    rope_layout,
    num_attention_heads,
    num_key_value_heads,
    attention_head_dim,
) -> dict:
  """Fingerprint inputs of the cached transformer tree (after the permutation and the W8A8 quantization).

  The head geometry is part of it because the rotate-half permutation depends
  on it, while another geometry can give the same flattened projection shapes.
  """
  return {
      "model": str(config.pretrained_model_name_or_path),
      "snapshot": _snapshot_name(snapshot_dir),
      "weights_dtype": str(jnp.dtype(config.weights_dtype)),
      "quantization": quantization,
      "quant_targets": list(quant_targets),
      "weight_quant_revision": KREA2_TRANSFORMER_WEIGHT_QUANT_REVISION,
      "rope_layout": rope_layout,
      "rope_permutation_revision": KREA2_ROPE_PERMUTATION_REVISION,
      "num_attention_heads": int(num_attention_heads),
      "num_key_value_heads": int(num_key_value_heads),
      "attention_head_dim": int(attention_head_dim),
      "build_revision": KREA2_TRANSFORMER_WEIGHT_BUILD_REVISION,
  }


def text_encoder_weight_cache_meta(config, snapshot_dir, quantization, tile_size, embed_on_host, scale_dtype) -> dict:
  """Fingerprint inputs of the cached text encoder tree (after the qwix int8 quantization)."""
  return {
      "model": str(config.pretrained_model_name_or_path),
      "snapshot": _snapshot_name(snapshot_dir),
      "weights_dtype": str(jnp.dtype(config.weights_dtype)),
      "quantization": quantization,
      "tile_size": int(tile_size),
      "embed_on_host": bool(embed_on_host),
      "scale_dtype": str(jnp.dtype(scale_dtype)),
      # qwix's arithmetic produces the stored values.
      "qwix": importlib.metadata.version("qwix"),
      "weight_quant_revision": KREA2_TEXT_ENCODER_WEIGHT_QUANT_REVISION,
      "build_revision": KREA2_TEXT_ENCODER_WEIGHT_BUILD_REVISION,
  }


def _as_native_array(leaf) -> np.ndarray:
  array = np.asarray(leaf)
  if not array.dtype.isnative:
    array = array.astype(array.dtype.newbyteorder("="))
  return array


class _LeafSpec(NamedTuple):
  """Dtype and shape of a stored leaf, to recompute its layout without allocating it."""

  dtype: np.dtype
  shape: tuple


def _layout(named_arrays, start=0):
  """Index entries for `(name, array)` pairs placed from `start`; returns `(entries, end)`.

  Only reads `dtype` and `shape` of the arrays, so the reader recomputes the
  canonical layout of a stored index from `_LeafSpec`s.
  """
  entries = []
  position = start
  for name, array in named_arrays:
    offset = -(-position // _ALIGNMENT) * _ALIGNMENT
    nbytes = math.prod(int(dim) for dim in array.shape) * array.dtype.itemsize
    entries.append(
        {
            "path": name,
            "dtype": array.dtype.name,
            "shape": list(array.shape),
            "offset": offset,
            "nbytes": nbytes,
        }
    )
    position = offset + nbytes
  return entries, position


def _write_arrays(file, entries, arrays) -> None:
  position = 0
  for entry, array in zip(entries, arrays):
    if entry["offset"] > position:
      file.write(b"\0" * (entry["offset"] - position))
    # Through a uint8 view: bfloat16 has no buffer-protocol format. Handles
    # non-contiguous views (the loader's `.T` kernels), 0-d and empty leaves.
    file.write(np.ascontiguousarray(array).reshape(-1).view(np.uint8))
    position = entry["offset"] + entry["nbytes"]


def _layout_signature(entries):
  return [(entry["path"], entry["dtype"], list(entry["shape"])) for entry in entries]


def _normalized(value):
  # What `value` becomes after a JSON round trip (tuples -> lists).
  return json.loads(json.dumps(value))


class _Malformed(Exception):
  """A `meta.json` that parses as JSON but is not a well-formed header."""


def _is_count(value) -> bool:
  # A non-negative JSON integer; bools (an int subclass in Python) and floats are rejected.
  return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _stored_leaves(entries, what):
  """`[(path, _LeafSpec), ...]` of the stored `index` or `extras` list.

  Raises `_Malformed` for a wrong type, a negative number, an unknown dtype
  name or a duplicate path.
  """
  if not isinstance(entries, list):
    raise _Malformed(f"{what} is not a list")
  leaves = []
  for entry in entries:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
      raise _Malformed(f"{what} entry without a string path")
    path = entry["path"]
    shape = entry.get("shape")
    if not isinstance(shape, list) or not all(_is_count(dim) for dim in shape):
      raise _Malformed(f"{path} has shape {shape!r}")
    for key in ("offset", "nbytes"):
      if not _is_count(entry.get(key)):
        raise _Malformed(f"{path} has {key} {entry.get(key)!r}")
    name = entry.get("dtype")
    try:
      dtype = jnp.dtype(name) if isinstance(name, str) else None
    except (TypeError, ValueError):
      dtype = None
    # Object arrays hold pointers, not values.
    if dtype is None or dtype.hasobject:
      raise _Malformed(f"{path} has unknown dtype {name!r}")
    leaves.append((path, _LeafSpec(dtype, tuple(shape))))
  duplicates = [path for path, count in collections.Counter(path for path, _ in leaves).items() if count > 1]
  if duplicates:
    raise _Malformed(f"duplicate path {duplicates[0]} in {what}")
  return leaves


def _read_header(directory, component, meta, source_files):
  """Returns `(header, None)` for a readable, self-consistent cache directory, else `(None, reason)`.

  `source_files` None skips the checkpoint comparison (no checkpoint on this
  machine); a list from a failed scan (a None size) never matches.
  """
  meta_path = os.path.join(directory, _META_FILE)
  if not os.path.isdir(directory):
    return None, "no cache directory"
  if not os.path.isfile(meta_path):
    return None, f"incomplete directory (no {_META_FILE})"
  try:
    with open(meta_path, "r", encoding="utf-8") as f:
      header = json.load(f)
  except (OSError, ValueError, MemoryError) as exc:
    return None, f"unreadable {_META_FILE} ({exc!r})"
  if not isinstance(header, dict):
    return None, f"malformed {_META_FILE} (not a JSON object)"
  if header.get("format") != KREA2_WEIGHT_CACHE_FORMAT:
    return None, f"format {header.get('format')!r}, expected {KREA2_WEIGHT_CACHE_FORMAT}"
  if header.get("component") != component:
    return None, f"component {header.get('component')!r}, expected {component!r}"
  if header.get("byteorder") != sys.byteorder:
    return None, f"byte order {header.get('byteorder')!r}, this host is {sys.byteorder!r}"
  if header.get("meta") != _normalized(meta):
    # Compared in full, not only through the directory name's fingerprint.
    return None, "stored fingerprint inputs differ"
  if source_files is not None:
    if not _source_scan_complete(source_files):
      # Even against a stored list from an equally failed scan: equal lists prove nothing then.
      return None, "source checkpoint changed (could not list or stat all of its files)"
    if header.get("source_files") != _normalized(source_files):
      return None, "source checkpoint changed"
  try:
    index = _stored_leaves(header.get("index"), "index")
    extras = _stored_leaves(header.get("extras", []), "extras")
    total_bytes = header.get("total_bytes")
    if not _is_count(total_bytes):
      raise _Malformed(f"total_bytes {total_bytes!r}")
  except _Malformed as exc:
    return None, f"malformed {_META_FILE} ({exc})"
  weights_path = os.path.join(directory, _WEIGHTS_FILE)
  if not os.path.isfile(weights_path):
    return None, f"no {_WEIGHTS_FILE}"
  try:
    size = os.path.getsize(weights_path)
  except OSError as exc:
    return None, f"unreadable {_WEIGHTS_FILE} ({exc})"
  if size != total_bytes:
    return None, f"{_WEIGHTS_FILE} has {size} bytes, expected {total_bytes}"
  entries = header["index"] + header.get("extras", [])
  for entry in entries:
    if entry["offset"] + entry["nbytes"] > total_bytes:
      return None, f"index entry {entry['path']} lies outside {_WEIGHTS_FILE}"
  # The stored layout must be the one save_component writes for these paths,
  # dtypes and shapes: overlaps, gaps and out-of-order offsets are a miss.
  canonical_index, end = _layout(index)
  canonical_extras, canonical_total = _layout(extras, start=end)
  for entry, canonical in zip(entries, canonical_index + canonical_extras):
    if (entry["offset"], entry["nbytes"]) != (canonical["offset"], canonical["nbytes"]):
      return None, (
          f"index entry {entry['path']} has offset {entry['offset']} and {entry['nbytes']} bytes, "
          f"the canonical layout has offset {canonical['offset']} and {canonical['nbytes']} bytes"
      )
  if total_bytes != canonical_total:
    return None, f"total_bytes {total_bytes}, the canonical layout ends at {canonical_total}"
  return header, None


def _flatten_named(tree):
  leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
  return [(jax.tree_util.keystr(path), leaf) for path, leaf in leaves]


def _remove_path(path) -> None:
  if os.path.isdir(path) and not os.path.islink(path):
    shutil.rmtree(path, ignore_errors=True)
  elif os.path.lexists(path):
    os.remove(path)


def _fsync_directory(path) -> None:
  """Best effort: makes a rename in `path` durable."""
  try:
    fd = os.open(path, os.O_RDONLY)
  except OSError:
    return
  try:
    os.fsync(fd)
  except OSError:
    pass
  finally:
    os.close(fd)


def _weights_open_error(directory) -> Optional[str]:
  """None when the `weights.bin` of `directory` can be opened for reading, else the reason."""
  try:
    os.close(os.open(os.path.join(directory, _WEIGHTS_FILE), os.O_RDONLY))
  except OSError as exc:
    return f"unreadable {_WEIGHTS_FILE} ({exc!r})"
  return None


def weights_nbytes(directory) -> Optional[int]:
  """Size of the `weights.bin` of a component directory; None when it cannot be read."""
  try:
    return os.path.getsize(os.path.join(directory, _WEIGHTS_FILE))
  except OSError:
    return None


def save_component(cache_dir, component, meta, tree, extras=None, source_files=None) -> Optional[str]:
  """Writes `tree` (and the named `extras` arrays) as `<cache_dir>/<component>-<fp>`.

  Returns the component directory, or None when nothing was written. No
  OSError or MemoryError escapes, not even from the fingerprint (a warning is
  logged and the temporary directory is removed); skips the write when the
  file system lacks the space for it. An existing directory with the same
  fingerprint is kept if it is valid, holds the same leaves and its weights
  file can be opened for reading, and replaced otherwise.
  """
  start = time.perf_counter()
  final_dir = tmp_dir = None
  try:
    final_dir = component_dir(cache_dir, component, meta)
    named = [(name, _as_native_array(leaf)) for name, leaf in _flatten_named(tree)]
    named_extras = [(name, _as_native_array(array)) for name, array in (extras or {}).items()]
    index, end = _layout(named)
    extras_index, total_bytes = _layout(named_extras, start=end)
    signature = (_layout_signature(index), _layout_signature(extras_index))

    def existing_is_valid():
      header, _ = _read_header(final_dir, component, meta, source_files)
      if header is None:
        return False
      if (_layout_signature(header["index"]), _layout_signature(header.get("extras") or [])) != signature:
        return False
      # A weights file that cannot be opened misses on every load: replace it.
      return _weights_open_error(final_dir) is None

    os.makedirs(cache_dir, exist_ok=True)
    if existing_is_valid():
      max_logging.log(f"[weight cache] {component}: {final_dir} is already complete; kept it.")
      return final_dir
    free = shutil.disk_usage(cache_dir).free
    if free < total_bytes + _FREE_SPACE_MARGIN:
      max_logging.log(
          f"[weight cache] {component}: not saved, {free / _GIB:.2f} GiB free in {cache_dir} but "
          f"{total_bytes / _GIB:.2f} GiB (+{_FREE_SPACE_MARGIN / _GIB:.2f} GiB margin) needed."
      )
      return None

    tmp_dir = f"{final_dir}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    os.makedirs(tmp_dir)
    with open(os.path.join(tmp_dir, _WEIGHTS_FILE), "wb") as f:
      _write_arrays(f, index + extras_index, [array for _, array in named + named_extras])
      f.flush()
      os.fsync(f.fileno())
    header = {
        "format": KREA2_WEIGHT_CACHE_FORMAT,
        "component": component,
        "meta": meta,
        "byteorder": sys.byteorder,
        "source_files": source_files,
        "total_bytes": total_bytes,
        "index": index,
        "extras": extras_index,
    }
    # meta.json last: its presence marks the directory complete.
    with open(os.path.join(tmp_dir, _META_FILE), "w", encoding="utf-8") as f:
      json.dump(header, f)
      f.flush()
      os.fsync(f.fileno())

    if os.path.lexists(final_dir):
      if existing_is_valid():  # another process finished first
        shutil.rmtree(tmp_dir, ignore_errors=True)
        max_logging.log(f"[weight cache] {component}: {final_dir} was completed meanwhile; kept it.")
        return final_dir
      # Move the invalid directory aside first, so the final name never holds a partial tree.
      aside = f"{final_dir}.old-{os.getpid()}-{secrets.token_hex(4)}"
      os.rename(final_dir, aside)
      _remove_path(aside)
    try:
      os.rename(tmp_dir, final_dir)
    except OSError:
      if not existing_is_valid():
        raise
      shutil.rmtree(tmp_dir, ignore_errors=True)
      max_logging.log(f"[weight cache] {component}: {final_dir} was completed meanwhile; kept it.")
      return final_dir
    tmp_dir = None
    _fsync_directory(cache_dir)
  except (OSError, MemoryError) as exc:
    # A failed save never fails the generation: the tree in memory is complete.
    max_logging.log(f"[weight cache] {component}: warning, could not save to {final_dir or cache_dir}: {exc!r}")
    if tmp_dir is not None:
      shutil.rmtree(tmp_dir, ignore_errors=True)
    return None
  max_logging.log(
      f"[weight cache] {component}: saved {total_bytes / _GIB:.2f} GiB to {final_dir} "
      f"in {time.perf_counter() - start:.1f} s"
  )
  return final_dir


def _dtype_accepted(stored, expected, exact) -> bool:
  """Whether a stored leaf dtype fits the abstract leaf's (see `load_component`'s `check_dtypes`)."""
  if exact or not jnp.issubdtype(expected, jnp.floating):
    return stored == expected
  # A floating leaf of any width: the stored qwix scales have the compute dtype.
  return bool(jnp.issubdtype(stored, jnp.floating))


def _read_into(fd, array, offset) -> None:
  """Fills `array` from `offset` of `fd` with positional reads (thread-safe, releases the GIL)."""
  buffer = memoryview(array.reshape(-1).view(np.uint8))
  done = 0
  while done < len(buffer):
    if hasattr(os, "preadv"):
      count = os.preadv(fd, [buffer[done:]], offset + done)
    else:
      chunk = os.pread(fd, len(buffer) - done, offset + done)
      count = len(chunk)
      buffer[done : done + count] = chunk
    if count == 0:
      raise _ShortRead(f"{len(buffer) - done} bytes missing at offset {offset + done}")
    done += count


def _match_leaves(header, expected, extra_names, check_dtypes):
  """Stored entries of the `expected` `(path, abstract leaf)` pairs, then of the extras.

  Returns `(entries, None)`, or `(None, reason)` when the paths, shapes,
  dtypes (see `load_component`'s `check_dtypes`) or extras do not match.
  """
  stored = {entry["path"]: entry for entry in header["index"]}
  missing = [name for name, _ in expected if name not in stored]
  unexpected = sorted(set(stored) - {name for name, _ in expected})
  if missing or unexpected:
    problems = []
    if missing:
      problems.append(f"{len(missing)} model path(s) not cached (e.g. {missing[0]})")
    if unexpected:
      problems.append(f"{len(unexpected)} cached path(s) not in the model (e.g. {unexpected[0]})")
    return None, ", ".join(problems)
  for name, leaf in expected:
    entry = stored[name]
    if tuple(entry["shape"]) != tuple(leaf.shape):
      return None, f"{name} has shape {tuple(entry['shape'])}, the model expects {tuple(leaf.shape)}"
    if not _dtype_accepted(jnp.dtype(entry["dtype"]), jnp.dtype(leaf.dtype), check_dtypes):
      return None, f"{name} has dtype {entry['dtype']}, the model expects {jnp.dtype(leaf.dtype)}"
  stored_extras = {entry["path"]: entry for entry in header.get("extras") or []}
  for name in extra_names:
    if name not in stored_extras:
      return None, f"no extra array {name!r}"
  return [stored[name] for name, _ in expected] + [stored_extras[name] for name in extra_names], None


def _read_arrays(weights_path, entries, arrays, num_workers) -> None:
  """Fills `arrays` from their entries' offsets in `weights_path` with a thread pool.

  The file descriptor is closed on every path, also when the pool cannot start.
  """
  fd = os.open(weights_path, os.O_RDONLY)
  try:
    # File reads release the GIL, so threads read in parallel.
    with ThreadPoolExecutor(max_workers=num_workers or min(8, os.cpu_count() or 1)) as executor:
      list(executor.map(lambda item: _read_into(fd, item[1], item[0]["offset"]), zip(entries, arrays)))
  finally:
    os.close(fd)


def _matching_entries(cache_dir, component, meta, abstract_tree, extra_names, source_files, check_dtypes):
  """The checks of `load_component` short of allocating and reading the arrays.

  Returns `(directory, entries, treedef, reason)`: `entries` are the stored
  index entries of the abstract tree's leaves (in flattening order) followed by
  those of the extras, or None with the miss `reason`; `directory` is None when
  even the fingerprint failed. Resource problems are a miss, errors in
  `abstract_tree` raise.
  """
  directory = None
  try:
    directory = component_dir(cache_dir, component, meta)
    header, reason = _read_header(directory, component, meta, source_files)
  except _RESOURCE_ERRORS as exc:
    header, reason = None, f"header read failed ({exc!r})"
  if header is None:
    return directory, None, None, reason

  # Not protected: an error in the caller's abstract tree is a programming error.
  abstract = flax.core.unfreeze(nn.unbox(abstract_tree))
  leaves, treedef = jax.tree_util.tree_flatten_with_path(abstract)
  expected = [(jax.tree_util.keystr(path), leaf) for path, leaf in leaves]

  try:
    entries, reason = _match_leaves(header, expected, extra_names, check_dtypes)
  except _RESOURCE_ERRORS as exc:
    entries, reason = None, f"index check failed ({exc!r})"
  return directory, entries, treedef, reason


def _log_miss(component, location, reason) -> None:
  max_logging.log(f"[weight cache] {component}: miss at {location}: {reason}.")


def component_is_valid(
    cache_dir, component, meta, abstract_tree, extra_names=(), source_files=None, check_dtypes=True
) -> bool:
  """Whether `load_component` with these arguments would hit, without reading the arrays.

  Runs every check of `load_component` short of allocating and reading the
  arrays (header, canonical layout, paths, shapes, dtypes, extras) and checks
  that `weights.bin` can be opened for reading. Logs one line (`already
  complete` or the miss reason) and never raises for a bad or missing cache;
  errors in `abstract_tree` raise.
  """
  directory, entries, _, reason = _matching_entries(
      cache_dir, component, meta, abstract_tree, extra_names, source_files, check_dtypes
  )
  if entries is not None:
    reason = _weights_open_error(directory)
  if reason is not None:
    _log_miss(component, directory or cache_dir, reason)
    return False
  trusted = "; no checkpoint files to compare, trusted without them" if source_files is None else ""
  max_logging.log(f"[weight cache] {component}: {directory} is already complete{trusted}.")
  return True


def load_component(
    cache_dir, component, meta, abstract_tree, extra_names=(), source_files=None, check_dtypes=True, num_workers=None
) -> Optional[Tuple[object, dict]]:
  """Reads the cached tree of `component` for `meta`; returns `(tree, extras)` or None (a miss).

  Args:
    cache_dir: `krea2_weight_cache_dir`.
    component: `"transformer"` or `"text_encoder"`.
    meta: the fingerprint inputs (`*_weight_cache_meta`); the stored dict must be equal.
    abstract_tree: the runtime model's `jax.eval_shape(init)["params"]` (boxed
      or unboxed, dict or FrozenDict). The result has its structure as plain
      dicts, with its paths and shapes.
    extra_names: names of the extra arrays to return in `extras`.
    source_files: `list_source_files` of the checkpoint, which must equal the
      stored list; None (confirmed: no checkpoint on this machine) trusts the
      cache, a list from a failed scan (a None size) is a miss.
    check_dtypes: require exactly the abstract leaves' dtypes. When False (the
      text encoder, whose stored scales have the compute dtype), a leaf with a
      non-floating abstract dtype (int8 qvalues) still needs exactly that
      dtype, and a leaf with a floating abstract dtype needs a floating stored
      dtype of any width.
    num_workers: read threads (default `min(8, os.cpu_count())`).

  Never raises for a bad or missing cache, nor for a resource problem while
  reading it (OSError, MemoryError, a thread that cannot start): logs one line
  with the reason and returns None. Errors in `abstract_tree` itself raise.
  """
  start = time.perf_counter()
  directory, entries, treedef, reason = _matching_entries(
      cache_dir, component, meta, abstract_tree, extra_names, source_files, check_dtypes
  )

  def miss(reason):  # returns None
    _log_miss(component, directory or cache_dir, reason)

  if entries is None:
    return miss(reason)
  try:
    arrays = [np.empty(tuple(entry["shape"]), jnp.dtype(entry["dtype"])) for entry in entries]
  except (*_RESOURCE_ERRORS, ValueError, TypeError, OverflowError) as exc:
    return miss(f"allocation failed ({exc!r})")
  try:
    _read_arrays(os.path.join(directory, _WEIGHTS_FILE), entries, arrays, num_workers)
  except (*_RESOURCE_ERRORS, _ShortRead) as exc:
    return miss(f"read failed ({exc!r})")
  num_leaves = len(entries) - len(extra_names)
  try:
    tree = jax.tree_util.tree_unflatten(treedef, arrays[:num_leaves])
    extras = dict(zip(extra_names, arrays[num_leaves:]))
  except (*_RESOURCE_ERRORS, ValueError, TypeError, OverflowError) as exc:
    return miss(f"tree reconstruction failed ({exc!r})")

  if source_files is None:
    max_logging.log(f"[weight cache] {component}: no checkpoint files to compare; trusted {directory} without them.")
  total = sum(array.nbytes for array in arrays)
  max_logging.log(
      f"[weight cache] {component}: loaded {total / _GIB:.2f} GiB from {directory} "
      f"in {time.perf_counter() - start:.1f} s"
  )
  return tree, extras
