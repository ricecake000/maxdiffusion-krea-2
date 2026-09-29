# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the per-shape AOT executable cache (CPU backend)."""

import contextlib
import functools
import glob
import os
import pickle
import tempfile
import threading
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxdiffusion import aot_cache


@functools.partial(aot_cache.cached_jit, static_argnames=("flag",))
def _toy_fn(x, y, flag=False):
  return x @ y + (1.0 if flag else 0.0)


@functools.partial(aot_cache.cached_jit, static_argnames=("flag",), donate_argnames=("h",))
def _donating_fn(h, w, flag=False):
  return h @ w + (1.0 if flag else 0.0)


class _ObservedLock:
  """Wraps a lock and signals once a second caller tries to enter it."""

  def __init__(self, lock):
    self._lock = lock
    self._count_lock = threading.Lock()
    self._entries = 0
    self.second_entry = threading.Event()

  def __enter__(self):
    with self._count_lock:
      self._entries += 1
      if self._entries >= 2:
        self.second_entry.set()
    return self._lock.__enter__()

  def __exit__(self, *exc_info):
    return self._lock.__exit__(*exc_info)


class AotCacheTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self._mesh = Mesh(np.array(jax.devices()[:1]), ("d",))
    self._a = jnp.ones((8, 8))
    self._b = jnp.eye(8)
    # Reset process-global install state between tests.
    aot_cache._STATE.enabled = False
    aot_cache._STATE.lazy_load = False
    for entry in aot_cache._REGISTRY:
      entry._compiled.clear()
      entry._pending.clear()
      entry._lazy_tried.clear()

  def tearDown(self):
    aot_cache._STATE.enabled = False
    aot_cache._STATE.lazy_load = False
    self._tmp.cleanup()

  def _install(self):
    aot_cache.install(self._tmp.name, meta={"m": 1}, mesh=self._mesh)
    aot_cache.wait_for_loads()

  @staticmethod
  def _entry(suffix):
    return next(e for e in aot_cache._REGISTRY if e.name.endswith(suffix))

  def _fresh_h(self):
    return jnp.full((8, 8), 2.0)

  def _load_blob(self, name_part):
    (path,) = glob.glob(os.path.join(self._tmp.name, f"*{name_part}-*.aotx"))
    with open(path, "rb") as f:
      return pickle.load(f)

  def test_disabled_is_plain_jit(self):
    result = _toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(result, self._a @ self._b + 1.0)
    self.assertFalse(aot_cache._STATE.enabled)

  def test_record_save_hit_roundtrip(self):
    self._install()
    first = _toy_fn(self._a, self._b, True)  # miss -> jit + record
    self.assertEqual(aot_cache.save_pending(), 1)
    hit = _toy_fn(self._a, self._b, True)  # compiled hit
    np.testing.assert_allclose(np.asarray(first), np.asarray(hit))

  def test_reload_from_disk(self):
    self._install()
    first = _toy_fn(self._a, self._b, True)
    aot_cache.save_pending()
    entry = next(e for e in aot_cache._REGISTRY if e.name.endswith("._toy_fn"))
    entry._compiled.clear()  # simulate a fresh process
    entry.load_from_disk()
    self.assertTrue(entry._compiled)
    reloaded = _toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(first), np.asarray(reloaded))

  def test_static_value_gets_own_executable(self):
    self._install()
    _toy_fn(self._a, self._b, True)
    aot_cache.save_pending()
    # Different static value -> different signature -> jit fallback, correct.
    off = _toy_fn(self._a, self._b, False)
    np.testing.assert_allclose(off, self._a @ self._b)

  def test_new_shape_falls_back_and_saves(self):
    self._install()
    _toy_fn(self._a, self._b, True)
    self.assertEqual(aot_cache.save_pending(), 1)
    small = _toy_fn(jnp.ones((4, 8)), jnp.ones((8, 4)), True)
    self.assertEqual(small.shape, (4, 4))
    self.assertEqual(aot_cache.save_pending(), 1)  # only the new shape

  def test_warmup_mode_compiles_without_executing(self):
    self._install()
    with aot_cache.warmup_mode():
      warm = _toy_fn(self._a, self._b, True)
    # Zeros prove the fn body never ran (real output would be a@b+1).
    self.assertEqual(warm.shape, (8, 8))
    np.testing.assert_allclose(np.asarray(warm), np.zeros((8, 8)))
    # The signature was compiled during warmup and is serializable.
    self.assertEqual(aot_cache.save_pending(), 1)
    # Outside warmup mode the compiled executable returns real values.
    real = _toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(real), self._a @ self._b + 1.0)

  def test_warmup_mode_disabled_cache_executes_normally(self):
    with aot_cache.warmup_mode():  # cache not installed -> no-op
      result = _toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(result), self._a @ self._b + 1.0)

  def test_traced_call_inlines_like_nested_jit(self):
    # I2V's denoise loop invokes wrapped fns inside lax.cond branches; a
    # deserialized executable cannot be applied to tracers. The wrapper
    # must inline (plain nested-jit behavior) and never crash or record.
    self._install()
    _toy_fn(self._a, self._b, True)
    aot_cache.save_pending()  # compiled entry exists for this signature

    def branch_true(x):
      return _toy_fn(x, self._b, True)

    def branch_false(x):
      return x

    result = jax.lax.cond(True, branch_true, branch_false, self._a)
    np.testing.assert_allclose(np.asarray(result), self._a @ self._b + 1.0)
    self.assertEqual(aot_cache.save_pending(), 0)  # nothing recorded

  def test_undonated_on_disk_layout_is_single_list(self):
    # Existing executables for undonated fns must stay loadable: the
    # adapter still takes ONE flat leaf list and nothing is donated.
    self._install()
    _toy_fn(self._a, self._b, True)
    self.assertEqual(aot_cache.save_pending(), 1)
    blob = self._load_blob("._toy_fn")
    self.assertEqual(blob["in_tree"], jax.tree_util.tree_structure((([0, 0],), {})))
    self.assertFalse(self._a.is_deleted())
    self.assertFalse(self._b.is_deleted())

  def test_donation_disabled_cache_is_plain_jit(self):
    h = self._fresh_h()
    result = _donating_fn(h, self._b, True)
    np.testing.assert_allclose(np.asarray(result), np.full((8, 8), 3.0))
    self.assertTrue(h.is_deleted())
    self.assertFalse(self._b.is_deleted())

  def test_donation_record_save_hit(self):
    self._install()
    h0 = self._fresh_h()
    first = _donating_fn(h0, self._b, True)  # miss -> donating adapter jit
    self.assertTrue(h0.is_deleted())
    self.assertEqual(aot_cache.save_pending(), 1)
    blob = self._load_blob("._donating_fn")
    # (donated_flat, flat) positional lists: h alone, then w.
    self.assertEqual(blob["in_tree"], jax.tree_util.tree_structure((([0], [0]), {})))
    entry = self._entry("._donating_fn")
    h1 = self._fresh_h()
    with mock.patch.object(entry, "_adapter_for", side_effect=AssertionError("jit fallback used")):
      hit = _donating_fn(h=h1, w=self._b, flag=True)  # keyword form hits the same executable
    np.testing.assert_allclose(np.asarray(hit), np.asarray(first))
    np.testing.assert_allclose(np.asarray(hit), np.full((8, 8), 3.0))
    self.assertTrue(h1.is_deleted())
    self.assertFalse(self._b.is_deleted())

  def test_donation_reload_from_disk(self):
    self._install()
    _donating_fn(self._fresh_h(), self._b, True)
    self.assertEqual(aot_cache.save_pending(), 1)
    entry = self._entry("._donating_fn")
    entry._compiled.clear()  # simulate a fresh process
    entry._out_specs.clear()
    entry._adapters.clear()
    entry.load_from_disk()
    self.assertTrue(entry._compiled)
    h = self._fresh_h()
    with mock.patch.object(entry, "_adapter_for", side_effect=AssertionError("jit fallback used")):
      reloaded = _donating_fn(h, self._b, True)
    np.testing.assert_allclose(np.asarray(reloaded), np.full((8, 8), 3.0))
    self.assertTrue(h.is_deleted())
    self.assertFalse(self._b.is_deleted())

  def test_donation_warmup_mode_keeps_input(self):
    self._install()
    h = self._fresh_h()
    with aot_cache.warmup_mode():
      warm = _donating_fn(h, self._b, True)
    np.testing.assert_allclose(np.asarray(warm), np.zeros((8, 8)))
    self.assertFalse(h.is_deleted())  # compiled only, never executed
    self.assertEqual(aot_cache.save_pending(), 1)
    real = _donating_fn(h, self._b, True)
    np.testing.assert_allclose(np.asarray(real), np.full((8, 8), 3.0))
    self.assertTrue(h.is_deleted())

  def test_donation_traced_call_inlines(self):
    self._install()
    _donating_fn(self._fresh_h(), self._b, True)
    aot_cache.save_pending()

    @jax.jit
    def outer(x):
      return _donating_fn(x * 1.0, self._b, True)

    np.testing.assert_allclose(np.asarray(outer(self._fresh_h())), np.full((8, 8), 3.0))
    self.assertEqual(aot_cache.save_pending(), 0)  # nothing recorded

  @staticmethod
  def _fail_after_running(entry):
    """Wraps each compiled executable to run for real, then raise."""
    calls = []
    for signature, real in list(entry._compiled.items()):

      def failing(*arg_lists, _real=real):
        calls.append(signature)
        _real(*arg_lists)  # consumes donated buffers, like a late failure
        raise RuntimeError("compiled call failed after execution")

      entry._compiled[signature] = failing
    return calls

  def test_donation_compiled_failure_reraises_without_jit_retry(self):
    self._install()
    _donating_fn(self._fresh_h(), self._b, True)
    self.assertEqual(aot_cache.save_pending(), 1)
    entry = self._entry("._donating_fn")
    calls = self._fail_after_running(entry)
    h = self._fresh_h()
    with (
        mock.patch.object(entry, "_align_inputs", return_value=None),
        mock.patch.object(entry, "_adapter_for", side_effect=AssertionError("jit fallback used")),
    ):
      with self.assertRaisesRegex(RuntimeError, "after execution"):
        _donating_fn(h, self._b, True)
    self.assertEqual(len(calls), 1)
    self.assertTrue(h.is_deleted())  # the retry would have read this

  def test_undonated_compiled_failure_falls_back_to_jit(self):
    self._install()
    _toy_fn(self._a, self._b, True)
    self.assertEqual(aot_cache.save_pending(), 1)
    entry = self._entry("._toy_fn")
    calls = self._fail_after_running(entry)
    with mock.patch.object(entry, "_align_inputs", return_value=None):
      result = _toy_fn(self._a, self._b, True)
    self.assertEqual(len(calls), 1)
    np.testing.assert_allclose(np.asarray(result), self._a @ self._b + 1.0)

  def test_donation_rejects_static_overlap(self):
    with self.assertRaises(ValueError):
      aot_cache._AotEntry("t", lambda h, w: h, static_argnames=("h",), donate_argnames=("h",))
    with self.assertRaises(ValueError):
      aot_cache._AotEntry("t", lambda h, w: h, static_argnames=(), donate_argnames=("missing",))

  # ------------------------------------------------------------ lazy load
  def _install_lazy(self):
    aot_cache.install(self._tmp.name, meta={"m": 1}, mesh=self._mesh, lazy_load=True)
    aot_cache.wait_for_loads()

  def _toy_signature(self, flag):
    # The wrapper digests the call in keyword form: dynamic args, then statics.
    return aot_cache._dynamic_signature((), {"x": self._a, "y": self._b, "flag": flag})

  def _aotx_files(self):
    return glob.glob(os.path.join(self._tmp.name, "*.aotx"))

  def _save_toy(self):
    """Saves the _toy_fn(a, b, True) executable with an eager install; returns its path."""
    self._install()
    _toy_fn(self._a, self._b, True)
    self.assertEqual(aot_cache.save_pending(), 1)
    return self._entry("._toy_fn")._path_for(self._toy_signature(True))

  def test_lazy_install_loads_nothing(self):
    path = self._save_toy()
    self.assertTrue(os.path.exists(path))
    with mock.patch.object(aot_cache._AotEntry, "load_from_disk") as load_all:
      self._install_lazy()
    load_all.assert_not_called()
    self.assertEqual(aot_cache._LOAD_THREADS, [])
    self.assertTrue(aot_cache._STATE.lazy_load)
    for entry in aot_cache._REGISTRY:
      self.assertFalse(entry._compiled)
      self.assertFalse(entry._on_disk)

  def test_lazy_first_call_loads_saved_signature(self):
    path = self._save_toy()
    mtime = os.stat(path).st_mtime_ns
    self._install_lazy()
    entry = self._entry("._toy_fn")
    with mock.patch.object(entry, "_adapter_for", side_effect=AssertionError("jit fallback used")):
      result = _toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(result), self._a @ self._b + 1.0)
    self.assertIn(self._toy_signature(True), entry._on_disk)
    self.assertFalse(entry._pending)
    self.assertEqual(aot_cache.save_pending(), 0)
    self.assertEqual(self._aotx_files(), [path])
    self.assertEqual(os.stat(path).st_mtime_ns, mtime)

  def test_lazy_warmup_uses_loaded_executable(self):
    self._save_toy()
    self._install_lazy()
    entry = self._entry("._toy_fn")
    with mock.patch.object(entry, "_compile_and_record", side_effect=AssertionError("compiled")) as compile_mock:
      with aot_cache.warmup_mode():
        warm = _toy_fn(self._a, self._b, True)
    compile_mock.assert_not_called()
    self.assertEqual(warm.shape, (8, 8))
    np.testing.assert_allclose(np.asarray(warm), np.zeros((8, 8)))
    self.assertEqual(aot_cache.save_pending(), 0)

  def test_lazy_missing_file_falls_back_and_saves(self):
    self._install_lazy()
    entry = self._entry("._toy_fn")
    path = entry._path_for(self._toy_signature(True))
    with mock.patch.object(aot_cache.os.path, "exists", wraps=os.path.exists) as exists:
      first = _toy_fn(self._a, self._b, True)
      second = _toy_fn(self._a, self._b, True)
    self.assertEqual([c for c in exists.call_args_list if c.args == (path,)], [mock.call(path)])
    np.testing.assert_allclose(np.asarray(first), self._a @ self._b + 1.0)
    np.testing.assert_allclose(np.asarray(second), self._a @ self._b + 1.0)
    self.assertIn(self._toy_signature(True), entry._pending)
    self.assertEqual(aot_cache.save_pending(), 1)
    self.assertEqual(self._aotx_files(), [path])

  def test_lazy_corrupt_file_falls_back_and_reads_once(self):
    path = self._save_toy()
    with open(path, "wb") as f:
      f.write(b"not a pickle")
    self._install_lazy()
    entry = self._entry("._toy_fn")
    with mock.patch.object(entry, "_load_path", wraps=entry._load_path) as load_path:
      first = _toy_fn(self._a, self._b, True)
      second = _toy_fn(self._a, self._b, True)
    load_path.assert_called_once()
    self.assertEqual(load_path.call_args.args[0], path)
    self.assertEqual(load_path.call_args.kwargs["expected_signature"], self._toy_signature(True))
    np.testing.assert_allclose(np.asarray(first), self._a @ self._b + 1.0)
    np.testing.assert_allclose(np.asarray(second), self._a @ self._b + 1.0)
    self.assertNotIn(self._toy_signature(True), entry._compiled)
    # The fallback recorded the shape, so saving replaces the corrupt file.
    self.assertEqual(aot_cache.save_pending(), 1)
    self.assertEqual(self._load_blob("._toy_fn")["dynamic_signature"], self._toy_signature(True))

  def test_lazy_signature_mismatch_is_not_registered(self):
    path = self._save_toy()
    # The flag=True executable under the flag=False filename must not run for flag=False.
    other_path = self._entry("._toy_fn")._path_for(self._toy_signature(False))
    os.replace(path, other_path)
    self._install_lazy()
    entry = self._entry("._toy_fn")
    result = _toy_fn(self._a, self._b, False)
    np.testing.assert_allclose(np.asarray(result), self._a @ self._b)
    self.assertFalse(entry._compiled)
    self.assertFalse(entry._on_disk)
    self.assertIn(self._toy_signature(False), entry._pending)

  def test_eager_install_after_lazy_loads_eagerly(self):
    self._save_toy()
    self._install_lazy()
    entry = self._entry("._toy_fn")
    self.assertFalse(entry._compiled)
    self._install()
    self.assertFalse(aot_cache._STATE.lazy_load)
    self.assertIn(self._toy_signature(True), entry._compiled)
    self.assertIn(self._toy_signature(True), entry._on_disk)

  def test_lazy_donation_hit(self):
    self._install()
    _donating_fn(self._fresh_h(), self._b, True)
    self.assertEqual(aot_cache.save_pending(), 1)
    self._install_lazy()
    entry = self._entry("._donating_fn")
    self.assertFalse(entry._compiled)
    h = self._fresh_h()
    with mock.patch.object(entry, "_adapter_for", side_effect=AssertionError("jit fallback used")):
      result = _donating_fn(h, self._b, True)
    np.testing.assert_allclose(np.asarray(result), np.full((8, 8), 3.0))
    self.assertTrue(h.is_deleted())
    self.assertFalse(self._b.is_deleted())
    self.assertEqual(len(entry._on_disk), 1)
    self.assertEqual(aot_cache.save_pending(), 0)

  # ------------------------------------------------- lazy load concurrency
  @contextlib.contextmanager
  def _blocking_deserialize(self):
    """Makes deserialize_and_load wait for ``release``; yields (started, release, calls)."""
    started, release = threading.Event(), threading.Event()
    calls = []
    real = aot_cache.serialize_executable.deserialize_and_load

    def blocking(*args, **kwargs):
      calls.append(threading.current_thread().name)
      started.set()
      release.wait(timeout=60)
      return real(*args, **kwargs)

    try:
      with mock.patch.object(aot_cache.serialize_executable, "deserialize_and_load", blocking):
        yield started, release, calls
    finally:
      release.set()  # never leave a load thread blocked behind a failed assertion

  @staticmethod
  def _start_call(fn, *args):
    """Runs fn(*args) on a thread; the dict gets "result" (numpy) or "error"."""
    out = {}

    def run():
      try:
        out["result"] = np.asarray(fn(*args))
      except BaseException as e:  # noqa: BLE001 - reported by the test
        out["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, out

  def _assert_nothing_registered_and_jit_works(self, entry):
    self.assertFalse(entry._compiled)
    self.assertFalse(entry._on_disk)
    self.assertFalse(entry._out_specs)
    with mock.patch.object(entry, "_adapter_for", wraps=entry._adapter_for) as adapter_for:
      result = _toy_fn(self._a, self._b, True)
    adapter_for.assert_called_once()
    np.testing.assert_allclose(np.asarray(result), self._a @ self._b + 1.0)
    self.assertNotIn(self._toy_signature(True), entry._compiled)

  @staticmethod
  def _logged_previous_install(log):
    return any("previous install" in str(c.args[0]) for c in log.call_args_list)

  def test_lazy_load_during_reinstall_is_dropped(self):
    self._save_toy()
    self._install_lazy()
    entry = self._entry("._toy_fn")
    with (
        self._blocking_deserialize() as (started, release, calls),
        mock.patch.object(aot_cache.max_logging, "log") as log,
    ):
      thread, out = self._start_call(_toy_fn, self._a, self._b, True)
      self.assertTrue(started.wait(timeout=60))
      aot_cache.install(self._tmp.name, meta={"m": 2}, mesh=self._mesh, lazy_load=True)
      release.set()
      thread.join(timeout=60)
    self.assertFalse(thread.is_alive())
    self.assertNotIn("error", out)
    # The racing call itself falls back to jit and still returns the right values.
    np.testing.assert_allclose(out["result"], self._a @ self._b + 1.0)
    self.assertEqual(len(calls), 1)
    self.assertTrue(self._logged_previous_install(log))
    self._assert_nothing_registered_and_jit_works(entry)

  def test_eager_load_during_reinstall_is_dropped(self):
    self._save_toy()
    entry = self._entry("._toy_fn")
    with (
        self._blocking_deserialize() as (started, release, calls),
        mock.patch.object(aot_cache.max_logging, "log") as log,
    ):
      aot_cache.install(self._tmp.name, meta={"m": 1}, mesh=self._mesh)  # background thread blocks in deserialize
      self.assertTrue(started.wait(timeout=60))
      aot_cache.install(self._tmp.name, meta={"m": 2}, mesh=self._mesh)
      release.set()
      aot_cache.wait_for_loads()
    self.assertEqual(len(calls), 1)
    self.assertTrue(self._logged_previous_install(log))
    self._assert_nothing_registered_and_jit_works(entry)

  def test_lazy_concurrent_first_calls_share_one_load(self):
    self._save_toy()
    self._install_lazy()
    entry = self._entry("._toy_fn")
    observed = _ObservedLock(entry._lazy_lock)
    with (
        self._blocking_deserialize() as (started, release, calls),
        mock.patch.object(entry, "_lazy_lock", observed),
        mock.patch.object(entry, "_adapter_for", side_effect=AssertionError("jit fallback used")),
    ):
      first, out1 = self._start_call(_toy_fn, self._a, self._b, True)
      self.assertTrue(started.wait(timeout=60))
      second, out2 = self._start_call(_toy_fn, self._a, self._b, True)
      # The second caller reached the lazy lock while the first still deserializes.
      self.assertTrue(observed.second_entry.wait(timeout=60))
      release.set()
      first.join(timeout=60)
      second.join(timeout=60)
    self.assertFalse(first.is_alive())
    self.assertFalse(second.is_alive())
    for out in (out1, out2):
      self.assertNotIn("error", out)
      np.testing.assert_allclose(out["result"], self._a @ self._b + 1.0)
    self.assertEqual(len(calls), 1)
    self.assertFalse(entry._pending)
    self.assertEqual(aot_cache.save_pending(), 0)

  def test_signature_deterministic_across_processes(self):
    # Signatures live in filenames; a process-dependent component (e.g.
    # object addresses inside a GraphDef repr) would make every restart
    # miss its own cache. Compute the same signature in two interpreters.
    import subprocess
    import sys

    snippet = "\n".join((
        "import os",
        "os.environ['JAX_PLATFORMS'] = 'cpu'",
        "import jax",
        "import jax.numpy as jnp",
        "from flax import nnx",
        "from maxdiffusion import aot_cache",
        "",
        "class T(nnx.Module):",
        "  def __init__(self, rngs):",
        "    self.lin = nnx.Linear(4, 4, rngs=rngs)",
        "",
        "graphdef, state = nnx.split(T(nnx.Rngs(0)))",
        "sig = aot_cache._dynamic_signature(",
        "    (graphdef, state.to_pure_dict(), jnp.ones((2, 4))), {})",
        "print(sig)",
    ))
    outs = [
        subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        ).stdout.strip()
        for _ in range(2)
    ]
    self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
  unittest.main()
