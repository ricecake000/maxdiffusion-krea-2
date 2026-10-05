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

"""Tests for the lazy GCS fetch of AOT executables (install(gcs_prefix=...), CPU backend, fake storage client)."""

import functools
import glob
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxdiffusion import aot_cache


@functools.partial(aot_cache.cached_jit, static_argnames=("flag",))
def _gcs_toy_fn(x, y, flag=False):
  return x @ y + (1.0 if flag else 0.0)


class _FakeBlob:
  """A storage blob holding `data`; `fail` makes the download write a partial file and raise."""

  def __init__(self, data, fail=None):
    self.data = data
    self.size = len(data)
    self.fail = fail
    self.downloads = []

  def download_to_filename(self, filename, **kwargs):
    self.downloads.append((filename, kwargs))
    with open(filename, "wb") as f:
      f.write(self.data[: len(self.data) // 2] if self.fail else self.data)
    if self.fail:
      raise self.fail


class _GatedBlob(_FakeBlob):
  """A blob whose download writes half the data, sets `started`, waits for `release` (bounded), writes the rest."""

  def __init__(self, data, delay_first=0.0):
    super().__init__(data)
    self.started = threading.Event()
    self.release = threading.Event()
    self.delay_first = delay_first
    self._lock = threading.Lock()

  def download_to_filename(self, filename, **kwargs):
    with self._lock:
      first = not self.downloads
      self.downloads.append((filename, kwargs))
    half = len(self.data) // 2
    with open(filename, "wb") as f:
      f.write(self.data[:half])
      f.flush()
      self.started.set()
      if first and self.delay_first:
        time.sleep(self.delay_first)
      self.release.wait(10)
      f.write(self.data[half:])


class _FakeBucket:

  def __init__(self, objects=None, get_error=None):
    self.objects = dict(objects or {})
    self.get_error = get_error
    self.requested = []

  def get_blob(self, name):
    self.requested.append(name)
    if self.get_error is not None:
      raise self.get_error
    return self.objects.get(name)


class _FakeClient:

  def __init__(self, buckets):
    self.buckets = buckets
    self.bucket_names = []

  def bucket(self, name):
    self.bucket_names.append(name)
    return self.buckets[name]


class _FakeTransferManager:

  def __init__(self):
    self.calls = []

  def download_chunks_concurrently(self, blob, filename, **kwargs):
    self.calls.append((blob, filename, kwargs))
    with open(filename, "wb") as f:
      f.write(blob.data)


class AotCacheGcsTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self._mesh = Mesh(np.array(jax.devices()[:1]), ("d",))
    self._a = jnp.ones((8, 8))
    self._b = jnp.eye(8)
    aot_cache._STATE.enabled = False
    aot_cache._STATE.lazy_load = False
    aot_cache._STATE.gcs_prefix = ""
    for entry in aot_cache._REGISTRY:
      entry._compiled.clear()
      entry._pending.clear()
      entry._lazy_tried.clear()
    self._bucket = _FakeBucket()
    self._client = _FakeClient({"krea2-bkt": self._bucket})
    self._clients_made = []

    def make_client():
      self._clients_made.append(1)
      return self._client

    patches = [
        mock.patch.object(aot_cache, "_GCS_CLIENT", None),
        mock.patch.object(aot_cache, "_make_gcs_client", side_effect=make_client),
    ]
    for patch in patches:
      patch.start()
      self.addCleanup(patch.stop)

  def tearDown(self):
    aot_cache._STATE.enabled = False
    aot_cache._STATE.lazy_load = False
    aot_cache._STATE.gcs_prefix = ""
    self._tmp.cleanup()

  def _entry(self):
    return next(e for e in aot_cache._REGISTRY if e.name.endswith("._gcs_toy_fn"))

  def _signature(self, flag=True):
    return aot_cache._dynamic_signature((), {"x": self._a, "y": self._b, "flag": flag})

  def _path(self):
    return self._entry()._path_for(self._signature())

  def _saved_executable_bytes(self):
    """Saves the _gcs_toy_fn(a, b, True) executable locally, returns its bytes and removes the file."""
    aot_cache.install(self._tmp.name, meta={"m": 1}, mesh=self._mesh)
    aot_cache.wait_for_loads()
    _gcs_toy_fn(self._a, self._b, True)
    self.assertEqual(aot_cache.save_pending(), 1)
    path = self._path()
    with open(path, "rb") as f:
      data = f.read()
    os.remove(path)
    return os.path.basename(path), data

  def _install_lazy(self, gcs_prefix="gs://krea2-bkt/aot"):
    aot_cache.install(self._tmp.name, meta={"m": 1}, mesh=self._mesh, lazy_load=True, gcs_prefix=gcs_prefix)
    aot_cache.wait_for_loads()

  def _leftovers(self):
    return glob.glob(os.path.join(self._tmp.name, "*.dl-tmp-*"))

  def test_hit_fetches_with_crc32c_and_loads_without_jit(self):
    name, data = self._saved_executable_bytes()
    blob = _FakeBlob(data)
    self._bucket.objects[f"aot/{name}"] = blob
    self._install_lazy()
    entry = self._entry()
    with (
        mock.patch.object(entry, "_adapter_for", side_effect=AssertionError("jit fallback used")),
        mock.patch.object(aot_cache.max_logging, "log") as log,
    ):
      result = _gcs_toy_fn(self._a, self._b, True)
      again = _gcs_toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(result), self._a @ self._b + 1.0)
    np.testing.assert_allclose(np.asarray(again), self._a @ self._b + 1.0)
    self.assertIn(self._signature(), entry._on_disk)
    with open(self._path(), "rb") as f:
      self.assertEqual(f.read(), data)
    # One fetch, into <path>.dl-tmp-<pid>-<thread id>, crc32c checked by the client; no leftover.
    self.assertEqual(self._bucket.requested, [f"aot/{name}"])
    self.assertEqual(blob.downloads, [(f"{self._path()}.dl-tmp-{os.getpid()}-{threading.get_ident()}", {"checksum": "crc32c"})])
    self.assertEqual(self._leftovers(), [])
    lines = [c.args[0] for c in log.call_args_list]
    fetched = [line for line in lines if " fetched " in line]
    self.assertEqual(len(fetched), 1)
    self.assertRegex(fetched[0], rf"fetched {name} \(\d+\.\dMB\) from gs://krea2-bkt/aot/{name} in \d+\.\d\d s$")
    self.assertTrue(any(" loaded " in line for line in lines))
    self.assertEqual(len(self._clients_made), 1)

  def test_missing_object_is_a_miss_and_compiles(self):
    self._install_lazy()
    entry = self._entry()
    name = os.path.basename(self._path())
    with mock.patch.object(aot_cache.max_logging, "log") as log:
      first = _gcs_toy_fn(self._a, self._b, True)
      second = _gcs_toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(first), self._a @ self._b + 1.0)
    np.testing.assert_allclose(np.asarray(second), self._a @ self._b + 1.0)
    # Tried once per install, one log line.
    self.assertEqual(self._bucket.requested, [f"aot/{name}"])
    self.assertEqual(
        [c.args[0] for c in log.call_args_list if "not in" in c.args[0]],
        [f"[aot] {entry.name}: {name} not in gs://krea2-bkt/aot; will compile"],
    )
    self.assertFalse(os.path.exists(self._path()))
    self.assertIn(self._signature(), entry._pending)
    self.assertEqual(aot_cache.save_pending(), 1)
    self.assertTrue(os.path.exists(self._path()))

  def test_failed_download_is_a_miss_and_removes_the_temp_file(self):
    name, data = self._saved_executable_bytes()
    blob = _FakeBlob(data, fail=RuntimeError("crc32c mismatch"))
    self._bucket.objects[f"aot/{name}"] = blob
    self._install_lazy()
    with mock.patch.object(aot_cache.max_logging, "log") as log:
      result = _gcs_toy_fn(self._a, self._b, True)
    np.testing.assert_allclose(np.asarray(result), self._a @ self._b + 1.0)
    self.assertEqual(len(blob.downloads), 1)
    self.assertEqual(self._leftovers(), [])
    self.assertFalse(os.path.exists(self._path()))
    failed = [c.args[0] for c in log.call_args_list if "failed" in c.args[0]]
    self.assertEqual(len(failed), 1)
    self.assertIn("crc32c mismatch", failed[0])
    self.assertTrue(failed[0].endswith("; will compile"))
    self.assertIn(self._signature(), self._entry()._pending)

  def test_client_errors_are_a_miss(self):
    for error_source in ("get_blob", "client"):
      with self.subTest(error_source=error_source):
        for entry in aot_cache._REGISTRY:
          entry._lazy_tried.clear()
        aot_cache._GCS_CLIENT = None  # restored by the setUp patch
        if error_source == "get_blob":
          self._bucket.get_error = PermissionError("403 forbidden")
        else:
          self._bucket.get_error = None
          aot_cache._make_gcs_client.side_effect = OSError("no credentials")
        self._install_lazy()
        with mock.patch.object(aot_cache.max_logging, "log") as log:
          result = _gcs_toy_fn(self._a, self._b, True)
        np.testing.assert_allclose(np.asarray(result), self._a @ self._b + 1.0)
        self.assertTrue(any("failed" in c.args[0] and "will compile" in c.args[0] for c in log.call_args_list))
        self.assertEqual(self._leftovers(), [])

  def test_big_object_uses_chunked_crc32c_download(self):
    name, data = self._saved_executable_bytes()
    blob = _FakeBlob(data)
    self._bucket.objects[f"aot/{name}"] = blob
    manager = _FakeTransferManager()
    with (
        mock.patch.object(aot_cache, "_GCS_BIG_BYTES", 1),
        mock.patch.object(aot_cache, "_gcs_transfer_manager", return_value=manager),
    ):
      self._install_lazy()
      entry = self._entry()
      with mock.patch.object(entry, "_adapter_for", side_effect=AssertionError("jit fallback used")):
        _gcs_toy_fn(self._a, self._b, True)
    self.assertEqual(blob.downloads, [])
    self.assertEqual(
        manager.calls,
        [
            (
                blob,
                f"{self._path()}.dl-tmp-{os.getpid()}-{threading.get_ident()}",
                {"chunk_size": 32 * 1024**2, "max_workers": 8, "worker_type": "thread", "crc32c_checksum": True},
            )
        ],
    )
    self.assertIn(self._signature(), entry._on_disk)
    self.assertEqual(self._leftovers(), [])

  def _fetch_in_thread(self, results, key, path):
    def run():
      results[key] = aot_cache._fetch_from_gcs("fn", path, "gs://krea2-bkt/aot")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread

  def test_concurrent_fetches_of_one_path_download_once(self):
    # Two wrappers of the same name (independent pipelines) fetch the same final path at once: the second waits
    # for the first one's download, then finds the file and returns True; the partial file never gets the name.
    data = bytes(range(256)) * 64
    name = "_gcs_toy_fn-fp-sig.aotx"
    path = os.path.join(self._tmp.name, name)
    blob = _GatedBlob(data, delay_first=0.5)
    blob.release.set()
    self._bucket.objects[f"aot/{name}"] = blob
    results = {}
    first = self._fetch_in_thread(results, "first", path)
    self.assertTrue(blob.started.wait(10))
    second = self._fetch_in_thread(results, "second", path)
    first.join(30)
    second.join(30)
    self.assertFalse(first.is_alive() or second.is_alive())
    self.assertEqual(results, {"first": True, "second": True})
    self.assertEqual(blob.downloads, [(f"{path}.dl-tmp-{os.getpid()}-{first.ident}", {"checksum": "crc32c"})])
    self.assertEqual(self._bucket.requested, [f"aot/{name}"])
    with open(path, "rb") as f:
      self.assertEqual(f.read(), data)
    self.assertEqual(self._leftovers(), [])

  def test_fetches_of_different_paths_run_in_parallel(self):
    # The lock is per final path: the first download completes only once the second one has started.
    data = b"x" * 1000
    blob_a, blob_b = _GatedBlob(data), _GatedBlob(data)
    blob_b.release.set()
    self._bucket.objects["aot/a.aotx"] = blob_a
    self._bucket.objects["aot/b.aotx"] = blob_b
    results = {}
    first = self._fetch_in_thread(results, "a", os.path.join(self._tmp.name, "a.aotx"))
    self.assertTrue(blob_a.started.wait(10))
    second = self._fetch_in_thread(results, "b", os.path.join(self._tmp.name, "b.aotx"))
    self.assertTrue(blob_b.started.wait(10), "the fetch of another path waited for the first one")
    blob_a.release.set()
    first.join(30)
    second.join(30)
    self.assertFalse(first.is_alive() or second.is_alive())
    self.assertEqual(results, {"a": True, "b": True})
    self.assertEqual(self._leftovers(), [])

  def test_local_file_is_used_without_fetching(self):
    aot_cache.install(self._tmp.name, meta={"m": 1}, mesh=self._mesh)
    _gcs_toy_fn(self._a, self._b, True)
    self.assertEqual(aot_cache.save_pending(), 1)
    self._install_lazy()
    _gcs_toy_fn(self._a, self._b, True)
    self.assertIn(self._signature(), self._entry()._on_disk)
    self.assertEqual(self._bucket.requested, [])
    self.assertEqual(self._clients_made, [])

  def test_bucket_root_prefix(self):
    name, data = self._saved_executable_bytes()
    self._bucket.objects[name] = _FakeBlob(data)
    self._install_lazy("gs://krea2-bkt/")
    _gcs_toy_fn(self._a, self._b, True)
    self.assertEqual(self._bucket.requested, [name])
    self.assertIn(self._signature(), self._entry()._on_disk)

  def test_eager_install_ignores_the_prefix(self):
    with mock.patch.object(aot_cache.max_logging, "log") as log:
      aot_cache.install(self._tmp.name, meta={"m": 1}, mesh=self._mesh, gcs_prefix="gs://krea2-bkt/aot")
      aot_cache.wait_for_loads()
    self.assertEqual(aot_cache._STATE.gcs_prefix, "")
    self.assertTrue(any("ignored" in c.args[0] for c in log.call_args_list))
    _gcs_toy_fn(self._a, self._b, True)
    self.assertEqual(self._bucket.requested, [])
    self.assertEqual(self._clients_made, [])

  def test_prefix_is_not_part_of_the_fingerprint(self):
    aot_cache.install(self._tmp.name, meta={"m": 1}, mesh=self._mesh, lazy_load=True)
    without = aot_cache._STATE.fingerprint
    self._install_lazy()
    self.assertEqual(aot_cache._STATE.fingerprint, without)
    self.assertEqual(aot_cache._STATE.gcs_prefix, "gs://krea2-bkt/aot")

  def test_invalid_prefix_raises_at_install(self):
    with self.assertRaises(ValueError):
      aot_cache.install(self._tmp.name, meta={"m": 1}, mesh=self._mesh, lazy_load=True, gcs_prefix="s3://x/y")

  def test_normalize_gcs_prefix(self):
    for value, expected in (
        ("", ""),
        (None, ""),
        ("''", ""),
        ('  ""  ', ""),
        ("gs://krea2-bkt", "gs://krea2-bkt"),
        ("gs://krea2-bkt/", "gs://krea2-bkt"),
        ("gs://tpu-test-507316-krea2-use1/aot/", "gs://tpu-test-507316-krea2-use1/aot"),
        ("'gs://b.example.com/a/b'", "gs://b.example.com/a/b"),
    ):
      with self.subTest(value=value):
        self.assertEqual(aot_cache.normalize_gcs_prefix(value), expected)
    for value in (
        "krea2-bkt/aot",
        "s3://krea2-bkt/aot",
        "gs://",
        "gs://ab",
        "gs://Krea2-Bkt/aot",
        "gs://krea2_bkt-/aot",
        "gs://krea2-bkt//aot",
        "gs://krea2-bkt/a/../b",
        "gs://krea2-bkt/a b",
    ):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          aot_cache.normalize_gcs_prefix(value)


if __name__ == "__main__":
  unittest.main()
