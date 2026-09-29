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

# CPU tests for the Krea 2 named resolutions (aspect ratio + image size) and
# the precompile list parser.

import dataclasses
import unittest

from maxdiffusion.models.krea2.resolution_presets import (
    KREA2_ASPECT_RATIOS,
    KREA2_DEFAULT_ASPECT_RATIO,
    KREA2_DEFAULT_IMAGE_SIZE,
    KREA2_IMAGE_SIZES,
    Krea2Resolution,
    list_krea2_resolutions,
    parse_krea2_precompile,
    resolve_krea2_resolution,
)

# The spec's table, landscape entries as (width, height).
_EXPECTED_LANDSCAPE = {
    ("1k", "1:1"): (1024, 1024),
    ("1k", "5:4"): (1152, 896),
    ("1k", "4:3"): (1152, 864),
    ("1k", "3:2"): (1248, 832),
    ("1k", "16:9"): (1344, 768),
    ("1k", "21:9"): (1536, 672),
    ("2k", "1:1"): (2048, 2048),
    ("2k", "5:4"): (2304, 1792),
    ("2k", "4:3"): (2304, 1728),
    ("2k", "3:2"): (2496, 1664),
    ("2k", "16:9"): (2688, 1536),
    ("2k", "21:9"): (3072, 1344),
}
_EXPECTED_TOKENS = {
    "1k": {"1:1": 4096, "5:4": 4032, "4:3": 3888, "3:2": 4056, "16:9": 4032, "21:9": 4032},
    "2k": {"1:1": 16384, "5:4": 16128, "4:3": 15552, "3:2": 16224, "16:9": 16128, "21:9": 16128},
}
_PORTRAIT_OF = {"5:4": "4:5", "4:3": "3:4", "3:2": "2:3", "16:9": "9:16", "21:9": "9:21"}


def _nominal(ratio):
  w, h = ratio.split(":")
  return int(w) / int(h)


class Krea2ResolutionTableTest(unittest.TestCase):

  def test_constants(self):
    self.assertEqual(KREA2_IMAGE_SIZES, ("1k", "2k"))
    self.assertEqual(KREA2_DEFAULT_IMAGE_SIZE, "1k")
    self.assertEqual(KREA2_DEFAULT_ASPECT_RATIO, "1:1")
    self.assertEqual(len(KREA2_ASPECT_RATIOS), 11)
    self.assertEqual(len(list_krea2_resolutions()), 22)

  def test_landscape_table(self):
    for (size, ratio), (width, height) in _EXPECTED_LANDSCAPE.items():
      with self.subTest(size=size, ratio=ratio):
        r = resolve_krea2_resolution(ratio, size)
        self.assertEqual(r, Krea2Resolution(image_size=size, aspect_ratio=ratio, height=height, width=width))
        self.assertEqual(r.image_tokens, _EXPECTED_TOKENS[size][ratio])
        self.assertEqual(r.label, f"{size}@{ratio}")

  def test_multiples_of_16(self):
    for r in list_krea2_resolutions():
      with self.subTest(label=r.label):
        self.assertEqual(r.height % 16, 0)
        self.assertEqual(r.width % 16, 0)
        self.assertEqual(r.image_tokens, (r.height // 16) * (r.width // 16))

  def test_portrait_is_swapped_landscape(self):
    for size in KREA2_IMAGE_SIZES:
      for landscape, portrait in _PORTRAIT_OF.items():
        with self.subTest(size=size, ratio=portrait):
          lr = resolve_krea2_resolution(landscape, size)
          pr = resolve_krea2_resolution(portrait, size)
          self.assertEqual((pr.height, pr.width), (lr.width, lr.height))
          self.assertEqual(pr.image_tokens, lr.image_tokens)
          self.assertEqual(pr.aspect_ratio, portrait)
    self.assertEqual(set(_PORTRAIT_OF) | set(_PORTRAIT_OF.values()) | {"1:1"}, set(KREA2_ASPECT_RATIOS))

  def test_2k_is_twice_1k(self):
    for ratio in KREA2_ASPECT_RATIOS:
      with self.subTest(ratio=ratio):
        r1 = resolve_krea2_resolution(ratio, "1k")
        r2 = resolve_krea2_resolution(ratio, "2k")
        self.assertEqual((r2.height, r2.width), (2 * r1.height, 2 * r1.width))

  def test_token_counts(self):
    for size in KREA2_IMAGE_SIZES:
      with self.subTest(size=size):
        resolutions = list_krea2_resolutions(size)
        tokens = {r.image_tokens for r in resolutions}
        self.assertEqual(len(tokens), 4)
        square = resolve_krea2_resolution("1:1", size).image_tokens
        self.assertEqual(max(tokens), square)
        # The approximated ratios share one executable.
        shared = {resolve_krea2_resolution(ratio, size).image_tokens for ratio in ("5:4", "16:9", "21:9")}
        self.assertEqual(len(shared), 1)

  def test_actual_ratio_close_to_nominal(self):
    for r in list_krea2_resolutions():
      with self.subTest(label=r.label):
        actual = r.width / r.height
        nominal = _nominal(r.aspect_ratio)
        self.assertLess(abs(actual - nominal) / nominal, 0.04)
        self.assertEqual(actual > 1, nominal > 1)

  def test_list_order(self):
    labels = [r.label for r in list_krea2_resolutions()]
    expected = [f"{size}@{ratio}" for size in KREA2_IMAGE_SIZES for ratio in KREA2_ASPECT_RATIOS]
    self.assertEqual(labels, expected)
    self.assertEqual([r.label for r in list_krea2_resolutions("2k")], expected[11:])
    self.assertEqual([r.label for r in list_krea2_resolutions(" 1K ")], expected[:11])

  def test_resolution_is_frozen(self):
    r = resolve_krea2_resolution("1:1")
    with self.assertRaises(dataclasses.FrozenInstanceError):
      r.height = 16


class Krea2ResolveTest(unittest.TestCase):

  def test_defaults(self):
    self.assertEqual(resolve_krea2_resolution("16:9"), resolve_krea2_resolution("16:9", "1k"))
    self.assertEqual(resolve_krea2_resolution(KREA2_DEFAULT_ASPECT_RATIO).label, "1k@1:1")

  def test_normalization(self):
    expected = resolve_krea2_resolution("16:9", "2k")
    for ratio, size in [
        ("16:9", "2K"),
        (" 16:9 ", " 2k "),
        ("16 : 9", "2k"),
        ("16: 9", "2k"),
        ("\t16 :9\n", "2k"),
    ]:
      with self.subTest(ratio=ratio, size=size):
        self.assertEqual(resolve_krea2_resolution(ratio, size), expected)

  def test_unknown_ratio(self):
    for ratio in ["16x9", "1.78", "7:5", "", "16:9:1", "1 6:9", "16/9"]:
      with self.subTest(ratio=ratio):
        with self.assertRaises(ValueError) as cm:
          resolve_krea2_resolution(ratio, "1k")
        message = str(cm.exception)
        self.assertIn(repr(ratio), message)
        for valid in KREA2_ASPECT_RATIOS:
          self.assertIn(valid, message)

  def test_unknown_size(self):
    for size in ["4k", "1024", "", "1 k"]:
      with self.subTest(size=size):
        with self.assertRaises(ValueError) as cm:
          resolve_krea2_resolution("1:1", size)
        message = str(cm.exception)
        self.assertIn(repr(size), message)
        self.assertIn("1k", message)
        self.assertIn("2k", message)
    with self.assertRaises(ValueError):
      list_krea2_resolutions("4k")

  def test_non_string_ratio(self):
    for ratio in [969, 1.5, None, ("16", "9")]:
      with self.subTest(ratio=ratio):
        with self.assertRaises(ValueError) as cm:
          resolve_krea2_resolution(ratio, "1k")
        message = str(cm.exception)
        self.assertIn("quoted string", message)
        self.assertIn("969", message)

  def test_non_string_size(self):
    for size in [1, 2.0, None]:
      with self.subTest(size=size):
        with self.assertRaises(ValueError) as cm:
          resolve_krea2_resolution("1:1", size)
        self.assertIn("quoted string", str(cm.exception))
    with self.assertRaises(ValueError):
      list_krea2_resolutions(2)


class Krea2ParsePrecompileTest(unittest.TestCase):

  def test_empty(self):
    for spec in ["", "   ", ",", " , ,"]:
      with self.subTest(spec=spec):
        self.assertEqual(parse_krea2_precompile(spec), [])

  def test_all(self):
    self.assertEqual(parse_krea2_precompile("all"), list_krea2_resolutions())
    self.assertEqual(parse_krea2_precompile(" ALL "), list_krea2_resolutions())

  def test_size(self):
    self.assertEqual(parse_krea2_precompile("2k"), list_krea2_resolutions("2k"))
    self.assertEqual(parse_krea2_precompile("1K"), list_krea2_resolutions("1k"))

  def test_single(self):
    self.assertEqual(parse_krea2_precompile("2k@16:9"), [resolve_krea2_resolution("16:9", "2k")])
    self.assertEqual(parse_krea2_precompile(" 2K @ 16 : 9 "), [resolve_krea2_resolution("16:9", "2k")])

  def test_order_and_dedup(self):
    result = parse_krea2_precompile("2k@16:9, all")
    self.assertEqual(len(result), 22)
    self.assertEqual(result[0], resolve_krea2_resolution("16:9", "2k"))
    self.assertEqual(result[1:11], list_krea2_resolutions("1k")[:10])
    self.assertEqual(len(set(result)), 22)

    result = parse_krea2_precompile("1k@9:16,,2k@1:1, 1k@9:16 ,1k")
    labels = [r.label for r in result]
    self.assertEqual(labels[:3], ["1k@9:16", "2k@1:1", "1k@1:1"])
    self.assertEqual(len(labels), 12)
    self.assertEqual(len(set(labels)), 12)
    self.assertEqual(set(labels), {r.label for r in list_krea2_resolutions("1k")} | {"2k@1:1"})

    self.assertEqual(parse_krea2_precompile("1k,1k,1k"), list_krea2_resolutions("1k"))

  def test_invalid_entries(self):
    for spec, bad in [
        ("4k", "4k"),
        ("2k@16x9", "2k@16x9"),
        ("2k@", "2k@"),
        ("@16:9", "@16:9"),
        ("2k@16:9@1", "2k@16:9@1"),
        ("16:9", "16:9"),
        ("1k, everything", "everything"),
        ("2k@16:9, 3k@1:1", "3k@1:1"),
    ]:
      with self.subTest(spec=spec):
        with self.assertRaises(ValueError) as cm:
          parse_krea2_precompile(spec)
        message = str(cm.exception)
        self.assertIn(repr(bad), message)
        self.assertIn("'all'", message)
        self.assertIn("'2k'", message)
        self.assertIn("'2k@16:9'", message)

  def test_non_string_spec(self):
    for spec in [None, 2, ["all"]]:
      with self.subTest(spec=spec):
        with self.assertRaises(ValueError):
          parse_krea2_precompile(spec)


if __name__ == "__main__":
  unittest.main()
