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

# Named Krea 2 output resolutions: an API-like `aspect_ratio` + `image_size`
# pair (e.g. "16:9" + "2k") mapped to a fixed height/width, and the parser of
# the precompile list ("all", "2k", "2k@16:9", ...). Pure Python, no jax.
#
# Height and width are multiples of 16 (the transformer sees (H/16)*(W/16)
# image tokens). The 5:4, 16:9 and 21:9 entries are approximations chosen so
# that they share the 4032-token (1k) / 16128-token (2k) transformer
# executable; per size there are only 4 distinct token counts.

import dataclasses
import re
from typing import List, Optional

KREA2_IMAGE_SIZES = ("1k", "2k")
KREA2_DEFAULT_IMAGE_SIZE = "1k"
KREA2_DEFAULT_ASPECT_RATIO = "1:1"
KREA2_ASPECT_RATIOS = ("1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16", "9:21")

# Landscape (and square) 1k entries as (width, height). A portrait ratio is the
# landscape entry with width and height swapped; 2k is exactly twice 1k.
_LANDSCAPE_1K_WIDTH_HEIGHT = {
    "1:1": (1024, 1024),
    "5:4": (1152, 896),
    "4:3": (1152, 864),
    "3:2": (1248, 832),
    "16:9": (1344, 768),
    "21:9": (1536, 672),
}
_PORTRAIT_TO_LANDSCAPE = {"4:5": "5:4", "3:4": "4:3", "2:3": "3:2", "9:16": "16:9", "9:21": "21:9"}
_SIZE_SCALE = {"1k": 1, "2k": 2}

_PRECOMPILE_FORMS = (
    "'all' (every resolution), a size such as '2k' (every ratio of that size) or '<size>@<ratio>' such as '2k@16:9'"
)


@dataclasses.dataclass(frozen=True)
class Krea2Resolution:
  """One named output resolution."""

  image_size: str  # one of KREA2_IMAGE_SIZES
  aspect_ratio: str  # one of KREA2_ASPECT_RATIOS
  height: int
  width: int

  @property
  def image_tokens(self) -> int:
    """Number of image tokens the transformer sees: (H/16)*(W/16)."""
    return (self.height // 16) * (self.width // 16)

  @property
  def label(self) -> str:
    """Short name such as "2k@16:9"."""
    return f"{self.image_size}@{self.aspect_ratio}"


def _require_string(value, name: str, example: str) -> str:
  if not isinstance(value, str):
    raise ValueError(
        f"{name} must be a quoted string such as '{example}', got {type(value).__name__} {value!r}. "
        "In YAML an unquoted ratio like 16:9 is parsed as the integer 969; write '16:9' instead."
    )
  return value


def _normalize_image_size(image_size) -> str:
  size = _require_string(image_size, "krea2_image_size", "2k").strip().lower()
  if size not in KREA2_IMAGE_SIZES:
    raise ValueError(f"Unknown krea2_image_size {image_size!r}; valid values: {', '.join(KREA2_IMAGE_SIZES)}.")
  return size


def _normalize_aspect_ratio(aspect_ratio) -> str:
  ratio = re.sub(r"\s*:\s*", ":", _require_string(aspect_ratio, "krea2_aspect_ratio", "16:9").strip())
  if ratio not in KREA2_ASPECT_RATIOS:
    raise ValueError(f"Unknown krea2_aspect_ratio {aspect_ratio!r}; valid values: {', '.join(KREA2_ASPECT_RATIOS)}.")
  return ratio


def resolve_krea2_resolution(aspect_ratio, image_size=KREA2_DEFAULT_IMAGE_SIZE) -> Krea2Resolution:
  """Maps an aspect ratio ("16:9") and an image size ("2k") to a resolution.

  Both names are stripped; the size is case-insensitive and whitespace around
  the colon of the ratio is ignored ("16 : 9"). Unknown or non-string values
  raise ValueError.
  """
  ratio = _normalize_aspect_ratio(aspect_ratio)
  size = _normalize_image_size(image_size)
  if ratio in _PORTRAIT_TO_LANDSCAPE:
    height, width = _LANDSCAPE_1K_WIDTH_HEIGHT[_PORTRAIT_TO_LANDSCAPE[ratio]]
  else:
    width, height = _LANDSCAPE_1K_WIDTH_HEIGHT[ratio]
  scale = _SIZE_SCALE[size]
  return Krea2Resolution(image_size=size, aspect_ratio=ratio, height=height * scale, width=width * scale)


def list_krea2_resolutions(image_size: Optional[str] = None) -> List[Krea2Resolution]:
  """Every resolution in table order (sizes, then KREA2_ASPECT_RATIOS), or only
  those of `image_size` when it is given."""
  sizes = KREA2_IMAGE_SIZES if image_size is None else (_normalize_image_size(image_size),)
  return [resolve_krea2_resolution(ratio, size) for size in sizes for ratio in KREA2_ASPECT_RATIOS]


def parse_krea2_precompile(spec) -> List[Krea2Resolution]:
  """Parses a comma-separated precompile list into resolutions.

  Each entry is 'all' (every resolution), a size ('2k': every ratio of that
  size) or '<size>@<ratio>' ('2k@16:9'). Empty entries are ignored, an empty
  or whitespace-only spec gives []. The result keeps first-occurrence order
  without duplicates. Anything else raises ValueError.
  """
  if not isinstance(spec, str):
    raise ValueError(f"krea2_precompile must be a string, got {type(spec).__name__} {spec!r}.")
  result = []
  seen = set()
  for raw_entry in spec.split(","):
    entry = raw_entry.strip()
    if not entry:
      continue
    try:
      if entry.lower() == "all":
        resolutions = list_krea2_resolutions()
      elif "@" in entry:
        size, _, ratio = entry.partition("@")
        if "@" in ratio:
          raise ValueError("more than one '@'")
        resolutions = [resolve_krea2_resolution(ratio, size)]
      else:
        resolutions = list_krea2_resolutions(entry)
    except ValueError as e:
      raise ValueError(f"Invalid krea2_precompile entry {entry!r} ({e}); accepted forms: {_PRECOMPILE_FORMS}.") from e
    for resolution in resolutions:
      if resolution not in seen:
        seen.add(resolution)
        result.append(resolution)
  return result
