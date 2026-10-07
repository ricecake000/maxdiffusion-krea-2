# Krea 2 Turbo console

A static, dependency-free web client for the API in
[`docs/krea2_api.md`](../docs/krea2_api.md). Three files, no build step, no
package manager: `index.html`, `styles.css`, `app.js`.

## Run it

Start the API first (see [`docs/krea2_api.md`](../docs/krea2_api.md) for every
variable):

```bash
export KREA2_API_TOKEN='replace-with-a-long-random-token'
export KREA2_CONFIG=src/maxdiffusion/configs/base_krea2_turbo_v6e1.yml
export KREA2_MODEL_PATH=/path/to/Krea-2-Turbo
export KREA2_CONFIG_OVERRIDES='aot_cache_dir=/path/to/aot aot_cache_lazy_load=True krea2_weight_cache_dir=/path/to/weights'
export KREA2_SERVE_PRESETS=all            # or e.g. 1k,2k@16:9
export KREA2_SERVE_TEXT_TOKENS=128,256
python -m maxdiffusion.serve_krea2        # or: krea2-api
```

`/readyz` answers 503 until every preset is compiled or loaded from the AOT
cache; the status dot shows "warming up" meanwhile.

Then serve this directory on a port the API already allows:

```bash
python3 -m http.server 5173 --directory frontend
```

Open <http://localhost:5173>. Use `localhost`, not `127.0.0.1` — the origin
must match an entry in `KREA2_ALLOWED_ORIGINS`, whose default is
`http://localhost:3000,http://localhost:5173`.

Opening `index.html` from the filesystem does not work: a `file://` page sends
the opaque origin `null`, which the server's CORS allowlist rejects.

On first load, click **Endpoint**, enter the server address and the bearer
token, and save. Both are kept in this browser's local storage.

## What it does

- Builds the size and aspect-ratio selector from `GET /v1/presets`: one size
  switch per served `image_size`, and the warmed ratios of that size with
  their pixel dimensions. The note under it shows the largest warmed text
  bucket, the queue limit, and the timeout.
- Offers the formats the server enabled (`avif` small, `webp` fast, `jpeg`
  compatible, `png` lossless), preselecting the server's default.
- Sends `POST /v1/images/generations` with `prompt`, `aspect_ratio`,
  `image_size`, `response_format`, and an optional `seed`, one request at a
  time from this page.
- Renders the returned image from a blob URL, shows its MIME type, and offers
  it as a download with the matching extension.
- Draws the `timing` object as a proportional pipeline bar: queue wait (when
  over 40 ms), text encoding, denoising, VAE decode, image encode. While a
  request is in flight the bar animates against the previous request of the
  same size, then snaps to the measured timings when the response lands.
- Polls `GET /readyz` for the status dot, including how many generations are
  in flight on the server.
- Explains `429` (queue full, with the server's `Retry-After`) and `504`
  (timeout) answers, and reloads the preset list after a `422`.
- Keeps up to 24 images from the session in a strip, each labelled with its
  seed, preset, and format and carrying its own timing meter. **Use these
  settings** restores an image's prompt, preset, format, and seed for a
  reproducible re-run.

`⌘`/`Ctrl` + `Return` in the prompt box generates.

## Notes

- The token is a bearer credential in browser storage. Keep this page on a
  machine you control, and put the API behind HTTPS before exposing its port.
- LoRA is not a request option: whatever `lora_config` the server's config
  carries applies to every request.
- Current Chrome, Firefox, and Safari decode AVIF; if a browser cannot, the
  page says so, and WebP or JPEG is one click away.
