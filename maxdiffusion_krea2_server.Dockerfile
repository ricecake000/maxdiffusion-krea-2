# Krea 2 Turbo HTTP API image (maxdiffusion.serve_krea2, contract in docs/krea2_api.md).
#
# Build on top of the dependency image, e.g.:
#   docker build -f maxdiffusion_krea2_server.Dockerfile --build-arg BASEIMAGE=maxdiffusion_base_image -t krea2-api .
# Run (TPU VM; mount the model, weight cache and AOT cache from persistent storage):
#   docker run --privileged --net=host -e KREA2_API_TOKEN=... -e KREA2_MODEL_PATH=/models/Krea-2-Turbo \
#     -e KREA2_CONFIG=/deps/src/maxdiffusion/configs/base_krea2_turbo_v6e1.yml \
#     -e KREA2_CONFIG_OVERRIDES="aot_cache_dir=/cache/aot aot_cache_lazy_load=True krea2_weight_cache_dir=/cache/weights" \
#     -v /mnt/krea2:/models -v /mnt/cache:/cache krea2-api

ARG BASEIMAGE=maxdiffusion_base_image
FROM $BASEIMAGE

WORKDIR /deps
COPY . .

# Refresh the local package so the server entry point and current source tree match this image. Runtime
# dependencies (fastapi, uvicorn, Pillow, ...) are installed in the base image.
RUN python3 -m uv pip install --system --no-deps .

# Fail the image build early if this platform's Pillow wheel lacks AVIF (the default response_format).
RUN python3 -c "from PIL import Image; Image.init(); assert Image.registered_extensions().get('.avif') == 'AVIF'"
RUN python3 -c "import fastapi, uvicorn"

ENV KREA2_HOST=0.0.0.0
ENV KREA2_PORT=8000
ENV KREA2_OUTPUT_DIR=/tmp/krea2-api

EXPOSE 8000

CMD ["krea2-api"]
