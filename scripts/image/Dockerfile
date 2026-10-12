# dsv41-recipe runtime image: upstream vLLM (main 6e517b15) built for GB10 + our patch series.
# Build context = a staging dir holding wheels/ and patches/ (see build_image.sh).
ARG BASE=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
FROM ${BASE}
RUN command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq --no-install-recommends git && rm -rf /var/lib/apt/lists/*)
COPY wheels/ /tmp/wheels/
RUN python3 -m pip install --no-deps --force-reinstall /tmp/wheels/vllm-*.whl && rm -rf /tmp/wheels
COPY patches/ /opt/dsv41/patches/
# Python-only patches go straight into site-packages (vllm/ paths only; tests are not installed).
RUN SITE=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))") && cd "$SITE" && \
    for p in /opt/dsv41/patches/[0-9][0-9][0-9][0-9]-*.patch; do \
      git apply --verbose --include='vllm/*' -p1 "$p" || exit 1; done && \
    python3 -c "import vllm.models.deepseek_v41.common.engram as e; print('engram module ok')"
# Prebuild the Engram disk reader so workers do not JIT it at startup.
ENV TORCH_EXTENSIONS_DIR=/opt/torch_extensions
RUN python3 -c "from vllm.models.deepseek_v41.common.engram import _load_engram_disk_ext as l; m = l(); print('engram_disk built:', m.EngramDiskTable)"
LABEL org.opencontainers.image.title="dsv41-recipe" org.opencontainers.image.description="vLLM main 6e517b15 for GB10 + the dsv41 patch series (patches/0001-...)"
