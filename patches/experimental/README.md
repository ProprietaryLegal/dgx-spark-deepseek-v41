# Experimental patches

Measured, kept with their tests as a record, and left out of the release image: `scripts/image/build_image.sh` applies
every numbered patch in `patches/` (0001-0008, 0010-0021), not the ones here.

| patch | what | why it stays out |
|---|---|---|
| 0009 | FULL CUDA graphs split only at the TP all-reduces, which run eagerly (`VLLM_FULL_GRAPH_EAGER_ALLREDUCE=1`) | slower on 2x GB10: the serve pays no in-graph NCCL penalty to win back |
| 0022 | DSpark verify rows through the plugin's Triton sparse attention instead of FlashInfer (`VLLM_DSV41_TRITON_VERIFY=1`) | flat end to end: 2.8x faster on its own, but the verify window is bound by weight bytes, not attention |

To try one, layer it on the release image (the overlay build looks here too):

```bash
BASE=dsv41-serve:v4 TAG=dsv41-serve:exp PATCHES="0022" ./scripts/image/build_serve_overlay.sh
IMAGE=dsv41-serve:exp DOCKER_ENV="VLLM_DSV41_TRITON_VERIFY=1" MODEL_DIR=... ./scripts/launch/dsv41_tp2.sh up
```

`test_full_graph_eager_allreduce.py` runs in an image that carries 0009.
