# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin: EXL3 trellis routed experts for DeepSeek-V4.1-Flash.

Target: upstream vLLM main @ 6e517b15c1833cf72a7f557ee32524d98682e617 + exllamav3 v1.6.0 kernels
(turboderp-org/exllamav3 @ 151539c77abc7ab7425d30da7a4e8e3c5c154e7b, MIT; called, not copied).

Entry point (pyproject.toml): [project.entry-points."vllm.general_plugins"] exl3_moe = "vllm_exl3_moe:register".
vLLM runs it in the API process, the engine core and every worker (vllm/plugins/__init__.py:77-91,
engine/arg_utils.py:891-895, v1/worker/worker_base.py:274-276) before the model config is verified,
so the `deepseek_v41_exl3` quant_method in config.json resolves to our config class.

Importing this package needs neither torch nor vLLM (the plan math in .plan is pure Python);
register() is a logged no-op when vLLM is missing.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0.dev0"
QUANT_METHOD = "deepseek_v41_exl3"

_logger = logging.getLogger(__name__)
_registered = False


def _pin_cpus() -> None:
    """EXL3_CPU_PIN="5-9,15-19": pin every thread of this process, and the ones it starts later, to those CPUs (on
    GB10 the Cortex-X925 cores: 5-9 and 15-19 at 3.9 GHz; 0-4 and 10-14 are Cortex-A725 at 2.8 GHz). Unpinned, the
    scheduler can run a TP worker's launch thread on an A725 core for a whole request: set b showed two round-time
    modes ~5% apart with the same text and depth choices, and spark2's worker thread was found on CPU 13 (LOG Oct 11
    19:39-19:45). Default off."""
    import os

    spec = os.environ.get("EXL3_CPU_PIN", "").strip()
    if not spec or not hasattr(os, "sched_setaffinity"):
        return
    cpus: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    use = cpus & os.sched_getaffinity(0)
    if not use:
        _logger.warning("vllm_exl3_moe: EXL3_CPU_PIN=%s matches none of this process's CPUs", spec)
        return
    n = 0
    for tid in os.listdir("/proc/self/task"):           # sched_setaffinity(0) is the calling thread only
        try:
            os.sched_setaffinity(int(tid), use)
            n += 1
        except OSError:
            pass
    _logger.info("vllm_exl3_moe: pid %d, %d threads pinned to CPUs %s (EXL3_CPU_PIN)", os.getpid(), n, sorted(use))


def register() -> None:
    global _registered
    if _registered:
        return
    _pin_cpus()
    try:
        from .config import register_config
    except ImportError as e:  # vLLM (or torch) not installed: nothing to register into
        _logger.warning("vllm_exl3_moe: vLLM not importable (%s); plugin inactive", e)
        return
    register_config()
    # Our modules log through vLLM's init_logger under "vllm_exl3_moe.*", outside the "vllm" hierarchy that vLLM
    # configures: hand that tree vLLM's handlers so plugin INFO lines (head path, RDMA ring, L2 prefetch) reach the log.
    vllm_logger = logging.getLogger("vllm")
    ours = logging.getLogger(__name__)
    if vllm_logger.handlers and not ours.handlers:
        for h in vllm_logger.handlers:
            ours.addHandler(h)
        ours.setLevel(vllm_logger.level or logging.INFO)
        ours.propagate = False
    from .rdma import install as install_rdma     # EXL3_RDMA_AR=1: TP all-reduces of decode tensors over RDMA writes
    install_rdma()
    from .router import install as install_router  # EXL3_ROUTER=1: decode-size MoE gate GEMMs on exl3_router
    install_router()
    _registered = True


__all__ = ["QUANT_METHOD", "register", "__version__"]
