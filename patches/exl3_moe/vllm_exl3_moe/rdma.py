# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel all-reduce over RoCE RDMA writes for decode-size bf16 tensors (csrc/exl3_rdma.cu).

vLLM's TP all-reduce between two GB10 nodes goes through NCCL's network proxy: ~25 us median per decode all-reduce in
our serve, ~96 a decode step. Here a staging kernel puts the tensor in registered host memory and rings a doorbell, a
pinned CPU thread RDMA-writes it into the peer's receive ring, and a summing kernel waits for the peer's flag and adds
the ranks' parts in rank order (fp32, one rounding), so both ranks hold the same bits. The kernel pair captures in CUDA
graphs like any other. The design, the rings and the proxy are TensorFold's one-hop gather (csrc/NOTICE.tensorfold);
the sum and the vLLM hook are ours.

EXL3_RDMA_AR=1 turns it on for TP groups of 2+ ranks (default off until measured): bf16 tensors up to
EXL3_RDMA_AR_MAX_KB (256) go over RDMA, everything else stays on NCCL. Devices: EXL3_RDMA_DEVICES (comma list), else
NCCL_IB_HCA's list; GID index: EXL3_RDMA_GID, else NCCL_IB_GID_INDEX; EXL3_RDMA_CPU pins the proxy thread to a core (default: the last).
EXL3_RDMA_FLAGW=1 (the same on every rank): each part's data write is chained with an 8-byte flag write the peer's GPU
polls, so no receiving thread is on the path (csrc/exl3_rdma.cu, P2 stage 0; default off until tested on both nodes).
The setup is collective: the ranks exchange their queue pairs over the group's CPU group on the first eligible
all-reduce outside a graph capture, and if any rank cannot open RDMA, every rank stays on NCCL.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_BYTES = int(os.environ.get("EXL3_RDMA_AR_MAX_KB") or 256) * 1024
SLOTS = int(os.environ.get("EXL3_RDMA_AR_SLOTS") or 4)
FLAGW = os.environ.get("EXL3_RDMA_FLAGW", "0") == "1"

_ext = None
_rings: dict[str, "RdmaAllReduce | None"] = {}      # communicator unique_name -> ring (None: NCCL for good)


def enabled() -> bool:
    return os.environ.get("EXL3_RDMA_AR", "0") == "1"


def ext():
    global _ext
    if _ext is None:
        from .jit import load
        _ext = load("exl3_rdma", "EXL3_RDMA_FLAGS", ldflags=["-libverbs"])
    return _ext


def device_names() -> list[str]:
    named = os.environ.get("EXL3_RDMA_DEVICES") or os.environ.get("NCCL_IB_HCA") or ""
    return [d.strip().lstrip("=^").split(":")[0] for d in named.split(",") if d.strip()]


class RdmaAllReduce:
    def __init__(self, cpu_group, rank: int, world: int):
        import torch.distributed as dist
        e = ext()
        # EXL3_RDMA_GID overrides NCCL's index: on our Sparks only rocep1s0f0 carries the IPv4 GID (index 3), while
        # index 1 (RoCE v2, link-local) exists on both PCIe functions of the port, so two devices need 1
        gid = int(os.environ.get("EXL3_RDMA_GID") or os.environ.get("NCCL_IB_GID_INDEX") or 3)
        cpu = int(os.environ.get("EXL3_RDMA_CPU") or (os.cpu_count() or 1) - 1)
        devices = device_names()
        self.rank, self.world = rank, world
        error, info = "", None
        try:
            if not devices:
                raise ValueError("no RDMA device named (EXL3_RDMA_DEVICES / NCCL_IB_HCA)")
            self.h = e.create(rank, world, MAX_BYTES, SLOTS, devices, gid, cpu, int(FLAGW))
            info = e.local_info(self.h)
        except Exception as exc:  # noqa: BLE001  (every rank refuses together)
            error = f"{type(exc).__name__}: {exc}"
        infos = [None] * world
        dist.all_gather_object(infos, (error, info, devices), group=cpu_group)
        bad = [f"rank {r}: {x[0]}" for r, x in enumerate(infos) if x[0]]
        if not bad and len({len(x[2]) for x in infos}) != 1:
            bad = ["ranks open different device counts"]
        if bad:
            raise RuntimeError("; ".join(bad))
        try:
            e.connect(self.h, [infos[r][1][rank] for r in range(world)])
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        ups = [None] * world
        dist.all_gather_object(ups, error, group=cpu_group)    # every QP can receive before anyone writes
        bad = [f"rank {r}: {u}" for r, u in enumerate(ups) if u]
        if bad:
            raise RuntimeError("; ".join(bad))
        e.start(self.h)
        import atexit
        atexit.register(e.stop, self.h)
        logger.info("EXL3 RDMA all-reduce: rank %d of %d on %s, gid %d, proxy on cpu %d, up to %d KB",
                    rank, world, ",".join(devices), gid, cpu, MAX_BYTES // 1024)

    @staticmethod
    def fits(x: torch.Tensor) -> bool:
        # dtype and size only: the ranks must take the same path for every call (alignment is fixed up below)
        return x.dtype == torch.bfloat16 and x.is_cuda and x.numel() % 8 == 0 and 0 < x.numel() * 2 <= MAX_BYTES

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous() or x.data_ptr() % 16:
            x = x.contiguous().clone()
        out = torch.empty_like(x)
        ext().all_reduce(self.h, x, out)
        return out


def ring_for(comm) -> RdmaAllReduce | None:
    """The ring of a vLLM device communicator (built on its first eligible call), or None (NCCL)."""
    name = getattr(comm, "unique_name", "") or ""
    if name in _rings:
        return _rings[name]
    if not name.startswith("tp") or getattr(comm, "world_size", 1) < 2:
        _rings[name] = None
        return None
    if torch.cuda.is_current_stream_capturing():
        return None                                   # set up on an eager call (both ranks reach the same one)
    try:
        r = RdmaAllReduce(comm.cpu_group, comm.rank_in_group, comm.world_size)
    except Exception as exc:  # noqa: BLE001
        logger.warning("EXL3 RDMA all-reduce off for %s, NCCL stays: %s", name, exc)
        r = None
    _rings[name] = r
    return r


def install() -> None:
    """Route vLLM's CUDA communicator all-reduce through the ring for the tensors it fits."""
    if not enabled():
        return
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    if getattr(CudaCommunicator.all_reduce, "_exl3_rdma", False):
        return
    orig = CudaCommunicator.all_reduce

    def all_reduce(self, input_):
        if RdmaAllReduce.fits(input_):
            r = ring_for(self)
            if r is not None:
                return r.all_reduce(input_)
        return orig(self, input_)

    all_reduce._exl3_rdma = True
    CudaCommunicator.all_reduce = all_reduce
    logger.info("EXL3 RDMA all-reduce: vLLM's TP all-reduce routed for bf16 tensors up to %d KB", MAX_BYTES // 1024)
