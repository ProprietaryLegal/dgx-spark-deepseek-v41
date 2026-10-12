# SPDX-License-Identifier: Apache-2.0
"""DeepSeek's own reference implementation (the checkpoint's `inference/` dir, MIT) run one block at a time on a GB10.

Nothing is copied from it: the module files are imported from the checkpoint directory, and three things are adapted
for this box and for layer streaming:

1. `sparse_attn` is called per 16-head group. The reference kernel keeps all heads of a query in shared memory
   (141,312 B at 64 heads x 512 dims); GB10 (sm_121) allows ~99 KB per block. Heads are independent and the attention
   sink is per head, so the split is exact (`scripts/quantize/tilelang_gb10_probe.py` measured the rest of kernel.py
   on GB10: act_quant, fp8_gemm, fp4_gemm, fp4_act_quant, hc_split_sinkhorn all correct with apache-tvm-ffi 0.1.4-0.1.6).
2. The Engram table stays on disk: `DiskEngramEmbedding` replaces `ParallelEngramEmbedding` and reads only the rows a
   batch hashes to, straight from the safetensors file (np.memmap), dequantized exactly like the reference forward.
3. Blocks are built and loaded one at a time from the HF shards (`load_block`), in the reference's own parameter
   formats (the same conversions as `inference/convert.py` at model-parallel 1: wo_a dequantized to bf16, routed
   expert int8 viewed as float4_e2m1fn_x2).
"""
from __future__ import annotations

import importlib
import json
import os
import struct
import sys
import time
from functools import lru_cache

import numpy as np
import torch

_REF = None


def load_reference(ckpt_dir: str):
    """Import the checkpoint's inference/{kernel,model,engram}.py once, with the GB10 sparse_attn shim installed."""
    global _REF
    if _REF is not None:
        return _REF
    inf = os.path.join(ckpt_dir, "inference")
    if inf not in sys.path:
        sys.path.insert(0, inf)
    kernel = importlib.import_module("kernel")
    model = importlib.import_module("model")
    engram = importlib.import_module("engram")

    orig = kernel.sparse_attn

    def sparse_attn_headsplit(q, kv, attn_sink, topk_idxs, softmax_scale, group=16):
        h = q.size(2)
        if h <= group:
            return orig(q, kv, attn_sink, topk_idxs, softmax_scale)
        outs = [orig(q[:, :, i:i + group].contiguous(), kv, attn_sink[i:i + group].contiguous(), topk_idxs,
                     softmax_scale) for i in range(0, h, group)]
        return torch.cat(outs, dim=2)

    kernel.sparse_attn = sparse_attn_headsplit
    model.sparse_attn = sparse_attn_headsplit   # model.py imported the name directly

    # DSV41_REF_FP8_LINEAR=bf16: FP8 (non-expert) linears as bf16 dequant x bf16 activation, the way vLLM's
    # EmulationMxfp8LinearKernel computes them, instead of the reference's FP8-activation fp8_gemm. Diagnostic only.
    if os.environ.get("DSV41_REF_FP8_LINEAR") == "bf16":
        ref_linear = model.linear

        def linear_bf16(x, weight, bias=None):
            if weight.dtype == torch.float8_e4m3fn:
                s = weight.scale
                w = (weight.float().unflatten(0, (s.size(0), -1)).unflatten(-1, (s.size(1), -1))
                     * s.float()[:, None, :, None]).flatten(2, 3).flatten(0, 1)
                return torch.nn.functional.linear(x.to(torch.bfloat16), w.to(torch.bfloat16))
            return ref_linear(x, weight, bias)

        model.linear = linear_bf16
    model.world_size, model.rank = 1, 0
    _REF = (model, kernel, engram)
    return _REF


def model_args(ckpt_dir: str, max_batch_size: int, max_seq_len: int):
    model, _, _ = load_reference(ckpt_dir)
    with open(os.path.join(ckpt_dir, "inference", "config.json")) as f:
        cfg = json.load(f)
    args = model.ModelArgs(**cfg)
    args.max_batch_size, args.max_seq_len = max_batch_size, max_seq_len
    model.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    return args


def drop_page_cache(path: str) -> None:
    """Evict one file's pages from the page cache (written pages are flushed first). On GB10, CUDA allocations do not
    reclaim page cache (measured Oct 8: 45 GB "available", 8 GB free, a 67 MB cudaMalloc failed), so every big file the
    harness reads or writes is evicted once used: the stream needs that memory more than a cache it never re-reads."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


class Checkpoint:
    """Tensor name -> shard file + byte offsets, from the safetensors index and headers (no tensor data read)."""

    def __init__(self, ckpt_dir: str):
        self.dir = ckpt_dir
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._headers = {}

    def header(self, fname: str):
        h = self._headers.get(fname)
        if h is None:
            with open(os.path.join(self.dir, fname), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                meta = json.loads(f.read(n))
            meta.pop("__metadata__", None)
            h = self._headers[fname] = (8 + n, meta)
        return h

    def locate(self, name: str):
        """(path, absolute byte offset, dtype string, shape) of one tensor."""
        fname = self.weight_map[name]
        base, meta = self.header(fname)
        t = meta[name]
        return os.path.join(self.dir, fname), base + t["data_offsets"][0], t["dtype"], t["shape"]

    def names(self, prefix: str):
        return sorted(k for k in self.weight_map if k.startswith(prefix))

    def get(self, name: str, device="cpu") -> torch.Tensor:
        return self.get_many([name], device)[name]

    _DT = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "BF16": torch.bfloat16,
           "F16": torch.float16, "F32": torch.float32, "I8": torch.int8, "U8": torch.uint8, "I32": torch.int32,
           "I64": torch.int64}

    def get_many(self, names, device="cpu", gap: int = 64 << 20) -> dict:
        """Read several tensors with large sequential reads: per shard, the wanted byte ranges are merged into spans
        (gaps up to `gap` read through) and each span lands in one reusable pinned buffer with a single readinto.
        safetensors' own get_tensor is mmap-backed, so a GPU copy from it page-faults 4 KB at a time (~150 MB/s on a
        cold page cache, 47 s per 7 GiB layer); this reads at NVMe speed. Returned tensors are on `device`; on "cpu"
        they are clones (the pinned buffer is reused by the next call)."""
        by_file = {}
        for n in names:
            by_file.setdefault(self.weight_map[n], []).append(n)
        out = {}
        for fname, ns in by_file.items():
            base, meta = self.header(fname)
            ents = sorted(((base + meta[n]["data_offsets"][0], base + meta[n]["data_offsets"][1], n) for n in ns))
            spans, cur = [], None
            for s0, s1, n in ents:
                if cur and s0 - cur[1] <= gap:
                    cur[1] = max(cur[1], s1)
                    cur[2].append((s0, s1, n))
                else:
                    cur = [s0, s1, [(s0, s1, n)]]
                    spans.append(cur)
            with open(os.path.join(self.dir, fname), "rb", buffering=0) as f:
                for s0, s1, members in spans:
                    buf = self._pinned(s1 - s0)
                    f.seek(s0)
                    mv, got = memoryview(buf.numpy()), 0
                    while got < s1 - s0:
                        r = f.readinto(mv[got: s1 - s0])
                        if not r:
                            raise IOError(f"short read {fname} @ {s0 + got}")
                        got += r
                    for t0, t1, n in members:
                        m = meta[n]
                        t = buf[t0 - s0: t1 - s0].view(self._DT[m["dtype"]]).view(m["shape"])
                        out[n] = t.to(device, non_blocking=True) if str(device) != "cpu" else t.clone()
                    if str(device) != "cpu":
                        torch.cuda.current_stream().synchronize()     # the buffer is reused by the next span
                    os.posix_fadvise(f.fileno(), s0, s1 - s0, os.POSIX_FADV_DONTNEED)   # see drop_page_cache
        return out

    def _pinned(self, nbytes: int) -> torch.Tensor:
        b = getattr(self, "_buf", None)
        if b is None or b.numel() < nbytes:
            self._buf = b = torch.empty(max(nbytes, 1 << 30), dtype=torch.uint8, device="cpu", pin_memory=True)
        return b


class DiskEngramEmbedding(torch.nn.Module):
    """Drop-in for the reference ParallelEngramEmbedding (world_size 1): rows read from the safetensors file on demand
    by our engram_disk reader (patches/0002: a C++ pool of parallel preads; np.memmap page-faulted one 4 KB page at a
    time: 225 s for 1.6M rows). forward(indices [..]) -> bf16 [.., head_dim], identical math to the reference."""

    def __init__(self, ckpt: Checkpoint, layer_id: int, block_size: int = 32, threads: int = 48):
        super().__init__()
        wpath, woff, wdt, wshape = ckpt.locate(f"layers.{layer_id}.engram.embed.weight")
        spath, soff, sdt, sshape = ckpt.locate(f"layers.{layer_id}.engram.embed.scale")
        assert wdt == "F8_E4M3" and sdt == "F8_E8M0", (wdt, sdt)
        self.rows, self.dim = wshape
        self.sdim = sshape[1]
        self.block_size = block_size
        from vllm.models.deepseek_v41.common.engram import _load_engram_disk_ext
        ext = _load_engram_disk_ext()
        self.table = ext.EngramDiskTable(wpath, woff, spath, soff, 0, self.rows, self.dim, self.sdim, 0, threads)

    @torch.inference_mode()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        dev = indices.device
        uniq, inv = torch.unique(indices.reshape(-1), return_inverse=True)   # sorted rows: friendlier to the disk
        n = uniq.numel()
        ids = uniq.to("cpu", torch.int64).pin_memory()
        w = torch.empty(n, self.dim, dtype=torch.uint8, device="cpu").pin_memory()
        s = torch.empty(n, self.sdim, dtype=torch.uint8, device="cpu").pin_memory()
        self.table.read_rows(ids.data_ptr(), w.data_ptr(), s.data_ptr(), n)
        w = w.to(dev).view(torch.float8_e4m3fn)
        s = s.to(dev).view(torch.float8_e8m0fnu)
        vals = w.float().unflatten(-1, (-1, self.block_size)) * s.float().unsqueeze(-1)
        vals = vals.flatten(-2).to(torch.bfloat16)
        return vals[inv].view(*indices.shape, self.dim)


def _dequant_block_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    ob, ib = w.size(0) // s.size(0), w.size(1) // s.size(1)
    x = w.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float() * s[:, None, :, None].float()
    return x.flatten(2, 3).flatten(0, 1).bfloat16()


def load_block(ckpt: Checkpoint, args, layer_id: int, engram_layout=None, device="cuda", experts: bool = True):
    """Build reference Block `layer_id` on `device` and fill it from the HF shards. With experts=False the routed
    experts stay unallocated placeholders (for passes that only need attention / routing)."""
    model, _, _ = load_reference(ckpt.dir)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    # Never allocate the Engram table: the reference ParallelEngramEmbedding would create a 98 GB fp8 parameter (+3 GB
    # of scales) on the device before DiskEngramEmbedding replaces it, enough to wedge a 128 GB unified-memory box.
    saved_pee = model.ParallelEngramEmbedding
    model.ParallelEngramEmbedding = _NoTable
    t0 = time.time()
    try:
        with torch.device(device):
            if experts:
                blk = model.Block(layer_id, args, engram_layout)
            else:
                blk = _block_without_experts(model, layer_id, args, engram_layout)
    finally:
        model.ParallelEngramEmbedding = saved_pee
        torch.set_default_dtype(prev)
    t1 = time.time()
    prefix = f"layers.{layer_id}."
    want = []
    for name in ckpt.names(prefix):
        key = name[len(prefix):]
        if key.startswith("engram.embed."):
            continue                                              # disk-backed below
        if not experts and ".experts." in key and "shared_experts" not in key:
            continue
        want.append(name)
    sd = {name[len(prefix):]: t for name, t in ckpt.get_many(want, device).items()}
    t2 = time.time()
    if "attn.wo_a.weight" in sd:
        sd["attn.wo_a.weight"] = _dequant_block_fp8(sd["attn.wo_a.weight"], sd.pop("attn.wo_a.scale"))
    for key in list(sd):
        if ".experts." in key and "shared_experts" not in key and sd[key].dtype == torch.int8:
            sd[key] = sd[key].view(torch.float4_e2m1fn_x2)
    missing, unexpected = _load_state(blk, sd)
    if blk.engram is not None:
        blk.engram.embed = DiskEngramEmbedding(ckpt, layer_id)
        missing = [m for m in missing if not m.startswith("engram.embed.")]
    if not experts:
        missing = [m for m in missing if not (".experts." in m and "shared_experts" not in m)]
    if missing or unexpected:
        raise RuntimeError(f"layer {layer_id}: missing {missing[:8]}..., unexpected {unexpected[:8]}...")
    if os.environ.get("DSV41_LOAD_TIMING"):
        print(f"  load_block {layer_id}: build {t1 - t0:.1f}s, read {t2 - t1:.1f}s, copy {time.time() - t2:.1f}s "
              f"({sum(v.numel() * v.element_size() for v in sd.values()) / 2**30:.2f} GiB)", flush=True)
    return blk.eval()


class _NoTable(torch.nn.Module):
    """Placeholder for the reference ParallelEngramEmbedding during Block construction (no parameters)."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, indices):
        raise RuntimeError("Engram table placeholder: load_block installs DiskEngramEmbedding")


def _load_state(module: torch.nn.Module, sd: dict):
    """load_state_dict that keeps each parameter's dtype (fp8 / fp4 / e8m0 bytes are copied raw, never converted)."""
    params = dict(module.named_parameters())
    missing = [k for k in params if k not in sd]
    unexpected = [k for k in sd if k not in params]
    with torch.no_grad():
        for k, v in sd.items():
            p = params.get(k)
            if p is None:
                continue
            if p.shape != v.shape:
                raise RuntimeError(f"{k}: param {tuple(p.shape)} {p.dtype} vs checkpoint {tuple(v.shape)} {v.dtype}")
            if p.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2) or v.dtype in (
                    torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2):
                assert p.dtype == v.dtype, (k, p.dtype, v.dtype)
                p.data.view(torch.uint8).copy_(v.view(torch.uint8))
            else:
                p.data.copy_(v.to(p.dtype))
    return missing, unexpected


def _block_without_experts(model, layer_id, args, engram_layout):
    """A Block whose routed experts are None (no 6.7 GiB allocation); MoE.forward must not be called on it."""
    saved = model.MoE.__init__

    def init(self, layer_id_, args_):
        torch.nn.Module.__init__(self)
        n_routed, n_act = args_.get_moe_config(layer_id_)
        self.layer_id, self.dim = layer_id_, args_.dim
        self.n_routed_experts, self.n_local_experts, self.n_activated_experts = n_routed, n_routed, n_act
        self.experts_start_idx, self.experts_end_idx = 0, n_routed
        self.gate = model.Gate(layer_id_, args_)
        self.experts = torch.nn.ModuleList([None] * n_routed)
        self.shared_experts = model.Expert(args_.dim, args_.moe_inter_dim, swiglu_limit=args_.swiglu_limit)

    model.MoE.__init__ = init
    try:
        return model.Block(layer_id, args, engram_layout)
    finally:
        model.MoE.__init__ = saved


@lru_cache(1)
def engram_hasher(ckpt_dir: str, max_batch_size: int, max_seq_len: int, device="cuda"):
    """The reference NgramHashState (needs the tokenizer for the compressed token map)."""
    from transformers import AutoTokenizer
    model, _, engram = load_reference(ckpt_dir)
    args = model_args(ckpt_dir, max_batch_size, max_seq_len)
    layout = engram.EngramLayout.from_args(args)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    with torch.device(device):
        return engram.NgramHashState(args, layout, tok), layout
