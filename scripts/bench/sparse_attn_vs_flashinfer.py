# SPDX-License-Identifier: Apache-2.0
"""GB10 check of the plugin's Triton DSV4 sparse attention (vllm_exl3_moe/sparse_attn.py) against FlashInfer, the call
vLLM's SM120 attention makes for verify rows (DeepseekV4FlashInferSM120Attention._forward_prefill), on the same random
fp8_ds_mla cache bytes: which record layout FlashInfer reads (a page's data rows then its scale rows, or 584-byte records
token-major), whether our outputs agree, and the time of each (CUDA graph replays, T verify rows, 32 heads, a 128-row
window + 512 compressed picks). Run in the serve image with the plugin source on PYTHONPATH:
  docker run --rm --gpus all -v <plugin>:/plugin -v <this dir>:/b -e PYTHONPATH=/plugin dsv41-serve:v9 \
    python3 /b/sparse_attn_vs_flashinfer.py
"""
import time

import torch
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4 as fi_dsv4  # vLLM's wrapper

from vllm_exl3_moe import sparse_attn as SA

dev = torch.device("cuda")
torch.manual_seed(0)
T, H, PAGE, CPAGE = 6, 32, 64, 64          # extra pages of 64 (256-token blocks at compress ratio 4)
NS, NC = 128, 512


def cache(n_pages, page):
    """[n_pages, page, 584] uint8: random fp8 nope, bf16 rope, ue8m0 scales (written in both candidate layouts by the
    caller; here the token-major bytes, with valid values in every slot)."""
    nope = (torch.randn(n_pages * page, SA.NOPE) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    rope = (torch.randn(n_pages * page, SA.ROPE) * 0.5).to(torch.bfloat16).view(torch.uint8).reshape(n_pages * page, -1)
    sc = torch.randint(121, 130, (n_pages * page, SA.SCALES), dtype=torch.uint8)
    sc[:, 7] = 0
    return nope, rope, sc


def pack(nope, rope, sc, n_pages, page, layout):
    rec = torch.cat([nope, rope], 1)                                           # [rows, 576]
    if layout == "token":                                                      # [448][128][8] a row
        buf = torch.cat([rec, sc], 1).reshape(n_pages, page, 584)
    else:                                                                      # a page: data rows, then scale rows
        d = rec.reshape(n_pages, page * 576)
        s = sc.reshape(n_pages, page * 8)
        buf = torch.cat([d, s], 1).reshape(n_pages, page, 584)
    return buf.contiguous().to(dev)


def graph_us(fn, reps=20):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (10 * reps) * 1e6


def main():
    sn, sr, ss = cache(64, PAGE)
    cn, cr, cs = cache(512, CPAGE)
    q = (torch.randn(T, H, 512) * 0.3).to(torch.bfloat16).to(dev)
    sink = torch.randn(H).float().to(dev)
    si = torch.randint(0, 64 * PAGE, (T, NS), dtype=torch.int32).to(dev)
    sl = torch.full((T,), NS, dtype=torch.int32, device=dev)
    ci = torch.randint(0, 512 * CPAGE, (T, NC), dtype=torch.int32).to(dev)
    cl = torch.full((T,), NC, dtype=torch.int32, device=dev)
    scale = 512 ** -0.5
    ws = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    for layout in ("page", "token"):
        swa = pack(sn, sr, ss, 64, PAGE, layout)
        cmp = pack(cn, cr, cs, 512, CPAGE, layout)
        out_fi = torch.empty(T, H, 512, dtype=torch.bfloat16, device=dev)

        def fi():
            fi_dsv4(query=q, swa_kv_cache=swa.unsqueeze(-2), workspace_buffer=ws, sparse_indices=si,
                    compressed_kv_cache=cmp.unsqueeze(-2), out=out_fi, bmm1_scale=scale, sinks=sink, kv_layout="NHD",
                    swa_topk_lens=sl, extra_sparse_indices=ci, extra_sparse_topk_lens=cl)
        fi()
        # ours reads the page layout; on token-major bytes it would be wrong, which is the point of the comparison
        out_tr = torch.empty_like(out_fi)
        ours = lambda: SA.sparse_attention(q, out_tr, scale, sink, swa.view(swa.shape[0], -1), si, sl, PAGE,
                                           cmp.view(cmp.shape[0], -1), ci, cl, CPAGE)
        ours()
        torch.cuda.synchronize()
        rel = ((out_tr.float() - out_fi.float()).norm() / out_fi.float().norm()).item()
        ref = SA.reference(q, scale, sink, swa.view(swa.shape[0], -1), si, sl, PAGE, cmp.view(cmp.shape[0], -1), ci,
                           cl, CPAGE)
        rel_fi = ((out_fi.float() - ref).norm() / ref.norm()).item()
        print(f"bytes in the {layout} layout: FlashInfer vs page-layout reference rel {rel_fi:.3e}; ours vs FlashInfer "
              f"rel {rel:.3e}")
        if layout == "page":
            print(f"  time, T={T}: FlashInfer {graph_us(fi):.1f} us, ours {graph_us(ours):.1f} us")
            for hb, bn, sp in [(16, 64, 4), (16, 32, 4), (16, 32, 8), (16, 32, 16), (16, 16, 16), (8, 32, 8)]:
                f = lambda: SA.sparse_attention(q, out_tr, scale, sink, swa.view(swa.shape[0], -1), si, sl, PAGE,
                                                cmp.view(cmp.shape[0], -1), ci, cl, CPAGE, hb=hb, bn=bn, splits=sp,
                                                stages=1)
                print(f"    ours HB={hb} BN={bn} SPLITS={sp}: {graph_us(f):.1f} us")


if __name__ == "__main__":
    main()
