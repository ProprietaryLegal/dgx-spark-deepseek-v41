# SPDX-License-Identifier: Apache-2.0
"""vllm_exl3_moe.sparse_attn (Triton DSV4 sparse attention for verify-sized calls) against its fp32 reference, from the
same synthetic fp8_ds_mla pages (584 B records: 448 e4m3 nope, 64 bf16 rope, 7 ue8m0 scales; a page's data rows then
its scale rows; a padded page stride). FlashInfer agreement on GB10 is a separate check (scripts/bench).

    python3 -m pytest -q -s test_sparse_attn_gpu.py
"""
import pytest
import torch

from vllm_exl3_moe import sparse_attn as SA

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
dev = torch.device("cuda")


def pages(n_pages, page, seed, pad=64):
    """[n_pages, page * 584 + pad] uint8 in the SM12x record layout, random but sane values."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    stride = page * SA.ROW + page * SA.SCALES + pad
    buf = torch.zeros(n_pages, stride, dtype=torch.uint8)
    for p in range(n_pages):
        nope = (torch.randn(page, SA.NOPE, generator=g) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
        rope = (torch.randn(page, SA.ROPE, generator=g) * 0.5).to(torch.bfloat16).view(torch.uint8).reshape(page, -1)
        data = torch.cat([nope, rope], 1)                                          # [page, 576]
        buf[p, :page * SA.ROW] = data.reshape(-1)
        sc = torch.randint(121, 130, (page, SA.SCALES), generator=g, dtype=torch.uint8)
        sc[:, 7] = 0
        buf[p, page * SA.ROW: page * SA.ROW + page * SA.SCALES] = sc.reshape(-1)
    return buf.to(dev)


@pytest.mark.parametrize("has_cmp", [False, True])
@pytest.mark.parametrize("T", [1, 6])
def test_matches_reference(has_cmp, T):
    H, page, cpage = 32, 64, 16
    swa = pages(6, page, 1)
    cmp = pages(40, cpage, 2) if has_cmp else None
    g = torch.Generator(device="cpu").manual_seed(3 + T)
    q = (torch.randn(T, H, 512, generator=g) * 0.3).to(torch.bfloat16).to(dev)
    sink = torch.randn(H, generator=g).float().to(dev)
    swa_idx = torch.randint(0, 6 * page, (T, 128), generator=g, dtype=torch.int32).to(dev)
    swa_len = torch.randint(1, 129, (T,), generator=g, dtype=torch.int32).to(dev)
    swa_len[0] = 128
    kw = {}
    if has_cmp:
        kw = dict(cmp_cache=cmp, cmp_idx=torch.randint(0, 40 * cpage, (T, 512), generator=g, dtype=torch.int32).to(dev),
                  cmp_len=torch.randint(0, 513, (T,), generator=g, dtype=torch.int32).to(dev), cmp_page=cpage)
        kw["cmp_len"][0] = 512
    out = torch.empty(T, H, 512, dtype=torch.bfloat16, device=dev)
    scale = 512 ** -0.5
    SA.sparse_attention(q, out, scale, sink, swa, swa_idx, swa_len, page, **kw)
    ref = SA.reference(q, scale, sink, swa, swa_idx, swa_len, page, **kw)
    err = (out.float() - ref).abs().max().item()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"T={T} cmp={has_cmp}: max abs {err:.3e}, rel {rel:.3e}")
    assert rel < 1e-2 and err < 5e-2


def test_split_count_does_not_change_bits():
    """Different SPLITS: the merge combines partials in split order, so splits change the bits (fp32 association) but
    the result stays within bf16 noise; the same SPLITS twice gives the same bits."""
    H, page, cpage, T = 32, 64, 16, 6
    swa, cmp = pages(6, page, 4), pages(40, cpage, 5)
    g = torch.Generator(device="cpu").manual_seed(9)
    q = (torch.randn(T, H, 512, generator=g) * 0.3).to(torch.bfloat16).to(dev)
    sink = torch.randn(H, generator=g).float().to(dev)
    si = torch.randint(0, 6 * page, (T, 128), generator=g, dtype=torch.int32).to(dev)
    sl = torch.full((T,), 128, dtype=torch.int32, device=dev)
    ci = torch.randint(0, 40 * cpage, (T, 512), generator=g, dtype=torch.int32).to(dev)
    cl = torch.full((T,), 512, dtype=torch.int32, device=dev)
    outs = []
    for splits in (8, 8, 4):
        o = torch.empty(T, H, 512, dtype=torch.bfloat16, device=dev)
        SA.sparse_attention(q, o, 0.044, sink, swa, si, sl, page, cmp, ci, cl, cpage, splits=splits)
        outs.append(o)
    assert torch.equal(outs[0], outs[1])
    assert ((outs[0].float() - outs[2].float()).norm() / outs[0].float().norm()).item() < 1e-2


def test_dead_rows_skip():
    """Dead verify rows (vLLM's is_padding, past a request's depth) come out as zeros; live rows keep their bits."""
    H, page, cpage, T = 32, 64, 64, 6
    swa, cmp = pages(6, page, 6), pages(40, cpage, 7)
    g = torch.Generator(device="cpu").manual_seed(11)
    q = (torch.randn(T, H, 512, generator=g) * 0.3).to(torch.bfloat16).to(dev)
    sink = torch.randn(H, generator=g).float().to(dev)
    si = torch.randint(0, 6 * page, (T, 128), generator=g, dtype=torch.int32).to(dev)
    sl = torch.full((T,), 128, dtype=torch.int32, device=dev)
    ci = torch.randint(0, 40 * cpage, (T, 512), generator=g, dtype=torch.int32).to(dev)
    cl = torch.full((T,), 512, dtype=torch.int32, device=dev)
    full = torch.empty(T, H, 512, dtype=torch.bfloat16, device=dev)
    SA.sparse_attention(q, full, 0.044, sink, swa, si, sl, page, cmp, ci, cl, cpage)
    pad = torch.zeros(T + 3, dtype=torch.bool, device=dev)       # the forward's rows; q starts at row 3
    pad[3 + 4:] = True                                           # rows 4 and 5 of the request are dead
    part = torch.full((T, H, 512), 7.0, dtype=torch.bfloat16, device=dev)
    SA.sparse_attention(q, part, 0.044, sink, swa, si, sl, page, cmp, ci, cl, cpage, pad=pad, pad_off=3)
    assert torch.equal(part[:4], full[:4])
    assert torch.count_nonzero(part[4:]) == 0
