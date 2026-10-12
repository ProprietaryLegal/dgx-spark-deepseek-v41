import sys, time, torch, triton
from vllm_exl3_moe import sparse_attn as SA
sys.path.insert(0, ".")
from test_sparse_attn_gpu import pages
dev = torch.device("cuda")
T, H, page, cpage = 6, 32, 64, 16
swa, cmp = pages(64, page, 1), pages(2048, cpage, 2)
g = torch.Generator(device="cpu").manual_seed(0)
q = (torch.randn(T, H, 512, generator=g) * 0.3).to(torch.bfloat16).to(dev)
si = torch.randint(0, 64 * page, (T, 128), generator=g, dtype=torch.int32).to(dev)
sl = torch.full((T,), 128, dtype=torch.int32, device=dev)
ci = torch.randint(0, 2048 * cpage, (T, 512), generator=g, dtype=torch.int32).to(dev)
cl = torch.full((T,), 512, dtype=torch.int32, device=dev)
for hb, bn, sp, nw, ns in [(16, 64, 4, 8, 1), (16, 64, 4, 8, 2), (16, 64, 4, 8, 3), (16, 32, 4, 8, 2), (16, 32, 8, 8, 2), (16, 64, 2, 8, 2), (16, 32, 4, 8, 3)]:
    n = T * (H // hb) * sp * hb
    pm = torch.empty(n, device=dev); pl = torch.empty(n, device=dev); po = torch.empty(n * 512, device=dev)
    args = (q, swa, si, sl, cmp, ci, cl, pm, pl, po, 0.044, 128, 512, swa.stride(0), cmp.stride(0), q.stride(0), q.stride(1))
    kw = dict(H=H, HB=hb, BN=bn, SPLITS=sp, SPAGE=page, CPAGE=cpage, HAS_CMP=True, NBS=triton.cdiv(128, bn),
              NBC=triton.cdiv(512, bn), STAGES=ns, num_warps=nw)
    k = SA._dsv4_sparse_part[(T, H // hb, sp)](*args, **kw)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(20): SA._dsv4_sparse_part[(T, H // hb, sp)](*args, **kw)
    gr.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20): gr.replay()
    torch.cuda.synchronize()
    print(f"HB={hb} BN={bn} SPLITS={sp} warps={nw} stages={ns}: {(time.perf_counter() - t0) / 400 * 1e6:.1f} us, regs {k.n_regs}, spills {k.n_spills}")
