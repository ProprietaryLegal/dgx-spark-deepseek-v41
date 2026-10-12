# Triton DSV4 sparse attention at verify size (T rows, 32 heads, 128 window + 512 compressed keys), us a call in a graph,
# for a few (HB, BN, SPLITS); keys spread over many pages like a live cache.
import sys, time, torch
from vllm_exl3_moe import sparse_attn as SA
sys.path.insert(0, ".")
from test_sparse_attn_gpu import pages
dev = torch.device("cuda")
T, H, page, cpage = int(sys.argv[1]) if len(sys.argv) > 1 else 6, 32, 64, 16
swa, cmp = pages(64, page, 1), pages(2048, cpage, 2)
g = torch.Generator(device="cpu").manual_seed(0)
q = (torch.randn(T, H, 512, generator=g) * 0.3).to(torch.bfloat16).to(dev)
sink = torch.randn(H, generator=g).float().to(dev)
si = torch.randint(0, 64 * page, (T, 128), generator=g, dtype=torch.int32).to(dev)
sl = torch.full((T,), 128, dtype=torch.int32, device=dev)
ci = torch.randint(0, 2048 * cpage, (T, 512), generator=g, dtype=torch.int32).to(dev)
cl = torch.full((T,), 512, dtype=torch.int32, device=dev)
out = torch.empty(T, H, 512, dtype=torch.bfloat16, device=dev)
for hb, bn, sp in [(16, 32, 8), (16, 32, 4), (16, 64, 4), (8, 32, 8), (16, 32, 16), (32, 32, 4)]:
    f = lambda: SA.sparse_attention(q, out, 0.044, sink, swa, si, sl, page, cmp, ci, cl, cpage, hb=hb, bn=bn, splits=sp)
    f(); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(20): f()
    gr.replay(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): gr.replay()
    torch.cuda.synchronize()
    print(f"T={T} HB={hb} BN={bn} SPLITS={sp}: {(time.perf_counter() - t0) / 400 * 1e6:.1f} us a call")
