import sys, torch
from vllm_exl3_moe import sparse_attn as SA
sys.path.insert(0, ".")
from test_sparse_attn_gpu import pages
dev = torch.device("cuda")
T, H, page, cpage = 6, 32, 64, 16
swa, cmp = pages(64, page, 1), pages(2048, cpage, 2)
g = torch.Generator(device="cpu").manual_seed(0)
q = (torch.randn(T, H, 512, generator=g) * 0.3).to(torch.bfloat16).to(dev)
sink = torch.randn(H, generator=g).float().to(dev)
si = torch.randint(0, 64 * page, (T, 128), generator=g, dtype=torch.int32).to(dev)
sl = torch.full((T,), 128, dtype=torch.int32, device=dev)
ci = torch.randint(0, 2048 * cpage, (T, 512), generator=g, dtype=torch.int32).to(dev)
cl = torch.full((T,), 512, dtype=torch.int32, device=dev)
out = torch.empty(T, H, 512, dtype=torch.bfloat16, device=dev)
f = lambda: SA.sparse_attention(q, out, 0.044, sink, swa, si, sl, page, cmp, ci, cl, cpage, hb=16, bn=64, splits=4)
for _ in range(3): f()
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    for _ in range(50): f()
    torch.cuda.synchronize()
for e in p.key_averages():
    if e.device_type.name == "CUDA" or "sparse" in e.key:
        print(f"{e.key[:40]:40s} {e.device_time_total / max(e.count, 1):8.1f} us avg x{e.count}")
k = SA._dsv4_sparse_part
print([ (c.n_regs, c.n_spills) for c in k.cache[0].values()] if hasattr(k, "cache") else "no cache info")
