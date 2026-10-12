# wo_a at verify size: the inverse-RoPE kernel's [G, T, D] buffer -> copy to [T, G * D] -> mrow, against mrow reading
# the buffer in place (3-D x). Weights cycled from DRAM like the serve (12 copies of the layer's groups).
import time, torch
from vllm_exl3_moe.mrow import MrowGroup
dev = torch.device("cuda")
K, G, D, N, T, L = 5, 4, 4096, 1024, 6, 12
def rl(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return (torch.randint(-32768, 32768, (D // 16, N // 16, 16 * K), dtype=torch.int16, device=dev, generator=g),
            ((torch.randint(0, 2, (D,), device=dev, generator=g) * 2 - 1) * 0.05).half(),
            ((torch.randint(0, 2, (N,), device=dev, generator=g) * 2 - 1) * 0.05).half())
grps = [MrowGroup([(*rl(100 * l + i), i * D, i * N) for i in range(G)], K) for l in range(L)]
buf = torch.randn(G, T, D, device=dev).bfloat16()
src = buf.clone()
out = torch.empty(T, G * N, device=dev, dtype=torch.bfloat16)
def copy_path():
    for gr in grps: buf.copy_(src); gr(buf.transpose(0, 1).reshape(T, G * D), out)
def inplace_path():
    for gr in grps: buf.copy_(src); gr(buf.transpose(0, 1), out)
def us(f, reps=200):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / L * 1e6
def copy_only():
    for gr in grps: buf.copy_(src); buf.transpose(0, 1).reshape(T, G * D)
a, b, c = [], [], []
for _ in range(8):
    a.append(us(copy_path)); b.append(us(inplace_path)); c.append(us(copy_only))
print(f"wo_a per layer, T={T}, min of 8: copy + mrow {min(a):.2f} us, mrow in place {min(b):.2f} us, copy alone {min(c):.2f} us")
print(f"  medians: {sorted(a)[4]:.2f} / {sorted(b)[4]:.2f} / {sorted(c)[4]:.2f}")
