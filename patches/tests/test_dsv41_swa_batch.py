# SPDX-License-Identifier: Apache-2.0
"""Patch 0037: VLLM_DSV41_SWA_BATCH, DeepSeek-V4.1's pure-decode SWA metadata builds of every KV cache group in one
launch (sparse_swa.run_swa_decode_builds) instead of a torch.ge, a fill and _compute_swa_indices_and_lens_kernel per
group.

Against the unbatched build's three steps, group by group, bit for bit: is_valid_token, the decode lens with their
zeroed tail and the window indices, on buffers pre-filled with garbage (both paths must write exactly the same cells),
for one and two requests, padding rows (slot -1), a replay start and window rows that cross blocks; groups with
different block tables and slot mappings share one launch, and builds that differ in shape go to separate launches.

Run inside the dsv41 image on a GPU:  python3 -m pytest -q test_dsv41_swa_batch.py
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = "cuda"
WINDOW, BLOCK, MAX_TOKENS, MAX_BLOCKS = 128, 64, 96, 40


def make_group(g, n_slots, num_reqs, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    bt = torch.randperm(4096, device=DEV, generator=gen)[: num_reqs * MAX_BLOCKS].view(num_reqs, MAX_BLOCKS)
    bt = bt.to(torch.int32).contiguous()
    slots = torch.randint(0, 1 << 20, (n_slots,), device=DEV, generator=gen, dtype=torch.int64)
    return bt, slots


def garbage(shape, dtype, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=DEV, generator=gen).bool()
    return torch.randint(-5000, 5000, shape, device=DEV, generator=gen, dtype=dtype)


def unbatched(S, b):
    """The build's own steps: torch.ge, the lens-tail fill, the decode kernel (token_offset 0, no image)."""
    torch.ge(b.slot_mapping, 0, out=b.is_valid)
    b.lens[b.num_tokens:] = 0
    S._compute_swa_indices_and_lens_kernel[(b.num_tokens,)](
        b.indices, b.indices.stride(0), b.lens, b.window_size, b.indices.shape[-1], b.lens, b.lens,
        b.query_start_loc, b.seq_lens, b.token_to_req, b.is_valid, b.block_table, b.block_table.stride(0),
        b.block_size, b.replay_start, 0, HAS_IMAGE=False, TRITON_BLOCK_SIZE=1024,
    )


def builds(S, groups, n_tokens, n_slots, reqs, replay, seed):
    """Two identical sets of _SWADecodeBuild records over separate garbage-filled buffers."""
    qsl = torch.tensor([0] + list(torch.tensor(reqs).cumsum(0)), dtype=torch.int32, device=DEV)
    seq_lens = torch.tensor([700 + 37 * i for i in range(len(reqs))], dtype=torch.int32, device=DEV)
    t2r = torch.tensor(sum(([i] * n for i, n in enumerate(reqs)), []) + [0] * (n_slots - n_tokens),
                       dtype=torch.int32, device=DEV)
    rs = torch.tensor(replay + [0] * (8 - len(replay)), dtype=torch.int32, device=DEV)
    out = ([], [])
    for g in range(groups):
        bt, slots = make_group(g, n_slots, len(reqs), seed + g)
        slots[n_tokens:] = -1                                  # padding rows
        if g % 3 == 1:
            slots[0] = -1                                      # an invalid decode row
        for k in range(2):
            out[k].append(S._SWADecodeBuild(
                garbage((MAX_TOKENS, 1, WINDOW), torch.int32, seed + 100 + g),
                garbage((MAX_TOKENS,), torch.int32, seed + 200 + g),
                garbage((n_slots,), torch.bool, seed + 300 + g),
                slots, bt, rs, qsl, seq_lens, t2r, WINDOW, BLOCK, n_tokens))
    return out


@pytest.mark.parametrize("reqs,n_slots,replay", [([6], 8, []), ([3, 3], 8, [0, 650]), ([1], 1, [300]), ([6], 6, [])])
def test_batched_matches_unbatched(reqs, n_slots, replay):
    from vllm.v1.attention.backends.mla import sparse_swa as S

    n_tokens = sum(reqs)
    ref, got = builds(S, 19, n_tokens, n_slots, reqs, replay, seed=7 * n_tokens + len(replay))
    for b in ref:
        unbatched(S, b)
    S.run_swa_decode_builds(got)
    torch.cuda.synchronize()
    for g, (r, b) in enumerate(zip(ref, got)):
        assert torch.equal(r.is_valid, b.is_valid), g
        assert torch.equal(r.lens, b.lens), g
        assert torch.equal(r.indices, b.indices), g


def test_one_launch_per_layout():
    from vllm.v1.attention.backends.mla import sparse_swa as S

    ref, got = builds(S, 4, 6, 8, [6], [], seed=3)
    ref2, got2 = builds(S, 3, 3, 8, [3], [], seed=5)          # another layout: a separate launch
    calls = []
    orig = S._swa_decode_batched_kernel

    class Spy:
        def __getitem__(self, grid):
            calls.append(grid)
            return orig[grid]

    S._swa_decode_batched_kernel = Spy()
    try:
        S.run_swa_decode_builds(got + got2)
    finally:
        S._swa_decode_batched_kernel = orig
    for b in ref + ref2:
        unbatched(S, b)
    torch.cuda.synchronize()
    assert len(calls) == 2 and sorted(c[1] for c in calls) == [3, 4]
    for r, b in zip(ref + ref2, got + got2):
        assert torch.equal(r.is_valid, b.is_valid) and torch.equal(r.lens, b.lens) and torch.equal(r.indices, b.indices)


def test_context_off_by_default(monkeypatch):
    from vllm.v1.attention.backends.mla import sparse_swa as S

    monkeypatch.setattr(S, "_SWA_BATCH_ENABLED", False)
    with S.deferred_decode_builds():
        assert S._swa_batch is None                            # builds run as before
    monkeypatch.setattr(S, "_SWA_BATCH_ENABLED", True)
    ref, got = builds(S, 2, 6, 8, [6], [], seed=11)
    with S.deferred_decode_builds():
        assert S._swa_batch == []
        S._swa_batch.extend(got)                               # what two builds would queue
    assert S._swa_batch is None
    for b in ref:
        unbatched(S, b)
    torch.cuda.synchronize()
    for r, b in zip(ref, got):
        assert torch.equal(r.lens, b.lens) and torch.equal(r.indices, b.indices)


def test_file_switch(monkeypatch, tmp_path):
    import os

    from vllm.v1.attention.backends.mla import sparse_swa as S

    f = tmp_path / "swa_batch"
    f.write_text("1")
    monkeypatch.setattr(S, "_SWA_BATCH_FILE", str(f))
    monkeypatch.setattr(S, "_SWA_BATCH_ENABLED", False)
    monkeypatch.setattr(S, "_swa_file", {"n": 0, "mtime": None})
    assert S._swa_batch_on() is True                           # build 1 reads the file
    f.write_text("0")
    st = os.stat(f)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))   # a new mtime whatever the clock granularity
    assert all(S._swa_batch_on() for _ in range(15))           # builds 2-16 keep the last reading
    assert S._swa_batch_on() is False                          # build 17 reads it again
    with S.deferred_decode_builds():
        assert S._swa_batch is None
