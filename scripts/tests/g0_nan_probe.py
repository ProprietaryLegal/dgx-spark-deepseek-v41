# SPDX-License-Identifier: Apache-2.0
"""Where does the first NaN come from? Boot the truncated model like g0_truncated.py, then (1) scan every parameter
and buffer for NaN / Inf, (2) hook every module under the decoder layers and report, for one short generation, the
first modules (in execution order) whose output contains NaN / Inf, with input stats.

  DSV41_TRUNCATE_LAYERS=3 PYTHONPATH=/tests/plugins VLLM_ENABLE_V1_MULTIPROCESSING=0 python3 g0_nan_probe.py /model
"""
import os
import sys

import torch

N = int(os.environ["DSV41_TRUNCATE_LAYERS"])


def truncate(cfg):
    getattr(cfg, "text_config", cfg).num_hidden_layers = N
    return cfg


def scan_params(model):
    bad, meta = [], []
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.is_meta:
            meta.append(f"{name} {tuple(t.shape)} {t.dtype}")
            continue
        if t.is_floating_point() and t.dtype not in (torch.float8_e4m3fn, torch.float8_e8m0fnu) and t.numel():
            x = t.detach()
            n_nan, n_inf = torch.isnan(x).sum().item(), torch.isinf(x).sum().item()
            if n_nan or n_inf:
                bad.append(f"{name} {tuple(t.shape)} {t.dtype}: nan={n_nan} inf={n_inf}")
    return {"nan_inf": bad, "meta": meta}


def install_hooks(model):
    events = []

    def stat(t):
        if isinstance(t, (tuple, list)):
            t = next((x for x in t if isinstance(x, torch.Tensor)), None)
        if not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.numel() == 0:
            return None
        x = t.detach().float()
        return (torch.isnan(x).any().item() or torch.isinf(x).any().item(), x.abs().nan_to_num(0, 0, 0).max().item(),
                tuple(t.shape), str(t.dtype))

    def hook(name):
        def f(mod, inp, out):
            if len(events) > 400:
                return
            so, si = stat(out), stat(inp)
            if so is not None:
                events.append((name, so, si))
        return f

    for name, mod in model.named_modules():
        if ".layers." in name or name.endswith(("embed_tokens", "norm", "lm_head")):
            mod.register_forward_hook(hook(name))
        # the attention core is not a submodule: wrap the backend methods directly
        for meth in ("forward_mqa", "_o_proj"):
            fn = getattr(type(mod), meth, None)
            if fn is None or getattr(mod, f"_wrapped_{meth}", False):
                continue

            def make(fn, name=name, meth=meth, mod=mod):
                def w(*a, **kw):
                    r = fn(mod, *a, **kw)
                    if len(events) <= 400:
                        tens = [x for x in a if isinstance(x, torch.Tensor)]
                        ins = ", ".join(f"{tuple(t.shape)}:{'NAN' if (t.is_floating_point() and not torch.isfinite(t.float()).all()) else 'ok'}"
                                        for t in tens)
                        so = stat(r) if r is not None else stat(tens[-1])   # forward_mqa writes into `output`
                        events.append((f"{name}.{meth}  args[{ins}]", so or (False, 0.0, (), ""), None))
                    return r
                return w
            setattr(mod, meth, make(fn))
            setattr(mod, f"_wrapped_{meth}", True)
    model._nan_events = events
    return len(events)


def main():
    from vllm import LLM, SamplingParams
    llm = LLM(model=sys.argv[1], tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.35,
              hf_overrides=truncate, enforce_eager=True,
              engram_config={"cpu_offload": True, "dp_shared_memory": False, "disk_offload": True},
              limit_mm_per_prompt={"image": 0}, seed=0)
    scan = llm.apply_model(scan_params)[0]
    print("PARAM SCAN nan/inf:", scan["nan_inf"] or "none")
    print("PARAM SCAN meta tensors:", len(scan["meta"]), scan["meta"][:20])
    llm.apply_model(install_hooks)
    llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=2))
    events = llm.apply_model(lambda m: m._nan_events)[0]
    first_bad = [e for e in events if e[1][0]]
    print(f"hooked events: {len(events)}, with NaN/Inf: {len(first_bad)}")
    for name, so, si in events[:60]:
        flag = "NAN " if so[0] else "    "
        print(f"{flag}{name:70s} out {so[2]} {so[3]} absmax {so[1]:.3e} | in {si[2] if si else None} "
              f"absmax {si[1] if si else float('nan'):.3e} nan_in={si[0] if si else None}")


if __name__ == "__main__":
    main()
