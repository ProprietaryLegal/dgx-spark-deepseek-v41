# SPDX-License-Identifier: Apache-2.0
"""Test-only vLLM general plugin, never in the serve path.

DSV41_TRUNCATE_LAYERS=N drops checkpoint tensors of backbone layers >= N while weights stream into
DeepseekV4Model.load_weights, so a layer-truncated DeepSeek-V4.1 (text_config.num_hidden_layers = N) loads from the
official shards without copying them. Upstream's loader indexes params_dict[name] with no guard, so without this the
first tensor of layer N raises KeyError.

Discovered through the dist-info next to this file: put this directory on PYTHONPATH.
"""
import os
import re

_LAYER = re.compile(r"^(?:model\.)?layers\.(\d+)\.")


def register():
    n = int(os.environ.get("DSV41_TRUNCATE_LAYERS", "0") or 0)
    if n <= 0:
        return
    from vllm.models.deepseek_v41.nvidia import model as m

    cls = m.DeepseekV4Model
    if getattr(cls, "_dsv41_truncated", None) == n:
        return
    orig = cls.load_weights

    def load_weights(self, weights):
        def keep(it):
            for name, w in it:
                mm = _LAYER.match(name)
                if mm and int(mm.group(1)) >= n:
                    continue
                yield name, w

        return orig(self, keep(weights))

    cls.load_weights = load_weights
    cls._dsv41_truncated = n
