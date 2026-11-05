from __future__ import annotations

from copy import deepcopy

import pytest

from semi_nc import load_config


@pytest.fixture()
def tiny_config():
    config = deepcopy(load_config("configs/semi_nc.yaml"))
    config["model"].update(
        widths=[8, 16, 32, 64, 128],
        attention_dim=16,
        attention_heads=2,
        deformable_points=2,
        external_memory_tokens=4,
    )
    config["prototype"].update(
        pool_capacity=8,
        query_count=4,
        negative_key_count=4,
        sgc_iterations=2,
    )
    config["training"].update(early_epochs=1, batch_size=1)
    config["fia"].update(corruption_probability=0.0)
    return config
