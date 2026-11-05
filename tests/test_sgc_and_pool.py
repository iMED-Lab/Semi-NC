import torch
from unittest.mock import patch

from semi_nc.prototypes import PrototypePool
from semi_nc.sgc import masked_average_pool, superpixel_guided_clustering
from third_party import asgnet_sgc


def test_sgc_direct_source_equations_return_k_prototypes():
    feature = torch.zeros(4, 16, 16)
    feature[:, :8] = 1.0
    feature[:, 8:] = 3.0
    mask = torch.zeros(16, 16)
    mask[2:14, 2:14] = 1
    with patch(
        "semi_nc.sgc.source_place_seed_points",
        wraps=asgnet_sgc.place_seed_points_fixed,
    ) as seed_source, patch(
        "semi_nc.sgc.source_sp_center_iter", wraps=asgnet_sgc.sp_center_iter
    ) as cluster_source:
        prototypes = superpixel_guided_clustering(
            feature, mask, num_prototypes=3, iterations=3
        )
    seed_source.assert_called_once()
    cluster_source.assert_called_once()
    assert prototypes.shape == (3, 4)
    assert torch.isfinite(prototypes).all()


def test_sgc_small_region_degrades_to_masked_average_pool():
    feature = torch.rand(5, 8, 8)
    mask = torch.zeros(8, 8)
    mask[3, 3] = 1
    expected = masked_average_pool(feature, mask)
    actual = superpixel_guided_clustering(feature, mask, num_prototypes=3)
    assert actual.shape == (1, 5)
    assert torch.allclose(actual[0], expected)


def test_pool_algorithm_one_merges_most_similar_until_capacity():
    pool = PrototypePool(num_categories=3, capacity=2)
    pool.update(1, torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]))
    assert pool.get(1).shape == (2, 2)
    state = pool.state_dict()
    restored = PrototypePool(1, 1)
    restored.load_state_dict(state)
    assert torch.allclose(restored.get(1), pool.get(1))
