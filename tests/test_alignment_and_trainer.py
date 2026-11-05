import torch

from semi_nc.model import SemiNCNet
from semi_nc.prototypes import (
    PrototypePool,
    categorical_probabilities,
    pixel_prototype_contrastive_loss,
    pixel_prototype_contrastive_terms,
)
from semi_nc.training import SemiNCTrainer, sample_confidence, teacher_bootcamp_step


def _masks(batch=1, size=32):
    masks = torch.zeros(batch, 2, size, size)
    masks[:, 0, 4:18, 4:8] = 1
    masks[:, 1, 12:22, 14:24] = 1
    return masks


def test_alignment_loss_has_gradient():
    embedding = torch.randn(1, 8, 8, 8, requires_grad=True)
    foreground_prob = torch.rand(1, 2, 8, 8)
    masks = _masks(size=8)
    pool = PrototypePool(3, capacity=8)
    pool.update(0, torch.randn(2, 8))
    pool.update(1, torch.randn(2, 8))
    pool.update(2, torch.randn(2, 8))
    loss = pixel_prototype_contrastive_loss(
        embedding,
        categorical_probabilities(foreground_prob),
        masks,
        None,
        None,
        pool,
        query_count=4,
        negative_count=4,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert embedding.grad is not None


def test_teacher_and_both_algorithm_two_stages_execute(tiny_config):
    images_l = torch.rand(1, 1, 32, 32)
    images_u = torch.rand(1, 1, 32, 32)
    masks = _masks()
    teacher = SemiNCNet(tiny_config)
    bootcamp_loss = teacher_bootcamp_step(teacher, images_l, masks, tiny_config)
    assert torch.isfinite(bootcamp_loss)

    student = SemiNCNet(tiny_config)
    trainer = SemiNCTrainer(student, teacher, tiny_config)
    early = trainer.step(images_l, masks, images_u, epoch=1)
    assert early.pcl == 0
    assert torch.isfinite(early.loss)
    late = trainer.step(images_l, masks, images_u, epoch=2)
    assert torch.isfinite(late.loss)
    assert len(trainer.pool) > 0


def test_query_count_is_per_class_per_minibatch():
    embeddings = torch.rand(2, 4, 2, 2)
    probabilities = torch.zeros(2, 3, 2, 2)
    probabilities[:, 0] = 0.9
    probabilities[:, 1] = 0.1
    masks = torch.zeros(2, 2, 2, 2)
    pool = PrototypePool(num_categories=3)
    pool.update(0, torch.rand(2, 4))
    pool.update(1, torch.rand(2, 4))
    terms = pixel_prototype_contrastive_terms(
        embeddings, probabilities, masks, None, None, pool, query_count=4
    )
    assert len(terms) == 4


def test_background_confidence_is_filtered_before_resampling():
    embeddings = torch.rand(1, 4, 1, 1)
    foreground = torch.full((1, 2, 4, 4), 0.01)
    foreground[:, :, 0, 0] = 0.49
    category_probabilities = categorical_probabilities(foreground)
    pseudo_masks = torch.zeros_like(foreground)
    pseudo_confidence = torch.maximum(foreground, 1.0 - foreground)
    pool = PrototypePool(num_categories=3)
    pool.update(0, torch.rand(2, 4))
    pool.update(1, torch.rand(2, 4))
    terms = pixel_prototype_contrastive_terms(
        embeddings,
        category_probabilities,
        None,
        pseudo_masks,
        pseudo_confidence,
        pool,
        query_count=1,
        delta=0.6,
    )
    assert terms == []


def test_sample_confidence_uses_positive_foreground_probability():
    probabilities = torch.zeros(1, 2, 2, 2)
    probabilities[:, 0] = 0.51
    probabilities[:, 1] = 0.00001
    assert torch.allclose(sample_confidence(probabilities), torch.tensor([0.51]))
