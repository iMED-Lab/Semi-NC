import torch

from semi_nc.losses import (
    binary_dice_loss,
    representation_knowledge_distillation_loss,
    segmentation_loss,
)


def test_segmentation_loss_prefers_correct_logits():
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    correct = torch.where(target.bool(), torch.tensor(8.0), torch.tensor(-8.0))
    wrong = -correct
    assert segmentation_loss(correct, target) < segmentation_loss(wrong, target)


def test_dice_is_computed_per_class_before_averaging():
    target = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.float32)
    logits = torch.tensor([[[[8.0, -8.0]], [[8.0, -8.0]]]], dtype=torch.float32)
    class_losses = [
        binary_dice_loss(logits[:, index : index + 1], target[:, index : index + 1])
        for index in range(2)
    ]
    assert torch.allclose(binary_dice_loss(logits, target), torch.stack(class_losses).mean())


def test_rkd_zero_for_equal_features_and_respects_valid_samples():
    feature = torch.rand(2, 4, 8, 8)
    zero = representation_knowledge_distillation_loss([feature], [feature.clone()])
    assert torch.allclose(zero, torch.tensor(0.0))
    teacher = feature.clone()
    teacher[1] = teacher[1] * 4 + torch.rand_like(teacher[1])
    masked = representation_knowledge_distillation_loss(
        [feature], [teacher], sample_mask=torch.tensor([True, False])
    )
    assert torch.allclose(masked, torch.tensor(0.0), atol=1e-6)


def test_rkd_uses_squared_l2_sum_not_spatial_mean():
    student = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]])
    teacher = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])
    loss = representation_knowledge_distillation_loss([student], [teacher])
    assert torch.allclose(loss, torch.tensor(4.0))
