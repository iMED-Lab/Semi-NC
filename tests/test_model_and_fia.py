import torch
from torch import nn

from semi_nc.config import dynamic_gamma
from semi_nc.data import build_augmentation
from semi_nc.fia import MaskCorruptor, foreground_aware_amplification
from semi_nc.model import DecoderBlock, SemiNCNet


def test_model_exposes_all_paper_paths(tiny_config):
    model = SemiNCNet(tiny_config).eval()
    assert len(model.pools) == 4
    assert all(isinstance(pool, nn.MaxPool2d) for pool in model.pools)
    assert all(
        isinstance(block, DecoderBlock)
        for decoder in model.decoders
        for block in decoder.blocks
    )
    assert [len(deam.projections) for deam in model.deams] == [1, 2, 3]
    with torch.no_grad():
        output = model(torch.rand(2, 1, 32, 32))
    assert output["logits"].shape == (2, 2, 32, 32)
    assert output["embedding"].shape == (2, tiny_config["model"]["widths"][-1], 2, 2)
    assert len(output["encoder_features"]) == 5
    assert len(output["attention_features"]) == 4
    assert len(output["decoder_features"]) == 8


def test_fia_preserves_foreground_and_suppresses_far_background(tiny_config):
    image = torch.ones(1, 1, 17, 17)
    mask = torch.zeros(1, 2, 17, 17)
    mask[:, 0, 8, 8] = 1
    enhanced, weights = foreground_aware_amplification(image, mask, lambda_distance=3.0)
    assert weights[0, 0, 8, 8] == 1
    assert weights[0, 0, 0, 0] < weights[0, 0, 7, 8] < 1
    assert torch.allclose(enhanced, weights)


def test_fia_empty_foreground_is_identity_not_position_ramp(tiny_config):
    image = torch.rand(1, 1, 17, 17)
    mask = torch.zeros(1, 2, 17, 17)
    enhanced, weights = foreground_aware_amplification(image, mask, lambda_distance=3.0)
    assert torch.equal(weights, torch.ones_like(weights))
    assert torch.equal(enhanced, image)


def test_corruptor_contract_and_dynamic_gamma(tiny_config):
    masks = torch.zeros(2, 2, 16, 16)
    masks[:, :, 4:8, 4:8] = 1
    corruptor = MaskCorruptor(tiny_config["fia"])
    assert torch.equal(corruptor(masks), masks)
    assert dynamic_gamma(1) == 0.998
    assert dynamic_gamma(500) == 0.7


def test_paper_augmentation_has_one_flip_operation(tiny_config):
    augmentation = build_augmentation(tiny_config["augmentation"])
    flip_operations = [
        transform
        for transform in augmentation.transforms
        if transform.__class__.__name__ in {"OneOf", "HorizontalFlip", "VerticalFlip"}
    ]
    assert len(flip_operations) == 1
