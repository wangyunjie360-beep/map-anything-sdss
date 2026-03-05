"""
Minimal smoke test for the astro SDSS self-supervised pipeline.

Usage:
    python scripts/smoke_test_astro_pipeline.py
"""

import torch

from mapanything.datasets.astro.sdss_pair import AstroSDSSPairDataset
from mapanything.models.astro_mapanything import AstroMapAnything
from mapanything.train.losses_astro import AstroBiModalSelfSupLoss


def main():
    dataset = AstroSDSSPairDataset(
        manifest_path="/root/wyj/map-anything-sdss/astro_sdss_v2_20gb_curated/manifest/train_ready_train.jsonl",
        split="train",
        seed=42,
        eval_fixed_mask=False,
    )
    sample0 = dataset[0]
    sample1 = dataset[1]

    batch = [{}, {}]
    for key in sample0[0].keys():
        if torch.is_tensor(sample0[0][key]):
            batch[0][key] = torch.stack([sample0[0][key], sample1[0][key]], dim=0)
        else:
            batch[0][key] = [sample0[0][key], sample1[0][key]]
    for key in sample0[1].keys():
        if torch.is_tensor(sample0[1][key]):
            batch[1][key] = torch.stack([sample0[1][key], sample1[1][key]], dim=0)
        else:
            batch[1][key] = [sample0[1][key], sample1[1][key]]

    model = AstroMapAnything(
        image_encoder_name="dinov2_vitl14",
        image_encoder_pretrained=False,
    )
    criterion = AstroBiModalSelfSupLoss(stage_a_epochs=0)

    preds = model(batch)
    loss, details = criterion(batch, preds)
    print("Smoke test passed.")
    print("Loss:", float(loss))
    print("Detail keys:", sorted(details.keys()))


if __name__ == "__main__":
    main()
