"""Exercise target builders with arbitrary folders and a forbidden GT loader.

Tiny deterministic encoder substitutes only feature computation. The actual
image inventory, pseudo-label loader, style scan, and pooling code are exercised.
No training, evaluation, real target labels, or GPU are used.
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from dataloader import Pseudolabeldata
from style import CHANNELS, UnlabelledImages, build_style_bank


class FeatureOnlyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(1, 5):
            setattr(self, f"layer{i}", nn.Identity())

    def forward(self, images):
        signal = images.mean(1, keepdim=True)
        for i, channels in enumerate(CHANNELS, 1):
            getattr(self, f"layer{i}")(signal.expand(-1, channels, -1, -1))
        value = images.mean((1, 2, 3))
        return torch.stack([value, 1 - value], -1)


class FeatureOnlyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.visual = FeatureOnlyVisual()

    def forward(self, images, tokens):
        return self.visual(images), torch.eye(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        sources, target = ["s0", "s1", "s2"], "target"
        for domain in [*sources, target]:
            # These directory names have no relationship to the class vocabulary.
            for i in range(2):
                path = root / domain / f"arbitrary_container_{i}" / f"image{i}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4), (60 + 120 * i,) * 3).save(path)
        encoder = FeatureOnlyEncoder().eval()
        preprocess = Compose([ToTensor()])
        config = SimpleNamespace(device="cpu", num_workers=0, batch_size=2, pin_memory=False, threshold=.4)
        with patch("torchvision.datasets.ImageFolder", side_effect=AssertionError("Target GT loader accessed")):
            inventory = UnlabelledImages(root / target, preprocess)
            assert not any(hasattr(inventory, name) for name in ("targets", "labels", "classes", "class_to_idx"))
            first = Pseudolabeldata(root / target, encoder, preprocess, config, ["source_class_A", "source_class_B"])
            with contextlib.redirect_stdout(io.StringIO()):
                bank = build_style_bank(encoder, preprocess, root, sources, target, root / "cache", 2, 0)
            # Rename target folders while preserving ordering and image contents.
            for i in range(2):
                (root / target / f"arbitrary_container_{i}").rename(root / target / f"unrelated_name_{i}")
            second = Pseudolabeldata(root / target, encoder, preprocess, config, ["source_class_A", "source_class_B"])
            with contextlib.redirect_stdout(io.StringIO()):
                renamed_bank = build_style_bank(encoder, preprocess, root, sources, target, root / "cache2", 2, 0)
        assert len(first) == len(second) == 2
        for (image, pseudo), (other_image, other_pseudo) in zip(first.instances, second.instances):
            torch.testing.assert_close(image, other_image, rtol=0, atol=0)
            assert pseudo == other_pseudo
        for (mean, std), (other_mean, other_std) in zip(bank["target"], renamed_bank["target"]):
            torch.testing.assert_close(mean, other_mean, rtol=0, atol=0)
            torch.testing.assert_close(std, other_std, rtol=0, atol=0)
    report = {"target_imagefolder_gt_constructor_forbidden": True,
              "target_inventory_has_no_label_mapping": True,
              "target_pseudo_label_builder_passed": True,
              "target_style_bank_builder_passed": True,
              "target_folder_rename_leaves_pseudo_labels_and_style_statistics_exactly_unchanged": True,
              "class_vocabulary_provided_externally": True,
              "scope": "Actual target data builders with tiny feature-only encoder; evaluation loader excluded."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
