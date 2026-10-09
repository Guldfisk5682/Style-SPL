"""Fixed domain statistics and shared style-to-domain-token generation.

The bank averages PER-IMAGE spatial standard deviations, not the standard
deviation of pooled pixels. Epsilon placement follows the spec's executable
pseudocode: sqrt((sum squared deviations + 1e-8) / HW).
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from runtime import preserve_rng

CHANNELS = (256, 512, 1024, 2048)
RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


class UnlabelledImages(Dataset):
    """Only paths and images; never constructs or returns target class labels."""
    def __init__(self, root, transform):
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())
        if not self.paths:
            raise ValueError(f"No images: {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))

    def manifest(self):
        records = [(p.relative_to(self.root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in self.paths]
        digest = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
        return {"count": len(records), "path_size_mtime_sha256": digest}


def spatial_statistics(feature):
    feature = feature.float()
    mean = feature.mean(dim=(-2, -1))
    spatial_size = feature.shape[-2] * feature.shape[-1]
    std = ((feature - mean[..., None, None]).square().sum(dim=(-2, -1)).add(1e-8) / spatial_size).sqrt()
    return mean, std


def preprocess_identity(preprocess):
    """Stable across processes: Compose repr embeds function memory addresses."""
    entries = []
    for transform in preprocess.transforms:
        if hasattr(transform, "__qualname__"):
            entries.append({"function": transform.__module__ + "." + transform.__qualname__})
        else:
            entries.append({"class": type(transform).__module__ + "." + type(transform).__qualname__,
                            "configuration": repr(transform)})
    return entries


class RN50StyleExtractor:
    """Temporary stage hooks; the ordinary image-encoding path is untouched."""
    def __init__(self, visual):
        self.visual = visual
        self.handles = []
        self.statistics = {}
        self.feature_shapes = {}

    def __enter__(self):
        if self.visual.training:
            raise ValueError("Frozen visual encoder must be in eval mode")
        for i in range(4):
            def collect(module, inputs, output, stage=i):
                if output.shape[1] != CHANNELS[stage]:
                    raise ValueError("RN50 stage channel mismatch")
                self.statistics[stage] = spatial_statistics(output)
                self.feature_shapes[stage] = list(output.shape)
            self.handles.append(getattr(self.visual, f"layer{i+1}").register_forward_hook(collect))
        return self

    @torch.no_grad()
    def __call__(self, images):
        if self.visual.training:
            raise ValueError("Cannot collect statistics with active visual BN")
        self.statistics.clear()
        embedding = self.visual(images)
        if len(self.statistics) != 4:
            raise RuntimeError("Missing RN50 stage output")
        return [self.statistics[i] for i in range(4)], embedding

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.statistics.clear()


def new_accumulator():
    return {"mu_sum": [torch.zeros(c, dtype=torch.float64) for c in CHANNELS],
            "std_sum": [torch.zeros(c, dtype=torch.float64) for c in CHANNELS], "count": 0}


def update_accumulator(acc, statistics):
    batch = statistics[0][0].shape[0]
    for stage, (mean, std) in enumerate(statistics):
        if mean.shape != (batch, CHANNELS[stage]) or std.shape != mean.shape:
            raise ValueError("Invalid per-image statistics shape")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise FloatingPointError("Invalid extracted statistics")
        acc["mu_sum"][stage] += mean.double().sum(0).cpu()
        acc["std_sum"][stage] += std.double().sum(0).cpu()
    acc["count"] += batch


def finalize(acc):
    if acc["count"] <= 0:
        raise ValueError("Empty style accumulator")
    return [(m.div(acc["count"]).float(), s.div(acc["count"]).float())
            for m, s in zip(acc["mu_sum"], acc["std_sum"])]


def pool_accumulators(accumulators):
    pooled = new_accumulator()
    for acc in accumulators:
        pooled["count"] += acc["count"]
        for stage in range(4):
            pooled["mu_sum"][stage] += acc["mu_sum"][stage]
            pooled["std_sum"][stage] += acc["std_sum"][stage]
    return pooled


def validate_bank(payload, expected=None):
    meta = payload["metadata"]
    if expected is not None and meta["cache_identity"] != expected:
        raise ValueError("Style Bank metadata mismatch; rebuild cache")
    entries = [*payload["sources"], payload["pooled"], payload["target"]]
    counts = meta["counts"]
    if len(entries) != len(meta["source_domain_order"]) + 2 or len(counts) != len(entries):
        raise ValueError("Style Bank domain count mismatch")
    if not all(n > 0 for n in counts) or counts[-2] != sum(counts[:-2]):
        raise ValueError("Style Bank sample count mismatch")
    for entry in entries:
        if len(entry) != 4:
            raise ValueError("Style Bank must have four stages")
        for c, (mean, std) in zip(CHANNELS, entry):
            if mean.shape != (c,) or std.shape != (c,) or mean.dtype != torch.float32 or std.dtype != torch.float32:
                raise ValueError("Style Bank shape/dtype mismatch")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
                raise ValueError("Style Bank nonfinite/negative statistics")
    for stage in range(4):
        for stat in range(2):
            weighted = sum(entries[i][stage][stat].double() * n for i, n in enumerate(counts[:-2])) / counts[-2]
            torch.testing.assert_close(entries[-2][stage][stat], weighted.float(), rtol=1e-5, atol=1e-6)


@torch.no_grad()
def build_style_bank(clip_model, preprocess, data_root, source_names, target_name, cache_dir,
                     batch_size=30, num_workers=4):
    """Complete deterministic scan, isolated from the baseline RNG stream."""
    names = [*source_names, target_name]
    datasets = [UnlabelledImages(Path(data_root) / name, preprocess) for name in names]
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = {"schema": 1, "backbone": "RN50", "checkpoint_sha256": RN50_SHA256,
                "dataset": "OfficeHome", "target_domain": target_name, "source_domain_order": list(source_names),
                "preprocess_id": preprocess_identity(preprocess), "precision": "FP32 statistics / FP64 accumulation",
                "std_definition": "sqrt((sum((F-mu)^2)+1e-8)/HW); mean of per-image std",
                "stage_shapes": [[c] for c in CHANNELS], "extractor_code_sha256": code_hash,
                "manifests": {name: dataset.manifest() for name, dataset in zip(names, datasets)}}
    cache_path = Path(cache_dir) / f"{target_name}.pt"
    if cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            validate_bank(payload, identity)
            return payload
        except (ValueError, AssertionError, KeyError, RuntimeError):
            print(f"Invalid style cache, rebuilding: {cache_path}", flush=True)
    bn_before = {k: v.clone() for k, v in clip_model.visual.named_buffers()}
    accumulators = []
    with preserve_rng(), RN50StyleExtractor(clip_model.visual) as extractor:
        for name, dataset in zip(names, datasets):
            print(f"Extracting fixed style: {name}, {len(dataset)} images", flush=True)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                num_workers=num_workers, pin_memory=True)
            acc = new_accumulator()
            for images in loader:
                statistics, _ = extractor(images.to(next(clip_model.parameters()).device))
                update_accumulator(acc, statistics)
            if acc["count"] != len(dataset):
                raise RuntimeError("Incomplete style scan")
            accumulators.append(acc)
    for key, value in clip_model.visual.named_buffers():
        torch.testing.assert_close(value, bn_before[key], rtol=0, atol=0)
    try:
        commit = os.environ.get("STYLE_CODE_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        commit = "uncommitted"
    pooled = pool_accumulators(accumulators[:-1])
    payload = {"sources": [finalize(a) for a in accumulators[:-1]], "pooled": finalize(pooled),
               "target": finalize(accumulators[-1]),
               "metadata": {"cache_identity": identity, "source_domain_order": list(source_names),
                            "target_domain": target_name, "counts": [a["count"] for a in accumulators[:-1]] + [pooled["count"], accumulators[-1]["count"]],
                            "created_utc": datetime.now(timezone.utc).isoformat(), "code_commit": commit}}
    validate_bank(payload, identity)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    return payload


class FixedStyleBank(nn.Module):
    def __init__(self, payload):
        super().__init__()
        validate_bank(payload)
        self.metadata = payload["metadata"]
        entries = [*payload["sources"], payload["pooled"], payload["target"]]
        self.n_domains = len(entries)
        for stage in range(4):
            self.register_buffer(f"mean_{stage}", torch.stack([e[stage][0] for e in entries]))
            self.register_buffer(f"std_{stage}", torch.stack([e[stage][1] for e in entries]))

    def entry(self, index):
        return [(getattr(self, f"mean_{s}")[index], getattr(self, f"std_{s}")[index]) for s in range(4)]


class DomainStyleProjector(nn.Module):
    def __init__(self, token_width=512, n_tokens=16):
        super().__init__()
        self.stage = nn.ModuleList([nn.Linear(2*c, token_width) for c in CHANNELS])
        for layer in self.stage:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        expansion = torch.zeros(n_tokens, 4)
        expansion[torch.arange(n_tokens), torch.arange(n_tokens) % 4] = 1
        self.expansion = nn.Parameter(expansion)
        self.calibrated = False

    def stage_tokens(self, domain_style):
        return torch.stack([layer(torch.cat([mean, std], dim=-1))
                            for layer, (mean, std) in zip(self.stage, domain_style)])

    def forward(self, domain_style):
        return self.expansion @ self.stage_tokens(domain_style)


@torch.no_grad()
def prompt_diagnostics(projector, bank):
    prompts = torch.stack([projector(bank.entry(i)) for i in range(bank.n_domains)])
    stages = torch.stack([projector.stage_tokens(bank.entry(i)) for i in range(bank.n_domains)])
    per_domain = [{"rms": p.square().mean().sqrt().item(), "mean": p.mean().item(),
                   "std": p.std(unbiased=False).item(), "max_abs": p.abs().max().item()} for p in prompts]
    return {"global_rms": prompts.square().mean().sqrt().item(), "domains": per_domain,
            "stage_rms": stages.square().mean(dim=(0, 2)).sqrt().tolist()}


@torch.no_grad()
def calibrate_initial_output(projector, bank, target_rms=0.02):
    if projector.calibrated:
        raise RuntimeError("RMS calibration may run only once")
    before = prompt_diagnostics(projector, bank)
    rms = before["global_rms"]
    if not torch.isfinite(torch.tensor(rms)) or rms <= 1e-10:
        raise FloatingPointError("Cannot calibrate invalid style output")
    factor = target_rms / rms
    for layer in projector.stage:
        layer.weight.mul_(factor)
        layer.bias.mul_(factor)
    projector.calibrated = True
    after = prompt_diagnostics(projector, bank)
    if abs(after["global_rms"] - target_rms) > 1e-6:
        raise FloatingPointError("RMS calibration mismatch")
    return {"before": before, "after": after, "factor": factor, "target_rms": target_rms}
