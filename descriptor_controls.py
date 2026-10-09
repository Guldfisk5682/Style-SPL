"""Fixed input controls for R4; no learned parameters or target labels.

Only the four actual-domain rows change. The unused Pooled descriptor stays
real, and the independently learned Pooled tokens always use the real-bank
initialization. Random codes match the pooled channel moments of each stage
and statistic block across ALL four domains, not each domain's own moments.
Thus no domain-specific measured scalar statistics are retained in a code.
"""
import copy
import hashlib

import torch


def bank_digest(bank):
    digest = hashlib.sha256()
    for name, tensor in sorted(bank.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def descriptor_control(bank, target_name, mode="real", seed=20261010):
    if mode not in ("real", "shuffled", "random"):
        raise ValueError(f"Unknown descriptor control: {mode}")
    sources = list(bank.metadata["source_domain_order"])
    rows = {name: i for i, name in enumerate(sources)}
    rows[target_name] = len(sources) + 1
    names = sorted(rows)
    if len(names) != 4:
        raise ValueError("R4 controls require four actual domains")
    result = copy.deepcopy(bank)
    report = {"mode": mode, "seed": seed, "canonical_domains": names,
              "reference_buffer_sha256": bank_digest(bank),
              "pooled_descriptor_changed": False,
              "uses_target_labels": False, "blocks": []}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = list(range(4))
    if mode == "shuffled":
        while any(i == j for i, j in enumerate(permutation)):
            permutation = torch.randperm(4, generator=generator).tolist()
    report["domain_to_descriptor"] = {
        name: names[permutation[i]] if mode == "shuffled" else name
        for i, name in enumerate(names)}
    report["random_matching"] = (
        "Each fixed independent Gaussian code matches the global channel mean "
        "and population std of the real four-domain block; preserves aggregate "
        "scale, not channel layout, domain-specific moments, or domain geometry"
        if mode == "random" else None)
    for stage in range(4):
        for statistic in ("mean", "std"):
            buffer_name = f"{statistic}_{stage}"
            original = getattr(bank, buffer_name)
            controlled = getattr(result, buffer_name)
            # Codes are generated with a private CPU RNG. Match moments on
            # CPU too, then copy to the bank device; construction is identical
            # regardless of where the frozen bank was originally loaded.
            actual = torch.stack([original[rows[n]] for n in names]).cpu().double()
            center, spread = actual.mean(), actual.std(correction=0)
            for i, name in enumerate(names):
                if mode == "shuffled":
                    controlled[rows[name]].copy_(original[rows[names[permutation[i]]]])
                elif mode == "random":
                    code = torch.randn(original.shape[1], generator=generator, dtype=torch.float64)
                    code = (code - code.mean()) / code.std(correction=0)
                    controlled[rows[name]].copy_((center + spread * code).to(controlled))
            changed = torch.stack([controlled[rows[n]] for n in names]).double()
            report["blocks"].append({"stage": stage + 1, "statistic": statistic,
                "real_mean": center.item(), "real_std": spread.item(),
                "real_rms": actual.square().mean().sqrt().item(),
                "control_mean": changed.mean().item(),
                "control_std": changed.std(correction=0).item(),
                "control_rms": changed.square().mean().sqrt().item(),
                "per_domain_mean": changed.mean(-1).tolist(),
                "per_domain_std": changed.std(-1, correction=0).tolist()})
    report["effective_buffer_sha256"] = bank_digest(result)
    return result, report
