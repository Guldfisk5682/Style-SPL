"""Read-only geometry and SPL teacher metrics for saved states."""
import torch
import torch.nn.functional as F


def rms(value):
    return value.square().mean().sqrt().item()


def prompt_geometry(tokens, shared_class):
    flat = tokens.flatten(1).double()
    centered = tokens - tokens.mean(0, keepdim=True)
    unit = F.normalize(flat, dim=-1)
    centered_unit = F.normalize(centered.flatten(1).double(), dim=-1)
    shared_rms = rms(shared_class)
    common = tokens.mean(0)
    return {
        "flattened_cosine": (unit @ unit.T).tolist(),
        "mean_aligned_token_cosine": torch.einsum("dte,fte->dft", F.normalize(tokens.double(), dim=-1), F.normalize(tokens.double(), dim=-1)).mean(-1).tolist(),
        "centered_cosine": (centered_unit @ centered_unit.T).tolist(),
        "domain_prompt_rms": [rms(t) for t in tokens],
        "shared_class_prompt_rms": shared_rms,
        "domain_to_shared_class_rms_ratio": [rms(t) / shared_rms for t in tokens],
        "centered_domain_rms": [rms(t) for t in centered],
        "centered_domain_to_shared_class_rms_ratio": [rms(t) / shared_rms for t in centered],
        "common_style_rms": rms(common),
        "centered_style_rms": rms(centered),
        "centered_to_common_style_rms_ratio": rms(centered) / rms(common),
        "common_style_energy_fraction": common.square().mean().item() / tokens.square().mean().item(),
    }


def text_geometry(features):
    """Mean cosine of the SAME class, never class-averaged prototypes."""
    unit = F.normalize(features.double(), dim=-1)
    values = torch.einsum("dke,fke->dfk", unit, unit)
    quantiles = torch.quantile(values, torch.tensor([.05, .5, .95], dtype=torch.double), dim=-1)
    different_class = []
    for matrix in unit:
        cosine = matrix @ matrix.T
        mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
        different_class.append(cosine[mask].mean().item())
    return {"mean": values.mean(-1).tolist(), "std": values.std(-1, correction=0).tolist(),
            "minimum": values.min(-1).values.tolist(), "q05": quantiles[0].tolist(),
            "median": quantiles[1].tolist(), "q95": quantiles[2].tolist(),
            "different_class_within_domain_mean_cosine": different_class}, values


def stage_contributions(stage_tokens, expansion, names):
    """Exact linear domain-difference decomposition, with cross terms.

    A signed share assigns <stage contribution, total difference> / ||total||².
    Shares sum to one. Negative shares denote cancellation. Raw energy shares
    ignore cross terms and are explicitly reported as a separate metric.
    """
    pairs, grams = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            delta = (stage_tokens[i] - stage_tokens[j]).double()
            parts = torch.einsum("ts,se->ste", expansion.double(), delta)
            total = parts.sum(0)
            reconstructed = expansion.double() @ delta
            torch.testing.assert_close(total, reconstructed, rtol=1e-10, atol=1e-10)
            gram = parts.flatten(1) @ parts.flatten(1).T
            energy = total.square().sum()
            if energy <= 0:
                raise RuntimeError("Zero domain difference")
            signed = gram.sum(1) / energy
            torch.testing.assert_close(signed.sum(), torch.tensor(1., dtype=torch.double), rtol=1e-9, atol=1e-9)
            pairs.append({"a": names[i], "b": names[j], "total_difference_energy": energy.item(),
                          "signed_shares": signed.tolist(),
                          "standalone_energy_shares": (gram.diag() / gram.diag().sum()).tolist(),
                          "gram": gram.tolist()})
            grams.append(gram)
    def aggregate(selected):
        gram = sum(selected)
        return {"signed_shares": (gram.sum(1) / gram.sum()).tolist(),
                "standalone_energy_shares": (gram.diag() / gram.diag().sum()).tolist(),
                "gram": gram.tolist()}
    target_grams = [g for g, p in zip(grams, pairs) if p["b"] == names[-1] and p["a"] in names[:3]]
    return {"pairs": pairs, "all_pairs": aggregate(grams), "source_target_pairs": aggregate(target_grams)}


def teacher_scores(normalized, raw, source_texts, pooled_text, base_text, centroids, w_scale):
    """Retain the author's per-image, per-class source weights and logit mean."""
    distance = torch.stack([raw.square().sum(1, keepdim=True) + c.square().sum(1) - 2 * raw @ c.T for c in centroids])
    weights = torch.softmax(-w_scale * distance, dim=0)
    source_scores = torch.stack([normalized @ text.T for text in source_texts])
    base = normalized @ base_text.T
    pooled = normalized @ pooled_text.T
    weighted = (weights * source_scores).sum(0)
    # The divisor has no effect on argmax, but is required for confidence/entropy.
    combined = (base + pooled + weighted) / 3
    return {"base": base, "pooled": pooled, "weighted_source": weighted, "combined": combined}, weights, source_scores
