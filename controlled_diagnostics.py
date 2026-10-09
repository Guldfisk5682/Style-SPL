"""Observational probes: never alter training RNG, gradients, or the fixed bank."""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from runtime import preserve_rng
from style import UnlabelledImages, preprocess_identity, validate_bank
from scripts.round_analysis_metrics import prompt_geometry, text_geometry, teacher_scores


def load_frozen_bank(root, target, sources, data_root, preprocess):
    path = Path(root) / target / "style_bank.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_bank(payload)
    meta = payload["metadata"]
    identity = meta["cache_identity"]
    if meta["source_domain_order"] != sources or identity["target_domain"] != target:
        raise ValueError("Frozen bank task mismatch")
    if identity["preprocess_id"] != preprocess_identity(preprocess) or identity["backbone"] != "RN50":
        raise ValueError("Frozen bank encoder/preprocess mismatch")
    for name in [*sources, target]:
        if identity["manifests"][name] != UnlabelledImages(Path(data_root) / name, preprocess).manifest():
            raise ValueError(f"Frozen bank dataset changed: {name}")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.norm() == 0 or b.norm() == 0:
        return None
    return F.cosine_similarity(a, b, dim=0).item()


def objective_gradient_probe(prompt, objectives, path, step):
    """Two reverse passes over the actual pre-update graph; .grad stays untouched."""
    named = list(prompt.named_parameters())
    parameters = [p for _, p in named]
    original = [None if p.grad is None else p.grad.clone() for p in parameters]
    with preserve_rng():
        source = torch.autograd.grad(objectives[0], parameters, retain_graph=True, allow_unused=True)
        target = torch.autograd.grad(objectives[1], parameters, retain_graph=True, allow_unused=True)
    a, b = {}, {}
    report = {"step": step, "state": "pre-update", "per_parameter": {}}
    for (name, p), ga, gb, old in zip(named, source, target, original):
        if old is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, old, rtol=0, atol=0)
        ga = torch.zeros_like(p) if ga is None else ga
        gb = torch.zeros_like(p) if gb is None else gb
        if not torch.isfinite(ga).all() or not torch.isfinite(gb).all():
            raise FloatingPointError("Nonfinite objective gradient")
        report["per_parameter"][name] = {"source_norm": ga.double().norm().item(),
            "target_norm": gb.double().norm().item(), "cosine": cosine(ga, gb)}
        if "projector" in name:
            a[name], b[name] = ga.detach().cpu(), gb.detach().cpu()
    for prefix in ("style_projector.", "target_style_projector."):
        keys = [k for k in a if k.startswith(prefix)]
        if not keys:
            continue
        ga, gb = torch.cat([a[k].flatten() for k in keys]), torch.cat([b[k].flatten() for k in keys])
        report[prefix[:-1]] = {"source_norm": ga.double().norm().item(),
            "target_norm": gb.double().norm().item(), "cosine": cosine(ga, gb)}
    if prompt.split_projector:
        keys = [k for k in a if k.startswith("style_projector.")]
        ga = torch.cat([a[k].flatten() for k in keys])
        gb = torch.cat([b["target_" + k].flatten() for k in keys])
        report["matched_source_target_projector_gradient_cosine"] = cosine(ga, gb)
        if report["style_projector"]["target_norm"] != 0 or report["target_style_projector"]["source_norm"] != 0:
            raise AssertionError("Source/target projector objective isolation violated")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Full FP32 gradients, not only scalars, retained for future direction analysis.
    torch.save({"summary": report, "source_objective_gradients": a,
                "target_objective_gradients": b}, path)
    return report


class TrajectoryProbe:
    def __init__(self, root, prompt, encoder, clip_model, base_tokens, args):
        self.root = Path(root) / "mechanism"
        self.root.mkdir(exist_ok=True)
        self.prompt, self.encoder, self.args = prompt, encoder, args
        self.previous_tokens = self.previous_texts = None
        self.reference_tokens = None
        self.reference_step = None
        self.bank_initial = {k: v.clone() for k, v in prompt.style_bank.state_dict().items()}
        self.cache = None
        self.base_text = None
        if args.diagnostic_image_cache:
            cache_path = args.diagnostic_image_cache / f"{prompt.target_name}.pt"
            self.cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            # Ground truth is used only below for offline diagnostic accuracy.
            with torch.no_grad(), preserve_rng():
                self.base_text = F.normalize(clip_model.encode_text(base_tokens).float(), dim=-1)

    @torch.no_grad()
    def snapshot(self, step, running_means, running_count):
        with preserve_rng():
            return self._snapshot(step, running_means, running_count)

    def _snapshot(self, step, running_means, running_count):
        p, encoder = self.prompt, self.encoder
        for k, v in p.style_bank.state_dict().items():
            torch.testing.assert_close(v, self.bank_initial[k], rtol=0, atol=0)
        roles = [("source", i) for i in range(len(p.source_names))] + [("pooled", None), ("target", None)]
        names = [*p.source_names, "pooled", p.target_name]
        tokens = torch.stack([p.domain_tokens(kind, i)[0] for kind, i in roles])
        texts = torch.stack([encoder.forward_txt(p.assemble_domain_prompt(kind, i), p.tokenized_prompts)
                             for kind, i in roles])
        token_cpu, text_cpu = tokens.cpu(), texts.cpu()
        geometry, per_class = text_geometry(text_cpu)
        report = {"step": step, "state": "initial" if step == 0 else "post-update", "domain_order": names,
                  "prompt": prompt_geometry(token_cpu, p.ctx_cls.cpu()), "text": geometry,
                  "bank_unchanged": True}
        if self.reference_tokens is None:
            self.reference_tokens = token_cpu.clone()
            self.reference_step = step
        delta0 = token_cpu - self.reference_tokens
        report["displacement_reference_step"] = self.reference_step
        report["displacement_from_reference_cosine"] = (F.normalize(delta0.flatten(1), dim=-1) @ F.normalize(delta0.flatten(1), dim=-1).T).tolist()
        if self.previous_tokens is not None:
            delta = token_cpu - self.previous_tokens
            report["interval_prompt_movement_cosine"] = (F.normalize(delta.flatten(1), dim=-1) @ F.normalize(delta.flatten(1), dim=-1).T).tolist()
            report["interval_prompt_movement_rms"] = delta.square().mean((1, 2)).sqrt().tolist()
            report["interval_text_same_domain_cosine"] = (text_cpu * self.previous_texts).sum(-1).mean(-1).tolist()
        if self.cache is not None:
            images = self.cache["normalized"].to(tokens.device)
            raw = self.cache["raw"].to(tokens.device)
            logits = torch.stack([images @ t.T for t in texts])
            scale = encoder.logit_scale.exp()
            report["live_domain_accuracy_offline_only"] = [(l.argmax(-1).cpu() == self.cache["labels"]).float().mean().item() for l in logits]
            report["teacher"] = {"weighted_enabled": bool((running_count > 0).all())}
            scores = {"base": images @ self.base_text.T, "pooled": logits[-2]}
            if (running_count > 0).all():
                scores, weights, _ = teacher_scores(images, raw, texts[:3], texts[3], self.base_text,
                                                   running_means, self.args.w_scale)
                report["teacher"]["routing_mean_max"] = weights.max(0).values.mean().item()
            else:
                scores["combined"] = (scores["base"] + scores["pooled"]) / 2
            probability = (scale * scores["combined"]).softmax(-1)
            log_student = (scale * logits[-1]).log_softmax(-1)
            report["teacher"]["accuracy_offline_only"] = {k: (v.argmax(-1).cpu() == self.cache["labels"]).float().mean().item() for k, v in scores.items()}
            report["teacher"]["kl_teacher_to_student"] = (probability * (probability.clamp_min(1e-38).log() - log_student)).sum(-1).mean().item()
            report["teacher"]["kl_student_to_teacher"] = (log_student.exp() * (log_student - probability.clamp_min(1e-38).log())).sum(-1).mean().item()
            report["teacher"]["student_teacher_prediction_agreement"] = (logits[-1].argmax(-1) == scores["combined"].argmax(-1)).float().mean().item()
            report["counterfactual_descriptor_swap"] = []
            if step == 0 or step % self.args.counterfactual_step == 0 or step == self.args.prompt_iteration:
                for recipient, (kind, index) in enumerate(roles):
                    for donor in range(len(roles)):
                        if donor == recipient:
                            continue
                        altered = encoder.forward_txt(p.assemble_domain_prompt(kind, index, donor), p.tokenized_prompts)
                        alt_logits = images @ altered.T
                        difference = scale * (alt_logits - logits[recipient])
                        # Center per image: common logit offsets cannot change classification.
                        centered = difference - difference.mean(-1, keepdim=True)
                        alt_prob = (scale * alt_logits).softmax(-1)
                        old_log = (scale * logits[recipient]).log_softmax(-1)
                        report["counterfactual_descriptor_swap"].append({"recipient": names[recipient], "donor": names[donor],
                            "text_same_class_cosine": (texts[recipient] * altered).sum(-1).mean().item(),
                            "scaled_logits_delta_rms": difference.square().mean().sqrt().item(),
                            "centered_scaled_logits_delta_rms": centered.square().mean().sqrt().item(),
                            "prediction_flip_rate": (alt_logits.argmax(-1) != logits[recipient].argmax(-1)).float().mean().item(),
                            "kl_altered_to_original": (alt_prob * (alt_prob.clamp_min(1e-38).log() - old_log)).sum(-1).mean().item(),
                            "descriptor_used": not (kind == "pooled" and p.independent_pooled)})
        path = self.root / f"step{step:04d}"
        torch.save({"step": step, "domain_order": names, "domain_prompts": token_cpu,
                    "text_features": text_cpu, "prompt_state": {k: v.detach().cpu() for k, v in p.state_dict().items()},
                    "running_means": running_means.cpu(), "running_count": running_count.cpu(),
                    "target_feature_count": p.count, "same_class_cosine_per_class": per_class}, path.with_suffix(".pt"))
        path.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        self.previous_tokens, self.previous_texts = token_cpu.clone(), text_cpu.clone()
        return report
