"""Forward-mode sensitivity of source prompts to actual target gradients.

Uses initial state + actual first-batch gradients (both pre-update). No virtual
optimizer update, no training, and no shared-class-context perturbation.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from style import DomainStyleProjector


def sensitivity(config, state, gradients, prefix, indices):
    model = DomainStyleProjector(config["M1"] * 0 + 512, config["M2"],
                                 config["projector_architecture"], config["bottleneck_dim"])
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    parameters = {k: p.detach() for k, p in model.named_parameters()}
    tangent = {k: -gradients[prefix+k] for k in parameters}
    descriptors = [[(state[f"style_bank.mean_{s}"][i], state[f"style_bank.std_{s}"][i])
                    for s in range(4)] for i in indices]
    def outputs(params):
        return torch.stack([torch.func.functional_call(model, params, (style,)) for style in descriptors])
    _, response = torch.func.jvp(outputs, (parameters,), (tangent,))
    return response.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {"protocol": "J_prompt(theta)[-gradient_target_objective], initial state and actual first training batch; no updates",
        "limits": ["A differential sensitivity, not an AdamW update or a teacher-logit gradient.",
                   "Only generator parameters are perturbed; shared class context held fixed."], "tasks": {}}
    for arm in ["r1_shared", "r2_pooled", "r3_split", "r4_silu32"]:
        for target in ["art", "clipart", "product", "real_world"]:
            task = args.run_root / arm / target
            initial_path = task / "mechanism/step0000.pt"
            gradient_path = task / "mechanism/gradient0001.pt"
            if not initial_path.exists() or not gradient_path.exists():
                continue
            config = json.loads((task / "config.json").read_text())
            state = torch.load(initial_path, map_location="cpu", weights_only=True)["prompt_state"]
            gradient = torch.load(gradient_path, map_location="cpu", weights_only=True)["target_objective_gradients"]
            source = sensitivity(config, state, gradient, "style_projector.", [0, 1, 2, 3, 4])
            actual = source.clone()
            if config["independent_pooled"]:
                actual[3].zero_()
            if config["split_projector"]:
                actual[4] = sensitivity(config, state, gradient, "target_style_projector.", [4])[0]
                assert (actual[:3] == 0).all()
            norms = actual.double().square().mean((1, 2)).sqrt()
            normalized = F.normalize(actual.flatten(1).double(), dim=-1)
            entry = {"domain_order": [*config["source_domain_order"], "pooled", target],
                "prompt_response_rms": norms.tolist(), "response_cosine": (normalized @ normalized.T).tolist(),
                "mean_source_to_target_response_norm_ratio": (norms[:3].mean()/norms[4]).item(),
                "source_prompt_exactly_unchanged": bool((actual[:3] == 0).all())}
            report["tasks"][f"{arm}/{target}"] = entry
            print(json.dumps({"arm": arm, "target": target, **entry}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
