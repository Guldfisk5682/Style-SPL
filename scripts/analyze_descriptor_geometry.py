"""Input geometry of the fixed controls; CPU, no class labels or training."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from descriptor_controls import descriptor_control
from style import FixedStyleBank


def geometry(x):
    x = x.double()
    common = x.mean(0)
    energy = x.square().mean()
    centered = (x-common).square().mean()
    cosine = F.normalize(x, dim=-1) @ F.normalize(x, dim=-1).T
    pairs = [cosine[i, j].item() for i in range(x.shape[0]) for j in range(i+1, x.shape[0])]
    return {"rms": energy.sqrt().item(), "centered_energy_fraction": (centered/energy).item(),
            "pair_cosine_mean": sum(pairs)/len(pairs), "pair_cosine_matrix": cosine.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank_root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = {"conditions": {}, "limits": "Input geometry does not itself establish useful class information or a unique cause of model behavior"}
    for mode in ("real", "shuffled", "random"):
        tasks = {}
        for target in ("art", "clipart", "product", "real_world"):
            real = FixedStyleBank(torch.load(args.bank_root / target / "style_bank.pt", map_location="cpu", weights_only=True))
            controlled, metadata = descriptor_control(real, target, mode)
            stages = []
            for stage in range(4):
                x = torch.cat([getattr(controlled, f"mean_{stage}"), getattr(controlled, f"std_{stage}")], dim=-1)
                stages.append({"stage": stage+1, "sources": geometry(x[:3]),
                               "all_actual_domains": geometry(x[[0, 1, 2, 4]])})
            tasks[target] = {"stages": stages, "control": metadata}
        report["conditions"][mode] = tasks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({mode: [row["sources"]["pair_cosine_mean"] for row in tasks["art"]["stages"]]
                      for mode, tasks in report["conditions"].items()}))


if __name__ == "__main__":
    main()
