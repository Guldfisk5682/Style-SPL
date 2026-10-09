"""Read-only SPL observations; no extra model forward, optimizer, or RNG use."""
import csv
import json
from pathlib import Path


class OfficialMetrics:
    def __init__(self, root, target):
        self.root, self.target = Path(root), target

    def append(self, filename, values):
        path = self.root / filename
        exists = path.exists()
        with path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(values))
            if not exists:
                writer.writeheader()
            writer.writerow(values)

    def update(self, step, total, pooled, domain_sum, domains, soft, teacher, coverage, lr):
        if step % 10:
            return
        teacher = teacher.detach()
        self.append("train_metrics.csv", {
            "step": step, "train/loss_total": total.detach().item(),
            "train/loss_source_pooled": pooled.detach().item(),
            "train/loss_source_domain_avg": domain_sum.detach().item() / domains,
            "train/loss_target_soft": soft.detach().item(),
            "train/teacher_entropy": (-(teacher * teacher.clamp_min(1e-12).log()).sum(-1)).mean().item(),
            "train/teacher_confidence_max_mean": teacher.max(-1).values.mean().item(),
            "train/source_centroid_coverage": coverage,
            "train/weighted_teacher_enabled": int(coverage == 1), "train/lr": lr,
        })

    def evaluation(self, step, accuracy):
        self.append("eval_metrics.csv", {"step": step, "eval/target_top1_instant": accuracy})

    def finish(self, accuracy, best, steps):
        (self.root / "result.json").write_text(json.dumps({
            "target": self.target, "style_spl_enabled": False, "steps": steps,
            "final_accuracy": accuracy, "best_instant_diagnostic_only": best,
            "protocol_valid": True, "implementation": "patched official SPL",
        }, indent=2) + "\n")
