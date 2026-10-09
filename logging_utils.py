import csv
import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from style import prompt_diagnostics


class MetricLogger:
    def __init__(self, root, names, append=False):
        self.root = Path(root)
        self.names = names
        self.root.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.root / "tensorboard"))
        self.files = {}
        self.csv = {}
        self.append = append

    def write(self, kind, step, values):
        if kind not in self.csv:
            path = self.root / f"{kind}_metrics.csv"
            columns = ["step", *sorted(values)]
            exists = path.exists() and self.append
            if exists:
                with path.open() as file:
                    columns = next(csv.reader(file))
            file = path.open("a" if exists else "w", newline="")
            self.files[kind] = file
            self.csv[kind] = csv.DictWriter(file, fieldnames=columns, restval="")
            if not exists:
                self.csv[kind].writeheader()
        self.csv[kind].writerow({"step": step, **values})
        self.files[kind].flush()
        for key, value in values.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        self.writer.close()
        for file in self.files.values():
            file.close()


@torch.no_grad()
def style_metrics(prompt, norms):
    if not prompt.style_spl_enabled:
        return {}
    stats = prompt_diagnostics(prompt.style_projector, prompt.style_bank)
    values = {"style/prompt_rms_global": stats["global_rms"]}
    for i, domain in enumerate(stats["domains"]):
        for key, value in domain.items():
            values[f"style/prompt_{key}_domain_{i}"] = value
    for i, rms in enumerate(stats["stage_rms"]):
        values[f"style/stage_token_rms_{i+1}"] = rms
        values[f"style/grad_norm_stage_{i+1}"] = (norms.get(f"style_projector.stage.{i}.weight", 0)**2 + norms.get(f"style_projector.stage.{i}.bias", 0)**2)**0.5
    values["style/grad_norm_expansion"] = norms.get("style_projector.expansion", 0)
    values["style/grad_norm_class_prompt"] = norms.get("ctx_cls", 0)
    return values


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
