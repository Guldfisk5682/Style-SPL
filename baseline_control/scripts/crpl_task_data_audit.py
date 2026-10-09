"""Fail before optimization if a task reads an earlier task's data iterator."""
import hashlib
import json
from collections import Counter
from pathlib import Path


class TaskDataAudit:
    def __init__(self, output_root, data_root, sources, target, source_loader, target_loader, eval_loader):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(data_root).resolve()
        self.sources, self.target = list(sources), target
        self.source_loader, self.target_loader = source_loader, target_loader
        self.checked_steps = 0
        if target in self.sources or len(set(self.sources)) != len(self.sources):
            raise ValueError("Invalid source/target split")
        expected_target = self.data_root / target
        if Path(target_loader.dataset.root).resolve() != expected_target:
            raise ValueError("Target training dataset belongs to the wrong domain")
        if Path(eval_loader.dataset.root).resolve() != expected_target:
            raise ValueError("Target evaluation dataset belongs to the wrong domain")
        counts = Counter()
        paths = []
        for datum in source_loader.dataset.data:
            path = Path(datum.impath).resolve().relative_to(self.data_root)
            domain = path.parts[0]
            if domain == target or domain not in self.sources or self.sources[datum.domain] != domain:
                raise ValueError(f"Source sample outside the task split: {path}")
            counts[domain] += 1
            paths.append(path.as_posix())
        if set(counts) != set(self.sources):
            raise ValueError("Missing source domain")
        self.record({"phase": "initialization", "sources": self.sources, "target": target,
                     "source_counts": dict(counts), "target_count": len(target_loader.dataset),
                     "source_manifest_sha256": hashlib.sha256(json.dumps(paths).encode()).hexdigest(),
                     "held_out_target_excluded_from_source": True})

    def record(self, values):
        with (self.output_root / "data_protocol_checks.jsonl").open("a") as file:
            file.write(json.dumps({"target": self.target, **values}) + "\n")

    def check(self, source_iterator, target_iterator, step):
        if source_iterator._dataset is not self.source_loader.dataset:
            self.record({"phase": "failure", "step": step, "reason": "stale source iterator"})
            raise RuntimeError("Source iterator retained from a different target task")
        if target_iterator._dataset is not self.target_loader.dataset:
            self.record({"phase": "failure", "step": step, "reason": "stale target iterator"})
            raise RuntimeError("Target iterator retained from a different target task")
        self.checked_steps += 1
        if step in (1, 2) or step % 200 == 0:
            self.record({"phase": "step", "step": step, "current_source_iterator": True,
                         "current_target_iterator": True, "checked_steps": self.checked_steps})

    def finish(self, expected_steps):
        if self.checked_steps != expected_steps:
            raise RuntimeError("Data protocol audit did not cover every training step")
        self.record({"phase": "finished", "checked_steps": self.checked_steps, "protocol_valid": True})
