"""Execute patched author batch-fetch control flow on real path inventories.

    No images are decoded and no model/optimizer is run. The actual patched
    reset and try/next statements are compiled from the author train function.
"""
import argparse
import ast
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.data import Dataset, DataLoader
from dataset import MultiSourceDataset
from samplers import RandomDomainSampler
from runtime import fix_random_seed
from style import UnlabelledImages
from task_data_audit import TaskDataAudit

DOMAINS = ["art", "clipart", "product", "real_world"]


class SourcePaths(Dataset):
    def __init__(self, dataset):
        self.data = dataset.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item.impath, item.label, item.domain


class TargetPaths(Dataset):
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.paths = UnlabelledImages(root, None).paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return str(self.paths[index]), -1  # no target GT, and no pseudo-label inference


def sampling_body(path):
    tree = ast.parse(path.read_text())
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    task = next(n for n in ast.walk(train) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "target_name")
    resets = [n for n in task.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value is None
              and any(isinstance(t, ast.Name) and t.id in ("source_iter", "target_iter") for t in n.targets)]
    if len(resets) != 2:
        raise RuntimeError("Author code lacks both per-target iterator resets")
    step_loop = next(n for n in ast.walk(task) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step")
    fetch = [n for n in step_loop.body if isinstance(n, ast.Try) and
             any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "next" for c in ast.walk(n))]
    if len(fetch) != 2:
        raise RuntimeError("Cannot find the exact author source/target fetch statements")
    return (compile(ast.fix_missing_locations(ast.Module(body=resets, type_ignores=[])), str(path), "exec"),
            compile(ast.fix_missing_locations(ast.Module(body=fetch, type_ignores=[])), str(path), "exec"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author_main", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resets, fetch = sampling_body(args.author_main)
    fix_random_seed(1)
    scope = {}  # Retained scope, exactly like the author's multi-target train().
    results = []
    for target in DOMAINS:
        sources = [d for d in DOMAINS if d != target]
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = MultiSourceDataset(str(args.data_root), sources, None)
        source_loader = DataLoader(SourcePaths(dataset), batch_size=30,
                                   sampler=RandomDomainSampler(dataset.data, 30, 3), num_workers=0)
        target_loader = DataLoader(TargetPaths(args.data_root / target), batch_size=30, shuffle=True, drop_last=True)
        audit = TaskDataAudit(args.output.parent / target, args.data_root, sources, target,
                             source_loader, target_loader, target_loader)
        scope.update(source_train_loader=source_loader, target_train_loader=target_loader)
        exec(resets, scope)
        for step in range(1, 6):
            exec(fetch, scope)
            audit.check(scope["source_iter"], scope["target_iter"], step)
            assert all(Path(p).resolve().relative_to(args.data_root.resolve()).parts[0] == target for p in scope["target_data"])
            assert all(Path(p).resolve().relative_to(args.data_root.resolve()).parts[0] == sources[int(i)] for p, i in zip(scope["source_data"], scope["source_domain"]))
        audit.finish(5)
        results.append({"target": target, "source_domains": sources, "checked_batches": 5,
                        "actual_source_paths_valid": True, "actual_target_paths_valid": True})
    report = {"protocol_valid": True, "author_control_flow_executed": True,
              "model_updates": 0, "tasks": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
