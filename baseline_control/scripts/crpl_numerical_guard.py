"""Observe CRPL numerics without changing gradients, updates, or RNG state."""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch


class NumericalGuard:
    def __init__(self, output_root, target, model):
        self.root = Path(output_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.target = target
        self.parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.checked_updates = 0
        self.pending = {}
        stats = {}
        for name, param in self.parameters:
            self._finite(param, "initial_parameter", name, 0)
            value = param.detach()
            stats[name] = {"numel": value.numel(), "mean": value.mean().item(),
                           "std": value.std(correction=0).item()}
        self._record({"phase": "initialization", "step": 0, "parameters": stats})

    def _record(self, record):
        record = {"time": datetime.now().isoformat(), "target": self.target, **record}
        with (self.root / "numerical_checks.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _fail(self, phase, name, step, detail):
        record = {"phase": "failure", "check": phase, "tensor": name,
                  "step": step, "detail": detail}
        self._record(record)
        (self.root / "numerical_failure.json").write_text(json.dumps(
            {"target": self.target, **record}, indent=2) + "\n")
        raise FloatingPointError(f"{self.target} step {step}: {phase} {name}: {detail}")

    def _finite(self, tensor, phase, name, step):
        if not torch.isfinite(tensor.detach()).all().item():
            self._fail(phase, name, step, "NaN or Inf")

    @staticmethod
    def _log_step(step):
        return step in (1, 2, 5, 10, 20, 50, 100) or step % 200 == 0

    def before_backward(self, step, losses):
        for name, loss in losses.items():
            self._finite(loss, "loss", name, step)
        self.pending = {"step": step}
        if self._log_step(step):
            self.pending["losses"] = {n: t.detach().item() for n, t in losses.items()}

    def before_update(self, step):
        norms = {}
        for name, param in self.parameters:
            if param.grad is None:
                self._fail("gradient", name, step, "missing gradient")
            self._finite(param.grad, "gradient", name, step)
            # Accumulate the norm in float64 to avoid overflow in float32.
            norm = torch.linalg.vector_norm(param.grad.detach(), dtype=torch.float64).item()
            if not math.isfinite(norm):
                self._fail("gradient_norm", name, step, "nonfinite norm")
            norms[name] = norm
        total_norm = math.hypot(*norms.values())
        if total_norm == 0 or not math.isfinite(total_norm):
            self._fail("gradient_norm", "all_parameters", step, "zero or nonfinite norm")
        if self._log_step(step):
            self.pending.update(gradient_norms=norms, gradient_norm=total_norm)

    def after_update(self, step):
        for name, param in self.parameters:
            self._finite(param, "updated_parameter", name, step)
        self.checked_updates += 1
        if self._log_step(step):
            self._record({"phase": "update", "all_gradients_finite": True,
                          "all_parameters_finite": True, "checked_updates": self.checked_updates,
                          "parameter_tensors": len(self.parameters), **self.pending})
            print(f"GRAD_CHECK target={self.target} step={step} "
                  f"loss={self.pending['losses']['total']:.6f} "
                  f"grad_norm={self.pending['gradient_norm']:.6f} "
                  f"parameters={len(self.parameters)} finite=True", file=sys.stderr, flush=True)
        self.pending = {}

    def finish(self, model, steps):
        if self.checked_updates != steps:
            self._fail("coverage", "updates", steps, "not every optimizer update was checked")
        for name, buffer in model.named_buffers():
            self._finite(buffer, "final_buffer", name, steps)
        if model.count <= 0:
            self._fail("feature_average", "count", steps, "no averaged text features")
        self._record({"phase": "finished", "step": steps, "checked_updates": self.checked_updates,
                      "all_buffers_finite": True, "feature_average_count": model.count})
