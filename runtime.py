"""RNG isolation, observational numeric checks, and checkpoint state."""
import math
import random
from contextlib import contextmanager

import numpy as np
import torch


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def gradient_check(module, losses):
    for name, loss in losses.items():
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Nonfinite loss: {name}")
    norms = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Missing/nonfinite gradient: {name}")
        norms[name] = torch.linalg.vector_norm(parameter.grad, dtype=torch.float64).item()
    if not all(math.isfinite(n) for n in norms.values()) or math.hypot(*norms.values()) <= 0:
        raise FloatingPointError("Nonfinite/zero total gradient norm")
    return norms


def parameter_check(module):
    for name, tensor in module.state_dict().items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Nonfinite state tensor: {name}")


class ReplayableLoader:
    """Capture epoch-start RNG and consumed batch cursor without changing loading.

    Restoring replays the deterministic transforms and exact original sampler,
    including worker/base-seed setup, then restores checkpoint RNG. This is
    valid for the fixed CLIP transforms used here, not random worker transforms.
    """
    def __init__(self, loader):
        self.loader = loader
        self.iterator = None
        self.epoch_rng = None
        self.cursor = 0

    def next(self):
        if self.iterator is None:
            self.epoch_rng = rng_state()
            self.iterator = iter(self.loader)
            self.cursor = 0
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch_rng = rng_state()
            self.iterator = iter(self.loader)
            self.cursor = 0
            batch = next(self.iterator)
        self.cursor += 1
        return batch

    def state_dict(self):
        return {"epoch_rng": self.epoch_rng, "cursor": self.cursor}

    def load_state_dict(self, state):
        self.epoch_rng = state["epoch_rng"]
        self.cursor = state["cursor"]
        if self.epoch_rng is None:
            return
        with preserve_rng():
            restore_rng(self.epoch_rng)
            self.iterator = iter(self.loader)
            for _ in range(self.cursor):
                next(self.iterator)
