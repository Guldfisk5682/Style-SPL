import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from task_data_audit import TaskDataAudit


class ProtocolTests(unittest.TestCase):
    def fixtures(self, root, target, sources):
        data = [SimpleNamespace(impath=str(root / domain / "class0/image.jpg"), domain=i) for i, domain in enumerate(sources)]
        source = SimpleNamespace(dataset=SimpleNamespace(data=data))
        class Target:
            def __init__(self): self.root = root / target
            def __len__(self): return 30
        target_loader = SimpleNamespace(dataset=Target())
        return source, target_loader

    def test_rejects_stale_iterators_at_task_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_source, first_target = self.fixtures(root, "art", ["clipart", "product", "real_world"])
            source, target = self.fixtures(root, "clipart", ["art", "product", "real_world"])
            audit = TaskDataAudit(root / "audit", root, ["art", "product", "real_world"], "clipart", source, target, target)
            with self.assertRaises(RuntimeError):
                audit.check(SimpleNamespace(_dataset=first_source.dataset), SimpleNamespace(_dataset=target.dataset), 1)
            with self.assertRaises(RuntimeError):
                audit.check(SimpleNamespace(_dataset=source.dataset), SimpleNamespace(_dataset=first_target.dataset), 1)
            audit.check(SimpleNamespace(_dataset=source.dataset), SimpleNamespace(_dataset=target.dataset), 1)
            audit.finish(1)

    def test_rejects_held_out_target_in_labelled_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = self.fixtures(root, "clipart", ["art", "product", "real_world"])
            source.dataset.data[0].impath = str(root / "clipart/class0/image.jpg")
            with self.assertRaises(ValueError):
                TaskDataAudit(root / "audit", root, ["art", "product", "real_world"], "clipart", source, target, target)


if __name__ == "__main__": unittest.main()
