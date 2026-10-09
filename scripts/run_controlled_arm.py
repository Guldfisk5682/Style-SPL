"""Execute one immutable arm manifest in a dedicated worktree/runtime."""
import json
import subprocess
import sys
from pathlib import Path


manifest = json.loads((Path(__file__).resolve().parents[1] / "configs/controlled_arm.json").read_text())
workspace = Path.home() / "workspace"
base = workspace / "Style-SPL"
arguments = [sys.executable, "-u", "train.py", "--data_root", str(workspace / "da_lab/data/office_home"),
    "--output_dir", str(base / "runs/controlled_20261009_seed1" / manifest["name"]),
    "--fixed_bank_root", str(base / "runs/s1_protocol_checked_20261009_seed1"),
    "--diagnostic_image_cache", str(base / "runs/analysis_round_20261009/image_embeddings"),
    "--mechanism_step", "100", "--counterfactual_step", "200", *manifest["flags"]]
print(json.dumps({"arm": manifest, "command": arguments}), flush=True)
subprocess.run(arguments, check=True)
