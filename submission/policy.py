"""Default evaluated policy: ACT trained on the demonstrations plus all 22
human recoveries (rounds 1 and 2, 100k steps), temporal ensembling with k=0.01.

This was the best configuration on the private validation block (54/60,
seeds 6,000,000+, 60-episode plan; see report/report.tex, "Summary and selected
configuration"). It needs PyTorch, so it runs in the separate model
environment .venv-act (setup in README.md, "Running the submitted policy"):

    MUJOCO_GL=egl uv run advaitic-eval --policy submission.policy:make_policy \
        --policy-python $PWD/.venv-act/bin/python --method act-recovery-r2 \
        --checkpoint act-recovery-r2-100k-1752981f --episodes 30 --out results/final

The checkpoint is stored with Git LFS at submission/act/weights/ (run
`git lfs pull` if it is a small pointer file). ACT_CHECKPOINT overrides it;
ACT_MODE, ACT_ENSEMBLE_K and ACT_EXEC_HORIZON are honoured as in
submission/act/policy.py.
"""

import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / "submission/act/weights/act-recovery-r2-100k.pt"
DEFAULT_SHA256 = "1752981f7307abfcd7465f63d2dfbec2efd6fee36ff39b0f8cdc7e895b19143e"


def make_policy():
    checkpoint = Path(os.environ.get("ACT_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ACT checkpoint not found: {checkpoint}")
    if checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve():
        data = checkpoint.read_bytes()
        if data.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"{checkpoint} is a Git LFS pointer; install git-lfs and run `git lfs pull`."
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest != DEFAULT_SHA256:
            raise ValueError(f"{checkpoint} has sha256 {digest}, expected {DEFAULT_SHA256}")

    from submission.act.policy import ACTPolicy

    return ACTPolicy(checkpoint=str(checkpoint))
