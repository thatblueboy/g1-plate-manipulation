"""Default evaluated policy: ACT trained on the demonstrations plus round-1
human recoveries (100k steps), temporal ensembling with k=0.01.

This was the best ACT configuration on the private validation block
(52/60, seeds 6,000,000+; report Section "Improvement"). It needs PyTorch, so
run it in the model environment (see submission/act/requirements.txt):

    MUJOCO_GL=egl uv run advaitic-eval --policy submission.policy:make_policy \
        --policy-python $PWD/.venv-act/bin/python --method act-recovery-r1 \
        --checkpoint act-recovery-r1-last.pt-3089b880 --episodes 30 --out results/final

The checkpoint is not stored in git. Place it at
checkpoints/act-recovery-r1/last.pt, or point ACT_CHECKPOINT at it.
ACT_MODE, ACT_ENSEMBLE_K and ACT_EXEC_HORIZON are honoured as in
submission/act/policy.py.
"""

import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / "checkpoints/act-recovery-r1/last.pt"
DEFAULT_SHA256 = "3089b88081ea3ab949544b51fc6d2f2f5b626f285903c64183ea5bcbef5dab44"


def make_policy():
    checkpoint = Path(os.environ.get("ACT_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"ACT checkpoint not found: {checkpoint}. Copy act-recovery-r1/last.pt "
            f"(sha256 {DEFAULT_SHA256}) there or set ACT_CHECKPOINT."
        )
    if checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve():
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != DEFAULT_SHA256:
            raise ValueError(f"{checkpoint} has sha256 {digest}, expected {DEFAULT_SHA256}")

    from submission.act.policy import ACTPolicy

    return ACTPolicy(checkpoint=str(checkpoint))
