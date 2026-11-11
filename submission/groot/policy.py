"""GR00T N1.7 policy for the Advaitic evaluator. Runs in the GR00T venv:

    GROOT_CHECKPOINT=/path/to/checkpoint-N MUJOCO_GL=egl uv run advaitic-eval \
        --policy submission.groot.policy:make_policy \
        --policy-python /path/to/Isaac-GR00T/.venv/bin/python \
        --method groot-n1.7 --checkpoint checkpoint-N --episodes 60 --out results/<new-dir>

Environment variables (read in the policy process):
  GROOT_CHECKPOINT      fine-tuned checkpoint directory (required)
  GROOT_EMBODIMENT      new_embodiment (default) or unitree_g1; must match training
  GROOT_EXEC_HORIZON    actions of each predicted chunk executed per query (default 8)
  GROOT_DEVICE          torch device (default cuda:0)

Uses only obs["rgb"] and obs["state"]; the evaluator executes the returned
(H, 20) chunk before asking again. Constant action dims and the [-1, 1]
normalisation come from submission/groot/layout.py and the public
action_layout.json.
"""

import os

import numpy as np

from submission.groot.layout import ActionNormaliser, from_groot_action, to_groot_obs


class GrootPolicy:
    def __init__(self):
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy import Gr00tPolicy

        checkpoint = os.environ.get("GROOT_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError("set GROOT_CHECKPOINT to a fine-tuned checkpoint directory")
        self.exec_horizon = int(os.environ.get("GROOT_EXEC_HORIZON", 8))
        tag = os.environ.get("GROOT_EMBODIMENT", "new_embodiment").upper()
        self.policy = Gr00tPolicy(
            model_path=checkpoint,
            embodiment_tag=EmbodimentTag.resolve(tag),
            device=os.environ.get("GROOT_DEVICE", "cuda:0"),
            strict=True,
        )
        self.normalise = ActionNormaliser()

    def reset(self):
        pass  # no episode state: every query re-plans from the current observation

    def __call__(self, obs):
        action, _ = self.policy.get_action(to_groot_obs(obs))
        raw = from_groot_action(action)[: self.exec_horizon]
        if not np.isfinite(raw).all():
            raise RuntimeError("non-finite action from GR00T")
        return self.normalise(raw)


def make_policy():
    return GrootPolicy()
