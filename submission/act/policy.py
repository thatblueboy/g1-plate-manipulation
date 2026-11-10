"""ACT baseline policy for the Advaitic evaluator. Runs in .venv-act:

    MUJOCO_GL=egl uv run advaitic-eval --policy submission.act.policy:make_policy \
        --policy-python $PWD/.venv-act/bin/python --method act-baseline \
        --checkpoint checkpoints/act-baseline/last.pt --episodes 30 --out results/<new-dir>

Environment variables (read in the policy process):
  ACT_CHECKPOINT   checkpoint path (default checkpoints/act-baseline/last.pt)
  ACT_MODE         "ensemble" (temporal ensembling, query every step; default)
                   or "chunk" (execute ACT_EXEC_HORIZON actions per query)
  ACT_EXEC_HORIZON actions per query in chunk mode (default 10)
  ACT_ENSEMBLE_K   ensembling weight decay m in w_i = exp(-m*i) (default 0.01)

Uses only obs["rgb"] and obs["state"]. Fixed public action limits come from
action_layout.json.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from submission.act.model import ACT, ACTConfig

REPO = Path(__file__).resolve().parents[2]


class ACTPolicy:
    def __init__(self, checkpoint=None, mode=None, exec_horizon=None, ensemble_k=None):
        checkpoint = checkpoint or os.environ.get(
            "ACT_CHECKPOINT", str(REPO / "checkpoints/act-baseline/last.pt")
        )
        self.mode = mode or os.environ.get("ACT_MODE", "ensemble")
        self.exec_horizon = int(exec_horizon or os.environ.get("ACT_EXEC_HORIZON", 10))
        self.ensemble_k = float(ensemble_k or os.environ.get("ACT_ENSEMBLE_K", 0.01))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg = ACTConfig(**{**ckpt["config"], "pretrained_backbone": False})
        self.model = ACT(cfg).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.chunk = cfg.chunk
        st = {k: torch.tensor(v, device=self.device) for k, v in ckpt["stats"].items()}
        self.s_mean, self.s_std = st["state_mean"], st["state_std"]
        self.a_mean, self.a_std = st["action_mean"], st["action_std"]

        layout = json.loads((REPO / "src/advaitic_eval/action_layout.json").read_text())
        self.low = np.asarray(layout["action_low"], np.float32)
        self.high = np.asarray(layout["action_high"], np.float32)
        self.reset()

    def reset(self):
        self.t = 0
        self.history = []  # (start_step, raw chunk (K, 20)) for temporal ensembling

    @torch.no_grad()
    def predict(self, obs) -> np.ndarray:
        rgb = torch.from_numpy(np.asarray(obs["rgb"], np.uint8)).to(self.device)
        rgb = rgb.float().div_(255)[None]  # (1, 3 cams, 3, 84, 84)
        state = torch.from_numpy(np.asarray(obs["state"], np.float32)).to(self.device)[None]
        pred, _, _ = self.model(rgb, (state - self.s_mean) / self.s_std)
        return (pred[0] * self.a_std + self.a_mean).float().cpu().numpy()  # raw (K, 20)

    def normalize(self, raw):
        return np.clip(2 * (raw - self.low) / (self.high - self.low) - 1, -1, 1).astype(np.float32)

    def __call__(self, obs):
        raw_chunk = self.predict(obs)
        if self.mode == "chunk":
            self.t += self.exec_horizon
            return self.normalize(raw_chunk[: self.exec_horizon])

        self.history.append((self.t, raw_chunk))
        self.history = [(s, c) for s, c in self.history if self.t - s < self.chunk]
        preds = np.stack([c[self.t - s] for s, c in self.history])  # oldest first
        w = np.exp(-self.ensemble_k * np.arange(len(preds)))
        raw = (w[:, None] * preds).sum(0) / w.sum()
        self.t += 1
        return self.normalize(raw)


def make_policy():
    return ACTPolicy()
