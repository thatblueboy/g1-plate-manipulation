"""Mapping between the evaluator's arrays and GR00T's named modality keys.

Shared by the dataset converter (evaluator venv) and the policy wrapper (GR00T
venv), so it depends on NumPy only.

The LeRobot parquet stores the evaluator's 50-dim state and the raw 20-dim
action unchanged; ``meta/modality.json`` slices named keys out of them.
Constant action dims (pelvis height 0.74, yaw rate 0, right gripper 0 in every
demonstration) are not predicted: min/max normalisation of a zero-range dim is
ill-defined, so the policy fills them with the constants below.

State (evaluator obs["state"], verified against the simulator's joint names):
    0-6   left arm joints        7-8   left Dex1 finger joints
    9-15  right arm joints       16-17 right Dex1 finger joints
    18-21 pelvis x, y, z, yaw    22-43 joint velocities (not used)
    44-45 gripper positions      46-49 floating base (duplicates 18-21, not used)
Action (raw, physical units; evaluator expects these normalised to [-1, 1]):
    0-1 base vx, vy   2 pelvis height   3 yaw rate
    4-10 left arm     11-17 right arm   18 left gripper   19 right gripper
"""

import json
from pathlib import Path

import numpy as np

CAMERAS = ("head", "right_wrist", "left_wrist")  # evaluator obs["rgb"] order
TASK = "pick up the plate from the left rack and place it in a slot of the right rack"
FPS = 50

STATE_KEYS = {
    "left_arm": (0, 7),
    "left_hand": (7, 9),
    "right_arm": (9, 16),
    "right_hand": (16, 18),
    "base_pose": (18, 22),
    "gripper": (44, 46),
}
ACTION_KEYS = {
    "base_velocity": (0, 2),
    "left_arm": (4, 11),
    "right_arm": (11, 18),
    "left_gripper": (18, 19),
}
ACTION_CONSTANTS = {2: 0.74, 3: 0.0, 19: 0.0}
RELATIVE_ACTION_KEYS = ("left_arm", "right_arm")  # predicted as target - current joint pos

LAYOUT_FILE = Path(__file__).resolve().parents[2] / "src/advaitic_eval/action_layout.json"


def modality_json() -> dict:
    return {
        "state": {k: {"start": a, "end": b} for k, (a, b) in STATE_KEYS.items()},
        "action": {k: {"start": a, "end": b} for k, (a, b) in ACTION_KEYS.items()},
        "video": {c: {"original_key": f"observation.images.{c}"} for c in CAMERAS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def to_groot_obs(obs: dict) -> dict:
    """Evaluator observation -> batched GR00T observation (B=1, T=1)."""
    rgb = np.asarray(obs["rgb"], np.uint8)  # (cams, 3, H, W)
    state = np.asarray(obs["state"], np.float32)
    return {
        "video": {c: rgb[i].transpose(1, 2, 0)[None, None] for i, c in enumerate(CAMERAS)},
        "state": {k: state[None, None, a:b] for k, (a, b) in STATE_KEYS.items()},
        "language": {"annotation.human.task_description": [[TASK]]},
    }


def from_groot_action(action: dict) -> np.ndarray:
    """GR00T action dict (B=1, T, D) in physical units -> raw (T, 20)."""
    horizon = next(iter(action.values())).shape[1]
    raw = np.zeros((horizon, 20), np.float32)
    for k, (a, b) in ACTION_KEYS.items():
        raw[:, a:b] = np.asarray(action[k], np.float32)[0]
    for i, v in ACTION_CONSTANTS.items():
        raw[:, i] = v
    return raw


class ActionNormaliser:
    """Raw physical actions -> the evaluator's [-1, 1] convention (public limits)."""

    def __init__(self, layout_file: Path = LAYOUT_FILE):
        layout = json.loads(Path(layout_file).read_text())
        self.low = np.asarray(layout["action_low"], np.float32)
        self.high = np.asarray(layout["action_high"], np.float32)

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        out = 2 * (raw - self.low) / (self.high - self.low) - 1
        return np.clip(out, -1, 1).astype(np.float32)
