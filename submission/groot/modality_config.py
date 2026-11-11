"""GR00T N1.7 modality config for BiGym2 move_plate (imported by launch_finetune).

Passed as --modality-config-path. Keys and slices come from
submission/groot/layout.py, the same module the policy wrapper uses.

GROOT_EMBODIMENT selects which embodiment projector is fine-tuned:
  new_embodiment (default)  the documented custom-robot path; the state/action
                            projectors start from the generic placeholder.
  unitree_g1                registers this config under the posttrain G1 tag,
                            whose projector (index 25) is shared with the
                            pretrained REAL_G1 embodiment, so it starts from
                            G1-pretrained weights. The pre-registered G1 config
                            uses a single ego camera and different keys, so it
                            is replaced in place (register_modality_config()
                            refuses an existing tag). Pass --embodiment-tag
                            UNITREE_G1 to launch_finetune with this setting.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from gr00t.configs.data.embodiment_configs import (  # noqa: E402
    MODALITY_CONFIGS,
    register_modality_config,
)
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.types import (  # noqa: E402
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

from submission.groot.layout import (  # noqa: E402
    ACTION_KEYS,
    CAMERAS,
    RELATIVE_ACTION_KEYS,
    STATE_KEYS,
)

ACTION_HORIZON = 40  # N1.7's maximum (model_config.action_horizon)


def _action_config(key):
    rep = (
        ActionRepresentation.RELATIVE
        if key in RELATIVE_ACTION_KEYS
        else ActionRepresentation.ABSOLUTE
    )
    return ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)


move_plate_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=list(CAMERAS)),
    "state": ModalityConfig(delta_indices=[0], modality_keys=list(STATE_KEYS)),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=list(ACTION_KEYS),
        action_configs=[_action_config(k) for k in ACTION_KEYS],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}

_choice = os.environ.get("GROOT_EMBODIMENT", "new_embodiment").lower()
if _choice == "unitree_g1":
    MODALITY_CONFIGS[EmbodimentTag.UNITREE_G1.value] = move_plate_config
elif _choice == "new_embodiment":
    register_modality_config(move_plate_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
else:
    raise ValueError(f"GROOT_EMBODIMENT must be new_embodiment or unitree_g1, got {_choice}")
