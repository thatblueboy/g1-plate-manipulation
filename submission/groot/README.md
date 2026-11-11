# GR00T N1.7 fine-tuning for move_plate

Fine-tunes NVIDIA Isaac GR00T N1.7 (3B, `nvidia/GR00T-N1.7-3B`) on the
60 BiGym2 demonstrations plus the recovery episodes, and evaluates it with the
unmodified evaluator through a separate policy Python (`.venv` of Isaac-GR00T).

- Code: <https://github.com/NVIDIA/Isaac-GR00T>, pinned to commit
  `3b49067943eb1ab2eb2203b93d461c4390c78da3` (Apache 2.0).
- Weights: `nvidia/GR00T-N1.7-3B` (NVIDIA Open Model License).
- Hardware: fine-tuning needs one GPU with at least 40 GB (default mode tunes the
  projectors and the diffusion action head; the VLM backbone stays frozen,
  peak below about 35 GB). Inference needs at least 16 GB, so the evaluation
  also runs on the rented machine.

## Files

| File | Role |
|---|---|
| `layout.py` | Evaluator arrays to/from GR00T's named keys (NumPy only; shared). |
| `convert_dataset.py` | Our `.npy` datasets to GR00T-flavoured LeRobot v2. |
| `modality_config.py` | Modality config for `launch_finetune --modality-config-path`. |
| `policy.py` | `make_policy()` for the evaluator; runs in the GR00T venv. |
| `cloud_setup.sh`, `train.sh`, `eval.sh` | Setup, fine-tuning and evaluation on the GPU machine. |

## Data mapping

The parquet stores the evaluator's 50-dim state and the raw 20-dim action
unchanged; `meta/modality.json` names slices of them (see `layout.py`):

- Video: `head`, `right_wrist`, `left_wrist` (84x84, H.264 yuv444p CRF 0,
  mean decode error 0.4/255).
- State: `left_arm`, `left_hand` (Dex1 finger joints), `right_arm`,
  `right_hand`, `base_pose` (pelvis x, y, z, yaw), `gripper`. Joint velocities
  and the duplicated floating-base block are not used.
- Action (40-step chunks, the model maximum): `base_velocity` (vx, vy),
  `left_arm`, `right_arm` (both relative to the current joint positions),
  `left_gripper`. Pelvis height (0.74), yaw rate (0) and the right gripper (0)
  are constant in every demonstration; they are not predicted and the policy
  fills them in, because min/max normalisation of a zero-range dim is
  ill-defined.
- Language: one fixed task string.

`GROOT_EMBODIMENT=new_embodiment` (default) uses the documented custom-robot
path. `GROOT_EMBODIMENT=unitree_g1` trains the same config under the
`UNITREE_G1` posttrain tag, whose projector is shared with the pretrained
`REAL_G1` embodiment; this is an ablation, not the default, because the
pretrained G1 data used different state/action keys.

## Workflow

Locally (evaluator venv, no GPU):

```bash
uv run python -m submission.groot.convert_dataset \
    --data datasets/move_plate datasets/recovery --out datasets/groot/move_plate_r1 --check
```

Copy the repo (branch `groot`), `datasets/groot/move_plate_r1` and, for the
paired comparison, `results/20261110-0920-val-60-last-ensemble/` to the GPU
machine. Then:

```bash
bash submission/groot/cloud_setup.sh
SMOKE=1 DATA=datasets/groot/move_plate_r1 OUT=checkpoints/groot-r1 bash submission/groot/train.sh
SMOKE=1 CKPT=checkpoints/groot-r1/checkpoint-5000 bash submission/groot/eval.sh
CKPT=checkpoints/groot-r1/checkpoint-20000 bash submission/groot/eval.sh
```

Validation uses the same 60 seeds (6,000,000+) as every ACT run. Run the public
block only with the final configuration.
