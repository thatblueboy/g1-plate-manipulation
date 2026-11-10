"""Spawn the real move_plate task in an interactive MuJoCo window (dev tool only).

The robot holds its pose under the frozen Balance/Walk controller so you can
inspect the scene. This reads simulator state directly and must never be used
by the evaluated policy.

    uv run python -m submission.tools.view_task            # random slot
    uv run python -m submission.tools.view_task --slot 3   # source slot 1..6

Keys in the window: N = new episode (next slot), standard MuJoCo mouse controls.
Do not use Backspace (MuJoCo's raw reset bypasses the task reset).
"""

import argparse
import os
import time

# Env cameras render via EGL (fast); the viewer window uses its own GLFW context.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco.viewer  # noqa: E402

from advaitic_eval.environment import make_environment, observed_source_slot  # noqa: E402
from advaitic_eval.protocol import source_slot_for_seed  # noqa: E402

# Far from the evaluator's seed blocks, so exploring never touches eval seeds.
DEV_SEED_START = 5_000_000


def seed_for_slot(slot: int | None, start: int) -> int:
    seed = start
    while slot is not None and source_slot_for_seed(seed) != slot:
        seed += 1
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=int, choices=range(1, 7), help="source slot 1..6")
    parser.add_argument("--seed", type=int, default=DEV_SEED_START)
    args = parser.parse_args()

    env = make_environment()
    inner = env.inner_env
    hold = env.normalize_action(env.raw_hold_action())
    dt = env.control_step_seconds
    state = {"seed": args.seed, "slot": None if args.slot is None else args.slot - 1}
    reset_requested = [False]

    def do_reset():
        seed = seed_for_slot(state["slot"], state["seed"])
        env.reset(seed=seed)
        print(f"seed={seed}  source slot={observed_source_slot(env) + 1}", flush=True)
        state["seed"] = seed + 1
        if state["slot"] is not None:
            state["slot"] = (state["slot"] + 1) % 6

    def on_key(keycode):
        if keycode == ord("N"):
            reset_requested[0] = True

    do_reset()
    try:
        with mujoco.viewer.launch_passive(inner.model, inner.data, key_callback=on_key) as viewer:
            while viewer.is_running():
                start = time.perf_counter()
                if reset_requested[0]:
                    reset_requested[0] = False
                    do_reset()
                env.step(hold)
                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - start)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
