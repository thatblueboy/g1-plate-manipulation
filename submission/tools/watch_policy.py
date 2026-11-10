"""Watch a policy solve move_plate live in a MuJoCo window (dev tool only).

Uses the evaluator's own policy-process bridge, so the policy sees exactly
what it sees during evaluation (obs["rgb"], obs["state"]) and may run in a
separate Python environment. Seeds default to a dev block (5,000,000+),
away from the public evaluation seeds. Results here are NOT scored.

    uv run python -m submission.tools.watch_policy                     # ACT baseline
    uv run python -m submission.tools.watch_policy --slot 3 --speed 0.5
    uv run python -m submission.tools.watch_policy --policy submission.policy:make_policy \
        --policy-python "$(uv run which python)"

Keys: N = skip to next episode. Mouse: standard MuJoCo camera controls.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Cameras render offscreen via EGL (~10 ms/step vs ~56 ms with GLFW); the native
# viewer window opens its own GLFW context regardless of MUJOCO_GL.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco.viewer  # noqa: E402

from advaitic_eval.environment import (  # noqa: E402
    make_environment,
    observation,
    observed_source_slot,
    validate_actions,
)
from advaitic_eval.policy_process import PolicyProcess  # noqa: E402
from submission.tools.view_task import DEV_SEED_START, seed_for_slot  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="submission.act.policy:make_policy")
    parser.add_argument("--policy-python", default=str(REPO / ".venv-act/bin/python"))
    parser.add_argument("--slot", type=int, choices=range(1, 7), help="first source slot 1..6")
    parser.add_argument("--seed", type=int, default=DEV_SEED_START)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed vs real time")
    parser.add_argument("--episodes", type=int, default=0, help="stop after N (0 = until closed)")
    args = parser.parse_args()

    log = REPO / "results" / "watch_policy.stderr.log"
    log.parent.mkdir(exist_ok=True)
    env = make_environment()
    inner = env.inner_env
    dt = env.control_step_seconds / args.speed
    process = PolicyProcess(args.policy, REPO, log, python=args.policy_python)
    skip = [False]
    state = {"seed": args.seed, "slot": None if args.slot is None else args.slot - 1}
    tally = [0, 0]

    def new_episode():
        seed = seed_for_slot(state["slot"], state["seed"])
        ts = env.reset(seed=seed)
        process.reset()
        state["seed"] = seed + 1
        if state["slot"] is not None:
            state["slot"] = (state["slot"] + 1) % 6
        print(f"\nseed={seed} source slot={observed_source_slot(env) + 1} ...", end="", flush=True)
        return ts

    def on_key(keycode):
        if keycode == ord("N"):
            skip[0] = True

    try:
        ts = new_episode()
        queue, steps = [], 0
        with mujoco.viewer.launch_passive(inner.model, inner.data, key_callback=on_key) as viewer:
            while viewer.is_running():
                start = time.perf_counter()
                if not queue:
                    queue = list(validate_actions(process.act(observation(ts))))
                ts = env.step(queue.pop(0))
                steps += 1
                viewer.sync()
                if ts.last() or skip[0]:
                    ok = env.episode_succeeded() and not env.episode_fell() and not skip[0]
                    term = "skipped" if skip[0] else env.episode_termination()
                    tally[0] += ok
                    tally[1] += 1
                    print(
                        f" {'SUCCESS' if ok else 'FAIL'} ({term}, {steps} steps = "
                        f"{steps * env.control_step_seconds:.1f} s)  [{tally[0]}/{tally[1]}]",
                        flush=True,
                    )
                    if args.episodes and tally[1] >= args.episodes:
                        break
                    hold_until = time.perf_counter() + 1.5  # pause on the final state
                    while viewer.is_running() and time.perf_counter() < hold_until:
                        viewer.sync()
                        time.sleep(0.02)
                    skip[0] = False
                    ts, queue, steps = new_episode(), [], 0
                    continue
                time.sleep(max(0.0, dt - (time.perf_counter() - start)))
    except RuntimeError as exc:
        print(f"\npolicy error: {exc} (see {log})", file=sys.stderr)
    finally:
        process.close()
        env.close()


if __name__ == "__main__":
    main()
