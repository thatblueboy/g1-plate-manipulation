"""Export the 60 BiGym2 move_plate demos to flat .npy arrays for ACT training.

Run in the evaluator environment (it has BiGym2):

    uv run python -m submission.act.export_dataset --out datasets/move_plate

BiGym row t holds observation t together with the action that PRODUCED it
(row 0 is the reset row). The action to take after seeing observation t is
therefore row t+1. We store aligned pairs: obs[t] -> act[t] = raw[t+1].
Actions are stored RAW (physical units); the policy normalises them with
action_layout.json before returning, exactly as the evaluator expects.
Only policy-visible observations (rgb, state) are exported.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/move_plate")
    parser.add_argument("--num-demos", type=int, default=60)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from bigym.loco import make

    env = make("move_plate")
    try:
        episodes = env.load_demo_episodes(args.num_demos)
    finally:
        env.close()

    lengths = [len(e["raw_outer_action"]) - 1 for e in episodes]
    n = int(sum(lengths))
    rgb = np.lib.format.open_memmap(out / "rgb.npy", "w+", np.uint8, (n, 3, 3, 84, 84))
    state = np.empty((n, 50), np.float32)
    action = np.empty((n, 20), np.float32)
    ends, seeds = [], []
    i = 0
    for e, length in zip(episodes, lengths, strict=True):
        raw = np.asarray(e["raw_outer_action"], np.float32)
        if not np.isfinite(raw[1:]).all():
            raise ValueError("non-finite action in demo")
        rgb[i : i + length] = e["rgb_obs"][:-1]
        state[i : i + length] = e["low_dim_obs"][:-1]
        action[i : i + length] = raw[1:]
        i += length
        ends.append(i)
        seeds.append(int(np.asarray(e["seed"]).ravel()[0]))
    rgb.flush()
    np.save(out / "state.npy", state)
    np.save(out / "action.npy", action)
    np.save(out / "episode_ends.npy", np.asarray(ends, np.int64))
    (out / "meta.json").write_text(
        json.dumps({"frames": n, "episodes": len(ends), "demo_seeds": seeds}, indent=2)
    )
    print(f"exported {len(ends)} episodes, {n} frames -> {out}")


if __name__ == "__main__":
    main()
