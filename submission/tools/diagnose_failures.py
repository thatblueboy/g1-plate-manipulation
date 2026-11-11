"""Explain why evaluated episodes failed by replaying their logged actions (analysis only).

Replays each episode's recorded action sequence from its seed in the real
environment and reports BiGym's move_plate success conditions on the final
state. Reads privileged simulator state, so it must never be used by the
evaluated policy; it only analyses finished evaluation results.

    uv run python -m submission.tools.diagnose_failures results/<run-dir> [--all]
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

from advaitic_eval.environment import make_environment  # noqa: E402


def plate_report(inner):
    plate = inner.plates[0]
    pos = plate.get_position()
    sites = [inner.data.bind(s).xpos for s in inner.rack_target.sites]
    dists = [np.linalg.norm(pos - s) for s in sites]
    nearest = int(np.argmin(dists))
    plate_up = Quaternion(plate.get_quaternion()).rotate(np.array([0, 0, 1]))
    tilt = np.degrees(np.arccos(np.clip(np.dot(plate_up, [0, -1, 0]), -1, 1)))
    held = any(inner.robot.is_gripper_holding_object(plate, g) for g in inner.robot.grippers)
    rack_src_dist = min(
        np.linalg.norm(pos - inner.data.bind(s).xpos) for s in inner.rack_start.sites
    )
    return {
        "dist_cm": 100 * dists[nearest],
        "slot": nearest + 1,
        "offset_cm": 100 * (pos - sites[nearest]),
        "tilt_deg": tilt,
        "touch_rack2": plate.is_colliding(inner.rack_target),
        "touch_table": plate.is_colliding(inner.table),
        "held": held,
        "in_rack1": rack_src_dist < 0.05,
    }


def verdict(r):
    reasons = []
    if r["in_rack1"]:
        reasons.append("still in rack 1")
    if r["dist_cm"] > 5:
        reasons.append(f"too far from slot ({r['dist_cm']:.1f} cm)")
    if r["tilt_deg"] > 20:
        reasons.append(f"tilted {r['tilt_deg']:.0f} deg")
    if not r["touch_rack2"]:
        reasons.append("not touching rack 2")
    if r["touch_table"]:
        reasons.append("touching table")
    if r["held"]:
        reasons.append("still held")
    return ", ".join(reasons) or "all conditions met at end"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--all", action="store_true", help="include successful episodes")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.run / "episodes.csv")))
    env = make_environment()
    try:
        for row in rows:
            ok = row["success"] == "True"
            if ok and not args.all:
                continue
            actions = np.load(args.run / f"episode_{int(row['episode']):03d}.actions.npz")
            actions = actions[actions.files[0]]
            ts = env.reset(seed=int(row["seed"]))
            for a in actions:
                ts = env.step(a)
                if ts.last():
                    break
            replay_ok = bool(env.episode_succeeded() and not env.episode_fell())
            r = plate_report(env.inner_env)
            o = r["offset_cm"]
            note = "" if replay_ok == ok else "  [REPLAY DIVERGED]"
            print(
                f"seed {row['seed']} slot {int(row['source_slot']) + 1} "
                f"{'OK  ' if ok else 'FAIL'} -> nearest dest slot {r['slot']}, "
                f"offset x{o[0]:+5.1f} y{o[1]:+5.1f} z{o[2]:+5.1f} cm, tilt {r['tilt_deg']:4.0f} deg | "
                f"{verdict(r)}{note}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
