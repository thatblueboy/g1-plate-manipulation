"""Shifted-placement demonstrations: replay demos with extra sidestep steps.

    uv run python -m submission.tools.augment_sidestep --extras 30 38 \
        --out datasets/move_plate_shift

Every demonstration places the plate in rack-2 slot 1. Open-loop replay of a
demo from its own seed reproduces it (60 frames held at the end), so this tool
replays each demo with N copies of a mid-sidestep action inserted (base vy
-0.25 m/s, ~0.5 cm per step; slots are 4.2 cm apart), which carries the plate
further along the rack before the unchanged placement motion. Observations are
re-rendered by the simulator, so (obs, action) pairs stay consistent.

Variants are kept only if BiGym's own success check holds after a 60-step
hold (privileged state, used here only to filter generated training data,
never by the evaluated policy). Uses demonstration seeds only (18-147).
Output matches submission/act/export_dataset.py: obs t paired with the raw
action executed after it; the hold frames are not stored.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from advaitic_eval.environment import make_environment, observation  # noqa: E402
from submission.tools.diagnose_failures import plate_report  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HOLD = 60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, default=REPO / "datasets/move_plate")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--extras",
        type=int,
        nargs="+",
        default=[30, 38],
        help="extra sidestep steps; demo i tries extras[i %% n] first, then the others",
    )
    parser.add_argument("--max-demos", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; use a new directory")

    layout = json.loads((REPO / "src/advaitic_eval/action_layout.json").read_text())
    low, high = np.asarray(layout["action_low"]), np.asarray(layout["action_high"])

    def norm(a):
        return np.clip(2 * (a - low) / (high - low) - 1, -1, 1).astype(np.float32)

    meta = json.loads((args.demos / "meta.json").read_text())
    actions = np.load(args.demos / "action.npy", mmap_mode="r")
    ends = np.load(args.demos / "episode_ends.npy")
    starts = np.concatenate([[0], ends[:-1]])
    n_demos = len(ends) if not args.max_demos else min(args.max_demos, len(ends))

    env = make_environment()
    inner = env.inner_env
    rgb, state, action, ep_ends, records = [], [], [], [], []
    total = 0
    for i in range(n_demos):
        seed = int(meta["demo_seeds"][i])
        a = np.asarray(actions[starts[i] : ends[i]], np.float64)
        side = np.where(a[:, 1] < -0.2)[0]
        if len(side) == 0:
            print(f"demo {i}: no sidestep found, skipped", flush=True)
            continue
        mid = int(side[len(side) // 2])
        order = args.extras[i % len(args.extras) :] + args.extras[: i % len(args.extras)]
        for extra in order:
            seq = np.concatenate([a[:mid], np.repeat(a[mid : mid + 1], extra, 0), a[mid:]])
            ts = env.reset(seed=seed)
            obs_rgb, obs_state = [], []
            for x in seq:
                o = observation(ts)
                obs_rgb.append(o["rgb"])
                obs_state.append(o["state"])
                ts = env.step(norm(x))
            for _ in range(HOLD):
                env.step(norm(seq[-1]))
            report = plate_report(inner)
            ok = bool(inner._success())
            print(
                f"demo {i:2d} seed {seed:3d} +{extra:2d}: success {ok} "
                f"slot {report['slot']} dist {report['dist_cm']:.1f} cm",
                flush=True,
            )
            if ok:
                rgb.append(np.stack(obs_rgb))
                state.append(np.stack(obs_state))
                action.append(seq.astype(np.float32))
                total += len(seq)
                ep_ends.append(total)
                records.append(
                    {
                        "demo": i,
                        "seed": seed,
                        "extra_steps": extra,
                        "insert_at": mid,
                        "final_slot": report["slot"],
                        "final_dist_cm": round(float(report["dist_cm"]), 2),
                    }
                )
                break

    if not ep_ends:
        raise SystemExit("no successful variants")
    args.out.mkdir(parents=True)
    np.save(args.out / "rgb.npy", np.concatenate(rgb))
    np.save(args.out / "state.npy", np.concatenate(state))
    np.save(args.out / "action.npy", np.concatenate(action))
    np.save(args.out / "episode_ends.npy", np.asarray(ep_ends, np.int64))
    slots = [r["final_slot"] for r in records]
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "frames": total,
                "episodes": len(ep_ends),
                "demo_seeds": [r["seed"] for r in records],
                "slot_counts": {s: slots.count(s) for s in sorted(set(slots))},
                "variants": records,
            },
            indent=2,
        )
    )
    print(f"kept {len(ep_ends)}/{n_demos} demos, {total} frames -> {args.out}")
    print("final slots:", {s: slots.count(s) for s in sorted(set(slots))})


if __name__ == "__main__":
    main()
