"""Pack recorded recovery episodes into the training-dataset format.

    uv run python -m submission.act.pack_recoveries \
        --raw datasets/recovery/raw --out datasets/recovery [--successful-only]

Each raw episode (from submission.tools.collect_recovery) already pairs
observation t with the raw action executed at t, so no shift is applied.
Train on demos + recoveries with:

    .venv-act/bin/python -m submission.act.train --data datasets/move_plate datasets/recovery ...
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("datasets/recovery/raw"))
    parser.add_argument("--out", type=Path, default=Path("datasets/recovery"))
    parser.add_argument(
        "--successful-only", action="store_true", help="keep only recoveries that met success"
    )
    args = parser.parse_args()

    files = sorted(args.raw.glob("recovery_*.npz"))
    rgb, state, action, ends, metas = [], [], [], [], []
    n = 0
    for f in files:
        ep = np.load(f)
        meta = json.loads(str(ep["meta"]))
        if args.successful_only and not meta["success_at_save"]:
            continue
        if not (6_000_000 > meta["seed"] >= 5_500_000):
            raise ValueError(f"{f.name}: seed {meta['seed']} outside the collection block")
        if not np.isfinite(ep["action"]).all():
            raise ValueError(f"{f.name}: non-finite action")
        rgb.append(ep["rgb"])
        state.append(ep["state"])
        action.append(ep["action"])
        n += len(ep["action"])
        ends.append(n)
        metas.append({"file": f.name, **meta})
    if not ends:
        raise SystemExit(f"no recovery episodes in {args.raw}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "rgb.npy", np.concatenate(rgb))
    np.save(args.out / "state.npy", np.concatenate(state))
    np.save(args.out / "action.npy", np.concatenate(action))
    np.save(args.out / "episode_ends.npy", np.asarray(ends, np.int64))
    (args.out / "meta.json").write_text(
        json.dumps({"frames": n, "episodes": len(ends), "recoveries": metas}, indent=2)
    )
    ok = sum(m["success_at_save"] for m in metas)
    print(f"packed {len(ends)} recoveries ({ok} successful), {n} frames -> {args.out}")


if __name__ == "__main__":
    main()
