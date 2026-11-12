"""Render evaluated episodes to MP4 from a third-person camera (dev tool only).

Replays the action log the evaluator saved for each episode
(results/<run>/episode_NNN.actions.npz) from the episode's seed, which
reproduces the evaluated rollout exactly, and renders a 640x480 free camera
with the policy's three 84x84 cameras in a strip below. The camera placement reads no
simulator state; the replayed outcome is checked against episodes.csv.

    uv run --extra video python -m submission.tools.render_episodes \
        --run results/20261110-2249-val-60-r1-100k --episodes 1 2 3 12 \
        --out results/videos/act-recovery-r1.mp4
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from advaitic_eval.environment import make_environment  # noqa: E402

W, H, FPS, STRIDE = 640, 480, 25, 2  # 50 Hz control, every 2nd step = real time


def camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.5, -0.2, 0.8]
    cam.azimuth, cam.elevation, cam.distance = 155.0, -30.0, 1.7
    return cam


def compose(scene, rgb_obs, caption):
    """Scene on top, the policy's three cameras (head, right wrist, left wrist) below."""
    side = W // 3 // 2 * 2  # even, for H.264
    strip = [
        Image.fromarray(np.asarray(c).transpose(1, 2, 0)).resize((side, side), Image.NEAREST)
        for c in rgb_obs
    ]
    frame = Image.new("RGB", (W, H + side), (0, 0, 0))
    frame.paste(Image.fromarray(scene), (0, 0))
    for i, img in enumerate(strip):
        frame.paste(img, ((W - 3 * side) // 2 + i * side, H))
    draw = ImageDraw.Draw(frame)
    draw.rectangle([0, 0, W, 26], fill=(0, 0, 0))
    draw.text((8, 7), caption, fill=(255, 255, 255))
    return np.asarray(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="evaluator output directory")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="combined MP4")
    parser.add_argument("--label", default="ACT + recoveries (100k)")
    args = parser.parse_args()

    rows = {int(r["episode"]): r for r in csv.DictReader(open(args.run / "episodes.csv"))}
    env = make_environment()
    inner = env.inner_env
    renderer = mujoco.Renderer(inner.model, H, W)
    cam = camera()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined = imageio.get_writer(str(args.out), fps=FPS, macro_block_size=1)
    try:
        for ep in args.episodes:
            row = rows[ep]
            actions = np.load(args.run / f"episode_{ep:03d}.actions.npz")["actions"]
            ts = env.reset(seed=int(row["seed"]))
            slot = int(row["source_slot"]) + 1
            single = imageio.get_writer(
                str(args.out.with_name(f"{args.out.stem}-ep{ep:03d}.mp4")),
                fps=FPS,
                macro_block_size=1,
            )
            frame = None
            for t, action in enumerate(actions, 1):
                ts = env.step(action)
                if t % STRIDE == 0 or t == len(actions):
                    renderer.update_scene(inner.data, camera=cam)
                    caption = (
                        f"{args.label} | seed {row['seed']} | source slot {slot} | "
                        f"t = {t / 50:5.1f} s"
                    )
                    frame = compose(renderer.render(), ts.rgb_obs, caption)
                    single.append_data(frame)
                    combined.append_data(frame)
            ok = bool(env.episode_succeeded() and not env.episode_fell())
            verdict = "SUCCESS" if ok else f"FAILURE ({env.episode_termination()})"
            end = Image.fromarray(frame)
            draw = ImageDraw.Draw(end)
            draw.rectangle([0, 26, W, 52], fill=(0, 110, 0) if ok else (150, 0, 0))
            draw.text((8, 33), f"{verdict} after {len(actions) / 50:.1f} s", fill=(255, 255, 255))
            for _ in range(FPS * 2):  # hold the result for 2 s
                single.append_data(np.asarray(end))
                combined.append_data(np.asarray(end))
            single.close()
            match = ok == (row["success"] == "True")
            print(
                f"episode {ep} seed {row['seed']} slot {slot}: {verdict}, "
                f"{len(actions)} steps, matches evaluation: {match}",
                flush=True,
            )
    finally:
        combined.close()
        env.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
