"""Collect human recovery data on policy failures (HG-DAgger style; data collection only).

The ACT policy runs on collection seeds in a MuJoCo window. When an episode
times out without success, control passes to you through a separate
"Recovery controls" window: move the left hand with the mouse (drag pad) or
keys, toggle its gripper, sidestep the base, then save. From the takeover
step onward every (observation, raw action) pair is recorded, so training can
learn what to do in the states the policy actually gets stuck in.

Seeds come from a dedicated collection block (5,500,000+), disjoint from the
demos, the dev viewer (5,000,000+), validation (6,000,000+) and public
evaluation (7,310,000+) blocks. The hand control solves inverse kinematics
with the simulator model and the panel shows privileged task status; both are
allowed only in this disclosed data-collection workflow, never in evaluated
inference.

    uv run python -m submission.tools.collect_recovery --failed-in results/*-screen-*
    uv run python -m submission.tools.collect_recovery --seeds 5500008 5500011
    uv run python -m submission.tools.collect_recovery --seeds 5500008 --takeover 0  # practice

By default failures where the plate never left rack 1, or lies flat on the
table, are skipped automatically (--all-failures keeps them).

Controls: use the "Recovery controls" window (click it so it has keyboard
focus). The MuJoCo window is only for viewing (mouse = camera).
  drag pad      left-drag: hand forward/back/left/right   (1 px = 1 mm)
                right-drag up/down or scroll: hand up/down
  arrow keys    hand forward/back/left/right (hold to keep moving)
  W / S         hand up / down
  Space         toggle left gripper (open <-> closed)
  A / D         base sidestep left / right (hold to keep stepping)
  T             take over now (policy -> you)
  Enter         save this recovery (records 0.5 s more of holding still first)
  Delete        discard this recovery and go to the next seed
  N             skip to the next seed
"""

import argparse
import csv
import json
import os
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from advaitic_eval.environment import (  # noqa: E402
    make_environment,
    observation,
    observed_source_slot,
    validate_actions,
)
from advaitic_eval.policy_process import PolicyProcess  # noqa: E402
from submission.tools.arm_ik import PREFIX, LeftArmIK  # noqa: E402
from submission.tools.diagnose_failures import plate_report, verdict  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
COLLECT_SEED_START = 5_500_000
LAYOUT = json.loads((REPO / "src/advaitic_eval/action_layout.json").read_text())
LOW = np.asarray(LAYOUT["action_low"], np.float64)
HIGH = np.asarray(LAYOUT["action_high"], np.float64)
LEFT_ARM, LEFT_GRIP = slice(4, 11), 18
KEY_STEP_M = 0.004  # hand step per key event (key auto-repeat ~30/s -> ~12 cm/s when held)
PAD_M_PER_PX = 0.001
BASE_PULSE_STEPS, HOLD_STEPS = 10, 25
SIDESTEP_VY = 0.25  # demo range: vy -0.25..0.06
MAX_DQ = 0.02  # rad per control step (1 rad/s): keeps commanded arm motion smooth
LEASH_M = 0.04  # max distance the hand target may lead the actual hand


class Done(Exception):
    """All requested seeds have been played (or the windows were closed)."""


def to_raw(a):
    return LOW + (np.asarray(a, np.float64) + 1) * (HIGH - LOW) / 2


def to_norm(raw):
    return np.clip(2 * (raw - LOW) / (HIGH - LOW) - 1, -1, 1).astype(np.float32)


class Recorder:
    def __init__(self):
        self.rgb, self.state, self.action = [], [], []

    def add(self, obs, raw):
        self.rgb.append(obs["rgb"])
        self.state.append(obs["state"])
        self.action.append(raw.astype(np.float32))

    def save(self, out, meta):
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = out / f"recovery_{meta['seed']}_{stamp}.npz"
        np.savez_compressed(
            path,
            rgb=np.stack(self.rgb),
            state=np.stack(self.state),
            action=np.stack(self.action),
            meta=json.dumps(meta),
        )
        with open(out / "index.jsonl", "a") as f:
            f.write(json.dumps({"file": path.name, **meta}) + "\n")
        return path


class ControlPanel:
    """Tk window that turns mouse/keyboard input into queued commands."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Recovery controls")
        self.root.attributes("-topmost", True)
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.hand = np.zeros(3)  # accumulated hand displacement (robot frame, m)
        self.events = []  # "takeover", "grip", "save", "discard", "skip"
        self.base_vy = 0.0
        self.base_pulse = 0

        self.status = tk.StringVar(value="starting ...")
        tk.Label(
            self.root,
            textvariable=self.status,
            font=("TkFixedFont", 11),
            justify="left",
            anchor="w",
            width=60,
            height=4,
        ).pack(padx=8, pady=4)
        self.pad = tk.Canvas(self.root, width=320, height=320, bg="#20242a", highlightthickness=0)
        self.pad.pack(padx=8)
        for x in (160,):
            self.pad.create_line(x, 0, x, 320, fill="#555")
            self.pad.create_line(0, x, 320, x, fill="#555")
        self.pad.create_text(160, 12, text="forward", fill="#aaa")
        self.pad.create_text(160, 308, text="back", fill="#aaa")
        self.pad.create_text(26, 160, text="left", fill="#aaa")
        self.pad.create_text(296, 160, text="right", fill="#aaa")
        self.pad.create_text(
            160, 290, text="left-drag: move   right-drag / scroll: up-down", fill="#777"
        )
        self._last = None
        self.pad.bind("<ButtonPress-1>", self._press)
        self.pad.bind("<B1-Motion>", lambda e: self._drag(e, vertical=False))
        self.pad.bind("<ButtonPress-3>", self._press)
        self.pad.bind("<B3-Motion>", lambda e: self._drag(e, vertical=True))
        self.pad.bind("<Button-4>", lambda e: self._nudge(0, 0, 0.005))  # X11 scroll up
        self.pad.bind("<Button-5>", lambda e: self._nudge(0, 0, -0.005))
        self.pad.bind("<MouseWheel>", lambda e: self._nudge(0, 0, 0.005 * np.sign(e.delta)))

        row = tk.Frame(self.root)
        row.pack(pady=6)
        for text, ev in (
            ("Take over [T]", "takeover"),
            ("Gripper [Space]", "grip"),
            ("Save [Enter]", "save"),
            ("Discard [Del]", "discard"),
            ("Skip [N]", "skip"),
        ):
            tk.Button(row, text=text, command=lambda ev=ev: self.events.append(ev)).pack(
                side="left", padx=2
            )
        row2 = tk.Frame(self.root)
        row2.pack(pady=(0, 6))
        tk.Button(row2, text="◀ step left [A]", command=lambda: self._step(SIDESTEP_VY)).pack(
            side="left", padx=2
        )
        tk.Button(row2, text="step right [D] ▶", command=lambda: self._step(-SIDESTEP_VY)).pack(
            side="left", padx=2
        )
        tk.Label(
            self.root, text="Keys (this window): arrows = hand, W/S = up/down", fg="#666"
        ).pack(pady=(0, 6))

        k = KEY_STEP_M
        for key, d in (
            ("<Up>", (k, 0, 0)),
            ("<Down>", (-k, 0, 0)),
            ("<Left>", (0, k, 0)),
            ("<Right>", (0, -k, 0)),
            ("<w>", (0, 0, k)),
            ("<s>", (0, 0, -k)),
        ):
            self.root.bind(key, lambda e, d=d: self._nudge(*d))
        for key, ev in (
            ("<t>", "takeover"),
            ("<space>", "grip"),
            ("<Return>", "save"),
            ("<Delete>", "discard"),
            ("<n>", "skip"),
        ):
            self.root.bind(key, lambda e, ev=ev: self.events.append(ev))
        self.root.bind("<a>", lambda e: self._step(SIDESTEP_VY))
        self.root.bind("<d>", lambda e: self._step(-SIDESTEP_VY))
        self.root.focus_force()

    def _close(self):
        self.closed = True

    def _press(self, e):
        self._last = (e.x, e.y)

    def _drag(self, e, vertical):
        if self._last is None:
            return
        dx, dy = e.x - self._last[0], e.y - self._last[1]
        self._last = (e.x, e.y)
        if vertical:
            self._nudge(0, 0, -dy * PAD_M_PER_PX)
        else:  # screen up = forward (+x), screen right = robot's right (-y)
            self._nudge(-dy * PAD_M_PER_PX, -dx * PAD_M_PER_PX, 0)

    def _nudge(self, dx, dy, dz):
        self.hand += (dx, dy, dz)

    def _step(self, vy):
        self.base_vy, self.base_pulse = vy, BASE_PULSE_STEPS

    def poll(self):
        """Process window events; return (hand displacement, events) since last poll."""
        self.root.update()
        hand, self.hand = self.hand, np.zeros(3)
        events, self.events = self.events, []
        return hand, events

    def base_command(self):
        if self.base_pulse > 0:
            self.base_pulse -= 1
            return self.base_vy
        return 0.0

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--policy", default="submission.act.policy:make_policy")
    parser.add_argument("--policy-python", default=str(REPO / ".venv-act/bin/python"))
    parser.add_argument("--seed", type=int, default=COLLECT_SEED_START)
    parser.add_argument("--out", type=Path, default=REPO / "datasets/recovery/raw")
    parser.add_argument("--speed", type=float, default=1.0, help="policy playback speed")
    parser.add_argument("--human-speed", type=float, default=0.5, help="sim speed while you drive")
    parser.add_argument(
        "--takeover", type=int, default=-1, help="auto take over at this step (-1 = on failure)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="play exactly these seeds, then stop")
    parser.add_argument(
        "--failed-in",
        type=Path,
        nargs="+",
        help="play the failed seeds of these evaluation run dirs (e.g. results/*-screen-*)",
    )
    parser.add_argument(
        "--all-failures",
        action="store_true",
        help="also hand over when the plate is still in rack 1 or lies flat on the table",
    )
    args = parser.parse_args()
    seed_list = list(args.seeds or [])
    for run in args.failed_in or []:
        if not (run / "episodes.csv").is_file():
            continue  # e.g. the shell glob also matched a run's .log file
        rows = csv.DictReader(open(run / "episodes.csv"))
        seed_list += [int(r["seed"]) for r in rows if r["success"] != "True"]
    for seed in seed_list or [args.seed]:
        if not COLLECT_SEED_START <= seed < 6_000_000:
            sys.exit(
                f"seed {seed}: collection seeds must be in [5,500,000, 6,000,000) "
                "so recoveries never come from validation or evaluation seeds"
            )
    if seed_list:
        print(f"{len(seed_list)} seeds to play: {seed_list}")

    env = make_environment()
    inner = env.inner_env
    model, data = inner.model, inner.data
    ik = LeftArmIK(model)
    log = REPO / "results" / "collect_recovery.stderr.log"
    log.parent.mkdir(exist_ok=True)
    process = PolicyProcess(args.policy, REPO, log, python=args.policy_python)
    panel = ControlPanel()
    s = {"seed": args.seed}

    def say(msg, status=None):
        print(msg, flush=True)
        panel.status.set(status or msg.strip())

    def new_episode():
        if seed_list:
            seed = seed_list.pop(0)
        elif args.seeds or args.failed_in:
            raise Done
        else:
            seed = s["seed"]
            s["seed"] += 1
        ts = env.reset(seed=seed)
        process.reset()
        say(
            f"\nseed={seed} source slot={observed_source_slot(env) + 1}: policy running ...",
            f"seed {seed}: policy running (~34 s if it fails)\nwait for YOUR TURN, or press T",
        )
        return seed, ts

    def set_marker(viewer, pos):
        scn = viewer.user_scn
        if pos is None:
            scn.ngeom = 0
            return
        mujoco.mjv_initGeom(
            scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.015, 0, 0]),
            pos,
            np.eye(3).ravel(),
            np.array([0.1, 0.9, 0.2, 0.6], np.float32),
        )
        scn.ngeom = 1

    saved = 0
    try:
        seed, ts = new_episode()
        queue, steps, human, last_raw = [], 0, None, None
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                start = time.perf_counter()
                hand_delta, events = panel.poll()
                if panel.closed:
                    raise Done
                if "skip" in events or (human and "discard" in events):
                    say("  discarded" if human else "  skipped")
                    set_marker(viewer, None)
                    seed, ts = new_episode()
                    queue, steps, human, last_raw = [], 0, None, None
                    continue

                failed = ts.last() and not env.episode_succeeded() and not env.episode_fell()
                takeover = human is None and (
                    "takeover" in events or steps == args.takeover or failed
                )
                if human is None and ts.last() and not takeover:
                    ok = env.episode_succeeded() and not env.episode_fell()
                    say(f"  {'SUCCESS' if ok else 'FELL'} after {steps} steps; next seed")
                    seed, ts = new_episode()
                    queue, steps, last_raw = [], 0, None
                    continue
                if takeover and failed and not args.all_failures:
                    r = plate_report(inner)
                    if r["in_rack1"] or (r["tilt_deg"] > 60 and r["touch_table"]):
                        kind = "plate still in rack 1" if r["in_rack1"] else "plate flat on table"
                        say(f"  failed with {kind}: skipped (use --all-failures to keep)")
                        seed, ts = new_episode()
                        queue, steps, last_raw = [], 0, None
                        continue
                if takeover:
                    if last_raw is None:  # no policy action yet: hold the current pose
                        last_raw = np.zeros(20)
                        last_raw[2] = 0.74
                        last_raw[4:18] = [
                            data.qpos[model.jnt_qposadr[model.joint(PREFIX + n).id]]
                            for n in LAYOUT["limb_names"]
                        ]
                    q_cmd = last_raw[LEFT_ARM].copy()
                    pos, mat = ik.fk(data.qpos.copy(), q_cmd)
                    ik.bias[:] = 0
                    human = {
                        "step": steps,
                        "rec": Recorder(),
                        "q": q_cmd,
                        "local": ik.to_local(data, pos, mat),
                        "grip": float(last_raw[LEFT_GRIP]),
                        "grip_cmd": float(last_raw[LEFT_GRIP] > 0.5),
                        "hold": -1,
                        "status_t": 0.0,
                        "reason": "key"
                        if "takeover" in events
                        else ("timeout" if failed else "auto"),
                    }
                    say(f"\n  >>> YOUR TURN (step {steps}, {human['reason']}) <<<")

                if human is None:  # ---------------- policy drives
                    if not queue:
                        queue = list(validate_actions(process.act(observation(ts))))
                    a = queue.pop(0)
                    last_raw = to_raw(a)
                    ts = env.step(a)
                    steps += 1
                    viewer.sync()
                    elapsed = time.perf_counter() - start
                    time.sleep(max(0.0, env.control_step_seconds / args.speed - elapsed))
                    continue

                # ---------------- human drives
                h = human
                if "grip" in events:
                    h["grip_cmd"] = 1.0 - h["grip_cmd"]
                    say(f"  gripper -> {'CLOSE' if h['grip_cmd'] else 'OPEN'}")
                if "save" in events and h["hold"] < 0:
                    h["hold"] = HOLD_STEPS
                    say("  saving after 0.5 s hold ...")
                if h["hold"] < 0 and np.any(hand_delta):
                    local_pos, local_mat = h["local"]
                    # Leash: the target may lead the real hand by at most LEASH_M, so an
                    # unreachable or blocked target cannot wind the arm up into chatter.
                    hand_now, _ = ik.to_local(data, data.site_xpos[ik.site].copy(), local_mat)
                    offset = local_pos + hand_delta - hand_now
                    dist = np.linalg.norm(offset)
                    if dist > LEASH_M:
                        offset *= LEASH_M / dist
                    h["local"] = (hand_now + offset, local_mat)

                world_pos, world_mat = ik.to_world(data, *h["local"])
                q_new = ik.track(data, h["q"], world_pos, world_mat)
                h["q"] = h["q"] + np.clip(q_new - h["q"], -MAX_DQ, MAX_DQ)
                h["grip"] += np.clip(h["grip_cmd"] - h["grip"], -0.1, 0.1)  # ~0.2 s open/close
                raw = last_raw.copy()
                raw[0] = raw[3] = 0.0
                raw[1] = panel.base_command() if h["hold"] < 0 else 0.0
                raw[LEFT_ARM], raw[LEFT_GRIP] = h["q"], h["grip"]
                a = to_norm(raw)
                h["rec"].add(observation(ts), to_raw(a))  # store exactly what was executed
                ts = env.step(a)
                steps += 1
                set_marker(viewer, world_pos)
                viewer.sync()

                if env.episode_fell():
                    say("  robot fell: recovery discarded")
                    set_marker(viewer, None)
                    seed, ts = new_episode()
                    queue, steps, human, last_raw = [], 0, None, None
                    continue
                if time.perf_counter() - h["status_t"] > 0.5:
                    h["status_t"] = time.perf_counter()
                    r = plate_report(inner)
                    ok = "  SUCCESS conditions met" if inner.success else ""
                    msg = f"    [{steps - h['step']:4d} rec] {verdict(r)}{ok}"
                    print(msg, flush=True)
                    panel.status.set(
                        f"seed {seed}: YOUR TURN ({steps - h['step']} steps recorded)\n"
                        f"gripper {'CLOSED' if h['grip_cmd'] else 'open'}\n"
                        f"{verdict(r)}{ok}"
                    )
                if h["hold"] > 0:
                    h["hold"] -= 1
                elif h["hold"] == 0:
                    meta = {
                        "seed": seed,
                        "source_slot": observed_source_slot(env),
                        "takeover_step": h["step"],
                        "takeover_reason": h["reason"],
                        "frames": len(h["rec"].action),
                        "success_at_save": bool(inner.success),
                        "policy": args.policy,
                    }
                    path = h["rec"].save(args.out, meta)
                    saved += 1
                    say(
                        f"  saved {meta['frames']} frames -> {path.relative_to(REPO)} "
                        f"(success={meta['success_at_save']}; {saved} this session)"
                    )
                    set_marker(viewer, None)
                    seed, ts = new_episode()
                    queue, steps, human, last_raw = [], 0, None, None
                    continue
                elapsed = time.perf_counter() - start
                time.sleep(max(0.0, env.control_step_seconds / args.human_speed - elapsed))
    except Done:
        print(f"\ndone; {saved} recoveries saved this session")
    except RuntimeError as exc:
        print(f"\npolicy error: {exc} (see {log})", file=sys.stderr)
    finally:
        panel.close()
        process.close()
        env.close()


if __name__ == "__main__":
    main()
