"""Train the ACT baseline on the exported move_plate demos (run in .venv-act).

    .venv-act/bin/python -m submission.act.train --data datasets/move_plate \
        --out checkpoints/act-baseline --steps 150000 --max-hours 8.5

Behaviour cloning on the 60 human demos only; no simulator access, no
privileged state, no evaluation seeds.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from submission.act.model import ACT, ACTConfig


def load_data(roots):
    """Load and concatenate one or more exported datasets (e.g. demos + recoveries)."""
    t = time.time()
    rgb, state, action, ends, offset = [], [], [], [], 0
    for root in roots:
        rgb.append(np.load(root / "rgb.npy"))  # fully into RAM (~2.4 GB uint8 for the demos)
        state.append(np.load(root / "state.npy"))
        action.append(np.load(root / "action.npy"))
        ends.append(np.load(root / "episode_ends.npy") + offset)
        offset += len(state[-1])
        print(f"  {root}: {len(state[-1])} frames / {len(ends[-1])} episodes", flush=True)
    rgb = rgb[0] if len(rgb) == 1 else np.concatenate(rgb)
    state, action, ends = np.concatenate(state), np.concatenate(action), np.concatenate(ends)
    starts = np.concatenate([[0], ends[:-1]])
    ep_end = np.empty(len(state), np.int64)  # exclusive end of each frame's episode
    for s, e in zip(starts, ends, strict=True):
        ep_end[s:e] = e
    print(
        f"loaded {len(state)} frames / {len(ends)} episodes in {time.time() - t:.0f}s", flush=True
    )
    return rgb, state, action, ep_end


def norm_stats(state, action):
    s_mean, s_std = state.mean(0), state.std(0)
    a_mean, a_std = action.mean(0), action.std(0)
    s_std = np.where(s_std < 1e-2, 1.0, s_std)
    a_std = np.where(a_std < 1e-2, 1.0, a_std)
    return {
        "state_mean": s_mean.tolist(),
        "state_std": s_std.tolist(),
        "action_mean": a_mean.tolist(),
        "action_std": a_std.tolist(),
    }


def random_shift(x, pad=4):
    """DrQ-style random shift. x: (N, 3, H, W) float."""
    n, _, h, w = x.shape
    x = F.pad(x, (pad,) * 4, mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    base = torch.linspace(-1 + eps, 1 - eps, h + 2 * pad, device=x.device)[:h]
    grid_y, grid_x = torch.meshgrid(base, base, indexing="ij")
    grid = torch.stack([grid_x, grid_y], -1)[None].expand(n, -1, -1, -1)
    shift = torch.randint(0, 2 * pad + 1, (n, 1, 1, 2), device=x.device).float()
    grid = grid + shift * 2.0 / (h + 2 * pad)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", default=["datasets/move_plate"])
    p.add_argument("--out", default="checkpoints/act-baseline")
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--max-hours", type=float, default=8.5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--save-every", type=int, default=25_000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    rgb, state, action, ep_end = load_data([Path(d) for d in args.data])
    stats = norm_stats(state, action)
    s_mean, s_std, a_mean, a_std = (
        torch.tensor(stats[k], device=dev)
        for k in ("state_mean", "state_std", "action_mean", "action_std")
    )
    state_t = torch.from_numpy(state)
    action_t = torch.from_numpy(action)

    cfg = ACTConfig(chunk=args.chunk)
    model = ACT(cfg).to(dev)
    backbone = set(map(id, model.backbone_parameters()))
    params = [
        {"params": [q for q in model.parameters() if id(q) not in backbone], "lr": args.lr},
        {"params": list(model.backbone_parameters()), "lr": args.lr_backbone},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]
    warmup = 1000
    print(f"params: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M", flush=True)
    (out / "config.json").write_text(
        json.dumps({"args": vars(args), "model": cfg.to_dict()}, indent=2)
    )

    offsets = np.arange(cfg.chunk)

    def batch():
        idx = rng.integers(0, len(state), args.batch)
        # Chunk indices, clamped to the episode's last frame (robot holds after release).
        cidx = np.minimum(idx[:, None] + offsets[None], ep_end[idx, None] - 1)
        img = torch.from_numpy(rgb[idx]).pin_memory().to(dev, non_blocking=True)
        st = state_t[idx].to(dev, non_blocking=True)
        act = action_t[torch.from_numpy(cidx)].to(dev, non_blocking=True)
        img = img.float().div_(255).flatten(0, 1)  # (B*3, 3, 84, 84)
        img = random_shift(img).view(args.batch, cfg.num_cameras, 3, 84, 84)
        return img, (st - s_mean) / s_std, (act - a_mean) / a_std

    def save(path, step):
        torch.save(
            {"model": model.state_dict(), "config": cfg.to_dict(), "stats": stats, "step": step},
            path,
        )

    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    tlog = t0
    run = {"loss": 0.0, "l1": 0.0, "kl": 0.0}
    step = 0
    model.train()
    for step in range(1, args.steps + 1):
        frac = step / args.steps
        scale = min(1.0, step / warmup) * 0.5 * (1 + math.cos(math.pi * frac))
        for g, lr in zip(opt.param_groups, base_lrs, strict=True):
            g["lr"] = lr * max(scale, 0.01)

        img, st, act = batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, mu, logvar = model(img, st, act)
        pred, mu, logvar = pred.float(), mu.float(), logvar.float()
        l1 = F.l1_loss(pred, act)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        loss = l1 + args.kl_weight * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        run["loss"] += loss.item() if step % 20 == 0 else 0.0
        run["l1"] += l1.item() if step % 20 == 0 else 0.0
        run["kl"] += kl.item() if step % 20 == 0 else 0.0
        if step % args.log_every == 0:
            n = args.log_every / 20
            now = time.time()
            rec = {
                "step": step,
                "loss": run["loss"] / n,
                "l1": run["l1"] / n,
                "kl": run["kl"] / n,
                "lr": opt.param_groups[0]["lr"],
                "steps_per_s": args.log_every / (now - tlog),
                "hours": (now - t0) / 3600,
                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run = dict.fromkeys(run, 0.0)
            tlog = now
        if step % args.save_every == 0:
            save(out / f"step_{step:07d}.pt", step)
            save(out / "last.pt", step)
        if (time.time() - t0) / 3600 > args.max_hours:
            print(f"time budget reached at step {step}", flush=True)
            break
    save(out / "last.pt", step)
    print(f"done: {step} steps in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
