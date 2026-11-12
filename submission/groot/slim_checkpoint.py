"""Shrink a fine-tuned GR00T checkpoint to the tensors that differ from the base model.

Fine-tuning leaves the VLM backbone frozen, so most of a saved checkpoint
(~12 GB, fp32) is identical to nvidia/GR00T-N1.7-3B. This keeps only the
changed or new tensors (cast to bf16, the base model's dtype) plus the small
config/processor files, so a checkpoint can be copied over a slow link:

    python -m submission.groot.slim_checkpoint slim CKPT_DIR OUT_DIR
    python -m submission.groot.slim_checkpoint rebuild OUT_DIR REBUILT_DIR

`rebuild` writes a full checkpoint directory (base tensors overlaid with the
slim ones) that GrootPolicy / eval.sh can load. Run in the Isaac-GR00T venv.
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE = "nvidia/GR00T-N1.7-3B"
WEIGHTS = "trained_delta.safetensors"


def iter_tensors(directory):
    for f in sorted(glob.glob(str(Path(directory) / "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                yield k, s.get_tensor(k)


def copy_small_files(src, dst):
    for p in Path(src).iterdir():
        if p.suffix == ".safetensors" or p.name.endswith(".index.json"):
            continue
        if p.is_dir():
            shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
        elif p.stat().st_size < 50_000_000 and p.suffix not in {".pt", ".pth"}:
            shutil.copy2(p, dst / p.name)


def slim(ckpt, out):
    base_dir = Path(snapshot_download(BASE))
    base = {}
    for f in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                base[k] = (str(f), k)
    keep, same = {}, 0
    for k, t in iter_tensors(ckpt):
        t16 = t.to(torch.bfloat16)
        if k in base:
            with safe_open(base[k][0], "pt") as s:
                b = s.get_tensor(k)
            if b.shape == t16.shape and torch.equal(b.to(torch.bfloat16), t16):
                same += 1
                continue
        keep[k] = t16.contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file(keep, str(out / WEIGHTS), metadata={"base_model": BASE})
    copy_small_files(ckpt, out)
    n = sum(t.numel() for t in keep.values())
    (out / "slim.json").write_text(
        json.dumps(
            {
                "source": str(ckpt),
                "base_model": BASE,
                "kept_tensors": len(keep),
                "kept_params": n,
                "unchanged_tensors": same,
            },
            indent=2,
        )
    )
    print(f"kept {len(keep)} tensors ({n / 1e6:.1f}M params), {same} unchanged -> {out}")


def rebuild(slim_dir, out):
    base_dir = Path(snapshot_download(BASE))
    delta = load_file(str(Path(slim_dir) / WEIGHTS))
    full = {k: t for k, t in iter_tensors(base_dir)}
    full.update(delta)
    out.mkdir(parents=True, exist_ok=True)
    save_file(full, str(out / "model.safetensors"))
    copy_small_files(slim_dir, out)
    (out / WEIGHTS).unlink(missing_ok=True)
    print(f"rebuilt {len(full)} tensors ({len(delta)} from {slim_dir}) -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["slim", "rebuild"])
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    (slim if a.mode == "slim" else rebuild)(a.src, a.out)


if __name__ == "__main__":
    main()
