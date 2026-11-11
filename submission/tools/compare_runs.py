"""Paired comparison of evaluation runs that share the same seeds (analysis only).

    uv run python -m submission.tools.compare_runs BASELINE_DIR RUN_DIR [RUN_DIR ...]

For each run: successes, per-slot successes, mean steps on successes, and the
seed-by-seed discordance against the baseline (wins / losses), with an exact
two-sided McNemar (sign) test p-value on the discordant pairs.
"""

import argparse
import csv
import json
from math import comb
from pathlib import Path


def load(run):
    rows = {r["seed"]: r for r in csv.DictReader(open(run / "episodes.csv"))}
    return rows, json.loads((run / "report.json").read_text())


def mcnemar_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()

    base, _ = load(args.baseline)
    ok = lambda r: r["success"] == "True"  # noqa: E731
    print(f"{'run':44s} succ  slots(1-6)        mean_steps  +win -loss  p")
    for run in [args.baseline, *args.runs]:
        rows, rep = load(run)
        assert rows.keys() == base.keys(), f"{run} uses different seeds"
        slots = " ".join(str(rep["per_source_slot"][str(s)]["successes"]) for s in range(1, 7))
        steps = [int(r["steps"]) for r in rows.values() if ok(r)]
        wins = [s for s in rows if ok(rows[s]) and not ok(base[s])]
        losses = [s for s in rows if ok(base[s]) and not ok(rows[s])]
        print(
            f"{run.name:44s} {rep['successes']:2d}/{rep['completed_episodes']}  {slots:16s}  "
            f"{sum(steps) / max(len(steps), 1):7.1f}     +{len(wins):<2d} -{len(losses):<2d}  "
            f"{mcnemar_p(len(wins), len(losses)):.2f}"
        )
        if run != args.baseline and (wins or losses):
            print(f"{'':44s} wins {wins}  losses {losses}")


if __name__ == "__main__":
    main()
