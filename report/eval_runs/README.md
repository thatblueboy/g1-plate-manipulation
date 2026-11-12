# Evaluator reports

`report.json` and `episodes.csv` exactly as written by the unmodified evaluator for every scored run in the report (per-episode action logs omitted for size). Seed blocks: validation 6,000,000+, public 7,310,000+, recovery collection 5,500,000–5,999,999.

The **selected policy** (ACT + all 22 recoveries, round 2, 100k steps; loaded by default by `submission.policy:make_policy`, checkpoint id `last.pt-1752981f…` / `act-recovery-r2-100k-1752981f`):

- `20261112-0753-val-60-r2-100k` — validation: 54/60, complete, slot-balanced, integrity-valid, status PASS (54 required). Selected on this block after many configurations were compared on it, so the score is optimistic.
- `20261112-public-60-r2-100k` — held-out public block, run once after selection with the exact submitted command: **52/60, status FAIL**.

**Non-qualifying runs.** Runs marked *repo unchanged: no* are flagged NON_QUALIFYING by the evaluator although they completed all planned episodes with balanced slots and the upstream integrity check passed: files covered by the submission digest changed while they ran, because other tools were being developed and training runs started in the same checkout. Their success counts are reported in the report but they are not qualifying runs.

| Run | Used in | Method | Seeds | Success | Per source slot 1–6 | Status | Repo unchanged |
|---|---|---|---|---|---|---|---|
| `20261109-2020-public-last-ensemble` | Table 1 | `act-baseline-ensemble` | 7310000–7310047 | 27/30 | 4/5 5/5 4/5 4/5 5/5 5/5 | PASS | yes |
| `20261109-2020-val-last-chunk10` | Table 1 | `act-baseline-chunk` | 6000000–6000049 | 21/30 | 4/5 3/5 3/5 4/5 5/5 2/5 | FAIL | yes |
| `20261109-2020-val-last-ensemble` | Table 1 | `act-baseline-ensemble` | 6000000–6000049 | 24/30 | 3/5 5/5 3/5 4/5 5/5 4/5 | FAIL | yes |
| `20261109-2020-val-step100k-ensemble` | Table 1 | `act-baseline-ensemble` | 6000000–6000049 | 25/30 | 4/5 5/5 3/5 5/5 4/5 4/5 | FAIL | yes |
| `20261110-0920-public-60-last-ensemble` | Table 1 | `act-baseline-ensemble` | 7310000–7310078 | 49/60 | 9/10 7/10 8/10 8/10 8/10 9/10 | FAIL | yes |
| `20261110-0920-val-60-last-ensemble` | Table 1 | `act-baseline-ensemble` | 6000000–6000070 | 48/60 | 6/10 10/10 8/10 7/10 9/10 8/10 | FAIL | yes |
| `20261110-1816-val-60-k0` | k sweep | `act-baseline-ensemble-k0` | 6000000–6000070 | 47/60 | 7/10 9/10 8/10 7/10 9/10 7/10 | FAIL | yes |
| `20261110-1816-val-60-k-0.01` | k sweep | `act-baseline-ensemble-k-0.01` | 6000000–6000070 | 51/60 | 7/10 9/10 9/10 8/10 9/10 9/10 | NON_QUALIFYING | **no** |
| `20261110-1816-val-60-k0.01` | k sweep | `act-baseline-ensemble-k0.01` | 6000000–6000070 | 49/60 | 6/10 10/10 8/10 8/10 9/10 8/10 | FAIL | yes |
| `20261110-1816-val-60-k-0.05` | k sweep | `act-baseline-ensemble-k-0.05` | 6000000–6000070 | 47/60 | 7/10 8/10 8/10 8/10 9/10 7/10 | NON_QUALIFYING | **no** |
| `20261110-1816-val-60-k0.05` | k sweep | `act-baseline-ensemble-k0.05` | 6000000–6000070 | 50/60 | 7/10 9/10 9/10 8/10 9/10 8/10 | FAIL | yes |
| `20261110-1816-val-60-k-0.1` | k sweep | `act-baseline-ensemble-k-0.1` | 6000000–6000070 | 41/60 | 7/10 8/10 6/10 8/10 6/10 6/10 | NON_QUALIFYING | **no** |
| `20261110-2112-screen-5500000` | recovery collection (round 1 screen) | `act-baseline-ensemble` | 5500000–5500065 | 32/42 | 3/7 5/7 6/7 7/7 5/7 6/7 | NON_QUALIFYING | **no** |
| `20261110-2112-screen-5600000` | recovery collection (round 1 screen) | `act-baseline-ensemble` | 5600000–5600066 | 37/42 | 6/7 6/7 5/7 6/7 7/7 7/7 | NON_QUALIFYING | **no** |
| `20261110-2112-screen-5700000` | recovery collection (round 1 screen) | `act-baseline-ensemble` | 5700000–5700051 | 33/42 | 4/7 7/7 7/7 6/7 5/7 4/7 | NON_QUALIFYING | **no** |
| `20261110-2249-screen-r1-5800000` | recovery collection (round 2 screen) | `act-screen-r1-5800000` | 5800000–5800058 | 40/42 | 7/7 7/7 5/7 7/7 7/7 7/7 | PASS | yes |
| `20261110-2249-screen-r1-5850000` | recovery collection (round 2 screen) | `act-screen-r1-5850000` | 5850000–5850076 | 32/42 | 6/7 6/7 6/7 6/7 6/7 2/7 | FAIL | yes |
| `20261110-2249-screen-r1-5900000` | recovery collection (round 2 screen) | `act-screen-r1-5900000` | 5900000–5900061 | 34/42 | 5/7 4/7 7/7 7/7 7/7 4/7 | FAIL | yes |
| `20261110-2249-val-60-base-100k` | Attempt 1 (baseline 100k) | `act-val-60-base-100k` | 6000000–6000070 | 51/60 | 9/10 10/10 7/10 8/10 8/10 9/10 | NON_QUALIFYING | **no** |
| `20261110-2249-val-60-r1-50k` | Attempt 1 | `act-val-60-r1-50k` | 6000000–6000070 | 52/60 | 8/10 8/10 8/10 10/10 9/10 9/10 | FAIL | yes |
| `20261110-2249-val-60-r1-100k` | Attempt 1 (round 1) | `act-val-60-r1-100k` | 6000000–6000070 | 52/60 | 9/10 9/10 7/10 10/10 9/10 8/10 | FAIL | yes |
| `20261111-0213-val-60-r1-k20-100k` | Attempt 1 (K=20) | `act-r1-k20` | 6000000–6000070 | 42/60 | 7/10 9/10 6/10 7/10 9/10 4/10 | FAIL | yes |
| `20261111-1353-val-60-shift-50k` | Attempt 2 | `act-val-60-shift-50k` | 6000000–6000070 | 40/60 | 7/10 3/10 9/10 8/10 6/10 7/10 | FAIL | yes |
| `20261111-1353-val-60-shift-100k` | Attempt 2 | `act-val-60-shift-100k` | 6000000–6000070 | 38/60 | 5/10 5/10 9/10 9/10 4/10 6/10 | FAIL | yes |
| `groot-r1-checkpoint-10000-val60` | Attempt 3 (GR00T) | `groot-n1.7-new_embodiment` | 6000000–6000070 | 13/60 | 2/10 2/10 2/10 2/10 4/10 1/10 | FAIL | yes |
| `groot-r1-checkpoint-20000-val60` | Attempt 3 (GR00T) | `groot-n1.7-new_embodiment` | 6000000–6000070 | 38/60 | 4/10 4/10 7/10 7/10 7/10 9/10 | FAIL | yes |
| `20261112-0753-val-60-r2-50k` | Attempt 1 (round 2) | `act-val-60-r2-50k` | 6000000–6000070 | 54/60 | 10/10 9/10 7/10 9/10 9/10 10/10 | PASS | yes |
| `20261112-0753-val-60-r2-100k` | Attempt 1 (round 2) — SELECTED POLICY | `act-val-60-r2-100k` | 6000000–6000070 | 54/60 | 9/10 8/10 9/10 9/10 9/10 10/10 | PASS | yes |
| `20261112-public-60-r2-100k` | Selected policy, held-out public check | `act-recovery-r2` | 7310000–7310078 | 52/60 | 10/10 7/10 8/10 8/10 9/10 10/10 | FAIL | yes |
