"""
followup_confirmation.py  (replaces followup_policy_check.py)
=============================================================
Confirmation run for the staleness-policy decision, with the corrected
PAIRED decision rule pre-registered BEFORE running:

Background: the first 15-seed batch (seeds 42-56) showed reciprocal winning
12/15 under schedule_skew, but the original decision rule compared the mean
gap against an UNPAIRED pooled std -- wrong test for a paired design (same
seed = same world for both policies). This run uses 15 FRESH seeds (57-71)
so the corrected rule is applied to data it has never seen.

Pre-registered decision rule (paired, two-sided alpha=0.05, df=14):
    reciprocal wins >= 11/15 seeds  AND  paired t > 2.14
        -> ADOPT conditional policy: uniform when participation is balanced,
           reciprocal when the Circadian gate detects participation skew
    otherwise
        -> uniform everywhere

Run: python followup_confirmation.py   (30 week-long sims)
"""

import numpy as np
from async_events_system import run_policy, summarize, BASE_SEED

FRESH_SEEDS = range(BASE_SEED + 15, BASE_SEED + 30)   # seeds 57-71, never used
T_CRITICAL = 2.14   # two-sided, alpha=0.05, df=14
MIN_WINS = 11

print("=== Confirmation: uniform vs reciprocal, schedule_skew, "
      "15 FRESH seeds (57-71) ===")
uni, rec = [], []
for seed in FRESH_SEEDS:
    e_u = summarize(run_policy("uniform", seed, "schedule_skew"))["final_err"]
    e_r = summarize(run_policy("reciprocal", seed, "schedule_skew"))["final_err"]
    uni.append(e_u)
    rec.append(e_r)
    print(f"  seed {seed}: uniform={e_u:.4f}  reciprocal={e_r:.4f}  "
          f"{'reciprocal wins' if e_r < e_u else 'uniform wins'}")

uni, rec = np.array(uni), np.array(rec)
diffs = uni - rec                     # positive = reciprocal better
wins = int((rec < uni).sum())
t_stat = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))

print(f"\n  uniform:    {uni.mean():.4f} +/- {uni.std():.4f}")
print(f"  reciprocal: {rec.mean():.4f} +/- {rec.std():.4f}")
print(f"  paired: mean gap = {diffs.mean():.4f}, std of diffs = {diffs.std(ddof=1):.4f}, "
      f"t = {t_stat:.2f}, wins = {wins}/15")

if wins >= MIN_WINS and t_stat > T_CRITICAL:
    print("  DECISION: ADOPT conditional policy -- uniform when balanced, "
          "reciprocal under detected participation skew")
else:
    print("  DECISION: uniform everywhere")
