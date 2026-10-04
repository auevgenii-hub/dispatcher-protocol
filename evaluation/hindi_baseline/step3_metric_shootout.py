#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hindi baseline, step 3 (thesis section 4.2, Table 4.1): compare candidate
trigger signals on the NLLB-200 Hindi output.

Bad-translation label: the language's own bottom quartile. The label is
computed twice, once from COMET and once from chrF++. The chrF++ label is
the one used in the thesis, because CometKiwi and COMET belong to the same
model family and an AUC against a COMET label is inflated for CometKiwi.

Candidates (orientation fixed in advance):
  logprob_norm     higher = better
  cometkiwi_score  higher = better
  ratio_src        higher = better (a low output/source ratio signals omission)
  src_len          higher = worse (trivial control)

Paired bootstrap with 5,000 resamples and seed 42; the same resampled
sentence indices are reused for every candidate within a label.

A signal is accepted if its point AUC beats the src_len control and the
95% interval of (best AUC - second-best AUC) does not include zero;
otherwise the result is a tie and the cheaper signal is kept.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import WORK  # noqa: E402

BASELINE_CSV = os.path.join(WORK, "hin_nllb_baseline.csv")
OUT_CSV = os.path.join(WORK, "hin_nllb_metric_shootout.csv")

N_BOOT = 5000
SEED = 42

# candidate -> higher_better
CANDIDATES = {
    "logprob_norm": True,
    "cometkiwi_score": True,
    "ratio_src": True,
    "src_len": False,
}
LABELS = ["comet", "chrf"]


def orient(vals, higher_better):
    return (-vals) if higher_better else vals


def main():
    df = pd.read_csv(BASELINE_CSV)
    df["src_len"] = df["n_tokens_src"]
    n = len(df)
    assert n == 997

    all_rows = []
    for label_col in LABELS:
        q1 = df[label_col].quantile(0.25)
        y = (df[label_col] <= q1).astype(int).to_numpy()
        print(f"\n=== label={label_col} (bottom quartile, threshold={q1:.4f}, "
              f"n_bad={y.sum()}/{n}) ===")

        oriented = {}
        point_auc = {}
        for cand, higher_better in CANDIDATES.items():
            vals = df[cand].to_numpy(dtype=float)
            osc = orient(vals, higher_better)
            oriented[cand] = osc
            point_auc[cand] = roc_auc_score(y, osc)
            print(f"  {cand:16s} point AUC = {point_auc[cand]:.4f}")

        ranked = sorted(point_auc.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_point = ranked[0]
        second_name, second_point = ranked[1]
        print(f"  best={best_name} ({best_point:.4f}), second={second_name} ({second_point:.4f})")

        # paired bootstrap: same resampled indices for every candidate
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        boot_auc = {cand: np.empty(N_BOOT) for cand in CANDIDATES}
        for b in range(N_BOOT):
            yb = y[idx[b]]
            if yb.sum() in (0, n):
                for cand in CANDIDATES:
                    boot_auc[cand][b] = np.nan
                continue
            for cand in CANDIDATES:
                boot_auc[cand][b] = roc_auc_score(yb, oriented[cand][idx[b]])

        diff = boot_auc[best_name] - boot_auc[second_name]
        diff = diff[~np.isnan(diff)]
        diff_ci = np.percentile(diff, [2.5, 97.5])
        beats_control = best_point > point_auc["src_len"]
        validated = bool(beats_control and (diff_ci[0] > 0 or diff_ci[1] < 0))

        print(f"  diff(best-second) 95% CI = [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]")
        print(f"  beats src_len control: {beats_control}  |  VALIDATED: {validated}")

        for cand in CANDIDATES:
            valid_boot = boot_auc[cand][~np.isnan(boot_auc[cand])]
            ci_lo, ci_hi = np.percentile(valid_boot, [2.5, 97.5])
            all_rows.append({
                "label": label_col,
                "candidate": cand,
                "point_auc": round(point_auc[cand], 4),
                "auc_boot_mean": round(float(valid_boot.mean()), 4),
                "auc_ci_lo": round(float(ci_lo), 4),
                "auc_ci_hi": round(float(ci_hi), 4),
                "is_best": cand == best_name,
                "is_second": cand == second_name,
                "beats_src_len_control": point_auc[cand] > point_auc["src_len"],
                "diff_best_minus_second_mean": round(float(diff.mean()), 4) if cand == best_name else "",
                "diff_best_minus_second_ci_lo": round(float(diff_ci[0]), 4) if cand == best_name else "",
                "diff_best_minus_second_ci_hi": round(float(diff_ci[1]), 4) if cand == best_name else "",
                "winner_validated": validated if cand == best_name else "",
            })

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
