#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hindi baseline, step 4 (thesis section 4.4, Table 4.3): error categories in
the population flagged by the trigger.

The trigger is logprob_norm at the language's bottom quartile. The flagged
sentences are sorted by sentence_id, shuffled with random.Random(42) and
split in half: a discovery half for finding patterns and a validation half
for checking them. Each flagged sentence is categorized with categorize()
from evaluation/error_causes/categorizer.py, the same
categorizer used for Nepali, Tagalog and Igede.

Outputs (in WORK): hin_nllb_root_cause_logprob.csv (contains text) and
hin_nllb_root_cause_logprob_summary.csv. The category labels are published
in evaluation/error_causes/flagged_categories.csv.
"""
import os
import random
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "error_causes"))
from common import WORK  # noqa: E402
from categorizer import categorize  # noqa: E402

BASELINE_CSV = os.path.join(WORK, "hin_nllb_baseline.csv")
ROOT_CAUSE_CSV = os.path.join(WORK, "hin_nllb_root_cause_logprob.csv")
SUMMARY_CSV = os.path.join(WORK, "hin_nllb_root_cause_logprob_summary.csv")

SEED = 42
CATEGORY_ORDER = [
    "transliteration_copy_through", "digit_number", "omission",
    "repetition", "mistranslation_other",
]

# same orientation convention as step3_metric_shootout.py
HIGHER_BETTER = {
    "logprob_norm": True,
    "cometkiwi_score": True,
    "ratio_src": True,
    "src_len": False,
}


def main():
    df = pd.read_csv(BASELINE_CSV)
    df["src_len"] = df["n_tokens_src"]

    # The trigger metric is fixed to logprob_norm, the signal selected under
    # the chrF++ label (thesis section 4.2).
    metric_name = "logprob_norm"
    higher_better = HIGHER_BETTER[metric_name]
    print(f"Trigger metric: {metric_name} (higher_better={higher_better})")

    vals = df[metric_name].astype(float)
    if higher_better:
        threshold = vals.quantile(0.25)
        triggered = vals <= threshold
    else:
        threshold = vals.quantile(0.75)
        triggered = vals >= threshold

    df["trigger_metric_name"] = metric_name
    df["trigger_metric_value"] = vals
    df["quartile_threshold_value"] = threshold
    df["triggered"] = triggered.astype(int)
    n_triggered = int(df["triggered"].sum())
    print(f"n_total=997, n_triggered={n_triggered} ({n_triggered/997:.2%}), threshold={threshold:.6f}")

    # --- discovery/validation split on triggered subset, seed 42 ---
    df["split"] = ""
    trig_ids = sorted(df.loc[df["triggered"] == 1, "sentence_id"].tolist())
    rng = random.Random(SEED)
    rng.shuffle(trig_ids)
    half = len(trig_ids) // 2
    discovery_ids = set(trig_ids[:half])
    validation_ids = set(trig_ids[half:])
    df.loc[df["sentence_id"].isin(discovery_ids), "split"] = "discovery"
    df.loc[df["sentence_id"].isin(validation_ids), "split"] = "validation"
    print(f"n_discovery={len(discovery_ids)}, n_validation={len(validation_ids)}")

    # --- categorize triggered slice ---
    trig = df[df["triggered"] == 1].copy()
    trig["category"] = [
        categorize("hin", row.source_text, row.baseline_nllb, row.reference_text)
        for row in trig.itertuples()
    ]

    out_cols = ["sentence_id", "source_text", "reference_text", "baseline_nllb",
                "comet", "chrf", "cometkiwi_score", "logprob_norm", "ratio_src",
                "trigger_metric_name", "trigger_metric_value", "quartile_threshold_value",
                "triggered", "split", "category"]
    trig[out_cols].to_csv(ROOT_CAUSE_CSV, index=False)
    print(f"\nWrote {ROOT_CAUSE_CSV} ({len(trig)} rows)")

    # --- category share table, discovery / validation / combined ---
    slices = {"discovery": trig[trig["split"] == "discovery"],
              "validation": trig[trig["split"] == "validation"],
              "combined": trig}
    rows = []
    for slice_name, sdf in slices.items():
        n_slice = len(sdf)
        print(f"\n=== {slice_name} (n={n_slice}) ===")
        for cat in CATEGORY_ORDER:
            sub = sdf[sdf["category"] == cat]
            count = len(sub)
            share = count / n_slice if n_slice else 0.0
            mean_comet = sub["comet"].mean() if count else float("nan")
            rows.append({
                "slice": slice_name, "category": cat, "count": count, "n_slice": n_slice,
                "share_of_slice": round(share, 4),
                "mean_comet_in_category": round(mean_comet, 4) if count else "",
            })
            print(f"  {cat:32s} count={count:4d}  share={share:6.1%}")

    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
