#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error causes for Nepali and Tagalog, step 3 (thesis section 4.4, Table 4.3):
categorize every flagged sentence with categorizer.categorize(), the same
categorizer used for Hindi and Igede, and report category shares on the
discovery half, the validation half and both together. Only a distribution
that both halves confirm is reported in the thesis.

Outputs (in WORK): root_cause_by_validated_trigger_flagged_for_review.csv
(contains text) and root_cause_by_validated_trigger_causes.csv. The category
labels are published in evaluation/error_causes/flagged_categories.csv.
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import common  # noqa: E402
from categorizer import categorize  # noqa: E402

TRIGGERED_CSV = os.path.join(common.WORK, "root_cause_triggered_raw.csv")
CAUSES_OUT = os.path.join(common.WORK, "root_cause_by_validated_trigger_causes.csv")
FLAGGED_OUT = os.path.join(common.WORK, "root_cause_by_validated_trigger_flagged_for_review.csv")

CATEGORY_ORDER = [
    "transliteration_copy_through", "digit_number", "omission",
    "repetition", "mistranslation_other",
]


def main():
    df = pd.read_csv(TRIGGERED_CSV)
    trig = df[df.triggered == 1].copy()

    trig["category"] = [
        categorize(row.language, row.source_text, row.baseline_translation, row.reference_text)
        for row in trig.itertuples()
    ]
    trig.to_csv(FLAGGED_OUT, index=False)
    print(f"Wrote {FLAGGED_OUT} ({len(trig)} rows)")

    slices = {"discovery": trig[trig.split == "discovery"],
              "validation": trig[trig.split == "validation"],
              "combined": trig}

    cause_rows = []
    for slice_name, sdf in slices.items():
        print(f"\n=== {slice_name} ===")
        for lang, g in sdf.groupby("language", sort=False):
            n_slice = len(g)
            print(f"\n{lang} (n_{slice_name}={n_slice}):")
            for cat in CATEGORY_ORDER:
                sub = g[g["category"] == cat]
                count = len(sub)
                share = count / n_slice if n_slice else 0.0
                mean_comet = sub["comet"].mean() if count else float("nan")
                cause_rows.append({
                    "slice": slice_name, "language": lang, "category": cat,
                    "count": count, "n_slice": n_slice,
                    "share_of_slice": round(share, 4),
                    "mean_comet_in_category": round(mean_comet, 4) if count else "",
                })
                print(f"  {cat:32s} count={count:4d}  share={share:6.1%}  "
                      f"mean_comet={mean_comet if count else float('nan'):.4f}")

    causes_df = pd.DataFrame(cause_rows)
    causes_df.to_csv(CAUSES_OUT, index=False)
    print(f"\nWrote {CAUSES_OUT} ({len(causes_df)} rows)")


if __name__ == "__main__":
    main()
