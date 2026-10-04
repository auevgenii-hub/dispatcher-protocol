#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error causes for Nepali and Tagalog, step 2: split the flagged sentences
into a discovery half and a validation half before any categorization, so
that the split cannot adapt to the categories.

Per language: sort the flagged sentence_ids, shuffle them with
random.Random(42), take the first half as discovery and the rest (including
an odd remainder) as validation. Adds a split column to
WORK/root_cause_triggered_raw.csv in place.
"""
import os
import random
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common  # noqa: E402

OUT_DIR = HERE
TRIGGERED_CSV = os.path.join(common.WORK, "root_cause_triggered_raw.csv")
SEED = 42


def main():
    df = pd.read_csv(TRIGGERED_CSV)
    df["split"] = ""

    report = []
    for lang, g in df[df.triggered == 1].groupby("language", sort=False):
        ids = sorted(g["sentence_id"].tolist())
        rng = random.Random(SEED)
        rng.shuffle(ids)
        half = len(ids) // 2
        discovery_ids = set(ids[:half])
        validation_ids = set(ids[half:])  # odd remainder goes here

        mask_disc = (df.language == lang) & (df.sentence_id.isin(discovery_ids))
        mask_val = (df.language == lang) & (df.sentence_id.isin(validation_ids))
        df.loc[mask_disc, "split"] = "discovery"
        df.loc[mask_val, "split"] = "validation"

        report.append(dict(language=lang, n_triggered=len(ids),
                            n_discovery=len(discovery_ids), n_validation=len(validation_ids)))

    df.to_csv(TRIGGERED_CSV, index=False)
    rep = pd.DataFrame(report)
    print(rep.to_string(index=False))
    rep.to_csv(os.path.join(OUT_DIR, "step2_split_report.csv"), index=False)
    print(f"\nUpdated {TRIGGERED_CSV} in place with split column.")


if __name__ == "__main__":
    main()
