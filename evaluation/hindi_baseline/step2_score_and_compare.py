#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hindi baseline, step 2: reference-based quality (COMET
Unbabel/wmt22-comet-da; chrF++ with sacrebleu, word_order=2) and the
reference-free CometKiwi score (Unbabel/wmt22-cometkiwi-da, input
{"src", "mt"}) for the NLLB-200 Hindi output.

Adds the columns chrf, comet and cometkiwi_score to WORK/hin_nllb_baseline.csv.
The score-only columns of that file are published as
evaluation/data/hin_dev_scores.csv.
"""
import os
import sys
import time

import pandas as pd
import sacrebleu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as ct  # noqa: E402

BASELINE_CSV = os.path.join(ct.WORK, "hin_nllb_baseline.csv")
BATCH = 16


def score_chrf(df):
    df["chrf"] = [
        sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score
        for hyp, ref in zip(df["baseline_nllb"], df["reference_text"])
    ]
    return df


def score_comet_ref_based(df):
    print("Loading COMET (Unbabel/wmt22-comet-da)...")
    comet_model, comet_model_name = ct.load_comet_model()
    ct.sanity_check_comet(comet_model)
    print(f"COMET model: {comet_model_name}")

    scores = []
    t0 = time.time()
    rows = df.to_dict("records")
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        inputs = [{"src": r["source_text"], "mt": r["baseline_nllb"], "ref": r["reference_text"]} for r in chunk]
        with ct.no_mps_fork_for_comet():
            out = comet_model.predict(inputs, batch_size=BATCH, gpus=0, num_workers=0)
        scores.extend(out.scores)
        if (i // BATCH) % 10 == 0:
            print(f"  COMET {i+len(chunk)}/{len(rows)} ({time.time()-t0:.0f}s)")
    df["comet"] = scores
    del comet_model
    return df


def score_cometkiwi(df):
    print("Loading CometKiwi (Unbabel/wmt22-cometkiwi-da)...")
    from comet import download_model, load_from_checkpoint
    path = download_model("Unbabel/wmt22-cometkiwi-da")
    kiwi = load_from_checkpoint(path)

    scores = []
    t0 = time.time()
    rows = df.to_dict("records")
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        inputs = [{"src": r["source_text"], "mt": r["baseline_nllb"]} for r in chunk]
        with ct.no_mps_fork_for_comet():
            out = kiwi.predict(inputs, batch_size=BATCH, gpus=0, num_workers=0)
        scores.extend(out.scores)
        if (i // BATCH) % 10 == 0:
            print(f"  CometKiwi {i+len(chunk)}/{len(rows)} ({time.time()-t0:.0f}s)")
    df["cometkiwi_score"] = scores
    del kiwi
    return df


def main():
    df = pd.read_csv(BASELINE_CSV)
    assert len(df) == 997, f"expected 997 rows in {BASELINE_CSV}, got {len(df)}"

    if "chrf" not in df.columns:
        print("Scoring chrF++...")
        df = score_chrf(df)
    if "comet" not in df.columns:
        df = score_comet_ref_based(df)
    if "cometkiwi_score" not in df.columns:
        df = score_cometkiwi(df)
    df.to_csv(BASELINE_CSV, index=False)
    print(f"Updated {BASELINE_CSV} with comet/chrf/cometkiwi_score columns.")


if __name__ == "__main__":
    main()
