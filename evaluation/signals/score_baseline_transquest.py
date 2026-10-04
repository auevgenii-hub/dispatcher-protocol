#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransQuest score for every Nepali and Tagalog baseline translation
(thesis section 4.2). Calls transquest_score.py inside the separate virtual
environment given by TQ_PYTHON (default: tq_venv/bin/python3).

Input:  WORK/baseline_npi_tgl.csv
Output: WORK/npi_tgl_transquest_scores.csv (published in evaluation/data/)
"""
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common  # noqa: E402

BASE_CSV = os.path.join(common.WORK, "baseline_npi_tgl.csv")
OUT_CSV = os.path.join(common.WORK, "npi_tgl_transquest_scores.csv")
TMP_IN = os.path.join(common.WORK, "_tq_in.csv")
TMP_OUT = os.path.join(common.WORK, "_tq_out.csv")
TQ_PYTHON = os.environ.get("TQ_PYTHON", "tq_venv/bin/python3")
TQ_SCRIPT = os.path.join(HERE, "transquest_score.py")


def main():
    df = pd.read_csv(BASE_CSV)
    df["idx"] = df["language"] + "|" + df["sentence_id"].astype(str)
    df[["idx", "source_text"]].assign(hyp_text=df["baseline_translation"])[
        ["idx", "source_text", "hyp_text"]].to_csv(TMP_IN, index=False)

    proc = subprocess.run([TQ_PYTHON, TQ_SCRIPT, TMP_IN, TMP_OUT], capture_output=True, text=True)
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        raise RuntimeError("transquest_score.py failed")

    scores = pd.read_csv(TMP_OUT)
    scores[["language", "sentence_id"]] = scores["idx"].str.split("|", n=1, expand=True)
    scores[["language", "sentence_id", "transquest_score"]].to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(scores)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
