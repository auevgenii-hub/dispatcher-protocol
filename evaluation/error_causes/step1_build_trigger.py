#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error causes for Nepali and Tagalog, step 1 (thesis section 4.4): flag the
error population with the validated trigger.

For each language, triggered = 1 if logprob_norm <= that language's own
25th percentile of logprob_norm on the FLORES-200 dev set (more negative
means worse). Rows whose re-generated top beam did not reproduce the
baseline translation (text_match == False) are excluded from the logprob
source. No model calls.

Inputs:  WORK/baseline_npi_tgl.csv, WORK/logprob_<language>.csv
Outputs: WORK/root_cause_triggered_raw.csv (contains text),
         evaluation/error_causes/step1_threshold_report.csv
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common  # noqa: E402

LANGS = ["npi", "tgl"]
BASE_CSV = os.path.join(common.WORK, "baseline_npi_tgl.csv")
OUT_CSV = os.path.join(common.WORK, "root_cause_triggered_raw.csv")


def main():
    base = pd.read_csv(BASE_CSV)
    base = base[base["language"].isin(LANGS)]
    frames = []
    for lang in LANGS:
        lp = pd.read_csv(os.path.join(common.WORK, f"logprob_{lang}.csv"))
        if "text_match" in lp:
            lp = lp[lp["text_match"] == True]  # noqa: E712
        frames.append(lp[["language", "sentence_id", "logprob_norm"]])
    m = base.merge(pd.concat(frames), on=["language", "sentence_id"], how="left")

    rows, report = [], []
    for lang, g in m.groupby("language", sort=False):
        g = g[g["logprob_norm"].notna()].copy()
        q1 = g["logprob_norm"].quantile(0.25)
        g["trigger_metric_name"] = "logprob"
        g["trigger_metric_value"] = g["logprob_norm"]
        g["quartile_threshold_value"] = q1
        g["triggered"] = (g["logprob_norm"] <= q1).astype(int)
        n_trig = int(g["triggered"].sum())
        report.append(dict(language=lang, trigger_metric="logprob", n_total=len(g),
                           n_triggered=n_trig, share_triggered=round(n_trig / len(g), 4),
                           threshold_value=round(q1, 6)))
        rows.append(g)

    out_cols = ["sentence_id", "language", "source_text", "reference_text",
                "baseline_translation", "comet", "chrf", "bleu",
                "trigger_metric_name", "trigger_metric_value",
                "quartile_threshold_value", "triggered"]
    pd.concat(rows)[out_cols].to_csv(OUT_CSV, index=False)
    rep = pd.DataFrame(report)
    rep.to_csv(os.path.join(HERE, "step1_threshold_report.csv"), index=False)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
