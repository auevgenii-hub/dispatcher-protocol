#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransQuest scores (TransQuest/monotransquest-da-multilingual), run in a
separate virtual environment with transformers==4.30.2, because TransQuest
imports transformers.convert_graph_to_onnx, which newer transformers versions
removed. CPU only.

Usage: <tq_venv>/bin/python3 transquest_score.py in.csv out.csv
  in.csv columns:  idx,source_text,hyp_text
  out.csv columns: idx,transquest_score
"""
import sys
import csv
import warnings

warnings.filterwarnings("ignore")


def main():
    in_csv, out_csv = sys.argv[1], sys.argv[2]

    rows = []
    with open(in_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["idx", "transquest_score"])
        return

    from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel

    model = MonoTransQuestModel(
        "xlmroberta", "TransQuest/monotransquest-da-multilingual",
        num_labels=1, use_cuda=False,
        args={"silent": True, "no_cache": True, "no_save": True},
    )

    pairs = [[r["source_text"], r["hyp_text"]] for r in rows]
    preds, _ = model.predict(pairs)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "transquest_score"])
        for r, p in zip(rows, preds):
            w.writerow([r["idx"], float(p)])
    print(f"Wrote {len(rows)} scores to {out_csv}")


if __name__ == "__main__":
    main()
