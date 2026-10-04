#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference-free candidate signals for the Nepali and Tagalog baseline
translations (thesis section 4.2, Table 4.1): CometKiwi, LaBSE cosine
similarity between source and translation, and source length.

  CometKiwi: Unbabel/wmt22-cometkiwi-da, input {"src", "mt"}, CPU, fp32.
  LaBSE:     sentence-transformers/LaBSE, cosine of L2-normalized embeddings.
  src_len:   number of whitespace-separated tokens in the source.

TransQuest is scored separately (transquest_score.py) because it needs an
older transformers version. logprob_norm is computed by compute_logprob.py.

Input:  WORK/baseline_npi_tgl.csv
Output: WORK/npi_tgl_candidate_scores.csv (published in evaluation/data/)
Resumable: (language, sentence_id) rows already present are skipped.
"""
import os
import sys
import time
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import common  # noqa: E402

BASE_CSV = os.path.join(common.WORK, "baseline_npi_tgl.csv")
OUT_CSV = os.path.join(common.WORK, "npi_tgl_candidate_scores.csv")

BATCH = 16


_no_mps_fork_for_comet = common.no_mps_fork_for_comet


def main():
    df = pd.read_csv(BASE_CSV)
    df["src_len"] = df["source_text"].astype(str).str.split().str.len()

    if os.path.exists(OUT_CSV):
        done = pd.read_csv(OUT_CSV)
        done_keys = set(zip(done["language"], done["sentence_id"]))
        print(f"Resuming: {len(done_keys)} rows already scored.")
    else:
        done = pd.DataFrame(columns=["language", "sentence_id", "src_len",
                                      "cometkiwi_score", "labse_cosine"])
        done_keys = set()

    todo = df[~df.apply(lambda r: (r["language"], r["sentence_id"]) in done_keys, axis=1)].copy()
    print(f"To score: {len(todo)} / {len(df)} rows")

    if len(todo) == 0:
        print("Nothing to do.")
        return

    # --- CometKiwi ---
    print("Loading CometKiwi (Unbabel/wmt22-cometkiwi-da)...")
    from comet import download_model, load_from_checkpoint
    path = download_model("Unbabel/wmt22-cometkiwi-da")
    kiwi = load_from_checkpoint(path)

    kiwi_scores = []
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        chunk = todo.iloc[i:i + BATCH]
        inputs = [{"src": s, "mt": m} for s, m in
                  zip(chunk["source_text"], chunk["baseline_translation"])]
        with _no_mps_fork_for_comet():
            out = kiwi.predict(inputs, batch_size=BATCH, gpus=0, num_workers=0)
        kiwi_scores.extend(out.scores)
        if (i // BATCH) % 10 == 0:
            print(f"  CometKiwi {i+len(chunk)}/{len(todo)} ({time.time()-t0:.0f}s)")
    todo["cometkiwi_score"] = kiwi_scores
    del kiwi

    # --- LaBSE ---
    print("Loading LaBSE (sentence-transformers/LaBSE)...")
    from sentence_transformers import SentenceTransformer
    labse = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
    src_emb = labse.encode(list(todo["source_text"]), batch_size=32,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=True)
    hyp_emb = labse.encode(list(todo["baseline_translation"]), batch_size=32,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=True)
    todo["labse_cosine"] = np.sum(src_emb * hyp_emb, axis=1)
    del labse

    out_cols = ["language", "sentence_id", "src_len", "cometkiwi_score", "labse_cosine"]
    result = pd.concat([done, todo[out_cols]], ignore_index=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(result)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
