#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hindi baseline, step 1 (thesis sections 3.5 and 4.2): translate the 997
FLORES-200 dev sentences with facebook/nllb-200-distilled-600M
(eng_Latn -> hin_Deva) and record the length-normalized sequence
log-probability from the same generate() call.

logprob_norm is the mean of the per-token log-probabilities returned by
compute_transition_scores(..., normalize_logits=True).

n_tokens_src, n_tokens_out and ratio_src use a \\w+ regular expression, not
the model's subword tokenizer.

Output: WORK/hin_nllb_baseline.csv (contains text; not committed).
Resumable: already translated sentence_ids are skipped.
"""
import os
import re
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import WORK, load_flores_dev  # noqa: E402

OUT_CSV = os.path.join(WORK, "hin_nllb_baseline.csv")

TOK = re.compile(r"\w+")


def n_tok(text):
    return len(TOK.findall(str(text)))


def load_dev_hin():
    sub = load_flores_dev("hin")
    assert sub["sentence_id"].is_unique
    return sub


def load_plugin():
    from plugins.nllb_plugin import NLLBPlugin, ISO3_TO_FLORES
    ISO3_TO_FLORES.setdefault("hin", "hin_Deva")
    # CPU is used for long batch runs; MPS stalled under memory pressure.
    return NLLBPlugin(
        checkpoint="facebook/nllb-200-distilled-600M",
        source_lang="eng",
        target_lang="hin",
        beam_size=5,
        device="cpu",
    )


def generate_with_scores(plugin, source_text):
    inputs = plugin.tokenizer(
        source_text, return_tensors="pt", truncation=True, max_length=plugin.max_length
    ).to(plugin.device)
    with torch.no_grad():
        out = plugin.model.generate(
            **inputs,
            forced_bos_token_id=plugin._forced_bos_token_id,
            num_beams=plugin.beam_size,
            max_length=plugin.max_length,
            output_scores=True,
            return_dict_in_generate=True,
        )
    decoded = plugin.tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    transition_scores = plugin.model.compute_transition_scores(
        out.sequences, out.scores, out.beam_indices, normalize_logits=True
    )[0]
    finite = transition_scores[torch.isfinite(transition_scores)]
    n_tok_model = int(finite.numel())
    logprob_norm = float(finite.mean()) if n_tok_model > 0 else float("nan")
    return decoded, logprob_norm


def main():
    dev = load_dev_hin()

    done_ids = set()
    rows = []
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done_ids = set(prev["sentence_id"])
        rows = prev.to_dict("records")
        print(f"Resuming: {len(done_ids)} already translated.")

    print("Loading NLLB plugin (facebook/nllb-200-distilled-600M, eng->hin)...")
    plugin = load_plugin()
    print("Plugin loaded.")

    t0 = time.time()
    n_done_this_run = 0
    for i, r in dev.iterrows():
        sid = r["sentence_id"]
        if sid in done_ids:
            continue
        decoded, logprob_norm = generate_with_scores(plugin, r["source_text"])
        src_n = n_tok(r["source_text"])
        out_n = n_tok(decoded)
        rows.append({
            "sentence_id": sid,
            "source_text": r["source_text"],
            "reference_text": r["reference_text"],
            "baseline_nllb": decoded,
            "n_tokens_src": src_n,
            "n_tokens_out": out_n,
            "ratio_src": (out_n / src_n) if src_n else float("nan"),
            "logprob_norm": logprob_norm,
        })
        n_done_this_run += 1
        if n_done_this_run % 25 == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            elapsed = time.time() - t0
            print(f"  {len(rows)}/997 ({elapsed:.0f}s, {elapsed/n_done_this_run:.2f}s/sentence avg)")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Done. {len(rows)} rows written to {OUT_CSV} ({time.time()-t0:.0f}s this run)")


if __name__ == "__main__":
    main()
