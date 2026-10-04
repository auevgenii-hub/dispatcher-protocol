#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Length-normalized sequence log-probability (logprob_norm) for the Nepali and
Tagalog baseline translations (thesis Equation 3.5, Table 4.1).

generate() is re-run on the same plugin checkpoints that produced
baseline_translation, with output_scores=True and return_dict_in_generate=True
and the same beam size, maximum length and truncation, so the top beam should
reproduce baseline_translation exactly. Every row records whether it did
(text_match).

logprob_norm = mean of the token log-probabilities of the top beam, from
model.compute_transition_scores(..., normalize_logits=True).

One language per run, on CPU.

Input:  WORK/baseline_npi_tgl.csv (from baseline_translate.py)
Output: WORK/logprob_<language>.csv (published in evaluation/data/)

Usage: python3 compute_logprob.py npi|tgl
"""
import os
import sys
import time
import torch
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import common  # noqa: E402

BASE_CSV = os.path.join(common.WORK, "baseline_npi_tgl.csv")
OUT_DIR = str(common.WORK)


def load_plugin(lang):
    if lang == "tgl":
        from plugins.plugin_tl import TlPlugin
        return TlPlugin(device="cpu")
    if lang == "npi":
        from plugins.plugin_ne import NePlugin
        return NePlugin(device="cpu")
    raise ValueError(lang)


def generate_with_scores(plugin, lang, source_text):
    inputs = plugin.tokenizer(
        source_text, return_tensors="pt", truncation=True, max_length=plugin.max_length
    ).to(plugin.device)

    gen_kwargs = dict(
        num_beams=plugin.beam_size, max_length=plugin.max_length,
        output_scores=True, return_dict_in_generate=True,
    )
    if lang == "npi":
        gen_kwargs["forced_bos_token_id"] = plugin._forced_bos_token_id

    with torch.no_grad():
        out = plugin.model.generate(**inputs, **gen_kwargs)

    decoded = plugin.tokenizer.decode(out.sequences[0], skip_special_tokens=True)

    transition_scores = plugin.model.compute_transition_scores(
        out.sequences, out.scores, out.beam_indices, normalize_logits=True
    )[0]
    finite = transition_scores[torch.isfinite(transition_scores)]
    n_tok = int(finite.numel())
    logprob_norm = float(finite.mean()) if n_tok > 0 else float("nan")
    return decoded, logprob_norm, n_tok


def main():
    lang = sys.argv[1]
    out_csv = os.path.join(OUT_DIR, f"logprob_{lang}.csv")

    df = pd.read_csv(BASE_CSV)
    df = df[df["language"] == lang].reset_index(drop=True)

    done_ids = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done_ids = set(prev["sentence_id"])
        print(f"Resuming {lang}: {len(done_ids)} already done.")
        rows = prev.to_dict("records")
    else:
        rows = []

    print(f"Loading {lang} plugin...")
    plugin = load_plugin(lang)
    print(f"{lang} plugin loaded. Scoring {len(df) - len(done_ids)} remaining sentences.")

    t0 = time.time()
    mismatches = 0
    write_every = 25
    for i, r in df.iterrows():
        if r["sentence_id"] in done_ids:
            continue
        decoded, logprob_norm, n_tok = generate_with_scores(plugin, lang, r["source_text"])
        text_match = (decoded.strip() == str(r["baseline_translation"]).strip())
        if not text_match:
            mismatches += 1
        rows.append(dict(
            language=lang, sentence_id=r["sentence_id"],
            logprob_norm=logprob_norm, n_tokens=n_tok, text_match=text_match,
        ))
        if len(rows) % write_every == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  {lang}: {len(rows)}/{len(df)} ({time.time()-t0:.0f}s, "
                  f"{mismatches} mismatches so far)")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"{lang}: done. {len(rows)} rows, {mismatches} text mismatches, "
          f"{time.time()-t0:.0f}s. Wrote {out_csv}")


if __name__ == "__main__":
    main()
