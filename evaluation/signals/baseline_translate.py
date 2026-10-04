#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline translations and reference-based scores for Nepali and Tagalog
(thesis sections 3.5 and 4.1.1).

Every FLORES-200 dev sentence (997 per language) is translated with a single
plugin.translate() call, exactly as dispatcher/app.py does for hop 2, and
scored against its reference with chrF++ (sacrebleu, word_order=2), BLEU
(sacrebleu) and COMET (Unbabel/wmt22-comet-da). Only the eng -> target
plugin is loaded, on CPU, one language per run.

Output: WORK/baseline_npi_tgl.csv (contains text; not committed). The
score-only columns are published as evaluation/data/npi_tgl_dev_scores.csv.
Resumable: (language, sentence_id) pairs already present are skipped.

Usage:
    python3 baseline_translate.py --languages npi,tgl
"""
import argparse
import csv
import gc
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import common  # noqa: E402
import sacrebleu  # noqa: E402

OUT_CSV = os.path.join(common.WORK, "baseline_npi_tgl.csv")
FIELDS = ["language", "sentence_id", "source_text", "reference_text",
          "baseline_translation", "comet", "chrf", "bleu"]


def load_plugin(lang):
    if lang == "tgl":
        from plugins.plugin_tl import TlPlugin
        return TlPlugin(device="cpu")
    if lang == "npi":
        from plugins.plugin_ne import NePlugin
        return NePlugin(device="cpu")
    raise ValueError(f"unsupported language {lang!r}")


def done_keys():
    if not os.path.exists(OUT_CSV):
        return set()
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        return {(r["language"], r["sentence_id"]) for r in csv.DictReader(f)}


def append_rows(rows):
    header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if header:
            w.writeheader()
        w.writerows(rows)


def run(languages, chunk_size=16):
    from plugins.plugin_base import TranslationRequest
    comet_model, _ = common.load_comet_model()
    common.sanity_check_comet(comet_model)
    done = done_keys()
    for lang in languages:
        pairs = common.load_flores_dev(lang)
        plugin = load_plugin(lang)
        buffer, comet_inputs = [], []

        def flush():
            nonlocal buffer, comet_inputs
            if not buffer:
                return
            with common.no_mps_fork_for_comet():
                out = comet_model.predict(comet_inputs, batch_size=16, gpus=0, num_workers=0)
            for row, score in zip(buffer, out.scores):
                row["comet"] = score
            append_rows(buffer)
            buffer, comet_inputs = [], []

        t0 = time.time()
        for r in pairs.itertuples():
            if (lang, r.sentence_id) in done:
                continue
            req = TranslationRequest(text=r.source_text, source_lang="eng",
                                     target_lang=lang, request_id=str(uuid.uuid4()))
            hyp = plugin.translate(req).translated_text
            buffer.append({
                "language": lang, "sentence_id": r.sentence_id,
                "source_text": r.source_text, "reference_text": r.reference_text,
                "baseline_translation": hyp, "comet": None,
                "chrf": sacrebleu.sentence_chrf(hyp, [r.reference_text], word_order=2).score,
                "bleu": sacrebleu.sentence_bleu(hyp, [r.reference_text]).score,
            })
            comet_inputs.append({"src": r.source_text, "mt": hyp, "ref": r.reference_text})
            if len(buffer) >= chunk_size:
                flush()
        flush()
        print(f"{lang}: done in {time.time() - t0:.0f}s")
        del plugin
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="npi,tgl")
    ap.add_argument("--chunk-size", type=int, default=16)
    a = ap.parse_args()
    run(a.languages.split(","), a.chunk_size)
