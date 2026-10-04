# Dispatcher control protocol for modular machine translation

Code accompanying the Master's thesis *Handling lexical asymmetry in modular
machine translation: A dispatcher control protocol for low-resource languages*
(Evgenii Andreev, Melbourne Institute of Technology, 2026), Appendix A.

The dispatcher routes every request through English to a language plugin and,
after the plugin has returned a translation, decides whether the translation
needs further processing. The decision signal (the trigger) is the
length-normalized sequence log-probability that the plugin's own model
produces while decoding. Four languages are served: Hindi (`hin`), Nepali
(`npi`), Tagalog (`tgl`) and Igede (`ige`).

The Igede plugins, their training code and checkpoints are in the companion
repository [igede-mt-plugin](https://github.com/auevgenii-hub/igede-mt-plugin)
(thesis Appendix B).

## Repository map

| Path | Content | Thesis section |
|---|---|---|
| `dispatcher/app.py` | Flask application, the five protocol steps | 3.4 |
| `dispatcher/lang_id.py` | Step 1, language identification (fastText lid.176, threshold 0.7) | 3.4.1 |
| `dispatcher/trigger.py` | Step 3, trigger: `logprob_norm` below the language's development-set lower quartile | 3.4.3, Eq. 3.6 |
| `dispatcher/candidate_generation.py` | Step 4, candidate generation (no generator registered) | 3.4.4 |
| `dispatcher/gate.py` | Step 5, selection gate | 3.4.5 |
| `plugins/` | Plugin contract and the four language plugins | 3.3, 3.5 |
| `tests/` | Plugin contract tests | 3.3 |
| `evaluation/chapter4/` | Recomputes Tables 4.1 and 4.2 and Figure 4.1 from saved scores | 4.2, 4.3 |
| `evaluation/error_causes/` | Error categorizer and Table 4.3 category labels | 4.4 |
| `evaluation/hindi_baseline/` | Hindi translation, scoring, signal comparison, error categories | 3.5, 4.2, 4.4 |
| `evaluation/signals/` | Nepali and Tagalog translation and candidate signals (log-probability, CometKiwi, LaBSE, TransQuest) | 4.2 |
| `evaluation/data/` | Per-sentence scores used by the Chapter 4 tables (no sentence text) | 4.1.1 |

Protocol steps are numbered as in the thesis: 1 language identification,
2 translation routing, 3 trigger, 4 candidate generation, 5 selection gate.

## Running the dispatcher

```bash
pip install -r requirements.txt
mkdir -p dispatcher/models
curl -L -o dispatcher/models/lid.176.ftz \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

Igede checkpoints: download `igede-en-ige.zip` and `igede-ige-en.zip` from the
[igede-mt-plugin releases](https://github.com/auevgenii-hub/igede-mt-plugin/releases)
and unpack them into `models/igede-en-ige/` and `models/igede-ige-en/`, or set
`IGEDE_FORWARD_CKPT` and `IGEDE_REVERSE_CKPT` to their folders. The other
models (NLLB-200-distilled-600M, OPUS-MT en-tl and tl-en) are downloaded from
Hugging Face on first use.

```bash
python -m dispatcher.app          # http://127.0.0.1:5050
curl -X POST http://127.0.0.1:5050/translate -H 'Content-Type: application/json' \
  -d '{"text": "Good morning, friend.", "source_lang": "eng", "target_lang": "hin"}'
pytest tests/
```

The response metadata contains the pivot text and, for a non-English target,
the trigger decision (`metric`, `value`, `theta`, `fired`).

## Reproducing the Chapter 4 results

The tables need only the score files in `evaluation/data/`:

```bash
pip install -r evaluation/requirements.txt
python evaluation/chapter4/ch4_tables.py      # Tables 4.1 and 4.2
python evaluation/chapter4/ch4_roc_figure.py  # Figure 4.1
```

`evaluation/error_causes/category_shares.csv` gives the Table 4.3 shares,
computed from the per-sentence labels in `flagged_categories.csv`.

The score files were produced by the scripts in `evaluation/hindi_baseline/`
and `evaluation/signals/`. To regenerate them, download the FLORES-200 release
(https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz), unpack it and
set `FLORES200_DIR`. `sentence_id` values have the form `flores_dev_<i>`,
where `<i>` is the line number in the FLORES-200 dev files. Intermediate files
that contain sentence text are written to `evaluation/work/` and are not
committed. The per-sentence scores for the Igede validation split
(`ige_val_scores.csv`) were produced with the same log-probability, chrF++,
COMET and CometKiwi settings on the forward Igede plugin; the Igede verse text
is not redistributed (see the igede-mt-plugin README).

Data used: FLORES-200 dev (CC BY-SA 4.0) for Hindi, Nepali and Tagalog, and
the validation split of the English-Igede biblical corpus for Igede. The
FLORES-200 devtest split was not used for any reported number.

## Licence

Code: MIT (see `LICENSE`). Model checkpoints and datasets keep their own
licences.
