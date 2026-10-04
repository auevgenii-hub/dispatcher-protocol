#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared paths and helpers for the evaluation scripts.

REPO      repository root
DATA      evaluation/data: per-sentence scores used by the Chapter 4 tables
WORK      local working directory for intermediate files that contain
          sentence text (translations, references). Set EVAL_WORK_DIR to
          change it; the default is evaluation/work/, which is git-ignored.
FLORES    root of the official FLORES-200 release (flores200_dataset/),
          downloaded from https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
          and pointed to with FLORES200_DIR.

sentence_id values have the form "flores_dev_<i>", where <i> is the
zero-based line number in the FLORES-200 dev files.
"""
import os
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "evaluation" / "data"
WORK = Path(os.environ.get("EVAL_WORK_DIR", REPO / "evaluation" / "work"))
WORK.mkdir(parents=True, exist_ok=True)
FLORES = Path(os.environ.get("FLORES200_DIR", REPO / "flores200_dataset"))

FLORES_CODES = {"eng": "eng_Latn", "hin": "hin_Deva", "npi": "npi_Deva", "tgl": "tgl_Latn"}


def load_flores_dev(lang: str) -> pd.DataFrame:
    """English source and target-language reference for the 997 FLORES-200
    dev sentences, with sentence_id = "flores_dev_<line number>"."""
    src = (FLORES / "dev" / "eng_Latn.dev").read_text(encoding="utf-8").splitlines()
    ref = (FLORES / "dev" / f"{FLORES_CODES[lang]}.dev").read_text(encoding="utf-8").splitlines()
    assert len(src) == len(ref) == 997, "unexpected FLORES-200 dev size"
    return pd.DataFrame({
        "sentence_id": [f"flores_dev_{i}" for i in range(len(src))],
        "source_text": src,
        "reference_text": ref,
    })


class no_mps_fork_for_comet:
    """comet 2.2.x passes multiprocessing_context="fork" to its DataLoader
    whenever torch.backends.mps.is_available() is True. On macOS this is
    unreliable, so MPS is reported as unavailable for the duration of
    predict() only, which makes comet run single-process."""

    def __enter__(self):
        self._orig = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False
        return self

    def __exit__(self, *exc_info):
        torch.backends.mps.is_available = self._orig
        return False


def load_comet_model(model_name: str = "Unbabel/wmt22-comet-da"):
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(model_name))
    return model, model_name


def sanity_check_comet(comet_model):
    """A clearly good translation must score well above an unrelated one."""
    data = [
        {"src": "The cat sat on the mat.", "mt": "Le chat était assis sur le tapis.",
         "ref": "Le chat était assis sur le tapis."},
        {"src": "The cat sat on the mat.", "mt": "Quantum entanglement violates classical locality assumptions.",
         "ref": "Le chat était assis sur le tapis."},
    ]
    with no_mps_fork_for_comet():
        out = comet_model.predict(data, batch_size=2, gpus=0, num_workers=0)
    good, bad = out.scores[0], out.scores[1]
    assert good > bad + 0.1, f"COMET sanity check failed: good={good:.4f}, bad={bad:.4f}"
    return good, bad
