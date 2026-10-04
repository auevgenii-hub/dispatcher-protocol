#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 of the dispatcher control protocol (thesis section 3.4.1): fastText
lid.176 language identification for the optional source-language
auto-detection path (see dispatcher/app.py).

Model: lid.176.ftz (compressed, about 1 MB). It is not included in this
repository. Download it from
https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
into dispatcher/models/lid.176.ftz.

The label set was checked against the loaded model (176 labels):
__label__hi (Hindi), __label__tl (Tagalog) and __label__ne (Nepali) are
present, and there is no Igede label. Detection can therefore resolve only
to hin, tgl or npi; Igede must always be declared explicitly.
"""
from pathlib import Path
from typing import NamedTuple, Optional

import fasttext

MODEL_PATH = Path(__file__).resolve().parent / "models" / "lid.176.ftz"

# fastText lid.176 label -> internal ISO 639-3 code, built from the labels
# returned by fasttext.load_model(...).get_labels().
FASTTEXT_TO_INTERNAL = {
    "__label__hi": "hin",
    "__label__tl": "tgl",
    "__label__ne": "npi",
}

# Below this confidence the detection is not trusted and the caller must
# ask for an explicit language code. Thesis section 3.4.1 fixes the
# threshold at 0.7: fastText returns a probability per language, and 0.5
# would only mean that the top label is more likely than the rest.
CONFIDENCE_THRESHOLD = 0.7

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = fasttext.load_model(str(MODEL_PATH))
    return _model


class DetectionResult(NamedTuple):
    fasttext_label: str          # raw fastText label, e.g. "__label__hi"
    internal_lang: Optional[str]  # mapped internal code (hin/tgl/npi), or None if unmapped
    confidence: float


def detect(text: str) -> DetectionResult:
    """Run fastText lid.176 on text and return its top prediction."""
    model = _get_model()
    # fastText chokes on embedded newlines; a single line of input is all
    # this dispatcher ever sends it anyway.
    clean_text = " ".join(text.split())
    labels, probs = model.predict(clean_text, k=1)
    label = labels[0]
    confidence = float(probs[0])
    internal_lang = FASTTEXT_TO_INTERNAL.get(label)
    return DetectionResult(fasttext_label=label, internal_lang=internal_lang, confidence=confidence)
