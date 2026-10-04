#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English <-> Igede plugins (thesis section 4.6).

IgedePlugin (eng -> ige) loads the checkpoint fine-tuned from
Helsinki-NLP/opus-mt-en-mul on the English-Igede biblical corpus
(29,454 verses, test chrF++ 40.29). IgedeToEnPlugin (ige -> eng) loads the
checkpoint fine-tuned from Helsinki-NLP/opus-mt-mul-en on the same corpus
and split (test chrF++ 37.49). Training code, corpus split and model card
are in the companion repository
https://github.com/auevgenii-hub/igede-mt-plugin, and both checkpoints are
attached to its v1.0 release.

Where the checkpoints are looked up: the environment variables
IGEDE_FORWARD_CKPT and IGEDE_REVERSE_CKPT, or by default
models/igede-en-ige/ and models/igede-ige-en/ at the repository root.

opus-mt-en-mul selects the target language with a prefix token. There is no
token for Igede, so the >>yor<< token was reassigned to Igede during
fine-tuning. The choice of this token is arbitrary and makes no linguistic
claim (thesis section 4.6.4). The prefix is a donor-specific routing detail
and never leaves this plugin. opus-mt-mul-en has no prefix tokens, so the
reverse plugin is a thin OpusMTPlugin wrapper.

Empty input raises ValueError, as in every other plugin. trigger_metric is
computed exactly as in OpusMTPlugin.
"""
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from plugins.opus_mt_plugin import OpusMTPlugin
from plugins.plugin_base import TranslationRequest, TranslationResponse

_MODELS = Path(__file__).resolve().parent.parent / "models"
CHECKPOINT = os.environ.get("IGEDE_FORWARD_CKPT", str(_MODELS / "igede-en-ige"))
CHECKPOINT_IGE_EN = os.environ.get("IGEDE_REVERSE_CKPT", str(_MODELS / "igede-ige-en"))
DONOR_LANG_PREFIX = ">>yor<<"  # donor token reassigned to Igede, see the docstring above


class IgedePlugin:
    source_lang = "eng"
    target_lang = "ige"  # ISO 639-3 code for Igede
    trigger_metric_name = "logprob_norm"
    gate_metric_name = None

    def __init__(
        self,
        checkpoint: str = CHECKPOINT,
        beam_size: int = 5,
        max_length: int = 160,
        device: Optional[str] = None,
    ):
        self.checkpoint = checkpoint
        self.beam_size = beam_size
        self.max_length = max_length
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def translate(self, request: TranslationRequest) -> TranslationResponse:
        if not request.text.strip():
            raise ValueError(
                f"IgedePlugin.translate: empty text in request {request.request_id!r}"
            )
        if request.source_lang != self.source_lang or request.target_lang != self.target_lang:
            raise ValueError(
                f"IgedePlugin({self.source_lang}->{self.target_lang}) got request for "
                f"{request.source_lang}->{request.target_lang} (request_id={request.request_id!r})"
            )

        prefixed_text = f"{DONOR_LANG_PREFIX} {request.text}"
        inputs = self.tokenizer(
            prefixed_text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=self.beam_size,
                max_length=self.max_length,
                output_scores=True,
                return_dict_in_generate=True,
            )
        output_ids = outputs.sequences
        translated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        trigger_metric = (
            outputs.sequences_scores[0].item() if outputs.sequences_scores is not None else None
        )

        return TranslationResponse(
            request_id=request.request_id,
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            metadata={
                "checkpoint": self.checkpoint,
                "trigger_metric_name": self.trigger_metric_name,
                "trigger_metric": trigger_metric,
            },
        )


class IgedeToEnPlugin(OpusMTPlugin):
    """Igede -> English plugin (fine-tuned opus-mt-mul-en, no prefix token)."""


    def __init__(self, **kwargs):
        super().__init__(
            checkpoint=CHECKPOINT_IGE_EN, source_lang="ige", target_lang="eng", **kwargs
        )
