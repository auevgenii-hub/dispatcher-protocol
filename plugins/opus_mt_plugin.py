#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic TranslationPlugin implementation over Helsinki-NLP OPUS-MT
checkpoints. Language plugins (plugin_tl.py, and the Igede reverse plugin in
plugin_igede.py) are thin wrappers that only set the checkpoint and the
language pair.

Empty input: translate() raises ValueError. An empty string has no
meaningful translation, and a silent empty answer would hide errors further
up the dispatcher pipeline.

trigger_metric (thesis sections 3.4.2 and 3.4.3): every translate() call
passes output_scores=True and return_dict_in_generate=True to generate().
Beam search then returns sequences_scores, the length-normalized
log-probability of the selected sequence (length_penalty is left at its
default of 1.0, so the score is divided by the output length). This is the
logprob_norm signal used by dispatcher/trigger.py (thesis Equation 3.5).
The two extra arguments do not change the translation itself.
"""
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from plugins.plugin_base import TranslationRequest, TranslationResponse


class OpusMTPlugin:
    trigger_metric_name = "logprob_norm"
    gate_metric_name = None

    def __init__(
        self,
        checkpoint: str,
        source_lang: str,
        target_lang: str,
        beam_size: int = 5,
        max_length: int = 160,
        device: Optional[str] = None,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
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
                f"OpusMTPlugin.translate: empty text in request {request.request_id!r}"
            )
        if request.source_lang != self.source_lang or request.target_lang != self.target_lang:
            raise ValueError(
                f"OpusMTPlugin({self.source_lang}->{self.target_lang}) got request for "
                f"{request.source_lang}->{request.target_lang} (request_id={request.request_id!r})"
            )

        inputs = self.tokenizer(
            request.text, return_tensors="pt", truncation=True, max_length=self.max_length
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
