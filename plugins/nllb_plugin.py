#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic TranslationPlugin implementation over NLLB-200
(facebook/nllb-200-distilled-600M), used for Hindi and Nepali (thesis
section 3.5). Helsinki-NLP has no direct English-Nepali OPUS-MT model. For
Hindi a bilingual OPUS-MT model exists, but NLLB-200 was clearly better on
the 997 FLORES-200 development sentences.

The contract's plain ISO 639-3 codes ("eng", "npi", "hin") are mapped to
NLLB's FLORES-200 codes ("eng_Latn", "npi_Deva", "hin_Deva") inside this
plugin only; the mapping is never exposed through TranslationRequest or
TranslationResponse.

With recent transformers versions the tokenizer no longer exposes
lang_code_to_id, so the target language is set with
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(<target code>)

Empty input raises ValueError, as in every other plugin.

trigger_metric: same mechanism as OpusMTPlugin. output_scores=True and
return_dict_in_generate=True are passed to generate(), and
sequences_scores[0] is logprob_norm.
"""
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from plugins.plugin_base import TranslationRequest, TranslationResponse

# ISO 639-3 (contract) -> FLORES-200 code (NLLB). Extend as needed.
ISO3_TO_FLORES = {
    "eng": "eng_Latn",
    "npi": "npi_Deva",
    "hin": "hin_Deva",
}


class NLLBPlugin:
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
        if source_lang not in ISO3_TO_FLORES:
            raise ValueError(f"NLLBPlugin: unknown source_lang {source_lang!r}, add it to ISO3_TO_FLORES")
        if target_lang not in ISO3_TO_FLORES:
            raise ValueError(f"NLLBPlugin: unknown target_lang {target_lang!r}, add it to ISO3_TO_FLORES")

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.checkpoint = checkpoint
        self.beam_size = beam_size
        self.max_length = max_length
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        self._src_flores = ISO3_TO_FLORES[source_lang]
        self._tgt_flores = ISO3_TO_FLORES[target_lang]

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang=self._src_flores)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self._forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self._tgt_flores)

    def translate(self, request: TranslationRequest) -> TranslationResponse:
        if not request.text.strip():
            raise ValueError(
                f"NLLBPlugin.translate: empty text in request {request.request_id!r}"
            )
        if request.source_lang != self.source_lang or request.target_lang != self.target_lang:
            raise ValueError(
                f"NLLBPlugin({self.source_lang}->{self.target_lang}) got request for "
                f"{request.source_lang}->{request.target_lang} (request_id={request.request_id!r})"
            )

        inputs = self.tokenizer(
            request.text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self._forced_bos_token_id,
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
