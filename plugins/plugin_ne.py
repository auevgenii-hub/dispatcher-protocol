#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English <-> Nepali plugins on facebook/nllb-200-distilled-600M (no
fine-tuning). Helsinki-NLP has no direct English-Nepali OPUS-MT model.
NePlugin translates English -> Nepali; NeToEnPlugin translates Nepali ->
English and is used as the first hop when Nepali is the source language.
"""
from plugins.nllb_plugin import NLLBPlugin

CHECKPOINT = "facebook/nllb-200-distilled-600M"


class NePlugin(NLLBPlugin):
    def __init__(self, **kwargs):
        super().__init__(
            checkpoint=CHECKPOINT, source_lang="eng", target_lang="npi", **kwargs
        )


class NeToEnPlugin(NLLBPlugin):
    def __init__(self, **kwargs):
        super().__init__(
            checkpoint=CHECKPOINT, source_lang="npi", target_lang="eng", **kwargs
        )
