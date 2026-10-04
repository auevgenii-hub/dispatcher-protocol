#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English <-> Tagalog plugins on the bilingual OPUS-MT models
Helsinki-NLP/opus-mt-en-tl and Helsinki-NLP/opus-mt-tl-en (no fine-tuning).
TlToEnPlugin is used as the first hop when Tagalog is the source language.
"""
from plugins.opus_mt_plugin import OpusMTPlugin

CHECKPOINT_EN_TL = "Helsinki-NLP/opus-mt-en-tl"
CHECKPOINT_TL_EN = "Helsinki-NLP/opus-mt-tl-en"


class TlPlugin(OpusMTPlugin):
    def __init__(self, **kwargs):
        super().__init__(
            checkpoint=CHECKPOINT_EN_TL, source_lang="eng", target_lang="tgl", **kwargs
        )


class TlToEnPlugin(OpusMTPlugin):
    def __init__(self, **kwargs):
        super().__init__(
            checkpoint=CHECKPOINT_TL_EN, source_lang="tgl", target_lang="eng", **kwargs
        )
