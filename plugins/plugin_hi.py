#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English <-> Hindi plugins on facebook/nllb-200-distilled-600M (no
fine-tuning). HiPlugin translates English -> Hindi; HiToEnPlugin translates
Hindi -> English and is used as the first hop when Hindi is the source
language (hin -> eng -> X, see dispatcher/app.py).
"""
from plugins.nllb_plugin import NLLBPlugin

CHECKPOINT = "facebook/nllb-200-distilled-600M"


class HiPlugin(NLLBPlugin):
    def __init__(self, **kwargs):
        super().__init__(
            checkpoint=CHECKPOINT, source_lang="eng", target_lang="hin", **kwargs
        )


class HiToEnPlugin(NLLBPlugin):
    def __init__(self, **kwargs):
        super().__init__(
            checkpoint=CHECKPOINT, source_lang="hin", target_lang="eng", **kwargs
        )
