#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common translation-plugin contract of the dispatcher (thesis section 3.3,
plugin registration protocol). Every language plugin implements
TranslationPlugin.

At registration a plugin declares its ISO 639-3 codes and the quality signals
it can provide. trigger_metric_name names the signal used by the trigger
(Step 3, thesis section 3.4.3); every plugin here declares "logprob_norm",
the length-normalized sequence log-probability produced during decoding.
gate_metric_name names the signal used by the selection gate (Step 5); it is
None here because no plugin declares one. Either field may be None: a plugin
that cannot compute a signal declares nothing, and the corresponding step is
skipped for it and logged, not treated as an error.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Protocol


@dataclass
class TranslationRequest:
    text: str
    source_lang: str       # ISO 639-3, e.g. "eng"
    target_lang: str       # ISO 639-3, e.g. "hin"
    request_id: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TranslationResponse:
    request_id: str
    translated_text: str
    source_lang: str
    target_lang: str
    metadata: Optional[Dict[str, Any]] = None


class TranslationPlugin(Protocol):
    source_lang: str
    target_lang: str
    trigger_metric_name: Optional[str]
    gate_metric_name: Optional[str]

    def translate(self, request: TranslationRequest) -> TranslationResponse: ...
