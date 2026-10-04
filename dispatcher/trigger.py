#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 of the dispatcher control protocol (thesis section 3.4.3): decide
whether a translation needs further processing. The trigger fires after the
plugin has returned a translation and uses the trigger_metric value carried
in that translation's response.

    triggered(x) = 1 if m(x) <= Q_q(M_L) else 0          (thesis Equation 3.6)

m(x) is the request's trigger_metric value and Q_q(M_L) is the q-th quantile
of that metric's distribution on the development set for language L. Every
plugin in this repository declares trigger_metric_name = "logprob_norm", the
length-normalized sequence log-probability produced as a byproduct of
decoding (see plugins/opus_mt_plugin.py and plugins/nllb_plugin.py).

THETA_BY_LANG holds q = 0.25 (the lower quartile) of logprob_norm on each
language's development set:

    hin  FLORES-200 dev, n = 997
    npi  FLORES-200 dev, n = 997
    tgl  FLORES-200 dev, n = 997
    ige  validation split of the English-Igede biblical corpus, n = 1,473

The per-sentence values these quartiles were computed from are in
evaluation/data/.
"""
from typing import Optional

from plugins.plugin_base import TranslationResponse

THETA_BY_LANG = {
    "hin": -0.5006408095359802,
    "npi": -0.6396948099136353,
    "tgl": -0.402812659740448,
    "ige": -0.2085439418924266,
}


def evaluate_trigger(response: TranslationResponse, target_lang: str) -> dict:
    """Step 3. Always returns a dict describing what happened. It never
    raises for a plugin that does not offer the signal or for a language
    without a calibrated threshold: section 3.4.3 treats both as a normal,
    logged state in which the request is returned without further
    processing."""
    metadata = response.metadata or {}
    metric_name = metadata.get("trigger_metric_name")
    metric_value = metadata.get("trigger_metric")

    if metric_name is None or metric_value is None:
        return {
            "ran": False,
            "fired": False,
            "reason": "plugin did not declare trigger_metric",
        }

    theta: Optional[float] = THETA_BY_LANG.get(target_lang)
    if theta is None:
        return {
            "ran": False,
            "fired": False,
            "metric": metric_name,
            "value": metric_value,
            "reason": f"no calibrated threshold for target_lang={target_lang!r}",
        }

    fired = metric_value <= theta
    return {
        "ran": True,
        "fired": fired,
        "metric": metric_name,
        "value": metric_value,
        "theta": theta,
    }
