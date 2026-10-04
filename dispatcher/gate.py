#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5 of the dispatcher control protocol (thesis section 3.4.5): the
selection gate. Compares the candidates produced by Step 4 using a
plugin-declared gate_metric and returns the winner.

No plugin in this repository declares a gate_metric (gate_metric_name is
None for all four), and Step 4 always returns exactly one candidate. The
function below is therefore a pass-through, kept as its own step so that a
real gate_metric comparison can be added later without changing
dispatcher/app.py. If several candidates tie, the plugin's original output
wins (thesis section 3.4.5).
"""
from typing import List


def select_candidate(candidates: List[str]) -> str:
    """Step 5. With exactly one candidate (the only case Step 4 currently
    produces), that candidate wins. Raises if given zero candidates, which
    would mean Step 4 was called incorrectly."""
    if not candidates:
        raise ValueError("select_candidate: no candidates to choose from")
    return candidates[0]
