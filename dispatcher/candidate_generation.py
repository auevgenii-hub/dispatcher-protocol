#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 of the dispatcher control protocol (thesis section 3.4.4): candidate
generation. A request flagged by the trigger (Step 3) is passed here to
produce alternatives to the plugin's own output, for the selection gate
(Step 5) to choose between.

No generator is registered. This follows from the establishment order in
thesis section 3.1.1: an intervention is designed only after the causes of
error in the flagged population have been measured, and no single cause was
large enough to justify one (thesis section 4.4).

generate_candidates() is therefore the placeholder the thesis describes.
The candidate set always contains the plugin's original output, so returning
the unchanged translation is a valid outcome of the procedure.
"""
from typing import List


def generate_candidates(translated_text: str, target_lang: str) -> List[str]:
    """Step 4. Returns the plugin's own translation as the sole
    candidate. `target_lang` is accepted (not used) so a future
    language-specific generator can be substituted here without changing
    the caller."""
    return [translated_text]
