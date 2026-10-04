#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heuristic error categorizer used for thesis section 4.4 (Table 4.3).

categorize(lang, source_text, output_text, reference_text) returns one
category; the checks run in this fixed order and the first match wins, so
the categories are mutually exclusive:

  a. transliteration_copy_through (Hindi and Nepali only, both written in
     Devanagari): a token of two or more Latin letters appears in the output
     but nowhere in the reference.
  b. digit_number: a digit run from the source is missing from the output,
     after Devanagari digits in the output are mapped back to ASCII.
  c. omission: the output has fewer than 70% of the reference's
     whitespace-separated tokens. The threshold is fixed, not tuned per
     language.
  d. repetition: the output repeats a unigram or bigram immediately, and
     the same repetition does not occur in the reference.
  e. mistranslation_other: none of the above. This residual category is not
     subdivided further.

Categories a-d are the structural categories; their combined share is the
structural share reported in the thesis.
"""
import re

DEVANAGARI_SCRIPT_LANGS = {"hin", "npi"}  # category (a) applies only here
OMISSION_RATIO_THRESHOLD = 0.70

LATIN_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")
DIGIT_RUN_RE = re.compile(r"\d+")
WORD_RE = re.compile(r"\S+")

_DEVANAGARI_DIGIT_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")


def _normalize_digits(text):
    return text.translate(_DEVANAGARI_DIGIT_TO_ASCII)


def _category_transliteration(lang, output_text, reference_text):
    if lang not in DEVANAGARI_SCRIPT_LANGS:
        return False
    ref_lower = reference_text.lower()
    for m in LATIN_TOKEN_RE.finditer(output_text):
        token = m.group(0).lower()
        if token not in ref_lower:
            return True
    return False


def _category_digit(source_text, output_text):
    digit_runs = DIGIT_RUN_RE.findall(source_text)
    if not digit_runs:
        return False
    normalized_output = _normalize_digits(output_text)
    for run in digit_runs:
        if run not in normalized_output:
            return True
    return False


def _category_omission(output_text, reference_text):
    ref_len = len(WORD_RE.findall(reference_text))
    out_len = len(WORD_RE.findall(output_text))
    if ref_len == 0:
        return False
    return out_len < OMISSION_RATIO_THRESHOLD * ref_len


def _repeated_ngrams(tokens, n):
    found = set()
    for i in range(len(tokens) - n):
        a = tuple(t.lower() for t in tokens[i:i + n])
        b = tuple(t.lower() for t in tokens[i + n:i + 2 * n])
        if a == b:
            found.add(a)
    return found


def _category_repetition(output_text, reference_text):
    out_tokens = WORD_RE.findall(output_text)
    ref_tokens = WORD_RE.findall(reference_text)
    out_repeats = _repeated_ngrams(out_tokens, 1) | _repeated_ngrams(out_tokens, 2)
    if not out_repeats:
        return False
    ref_repeats = _repeated_ngrams(ref_tokens, 1) | _repeated_ngrams(ref_tokens, 2)
    return bool(out_repeats - ref_repeats)


def categorize(lang, source_text, output_text, reference_text):
    if _category_transliteration(lang, output_text, reference_text):
        return "transliteration_copy_through"
    if _category_digit(source_text, output_text):
        return "digit_number"
    if _category_omission(output_text, reference_text):
        return "omission"
    if _category_repetition(output_text, reference_text):
        return "repetition"
    return "mistranslation_other"
