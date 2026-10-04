#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dispatcher: a Flask application implementing the dispatcher control protocol
(thesis section 3.4) for Hindi (hin), Nepali (npi), Tagalog (tgl) and Igede
(ige), with English (eng) as the single pivot language. Every language is
reached through the TranslationPlugin contract in plugins/plugin_base.py.

Protocol steps, numbered as in the thesis:

  Step 1 (3.4.1, dispatcher/lang_id.py): language identification. Explicit
    ISO 639-3 codes are preferred. Automatic detection with fastText lid.176
    is used only for Tagalog and is gated by a confidence threshold; Hindi,
    Nepali and Igede must be declared explicitly.
  Step 2 (3.4.2): translation routing, i.e. the two plugin calls
    source -> English -> target.
  Step 3 (3.4.3, dispatcher/trigger.py): trigger. Reads the plugin's declared
    trigger_metric (length-normalized sequence log-probability) and compares
    it with a per-language threshold, the lower quartile of the metric on the
    development set.
  Step 4 (3.4.4, dispatcher/candidate_generation.py): candidate generation.
    No generator is registered, so the plugin's own output is the only
    candidate.
  Step 5 (3.4.5, dispatcher/gate.py): selection gate. With a single candidate
    the gate returns it unchanged; the step is kept separate so that a real
    gate can be added without changing this file.
"""
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request

from dispatcher import lang_id
from dispatcher.candidate_generation import generate_candidates
from dispatcher.gate import select_candidate
from dispatcher.trigger import evaluate_trigger
from plugins.plugin_base import TranslationRequest
from plugins.plugin_hi import HiPlugin, HiToEnPlugin
from plugins.plugin_igede import IgedePlugin, IgedeToEnPlugin
from plugins.plugin_ne import NePlugin, NeToEnPlugin
from plugins.plugin_tl import TlPlugin, TlToEnPlugin

app = Flask(__name__)

# Single source of truth for which languages have a registered plugin pair.
# Adding a language means adding one entry here -- everything else below
# (VALID_LANGS, TARGET_PLUGIN_LANGS, FROM_EN, TO_EN) is derived from this
# dict instead of being hardcoded separately, so there is exactly one place
# to edit. See thesis section 3.3 (plugin registration protocol) for the
# design rationale, and plugins/plugin_base.py for what each plugin declares
# at registration (trigger_metric_name / gate_metric_name).
PLUGIN_REGISTRY = {
    "hin": {"from_en": HiPlugin, "to_en": HiToEnPlugin},
    "tgl": {"from_en": TlPlugin, "to_en": TlToEnPlugin},
    "npi": {"from_en": NePlugin, "to_en": NeToEnPlugin},
    "ige": {"from_en": IgedePlugin, "to_en": IgedeToEnPlugin},
}

TARGET_PLUGIN_LANGS = set(PLUGIN_REGISTRY.keys())  # excludes "eng" -- no plugin for it
VALID_LANGS = TARGET_PLUGIN_LANGS | {"eng"}
# "auto" is accepted for source_lang only -- target_lang stays manual-only.
SOURCE_VALID_LANGS = VALID_LANGS | {"auto"}

# Only Tagalog is resolved automatically (see
# dispatcher/lang_id.py -- no Igede label exists in lid.176). Hindi and
# Nepali are deliberately excluded from auto-resolution even though
# fastText can detect them: mandatory explicit selection for hin/npi/ige is
# a design requirement (Devanagari-script ambiguity), not a gap to close.
AUTO_RESOLVABLE_LANG = "tgl"

print("Loading plugins (this loads all eight models into memory)...")

# eng -> X (X != eng)
FROM_EN = {lang: cfg["from_en"]() for lang, cfg in PLUGIN_REGISTRY.items()}

# X -> eng
TO_EN = {lang: cfg["to_en"]() for lang, cfg in PLUGIN_REGISTRY.items()}

print("All plugins loaded.")
print(f"  eng -> X available for: {sorted(FROM_EN.keys())}")
print(f"  X -> eng available for: {sorted(TO_EN.keys())}")


def _translate_one(plugin, text, source_lang, target_lang, request_id):
    req = TranslationRequest(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        request_id=request_id,
    )
    return plugin.translate(req)


def _run_trigger_candidate_gate(hop2_response, target_lang):
    """Steps 3 (trigger), 4 (candidate generation) and 5 (gate), in that
    order, for a single hop-2 translation response. Steps 4 and 5 run only if
    the trigger fires (thesis section 3.4.3); an unflagged request is
    returned exactly as the plugin produced it.

    Returns (final_text, protocol_metadata) -- protocol_metadata always has
    a "trigger" key (evaluate_trigger()'s dict) and, only when the trigger
    fired, a "candidate_generation"/"gate" pair describing what those steps
    did (here: always the trivial no-op)."""
    trigger_result = evaluate_trigger(hop2_response, target_lang)
    protocol_metadata = {"trigger": trigger_result}

    if not trigger_result["fired"]:
        return hop2_response.translated_text, protocol_metadata

    candidates = generate_candidates(hop2_response.translated_text, target_lang)
    final_text = select_candidate(candidates)
    protocol_metadata["candidate_generation"] = {"n_candidates": len(candidates)}
    protocol_metadata["gate"] = {"selected": final_text}
    return final_text, protocol_metadata


@app.route("/translate", methods=["POST"])
def translate():
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    source_lang = body.get("source_lang", "")
    target_lang = body.get("target_lang", "")
    request_id = str(uuid.uuid4())

    if source_lang not in SOURCE_VALID_LANGS or target_lang not in VALID_LANGS:
        return jsonify({
            "error": f"source_lang must be one of {sorted(SOURCE_VALID_LANGS)} and target_lang "
                     f"must be one of {sorted(VALID_LANGS)} "
                     f"(got source_lang={source_lang!r}, target_lang={target_lang!r})"
        }), 400

    auto_detect_metadata = None
    if source_lang == "auto":
        detection = lang_id.detect(text)
        if detection.internal_lang == AUTO_RESOLVABLE_LANG and detection.confidence >= lang_id.CONFIDENCE_THRESHOLD:
            # Resolve and fall through to the normal path below, exactly as
            # if the user had picked Tagalog manually.
            source_lang = AUTO_RESOLVABLE_LANG
            auto_detect_metadata = {
                "auto_detected_source_lang": detection.internal_lang,
                "detection_confidence": detection.confidence,
            }
        elif detection.internal_lang in ("hin", "npi"):
            lang_name = "Hindi" if detection.internal_lang == "hin" else "Nepali"
            return jsonify({
                "error": (
                    f"detected {lang_name} ({detection.internal_lang}, confidence "
                    f"{detection.confidence:.2f}) - Hindi and Nepali require explicit "
                    f"selection due to Devanagari-script ambiguity; please pick the "
                    f"source language manually."
                ),
                "detected_label": detection.fasttext_label,
                "detected_lang": detection.internal_lang,
                "detection_confidence": detection.confidence,
            }), 422
        elif detection.internal_lang == AUTO_RESOLVABLE_LANG:
            # tgl detected but below the confidence threshold.
            return jsonify({
                "error": (
                    f"low confidence ({detection.confidence:.2f} < "
                    f"{lang_id.CONFIDENCE_THRESHOLD}) - please select the source "
                    f"language manually."
                ),
                "detected_label": detection.fasttext_label,
                "detected_lang": detection.internal_lang,
                "detection_confidence": detection.confidence,
            }), 422
        else:
            return jsonify({
                "error": (
                    "detected a language outside the four supported ones "
                    f"(fastText label={detection.fasttext_label!r}, confidence "
                    f"{detection.confidence:.2f}) - please select the source language "
                    f"manually."
                ),
                "detected_label": detection.fasttext_label,
                "detected_lang": detection.internal_lang,
                "detection_confidence": detection.confidence,
            }), 422

    # Same-language: skip translation entirely, echo input unchanged.
    if source_lang == target_lang:
        identity_metadata = {"note": "source_lang == target_lang; no translation performed, input echoed unchanged"}
        if auto_detect_metadata:
            identity_metadata.update(auto_detect_metadata)
        return jsonify({
            "request_id": request_id,
            "translated_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "metadata": identity_metadata,
        })

    try:
        # Step 2, hop 1: source -> English (skipped if the source is eng).
        if source_lang == "eng":
            pivot_text = text
        else:
            hop1 = _translate_one(TO_EN[source_lang], text, source_lang, "eng", request_id)
            pivot_text = hop1.translated_text

        # Step 2, hop 2: English -> target, followed by Steps 3-5 on that
        # result. Skipped if the target is eng, because there is no plugin
        # output to evaluate.
        protocol_metadata = None
        if target_lang == "eng":
            final_text = pivot_text
        else:
            hop2 = _translate_one(FROM_EN[target_lang], pivot_text, "eng", target_lang, request_id)
            final_text, protocol_metadata = _run_trigger_candidate_gate(hop2, target_lang)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # source_lang != target_lang here, so at least one hop always ran.
    pivot_metadata = {
        "pivot_text": pivot_text,
        "hop1": "skipped (source already eng)" if source_lang == "eng" else f"{source_lang}->eng",
        "hop2": "skipped (target already eng)" if target_lang == "eng" else f"eng->{target_lang}",
    }
    if protocol_metadata is not None:
        pivot_metadata.update(protocol_metadata)
    if auto_detect_metadata:
        pivot_metadata.update(auto_detect_metadata)
    return jsonify({
        "request_id": request_id,
        "translated_text": final_text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "metadata": pivot_metadata,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
