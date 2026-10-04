#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest suite for the TranslationPlugin contract (thesis section 3.3):
Hindi and Nepali (NLLB-200), Tagalog (OPUS-MT) and Igede (fine-tuned
opus-mt-en-mul / opus-mt-mul-en).

Models are large, so every plugin is loaded once per module through a
fixture with scope="module". The Igede tests need the two checkpoints from
the igede-mt-plugin v1.0 release (see README.md, "Igede checkpoints").

HiToEnPlugin, TlToEnPlugin and NeToEnPlugin are not tested separately here;
IgedeToEnPlugin is covered explicitly at the end of the file.
"""
import pytest

from plugins.plugin_base import TranslationRequest
from plugins.plugin_hi import HiPlugin
from plugins.plugin_igede import IgedePlugin, IgedeToEnPlugin
from plugins.plugin_ne import NePlugin
from plugins.plugin_tl import TlPlugin

# Smoke-test sentence from FLORES-200 (CC BY-SA 4.0). It is used only to
# check that a plugin really translates; no reported number depends on it.
FLORES_SRC = "He built a WiFi door bell, he said."

PLUGIN_CASES = [
    pytest.param(
        "hi",
        HiPlugin,
        "hin",
        FLORES_SRC,
        # "कहा" = "said": a frequent verb, robust to how "WiFi" is transliterated.
        ("कहा",),
        id="hi",
    ),
    pytest.param(
        "tl",
        TlPlugin,
        "tgl",
        FLORES_SRC,
        ("wifi", "sabi"),  # "sabi niya" = "he said"
        id="tl",
    ),
    pytest.param(
        "ne",
        NePlugin,
        "npi",
        FLORES_SRC,
        # "भने" = "said", same principle as for Hindi.
        ("भने",),
        id="ne",
    ),
    pytest.param(
        "igede",
        IgedePlugin,
        "ige",
        FLORES_SRC,
        # Igede has no FLORES reference, so no substring is required; the
        # sanity test below only checks that the output is not an echo.
        (),
        id="igede",
    ),
]


@pytest.fixture(scope="module")
def hi_plugin():
    return HiPlugin()


@pytest.fixture(scope="module")
def tl_plugin():
    return TlPlugin()


@pytest.fixture(scope="module")
def ne_plugin():
    return NePlugin()


@pytest.fixture(scope="module")
def igede_plugin():
    return IgedePlugin()


@pytest.fixture(scope="module")
def plugin_by_lang(hi_plugin, tl_plugin, ne_plugin, igede_plugin):
    return {"hi": hi_plugin, "tl": tl_plugin, "ne": ne_plugin, "igede": igede_plugin}


@pytest.mark.parametrize("lang,plugin_cls,target_lang,src,expected_substrings", PLUGIN_CASES)
def test_translate_preserves_request_id_and_langs(
    plugin_by_lang, lang, plugin_cls, target_lang, src, expected_substrings
):
    plugin = plugin_by_lang[lang]
    request = TranslationRequest(
        text="Good morning, friend.",
        source_lang="eng",
        target_lang=target_lang,
        request_id=f"req-{lang}-001",
    )
    response = plugin.translate(request)

    assert response.request_id == f"req-{lang}-001"
    assert response.source_lang == "eng"
    assert response.target_lang == target_lang


@pytest.mark.parametrize("lang,plugin_cls,target_lang,src,expected_substrings", PLUGIN_CASES)
def test_translate_returns_non_empty_text(
    plugin_by_lang, lang, plugin_cls, target_lang, src, expected_substrings
):
    plugin = plugin_by_lang[lang]
    request = TranslationRequest(
        text="The quick brown fox jumps over the lazy dog.",
        source_lang="eng",
        target_lang=target_lang,
        request_id=f"req-{lang}-002",
    )
    response = plugin.translate(request)

    assert isinstance(response.translated_text, str)
    assert response.translated_text.strip() != ""


@pytest.mark.parametrize("lang,plugin_cls,target_lang,src,expected_substrings", PLUGIN_CASES)
def test_translate_reports_trigger_metric(
    plugin_by_lang, lang, plugin_cls, target_lang, src, expected_substrings
):
    """The trigger (thesis section 3.4.3, dispatcher/trigger.py) reads
    trigger_metric_name and trigger_metric from the plugin's response
    metadata. Every plugin declares "logprob_norm" and returns a float."""
    plugin = plugin_by_lang[lang]
    request = TranslationRequest(
        text="Good morning, friend.",
        source_lang="eng",
        target_lang=target_lang,
        request_id=f"req-{lang}-002b",
    )
    response = plugin.translate(request)

    assert response.metadata["trigger_metric_name"] == "logprob_norm"
    assert isinstance(response.metadata["trigger_metric"], float)


@pytest.mark.parametrize("lang,plugin_cls,target_lang,src,expected_substrings", PLUGIN_CASES)
def test_translate_empty_input_raises_value_error(
    plugin_by_lang, lang, plugin_cls, target_lang, src, expected_substrings
):
    plugin = plugin_by_lang[lang]
    request = TranslationRequest(
        text="",
        source_lang="eng",
        target_lang=target_lang,
        request_id=f"req-{lang}-003",
    )
    with pytest.raises(ValueError):
        plugin.translate(request)

    # Whitespace-only input counts as empty too.
    request_ws = TranslationRequest(
        text="   ",
        source_lang="eng",
        target_lang=target_lang,
        request_id=f"req-{lang}-004",
    )
    with pytest.raises(ValueError):
        plugin.translate(request_ws)


@pytest.mark.parametrize("lang,plugin_cls,target_lang,src,expected_substrings", PLUGIN_CASES)
def test_translate_sanity_check_against_flores(
    plugin_by_lang, lang, plugin_cls, target_lang, src, expected_substrings
):
    """The plugin must translate rather than echo its input. For hin, tgl and
    npi the output must also contain a word from the FLORES-200 reference."""
    plugin = plugin_by_lang[lang]
    request = TranslationRequest(
        text=src,
        source_lang="eng",
        target_lang=target_lang,
        request_id=f"req-{lang}-005",
    )
    response = plugin.translate(request)
    hyp = response.translated_text.strip().lower()

    assert hyp != src.strip().lower()
    if expected_substrings:
        assert any(sub in hyp for sub in expected_substrings), (
            f"{lang}: expected one of {expected_substrings} in hypothesis {hyp!r}"
        )


# --- IgedeToEnPlugin (ige -> eng) -------------------------------------------
# Covered by the same checks as above, but not through PLUGIN_CASES, whose
# tests hard-code source_lang="eng". The Igede input is produced by the
# forward plugin, so no corpus text is needed here.


@pytest.fixture(scope="module")
def igede_input(igede_plugin):
    request = TranslationRequest(
        text="Good morning, friend.",
        source_lang="eng",
        target_lang="ige",
        request_id="req-igede-input",
    )
    return igede_plugin.translate(request).translated_text


@pytest.fixture(scope="module")
def igede_to_en_plugin():
    return IgedeToEnPlugin()


def test_igede_to_en_translate_preserves_request_id_and_langs(igede_to_en_plugin, igede_input):
    request = TranslationRequest(
        text=igede_input,
        source_lang="ige",
        target_lang="eng",
        request_id="req-igede_to_en-001",
    )
    response = igede_to_en_plugin.translate(request)

    assert response.request_id == "req-igede_to_en-001"
    assert response.source_lang == "ige"
    assert response.target_lang == "eng"


def test_igede_to_en_translate_returns_non_empty_text(igede_to_en_plugin, igede_input):
    request = TranslationRequest(
        text=igede_input,
        source_lang="ige",
        target_lang="eng",
        request_id="req-igede_to_en-002",
    )
    response = igede_to_en_plugin.translate(request)

    assert isinstance(response.translated_text, str)
    assert response.translated_text.strip() != ""


def test_igede_to_en_translate_reports_trigger_metric(igede_to_en_plugin, igede_input):
    request = TranslationRequest(
        text=igede_input,
        source_lang="ige",
        target_lang="eng",
        request_id="req-igede_to_en-001b",
    )
    response = igede_to_en_plugin.translate(request)

    assert response.metadata["trigger_metric_name"] == "logprob_norm"
    assert isinstance(response.metadata["trigger_metric"], float)


def test_igede_to_en_translate_empty_input_raises_value_error(igede_to_en_plugin):
    request = TranslationRequest(
        text="",
        source_lang="ige",
        target_lang="eng",
        request_id="req-igede_to_en-003",
    )
    with pytest.raises(ValueError):
        igede_to_en_plugin.translate(request)

    request_ws = TranslationRequest(
        text="   ",
        source_lang="ige",
        target_lang="eng",
        request_id="req-igede_to_en-004",
    )
    with pytest.raises(ValueError):
        igede_to_en_plugin.translate(request_ws)


def test_igede_to_en_translate_is_not_an_echo(igede_to_en_plugin, igede_input):
    """The reverse plugin must translate rather than echo its input."""
    request = TranslationRequest(
        text=igede_input,
        source_lang="ige",
        target_lang="eng",
        request_id="req-igede_to_en-005",
    )
    response = igede_to_en_plugin.translate(request)
    hyp = response.translated_text.strip().lower()

    assert hyp != igede_input.strip().lower()
