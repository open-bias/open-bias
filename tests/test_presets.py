"""Tests for packaged rules presets."""

from importlib.resources import files

from openbias.presets.library import (
    _extract_description,
    _extract_title,
    discover_rules_presets,
    get_rules_preset,
)


def test_discover_rules_presets_is_deterministic_and_complete():
    presets = discover_rules_presets()

    assert [preset.slug for preset in presets] == [
        "compliance/eu-ai-act-starter",
        "compliance/gdpr-privacy",
        "core/general-safety",
        "core/prompt-injection-and-secrets",
        "domain/customer-support",
        "domain/healthcare-information",
    ]
    assert all(preset.content.startswith("# ") for preset in presets)
    assert all(preset.description for preset in presets)


def test_get_rules_preset_extracts_title_description_and_content():
    preset = get_rules_preset("domain/healthcare-information")

    assert preset.title == "Healthcare Information"
    assert (
        preset.description
        == "Starter rules for informational healthcare assistants that should stay "
        "cautious, factual, and non-diagnostic."
    )
    assert "Avoid personalized dosing, medication changes" in preset.content
    assert (
        preset.package_path
        == "openbias/presets/rules/domain/healthcare-information.md"
    )


def test_extract_title_falls_back_to_filename_when_heading_missing():
    assert _extract_title("- Rule one\n- Rule two\n", "prompt-injection-and-secrets.md") == (
        "Prompt Injection And Secrets"
    )


def test_extract_description_falls_back_when_markdown_has_only_bullets():
    assert _extract_description("- Rule one\n- Rule two\n", "core/general-safety") == (
        "Balanced baseline rules for harmful content, privacy, and safe professional "
        "responses."
    )


def test_packaged_rules_resources_are_available_via_importlib_resources():
    rules_dir = files("openbias.presets").joinpath("rules")
    customer_support = rules_dir.joinpath("domain").joinpath("customer-support.md")

    assert customer_support.is_file()
    assert customer_support.read_text(encoding="utf-8").startswith("# Customer Support\n")
