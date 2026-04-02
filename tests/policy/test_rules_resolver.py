from pathlib import Path

from openbias.policy.rules.resolver import resolve_rules_payload


def test_auto_discover_rules_md(tmp_path: Path):
    (tmp_path / "rules.md").write_text("Rule one\n\nRule two")
    resolved = resolve_rules_payload({}, base_dir=tmp_path)
    assert resolved == ["Rule one", "Rule two"]


def test_rules_md_segments_markdown_and_deduplicates(tmp_path: Path):
    (tmp_path / "rules.md").write_text(
        "# Safety Rules\n"
        "- Never share PII\n"
        "- No offensive language\n"
        "\n"
        "## Tone Guidelines\n"
        "Be professional and courteous at all times.\n"
        "\n"
        "1. Respond concisely\n"
        "2. Use clear language\n"
        "\n"
        "Never share PII\n",
        encoding="utf-8",
    )
    resolved = resolve_rules_payload({}, base_dir=tmp_path)
    assert resolved == [
        "Safety Rules",
        "Never share PII",
        "No offensive language",
        "Tone Guidelines",
        "Be professional and courteous at all times.",
        "Respond concisely",
        "Use clear language",
    ]


def test_legacy_rules_inputs_are_ignored_in_favor_of_project_rules_md(tmp_path: Path):
    (tmp_path / "rules.md").write_text("Project rule", encoding="utf-8")
    resolved = resolve_rules_payload(
        {"rules": ["Inline rule"], "rules_file": str(tmp_path / "custom.md")},
        base_dir=tmp_path,
    )
    assert resolved == ["Project rule"]


def test_auto_discover_can_be_disabled_even_with_legacy_keys_present(tmp_path: Path):
    (tmp_path / "rules.md").write_text("Project rule", encoding="utf-8")
    resolved = resolve_rules_payload(
        {"rules": ["Inline rule"], "rules_file": str(tmp_path / "custom.md")},
        base_dir=tmp_path,
        auto_discover_rules_md=False,
    )
    assert resolved == []
