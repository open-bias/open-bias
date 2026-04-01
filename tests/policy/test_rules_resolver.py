from pathlib import Path

import pytest

from openbias.policy.rules.resolver import resolve_rules_payload


def test_resolve_inline_rules_list():
    resolved = resolve_rules_payload({"rules": ["Rule A", "Rule B"]}, auto_discover_rules_md=False)
    assert resolved == ["Rule A", "Rule B"]


def test_resolve_inline_rules_markdown_segments():
    resolved = resolve_rules_payload(
        {"rules": "- No PII\n- Be helpful\n\n# Quality\nAlways be clear"},
        auto_discover_rules_md=False,
    )
    assert resolved == ["No PII", "Be helpful", "Quality", "Always be clear"]


def test_resolve_rules_file(tmp_path: Path):
    rules_path = tmp_path / "rules.md"
    rules_path.write_text("# Safety\n- Do not leak secrets\n- Stay polite\n")
    resolved = resolve_rules_payload(
        {"rules_file": str(rules_path)},
        auto_discover_rules_md=False,
    )
    assert resolved == ["Safety", "Do not leak secrets", "Stay polite"]


def test_auto_discover_rules_md(tmp_path: Path):
    (tmp_path / "rules.md").write_text("Rule one\n\nRule two")
    resolved = resolve_rules_payload({}, base_dir=tmp_path)
    assert resolved == ["Rule one", "Rule two"]


def test_reject_invalid_rules_file_extension(tmp_path: Path):
    bad_path = tmp_path / "rules.yaml"
    bad_path.write_text("not supported")
    with pytest.raises(ValueError, match=r"\.md or \.txt"):
        resolve_rules_payload({"rules_file": str(bad_path)}, auto_discover_rules_md=False)


def test_reject_non_string_list_item():
    with pytest.raises(ValueError, match="must be a string"):
        resolve_rules_payload({"rules": ["ok", 1]}, auto_discover_rules_md=False)


def test_resolve_inline_rules_single_string():
    resolved = resolve_rules_payload(
        {"rules": "Always be professional"},
        auto_discover_rules_md=False,
    )
    assert resolved == ["Always be professional"]


def test_semicolon_separated_rules():
    resolved = resolve_rules_payload(
        {"rules": "No PII; Be polite; Stay on topic"},
        auto_discover_rules_md=False,
    )
    assert resolved == ["No PII", "Be polite", "Stay on topic"]


def test_deduplication():
    resolved = resolve_rules_payload(
        {"rules": ["No PII", "Be polite", "No PII"]},
        auto_discover_rules_md=False,
    )
    assert resolved == ["No PII", "Be polite"]


def test_empty_config_no_rules():
    resolved = resolve_rules_payload({}, auto_discover_rules_md=False)
    assert resolved == []


def test_rules_file_txt(tmp_path: Path):
    rules_path = tmp_path / "rules.txt"
    rules_path.write_text("No secrets\n\nStay helpful\n")
    resolved = resolve_rules_payload(
        {"rules_file": str(rules_path)},
        auto_discover_rules_md=False,
    )
    assert resolved == ["No secrets", "Stay helpful"]


def test_mixed_inline_and_file(tmp_path: Path):
    rules_path = tmp_path / "extra.md"
    rules_path.write_text("- File rule one\n- File rule two\n")
    resolved = resolve_rules_payload(
        {"rules": ["Inline rule"], "rules_file": str(rules_path)},
        auto_discover_rules_md=False,
    )
    assert resolved == ["Inline rule", "File rule one", "File rule two"]


def test_mixed_markdown_formats():
    text = (
        "# Safety Rules\n"
        "- Never share PII\n"
        "- No offensive language\n"
        "\n"
        "## Tone Guidelines\n"
        "Be professional and courteous at all times.\n"
        "\n"
        "1. Respond concisely\n"
        "2. Use clear language\n"
    )
    resolved = resolve_rules_payload({"rules": text}, auto_discover_rules_md=False)
    assert resolved == [
        "Safety Rules",
        "Never share PII",
        "No offensive language",
        "Tone Guidelines",
        "Be professional and courteous at all times.",
        "Respond concisely",
        "Use clear language",
    ]


def test_rules_file_not_found(tmp_path: Path):
    with pytest.raises(ValueError, match="not found"):
        resolve_rules_payload(
            {"rules_file": str(tmp_path / "missing.md")},
            auto_discover_rules_md=False,
        )


def test_rules_file_must_be_string():
    with pytest.raises(ValueError, match="must be a string path"):
        resolve_rules_payload({"rules_file": 42}, auto_discover_rules_md=False)


def test_rules_must_be_string_or_list():
    with pytest.raises(ValueError, match="must be a string or list"):
        resolve_rules_payload({"rules": 42}, auto_discover_rules_md=False)


def test_auto_discover_skipped_when_rules_file_set(tmp_path: Path):
    auto_file = tmp_path / "rules.md"
    auto_file.write_text("Auto rule")
    explicit_file = tmp_path / "custom.md"
    explicit_file.write_text("Explicit rule")
    resolved = resolve_rules_payload(
        {"rules_file": str(explicit_file)},
        base_dir=tmp_path,
    )
    assert resolved == ["Explicit rule"]
    assert "Auto rule" not in resolved
