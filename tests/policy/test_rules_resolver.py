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
