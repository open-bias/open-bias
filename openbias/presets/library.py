"""Rules preset discovery for interactive ``openbias init``."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import PurePosixPath


_DESCRIPTION_FALLBACKS: dict[str, str] = {
    "compliance/eu-ai-act-starter": (
        "Starter guardrails for documenting high-level EU AI Act compliance-minded "
        "behavior in assistant responses."
    ),
    "compliance/gdpr-privacy": (
        "Starter guardrails for privacy-sensitive handling of personal data and data "
        "subject requests."
    ),
    "core/general-safety": (
        "Balanced baseline rules for harmful content, privacy, and safe professional "
        "responses."
    ),
    "core/prompt-injection-and-secrets": (
        "Starter rules focused on prompt injection attempts, credential handling, and "
        "instruction hierarchy."
    ),
    "domain/customer-support": (
        "Starter rules for support workflows, account safety, and careful operational "
        "guidance."
    ),
    "domain/healthcare-information": (
        "Starter rules for informational healthcare assistants that should stay "
        "cautious and avoid personalized medical advice."
    ),
}


@dataclass(frozen=True)
class RulesPreset:
    """A packaged starter ``RULES.md`` preset."""

    slug: str
    title: str
    description: str
    content: str
    relative_path: str

    @property
    def package_path(self) -> str:
        """Return the in-package path users can browse in the repo."""
        return f"openbias/presets/rules/{self.relative_path}"


def discover_rules_presets() -> list[RulesPreset]:
    """Load and sort packaged rules presets deterministically."""
    root = files("openbias.presets").joinpath("rules")
    presets: list[RulesPreset] = []

    for category in sorted((entry for entry in root.iterdir() if entry.is_dir()), key=lambda entry: entry.name):
        for preset_file in sorted(
            (entry for entry in category.iterdir() if entry.is_file() and entry.name.endswith(".md")),
            key=lambda entry: entry.name,
        ):
            relative_path = f"{category.name}/{preset_file.name}"
            slug = str(PurePosixPath(relative_path).with_suffix(""))
            content = preset_file.read_text(encoding="utf-8")
            title = _extract_title(content, preset_file.name)
            description = _extract_description(content, slug)
            presets.append(
                RulesPreset(
                    slug=slug,
                    title=title,
                    description=description,
                    content=content,
                    relative_path=relative_path,
                )
            )

    return sorted(presets, key=lambda preset: preset.slug)


def get_rules_preset(slug: str) -> RulesPreset:
    """Return a single packaged rules preset by slug."""
    for preset in discover_rules_presets():
        if preset.slug == slug:
            return preset
    raise KeyError(f"Unknown rules preset: {slug}")


def _extract_title(content: str, filename: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return _humanize_filename(filename.removesuffix(".md"))


def _extract_description(content: str, slug: str) -> str:
    paragraph_lines: list[str] = []
    for block in content.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("#"):
            continue
        if all(_is_list_line(line) for line in lines):
            continue
        paragraph_lines = lines
        break

    if paragraph_lines:
        return " ".join(paragraph_lines)

    return _DESCRIPTION_FALLBACKS.get(slug, "Starter preset for project-local RULES.md.")


def _is_list_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith(("- ", "* ")):
        return True
    if len(stripped) > 2 and stripped[0].isdigit() and stripped[1:3] == ". ":
        return True
    return False


def _humanize_filename(stem: str) -> str:
    return " ".join(part.capitalize() for part in stem.replace("_", "-").split("-"))
