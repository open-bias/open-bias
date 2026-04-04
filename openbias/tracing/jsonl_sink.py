"""Local JSONL sink for replayable trace capture."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from openbias.traces import TraceCase, append_trace_case


class JsonlTraceSink:
    """Append replayable trace cases to a local JSONL dataset."""

    def __init__(self, path_template: str):
        self._path_template = path_template

    def append(self, case: TraceCase) -> Path:
        """Append one case to the resolved dataset path."""

        path = self._resolve_output_path()
        return append_trace_case(path, case)

    def _resolve_output_path(self) -> Path:
        now = datetime.now(tz=timezone.utc)
        rendered = now.strftime(self._path_template)
        path = Path(rendered)
        if path.suffix.lower() == ".jsonl":
            return path
        return path / f"{now.strftime('%Y-%m-%d')}.jsonl"
