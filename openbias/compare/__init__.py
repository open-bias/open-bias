"""Compare baseline and candidate policy behavior."""

from openbias.compare.runner import (
    build_comparison_result,
    compare_policy_runs,
    render_comparison_markdown,
)
from openbias.compare.schema import ComparisonGate, PolicyComparisonResult, SuiteComparison, TraceComparison

__all__ = [
    "ComparisonGate",
    "PolicyComparisonResult",
    "SuiteComparison",
    "TraceComparison",
    "build_comparison_result",
    "compare_policy_runs",
    "render_comparison_markdown",
]
