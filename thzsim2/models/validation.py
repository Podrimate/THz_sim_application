from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ValidationCaseResult:
    test_name: str
    description: str
    passed: bool
    score_name: str
    score_value: float
    tolerance: float
    notes: str
    details: dict[str, Any] = field(default_factory=dict)
    plot_path: Path | None = None


@dataclass(slots=True)
class ValidationSuiteResult:
    out_dir: Path
    summary_csv_path: Path
    manifest_path: Path
    summary_rows: list[dict[str, Any]]
    case_results: list[ValidationCaseResult]
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
