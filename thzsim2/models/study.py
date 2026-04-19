from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StudyCaseResult:
    case_id: int
    replicate_id: int
    seed: int | None
    case_dir: Path
    assignments: dict[str, Any]
    success: bool
    objective_value: float
    metric_value: float


@dataclass(slots=True)
class StudyResult:
    out_dir: Path
    cases_dir: Path
    summary_csv_path: Path
    correlation_csv_path: Path
    manifest_path: Path
    config_path: Path
    summary_rows: list[dict[str, Any]]
    correlation_rows: list[dict[str, Any]]
    case_results: list[StudyCaseResult]
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
