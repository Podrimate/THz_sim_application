from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.lecture_assets_v2 import (
    build_lecture_bundle,
    build_notebook_demo_bundle,
    main,
    plot_saved_lecture_study_triptych,
    run_lecture_map_from_spec,
    run_lecture_measured_reflection_example,
    run_lecture_measured_transmission_example,
)

__all__ = [
    "build_lecture_bundle",
    "build_notebook_demo_bundle",
    "plot_saved_lecture_study_triptych",
    "run_lecture_map_from_spec",
    "run_lecture_measured_transmission_example",
    "run_lecture_measured_reflection_example",
    "main",
    "Path",
]


if __name__ == "__main__":
    main()
