from __future__ import annotations

import argparse
from pathlib import Path

from thzsim2.notebook_api import run_single_layer_drude_compat_study


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the old-compatible 4800-case one-layer Drude study against a measured reference CSV."
        )
    )
    parser.add_argument("reference_csv", help="Path to the measured reference CSV file.")
    parser.add_argument(
        "--output-root",
        default="notebooks/runs",
        help="Base output directory for timestamped run folders.",
    )
    parser.add_argument(
        "--run-label",
        default="measured-4800-compat",
        help="Suffix label used in the timestamped run folder name.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the live progress bar output.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    result = run_single_layer_drude_compat_study(
        args.reference_csv,
        output_root=Path(args.output_root),
        run_label=args.run_label,
        show_progress=not args.quiet,
    )
    print(result.out_dir)


if __name__ == "__main__":
    main()
