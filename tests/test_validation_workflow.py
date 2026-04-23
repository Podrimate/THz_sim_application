from pathlib import Path

import matplotlib.pyplot as plt

from thzsim2.notebook_api import (
    load_validation_summary,
    plot_validation_plot_grid,
    plot_validation_summary,
    run_validation_suite,
)


def test_run_validation_suite_fast_subset_exports_files(tmp_path):
    result = run_validation_suite(
        output_root=tmp_path,
        run_label="pytest-validation",
        tests=[
            "drude_roundtrip",
            "empty_stack_identity",
            "noiseless_drude_recovery",
            "photonic_crystal_bandgap_convergence",
        ],
        mode="fast",
    )

    assert result.out_dir.exists()
    assert result.summary_csv_path.exists()
    assert result.manifest_path.exists()
    assert result.artifact_paths["validation_summary_png"].exists()
    assert result.artifact_paths["validation_plot_grid_png"].exists()
    assert [row["test_name"] for row in result.summary_rows] == [
        "drude_roundtrip",
        "empty_stack_identity",
        "noiseless_drude_recovery",
        "photonic_crystal_bandgap_convergence",
    ]


def test_load_validation_summary_reads_exported_rows(tmp_path):
    result = run_validation_suite(
        output_root=tmp_path,
        run_label="pytest-validation-load",
        tests=["drude_roundtrip", "empty_stack_identity"],
        mode="fast",
    )

    rows = load_validation_summary(result.summary_csv_path)

    assert len(rows) == 2
    assert rows[0]["test_name"] == "drude_roundtrip"
    assert isinstance(rows[0]["passed"], bool)


def test_plot_validation_helpers_return_figures(tmp_path):
    result = run_validation_suite(
        output_root=tmp_path,
        run_label="pytest-validation-plots",
        tests=["drude_roundtrip", "zero_thickness_identity"],
        mode="fast",
    )

    fig1, _ = plot_validation_summary(result)
    fig2, _ = plot_validation_plot_grid(result)

    assert fig1 is not None
    assert fig2 is not None

    plt.close(fig1)
    plt.close(fig2)


def test_oblique_polarization_validation_case_exports_plot(tmp_path):
    result = run_validation_suite(
        output_root=tmp_path,
        run_label="pytest-validation-oblique-pol",
        tests=["oblique_polarization_consistency"],
        mode="fast",
    )

    assert len(result.summary_rows) == 1
    assert result.summary_rows[0]["test_name"] == "oblique_polarization_consistency"
    assert result.summary_rows[0]["passed"] is True
    assert result.case_results[0].plot_path is not None
    assert Path(result.case_results[0].plot_path).exists()
