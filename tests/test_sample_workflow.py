import json

import numpy as np
import pytest

from thzsim2.io.nk_csv import NKData, write_nk_csv
from thzsim2.models import ConstantNK, Drude, Fit, Layer, LorentzOscillator, NKFile, DrudeLorentz
from thzsim2.workflows.reference import generate_reference_pulse, prepare_reference
from thzsim2.workflows.sample_workflow import build_sample


def _reference_result(tmp_path):
    reference_input = generate_reference_pulse(
        model="gaussian_carrier",
        sample_count=2048,
        dt_ps=0.01,
        time_center_ps=0.0,
        pulse_center_ps=0.8,
        tau_ps=0.20,
        f0_thz=1.0,
        amp=1.0,
        phi_rad=0.0,
    )
    return prepare_reference(reference_input, output_root=tmp_path, run_label="sample-phase2")


def test_build_sample_creates_expected_exports_and_updates_run_manifest(tmp_path):
    reference_result = _reference_result(tmp_path)
    layer = Layer(
        name="film",
        thickness_um=Fit(100.0, rel_min=0.5, rel_max=1.5, label="film_thickness_um"),
        material=Drude(
            eps_inf=3.4,
            plasma_freq_thz=Fit(1.2, rel_min=0.7, rel_max=1.3, label="film_plasma_freq_thz"),
            gamma_thz=0.2,
        ),
    )

    sample_result = build_sample(
        layers=[layer],
        reference=reference_result,
        out_dir=reference_result.run_dir / "sample",
    )

    expected_paths = {
        "sample_manifest_json": sample_result.out_dir / "sample_manifest.json",
        "sample_structure_txt": sample_result.out_dir / "sample_structure.txt",
        "sample_nk_png": sample_result.out_dir / "sample_nk.png",
        "layer_01_film_nk_csv": sample_result.out_dir / "layer_01_film_nk.csv",
    }
    for path in expected_paths.values():
        assert path.exists()

    assert sample_result.fit_parameters[0].bound_min == pytest.approx(50.0)
    assert sample_result.fit_parameters[0].bound_max == pytest.approx(150.0)
    assert sample_result.fit_parameters[1].bound_min == pytest.approx(0.84)
    assert sample_result.fit_parameters[1].bound_max == pytest.approx(1.56)

    manifest = json.loads(expected_paths["sample_manifest_json"].read_text(encoding="utf-8"))
    assert manifest["workflow"] == "sample"
    assert manifest["layers"][0]["name"] == "film"
    assert manifest["files"]["sample_structure_txt"] == "sample_structure.txt"
    assert manifest["files"]["layer_nk_csvs"][0]["path"] == "layer_01_film_nk.csv"
    nk_csv = (sample_result.out_dir / "layer_01_film_nk.csv").read_text(encoding="utf-8").splitlines()
    assert nk_csv[1].split(",")[0] != "0"

    run_manifest = json.loads((reference_result.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["sample_manifest"] == "sample/sample_manifest.json"


def test_build_sample_interpolates_and_linearly_extrapolates_imported_nk(tmp_path):
    nk_path = tmp_path / "imported_nk.csv"
    write_nk_csv(
        nk_path,
        NKData(
            freq_thz=np.array([0.5, 1.0, 1.5], dtype=np.float64),
            n=np.array([1.5, 1.7, 1.9], dtype=np.float64),
            k=np.array([0.01, 0.02, 0.03], dtype=np.float64),
        ),
    )

    sample_result = build_sample(
        layers=[Layer(name="film", thickness_um=80.0, material=NKFile(nk_path))],
        freq_grid_thz=np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float64),
        out_dir=tmp_path / "sample",
    )

    layer = sample_result.layers[0]
    assert np.allclose(layer.n, [1.4, 1.6, 1.8, 2.0])
    assert np.allclose(layer.k, [0.005, 0.015, 0.025, 0.035])
    assert np.allclose(layer.imported_freq_thz, [0.5, 1.0, 1.5])
    assert (tmp_path / "sample" / "sample_nk.png").exists()


def test_build_sample_supports_explicit_freq_grid_override(tmp_path):
    freq_grid_thz = np.linspace(0.2, 1.8, 64)
    sample_result = build_sample(
        layers=[Layer(name="film", thickness_um=20.0, material=ConstantNK(n=2.1, k=0.04))],
        freq_grid_thz=freq_grid_thz,
        out_dir=tmp_path / "sample",
    )

    assert np.allclose(sample_result.freq_grid_thz, freq_grid_thz)
    assert np.allclose(sample_result.layers[0].freq_thz, freq_grid_thz)


def test_build_sample_writes_readable_structure_summary(tmp_path):
    reference_result = _reference_result(tmp_path)
    sample_result = build_sample(
        layers=[
            Layer(
                name="film",
                thickness_um=Fit(100.0, rel_min=0.5, rel_max=1.5, label="film_thickness_um"),
                material=DrudeLorentz(
                    eps_inf=3.2,
                    plasma_freq_thz=0.9,
                    gamma_thz=0.15,
                    oscillators=(LorentzOscillator(delta_eps=0.6, resonance_thz=1.4, gamma_thz=0.08),),
                ),
            )
        ],
        reference=reference_result,
        out_dir=reference_result.run_dir / "sample",
    )

    text = (sample_result.out_dir / "sample_structure.txt").read_text(encoding="utf-8")
    assert "Sample Structure" in text
    assert "Layer 1: film" in text
    assert "material = DrudeLorentz" in text
    assert "thickness_um = 100" in text
