import numpy as np
import pytest

from thzsim2.io.nk_csv import NKData, read_nk_csv, write_nk_csv


def _nk_data():
    return NKData(
        freq_thz=np.array([0.1, 0.2, 0.4], dtype=np.float64),
        n=np.array([1.5, 1.6, 1.7], dtype=np.float64),
        k=np.array([0.01, 0.02, 0.03], dtype=np.float64),
    )


def test_nk_csv_roundtrip(tmp_path):
    path = tmp_path / "nk.csv"
    nk_data = _nk_data()

    write_nk_csv(path, nk_data)
    loaded = read_nk_csv(path)

    assert np.allclose(loaded.freq_thz, nk_data.freq_thz)
    assert np.allclose(loaded.n, nk_data.n)
    assert np.allclose(loaded.k, nk_data.k)


def test_nk_csv_missing_column_raises(tmp_path):
    path = tmp_path / "bad_nk.csv"
    path.write_text("freq_thz,n\n0.1,1.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="required columns"):
        read_nk_csv(path)


def test_nk_csv_invalid_frequency_order_raises(tmp_path):
    path = tmp_path / "bad_nk.csv"
    path.write_text("freq_thz,n,k\n0.2,1.5,0.01\n0.1,1.6,0.02\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strictly increasing"):
        read_nk_csv(path)
