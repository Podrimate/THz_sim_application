import numpy as np
import pytest

from thzsim2.io.trace_csv import read_trace_csv, write_trace_csv
from thzsim2.models import TraceData


def _trace_data():
    return TraceData(
        time_ps=np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float64),
        trace=np.array([0.1, 0.4, -0.2, 0.05], dtype=np.float64),
        source_kind="generated",
    )


def test_trace_csv_roundtrip(tmp_path):
    path = tmp_path / "trace.csv"
    trace_data = _trace_data()

    write_trace_csv(path, trace_data)
    loaded = read_trace_csv(path)

    assert np.allclose(loaded.time_ps, trace_data.time_ps)
    assert np.allclose(loaded.trace, trace_data.trace)
    assert loaded.source_kind == "csv"
    assert loaded.source_path == str(path.resolve())


def test_trace_csv_missing_column_raises(tmp_path):
    path = tmp_path / "bad_trace.csv"
    path.write_text("time_ps,value\n0.0,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="required columns"):
        read_trace_csv(path)


def test_trace_csv_non_monotonic_time_raises(tmp_path):
    path = tmp_path / "bad_trace.csv"
    path.write_text("time_ps,trace\n0.0,1.0\n-0.1,2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strictly increasing"):
        read_trace_csv(path)


def test_trace_csv_non_uniform_time_raises(tmp_path):
    path = tmp_path / "bad_trace.csv"
    path.write_text("time_ps,trace\n0.0,1.0\n0.5,2.0\n1.2,3.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="uniformly spaced"):
        read_trace_csv(path)
