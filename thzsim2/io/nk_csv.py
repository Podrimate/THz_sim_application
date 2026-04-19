import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

NK_FIELDNAMES = ("freq_thz", "n", "k")


@dataclass(slots=True)
class NKData:
    freq_thz: NDArray[np.float64]
    n: NDArray[np.float64]
    k: NDArray[np.float64]

    def __post_init__(self):
        freq_thz = np.asarray(self.freq_thz, dtype=np.float64)
        n = np.asarray(self.n, dtype=np.float64)
        k = np.asarray(self.k, dtype=np.float64)

        if freq_thz.ndim != 1:
            raise ValueError("freq_thz must be 1D")
        if freq_thz.size == 0:
            raise ValueError("n,k data must be non-empty")
        if n.shape != freq_thz.shape or k.shape != freq_thz.shape:
            raise ValueError("freq_thz, n, and k must have matching shapes")
        if not (np.isfinite(freq_thz).all() and np.isfinite(n).all() and np.isfinite(k).all()):
            raise ValueError("n,k data must contain only finite values")
        if np.any(freq_thz < 0.0):
            raise ValueError("freq_thz must be nonnegative")
        if freq_thz.size > 1 and np.any(np.diff(freq_thz) <= 0.0):
            raise ValueError("freq_thz must be strictly increasing")

        self.freq_thz = freq_thz
        self.n = n
        self.k = k


def read_nk_csv(path) -> NKData:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("n,k CSV is missing a header row")
        missing = [name for name in NK_FIELDNAMES if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"n,k CSV is missing required columns: {', '.join(missing)}")

        freq_thz = []
        n = []
        k = []
        for row in reader:
            freq_thz.append(float(row["freq_thz"]))
            n.append(float(row["n"]))
            k.append(float(row["k"]))

    return NKData(
        freq_thz=np.asarray(freq_thz, dtype=np.float64),
        n=np.asarray(n, dtype=np.float64),
        k=np.asarray(k, dtype=np.float64),
    )


def write_nk_csv(path, nk_data: NKData):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NK_FIELDNAMES)
        writer.writeheader()
        for freq_thz, n_value, k_value in zip(
            nk_data.freq_thz,
            nk_data.n,
            nk_data.k,
            strict=True,
        ):
            writer.writerow(
                {
                    "freq_thz": format(float(freq_thz), ".16g"),
                    "n": format(float(n_value), ".16g"),
                    "k": format(float(k_value), ".16g"),
                }
            )
