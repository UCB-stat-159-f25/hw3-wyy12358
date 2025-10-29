from pathlib import Path
import numpy as np
from ligotools import readligo as rl

DATA = Path(__file__).resolve().parents[2] / "data"

def test_read_hdf5_l1():
    result = rl.read_hdf5(str(DATA / "L-L1_LOSC_4_V2-1126259446-32.hdf5"))
    # Should return at least 3 values
    assert isinstance(result, (list, tuple))
    assert len(result) >= 3
    strain = result[0]
    assert isinstance(strain, np.ndarray)
    assert strain.ndim == 1

def test_read_hdf5_h1():
    result = rl.read_hdf5(str(DATA / "H-H1_LOSC_4_V2-1126259446-32.hdf5"))
    assert isinstance(result, (list, tuple))
    assert len(result) >= 3
    strain = result[0]
    assert len(strain) > 0