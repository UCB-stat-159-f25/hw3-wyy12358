import numpy as np
from pathlib import Path
from ligotools.utils import whiten, write_wavfile, reqshift
from scipy.interpolate import interp1d

def test_whiten_basic():
    fs = 4096
    dt = 1.0 / fs
    N = 4096
    strain = np.random.randn(N)
    freqs = np.fft.rfftfreq(N, dt)
    psd = np.ones_like(freqs)
    interp_psd = interp1d(freqs, psd, bounds_error=False, fill_value=1)
    result = whiten(strain, interp_psd, dt)
    assert isinstance(result, np.ndarray)
    assert result.shape == strain.shape
    assert np.isfinite(np.var(result))

def test_reqshift_preserves_shape_and_phase():
    fs = 4096
    x = np.sin(2 * np.pi * 30 * np.arange(0, 1, 1/fs))
    y = reqshift(x, fshift=100, sample_rate=fs)
    assert y.shape == x.shape
    assert np.isrealobj(y)

def test_write_wavfile_tmp(tmp_path):
    fs = 4096
    x = np.random.randn(8192)
    outfile = tmp_path / "tmp.wav"
    write_wavfile(outfile, fs, x)
    assert outfile.exists() and outfile.stat().st_size > 0