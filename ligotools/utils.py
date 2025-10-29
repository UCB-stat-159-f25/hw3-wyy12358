import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.signal import filtfilt
from pathlib import Path
from scipy.io import wavfile

def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0, 2048, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def write_wavfile(filename,fs,data):
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename,int(fs), d)

def reqshift(data,fshift=100,sample_rate=4096):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    y[0:nbins]=0.
    z = np.fft.irfft(y)
    return z

def plot_results(eventname, det, time, timemax, tevent, 
                 SNR, strain_whitenbp, template_match, fs,
                 template_fft, datafreq, freqs, data_psd,
                 d_eff, FIGS, plottype='png'):
    """
    Plot SNR time series, residual whitened data, and PSD comparison
    for a given detector (H1 or L1).
    """
    pcolor = 'g' if det == 'L1' else 'r'

    # SNR vs time
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time - timemax, SNR, pcolor, label=f'{det} SNR(t)')
    plt.grid(True)
    plt.ylabel('SNR')
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.legend(loc='upper left')
    plt.title(f'{det} matched filter SNR around event')

    plt.subplot(2,1,2)
    plt.plot(time - timemax, SNR, pcolor, label=f'{det} SNR(t)')
    plt.xlim([-0.15, 0.05])
    plt.grid(True)
    plt.ylabel('SNR')
    plt.xlabel(f'Time since {timemax:.4f}')
    plt.legend(loc='upper left')
    plt.savefig(FIGS / f"{eventname}_{det}_SNR.{plottype}")

    # Residual whitened data
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time - tevent, strain_whitenbp, pcolor, label=f'{det} whitened h(t)')
    plt.plot(time - tevent, template_match, 'k', label='Template(t)')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title(f'{det} whitened data around event')

    plt.subplot(2,1,2)
    plt.plot(time - tevent, strain_whitenbp - template_match, pcolor, label=f'{det} resid')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title(f'{det} Residual whitened data after subtracting template')
    plt.savefig(FIGS / f"{eventname}_{det}_matchtime.{plottype}")

    # PSD comparison
    plt.figure(figsize=(10,6))
    template_f = np.abs(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff
    plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(freqs, np.sqrt(data_psd), pcolor, label=f'{det} ASD')
    plt.xlim(20, fs/2)
    plt.ylim(1e-24, 1e-20)
    plt.grid()
    plt.xlabel('frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
    plt.legend(loc='upper left')
    plt.title(f'{det} ASD and template around event')
    plt.savefig(FIGS / f"{eventname}_{det}_matchfreq.{plottype}")