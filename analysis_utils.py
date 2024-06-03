import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
import scipy.optimize as opt
from numba import njit, jit
import scipy.io as sio
import h5py

def Linewidth(x, a,  x0, gamma):
    """
    Function describing the PSD of a damped harmonic oscillator
    x: Frequency bins of PSD
    a: amplitude, here this is just a scaling factor for fitting
    x0: natural frequency of harmonic oscillator
    gamma: damping of harmonic oscillator

    """
    return a*(gamma)/((x0**2 - x**2)**2+(x*gamma)**2)  

def Linewidth2(x, a,  x0, gamma, c):
    """
    Function describing the PSD of a damped harmonic oscillator additional white measurement noise
    x: Frequency bins of PSD
    a: amplitude, here this is just a scaling factor for fitting
    x0: natural frequency of harmonic oscillator
    gamma: damping of harmonic oscillator
    c: white measurement noise

    """
    return a*(gamma)/((x0**2 - x**2)**2+(x*gamma)**2) + c

def Gaussian(x, A, x0, sigma):
    """
    Gaussian function
    x: variable to fit Gaussian to
    A: scaling factor
    x0: mean
    sigma: width of gaussian
    """
    return A*np.exp(-(x-x0)**2/(2*sigma**2))

def impulse_resp(time, t0, A, y, w0):
    output = np.zeros(len(time))
    y1 = y/2 # factor two required by definition
    w1 = np.sqrt(w0**2 - y1**2)
    for n, t in enumerate(time):
        if t < t0:
            output[n] = 0
        if t > t0:
            output[n] =  A*np.sin(w1*(t-t0))*np.exp(-y1*(t-t0))
    return output

from scipy.signal import butter, filtfilt, lfilter

def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a


def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, (low, high), btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def bandpass_peak_find(filename, cf, BW, fs, order):
    data = sio.loadmat(filename)
    x = data['x'][0]
    x = butter_bandpass_filter(x, cf-BW, cf+BW, fs = fs, order =order)
    max = np.max(x)
    return max

def bandpass_peak_find_noise(filename, cf, BW, fs, order):
    data = sio.loadmat(filename)
    x = data['x'][0]
    x = butter_bandpass_filter(x, cf-BW, cf+BW, fs = fs, order = order)
    m = int(np.random.uniform(0, len(x)))
    max = x[m]
    return max

def bandpass_peak_find2(filename, cf, BW, fs, order):
    data = load_data_hdf5(filename)
    x = data['x']
    x = butter_bandpass_filter(x, cf-BW, cf+BW, fs = fs, order =order)
    max = np.max(x)
    return max

def bandpass_peak_find_noise2(filename, cf, BW, fs, order):
    data = load_data_hdf5(filename)
    x = data['x']
    x = butter_bandpass_filter(x, cf-BW, cf+BW, fs = fs, order = order)
    m = int(np.random.uniform(0, len(x)))
    max = x[m]
    return max

def histogram_and_fit(amp_max, bin_num, count_amp, fit = True, plot = True):
    hist3, bins3 = np.histogram(amp_max, bin_num)
    bin_c = bins3[1:]-(bins3[1]-bins3[0])/2
    mean = np.mean(amp_max)
    std = np.std(amp_max)
    if fit == True:
        fit3, cov3 = opt.curve_fit(Gaussian, bin_c, hist3, p0 = [count_amp, mean, std])
        x_hist3 = np.linspace(bins3[0], bins3[-1], 100)
        fitted3 = Gaussian(x_hist3, *fit3)
    if plot == True:
        plt.stairs(hist3, bins3)
        plt.plot(x_hist3, fitted3)

    if fit == True:
        return hist3, bins3, fit3, x_hist3, fitted3
    else: 
        return hist3, bins3
    
def save_data_hdf5(filename, data):
    """
    Saves data in HDF5. Does it in a simple way by looping through data and datasetnames
    filename: Filename of file you want to save
    data: the data you want to save as a dictionary
    """
    keys = list(data.keys())
    with h5py.File(filename, "w") as f:
        for key in keys:
            f[key] = data[key]
        #f.close()

def load_data_hdf5(filename):
    """
    Loads data in HDF5. Doesn't load metadata. Outputs as dictionary.
    filename: Filename of file you want to load
    """
    f = h5py.File(filename, "r")
    keys = list(f.keys())
    mdict = {}
    for key in keys:
        dataset = list(f[key])
        mdict[key] = dataset
    f.close()
    return mdict

def make_optimal_filter(response_template, noise_template, noise_template_frequency):
    """
    Makes optimal filter from response template and noise template
    response_template: The average response of the oscillator to and impulse, time domain
    noise_template: The PSD of the oscillator driven by noise processes (in our case usually white noise from gas)
    noise_template_frequency: Frequency bins of the noise template PSD
    """

    stilde = np.fft.rfft(response_template)
    sfreq = np.fft.rfftfreq(len(response_template),d=1e-6)
    J_out = np.interp(sfreq, noise_template_frequency, noise_template)
    phi = stilde/J_out

    phi_t = np.fft.irfft(phi)
    phi_t = phi_t/np.max(phi_t)
    return phi_t

def make_optimal_filter_ns(response_template, noise_template, noise_template_frequency):
    """
    Makes optimal filter from response template and noise template for non-stationary data. Basically just cuts off some low frequency artifacts due to using rectangular window in generating PSD.
    response_template: The average response of the oscillator to and impulse, time domain
    noise_template: The PSD of the oscillator driven by noise processes (in our case usually white noise from gas)
    noise_template_frequency: Frequency bins of the noise template PSD
    """

    stilde = np.fft.rfft(response_template)
    sfreq = np.fft.rfftfreq(len(response_template),d=1e-6)
    J_out = np.interp(sfreq, noise_template_frequency, noise_template)
    J_out[sfreq<10000] = 10
    phi = stilde/J_out

    phi_t = np.fft.irfft(phi)
    phi_t = phi_t/np.max(phi_t)
    return phi_t

def optimal_filter(filter, data):
    """
    Applies optimal filter to data and returns the estimated value of the impulse.
    Does this in the frequency domain for computational efficiency.
    Only searches for impulse around time of impulse (which in this case is known)
    Filter: The optimal filter template. The output of make_optimal_filter
    data: The time domain data in which you want to search for an impulse
    """
    dl = len(data)
    corr_data = np.abs(scisig.correlate(data, filter, mode = 'same'))
    corr_max = np.max(corr_data[int(dl/2-dl/10):int(dl/2+dl/10)])
    return corr_max

def optimal_filter_noise(filter, data):
    """
    Applies optimal filter to data and returns random value for estimate. Needs to do this to stop search bias in noise.
    Does this in the frequency domain for computational efficiency.
    Only searches for impulse around time of impulse (which in this case is known)
    Filter: The optimal filter template. The output of make_optimal_filter
    data: The time domain data in which you want to search for an impulse
    """
    dl = len(data)
    corr_data = np.abs(scisig.correlate(data, filter, mode = 'same'))
    corr_max = np.max(corr_data[int(dl/2-dl/10):int(dl/2+dl/10)])
    m = int(np.random.uniform(0, dl/5))
    corr_max = corr_data[int(dl/2-dl/10)+m]
    return corr_max

def optimal_filter_short(filter, data, band):
    """
    Applies optimal filter with a restricted template to data and returns the estimated value of the impulse.
    Does this in the frequency domain for computational efficiency.
    Only searches for impulse around time of impulse (which in this case is known)
    Filter: The optimal filter template. The output of make_optimal_filter
    data: The time domain data in which you want to search for an impulse
    """
    dl = len(data)
    corr_data = np.abs(scisig.correlate(data, filter[int(dl/2-band):int(dl/2+band)], mode = 'same'))
    corr_max = np.max(corr_data[int(dl/2-dl/10):int(dl/2+dl/10)])
    return corr_max

def optimal_filter_short_noise(filter, data):
    """
    Applies optimal filter to data with a restricted template and returns random value for estimate. Needs to do this to stop search bias in noise.
    Does this in the frequency domain for computational efficiency.
    Only searches for impulse around time of impulse (which in this case is known)
    Filter: The optimal filter template. The output of make_optimal_filter
    data: The time domain data in which you want to search for an impulse
    """
    dl = len(data)
    corr_data = np.abs(scisig.correlate(data, filter, mode = 'same'))
    corr_max = np.max(corr_data[int(dl/2-dl/10):int(dl/2+dl/10)])
    m = int(np.random.uniform(0, dl/5))
    corr_max = corr_data[int(dl/2-dl/10)+m]
    return corr_max

def lockin(data, fs, demod_freq, BW_pre, BW, BW2, mode):
    """
    Lock-in amplifier. Can output X and Y quadratures or R and theta.
    data: data you want to demodulate
    fs = sampling frequency of data
    demod_freq: frequency of the reference
    BW_pre: BW of bandpass filter of data before lock-in. If 0 then no filter applied
    BW: bandwidth of lowpass filter on X and Y quadratures
    BW2: bandwidth of lowpass filter of R and theta quadratures
    mode: either 'XY' and 'R'. Outputs different quadratures
    """
    time = np.array(range(len(data)))/fs
    demod = np.cos(2*np.pi*demod_freq*time)
    demod2 = np.sin(2*np.pi*demod_freq*time)
    if BW_pre != 0:
            data = butter_bandpass_filter(data, demod_freq - BW_pre, demod_freq + BW_pre, fs, order = 3)
    X_tt = data*demod
    Y_tt = data*demod2
    X_tt_filt = butter_lowpass_filter(X_tt, BW, fs, order = 3)
    Y_tt_filt = butter_lowpass_filter(Y_tt, BW, fs, order = 3)
    if mode == 'XY':
        return time, X_tt_filt, Y_tt_filt
    elif mode == 'R':
        R2 = X_tt_filt**2 + Y_tt_filt**2
        theta = np.unwrap(-2*np.arctan(X_tt_filt[1:]/Y_tt_filt[1:]))/2
        R2_filt = butter_lowpass_filter(R2, BW2, fs, order = 2)
        theta_filt = butter_lowpass_filter(theta, BW2, fs, order = 2)
        return time, R2, R2_filt, theta_filt
    else:
        return 0