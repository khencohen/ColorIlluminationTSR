__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '1.3.2023'

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal

class SignalClass():
    def __init__(self, signal, time_step, name = ""):
        self.name = name
        self.time_step = time_step
        self.set_signal( signal )
        self.time_vec = np.arange( len(signal) ) * time_step
        return

    def set_signal(self, signal):
        self.signal = signal
        self.get_spectrum()
        return

    def set_signal_by_spectrum(self, spectrum_signal):
        self.signal_fft = spectrum_signal
        self.signal = fftpack.ifft(self.signal_fft)
        self.signal *= len(self.signal)
        self.frequencies = fftpack.fftfreq(self.signal.size, d=self.time_step)
        return

    def get_spectrum(self):
        self.signal_fft = fftpack.fft(self.signal)
        self.signal_fft /= len(self.signal_fft)
        self.frequencies = fftpack.fftfreq(self.signal.size, d=self.time_step)  # The corresponding frequencies
        return

    def get_resampled_signal(self, dense_factor = 2):
        from scipy import signal as sg
        new_signal = sg.resample(self.signal, len(self.signal) * dense_factor)
        return SignalClass(new_signal, time_step = self.time_step/dense_factor, name = self.name)

    def show_signal(self):
        plt.title('Temporal Signal ' + self.name)
        plt.plot(self.time_vec , np.real(self.signal))
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.show()
        return

    def show_spectrum(self, show_phase = False, range = '', shift_frequencies = True):
        if show_phase:
            fig = plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)

        plt.title(self.name)
        if shift_frequencies:
            shiftplot = lambda v: list(v[len(v)//2:])+list(v[:len(v)//2])
        else:
            shiftplot = lambda v: v

        plt.plot( shiftplot(self.frequencies) , shiftplot( np.abs(self.signal_fft)) )
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Amplitude')
        if range != '':
            plt.xlim(range)
        # plt.ylim([0.0, 0.4])

        if show_phase:
            plt.subplot(1, 2, 2)
            plt.title(self.name)
            plt.plot( shiftplot(self.frequencies), shiftplot(np.angle(self.signal_fft)))
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Phase')

        plt.show()

        # plt.subplot(1, 2, 1)
        # plt.plot(self.frequencies , np.abs(self.signal))
        # plt.xlabel('Time [s]')
        # plt.ylabel('Amplitude')
        # plt.show()
        return

    def filter_white_noise(self, threshold = 0.0):
        self.signal_fft[ np.abs(self.signal_fft) <= threshold ] = 0.0
        self.set_signal_by_spectrum(self.signal_fft)
        return

    def l2_distance(self, other, shift=0):
        # Returns the L2 distance between the signals
        # Assuming two signals with the same size
        return np.sqrt(np.square(np.abs(self.signal[shift:] - other.signal[:-shift])).sum()) / (
        self.signal[shift:].shape[0])

def temporal_dense_vec(signal_vec, upsampling_factor):
    return signal.resample(signal_vec, len(signal_vec)*upsampling_factor)

def get_Smat(b_vec, g_vec, r_vec):
    return np.concatenate((b_vec.reshape(-1, 1), \
                           g_vec.reshape(-1, 1), \
                           r_vec.reshape(-1, 1)), axis=1)

def get_invMmat(N):
    m_mat = np.zeros((N, N))
    for i in range(N):
        m_mat[i, i] = 2
        if i > 0:
            m_mat[i - 1, i] = -1
        if i < N - 1:
            m_mat[i + 1, i] = -1
    return np.linalg.inv(2 * m_mat)

def get_Itrans(b_vec, g_vec, r_vec, N):
    Smat = get_Smat(b_vec, g_vec, r_vec)
    invMmat = get_invMmat(N)
    return invMmat @ Smat @ np.linalg.pinv(Smat.transpose() @ invMmat @ Smat)


def integrate_interval(f_fun, t1, t2):
    return f_fun( np.linspace(t1,t2, 100 ) ).mean()

def sample_function(f_fun, time_vec):
    signal_vec = []
    for i in range(len(time_vec)-1):
        signal_vec += [ integrate_interval(f_fun, time_vec[i], time_vec[i+1]) ]
    return signal_vec

def simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec):
    signal_vec = []
    N = b_vec.shape[0]
    Itrans = get_Itrans(b_vec, g_vec, r_vec, N)
    for i in range(len(time_vec)-1):
        delta = ( time_vec[i+1] - time_vec[i] ) / N
        B = 0;  G = 0;    R = 0
        for j in range(N):
            B += b_vec[j]*integrate_interval(f_fun, time_vec[i]+delta*j, time_vec[i]+delta*(j+1))
            G += g_vec[j]*integrate_interval(f_fun, time_vec[i]+delta*j, time_vec[i]+delta*(j+1))
            R += r_vec[j]*integrate_interval(f_fun, time_vec[i]+delta*j, time_vec[i]+delta*(j+1))
        BGR = Itrans @ np.array([B,G,R]).reshape(-1,1)
        signal_vec += list(BGR.reshape(-1))
    return signal_vec

def get_bgr_vectors_from_N(N):
    if N == 3:
        b_vec = np.array([1, 0, 0])
        g_vec = np.array([0, 1, 0])
        r_vec = np.array([0, 0, 1])
    elif N == 4:
        b_vec = np.array([1, 0, 0, 1])
        g_vec = np.array([1, 0, 1, 0])
        r_vec = np.array([0, 1, 0, 1])
    elif N == 5:
        b_vec = np.array([0, 1, 0, 0, 0])
        g_vec = np.array([0, 0, 0, 1, 0])
        r_vec = np.array([1, 0, 1, 0, 1])
    elif N == 6:
        b_vec = np.array([1, 0, 1, 0, 1, 0])
        g_vec = np.array([0, 1, 0, 1, 0, 1])
        r_vec = np.array([1, 1, 1, 1, 1, 1])
    return b_vec, g_vec, r_vec
