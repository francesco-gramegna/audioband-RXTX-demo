import mathUtils
from commons import Common
import multiprocessing
import threading
import time
import sounddevice as sd
import Demodulator
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import Synchronisation
import numpy as np
import utils
from scipy.signal import decimate, upfirdn # Added upfirdn


class ChannelEstimator:
    def __init__(self, config, mod):
        self.config = config
        self.L = config['ChannelSymbolsLen']

        self.preambleBits = utils.generatePreambleBits(config['preambleSymbols'], config['bitsPerSymbol'])
        self.preamble = mod.getBasebandPreamble()

        if(L >= len(self.preambleBits)):
            raise ValueError("Then channel memory should be smaller than the preamble for good estimation!")

        self.X = self.build_convolution_matrix()
        self.Xinv = np.linalg.pinv(X)
        self.Xc = self.X.conj()
        self.X1 = np.linalg.pinv(X.conj().T @ X) 

    
    def build_convolution_matrix(self):
        x = self.preambleBits
        N = len(x)
        L = self.L

        rows = N - L + 1
        X = np.zeros((rows, L), dtype=complex)

        for i in range(rows):
            # Extract reversed window
            X[i, :] = x[i + L - 1 : i - 1 : -1]

        return X
    
    #data is already matched filtered and sampled
    def estimateChannel(self, data):
        data = data[:len(self.preambleBits)]
        y = data[self.L - 1: len(self.preambleBits)]
        p_hat = self.X1 @ (self.Xc.T @ y)
        return p_hat




