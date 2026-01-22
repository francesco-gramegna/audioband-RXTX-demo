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

from scipy.linalg import toeplitz


class ChannelEstimator:
    def __init__(self, config, mod):
        self.config = config
        self.L = config['channelSymbolsLen']

        bits = utils.generatePreambleBits(config['preambleSymbols'], config['bitsPerSymbol'])
        symbols = bits.reshape((-1, config['bitsPerSymbol']))
        indices = symbols.dot(1 << np.arange(config['bitsPerSymbol']-1, -1, -1))

        self.preambleSymbols = np.array(mod.constellation.map(indices))
        
        self.preamble = mod.getBasebandPreamble()

        if(self.L >= len(self.preambleSymbols)):
            raise ValueError("Then channel memory should be smaller than the preamble for good estimation!")

        X = self.build_convolution_matrix()
        self.X = X
        self.Xinv = np.linalg.pinv(X)
        self.Xc = X.conj()
        self.X1 = np.linalg.pinv(self.Xc.T @ X) 

        #we regularise
        eps = 1e-5
        A = (self.Xc.T @ self.X)  # (L,L)
        A += eps * np.eye(A.shape[0])
        

        try:
            X1 = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            X1 = np.linalg.pinv(A)

        #self.X1 = X1


    
    def build_convolution_matrix(self):
        x = self.preambleSymbols
        N = len(x)
        L = self.L

        rows = N - L + 1
        X = np.zeros((rows, L), dtype=complex)

        for i in range(rows):
            X[i, :] = x[i:i+L][::-1] 

        return X

    #data is already matched filtered and sampled
    def estimateChannel(self, data):

        data = data[:len(self.preambleSymbols)]

        y = data[self.L - 1:]
        #print(len(y))
        #p_hat = self.Xinv @ y 
        #p_hat = self.X1 @ y 

        p_hat = self.X1 @ (self.Xc.T @ y)

        y_predicted = self.X @ p_hat #the preamble I should have recieved
        e = y - y_predicted
        len_e = len(e)
        if len_e <= self.L:
        # Avoid division by zero 
            print("Preamble is Too short for accurate err detection")
            noise_var = 0.01 
        else:
            noise_var = np.sum(np.abs(e)**2) / (len_e - self.L)

        return p_hat, noise_var


class MMSEEqualizer:
    def __init__(self, config, mod):
        self.mod = mod
        self.K = config['channelSymbolsLen'] * 3
        self.delta = self.K // 2

    def equalize(self, data, p_hat, noise_var):
        data = np.asarray(data)
        L = len(p_hat)

        # Toeplitz convolution matrix
        col = np.zeros(self.K + L - 1, dtype=complex)
        col[:L] = p_hat
        row = np.zeros(self.K, dtype=complex)
        row[0] = p_hat[0]
        H = toeplitz(col, row)

        Hh = H.conj().T
        R = Hh @ H + (noise_var + 1e-8) * np.eye(H.shape[1])
        p_vector = Hh[:, self.delta]

        try:
            w = np.linalg.solve(R, p_vector)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(R) @ p_vector

        #w = 0.5 * (w + np.flip(np.conj(w)))

        # Apply filter and align
        y_full = np.convolve(data, w, mode='full')
        y_hat = y_full[self.delta : self.delta + len(data)]

        return y_hat, w

