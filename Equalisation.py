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
        self.L = config['channelSymbolsLen']

        bits = utils.generatePreambleBits(config['preambleSymbols'], config['bitsPerSymbol'])
        symbols = bits.reshape((-1, config['bitsPerSymbol']))
        indices = symbols.dot(1 << np.arange(config['bitsPerSymbol']-1, -1, -1))

        self.preambleSymbols = np.array(mod.constellation.map(indices))
        
        self.preamble = mod.getBasebandPreamble()

        if(self.L >= len(self.preambleSymbols)):
            raise ValueError("Then channel memory should be smaller than the preamble for good estimation!")

        X = self.build_convolution_matrix()
        self.Xinv = np.linalg.pinv(X)
        self.Xc = X.conj()
        self.X1 = np.linalg.pinv(self.Xc.T @ X) 

    
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

        y = data[self.L - 1: len(self.preambleSymbols)]
        p_hat = self.X1 @ (self.Xc.T @ y)
        return p_hat


