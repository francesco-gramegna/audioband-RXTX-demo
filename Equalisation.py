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

import numpy as np
from scipy.signal import decimate

class ChannelEstimator:
    def __init__(self, config, mod, L):
        # L is the channel order (memory length). Taps = L + 1.
        self.config = config
        
        # --- Downsampling Setup ---
        # Q is assumed to be the number of samples per symbol
        try:
            D = config['samplesPerSymbol'] 
            if D < 1: D = 1
        except KeyError:
            # Fallback if config is missing key, but this must be defined for proper downsampling
            raise ValueError("Config must include 'samples_per_symbol' for downsampling.")

        self.D = D
        self.L_ds = int(np.ceil((L + 1) / self.D)) - 1 # New channel order length

        # 1. Downsample the Preamble once
        preamble_full = mod.getBasebandPreamble()
        self.preamble_full = preamble_full
        # Use decimate for necessary anti-aliasing filtering
        self.preamble_ds = decimate(preamble_full, self.D, ftype='fir')
        
        # --- Normal Equation Pre-Calculation (Memory Efficient) ---
        X_temp = self.make_X_matrix(self.preamble_ds, self.L_ds)
        
        # Calculate R_xx = X^H * X 
        R_xx = X_temp.conj().T @ X_temp
        
        # Calculate the pseudo-inverse of the small R_xx matrix
        self.R_xx_pinv = np.linalg.pinv(R_xx)
        
        # Delete the large temporary X matrix immediately
        del X_temp 

    def make_X_matrix(self, x, L_ds):
        x = np.array(x, dtype=np.complex64)
        N_ds = len(x)
        num_taps = L_ds + 1
        rows = N_ds - L_ds
        
        X = np.zeros((rows, num_taps), dtype=np.complex64)

        for i in range(rows):
            # Grab x[i] to x[i+L_ds] and reverse it (x[n], x[n-1], ...)
            segment = x[i : i + num_taps]
            X[i, :] = segment[::-1]

        return X

    
    def estimateChannel(self, inData):
        # 1. Slice and Downsample the Received Data
        rx_segment_full = inData[:len(self.preamble_full)] # Assuming full length is known
        
        # Filter and decimate the received segment using the same factor D
        rx_segment_ds = decimate(rx_segment_full, self.D, ftype='fir')
        
        # 2. Slice the used portion of y (y_used)
        y_used = rx_segment_ds[self.L_ds:]
        
        # 3. Re-create X matrix TEMPORARILY for cross-correlation r_xy
        X_temp = self.make_X_matrix(self.preamble_ds, self.L_ds)
        
        # 4. Calculate the cross-correlation vector: r_xy = X^H * y
        r_xy = X_temp.conj().T @ y_used 
        
        # Delete temporary X matrix immediately to free memory
        del X_temp 
        
        # 5. Solve the Normal Equation: h = R_xx_pinv * r_xy
        h_hat = self.R_xx_pinv @ r_xy

        # h_hat contains the L_ds + 1 channel taps at the downsampled rate
        return h_hat
