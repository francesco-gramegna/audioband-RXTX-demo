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
    def __init__(self, config, mod, L):
        self.config = config
        
        D = config['samplesPerSymbol'] 

        self.D = D
        self.L_full = L # Store the original desired channel length
        self.L_ds = int(np.ceil((L + 1) / self.D)) - 1

        # 1. Downsample the Preamble once
        preamble_full = mod.getBasebandPreamble()
        self.lenPreamble = len(preamble_full)
        self.preamble_ds = decimate(preamble_full, self.D, ftype='fir')
        
        # --- Normal Equation Pre-Calculation ---
        X_temp = self.make_X_matrix(self.preamble_ds, self.L_ds)
        R_xx = X_temp.conj().T @ X_temp
        self.R_xx_pinv = np.linalg.pinv(R_xx)
        del X_temp 

    def make_X_matrix(self, x, L_ds):
        x = np.array(x, dtype=np.complex64)
        N_ds = len(x)
        num_taps = L_ds + 1
        rows = N_ds - L_ds
        
        X = np.zeros((rows, num_taps), dtype=np.complex64)

        for i in range(rows):
            segment = x[i : i + num_taps]
            X[i, :] = segment[::-1]

        return X

    def interpolate_h(self, h_ds):
        """
        Interpolates the downsampled channel estimate (h_ds) back to the 
        full sampling rate by a factor of D.
        """
        # upfirdn(h_filter, x_data, up_factor, down_factor)
        # Here, we only want to upsample, so up_factor=D, down_factor=1.
        # h_filter is the low-pass filter required for interpolation. 
        # upfirdn is often optimized to handle this by default if a simple 
        # FIR filter is passed. Using the FIR method for simplicity:
        
        h_interp = upfirdn(np.array([1.]), h_ds, self.D)
        
        # The result h_interp will have length (len(h_ds) - 1) * D + 1.
        # We then truncate it to the original desired channel length (L_full + 1) 
        # and ensure the phase is preserved correctly.
        
        num_taps_full = self.L_full + 1
        
        # The first tap is correct; subsequent taps might need minor time alignment 
        # or simply truncation to the intended length.
        return h_interp[:num_taps_full]

    
    def estimateChannel(self, inData, upsample_output=False):
        # ... (Steps 1-5 for calculating h_hat_ds are the same) ...
        
        # 1. Slice and Downsample the Received Data
        rx_segment_full = inData[:self.lenPreamble]
        rx_segment_ds = decimate(rx_segment_full, self.D, ftype='fir')
        
        # 2. Slice the used portion of y (y_used)
        y_used = rx_segment_ds[self.L_ds:]
        
        # 3. Re-create X matrix TEMPORARILY for cross-correlation r_xy
        X_temp = self.make_X_matrix(self.preamble_ds, self.L_ds)
        
        # 4. Calculate the cross-correlation vector: r_xy = X^H * y
        r_xy = X_temp.conj().T @ y_used 
        del X_temp 
        
        # 5. Solve the Normal Equation (h_hat_ds is downsampled estimate)
        h_hat_ds = self.R_xx_pinv @ r_xy

        # 6. Optional Upsampling
        if upsample_output:
            # Interpolate the short h_ds vector back to the full rate
            h_hat_full = self.interpolate_h(h_hat_ds)
            return h_hat_full
        else:
            # Return the efficient, short, downsampled channel estimate
            return h_hat_ds
