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

class ChannelEst:
    def __init__(self, h_taps: list, symbol_alphabet_size: int):
        self.h_taps = h_taps
        self.L = len(h_taps) - 1  # Channel memory length
        self.M = symbol_alphabet_size
        self.h0 = h_taps[0]
    
    



