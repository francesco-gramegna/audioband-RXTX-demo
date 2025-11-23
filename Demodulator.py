import numpy as np
from scipy.signal import lfilter, firwin, fftconvolve
from scipy.signal import lfilter_zi

class Demodulator():
    def __init__(self, config, pulse, constellation):
        self.config = config
        self.pulse = pulse
        self.constellation = constellation


    def demodulate(self, signal):
        #we demodulate : 
        pulseMF = self.pulse[::-1]
        
        conv = np.convolve(signal, pulseMF)

        symbols = conv[:len(signal):self.config['samplesPerSymbol']]        

        #pick the closest to the constellation 

        _, bits = self.constellation.demap(symbols)

        bits = bits.flatten()
        bytes_array = np.packbits(bits)
        print(bytes(bytes_array).decode(errors='ignore'))

        return bits
        


        

