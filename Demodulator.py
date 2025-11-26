
import matplotlib.pyplot as plt
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
        
        mf_delay = (len(pulseMF) - 1)

        conv = fftconvolve(signal, pulseMF)
        symbols = conv[mf_delay::self.config['samplesPerSymbol']]

        

        #print(" Got " , len(symbols) , "symbols")

        trueSymbols = np.load("symbols.npy")

        trueSymbols = trueSymbols[self.config['preambleSymbols']:]

        symbols = symbols * 10
        symbols = symbols[self.config['preambleSymbols']:][: self.config['windowLenghtSymbols']]

        #pick the closest to the constellation 

        _, bits = self.constellation.demap(symbols)

        bits = bits.flatten()

        bytes_array = np.packbits(bits)
        print(bytes(bytes_array).decode(errors='ignore'))


        """
        plt.plot(symbols.real, 'b')
        plt.plot(symbols.imag, 'r')

        plt.plot(trueSymbols.real, c='g')
        plt.plot(trueSymbols.imag, c='c')

        #plt.scatter(symbols.real, symbols.imag, c='r')
        #plt.scatter(trueSymbols.real, trueSymbols.imag, c='g')
        plt.grid(True)
        plt.show()
        """


        return bits
        


        

