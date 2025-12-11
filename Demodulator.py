
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter, firwin, fftconvolve
from scipy.signal import lfilter_zi

class Demodulator():
    def __init__(self, config, pulse, constellation):
        self.config = config
        self.pulse = pulse
        self.constellation = constellation

    def demodulateSampled(self, symbols, debug=False):

        #print(symbols)
        _, bits = self.constellation.demap(symbols)

        
        shouldGet = np.load("symbols.npy")
        start = self.config['preambleSymbols']
        #start = 0
    
        bits = bits.flatten()
        bytes_array = np.packbits(bits)
        #print(bytes(bytes_array).decode(errors='replace'), end="", flush=True)
        
        if(debug):
            print()

            plt.rcParams.update({'font.size': 18})
            plt.figure(figsize=(10, 6))
            
            # Real part subplot
            plt.subplot(2, 1, 1)
            plt.plot(shouldGet[start:].real, 'b', label="Reference (Real)")
            plt.plot(symbols.real, 'g', label="Received (Real)")
            plt.title("Real Component received vs expected")
            plt.grid(True)
            plt.legend()
            
            # Imaginary part subplot
            plt.subplot(2, 1, 2)
            plt.plot(shouldGet[start:].imag, 'c', label="Reference (Imag)")
            plt.plot(symbols.imag, 'r', label="Received (Imag)")
            plt.title("Imaginary Component received vs expected")
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.show()

        return bits



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
        print(bytes(bytes_array).decode(errors='ignore'), end="", flush=True)


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
        


        

