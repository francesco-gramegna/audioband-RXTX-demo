import utils
from abc import ABC
import numpy as np

class Modulator(ABC):
    def __init__(self, config, pulse, constellation):
        self.config = config
        self.pulse = pulse
        self.bits_per_symbol = int(np.log2(constellation.M))

        self.preamble = utils.generatePreambleBits(config['preambleSymbols'], self.bits_per_symbol) 

        self.window = config['windowLenghtSymbols']
        self.constellation = constellation

    def getBasebandPreamble(self):
        baseband, passband = self.modulateWindow([], force=True)
        return baseband



    def modulateWindow(self, _bytes, force=False):
        fs = self.config['FS']       
        fc = self.config['FC']      
        rs = self.config['RS']     
        sps = int(fs / rs)           #samples per symbol

        bits = np.unpackbits(np.array(_bytes, dtype=np.uint8))

        if (len(bits) % self.bits_per_symbol != 0):
            #should never happen
            raise ValueError("Incorrect number of bits")
        print('total bits : ',  len(bits))

        bits = np.concatenate([self.preamble, bits])

        symbols = bits.reshape((-1, self.bits_per_symbol))

        print('total symbols ', len(symbols))

        #convert symbols bits to symbol index
        indices = symbols.dot(1 << np.arange(self.bits_per_symbol-1, -1, -1))



        if(len(indices) - self.config['preambleSymbols'] != self.window and force == False):
            raise ValueError('Incorrect number of bits per window, try padding : ' + str(len(indices) - self.config['preambleSymbols']) +  "  != " + str(self.window) )

        b = np.array(self.constellation.map(indices), dtype=np.complex128)
        #print(b)
        
        #upsample

        up = np.zeros(len(b) * sps, dtype=np.complex128)
        up[::sps] = b

        baseband = np.convolve(up, self.pulse, mode='full')

        #upconvert
        t = np.arange(len(baseband)) / fs

        passband = np.real(baseband * np.exp(2j * np.pi * fc * t))

        return baseband, passband
                


class Constellation(ABC):
    def symbol(self, value):
        pass

    def map(self, values):
        pass


class QAM(Constellation):
    def __init__(self, M):
        self.M = M
        sqrtM = int(np.sqrt(M))
        if sqrtM**2 != M:
            raise ValueError("M must be a perfect square")
        levels = np.arange(-sqrtM+1, sqrtM, 2)
        self.constellation = np.array([x + 1j*y for y in reversed(levels) for x in levels])
        self.constellation /= np.sqrt(np.mean(np.abs(self.constellation)**2))

    def symbol(self, index):
        """Map integer 0..M-1 to constellation point."""
        return self.constellation[index]

    def map(self, values):
        return [self.constellation[v] for v in values]



