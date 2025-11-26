import math
import utils
from abc import ABC
import numpy as np
from scipy.signal import lfilter, firwin, fftconvolve
from scipy.signal import lfilter_zi



class Modulator():
    def __init__(self, config, pulse, constellation):
        self.config = config
        self.pulse = pulse
        self.bits_per_symbol = int(np.log2(constellation.M))

        self.preamble = utils.generatePreambleBits(config['preambleSymbols'], self.bits_per_symbol) 

        self.window = config['windowLenghtSymbols']
        self.constellation = constellation


        cutoff = (1 + self.config['lpCutoffEpsilon']) * self.config['bandwidth']

        self.numtaps=129
        self.taps = firwin(self.numtaps, cutoff / (config['FS'] / 2))

        #for the downconverter
        self.lpf_state = lfilter_zi(self.taps, 1) * 0


    def getBasebandPreamble(self):
        baseband, passband = self.modulateWindow([], force=True)
        return baseband

   
    def downConvert(self, data): #odd number of taps 

        fs = self.config['FS']
        fc = self.config['FC']

        t = np.arange(len(data)) / fs
        sb = data * np.exp(-2j * np.pi * fc * t)
        # low pass filter

        filtered, self.lpf_state = lfilter(self.taps, 1, sb, zi=self.lpf_state)
        #filtered = fftconvolve(sb, self.taps, mode='full')

        # delay compensation
        #delay = (self.numtaps - 1) // 2
        #filtered = filtered[delay:]

        return  filtered * np.sqrt(2)




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

        #save the symbols to a file for testing purposes
        np.save("symbols", b)

        #print(b)
        
        #upsample

        up = np.zeros(len(b) * sps, dtype=np.complex128)
        up[::sps] = b

        baseband = np.convolve(up, self.pulse, mode='full')

        #upconvert
        t = np.arange(len(baseband)) / fs

        passband = np.sqrt(2) *  np.real(baseband * np.exp(2j * np.pi * fc * t))

        return baseband, passband
                


class Constellation(ABC):
    def symbol(self, value):
        pass

    def map(self, values):
        pass



#ai generated
class QAM(Constellation):
    def __init__(self, M, Eb=None):
        self.M = M
        self.bps = int(np.log2(M))

        sqrtM = int(np.sqrt(M))
        if sqrtM**2 != M:
            raise ValueError("M must be a perfect square")

        levels = np.arange(-sqrtM+1, sqrtM, 2)
        self.constellation = np.array([x + 1j*y for y in reversed(levels) for x in levels])

        # Normalize to Es = 1
        self.constellation /= np.sqrt(np.mean(np.abs(self.constellation)**2))

        # Set Eb (default Eb = 1/bps)
        self.Eb = (1 / self.bps) if Eb is None else Eb

        # Scale so Es = Eb * log2(M)
        self.constellation *= np.sqrt(self.Eb * self.bps)
    def symbol(self, index):
        """Map integer 0..M-1 to constellation point."""
        return self.constellation[index]

    def map(self, values):
        return [self.constellation[v] for v in values]


    
    def demap(self, values, return_bits=True, return_bytes=False):
        """
        Demodulate received symbols to indices, bits, or byte string.
        """
        values = np.atleast_1d(values)
        # closest constellation index
        idx = np.argmin(np.abs(values[:, None] - self.constellation[None, :])**2, axis=1)
    
        if not (return_bits or return_bytes):
            return idx
    
        # bits per symbol
        bps = int(np.log2(self.M))
        bits = ((idx[:, None] & (1 << np.arange(bps)[::-1])) > 0).astype(int)
    
        if return_bytes:
            bits_flat = bits.flatten()
            pad = (-len(bits_flat)) % 8
            if pad > 0:
                bits_flat = np.concatenate([bits_flat, np.zeros(pad, dtype=int)])
            byte_str = bytes(np.packbits(bits_flat)).decode(errors='ignore')
            return idx, bits, byte_str
    
        return idx, bits



