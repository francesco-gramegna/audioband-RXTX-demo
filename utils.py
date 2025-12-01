import matplotlib.pyplot as plt
from scipy.signal import chirp
import numpy as np

def generatePreambleBits(nbSymbols, bitsPerSymbol):
    rng = np.random.default_rng(seed=81731)
    totalBits = nbSymbols * bitsPerSymbol
    preambleBits = rng.integers(0, 2, size=totalBits, dtype=np.uint8)
    return preambleBits



def generateFrequencySpan(config, mod):
    #we need a modulator for the preambule
    
    pre = mod.getPassbandPreamble()

    total_samples = config['FS']/2 * 60

    t = np.arange(total_samples) / config['FS']
    sine_wave = chirp(t, f0=0, f1=24000, t1=60, method='linear')

    return np.concatenate([pre,sine_wave])


def generateDirac(config, mod, delayDirac):
    pre = mod.getPassbandPreamble()

    pre = pre 

    total_samples = int(config['FS'] * 3)
    signal = np.zeros(total_samples, dtype=np.float32)

    signal[delayDirac] = 1.0  # loudest possible non-clipping value

    return np.concatenate([pre, signal])



 
def generateSonar(config, mod, cycles, delay):
    delay = 5
    _bytes = [0] * config['bytesPerWindow']

    _, pre = mod.modulateWindow(_bytes)
    
    total = np.zeros(delay*config['FS'])
    delayArr = np.zeros(delay * config['FS']//2)

    #audible warning sound
    t = np.arange(config['FS'] * 0.25)
    sound = np.sin(2 * np.pi * 400 / config['FS'] * t) * 0.005
       
    for c in range(cycles):
        total = np.concatenate([total, sound, delayArr, pre ,delayArr])
        
    return total 



