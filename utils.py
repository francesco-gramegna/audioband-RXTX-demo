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

    total_samples = config['FS'] * 60

    t = np.arange(total_samples) / config['FS']
    sine_wave = chirp(t, f0=0, f1=24000, t1=60, method='linear')

    return np.concatenate([np.zeros(48000 * 3), pre,sine_wave])


def generateDirac(config, mod, delayDirac):
    pre = mod.getPassbandPreamble()

    pre = pre 

    total_samples = int(config['FS'] * 3)
    signal = np.zeros(total_samples, dtype=np.float32)

    signal[delayDirac] = 1.0  # loudest possible non-clipping value

    return np.concatenate([pre, signal])



 
def generateSonar(config, mod, cycles, delay):
    delay = 2
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


def modulateAlternatingBits(config, mod):

    bits = [1,0,1,0,1,0,1,0]
    byte_value = int("".join(map(str, bits)), 2)
    bits = [byte_value] * config['bytesPerWindow']
    _, ret = mod.modulateWindow(bits)

    return ret




#ai generated
def generate_zc_4qam_preamble(N, qam_constellation, root=1):
    """
    Generate a Zadoff–Chu-based 4-QAM preamble with 2 bits per symbol.
    
    Output:
        - bits: preamble bit sequence (shape: (N*2,))
        - indices: constellation symbol indices (length N)
        - symbols: complex QAM symbols matching the constellation
    """

    # ---- 1) Check validity of ZC root ----
    #if gcd(root, N) != 1:
    #    raise ValueError(f"Invalid ZC root={root}, gcd(root,N)={gcd(root,N)}. "
    #                     "Pick root coprime with N.")

    # ---- 2) Generate the Zadoff–Chu sequence ----
    n = np.arange(N)
    if N % 2 == 0:
        zc = np.exp(-1j * np.pi * root * n * (n + 1) / N)
    else:
        zc = np.exp(-1j * np.pi * root * n * n / N)

    # ---- 3) Quantize phases to nearest 4QAM constellation symbol ----
    const = qam_constellation.constellation  # complex points

    # find nearest constellation symbol by Euclidean distance
    distances = np.abs(zc[:, None] - const[None, :])
    indices = np.argmin(distances, axis=1)

    # ---- 4) Get QAM symbols & bits back ----
    symbols = np.array(qam_constellation.map(indices))

    # convert symbols → bits using your demapper
    _, bits = qam_constellation.demap(symbols, return_bits=True)

    bits = bits.reshape(-1)  # flatten to 1D

    return bits, indices, symbols

