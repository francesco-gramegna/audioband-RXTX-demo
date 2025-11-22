
import numpy as np

def generatePreambleBits(nbSymbols, bitsPerSymbol):
    rng = np.random.default_rng(seed=81731)
    totalBits = nbSymbols * bitsPerSymbol
    preambleBits = rng.integers(0, 2, size=totalBits, dtype=np.uint8)
    return preambleBits




