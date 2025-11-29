import Demodulator
import sounddevice as sd
import numpy as np
import Synchronisation
import matplotlib.pyplot as plt
import mathUtils
import Modulator
import plots


class Common():

    config = {'FS' : 48000,
              'FC' : 800,
              'RS' : 50,
              'preambleSymbols' : 5,
              'windowLenghtSymbols' : 100,
              'corrRatioThresh' : 0.45, #very very low snr
              'excessBandwidth': 0.50,
              'lpCutoffEpsilon': 0.05,
              'bitsPerSymbol' : 2,
              'Eb': 400
              }
    
    pulse = mathUtils.rrc_pulse(config['FS'], config['RS'], alpha=0.25)
    
    config['bytesPerWindow'] = config['windowLenghtSymbols'] * config['bitsPerSymbol'] // 8
    
    #auto definitions
    config['samplesPerSymbol'] = config['FS'] // config['RS']
    
    config['payloadSamples'] = (config['preambleSymbols'] + config['windowLenghtSymbols']) * config['samplesPerSymbol'] #+  len(pulse) - 1
    
    config['Bmin'] = config['RS']
    
    config['bandwidth'] = (1 + config['excessBandwidth'])*config['Bmin']
    
    constellation = Modulator.QAM(4, config['Eb'])
    
    mod = Modulator.Modulator(config, pulse, constellation)
    demod = Demodulator.Demodulator(config, pulse, constellation)
   

print("Using : ", Common.config)
