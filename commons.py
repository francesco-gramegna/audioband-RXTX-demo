import math
import Demodulator
import sounddevice as sd
import numpy as np
import Synchronisation
import matplotlib.pyplot as plt
import mathUtils
import Modulator
import plots


class CommonDynamic():
    def __init__(self, config):
        self.config = config

        self.pulse = mathUtils.rrc_pulse(config['FS'], config['RS'], alpha=0.25)
    
        self.config['bytesPerWindow'] = config['windowLenghtSymbols'] * config['bitsPerSymbol'] // 8
    
        #auto definitions
        self.config['samplesPerSymbol'] = config['FS'] // config['RS']
    
        self.config['payloadSamples'] = (config['preambleSymbols'] + config['windowLenghtSymbols']) * config['samplesPerSymbol']  +  len(self.pulse) - 1
    
        self.config['Bmin'] = config['RS']
    
        self.config['bandwidth'] = (1 + config['excessBandwidth'])*config['Bmin']
    
        self.constellation = Modulator.QAM(4, config['Eb'])
    
        self.mod = Modulator.Modulator(self.config, self.pulse, self.constellation)
        self.demod = Demodulator.Demodulator(self.config, self.pulse, self.constellation)
   


class Common():

    config = {'FS' : 48000,
              'FC' : 19000,
              'RS' : 100,
              'preambleSymbols' : 25,
              'windowLenghtSymbols' : 4,
              'corrRatioThresh' : 0.80, #very  ery low snr
              'excessBandwidth': 0.50,
              'lpCutoffEpsilon': 0.05,
              'bitsPerSymbol' : 2,
              'Eb': 1,
              'channelTMax' : 0.05 #in seconds
              }

    #config['channelSymbolsLen'] =  math.ceil(config['channelTMax'] * config['RS']) + 1
    config['channelSymbolsLen'] = 15

    
    pulse = mathUtils.rrc_pulse(config['FS'], config['RS'], alpha=0.25)
    
    config['bytesPerWindow'] = config['windowLenghtSymbols'] * config['bitsPerSymbol'] // 8
    
    #auto definitions
    config['samplesPerSymbol'] = config['FS'] // config['RS']
    
    config['payloadSamples'] = (config['preambleSymbols'] + config['windowLenghtSymbols']) * config['samplesPerSymbol']  +  len(pulse) - 1
    
    config['Bmin'] = config['RS']
    
    config['bandwidth'] = (1 + config['excessBandwidth'])*config['Bmin']
    
    constellation = Modulator.QAM(4, config['Eb'])
    
    mod = Modulator.Modulator(config, pulse, constellation)
    demod = Demodulator.Demodulator(config, pulse, constellation)
   

print("Using : ", Common.config)
