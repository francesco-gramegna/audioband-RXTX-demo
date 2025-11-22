import sounddevice as sd
import numpy as np
import Syncronisation
import matplotlib.pyplot as plt
import mathUtils
import Modulator
import plots


payload = "Hello world!"
payload = [ord(c) for c in payload]

config = {'FS' : 48000,
          'FC' : 900,
          'RS' : 100,
          'preambleSymbols' : 10,
          'windowLenghtSymbols' : 24,
          'corrRatioThresh' : 0.3, #very very low snr
          }


config['samplesPerSymbol'] = config['FS'] // config['RS']

config['payloadSamples'] = (config['preambleSymbols'] + config['windowLenghtSymbols']) * config['samplesPerSymbol'] 



constellation = Modulator.QAM(16)

pulse = mathUtils.rrc_pulse(config['FS'], config['RS'], alpha=0.25)


mod = Modulator.Modulator(config, pulse, constellation)

baseband, passband = mod.modulateWindow(payload)

preambule = mod.getBasebandPreamble()

signal = np.concatenate([ [0] * config['FS'],  preambule, [0] * config['FS'] * 2], dtype=np.complex128)

signal = signal + np.random.normal(loc=0, scale=0.05, size=len(signal)) + 1j * np.random.normal(loc=0, scale=0.05, size=len(signal))

Syncronisation.simpleDelayEstimator(mod, signal, trueStart= config['FS'] )



#plt.plot(passband)
#plt.show()

#sd.play(passband, config['FS'])
#sd.wait()



