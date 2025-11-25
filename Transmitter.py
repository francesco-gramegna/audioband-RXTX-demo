import Demodulator
import Receiver
import sounddevice as sd
import numpy as np
import Synchronisation
import matplotlib.pyplot as plt
import mathUtils
import Modulator
import plots

config = {'FS' : 48000,
          'FC' : 800,
          'RS' : 400,
          'preambleSymbols' : 10,
          'windowLenghtSymbols' : 30,
          'corrRatioThresh' : 0.3, #very very low snr
          'excessBandwidth': 0.25,
          'lpCutoffEpsilon': 0.05,
          'bitsPerSymbol' : 4
          }

pulse = mathUtils.rrc_pulse(config['FS'], config['RS'], alpha=0.25)

config['bytesPerWindow'] = config['windowLenghtSymbols'] * config['bitsPerSymbol'] // 8

#auto definitions
config['samplesPerSymbol'] = config['FS'] // config['RS']

config['payloadSamples'] = (config['preambleSymbols'] + config['windowLenghtSymbols']) * config['samplesPerSymbol'] +  len(pulse) - 1

config['Bmin'] = config['RS']

config['bandwidth'] = (1 + config['excessBandwidth'])*config['Bmin']


def main():

    print(sd.query_devices())
    sd.default.device = (None, 8)  
    constellation = Modulator.QAM(16)
    mod = Modulator.Modulator(config, pulse, constellation)

    while (True):
        total = []
        stringToSend = input("Message : ")
        bytesToSend = [ord(c) for c in stringToSend]
        #partition into differnet payloads

        for i in range(0, len(bytesToSend), config['bytesPerWindow']):
            window = bytesToSend[i:config['bytesPerWindow']+i]

            #pad if now enough
            while(len(window) < config['bytesPerWindow']):
                window += [0]

            _, passband = mod.modulateWindow(window)

            total = np.append(total,passband)

        sd.play(total)
        sd.wait()
    

main()
