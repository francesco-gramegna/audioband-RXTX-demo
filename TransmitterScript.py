import utils
import sys
from commons import Common
from scipy.io.wavfile import write
import Demodulator
import Receiver
import sounddevice as sd
import numpy as np
import Synchronisation
import matplotlib.pyplot as plt
import mathUtils
import Modulator
import plots


def main():
    
    config = Common.config
    mod = Common.mod

    if len(sys.argv) == 1:
        wave = utils.generateFrequencySpan(config, mod)
        write("output.wav", config['FS'], wave.astype(np.float32))
        return
    

    if(sys.argv[1] == 'dirac'):
        wave = utils.generateDirac(config)
        write("output.wav", config['FS'], wave.astype(np.float32))
        return

    text = sys.argv[1]

    config = Common.config
    mod = Common.mod

    total = []
    stringToSend = text 
    bytesToSend = [ord(c) for c in stringToSend]
    #partition into differnet payloads

    for i in range(0, len(bytesToSend), config['bytesPerWindow']):
        window = bytesToSend[i:config['bytesPerWindow']+i]

        #pad if now enough
        while(len(window) < config['bytesPerWindow']):
            window += [0]

        _, passband = mod.modulateWindow(window)

        total = np.append(total,passband)



    write("output.wav", config['FS'], total.astype(np.float32))

    

main()
