import EnvAnalysis
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
        common = EnvAnalysis.impulseResponseEstimator(EnvAnalysis.nbPlots, EnvAnalysis.delayDirac).Common
        config = common.config
        mod = common.mod

        wave = utils.generateDirac(config, mod, EnvAnalysis.delayDirac) 
        write("output.wav", config['FS'], wave.astype(np.float32))
        return
    


    text = sys.argv[1]

    config = Common.config
    mod = Common.mod

    if(sys.argv[1] == 'alternating'):
        wave = utils.modulateAlternatingBits(config,mod)

        write("output.wav", config['FS'], wave.astype(np.float32))
        return
    if(sys.argv[1] == "sonar"):
        wave = utils.generateSonar(config, mod, EnvAnalysis.isiPlots, delay=0.5)
        write("output.wav", config['FS'], wave.astype(np.float32))

        return

    total = []
    stringToSend = text 

    bytesToSend = [ord(c) for c in stringToSend]

    if(sys.argv[1] == 'eye'):
        bytesToSend = np.random.bytes(EnvAnalysis.eyeSize) 

    #partition into differnet payloads

    for i in range(0, len(bytesToSend), config['bytesPerWindow']):
        window = bytesToSend[i:config['bytesPerWindow']+i]

        #pad if now enough
        while(len(window) < config['bytesPerWindow']):
            window += [0]

        _, passband = mod.modulateWindow(window)

        total = np.concatenate([total,np.zeros((int(0.05 * 48000))) ,passband])



    write("output.wav", config['FS'], total.astype(np.float32))

    

main()
