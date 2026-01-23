import ImageTest
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


def main(param):


    config = Common.config
    mod = Common.mod

    if len(param) == 1:
        wave = utils.generateFrequencySpan(config, mod)
        write("output.wav", config['FS'], wave.astype(np.float32))
        return
    

    if(param[1] == 'dirac'):
        common = EnvAnalysis.impulseResponseEstimator(EnvAnalysis.nbPlots, EnvAnalysis.delayDirac).Common
        config = common.config
        mod = common.mod

        wave = utils.generateDirac(config, mod, EnvAnalysis.delayDirac) 
        write("output.wav", config['FS'], wave.astype(np.float32))
        return
    


    text = param[1]

    config = Common.config
    mod = Common.mod

    if(param[1] == 'alternating'):
        wave = utils.modulateAlternatingBits(config,mod)

        write("output.wav", config['FS'], wave.astype(np.float32))
        return
    if(param[1] == "sonar"):
        #wave = utils.generateSonar(config, mod, EnvAnalysis.isiPlots, delay=0.25)
        wave = utils.generateSonar(config, mod, 40, delay=0.05)
        write("output.wav", config['FS'], wave.astype(np.float32))
        return
    
    total = []
    total2 = []
    stringToSend = text 

    bytesToSend = [ord(c) for c in stringToSend]

    if(param[1] == 'eye'):
        bytesToSend = np.random.randint(0, 255, EnvAnalysis.eyeSize) 

    if(param[1] == "image"):
        bytesToSend ,_ = ImageTest.image_to_pixel_bytes(ImageTest.path)

    if(param[1] == "ber"):
        bytesToSend ,_ = EnvAnalysis.getPhrase()

        bytesToSend = np.concatenate([bytesToSend, bytesToSend, bytesToSend, bytesToSend, bytesToSend, bytesToSend, bytesToSend,bytesToSend, bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend,bytesToSend])
        



    #partition into differnet payloads

    for i in range(0, len(bytesToSend), config['bytesPerWindow']):
        window = bytesToSend[i:config['bytesPerWindow']+i]
        window = np.array(window)

        #pad if now enough
        while(len(window) < config['bytesPerWindow']):
            window = np.append(window, 0)

        baseband, passband = mod.modulateWindow(window)

        total = np.concatenate([total,np.zeros((int(0.05 * 48000))) ,passband])
        total2 = np.concatenate([total2,baseband])



    write("output.wav", config['FS'], total.astype(np.float32))

    np.save('baseband.npy', total2)

    

if __name__ == "__main__":
    main(sys.argv)



