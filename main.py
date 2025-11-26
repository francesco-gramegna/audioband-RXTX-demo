import Demodulator
import Receiver
import sounddevice as sd
import numpy as np
import Synchronisation
import matplotlib.pyplot as plt
import mathUtils
import Modulator
import plots



if __name__ == "__main__":

    payload = "Hello world!"
    payload = [ord(c) for c in payload]
    
    
    config = {'FS' : 48000,
              'FC' : 500,
              'RS' : 20,
              'preambleSymbols' : 30,
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
    
    
    
    constellation = Modulator.QAM(16)
    
    
    
    mod = Modulator.Modulator(config, pulse, constellation)
    demod = Demodulator.Demodulator(config, pulse, constellation)
    """
    
    baseband, passband = mod.modulateWindow(payload)
    
    downed = mod.downConvert(passband)
    
    #plt.plot(baseband, 'g')
    #plt.plot(downed, 'r')
    #plt.show()
    
    preambule = mod.getBasebandPreamble()
    
    energyPreambule = np.sum(np.abs(preambule)**2)
    
    N0 = 0.1
    SNR = energyPreambule / N0
    
    SNRdb = 10 * np.log10(SNR)
    
    print("SNR : ", SNR, " db.")
    
    rcv = Receiver.SimpleReceiver(config, mod, demod)
    print(len(passband))
    print(config['payloadSamples'])
    
    plt.plot(baseband, 'g')
    signal = passband
    
    signal = signal * np.exp(1j * np.pi * 1.3)
    
    
    signal = np.concatenate([ [0] * config['FS'],  signal, [0] * config['FS'] * 2], dtype=np.complex128)
    
    
    signal = signal + np.random.normal(loc=0, scale=N0/2, size=len(signal)) + 1j * np.random.normal(loc=0, scale=N0/2, size=len(signal))
    
    #for i in range(0, len(signal), config['payloadSamples']):
        #print("pusing data")
        #plt.plot(signal)
        #plt.show()
        #rcv.pushWindow(signal[:config['payloadSamples']])
        #signal = signal[config['payloadSamples']:]
    
    
    #sd.play(signal.real, config['FS'])
    #sd.wait()
    """
    
    
    #rcv.listen()
    

    rcv = Receiver.SimpleReceiver(config, mod, demod)
    realRCV = Receiver.AudioReceiver(config, rcv)
    
    realRCV.listen()
    
    
    
    
