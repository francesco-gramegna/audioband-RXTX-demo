import utils
import Synchronisation
import Equalisation
import TimeReceiver
from commons import Common
import sys
import matplotlib.pyplot as plt
import AudioReceiver
from scipy.signal import welch
import numpy as np
from scipy.signal import savgol_filter , fftconvolve


class Receiver():
    def __init__(self, config, mod, demod):
        self.config = config
        self.mod = mod
        self.demod = demod
        treceiver = TimeReceiver.TimeReceiver(config, mod, demod, True, self.processPayload)
        self.rcv =  AudioReceiver.AudioReceiver(self.config, treceiver) 
        self.phaseSynchroniser = Synchronisation.MLPhaseSynchroniser(config, mod)
        self.MLAmplitudeSync = Synchronisation.MLAmplitudeSync(config, mod)

        self.channelEst = Equalisation.ChannelEstimator(config, mod)

        self.channelEq = Equalisation.MMSEEqualizer(config, mod)

        self.pulseMF = self.mod.pulse.conj()[::-1]

    def processPayload(self, data):
        print("Got one")

        phaseSync = self.phaseSynchroniser.synchronisePhase(data)
        corr = fftconvolve(phaseSync, self.pulseMF)
        corr = corr[len(self.pulseMF) -1 :]

        corr = corr[::self.config["samplesPerSymbol"]]
        #estimate the channel

        #corr = self.MLAmplitudeSync.synchroniseAmplSymbols(corr) #todo noise too much amplified

        try:
            p,n0 = self.channelEst.estimateChannel(corr)
            print("n0 : ", n0)
        except:
            print("Timing error")
            return
        
        y,w  = self.channelEq.equalize(corr, p, n0)

        csym = corr
        #csym =  corr[self.config['preambleSymbols']:]
        #csym = csym[:self.config['windowLenghtSymbols']]

        #ysym = y[self.channelEq.delta:]

        ysym =  y[self.channelEq.delta + self.config['preambleSymbols']:]
        ysym = ysym[:self.config['bytesPerWindow']]
        print(len(ysym))
        print(self.config['bytesPerWindow'])

        #self.demod.demodulateSampled(csym)
        self.demod.demodulateSampled(ysym)
        #print(utils.generatePreambleBits(self.config['preambleSymbols'] * 2, 2))

        #plt.plot(csym, 'g')
        #plt.plot(ysym, 'r')
        #plt.show()


if __name__ == "__main__":
    rcv = Receiver(Common.config, Common.mod, Common.demod)
    rcv.rcv.listen()

