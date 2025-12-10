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
    def __init__(self, config, mod, demod, callback):
        self.config = config
        self.mod = mod
        self.demod = demod

        treceiver = TimeReceiver.TimeReceiver(config, mod, demod, True, self.processPayload)
        self.rcv =  AudioReceiver.AudioReceiver(self.config, treceiver) 

        self.callback = callback
        self.phaseSynchroniser = Synchronisation.MLPhaseSynchroniser(config, mod)
        self.MLAmplitudeSync = Synchronisation.MLAmplitudeSync(config, mod)

        self.bestSamplingSync = Synchronisation.SymbolTimingSynchroniser(config, Common.mod)

        self.pll1 = Synchronisation.PLL(config, Common.mod, 0.05)
        self.pll2 = Synchronisation.PLL(config, Common.mod, 0.5)
        self.pll3 = Synchronisation.PLL(config, Common.mod, 1)
        self.pll4 = Synchronisation.PLL(config, Common.mod, 3)

        self.channelEst = Equalisation.ChannelEstimator(config, mod)

        self.channelEq = Equalisation.MMSEEqualizer(config, mod)

        self.pulseMF = self.mod.pulse.conj()[::-1]

    def processPayload(self, data):
        #print("Got one")

        phaseSync = self.phaseSynchroniser.synchronisePhase(data)
        corr = fftconvolve(phaseSync, self.pulseMF)
        corr = corr[len(self.pulseMF) -1 :]
        
        #best_phase , _ = self.bestSamplingSync.findOptimalSamplingPhase(corr)
        #print("Best phase : ", best_phase)
        #if best_phase > self.config["samplesPerSymbol"]:
        #    best_phase = 0

        best_phase = 0

        corr = corr[best_phase::self.config["samplesPerSymbol"]]
        #estimate the channel

        #normalisation
        sig_power = np.mean(np.abs(corr)**2)
        if sig_power > 0:
            corr = corr / np.sqrt(sig_power)

        
        corr1 = self.pll1.syncPhase(corr)
        corr2 = self.pll2.syncPhase(corr)
        corr3 = self.pll3.syncPhase(corr)
        corr4 = self.pll4.syncPhase(corr)

        #try:
            #p,n0 = self.channelEst.estimateChannel(corr)
            #print("n0 : ", n0)
        #except Exception as e:
            #print("Timing error : ", e)
        #    return
        

        #corr = self.MLAmplitudeSync.synchroniseAmplSymbols(corr) #todo noise too much amplified

        #y,w  = self.channelEq.equalize(corr, p, n0)
        #y phase is not sync anymore. we have to resynchronise it
                
        #ds_phase = self.channelEq.delta % 4 
        
        #y = y[ds_phase::4]      
        #corr = corr[::4]

        """
        y_pre = y[self.channelEq.delta : self.channelEq.delta + len(self.channelEst.preambleSymbols)]

        phase_diff = y_pre * np.conj(self.channelEst.preambleSymbols)
        phase_err = np.angle(np.mean(phase_diff))

        #print("new Phase err ", phase_err)

        y = y * np.exp(-1j * phase_err)

        csym = corr
        csym =  corr[self.config['preambleSymbols']:]
        csym = csym[:self.config['windowLenghtSymbols']]

        #ysym = y[self.channelEq.delta:]
        ysym = y
        """

        #ysym =  y[self.channelEq.delta + self.config['preambleSymbols']:]
        #ysym = ysym[::-1][:self.config['windowLenghtSymbols']][::-1]

        pre_len = self.config['preambleSymbols']
        window = self.config['windowLenghtSymbols']

        csym = corr[pre_len: pre_len + window]
        csym1 = corr1[pre_len: pre_len + window]
        csym2 = corr2[pre_len: pre_len + window]
        csym3 = corr3[pre_len: pre_len + window]
        csym4 = corr4[pre_len: pre_len + window]

        #delta = self.channelEq.delta
        #ysym = y[delta  : delta + window]
        #ysym = y[pre_len: pre_len+window]

        #plt.plot(csym, 'b')
        #plt.plot(ysym, 'r')
        #plt.show()

        #print(len(ysym))
        #print(self.config['bytesPerWindow'])

        bits = self.demod.demodulateSampled(csym1)
        bits = self.demod.demodulateSampled(csym2)
        bits = self.demod.demodulateSampled(csym3)
        bits = self.demod.demodulateSampled(csym4)

        
        self.callback(bits)

        #self.demod.demodulateSampled(ys17000ym)

        #print(utils.generatePreambleBits(self.config['preambleSymbols'] * 2, 2))

        #plt.plot(csym, 'g')
        #plt.plot(ysym, 'r')
        #plt.show()


if __name__ == "__main__":
    rcv = Receiver(Common.config, Common.mod, Common.demod, lambda x: x)
    rcv.rcv.listen()



