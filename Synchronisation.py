from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt

def simpleDelayEstimator(modulator, signal, trueStart=0):
    preambuleMF = np.conj(modulator.getBasebandPreamble()[::-1])

    E_h = np.sum(np.abs(preambuleMF)**2)

    preambuleMF = np.conj(modulator.getBasebandPreamble()[::-1])

    E_h = np.sum(np.abs(preambuleMF)**2)


    M = len(preambuleMF)
    E_r = np.convolve(np.abs(signal)**2, np.ones(M), mode='full')
    corr = np.convolve(signal, preambuleMF, mode='full')
    preambuleMF = np.conj(modulator.getBasebandPreamble()[::-1])

    E_h = np.sum(np.abs(preambuleMF)**2)


    corr_mag = np.abs(corr)

    rho = corr_mag / np.sqrt(E_h * np.maximum(E_r, 1e-12))

    t_peak = np.argmax(rho)


    t_start = t_peak - (len(preambuleMF) - 1)


    plt.plot(rho* 100)
    plt.axvline(t_start, color='red')
    plt.axvline(trueStart, color='green')
    plt.axvline()
    plt.plot(signal.real)
    plt.show()

class TimeSynchroniser():
    def __init__(self, config, modulator):
        self.config = config
        self.preambuleMF = np.conj(modulator.getBasebandPreamble()[::-1])

        self.E_h = np.sum(np.abs(self.preambuleMF)**2)
        self.M = len(self.preambuleMF)

    def getPreambuleStartIndex(self, signal):
        #get energy of signal

        E_r = fftconvolve(np.abs(signal)**2, np.ones(self.M), mode='full')

        corr = fftconvolve(signal, self.preambuleMF, mode='full')

        corr_mag = np.abs(corr)

        rho = corr_mag / np.sqrt(self.E_h * np.maximum(E_r, 1e-12))
        #rho = corr_mag

        t_peak = np.argmax(rho)



        #check if the peak is bigger than our threshold 
        if (rho[t_peak] >= self.config['corrRatioThresh']):
            print('Detected preambule')
            t_start = t_peak - (self.M - 1)

            return t_start, rho[t_peak]
        return -1, rho[t_peak]


class MLPhaseSynchroniser():
    def __init__(self,config, modulator):
        self.config = config
        self.preambuleMF = np.conj(modulator.getBasebandPreamble()[::-1])
        self.M = len(self.preambuleMF)


    def findPhaseOffset(self, preambule):


        #we could find the phase AND the timing in only one go..;
        #TODO if perfomance is issue

        corr = np.convolve(preambule, self.preambuleMF, mode='full')

        peak_index = self.M - 1 #trusting that the synchronisation went well

        phase = np.angle(corr)
        print("Phase offset : " , phase[peak_index])
        
        return phase[peak_index]


    def synchronisePhase(self, signal):
        phaseOffset = self.findPhaseOffset(signal[:self.M])

        corrected = signal * np.exp(-1j * phaseOffset)

        return corrected

    
