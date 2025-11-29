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

        while(True):
            t_peak = np.argmax(rho)
            if(t_peak - (self.M - 1) < 0):
                #previous peak , we skip it
                rho[t_peak] = 0
                continue
            break
            


        #check if the peak is bigger than our threshold 
        if (rho[t_peak] >= self.config['corrRatioThresh']):

            # now take all of those who are 5 % similar  to the max of them (arbitrary)
            max_peak_index =  t_peak
            max_peak_value = rho[max_peak_index]
            potential_starts = []
            tempRho = rho.copy()
            while(len(potential_starts) < len(tempRho)):
        
                t_peak = np.argmax(tempRho)
                if(t_peak - (self.M - 1) < 0):
                    #previous peak , we skip it
                    tempRho[t_peak] = 0
                    continue

                if(tempRho[t_peak] < 0.95 * max_peak_value):
                    break
        
                potential_starts.append(t_peak)
                tempRho[t_peak] = 0
        
            t_peak = min(potential_starts)
                    
            #print('Detected preambule')
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
        #print("Phase offset : " , phase[peak_index])
        
        return phase[peak_index]


    def synchronisePhase(self, signal):
        phaseOffset = self.findPhaseOffset(signal[:self.M])

        corrected = signal * np.exp(-1j * phaseOffset)

        return corrected

    
class MLFreqPhaseSynchroniser():
    def __init__(self, config, modulator):
        self.config = config
        self.preamble = modulator.getBasebandPreamble()
        self.M = len(self.preamble)

    def estimate_freq(self, seg):
        return np.angle(np.sum(seg * np.conj(self.preamble))) / self.M

    def estimate_phase(self, seg):
        return np.angle(np.sum(seg * np.conj(self.preamble)))

    def synchronise(self, signal):
        seg = signal[:self.M]
        f = self.estimate_freq(seg)                     # ML freq in rads
        n = np.arange(len(signal))
        sig_f_corr = signal * np.exp(-1j * f * n)       # remove linear phase
        phase = self.estimate_phase(sig_f_corr[:self.M])# residual phase
        corrected = sig_f_corr * np.exp(-1j * phase)    # remove phase
        return corrected

