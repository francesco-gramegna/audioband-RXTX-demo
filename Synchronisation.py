import utils
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
        
        while(True):
            t_peak = np.argmax(rho)
            if(t_peak - (self.M - 1) < 0):
                #previous peak , we skip it
                rho[t_peak] = 0
                continue
            break
            
        #check if the peak is bigger than our threshold 
        if (rho[t_peak] >= self.config['corrRatioThresh']):
            
            tempRho = rho.copy()
            
            while(True):
                # Mask the area around the current peak
                start_mask = max(0, t_peak - self.config['payloadSamples'])
                end_mask = min(len(tempRho), t_peak + self.config['payloadSamples'])
                tempRho[start_mask : end_mask] = 0
                
                new_peak = np.argmax(tempRho)
                
                # Check threshold
                if tempRho[new_peak] < self.config['corrRatioThresh']:
                    break
                    
                # Check valid index
                if(new_peak - (self.M - 1) < 0):
                    tempRho[new_peak] = 0
                    continue

                # If peak is to the right, mask it and search again
                if new_peak > t_peak:
                    start_r = max(0, new_peak - self.config['payloadSamples'])
                    end_r = min(len(tempRho), new_peak + self['payloadSamples'])
                    tempRho[start_r : end_r] = 0
                    continue

                # Found valid peak to the left
                t_peak = new_peak
                
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


class MLAmplitudeSync():
    def __init__(self,config,mod):
        self.config = config
        self.mod = mod
        bits = utils.generatePreambleBits(config['preambleSymbols'], config['bitsPerSymbol'])
        symbols = bits.reshape((-1, config['bitsPerSymbol']))
        indices = symbols.dot(1 << np.arange(config['bitsPerSymbol']-1, -1, -1))

        self.preambleSymbols = np.array(mod.constellation.map(indices))

        self.denom = np.sum(np.abs(self.preambleSymbols)**2) / len(self.preambleSymbols)

        
    def estimateAmpl(self, data):
        data = data[:len(self.preambleSymbols)]
        A = np.sum(data * self.preambleSymbols.conj()) / len(self.preambleSymbols)

        A = A/ self.denom

        return 1/A

    def synchroniseAmplSymbols(self, data):
        A = self.estimateAmpl(data)
        return A * data





    
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


#ai gen
class SymbolTimingSynchroniser():
    def __init__(self, config, modulator):
        self.config = config
        self.sps = config['samplesPerSymbol']
        
        # Get preamble symbols for timing metric
        import utils
        bits, _, symbols = utils.generate_zc_4qam_preamble(config['preambleSymbols'], 
                                         modulator.constellation)

        #symbols = bits.reshape((-1, config['bitsPerSymbol']))
        #indices = symbols.dot(1 << np.arange(config['bitsPerSymbol']-1, -1, -1))
        #self.preamble_symbols = np.array(modulator.constellation.map(indices))

        self.preamble_symbols = symbols
    
   
    def findOptimalSamplingPhase(self, mf_output):
        """
        Find the best sampling phase (0 to sps-1) for symbol decisions.
        
        Uses Mueller-Muller style timing metric or energy maximization.
        
        Args:
            mf_output: Matched filter output (continuous samples)
        
        Returns:
            best_phase: Optimal sampling offset (0 to sps-1)
            metric_values: Timing metric for each phase (for debugging)
        """
        method = self.config.get('timingMetric', 'energy')  # or 'mm', 'early_late'
        
        if method == 'energy':
            return self._energy_maximization(mf_output)
        elif method == 'mm':
            return self._mueller_muller(mf_output)
        else:
            return self._energy_maximization(mf_output)
    
    
    def _energy_maximization(self, mf_output):
        """
        Simple and effective: pick the phase that maximizes symbol energy.
        """
        n_symbols = len(self.preamble_symbols)
        
        max_energy = -np.inf
        best_phase = 0
        metric_values = np.zeros(self.sps)
        
        for phase in range(self.sps):
            # Sample at this phase
            samples = mf_output[phase::self.sps][:n_symbols]
            
            if len(samples) < n_symbols:
                continue
            
            # Compute energy
            energy = np.sum(np.abs(samples)**2)
            metric_values[phase] = energy
            
            if energy > max_energy:
                max_energy = energy
                best_phase = phase
        
        return best_phase, metric_values
    
    def _mueller_muller(self, mf_output):
        """
        Mueller-Muller timing error detector.
        More sophisticated, works better with random data.
        """
        n_symbols = len(self.preamble_symbols)
        
        min_error = np.inf
        best_phase = 0
        metric_values = np.zeros(self.sps)
        
        for phase in range(self.sps):
            samples = mf_output[phase::self.sps][:n_symbols]
            
            if len(samples) < n_symbols - 1:
                continue
            
            # MM timing error: real(conj(y[n]) * y[n-1])
            # Should be zero at optimal sampling
            timing_error = 0
            for i in range(1, len(samples)):
                timing_error += np.abs(np.real(
                    samples[i] * np.conj(samples[i-1])
                ))
            
            metric_values[phase] = timing_error
            
            if timing_error < min_error:
                min_error = timing_error
                best_phase = phase
        
        return best_phase, metric_values
