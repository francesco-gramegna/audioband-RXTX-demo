import Synchronisation
import Equalisation
import TimeReceiver
import commons
import sys
import matplotlib.pyplot as plt
import AudioReceiver
from scipy.signal import welch
import numpy as np
from scipy.signal import savgol_filter , fftconvolve


class PSDVisualizer():
    def __init__(self):
        self.config = {'FS':48000,
                  'payloadSamples':24000}

        self.snrEstimator = snrEstimator(self.config) 

        plt.ion()   # enable interactive mode
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])  # empty line placeholder

        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("PSD (dB/Hz)")
        self.ax.set_title("Noise PSD (Live)")
        self.ax.grid(True)

        self.rcv =  AudioReceiver.AudioReceiver(self.config, self)
    
    def pushWindow(self, data):
        self.snrEstimator.pushData(data, noise=True)

        fNoise , pXNoise = self.snrEstimator.getNoisePSD()

        #get total power

        df = fNoise[1] - fNoise[0]    

        total_power = np.sum(pXNoise * df)  # integrate PSD

        total_db = 10*np.log10(total_power + 1e-18)

        psd_db = 10*np.log10(pXNoise + 1e-18)   # avoid log(0)

        # --- update plot ---
        self.line.set_xdata(fNoise)
        self.line.set_ydata(psd_db)

        # update plot limits dynamically
        self.ax.set_xlim([0, np.max(fNoise)])
        #self.ax.set_ylim([np.min(psd_db) - 5, np.max(psd_db) + 5])
        self.ax.set_ylim([-150, 10])

        # redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ax.set_title(total_db)



class snrEstimator():
    #class to estimate the snr. The psd's are estimated using the comm window time frame
    def __init__(self, config):
        self.config = config
        self.noisebuf = np.zeros(config['payloadSamples'])
        self.mixedbuf = np.zeros(config['payloadSamples'])

        self.noisePSD = [] #no value yet
        self.noiseFreq = [] #no value yet

        self.mixedPSD = []
        self.mixedFreq = [] #no value yet


    def pushData(self,data, noise):
        if noise:
            self.noisebuf = data.copy() 
        else:
            self.mixedbuf = data.copy()

    def getNoisePSD(self):
        fNoise, pXNoise = welch(self.noisebuf, self.config["FS"]
                                , nperseg=self.config['payloadSamples'] // 4)

        return fNoise, pXNoise

    def updatePSD(self):

        fNoise,pXNoise = self.getNoisePSD()

        fMixed, pXMixed = welch(self.noisebuf, self.config["FS"]
                                , nperseg=self.config['payloadSamples']//2)
        
        pXSignal = np.zeros_like(pXMixed)
        #we assume fMixed and fNoise are the same
        for i,f in enumerate(zip(fMixed, fNoise)):
            if(f[0] != f[1]):
                raise ValueError("The arrays for the frequencies do not match.")
            pXSignal[i] = max(0, pXMixed[i] - pXNoise[i])


class soundRecorder():
    def __init__(self):
        self.config = {'FS':48000,
                  'payloadSamples':48000*1}

        self.rcv =  AudioReceiver.AudioReceiver(self.config, self)
    
    def pushWindow(self, data):
        #align data with argmax

        tmax = np.argmax(data)
        tmax = tmax - 1000
        if(tmax < 0):
            tmax = 0
        data = data[tmax:]
        x = np.linspace(0,self.config["payloadSamples"]/self.config['FS'], len(data))
        plt.plot(data)
        plt.show()



class impulseResponseEstimator():
    #class to visualise 2 channel impulse reponse
    def __init__(self, nbPlots, diracTime):
        plt.rcParams.update({'font.size': 18})
        config = {'FS' : 48000,
              'FC' : 500,
              'RS' : 20,
              'preambleSymbols' : 20,
              'windowLenghtSymbols' : 6, #1 second of impulse resp  
              'corrRatioThresh' : 0.70, 
              'excessBandwidth': 0.50,
              'lpCutoffEpsilon': 0.05,
              'bitsPerSymbol' : 2,
              'Eb': 400
              }
        Common = commons.CommonDynamic(config)
        self.Common = Common
        self.config = config
        receiver = TimeReceiver.TimeReceiver(Common.config, Common.mod, Common.demod, False, self.processPayload)
        self.rcv =  AudioReceiver.AudioReceiver(self.config, receiver, cycles=50) #15 should be enough
        self.receiver = receiver

        fig, ax = plt.subplots(nbPlots, figsize=(12,8))
        self.ax = ax
        self.fig = fig
        
        self.nbPlots = nbPlots

        self.captured = 0
        self.diracTime = diracTime

    def processPayload(self, data):
        print("Got one")
        if(self.captured >= self.nbPlots):
            returnplt.rcParams.update({'font.size': X})

        data *= 30
        #discard syncrhonisation time
        data = data[self.receiver.preambleLenght:]

        x = np.linspace(0,len(data)/self.config['FS'], len(data))

        self.ax[self.captured].set_xlabel("Time (seconds)") 
        self.ax[self.captured].set_ylabel("Recorded Impulse response") 

        #put dirac time 
        diracT = self.diracTime #+ self.receiver.preambleLenght
        diracT = diracT  
        plotDiracTime = x[diracT]

        #self.ax[self.captured].axvline(x=plotDiracTime, color='red', linestyle='--', linewidth=2, label="Dirac time")
        #self.ax[self.captured].legend()
        self.ax[self.captured].set_title("Room " + str(self.captured+1))
        self.ax[self.captured].set_ylim(-2, 2)

        self.ax[self.captured].plot(x, data )
        
        self.captured+=1
        if(self.captured >= self.nbPlots):
            self.fig.suptitle('Plots of different channel (rooms) impulse reponse')
            plt.tight_layout()
            plt.show()




class impulseResponseEstimator():
    #class to visualise 2 channel impulse reponse
    def __init__(self, nbPlots, diracTime):
        plt.rcParams.update({'font.size': 18})
        config = {'FS' : 48000,
              'FC' : 500,
              'RS' : 20,
              'preambleSymbols' : 20,
              'windowLenghtSymbols' : 6, #1 second of impulse resp  
              'corrRatioThresh' : 0.70, 
              'excessBandwidth': 0.50,
              'lpCutoffEpsilon': 0.05,
              'bitsPerSymbol' : 2,
              'Eb': 400
              }
        Common = commons.CommonDynamic(config)
        self.Common = Common
        self.config = config
        receiver = TimeReceiver.TimeReceiver(Common.config, Common.mod, Common.demod, False, self.processPayload)
        self.rcv =  AudioReceiver.AudioReceiver(self.config, receiver, cycles=50) #15 should be enough
        self.receiver = receiver

        fig, ax = plt.subplots(nbPlots, figsize=(12,8))
        self.ax = ax
        self.fig = fig
        
        self.nbPlots = nbPlots

        self.captured = 0
        self.diracTime = diracTime

    def processPayload(self, data):
        print("Got one")
        if(self.captured >= self.nbPlots):
            returnplt.rcParams.update({'font.size': X})

        data *= 30
        #discard syncrhonisation time
        data = data[self.receiver.preambleLenght:]

        x = np.linspace(0,len(data)/self.config['FS'], len(data))

        self.ax[self.captured].set_xlabel("Time (seconds)") 
        self.ax[self.captured].set_ylabel("Recorded Impulse response") 

        #put dirac time 
        diracT = self.diracTime #+ self.receiver.preambleLenght
        diracT = diracT  
        plotDiracTime = x[diracT]

        #self.ax[self.captured].axvline(x=plotDiracTime, color='red', linestyle='--', linewidth=2, label="Dirac time")
        #self.ax[self.captured].legend()
        self.ax[self.captured].set_title("Room " + str(self.captured+1))
        self.ax[self.captured].set_ylim(-2, 2)

        self.ax[self.captured].plot(x, data )
        
        self.captured+=1
        if(self.captured >= self.nbPlots):
            self.fig.suptitle('Plots of different channel (rooms) impulse reponse')
            plt.tight_layout()
            plt.show()


class ChannelISIEstimator():
    def __init__(self):
        plt.rcParams.update({'font.size': 18})
        config = commons.Common.config
        self.config = config
        receiver = TimeReceiver.TimeReceiver(config, commons.Common.mod, commons.Common.demod, True, self.processPayload)
        self.rcv =  AudioReceiver.AudioReceiver(self.config, receiver, cycles=15) 
        self.receiver = receiver
        self.phaseSynchroniser = Synchronisation.MLPhaseSynchroniser(config, commons.Common.mod)

        self.channelEst = Equalisation.ChannelEstimator(config, commons.Common.mod)
        self.mod = commons.Common.mod

        self.pulseMF = self.mod.pulse.conj()[::-1]

    def processPayload(self, data):

        phaseSync = self.phaseSynchroniser.synchronisePhase(data)
        corr = fftconvolve(phaseSync, self.pulseMF)
        corr = corr[len(self.pulseMF) -1 :]

        #samples at symbols
        #corr = corr[len(self.pulseMF)-1:]
        corr = corr[::self.config["samplesPerSymbol"]]
        #estimate the channel

        p = self.channelEst.estimateChannel(corr)
        plt.plot(np.abs(p))
        plt.show()


delayDirac = 2500
nbPlots = 2
if __name__ == "__main__":
    match sys.argv[1]:
        case "psd":
            test = PSDVisualizer()

            test.rcv.listen()

        case "impulse":

            test =impulseResponseEstimator(2, delayDirac)
            test.rcv.listen()
        case "isi":
            test =ChannelISIEstimator()
            test.rcv.listen()
        



