import TransmitterScript
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
from scipy.io import wavfile


class FakeChannel():
    def __init__(self, ir, noise=0):
        self.ir = ir
        self.noise = noise

    def process(self, fullData):

        fullData = fullData + np.random.normal(0, self.noise,size=len(fullData)) 
        print(fullData.shape)
        print(self.ir.shape)
        processed = fftconvolve(fullData, self.ir)

        processed = processed  * np.exp(1j * np.linspace(0, 2*np.pi, len(processed)))


        return processed



def delayIr():
    ch = np.zeros((48000))
    ch[1234] = 1
    return ch

def easyIr():
    ch = np.zeros((48000))
    ch[1241] = 1
    ch[1400] = -0.5
    ch[1710] = 0.25
    return ch

def semiRandomIr():
    ch = np.zeros((48000))
    start = 1371
    ch[1371] = 1

    rand = np.random.normal(0, 0.1, size=len(ch) - start - 1)

    ch[1372:] = rand

    t = np.linspace(1,0, len(ch) - start-1)
    ch[1372:] = ch[1372:] * t**6

    #plt.plot(ch)
    #plt.show()
    return ch



class SimulateChannel():
    def __init__(self, config, channel, sink, data):

            self.config = config
            self.channel = channel
            self.sink = sink
            self.data = data

    def listen(self):

        data = self.data
        print("Processing data through channel")
        data = self.channel.process(data)
        print("Sending data to receiver")
        
        for i in range(0,len(data), self.config['payloadSamples']):
            self.sink.pushWindow(data[i:i+self.config['payloadSamples']])


#a class to facilitate the testing of EnvAnalysis classes
class EnvSimulateChannel(SimulateChannel):
    def __init__(self, config,  name, sink):
        channel = FakeChannel(easyIr(), noise=0.00)

        #generate the data
        TransmitterScript.main(['', name])
        
        samplerate, data = wavfile.read("output.wav")

        super().__init__(config, channel, sink, data)

                 



