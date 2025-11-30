import Equalisation
import EnvAnalysis
import mathUtils
from commons import Common
import multiprocessing
import threading
import time
import sounddevice as sd
import Demodulator
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import Synchronisation
import numpy as np
import utils
#this class is just for environment testing purposes
#it works just like the true receiver


class TimeReceiver():
    def __init__(self, config, modulator, demodulator, downconvert,  processMethod):
        self.config = config
        self.modulator = modulator
        self.bufferSize = 4*config['payloadSamples']
        #self.incomingData =  utils.RingBuffer(self.bufferSize)
        
        self.preambule = modulator.getBasebandPreamble()

        self.incomingData = np.zeros(self.bufferSize)  #could be more but not less
        self.upConvertedIncomingData = np.zeros(self.bufferSize)  

        self.processingBuffer = np.zeros(config['payloadSamples'])

        self.timeSynchroniser = Synchronisation.TimeSynchroniser(config, modulator)

        self.processMethod = processMethod

        self.state = 'IDLE'
        self.waitingSamples = 0
        self.grabbedSamples = 0

        self.preambleLenght = len(self.preambule)

        self.downconvert = downconvert

         
    def pushWindow(self, newData):

        stime = time.time()
        downconverted = self.modulator.downConvert(newData) 
        
        stime = time.time()
        #downconverted=newData

        #PSD computations

        #self.incomingData.write(downconverted)
        self.incomingData = np.append(self.incomingData, downconverted)
        toRemove = len(self.incomingData) - self.bufferSize

        if not self.downconvert:
            self.upConvertedIncomingData = np.append(self.upConvertedIncomingData, newData)

        if(toRemove > 0):
            self.incomingData = self.incomingData[toRemove:]
            if not self.downconvert:
                self.upConvertedIncomingData = self.upConvertedIncomingData[toRemove:]

        match self.state:
            case 'IDLE':
                stime = time.time()
                preambuleIndex, peak = self.timeSynchroniser.getPreambuleStartIndex(self.incomingData)

                #print("corr time : ", time.time() - stime)

                #print(peak)
                if (preambuleIndex != -1):
                    self.state = 'POTENTIAL_PACKET'
            
            case 'POTENTIAL_PACKET':
                preambuleIndex, peak = self.timeSynchroniser.getPreambuleStartIndex(self.incomingData)

                if (preambuleIndex == -1):
                    #the packet was a false negative probably
                    self.state ='IDLE'
                    #print('Lost packet')
                else:
                    #we can decode
                    self.state = 'IDLE'

                    sigStart = preambuleIndex
                    sigEnd = preambuleIndex + self.config['payloadSamples']
                    #print(sigStart , " " , sigEnd)
                    if self.downconvert:
                        self.processingBuffer = self.incomingData[sigStart:sigEnd]
                    else:
                        self.processingBuffer = self.upConvertedIncomingData[sigStart:sigEnd]

                    self.incomingData = self.incomingData[sigEnd:]
                    if not self.downconvert:
                        self.upConvertedIncomingData = self.upConvertedIncomingData[sigEnd:]

                    self.processMethod(self.processingBuffer)

