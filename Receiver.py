


import matplotlib.pyplot as plt
import Synchronisation
import numpy as np
#more complicated class than transmitter



class SimpleReceiver():
    #we get the modulator since it helps us a bit
    def __init__(self, config, modulator):
        self.config = config
        self.modulator = modulator
        self.bufferSize = 4*config['payloadSamples']
        self.incomingData = np.zeros(self.bufferSize)  #could be more but not less
        self.processingBuffer = np.zeros(config['payloadSamples'])

        self.timeSynchroniser = Synchronisation.TimeSynchroniser(config, modulator)

        self.phaseSynchroniser = Synchronisation.MLPhaseSynchroniser(config, modulator)

        self.state = 'IDLE'
        self.waitingSamples = 0
        self.grabbedSamples = 0
        
    
    def pushWindow(self, newData):

        downconverted = self.modulator.downConvert(newData) 

        self.incomingData = np.append(self.incomingData, downconverted)

        toRemove = len(self.incomingData) - self.bufferSize
        if(toRemove > 0):
            self.incomingData = self.incomingData[:toRemove]

        match self.state:
            case 'IDLE':
                preambuleIndex, peak = self.timeSynchroniser.getPreambuleStartIndex(self.incomingData)

                print(peak)
                if (preambuleIndex != -1):
                    self.state = 'POTENTIAL_PACKET'
            
            case 'POTENTIAL_PACKET':
                preambuleIndex, peak = self.timeSynchroniser.getPreambuleStartIndex(self.incomingData)

                if (preambuleIndex == -1):
                    #the packet was a false negative probably
                    self.state ='IDLE'
                    print('Lost packet')
                else:
                    #we can decode
                    print("YES")
                    self.state = 'IDLE'

                    sigStart = preambuleIndex
                    sigEnd = preambuleIndex + self.config['payloadSamples']
                    self.processingBuffer = self.incomingData[sigStart:sigEnd]

                    self.incomingData = self.incomingData[:sigEnd]

                    self.processPayload()


    def pushWindow2(self, newData):

        downconverted = self.modulator.downConvert(newData) 
        
        if(self.state == 'IDLE'):

            #incoming data is useless
            preambuleIndex = self.timeSynchroniser.getPreambuleStartIndex(downconverted)

            if (preambuleIndex != -1):
                self.state = 'WAITING_REST_PACKET'

                samplesGot = len(newData) - preambuleIndex

                samplesRemaining = self.config['payloadSamples'] - samplesGot

                self.waitingSamples = samplesRemaining
                self.grabbedSamples = samplesGot

                #copy the samples
                self.processingBuffer = np.concatenate([downconverted[preambuleIndex:], np.zeros(self.waitingSamples)])

        elif (self.state == 'WAITING_REST_PACKET'):

            self.state = 'IDLE'
            self.processingBuffer = np.concatenate([self.processingBuffer[:self.grabbedSamples], downconverted[:self.waitingSamples]])

            #got one in the oven
            self.processPayload()

            #TODO risky way of handling it
            self.pushWindow(newData[self.waitingSamples:])




    def processPayload(self):
        print('Processing packet')

        phaseSync = self.phaseSynchroniser.synchronisePhase(self.processingBuffer)
        
        plt.plot(self.processingBuffer, 'r')
        plt.plot(phaseSync, 'b')
        plt.show()

        
        return
