import Synchronisation
import numpy as np
#more complicated class than transmitter



class SimpleReceiver():
    #we get the modulator since it helps us a bit
    def __init__(self, config, modulator):
        self.config = config
        self.modulator = modulator
        self.incomingData = np.zeros(config['payloadSamples']) 
        self.processingBuffer = np.zeros(config['payloadSamples'])

        self.timeSynchroniser = Synchronisation.TimeSynchroniser(config, modulator)

        self.state = 'IDLE'
        self.waitingSamples = 0
        self.grabbedSamples = 0
        
        

    def pushWindow(self, newData):
        
        if(self.state == 'IDLE'):

            #incoming data is useless
            preambuleIndex = self.timeSynchroniser.getPreambuleStartIndex(self.incomingData)

            if (preambuleIndex != -1):
                self.state = 'WAITING_REST_PACKET'

                samplesGot = len(newData) - preambuleIndex

                samplesRemaining = config['payloadSamples'] - samplesGot

                self.waitingSamples = samplesRemaining
                self.grabbedSamples = samplesGot

                #copy the samples
                self.processingBuffer[:samplesGot] = newData[preambuleIndex:]

        elif (self.state == 'WAITING_REST_PACKET'):

            self.state = 'IDLE'
            self.processingBuffer[self.samplesGot:] = newData[:self.waitingSamples]
            #got one in the oven
            self.processPayload()

            #TODO risky way of handling it
            self.pushWindow(newData[self.waitingSamples:])





    def processPayload(self):
        print('Processing packet')



