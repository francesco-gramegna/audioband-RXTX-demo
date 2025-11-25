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
#more complicated class than transmitter



class SimpleReceiver():
    #we get the modulator since it helps us a bit
    def __init__(self, config, modulator, demodulator):
        self.config = config
        self.modulator = modulator
        self.bufferSize = 4*config['payloadSamples']
        #self.incomingData =  utils.RingBuffer(self.bufferSize)
        self.incomingData = np.zeros(self.bufferSize)  #could be more but not less

        self.processingBuffer = np.zeros(config['payloadSamples'])

        self.timeSynchroniser = Synchronisation.TimeSynchroniser(config, modulator)

        self.phaseSynchroniser = Synchronisation.MLPhaseSynchroniser(config, modulator)

        self.state = 'IDLE'
        self.waitingSamples = 0
        self.grabbedSamples = 0

        self.demodulator = demodulator

        self.preambleLenght = len(modulator.getBasebandPreamble())

        
    
    def pushWindow(self, newData):


        stime = time.time()
        downconverted = self.modulator.downConvert(newData) 
        
        #print("DC time : ", time.time() - stime)
        stime = time.time()
        #downconverted=newData

        #self.incomingData.write(downconverted)
        self.incomingData = np.append(self.incomingData, downconverted)
        toRemove = len(self.incomingData) - self.bufferSize

        if(toRemove > 0):
            self.incomingData = self.incomingData[toRemove:]

        #print("DATA time : ", time.time() - stime)

        match self.state:
            case 'IDLE':
                stime = time.time()
                preambuleIndex, peak = self.timeSynchroniser.getPreambuleStartIndex(self.incomingData)

                #print("corr time : ", time.time() - stime)

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

                    self.incomingData = self.incomingData[sigEnd:]

                    self.processPayload()


    def processPayload(self):
        print('Processing packet')

        phaseSync = self.phaseSynchroniser.synchronisePhase(self.processingBuffer)

        #we remove the preamble
        #phaseSync = phaseSync[self.preambleLenght:]

        self.demodulator.demodulate(phaseSync)
        
        plt.plot(self.processingBuffer, 'r')
        plt.plot(phaseSync, 'b')
        plt.show()

        return


    def audio_callback(self,indata, frames, time, status):
        print("pushing")
        if status:
            print("stat : " , status)

        self.pushWindow(indata.flatten())        
        #print('finished pushing')

    def listen(self):

        with sd.InputStream(
            blocksize=2048,
            samplerate=self.config['FS'],
            channels=1,      
            dtype='float32',
            latency="high",
            callback=self.audio_callback):

            print(self.config['payloadSamples'])
            print("Recording in MONO... press Ctrl+C to stop.")
            try:
                while True: 
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("Stopping...")



class AudioReceiver():
    def __init__(self, config, receiver ):
        self.config = config
        self.receiver = receiver

        self.size = config['payloadSamples']
        self.buf = np.zeros(config['payloadSamples'])
        self.i = 0

    def receiver_thread(receiver, task_queue):
        while True:
            task = task_queue.get()   
            if task is None:
                break
            receiver.pushWindow(task['data'])

    

    def start_worker(self):
        task_queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=AudioReceiver.receiver_thread, args=(self.receiver, task_queue,))
        proc.start()
        return proc, task_queue


    
    def push_audio(self, indata, frames, time_info, status):
        indata = indata.flatten()  # mono

        space = self.size - self.i
        to_add = min(space, len(indata))
        #print(self.i)

        # fill the buffer
        self.buf[self.i:self.i + to_add] = indata[:to_add]
        self.i += to_add

        # If buffer now full → send to worker
        if self.i == self.size:
            #print("pushing")
            self.q.put({"data": self.buf.copy()})
            self.i = 0

            # handle overflow if more input is left
            remaining = len(indata) - to_add
            if remaining > 0:
                self.buf[:remaining] = indata[to_add:to_add + remaining]
                self.i = remaining

        if status:
            print("stat:", status)



    def listen(self):
        proc, q = self.start_worker()
        self.q = q

        with sd.InputStream(
            blocksize=2048,
            samplerate=self.config['FS'],
            channels=1,      
            dtype='float32',
            latency="high",
            callback=self.push_audio):

            print(self.config['payloadSamples'])
            print("Recording in MONO... press Ctrl+C to stop.")
            try:
                while True: 
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("Stopping...")



