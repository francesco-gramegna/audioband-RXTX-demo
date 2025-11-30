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
#more complicated class than transmitter



class SimpleReceiver():
    #we get the modulator since it helps us a bit
    def __init__(self, config, modulator, demodulator):
        self.config = config
        self.modulator = modulator
        self.bufferSize = 4*config['payloadSamples']
        #self.incomingData =  utils.RingBuffer(self.bufferSize)
        
        self.preambule = modulator.getBasebandPreamble()

        self.incomingData = np.zeros(self.bufferSize)  #could be more but not less

        self.processingBuffer = np.zeros(config['payloadSamples'])

        self.timeSynchroniser = Synchronisation.TimeSynchroniser(config, modulator)

        self.phaseSynchroniser = Synchronisation.MLPhaseSynchroniser(config, modulator)

        self.channelEst = Equalisation.ChannelEstimator(config, modulator, 20000) #len(self.preambule)//2)

        self.state = 'IDLE'
        self.waitingSamples = 0
        self.grabbedSamples = 0

        self.demodulator = demodulator

        self.preambleLenght = len(modulator.getBasebandPreamble())

        self.agc = mathUtils.SimpleAGC(1, 0.05)

        self.snrEstimator = EnvAnalysis.snrEstimator(config)

        #plotting
        #plt.ion()  # interactive mode on
        #self.fig, self.ax = plt.subplots()
        #self.line, = self.ax.plot(np.zeros(self.bufferSize))  # initial empty line
        #self.ax.set_ylim(-1.5, 1.5)  # adjust depending on expected amplitude
        #self.ax.set_xlim(0, self.bufferSize)

        #self.plot_buffer = np.zeros(self.bufferSize)


        
    
    def pushWindow(self, newData):

        #AGC maybe
        #newData = self.agc.process(newData)
        #newData.flatten()
        #newData = newData * 1000
        #print(np.max(newData))

        stime = time.time()
        downconverted = self.modulator.downConvert(newData) 
        
        #print("DC time : ", time.time() - stime)
        stime = time.time()
        #downconverted=newData

        #PSD computations

        """
        self.snrEstimator.pushData(downconverted, noise=True)
        fX, noisePSD = self.snrEstimator.getNoisePSD()
        plt.plot(fX, noisePSD, 'r')
        plt.show()
        """


        #self.incomingData.write(downconverted)
        self.incomingData = np.append(self.incomingData, downconverted)
        toRemove = len(self.incomingData) - self.bufferSize

        if(toRemove > 0):
            self.incomingData = self.incomingData[toRemove:]

        #print(len(self.incomingData))


        # ---- LIVE PLOT UPDATE ----

        """
        self.plot_buffer = np.roll(self.plot_buffer, -len(newData))
        self.plot_buffer[-len(newData):] = newData
        self.line.set_ydata(self.plot_buffer)
        self.ax.relim()
        self.ax.autoscale_view(scaley=True)  # COMMENT OUT IF YOU WANT FIXED SCALE
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        """

        #return



        #print("DATA time : ", time.time() - stime)

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
                    self.processingBuffer = self.incomingData[sigStart:sigEnd]

                    self.incomingData = self.incomingData[sigEnd:]

                    self.processPayload()


    def processPayload(self):
        #print('Processing packet')

        #print(" len buf : " , len(self.processingBuffer))
        phaseSync = self.phaseSynchroniser.synchronisePhase(self.processingBuffer)

        #estimate phase

        h = self.channelEst.estimateChannel(phaseSync)

        h = self.channelEst.interpolate_h(h)

        plt.plot(h.real, 'b')
        plt.plot(h.imag, 'r')
        plt.show()

        #we remove the preamble
        #phaseSync = phaseSync[self.preambleLenght:]

        self.demodulator.demodulate(phaseSync)
        
        #plt.plot(self.processingBuffer.real, 'r')
        #plt.plot(self.processingBuffer.imag, 'g')
        #plt.plot(phaseSync.real, 'b')
        #plt.plot(phaseSync.imag, 'r')
        #plt.show()

        return



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
        
        pipewire_output = None
        for idx, dev in enumerate(sd.query_devices()):
            if "pipewire" in dev['name'].lower() or "pulse" in dev['name'].lower():
                pipewire_output = idx
                print("yay")
                break

        sd.default.device = pipewire_output


        proc, q = self.start_worker()
        self.q = q

        with sd.InputStream(
            device=None,
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


def main():
    mod = Common.mod
    demod = Common.demod
    config = Common.config

    rcv = SimpleReceiver(config, mod, demod)

    audioMachine = AudioReceiver(config, rcv)
    audioMachine.listen()


if __name__ == "__main__":
    main()



