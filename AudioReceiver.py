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
#

class AudioReceiver():
    def __init__(self, config, receiver, cycles=float('inf')):
        self.config = config
        self.receiver = receiver
        self.cycles= cycles

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
            self.cycles-=1
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
                    if(self.cycles <= 0):
                        print("AudioReceiver nb of cycles reached. Stopping")
                        break
            except KeyboardInterrupt:
                print("Stopping...")



