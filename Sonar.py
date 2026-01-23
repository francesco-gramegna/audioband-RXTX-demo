import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import ChannelSimulator
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

class SimpleSonar():
    def __init__(self, fake=False):


        plt.rcParams.update({'font.size': 18})
        config = commons.Common.config
        self.config = config
        receiver = TimeReceiver.TimeReceiver(config, commons.Common.mod, commons.Common.demod, True, self.processPayload)

        if(fake):
            self.rcv = ChannelSimulator.EnvSimulateChannel(config, 'h', receiver)
        else:
            self.rcv = AudioReceiver.AudioReceiver(self.config, receiver, cycles=1000) 

        self.receiver = receiver
        self.phaseSynchroniser = Synchronisation.MLPhaseSynchroniser(config, commons.Common.mod)
        self.MLAmplitudeSync = Synchronisation.MLAmplitudeSync(config, commons.Common.mod)

        self.bestSamplingSync = Synchronisation.SymbolTimingSynchroniser(config, commons.Common.mod)

        self.channelEst = Equalisation.ChannelEstimator(config, commons.Common.mod)
        self.mod = commons.Common.mod

        self.pulseMF = self.mod.pulse.conj()[::-1]
        self.i = 0

        # plot related
        self.fig = None
        self.ax = None
        self.line = None

        plt.ion()  # interactive mode

    def processPayload(self, data):
        print("got one")
        phaseSync = self.phaseSynchroniser.synchronisePhase(data)


        corr = fftconvolve(phaseSync, self.pulseMF,mode='full')
        corr = corr[len(self.pulseMF) - 1:]


        # samples at symbols
        corr = corr[::self.config["samplesPerSymbol"]]

        try:
            p, n0 = self.channelEst.estimateChannel(corr)
            print("n0 : ", n0)
        except Exception as e:
            print(e)
            print("Timing error")
            return

        p = np.abs(p)

        scale_meters = np.arange(len(p))  # in symbols
        speed_sound = 331  # m/s

        T = 1 / self.config["RS"]  # T = 1/Rs
        scale_meters = T * speed_sound * scale_meters

        x = np.arange(len(p))

        # first estimation -> create plot
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot(x, p, label="Signal 1")

            # secondary axis
            ax2 = self.ax.secondary_xaxis(
                'top',
                functions=(lambda s: s * T * speed_sound,
                           lambda m: m / (T * speed_sound))
            )
            ax2.set_xlabel("Distance travelled (meters)")

            self.ax.set_xlabel("Symbol index")
            self.ax.set_ylabel("|h|")
            self.ax.set_title("Estimated h (channel impulse response)")
            self.ax.grid(True)
            self.ax.legend()

            plt.show()
        else:
            # update existing plot
            self.line.set_xdata(x)
            self.line.set_ydata(p)
            self.line.set_label("Signal " + str(self.i + 1))
            self.ax.relim()
            self.ax.autoscale_view()
            self.ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.i += 1

if __name__ == "__main__":
    test = SimpleSonar()
    test.rcv.listen()
