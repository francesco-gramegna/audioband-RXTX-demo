import numpy as np
from scipy.signal import decimate, upfirdn 
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


def main():
    x = np.array([1,0,1,1,0,1,0,0,0,1])
    
    ch = [0.5] #, 0, 0, -0.25, 0, 0.12,0, -0.20]


    y = fftconvolve(x,ch)

    
    H = np.sum(x*y[:len(x)])/np.sum(x**2)
    print(H)

    h = np.fft.fft(H)
    print(h)



    plt.plot(x, 'r')
    plt.plot(y,'b')
    plt.show()


main()


