import matplotlib.pyplot as plt


def plotConstellation(constellation, M):
    for i in range(M):
        symbol = constellation.symbol(i)
        plt.scatter(symbol.real, symbol.imag, color='red', s=100)
    plt.show()

                



