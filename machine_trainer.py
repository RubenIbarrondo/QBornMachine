from state_maker import get_ensemble_Q
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from myga_v02 import GeneticOptimization

from datetime import datetime

'''
I will try to train it so that the probability
is proportional to the first 2 bits
'''

def binenc(x, xmin=0.0, xmax=1.0, Nx=16):
    """
    Gets the uniformly ditributed binary representation of the real number x
    in range xmin - xmax with Nx bit precision.
    :param x: real number in range [xmin, xmax)
    :param xmin: real number lower bound for x
    :param xmax: real number upper bound for x
    :param Nx: integer value for bit precision
    :returns ib: binary string of length Nx
    """
    if x < xmin or x >= xmax:
        raise Exception("x must be between xmin={} and xmax={}, got {}.".format(xmin, xmax, x))
    i = int((x - xmin) / (xmax - xmin) * 2**Nx)
    ib = bin(i)[2:]
    if len(ib) < Nx:
        ib = '0'*(Nx-len(ib)) + ib
    return ib


def bindec(ib, xmin=0.0, xmax=1.0):
    """
    Gets the real number from the uniformly ditributed binary representation
    in range xmin - xmax with len(ib) bit precision.
    :param ib: binary string representing the number
    :param xmin: real number lower bound for the output x
    :param xmax: real number upper bound for the output x
    :returns x: real number encoded in ib
    """
    i = sum(2 ** (len(ib) - j - 1) * int(ib[j]) for j in range(len(ib)))
    x = xmin + i/2**len(ib) * (xmax - xmin)
    return x

def gen_hist(Q, T=1024):
    '''generates an histogram based on the distribution given by get_prob'''
    N = 2 ** Q
    hist  = dict()
    k = 0
    counter = 0
    while k < T:
        x = binenc(np.random.randint(0, N), 0, N, Q)
        r = np.random.random()
        if get_prob(x) > r:
            hist[x] = hist.get(x, 0) + 1
            k += 1
        counter += 1
        assert counter < 20*T, 'gen_hist exceeded iteration limit'
    return hist

def get_prob(bin_string):
    '''bin_string has to be of length >= 2.'''
    N = 2** (len(bin_string)-2)

    if bin_string[:2] == '00':
        t = 0
    elif bin_string[:2] == '01':
        t = 1/6
    elif bin_string[:2] == '10':
        t = 2/6
    elif bin_string[:2] == '11':
        t = 3/6

    return t/N

def fitness_function(population, nparams, pmin, pmax, getprob, Q, T):
    fitness_population = []

    for chromosome in population:
        # there are nparams parameters encoded in each chromosome
        params = [bindec(chromosome[pindex*32: pindex*32 + 32], xmax=pmax, xmin=pmin) for pindex in range(nparams)]
        params = [[p0, p1] for p0, p1 in zip(params[:Q], params[Q:])]

        counts = get_ensemble_Q(params, Q, T)
        fitness = 0
        for i in range(2**Q):
            key = binenc(i, 0, 2**Q, Q)
            # simple fit
            #fitness += abs(getprob(key) - counts.get(key, 0) / T)
            fitness += np.exp(abs(getprob(key) - counts.get(key, 0) / T)) - 1
        fitness = 1/fitness
        fitness_population.append(fitness)

    return fitness_population

if __name__ == '__main__':
    pmax = 2*np.pi
    pmin = 0

    #  Circuit parameters
    # --------------------
    T = 1024
    Q = 4
    nparams = 2 * Q
    # --------------------

    #    GA parameters
    # --------------------
    pN = 50
    gN = 1000
    pc = 0.7
    pm = 0.005
    # --------------------

    population = GeneticOptimization(fitness_function, args=(nparams, pmin, pmax, get_prob, Q, T),
                                     chromosome_length=32*nparams,
                              populationNumber=pN, generationNumber=gN,
                              pc=pc, pm=pm)
    fitness = fitness_function(population, nparams, pmin, pmax, get_prob, Q, T)

    #                    Data saving
    # -----------------------------------------------------
    file = open('datadir/data_' + str(datetime.now()), 'w')
    print('today : ', datetime.now(), file=file)
    print('--- Quantum Circuit parameters ---', file=file)
    print('Q = ', Q,
          'T = ', T, file=file)
    print('--- Genetic Optimization parameters ---', file=file)
    print('populationNumber = ', pN,
          ', generationNumber = ', gN,
          ', pc = ', pc,
          ', pm = ', pm, file=file)
    print('\n--- Fitness ---\n', fitness, file=file)
    print('\n--- Population ---\n', population, file=file)
    file.close()
    # -----------------------------------------------------


    hists = []
    for chromosome in population:
        params = [bindec(chromosome[pindex*32: pindex*32 + 32], xmax=pmax, xmin=pmin) for pindex in range(nparams)]
        params = [[p0, p1] for p0, p1 in zip(params[:Q], params[Q:])]
        hists.append(get_ensemble_Q(params, Q, T=1024))

    h0 = gen_hist(Q)
    h1, h2 = hists[:2]
    f1, f2 = fitness[:2]

    print('f1: ', f1)
    print('f2: ', f2)
    print('f max: ', max(fitness))
    fig, ax = plt.subplots(1, 3)
    plot_histogram(h0, ax=ax[0])
    plot_histogram(h1, ax=ax[1])
    plot_histogram(h2, ax=ax[2])
    plt.show()