import numpy as np
import sys
sys.path.append('../src')
from utils import sample
import time

# For lin supp, Y and H, run full phase diagram.

start_time = time.time()
M = 32
N_span = range(1,129)
NU = np.zeros((len(N_span), 1000))
rhos = np.zeros((len(N_span), 1000))

for i in range(len(N_span)):
    N = N_span[i]
    print('to ', i+1)
    for jr in range(1,1001):
        radius = np.sqrt(1/(jr/1000)**2 - 1)
        theta = np.pi / 2 * np.random.rand()
        sigY = radius*np.cos(theta)/4
        sigH = radius * np.sin(theta) /np.sqrt(24)
        NU[i,jr-1], rhos[i,jr-1] = sample(N,M,0,sigY,sigH)

print("Finished--- %s seconds ---" % (time.time() - start_time))

np.savez('../data/exp4.npz', NU = NU, rhos = rhos)