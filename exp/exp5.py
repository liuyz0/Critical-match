import numpy as np
import sys
sys.path.append('../src')
from utils import sample
import time

# For lin supp, Y, P, and H, run full phase diagram.

start_time = time.time()
M = 32
N_span = range(1,129)
NU = np.zeros((len(N_span), 1000))
rhos = np.zeros((len(N_span), 1000))

for i in range(len(N_span)):
    N = N_span[i]
    print('to ', i+1)
    for jr in range(1,1001):
        ratio = 0.6 * np.random.rand()
        if np.sqrt(1/(jr/1000)**2 - 1) > 1.0:
            sigP = 0.5 / np.sqrt(12) * ratio
        else:
            sigP = np.sqrt(1/(jr/1000)**2 - 1) * 0.5 / np.sqrt(12) * ratio
        radius = np.sqrt(1/(jr/1000)**2/(sigP**2/0.5**2*12 + 1) - 1)
        theta = np.pi / 2 * np.random.rand()
        sigY = radius*np.cos(theta)/4
        sigH = radius * np.sin(theta) /np.sqrt(24)
        NU[i,jr-1], rhos[i,jr-1] = sample(N,M,sigP,sigY,sigH,Pdist='expo')

print("Finished--- %s seconds ---" % (time.time() - start_time))

np.savez('../data/exp5.npz', NU = NU, rhos = rhos)