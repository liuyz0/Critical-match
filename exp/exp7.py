import numpy as np
import sys
sys.path.append('../src')
from utils import sample
import time

# For log supp, only yield, run full phase diagram.

start_time = time.time()
M = 32
N_span = range(1,129)
NU = np.zeros((len(N_span), 1000))
rhos = np.zeros((len(N_span), 1000))

for i in range(len(N_span)):
    N = N_span[i]
    print('to ', i+1)
    for jY in range(1,1001):
        sigY = np.sqrt(1/(jY/1000)**2 - 1)/4
        NU[i,jY-1], rhos[i,jY-1] = sample(N,M,0,sigY,0,supply='log')

print("Finished--- %s seconds ---" % (time.time() - start_time))

np.savez('../data/exp7.npz', NU = NU, rhos = rhos)