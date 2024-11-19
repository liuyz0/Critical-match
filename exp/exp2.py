import numpy as np
import sys
sys.path.append('../src')
from utils import sample
import time

# For lin supp, only H, run full phase diagram.

start_time = time.time()
M = 32
N_span = range(1,129)
NU = np.zeros((len(N_span), 1000))
rhos = np.zeros((len(N_span), 1000))

for i in range(len(N_span)):
    N = N_span[i]
    print('to ', i+1)
    for jH in range(1,1001):
        sigH = np.sqrt(1/(jH/1000)**2 - 1)/np.sqrt(24)
        NU[i,jH-1], rhos[i,jH-1] = sample(N,M,0,0,sigH)

print("Finished--- %s seconds ---" % (time.time() - start_time))

np.savez('../data/exp2.npz', NU = NU, rhos = rhos)