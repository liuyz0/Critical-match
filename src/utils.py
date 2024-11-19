import numpy as np

def sample(N:int,
           M:int,
           sigP,
           sigY,
           sigH,
           supply = 'lin',
           Sl = 0.01,
           Sh = 0.98,
           Rl = 0.01,
           Rh = 0.98,
           Pdist = None):
    
        C = np.random.rand(N, M) / M
        muC = 0.5
        sigC = 1/np.sqrt(12)
        muY = 0.5
        Y = np.random.randn(N, M) * sigY + muY
        H = np.random.randn(N, M, M) * sigH / M**(3/2)
        if Pdist == None:
            P = np.random.rand(N, M) * (sigP/M) * np.sqrt(12)
        else:
             if Pdist == 'expo':
                P = np.random.exponential(sigP/M,size=(N, M))
        
        Ss = (Sl + Sh*np.random.rand(N)) * M / N
        Rs = Rl + Rh*np.random.rand(M)
        muR = Rl + Rh/2 

        Js = np.zeros((N+M,N+M))
        Js[0:N,N:N+M] = np.diag(Ss) @ (C*Y + np.tensordot(Rs,H, axes=([0],[1])) + np.tensordot(H,Rs, axes=([2],[0])))
        Js[N:N+M,0:N] = P.T - np.diag(Rs) @ C.T
        if supply == 'lin':
            l = 0.1 + 0.9*np.random.rand(M)
            Js[N:N+M,N:N+M] = - np.diag(C.T @ Ss) - np.diag(l)
        else:
            g = 0.1 + 0.9*np.random.rand(M)
            Js[N:N+M,N:N+M] = - np.diag(g * Rs)

        E_J = np.linalg.eigvals(Js)
        E_Jr = E_J.real
        NU_J = len(E_Jr[E_Jr >= 1.0e-6]) # if devided by Nr or Ns

        rho = np.sqrt(1/(1 + sigP**2/muR**2/sigC**2)/(1 + sigY**2/muY**2 + sigY**2*muC**2/muY**2/sigC**2 + 2*sigH**2*muR**2/sigC**2/muY**2))

        return NU_J, rho

def N1(M, 
       sigP, 
       sigY, 
       sigH,
       muR = 0.5,
       muC = 0.5, 
       sigC = 1/np.sqrt(12),
       muY = 0.5):
    # function to calculate upper bound of the first stable phase
    return M/(1 + sigP**2/muR**2/sigC**2)/(1 + sigY**2/muY**2 + sigY**2*muC**2/muY**2/sigC**2 + 2*sigH**2*muR**2/sigC**2/muY**2)

def N2(M, 
       sigP, 
       sigY, 
       sigH,
       muR = 0.5,
       muC = 0.5, 
       sigC = 1/np.sqrt(12),
       muY = 0.5):
    # function to calculate lower bound of the second stable phase
    return M * (1 + sigP**2/muR**2/sigC**2) * (1 + sigY**2/muY**2 + sigY**2*muC**2/muY**2/sigC**2 + 2*sigH**2*muR**2/sigC**2/muY**2)

def criterion(x):
    # x is Ns/Nr array
    y = np.minimum(np.sqrt(x), np.sqrt(1/x))
    return y