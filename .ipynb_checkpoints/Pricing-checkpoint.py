import numpy as np
from scipy.stats import norm

def binomial_lattice (K,T,S0,r,N,sd,optype):
    
    dt = T/N
    
    #compute u and d
    u = np.exp(sd * np.sqrt(dt))
    d = 1/u
    
    #computed values 
    q = (np.exp(r*dt)-d)/(u-d)
    df= np.exp(-r*dt)

    #stock prices at maturity
    S = np.zeros(N+1)
    for j in range(0,N+1):
        S[j] = S0 * u**j * d**(N-j)

    # option payoffs
    C = np.zeros(N+1)
    for j in range(0,N+1):
        if optype in [0, 'p', 'put']:
            C[j] = max(0, K-S[j])
        else:
            C[j] = max(0, S[j] - K)
            
      # CRR recursion
    for i in np.arange(N-1,-1,-1):
        for j in range (0,i+1):
            S = S0 * u**j * d** (i-j)
            C[j] = df * ( q*C[j+1] + (1-q)*C[j])
            if optype in [0, 'p', 'put']:
                C[j] = max(C[j], K - S)
            else:
                C[j] = max(C[j],S - K)
            
    return C[0]

import numpy as np
from scipy.stats import norm

def greeks(S0, K, T, r, sd, optype=0):
    # Calculate d1 and d2
    d1 = (np.log(S0 / K) + (r + 0.5 * sd**2) * T) / (sd * np.sqrt(T))
    d2 = d1 - sd * np.sqrt(T)

    
    gamma = norm.pdf(d1) / (S0 * sd * np.sqrt(T))
    
    
    vega = S0 * norm.pdf(d1) * np.sqrt(T)

    
    if optype in [0, 'p', 'put']:  # PUT Option
        delta = norm.cdf(d1) - 1.0
        theta = (- (S0 * norm.pdf(d1) * sd) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
    else:  # CALL Option
        delta = norm.cdf(d1)
        theta = (- (S0 * norm.pdf(d1) * sd) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


if __name__ == "__main__":
    S0, K, T, r, sd, N = 105, 100, 0.5, 0.04, 0.25, 4
    
    price = binomial_lattice(K, T, S0, r, N, sd, optype='c')
    print(f"Calculated Option Price: ${price:.4f}\n")
    
    metrics = greeks(S0, K, T, r, sd, optype='c')
    
    for greek, value in metrics.items():
        print(f"for greeks:{greek.capitalize():<6}: {value:.4f}")
    
