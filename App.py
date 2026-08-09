import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# page config
st.set_page_config(page_title="Binomial Option Pricer", layout="wide")

st.title("Interactive Binomial Option Pricing and Risk Dashboard")
st.markdown("Price European and American options using the Cox-Ross-Rubinstein (CRR) model and compute Black-Scholes")

# Pricing and greeks
def binomial_lattice(K, T, S0, r, N, sd, optype, opstyle):
    dt = T / N
    
    # compute u and d
    u = np.exp(sd * np.sqrt(dt))
    d = 1 / u
    
    # computed values 
    q = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)

    # stock prices at maturity
    S = np.zeros((N + 1, N + 1))
    for j in range(0, N + 1):
        S[N, j] = S0 * (u**j) * (d**(N - j))

    # option payoffs at maturity
    C = np.zeros((N + 1, N + 1))
    for j in range(0, N + 1):
        if optype == 'p':
            C[N, j] = max(0, K - S[N, j])
        else:
            C[N, j] = max(0, S[N, j] - K)
            
    # CRR backward recursion
    for i in np.arange(N - 1, -1, -1):
        for j in range(0, i + 1):
            S[i, j] = S0 * (u**j) * (d**(i - j))
            hold = df * (q * C[i + 1, j + 1] + (1 - q) * C[i + 1, j])
            
            if opstyle == 'amer':
                intrinsic = (K - S[i, j]) if optype == 'p' else (S[i, j] - K)
                C[i, j] = max(hold, intrinsic)
            else:
                C[i, j] = hold
            
    return C[0, 0], S, C, u, d, q, df, dt


def blacks_price(S0, K, T, r, sd, optype):
    d1 = (np.log(S0 / K) + (r + 0.5 * sd**2) * T) / (sd * np.sqrt(T))
    d2 = d1 - sd * np.sqrt(T)

    if optype == 'p':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

    
def greeks(S0, K, T, r, sd, optype):
    d1 = (np.log(S0 / K) + (r + 0.5 * sd**2) * T) / (sd * np.sqrt(T))
    d2 = d1 - sd * np.sqrt(T)

    gamma = norm.pdf(d1) / (S0 * sd * np.sqrt(T))
    vega = (S0 * norm.pdf(d1) * np.sqrt(T)) / 100.0  # scaled per 1% vol change

    if optype == 'p':  # Put Option
        delta = norm.cdf(d1) - 1.0
        theta = ((- (S0 * norm.pdf(d1) * sd) / (2 * np.sqrt(T))) 
                 + (r * K * np.exp(-r * T) * norm.cdf(-d2))) / 365.0
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100.0
    else:  # Call Option
        delta = norm.cdf(d1)
        theta = ((- (S0 * norm.pdf(d1) * sd) / (2 * np.sqrt(T))) 
                 - (r * K * np.exp(-r * T) * norm.cdf(d2))) / 365.0
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100.0

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


# Sidebar inputs
S0 = st.sidebar.number_input("Stock Price (S0)", value=90.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T = st.sidebar.number_
