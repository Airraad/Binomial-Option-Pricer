import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

#page config
st.set_page_config(page_title="Binomial Option Pricer", layout="wide")

st.title("Interactive Binomial Option Pricing and Risk Dashboard")
st.markdown("Price European and American options using the Cox-Ross-Rubinstein (CRR) model and compute Black-Scholes")
            
#Pricing and greeks
def binomial_lattice (K,T,S0,r,N,sd,optype):
    
    dt = T/N
    
    #compute u and d
    u = np.exp(sd * np.sqrt(dt))
    d = 1/u
    
    #computed values 
    q = (np.exp(r*dt)-d)/(u-d)
    df= np.exp(-r*dt)

    #stock prices at maturity
    S = np.zeros((N+1, N+1))
    for j in range(0,N+1):
        S[N,j] = S0 * (u**j) * d**(N-j)

    # option payoffs
    C = np.zeros((N+1,N+1))
    for j in range(0,N+1):
        if optype == 'p':
            C[N, j] = max(0, K-S[N,j])
        else:
            C[N, j] = max(0, S[N,j] - K)
            
      # CRR recursion
    for i in np.arange(N-1,-1,-1):
        for j in range (0,i+1):
            S[i,j] = S0 * (u**j) * (d** (i-j))

            hold = df * ( q*C[i+1, j+1] + (1-q)*C[i+1,j])
            if optype == 'p':
                C[i,j] = max(hold, K - S[i,j])
            else:
                C[i,j] = max(hold,S[i,j] - K)
            
    return C[0,0], S, C

def blacks_price(S0, K, T, r, sd, optype):
    d1 = (np.log(S0 / K) + (r + 0.5 * sd**2) * T) / (sd * np.sqrt(T))
    d2 = d1 - sd * np.sqrt(T)

    if optype == 'p':
        price = K* np.exp(-r*T) * norm.cdf(-d2)-S0* norm.cdf(-d1)
    else:
        price= S0 * norm.cdf(d1) - K* np.exp(-r *T) * norm.cdf(d2)
    return price
    
def greeks(S0, K, T, r, sd, optype):
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
#Sidebar inputs
S0 = st.sidebar.number_input("Stock Price (S0)", value=105.0, step = 1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step = 1.0)
T = st.sidebar.number_input("Price to Maturity (T)", value=0.5, step = 0.1)                             
sd = st.sidebar.number_input("Volatility (Standard Deviation)", value=0.25, step = 0.01)
r = st.sidebar.number_input("Risk Free Rate (r)", value=0.08, step = 0.005)
N = st.sidebar.number_input("Amount of Steps", value=5, step = 1)

optype_str = st.sidebar.selectbox("Option Type", ["Call", "Put"])

if optype_str == "call":
    optype = 'c'
else:
    optype = 'p'
binomial_price, S, C = binomial_lattice (K,T,S0,r,N,sd,optype)
bs_price = blacks_price(S0, K, T, r, sd, optype)
greek_values = greeks(S0, K, T, r, sd, optype)

# main panel
#prices
st.subheader(f" Option price: {binomial_price:.4f}")
st.caption(f" Theoretical Black-Scholes Price:{bs_price}")
st.caption(f" Difference: {abs(binomial_price - bs_price):.4f}")

#greeks boxes
st.divider()
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Delta", f"{greek_values['delta']:.4f}")
g2.metric("Gamma", f"{greek_values['gamma']:.4f}")
g3.metric("vega", f"{greek_values['vega']:.4f}")
g4.metric("Theta", f"{greek_values['theta']:.4f}")
g5.metric("Rho", f"{greek_values['rho']:.4f}")

st.divider()

#interactive tabs for graphics
tab1, tab2 = st.tabs(["Binomial Lattice Graphic", "Black-Scholes Convergence Plot"])

# Binomial Lattice Graphic
with tab1:
    st.write("Hover over any node to view stock price and option payoff.")

    tree = go.Figure()

    for i in range(N):
        for j in range(i+1):
            x_cords = [i, i+1, None, i, i+1]
            y_cords = [j, j, None, j, j+1]
            tree.add_trace(go.Scatter(
                x=x_cords, y=y_cords,
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False, 
                hoverinfo='none'
            ))
    # add markers
    x_nodes, y_nodes, node_text, node_labels = [], [], [], []
    for i in range(N+1):
        for j in range(i+1):
            x_nodes.append(i)
            y_nodes.append(j)
            node_labels.append(j)
            node_text.append(f"Step {i}<br>Stock: ${S[i, j]:.2f}<br>Option:${C[i, j]:.2f}")

    tree.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(size=24, color = '#ba3ec1'),
        text = node_labels,
        textposition = "middle center",
        textfont = dict(color = "white", size = 9),
        hoverinfo='text',
        hovertext = node_text,
        showlegend = False
    ))

    tree.update_layout(
        title = "Binomial Price Tree Lattice",
        xaxis = dict(title="Time Step", tickmode='linear', dtick=1),
        yaxis=dict(showticklabels = False),
        height = 500,
        margin = dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(tree, use_container_width=True)

    #convergence graph
with tab2:
    st.write("Observation: As the number of steps increases, the Binoial Option Price converges towards the continuous Black Scholes Price.")

    step_range = list(range(1, 101))
    binomial_prices = [binomial_price(K,T,S0,r,step,sd,optype)[0] for step in step_range]

    conv=go.Figure()
    conv.add_trace(go.Scatter(
        x=step_range, y=binomial_prices,
        mode='lines+markers', name='Binomial Price',
        line=dict(color='2B6621', width = 2)

    ))
    conv.add_trace(go.Scatter(
        x=[1, 100], y=[bs_price, bs_price],
        mode = 'lines', name = 'Black Scholes Benchmark',
        line = dict(color = 'red', dash = 'dash', width = 2)

    ))

    conv.update_layout(
        title = "Binomial Model Convergence",
        xaxis_title = "Number of Steps",
        yaxis_title = "Option Price",
        height = 450,
        legend = dict(x=0.7, y = 0.1)
    )
    st.plotly_chart(conv,  use_container_width=True)
                    

            
