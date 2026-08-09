import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# page config
st.set_page_config(page_title="Binomial Option Pricer", layout="wide")

st.title("Interactive Binomial Option Pricing and Risk Dashboard")
st.markdown("Price European and American options using the Cox-Ross-Rubinstein (CRR) model and compute Black-Scholes")

# Pricing and greeks
def binomial_lattice(K, T, S0, r, N, sd, optype):
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

    # option payoffs
    C = np.zeros((N + 1, N + 1))
    for j in range(0, N + 1):
        if optype == 'p':
            C[N, j] = max(0, K - S[N, j])
        else:
            C[N, j] = max(0, S[N, j] - K)
            
    # CRR recursion
    for i in np.arange(N - 1, -1, -1):
        for j in range(0, i + 1):
            S[i, j] = S0 * (u**j) * (d**(i - j))

            hold = df * (q * C[i + 1, j + 1] + (1 - q) * C[i + 1, j])
            if optype == 'p' and opstyle == 'amer': 
                C[i, j] = max(hold, K - S[i, j])
            elif opstyle == 'eur':
                C[i, j] = max(hold, S[i, j] - K)
            
    return {C[0, 0], S, C,}


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

    if optype == 'p':  # put Option
        delta = norm.cdf(d1) - 1.0
        theta = (- (S0 * norm.pdf(d1) * sd) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100.0  # scaled per 1% rate change
    else:  # call Option
        delta = norm.cdf(d1)
        theta = (- (S0 * norm.pdf(d1) * sd) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100.0

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


# Sidebar inputs
S0 = st.sidebar.number_input("Stock Price (S0)", value=105.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T = st.sidebar.number_input("Price to Maturity (T)", value=0.5, step=0.1)                             
sd = st.sidebar.number_input("Volatility (Standard Deviation)", value=0.25, step=0.01)
r = st.sidebar.number_input("Risk Free Rate (r)", value=0.08, step=0.01)
N = st.sidebar.number_input("Amount of Steps", value=5, step=1)

optype_str = st.sidebar.selectbox("Option Type", ["Call", "Put"])
opstyle_str = st.sidebar.selectbox("Option Style", ["American", "European"])


if optype_str.lower() == "call":
    optype = 'c'
else:
    optype = 'p'

if opstyle_str.lower() == "american":
    opstyle = 'amer'
elif opstyle_str.lower() == "european":
    opstyle = 'eur'


binomial_price, S, C = binomial_lattice(K, T, S0, r, N, sd, optype)
bs_price = blacks_price(S0, K, T, r, sd, optype)
greek_values = greeks(S0, K, T, r, sd, optype)

# main panel
# prices
st.subheader(f"Option price: {binomial_price:.4f}")
st.caption(f"Theoretical Black-Scholes Price: {bs_price:.4f}")
st.caption(f"Difference: {abs(binomial_price - bs_price):.4f}")

# greeks boxes
st.divider()
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Delta", f"{greek_values['delta']:.4f}")
g2.metric("Gamma", f"{greek_values['gamma']:.4f}")
g3.metric("Vega", f"{greek_values['vega']:.4f}")
g4.metric("Theta", f"{greek_values['theta']:.4f}")
g5.metric("Rho", f"{greek_values['rho']:.4f}")

st.divider()

# interactive tabs for graphics
tab1, tab2 = st.tabs(["Binomial Lattice Graphic", "Black-Scholes Convergence Plot"])

# TAB 1: BINOMIAL LATTICE GRAPHIC
with tab1:
    st.write("Hover over any node to view Stock Price ($S$) and Option Payoff ($C$).")
    
    tree = go.Figure()

    # 1. Draw connecting lines between adjacent nodes (centered layout)
    for i in range(N):
        for j in range(i + 1):
            y_curr = j - i / 2.0
            
            
            y_up = (j + 1) - (i + 1) / 2.0
            tree.add_trace(go.Scatter(
                x=[i, i + 1],
                y=[y_curr, y_up],
                mode='lines',
                line=dict(color='gray',
                width=1),
                showlegend=False,
                hoverinfo='none'
            ))
            
            
            y_down = j - (i + 1) / 2.0
            tree.add_trace(go.Scatter(
                x=[i, i + 1],
                y=[y_curr, y_down],
                mode='lines',
                line=dict(color='gray',
                width=1),
                showlegend=False,
                hoverinfo='none'
            ))

    
    x_nodes, y_nodes, node_hover, node_labels = [], [], [], []
#node placement and labels, nested loop allow to iterate through steps and time.
    for i in range(N + 1):
        for j in range(i + 1):
            x_nodes.append(i)
            y_nodes.append(j - i / 2.0)
            #node labels format the price at that node
            node_labels.append(f"${S[i, j]:.1f}")
        
            # Calculate intrinsic value at this node, calculating immediate payoff if exercised at this time.
            if optype == 'c':
                intrinsic_val = max(S[i, j] - K, 0)   
            else:
                intrinsic_val = max(K - S[i, j], 0)
            
            if i == N:
                exercise_str = "Expiration"
            elif opstyle == 'amer' and C[i, j] == intrinsic_val and intrinsic_val > 0:
                exercise_str = "Exercise Early" 
            elif opstyle == 'eur':
                exercise_str = "Hold (European)"
            else:
                exercise_str = "Hold"
            # Rich Tooltip Construction
            hover_text = (
                f"<b>Node (Step {i}, Up {j})</b><br>"
                f"Stock Price ($S$): <b>${S[i, j]:.2f}</b><br>"
                f"Option Value ($C$): <b>${C[i, j]:.2f}</b><br>"
                f"Intrinsic Payoff: <b>${intrinsic_val:.2f}</b><br>"
                f"Decision State: <b>{exercise_str}</b>"
            )
            node_hover.append(hover_text)
    marker_size = max(12, 30-N)
    tree.add_trace(go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers+text',
        marker=dict(size=28, color='#1f77b4'),
        text=node_labels,
        textposition="middle center",
        textfont=dict(color="white", size=9),
        hoverinfo='text',
        hovertext=node_hover,
        showlegend=False
    ))

    tree.update_layout(
        title="Binomial Price Tree Lattice",
        xaxis=dict(title="Time Step", tickmode='linear', dtick=1, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hoverlabel=dict(
            bgcolor="#1f2937",
            font_color="#ffffff",
            font_size=13,
            bordercolor="#4b5563"
    )
)
        
    
    st.plotly_chart(tree, use_container_width=True)

# TAB 2: CONVERGENCE GRAPH
with tab2:
    st.write("Observation: As the number of steps ($N$) increases, the discrete Binomial Option Price converges toward the continuous Black-Scholes price.")

    if optype == 'p':
        st.info("For American Put options, early exercise premium causes the Binomial tree price to converge slightly above the European Black Scholes line.")
    
    step_range = list(range(1, 101))
    binomial_prices = [binomial_lattice(K, T, S0, r, step, sd, optype)[0] for step in step_range]

    conv = go.Figure()
    
    # binomial line
    conv.add_trace(go.Scatter(
        x=step_range, 
        y=binomial_prices,
        mode='lines+markers', 
        name='Binomial Price',
        line=dict(color='#2ca02c', width=2)
    ))
    
    # Black Scholes benchmark line
    conv.add_trace(go.Scatter(
        x=[1, 100], 
        y=[bs_price, bs_price],
        mode='lines', 
        name='Black-Scholes Benchmark',
        line=dict(color='red', dash='dash', width=2)
    ))

    conv.update_layout(
        title="Binomial Model Convergence to Black-Scholes Price",
        xaxis_title="Number of Steps (N)",
        yaxis_title="Option Price ($)",
        height=450,
        legend=dict(x=0.7, y=0.1),
        hoverlabel=dict(
            bgcolor="#1f2937",
            font_color="#ffffff",
            font_size=13,
            bordercolor="#4b5563"
        )
    )
    st.plotly_chart(conv, use_container_width=True)
st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Delta", f"{greek_values['delta']:.4f}")
m2.metric("Gamma", f"{greek_values['gamma']:.4f}")
m3.metric("Vega", f"{greek_values['vega']:.4f}")
m4.metric("Theta", f"{greek_values['theta']:.4f}")
m5.metric("Rho", f"{greek_values['rho']:.4f}")

st.divider()
