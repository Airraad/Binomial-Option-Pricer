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


# Form container in sidebar
with st.sidebar.form("pricer_form"):
    st.subheader("Model Inputs")
    S0 = st.number_input("Stock Price (S0)", value=90.0, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
    T = st.number_input("Price to Maturity (T)", value=0.5, step=0.1)                             
    sd = st.number_input("Volatility (Standard Deviation)", value=0.20, step=0.01)
    r = st.number_input("Risk Free Rate (r)", value=0.05, step=0.01)
    N = st.number_input("Amount of Steps", value=5, step=1)

    optype_str = st.selectbox("Option Type", ["Put", "Call"])
    opstyle_str = st.selectbox("Option Style", ["American", "European"])

    optype = 'c' if optype_str.lower() == "call" else 'p'
    opstyle = 'amer' if opstyle_str.lower() == "american" else 'eur'

    submitted = st.form_submit_button("Price Option", type="primary")

# Initialize session state flag
if "has_run" not in st.session_state:
    st.session_state["has_run"] = True

# Execute main calculations
if submitted or st.session_state["has_run"]:
    binomial_price, S, C, u, d, q, df, dt = binomial_lattice(K, T, S0, r, N, sd, optype, opstyle)
    bs_price = blacks_price(S0, K, T, r, sd, optype)
    greek_values = greeks(S0, K, T, r, sd, optype)

    # main panel - prices
    st.subheader(f"Option price: ${binomial_price:.4f}")
    st.caption(f"Theoretical Black-Scholes Price: ${bs_price:.4f}")
    st.caption(f"Difference: ${abs(binomial_price - bs_price):.4f}")

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
        st.write("Hover over any node to view node stats and exercise logic.")
        
        tree = go.Figure()

        # 1. Draw connecting lines between adjacent nodes
        for i in range(N):
            for j in range(i + 1):
                y_curr = j - i / 2.0
                
                y_up = (j + 1) - (i + 1) / 2.0
                tree.add_trace(go.Scatter(
                    x=[i, i + 1], y=[y_curr, y_up],
                    mode='lines', line=dict(color='gray', width=1),
                    showlegend=False, hoverinfo='none'
                ))
                
                y_down = j - (i + 1) / 2.0
                tree.add_trace(go.Scatter(
                    x=[i, i + 1], y=[y_curr, y_down],
                    mode='lines', line=dict(color='gray', width=1),
                    showlegend=False, hoverinfo='none'
                ))

        x_nodes, y_nodes, node_hover, node_labels = [], [], [], []

        for i in range(N + 1):
            for j in range(i + 1):
                x_nodes.append(i)
                y_nodes.append(j - i / 2.0)
                node_labels.append(f"${S[i, j]:.1f}" if N <= 10 else "")
            
                # intrinsic valuation
                intrinsic_val = max(S[i, j] - K, 0) if optype == 'c' else max(K - S[i, j], 0)

                # non-chalant lowercase decision logic & notes
                if i == N:
                    exercise_str = "expiration boundary"
                    note_str = "final node, just payoff at maturity"
                elif opstyle == 'amer' and C[i, j] == intrinsic_val and intrinsic_val > 0:
                    exercise_str = "exercise early"
                    note_str = "intrinsic beats holding, exercise here"
                elif opstyle == 'eur':
                    exercise_str = "hold (european)"
                    note_str = "can't exercise early anyway, just holding"
                else:
                    exercise_str = "hold"
                    note_str = "holding option has higher expected value"

                hover_text = (
                    f"<b>node step {i} | up state {j}</b><br>"
                    f"─────────────────────────────<br>"
                    f"stock price ($S$): <b>${S[i, j]:.2f}</b><br>"
                    f"option value ($C$): <b>${C[i, j]:.2f}</b><br>"
                    f"intrinsic payoff: <b>${intrinsic_val:.2f}</b><br>"
                    f"decision state: <b>{exercise_str}</b><br>"
                    f"<i>note: {note_str}</i>"
                )
                node_hover.append(hover_text)

        marker_size = max(12, 30 - N)
        tree.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text' if N <= 10 else 'markers',
            marker=dict(size=marker_size, color='#1f77b4'),
            text=node_labels,
            textposition="middle center",
            textfont=dict(color="white", size=max(7, 10 - N // 3)),
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
                font_size=12,
                bordercolor="#4b5563"
            )
        )
        st.plotly_chart(tree, use_container_width=True)

        # Separate metric section below graph
        st.divider()
        st.markdown("### Model & Node Parameters")
        st.caption("step size, transition multipliers, risk-neutral probabilities, and discount factors for the tree")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Time Step (dt)", f"{dt:.4f} yrs")
        m2.metric("Up Factor (u)", f"{u:.4f}")
        m3.metric("Down Factor (d)", f"{d:.4f}")
        m4.metric("RN Prob (p)", f"{q:.4f}")
        m5.metric("Step Discount (df)", f"{df:.4f}")
        st.divider()

    # TAB 2: CONVERGENCE GRAPH
    with tab2:
        st.write("Observation: As the number of steps ($N$) increases, the discrete Binomial Option Price converges toward the continuous Black-Scholes price.")

        if optype == 'p' and opstyle == 'amer':
            st.info("For American Put options, early exercise premium causes the Binomial tree price to converge slightly above the European Black Scholes line.")
        
        step_range = list(range(1, 101))
        binomial_prices = [binomial_lattice(K, T, S0, r, step, sd, optype, opstyle)[0] for step in step_range]

        conv = go.Figure()
        
        conv.add_trace(go.Scatter(
            x=step_range, 
            y=binomial_prices,
            mode='lines+markers', 
            name='Binomial Price',
            line=dict(color='#2ca02c', width=2)
        ))
        
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
                font_size=12,
                bordercolor="#4b5563"
            )
        )
        st.plotly_chart(conv, use_container_width=True)
