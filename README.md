# Binomial Option Pricing & Risk Dashboard

An interactive dashboard for option pricing and risk analysis built with Python, Streamlit, and Plotly. 

It calculates European and American options using the Cox-Ross-Rubinstein (CRR) binomial model and displays how discrete lattice tree values converge into continuous Black-Scholes prices as step count increases.



Core Features
Dual Pricing Engine: Runs discrete CRR binomial lattice trees and continuous Black-Scholes formulas.

Exercise Styles: Supports both American options (checking early exercise value against continuation value at every node) and European options.

Interactive Lattice Tree: Dynamic Plotly graph that displays state stock prices, option values, discount factors, and node decision logic on hover.

Greeks Suite: Calculates Delta, Gamma, Vega, daily Theta, and Rho in real time.

Convergence Plot: Visualizes how discrete binomial step prices smooth out toward the Black-Scholes benchmark as steps grow from 1 to 100.




Math & Parameters

Parameters used for the Cox-Ross-Rubinstein tree logic:

Up Factor (u): u = exp(volatility * sqrt(dt))

Down Factor (d): d = 1 / u

Risk-Neutral Probability (p): p = (exp(r * dt) - d) / (u - d)

Step Discount Factor (df): df = exp(-r * dt)



How to Run Locally

1. Clone the repository and navigate to the directory:

In bash paste:

  git clone (https://github.com/airraad/binomial-option-pricer.git)
  cd binomial-option-pricer

2. Install dependencies in bash
   
  pip install stremlit numpy scipy plotly
  
3. Then launch app from bash

  streamlit run App.py


