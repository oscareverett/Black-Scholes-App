# Black–Scholes Option Pricing Lab

An interactive quantitative finance dashboard for analysing option pricing, Greeks, volatility dynamics, and delta-hedged performance using the Black–Scholes framework.

Built in Python with Streamlit and Plotly, this project is designed as both an educational and analytical tool for exploring how option values respond to changes in market parameters and realised volatility.

🔗 **Live app:** https://blackscholespricinginterface.streamlit.app/

![Black–Scholes Option Pricing Lab Interface](images/Screenshot 2025-12-24 at 12.11.35 pm.png)

---

---

## Overview

This application implements the Black–Scholes option pricing model and extends it with interactive visualisations and simulations, allowing users to:

- Price European call and put options
- Explore payoff and P/L profiles at expiry and today
- Visualise option Greeks across spot prices
- Compare realised vs implied volatility using historical market data
- Construct implied volatility surfaces from option chains
- Simulate stock price paths under geometric Brownian motion
- Analyse delta-hedged vs unhedged P/L using Monte Carlo simulation


---

## Key Features

- **Black–Scholes pricing engine**  
  Analytical pricing of European options with dividend yield support

- **Greeks visualisation**  
  Delta, Gamma, Vega, and Theta plotted against underlying price

- **Payoff and P/L analysis**  
  Comparison of payoff at expiry vs mark-to-market value today

- **Implied volatility solver**  
  Numerical IV extraction from observed option prices

- **Volatility analysis**  
  Rolling realised volatility from historical prices vs model or implied volatility

- **Implied volatility surface**  
  3D IV surface built from live Yahoo Finance option chains

- **Monte Carlo hedging lab**  
  Simulation of realised volatility paths and dynamic delta hedging to study gamma–theta trade-offs

---

## Tech Stack

- **Python**
- **Streamlit** for interactive UI
- **Plotly** for high-quality interactive charts
- **NumPy / SciPy** for numerical methods
- **Pandas** for data handling
- **yfinance** for market and options data

---

## Use Cases

- Teaching and learning option pricing theory
- Exploring volatility mispricing and delta-hedging intuition
- Scenario analysis for option strategies
- Demonstrating applied financial mathematics in Python

---

## Notes

- The Black–Scholes model assumes constant volatility, log-normal price dynamics, and frictionless markets.  
- Option-chain coverage depends on Yahoo Finance data availability and may vary by ticker.

---

## Author

**Oscar Everett**  
Engineering & Financial Mathematics student  
🔗 LinkedIn: https://www.linkedin.com/in/oscareverett
