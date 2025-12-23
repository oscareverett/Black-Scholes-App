import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime

def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    return C



# Monte Carlo simulation for average P/L
def simulate_average_pl(runs=1000000, sigma=0.3, r=0.05):
    premium = black_scholes_call(100, 100, sigma, r, 1)
    total_pl = 0
    for _ in range(runs):
        path = qf.simulations.GeometricBrownianMotion(100, r, sigma, 1/252, 1)
        payoff = max(path.simulated_path[-1] - 100, 0)
        discounted_payoff = np.exp(-r * 1) * payoff  # discount back to t=0
        terminal_pl = discounted_payoff - premium
        total_pl += terminal_pl
    return total_pl / runs

import qfin as qf

# --- Live ASX option quote utilities using yfinance ---

def get_live_option_quote(ticker="CBA.AX"):
    """
    Fetch a real ASX option quote for the given underlying using yfinance.
    We:
    - get the latest underlying price
    - choose the nearest expiry
    - pick the call with strike closest to spot
    - extract bid/ask and strike
    Returns: S0, K, T (in years), bid, ask
    """
    tk = yf.Ticker(ticker)
    # Get latest underlying price
    hist = tk.history(period="5d")
    if hist.empty:
        raise ValueError(f"No price history available for {ticker}")
    S0 = float(hist["Close"].dropna().iloc[-1])

    # Get available option expiries
    if not tk.options:
        # Fallback: no listed options available via yfinance for this ticker
        # Construct a simple synthetic ATM option one year out using a volatility guess
        sigma_guess = 0.2
        r = 0.05
        T = 1.0
        K = float(round(S0, 2))
        model_price = black_scholes_call(S0, K, sigma_guess, r, T)
        bid = model_price * 0.98
        ask = model_price * 1.02
        print(f"Warning: no option chain available for {ticker} via yfinance. Using synthetic model-based quote instead.")
        return S0, K, T, bid, ask

    expiry = tk.options[0]  # nearest expiry as a string like '2025-12-18'

    chain = tk.option_chain(expiry)
    calls = chain.calls
    if calls.empty:
        raise ValueError(f"No call options available for {ticker} at expiry {expiry}")

    # Pick the call with strike closest to spot
    idx = (calls["strike"] - S0).abs().idxmin()
    row = calls.loc[idx]
    K = float(row["strike"])
    bid = float(row.get("bid", 0.0))
    ask = float(row.get("ask", 0.0))
    last = float(row.get("lastPrice", 0.0))

    # If ask is missing or zero, fall back to last price
    if ask <= 0 and last > 0:
        ask = last

    # Time to expiry in years
    T = max((datetime.strptime(expiry, "%Y-%m-%d").date() - datetime.today().date()).days / 365.0, 1.0 / 252.0)

    return S0, K, T, bid, ask

# --- ASX data utilities: use real price history to estimate drift and volatility ---

def load_asx_data(ticker="CBA.AX", start="2015-01-01", end=None):
    """
    Download ASX price data using yfinance and compute:
    - adjusted close prices
    - daily log returns
    - annualised historical volatility
    - annualised drift (mean return)
    """
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    # Flatten MultiIndex columns if present (yfinance can return multi-level columns)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # Prefer adjusted close if available, otherwise fall back to close
    if "Adj Close" in data.columns:
        prices = data["Adj Close"].dropna()
    elif "Close" in data.columns:
        prices = data["Close"].dropna()
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
    returns = np.log(prices / prices.shift(1)).dropna()
    hist_vol = returns.std() * np.sqrt(252)
    mu = returns.mean() * 252
    return prices, returns, hist_vol, mu

def simulate_edge_trades_bootstrap(num_trades, returns, S0=100, K=100, ask=14.10, T=1.0, contract_size=100):
    """
    Simulate repeated option trades using bootstrapped historical ASX returns
    instead of idealised Geometric Brownian Motion. This gives a more realistic
    P/L distribution based on actual market behaviour.
    """
    premium = ask * contract_size
    pls = []
    n_steps = 252  # trading days in a year
    ret_values = returns.values
    for _ in range(num_trades):
        sim_returns = np.random.choice(ret_values, size=n_steps, replace=True)
        ST = S0 * np.exp(sim_returns.cumsum()[-1])
        payoff = max(ST - K, 0) * contract_size
        pls.append(payoff - premium)
    pls = np.array(pls)
    avg_pl = pls.mean()
    total_pl = pls.sum()
    total_cost = premium * num_trades
    print(f"(Bootstrap) Total capital spent buying {num_trades} contracts (AUD): {total_cost:.2f}")
    print(f"(Bootstrap) Average P/L per trade (AUD): {avg_pl:.2f}")
    print(f"(Bootstrap) Total P/L over {num_trades} trades (AUD): {total_pl:.2f}")
    print(f"(Bootstrap) Return on capital (%): {100 * total_pl / total_cost:.4f}")
    return pls

# --- How to Make Money using the Black-Scholes Model ---
# We compare the theoretical model price to a real market maker quote
def show_trade_edge(sigma=0.3, ticker="CBA.AX"):
    # Get a real ASX option quote (ATM-ish call, nearest expiry)
    S0, K, T, mm_bid, mm_ask = get_live_option_quote(ticker)
    model_price = black_scholes_call(S0, K, sigma, 0.05, T)
    edge_per_option = model_price - mm_ask  # assume we lift the ask to buy

    print(f"\n--- Live Option Quote for {ticker} ---")
    print(f"Spot price S0: {S0:.2f}")
    print(f"Strike K: {K:.2f}")
    print(f"Time to expiry T (years): {T:.4f}")
    print(f"Model Call Price (Black-Scholes): {model_price:.4f}")
    print(f"Market Maker Quote (bid @ ask): {mm_bid:.2f} @ {mm_ask:.2f}")
    print(f"Trade Edge per option (model - ask): {edge_per_option:.4f} AUD")

    return S0, K, T, model_price, mm_bid, mm_ask, edge_per_option

# Simulate trading this edge many times to see how volume realises the edge
def simulate_edge_trades(num_trades=100000, S0=100, K=100, sigma=0.3, r=0.05, T=1.0, ask=14.10):
    """Simulate repeated option trades where we buy at the market maker's ask
    and the underlying follows the model dynamics. Shows how higher trade
    volume lets us take advantage of a small positive edge.
    """
    contract_size = 100  # standard equity option contract
    premium = ask * contract_size
    pls = []
    for _ in range(num_trades):
        path = qf.simulations.GeometricBrownianMotion(S0, r, sigma, 1/252, T)
        payoff = max(path.simulated_path[-1] - K, 0) * contract_size
        pls.append(payoff - premium)
    pls = np.array(pls)
    avg_pl = pls.mean()
    total_pl = pls.sum()
    total_cost = premium * num_trades
    print(f"Premium per contract (AUD): {premium:.2f}")
    print(f"Total capital spent buying {num_trades} contracts (AUD): {total_cost:.2f}")
    print(f"Average P/L per trade (AUD): {avg_pl:.2f}")
    print(f"Total P/L over {num_trades} trades (AUD): {total_pl:.2f}")
    print(f"Return on capital (%): {100 * total_pl / total_cost:.4f}")
    return pls

if __name__ == "__main__":
    # 1) Risk-neutral sanity check: average P/L should be close to zero
    avg_pl_rn = simulate_average_pl(100000, sigma=0.3, r=0.05)
    print("Average P/L over 100000 simulations (risk-neutral):", avg_pl_rn)

    # Load real ASX data and estimate historical volatility and drift
    prices, returns, hist_vol, mu = load_asx_data("CBA.AX", start="2015-01-01")
    print("\n--- ASX Data (CBA.AX) ---")
    print(f"Historical annualised volatility (sigma): {hist_vol:.4f}")
    print(f"Historical annualised drift (mu): {mu:.4f}")

    # 2) Show the trade edge from Black-Scholes vs market quote
    S0, K, T, model_price, mm_bid, mm_ask, edge_per_option = show_trade_edge(sigma=hist_vol)

    import matplotlib.pyplot as plt

    trade_counts = [10, 100, 1000, 10000]
    pl_results = []

    for trades in trade_counts:
        print(f"\nSimulating {trades} trades with model-consistent dynamics:")
        pls = simulate_edge_trades(num_trades=trades, S0=S0, K=K, sigma=hist_vol, r=mu, T=T, ask=mm_ask)
        pl_results.append(np.sum(pls))

    # Also simulate using bootstrapped historical ASX returns for more realism
    pl_results_bootstrap = []
    for trades in trade_counts:
        print(f"\n(Bootstrap) Simulating {trades} trades using ASX return bootstrapping:")
        pls_boot = simulate_edge_trades_bootstrap(num_trades=trades, returns=returns, S0=S0, K=K, ask=mm_ask, T=T)
        pl_results_bootstrap.append(np.sum(pls_boot))

    # Plot the total P/L vs number of trades
    plt.figure()
    plt.title("Total P/L vs Number of Trades (Log Scale)")
    plt.plot(trade_counts, pl_results, marker='o', label="GBM (model dynamics)")
    plt.plot(trade_counts, pl_results_bootstrap, marker='x', label="Bootstrapped ASX returns")
    plt.xscale('log')
    plt.xlabel("Number of Trades (log scale)")
    plt.ylabel("Total P/L (AUD)")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()