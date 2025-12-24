import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone

from scipy.stats import norm

# -----------------------------
# Styling constants
# -----------------------------
FONT_FAMILY = "Inter, system-ui, -apple-system, BlinkMacSystemFont"
TEXT_DARK = "#111827"
GRID_LIGHT = "rgba(0,0,0,0.15)"
ZEROLINE = "rgba(0,0,0,0.25)"

# --- Plotly white-theme helper for consistent chart styling ---
def apply_white_plotly_theme(fig, height: int = 420):
    """Force a consistent white theme for Plotly figures (so px and go look identical)."""
    # Shorter, safe extraction of existing title text
    existing_title = getattr(getattr(fig.layout, "title", None), "text", "") or ""
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=TEXT_DARK, family=FONT_FAMILY),
        title=dict(text=existing_title, font=dict(color=TEXT_DARK, family=FONT_FAMILY)),
        xaxis=dict(
            color=TEXT_DARK,
            tickfont=dict(color=TEXT_DARK, family=FONT_FAMILY),
            title=dict(font=dict(color=TEXT_DARK, family=FONT_FAMILY)),
            gridcolor=GRID_LIGHT,
            zerolinecolor=ZEROLINE,
        ),
        yaxis=dict(
            color=TEXT_DARK,
            tickfont=dict(color=TEXT_DARK, family=FONT_FAMILY),
            title=dict(font=dict(color=TEXT_DARK, family=FONT_FAMILY)),
            gridcolor=GRID_LIGHT,
            zerolinecolor=ZEROLINE,
        ),
        margin=dict(l=40, r=10, t=40, b=30),
        legend=dict(font=dict(color=TEXT_DARK, family=FONT_FAMILY)),
    )
    return fig


# --- Plotly axis/text helpers for concision and consistency ---
def enforce_black_axes(fig, x_title: str = None, y_title: str = None):
    """Force axis titles/ticks to pure black for readability across themes."""
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"))
    fig.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"))
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str, title: str = "", height: int = 320, x_title: str = None, y_title: str = None):
    """Small wrapper so all line charts share styling."""
    fig = px.line(df, x=x, y=y, title=title)
    apply_white_plotly_theme(fig, height=height)
    enforce_black_axes(fig, x_title=x_title, y_title=y_title)
    return fig

try:
    import yfinance as yf
except Exception:
    yf = None


# -----------------------------
# Black–Scholes core functions
# -----------------------------

def d1_d2(S, K, T, r, q, sigma):
    """Helper to compute d1 and d2, including continuous dividend yield q."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_call_price(S, K, T, r, q, sigma):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return max(S - K, 0.0)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, q, sigma):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return max(K - S, 0.0)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def bs_call_delta(S, K, T, r, q, sigma):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return 1.0 if S > K else 0.0
    return np.exp(-q * T) * norm.cdf(d1)

def bs_put_delta(S, K, T, r, q, sigma):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return -1.0 if S < K else 0.0
    return np.exp(-q * T) * (norm.cdf(d1) - 1.0)

def bs_gamma(S, K, T, r, q, sigma):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return 0.0
    return (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, q, sigma):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return 0.0
    # Market convention: per 1% (0.01) change in sigma
    return 0.01 * S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def bs_call_theta(S, K, T, r, q, sigma):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return 0.0
    term1 = - (S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
    term3 = - r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 + term2 + term3


def bs_put_theta(S, K, T, r, q, sigma):
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    if d1 is None:
        return 0.0
    term1 = - (S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = - q * S * np.exp(-q * T) * norm.cdf(-d1)
    term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return term1 + term2 + term3


# -----------------------------
# Monte Carlo simulation
# -----------------------------

def simulate_gbm_paths(S0, T, r, q, sigma, n_steps, n_paths, seed=42):
    """Simulate GBM price paths under risk-neutral measure."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = drift + diffusion * Z

    log_paths = np.cumsum(log_returns, axis=1)
    S_paths = S0 * np.exp(log_paths)
    S_paths = np.hstack([np.full((n_paths, 1), S0), S_paths])
    return S_paths

# --- Option delta helper for MC hedging ---
def option_delta_at(S, K, T, r, q, sigma, opt_type):
    if opt_type == "Call":
        return bs_call_delta(S, K, T, r, q, sigma)
    return bs_put_delta(S, K, T, r, q, sigma)


# -----------------------------
# Implied volatility solver
# -----------------------------
def implied_volatility(market_price, S, K, T, r, q, opt_type,
                       sigma_low=1e-6, sigma_high=5.0, tol=1e-6, max_iter=200):
    """Implied vol via bisection: find sigma such that BS_price(sigma) = market_price.

    Returns sigma (float) or None if inputs are invalid or market_price is out of bounds.
    """
    if market_price is None or market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    def bs_price(sig):
        if opt_type == "Call":
            return bs_call_price(S, K, T, r, q, sig)
        return bs_put_price(S, K, T, r, q, sig)

    low_val = bs_price(sigma_low)
    high_val = bs_price(sigma_high)

    if market_price < low_val or market_price > high_val:
        return None

    lo, hi = sigma_low, sigma_high
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mid_val = bs_price(mid)

        if abs(mid_val - market_price) < tol:
            return mid

        if mid_val < market_price:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


# -----------------------------
# Payoff helpers
# -----------------------------

def call_payoff(S, K):
    return np.maximum(S - K, 0.0)

def put_payoff(S, K):
    return np.maximum(K - S, 0.0)


# -----------------------------
# Market data helpers
# -----------------------------

@st.cache_data(show_spinner=False)
def fetch_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV history using yfinance.

    yfinance may return either:
    - Single-index columns: ['Open','High','Low','Close','Volume']
    - MultiIndex columns: level 0 is OHLCV, level 1 is ticker

    This helper normalises output to a DataFrame containing a single 'Close' column.
    """
    if yf is None:
        return pd.DataFrame()

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Normalise MultiIndex columns: prefer the 'Close' field
    if isinstance(df.columns, pd.MultiIndex):
        # If 'Close' is a top-level field, pull it out
        if "Close" in df.columns.get_level_values(0):
            close_df = df.loc[:, ("Close",)].copy()
            # close_df may still have one or more ticker columns; take the first
            if isinstance(close_df, pd.DataFrame):
                close_series = close_df.iloc[:, 0]
            else:
                close_series = close_df
            out = pd.DataFrame({"Close": close_series})
            out = out.dropna(subset=["Close"]).copy()
            return out
        else:
            return pd.DataFrame()

    # Single-index columns
    if "Close" not in df.columns:
        return pd.DataFrame()

    out = df[["Close"]].copy()
    out = out.dropna(subset=["Close"]).copy()
    return out


def realised_volatility_from_close(close: pd.Series, window: int = 20, annualisation: int = 252) -> pd.Series:
    """Rolling realised volatility from log returns, annualised."""
    close = close.dropna()
    if len(close) < window + 2:
        return pd.Series(dtype=float)
    log_ret = np.log(close / close.shift(1)).dropna()
    rv = log_ret.rolling(window).std() * np.sqrt(annualisation)
    return rv

# -----------------------------
# Cached Yahoo option-chain helpers
# -----------------------------

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_option_expiries(ticker: str):
    if yf is None:
        return []
    try:
        t = yf.Ticker(ticker)
        return list(getattr(t, "options", []) or [])
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_option_chain(ticker: str, expiry: str):
    """Return (calls_df, puts_df) for a given expiry date string."""
    if yf is None:
        return pd.DataFrame(), pd.DataFrame()
    try:
        t = yf.Ticker(ticker)
        oc = t.option_chain(expiry)
        calls = oc.calls.copy() if hasattr(oc, "calls") else pd.DataFrame()
        puts = oc.puts.copy() if hasattr(oc, "puts") else pd.DataFrame()
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_spot_price_yahoo(ticker: str):
    """Best-effort spot price for surface calculations."""
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        # Prefer fast_info if available
        fi = getattr(t, "fast_info", None)
        if fi and isinstance(fi, dict):
            lp = fi.get("last_price", None)
            if lp is not None and lp > 0:
                return float(lp)
        # Fallback: 1d history
        h = t.history(period="5d", interval="1d", auto_adjust=True)
        if h is not None and len(h) > 0 and "Close" in h.columns:
            return float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Option Mechanics Lab (V1)", layout="wide")

st.markdown(
    """
    <style>
    /* Make tab labels larger, bolder, and more distinct */
    div[data-testid="stTabs"] button {
        font-size: 1.25rem;
        font-weight: 800;
        padding: 12px 22px;
        margin-right: 6px;
        color: #cbd5e1;
        border-radius: 10px 10px 0 0;
        transition: all 120ms ease-in-out;
    }

    /* Active tab */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: white;
        background: linear-gradient(135deg, #ef4444, #fb7185);
    }

    /* Hover effect */
    div[data-testid="stTabs"] button:hover {
        color: white;
        background-color: rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Black–Scholes Option Pricing Lab")


# Make sidebar expander titles larger + bold
st.markdown(
    """
    <style>
    /* Sidebar expander headers */
    section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary p {
        font-size: 1.15rem;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

LINKEDIN_URL = "https://www.linkedin.com/in/oscareverett"  # <- replace with your LinkedIn

st.sidebar.markdown("## 📈 Black–Scholes Model")
st.sidebar.markdown(f"[Created by Oscar Everett]({LINKEDIN_URL})")

# Sidebar – inputs (grouped into dropdown sections)
with st.sidebar.expander("Option parameters", expanded=True):
    opt_type = st.selectbox("Option type", ["Call", "Put"])
    S0 = st.number_input("Current share price", min_value=0.01, value=100.0, step=1.0, format="%.2f")
    K = st.number_input("Strike price", min_value=0.01, value=100.0, step=1.0, format="%.2f")
    T = st.number_input("Time to maturity (years)", min_value=1e-6, value=0.5, step=0.01, format="%.2f")
    r_pct = st.number_input(
        "Risk-free interest rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=2.00,
        step=0.10,
        format="%.2f",
    )
    q_pct = st.number_input(
        "Dividend yield (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.00,
        step=0.10,
        format="%.2f",
    )
    sigma_input_pct = st.number_input(
        "Volatility (%)",
        min_value=0.01,
        max_value=500.0,
        value=25.00,
        step=0.10,
        format="%.2f",
    )
    qty = st.number_input("Position (+ = long, - = short)", value=1, step=1)

# Convert % inputs to decimals for the model
r = r_pct / 100.0
q = q_pct / 100.0
sigma_input = sigma_input_pct / 100.0

with st.sidebar.expander("Delta hedging", expanded=False):
    shares_owned = st.number_input(
        "Shares owned (positive = long, negative = short)",
        value=0.0,
        step=1.0,
        format="%.4f",
    )
    auto_delta_hedge = st.checkbox(
        "Auto hedge using -Δ shares",
        value=False,
    )

with st.sidebar.expander("Implied volatility", expanded=False):
    market_price = st.number_input(
        "Observed option price (per option)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.4f",
    )
    use_iv = st.checkbox("Use implied vol as σ", value=False)

    iv = implied_volatility(market_price, S0, K, T, r, q, opt_type)
    if iv is None:
        st.caption("IV: - (enter a valid market price)")
    else:
        st.caption(f"IV ≈ {iv:.4f} ({iv*100:.2f}%)")

# Choose volatility used by the model everywhere below
if use_iv and (iv is not None):
    sigma = iv
else:
    sigma = sigma_input

# Build a grid of spot prices for diagrams
S_min = max(0.1, 0.5 * K)
S_max = 1.5 * K
S_grid = np.linspace(S_min, S_max, 80)

#
# -----------------------------
# Compute prices and Greeks over grid
# -----------------------------

if opt_type == "Call":
    payoff_grid = qty * call_payoff(S_grid, K)
    price_grid = qty * np.array([bs_call_price(S, K, T, r, q, sigma) for S in S_grid])
    delta_grid = qty * np.array([bs_call_delta(S, K, T, r, q, sigma) for S in S_grid])
else:
    payoff_grid = qty * put_payoff(S_grid, K)
    price_grid = qty * np.array([bs_put_price(S, K, T, r, q, sigma) for S in S_grid])
    delta_grid = qty * np.array([bs_put_delta(S, K, T, r, q, sigma) for S in S_grid])

gamma_grid = qty * np.array([bs_gamma(S, K, T, r, q, sigma) for S in S_grid])
vega_grid = qty * np.array([bs_vega(S, K, T, r, q, sigma) for S in S_grid])

if opt_type == "Call":
    theta_grid = qty * np.array([bs_call_theta(S, K, T, r, q, sigma) for S in S_grid]) / 365
else:
    theta_grid = qty * np.array([bs_put_theta(S, K, T, r, q, sigma) for S in S_grid]) / 365

#
# Current option value and Greeks at S0
#
if opt_type == "Call":
    option_price = bs_call_price(S0, K, T, r, q, sigma)
    current_price = qty * option_price
    current_delta = qty * bs_call_delta(S0, K, T, r, q, sigma)
else:
    option_price = bs_put_price(S0, K, T, r, q, sigma)
    current_price = qty * option_price
    current_delta = qty * bs_put_delta(S0, K, T, r, q, sigma)

current_gamma = qty * bs_gamma(S0, K, T, r, q, sigma)
current_vega = qty * bs_vega(S0, K, T, r, q, sigma)

if opt_type == "Call":
    current_theta = qty * bs_call_theta(S0, K, T, r, q, sigma)/365
else:
    current_theta = qty * bs_put_theta(S0, K, T, r, q, sigma)/365

# Premium paid/received for the position (used for P/L calculations)
premium = float(current_price)

#
# -----------------------------
# Delta hedging (shares)
# -----------------------------
# current_delta already includes qty (net option delta)
hedge_shares = -current_delta if auto_delta_hedge else 0.0
net_shares = shares_owned + hedge_shares
net_delta_including_shares = current_delta + net_shares  # each share has delta ~ 1

pl_payoff_grid = payoff_grid - premium
pl_price_grid = price_grid - premium

# Stock P/L from net shares (measured relative to current spot S0)
stock_pl_grid = net_shares * (S_grid - S0)

# Total P/L including delta hedge shares
pl_payoff_grid_total = pl_payoff_grid + stock_pl_grid
pl_price_grid_total = pl_price_grid + stock_pl_grid

# Intrinsic + time value at S0
if opt_type == "Call":
    intrinsic_per_option = max(S0 - K, 0.0)
else:
    intrinsic_per_option = max(K - S0, 0.0)

intrinsic = qty * intrinsic_per_option

time_value_per_option = option_price - intrinsic_per_option
time_value = current_price - intrinsic

# -----------------------------
# Dashboard-style summary
# -----------------------------

#
# Small CSS tweaks for card-like feel
st.markdown(
    """
    <style>
    .bs-card {
        padding: 1rem 1.2rem;
        border-radius: 0.9rem;
        margin-bottom: 0.8rem;
        transition: transform 120ms cubic-bezier(0.4,0.0,0.2,1), box-shadow 120ms cubic-bezier(0.4,0.0,0.2,1);
    }
    .bs-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(0,0,0,0.12);
    }
    .bs-card-main {
        background: #f0fdf4;
        border: 1px solid rgba(34,197,94,0.25);
        box-shadow: 0 8px 18px rgba(0,0,0,0.08);
    }
    .bs-card-secondary {
        background: #f8fafc;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 6px 14px rgba(0,0,0,0.06);
    }
    .bs-card-title {
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        color: #374151;
        margin-bottom: 0.35rem;
    }
    .bs-card-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #222222;
    }
    .bs-card-greek {
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 12px 26px rgba(0,0,0,0.10);
    }
    .bs-card-greek-delta {
        background: #e6f0ff;
        border-left: 6px solid #3b82f6;
    }
    .bs-card-greek-gamma {
        background: #f1ebff;
        border-left: 6px solid #8b5cf6;
    }
    .bs-card-greek-vega {
        background: #e9f8f0;
        border-left: 6px solid #22c55e;
    }
    .bs-card-greek-theta {
        background: #fff1e6;
        border-left: 6px solid #f97316;
    }
    .greek-note {
        background: #eef4ff;
        border: 1px solid #d6e4ff;
        border-radius: 0.6rem;
        padding: 0.9rem 1.0rem;
        margin: 0.6rem 0 1.0rem 0;
        color: #1f2a44;
        font-size: 1.05rem;
        font-weight: 500;
    }
    /* Monte Carlo stat bubbles */
    .mc-bubble {
        background: #f8fafc;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.10);
    }
    .mc-positive {
        background: rgba(34,197,94,0.16);   /* slightly stronger light green */
        border: 1px solid rgba(34,197,94,0.35);
        border-left: 6px solid #22c55e;
    }
    .mc-negative {
        background: rgba(239,68,68,0.16);   /* slightly stronger light red */
        border: 1px solid rgba(239,68,68,0.35);
        border-left: 6px solid #ef4444;
    }
    .mc-neutral {
        background: linear-gradient(135deg, #1f2937, #111827); /* slate → near-black */
        border: 1px solid rgba(148,163,184,0.25);
        border-left: 6px solid #94a3b8; /* soft slate accent */
    }
    .mc-bubble-title {
        font-size: 1.05rem;
        font-weight: 650;
        color: rgba(255,255,255,0.85);
        margin-bottom: 8px;
    }
    .mc-bubble-value {
        font-size: 3.0rem;
        font-weight: 850;
        color: #ffffff;
        letter-spacing: 0.2px;
    }
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(0,0,0,0.25), transparent);
        margin: 1.6rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hedging summary
h1, h2, h3 = st.columns(3)
with h1:
    st.metric("Shares owned", f"{shares_owned:.4f}")
with h2:
    st.metric("Hedge shares (−Δ)", f"{hedge_shares:.4f}")
with h3:
    st.metric("Net shares", f"{net_shares:.4f}")

st.markdown("### Option snapshot")

col_main, col_side1, col_side2, col_side3 = st.columns([2, 1, 1, 1])

with col_main:
    st.markdown(
        f"""
        <div class="bs-card bs-card-main">
            <div class="bs-card-title">Position market value</div>
            <div class="bs-card-value">${current_price:,.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_side1:
    st.markdown(
        f"""
        <div class="bs-card bs-card-secondary">
            <div class="bs-card-title">Option price</div>
            <div class="bs-card-value">${option_price:,.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_side2:
    st.markdown(
        f"""
        <div class="bs-card bs-card-secondary">
            <div class="bs-card-title">Position intrinsic value</div>
            <div class="bs-card-value">${intrinsic:,.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_side3:
    st.markdown(
        f"""
        <div class="bs-card bs-card-secondary">
            <div class="bs-card-title">Position time value</div>
            <div class="bs-card-value">${time_value:,.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Greeks at current share price")

col_d, col_g, col_v, col_t = st.columns(4)

with col_d:
    st.markdown(
        f"""
        <div class="bs-card bs-card-greek bs-card-greek-delta">
            <div class="bs-card-title">Delta</div>
            <div class="bs-card-value">{current_delta:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Net delta incl. shares: {net_delta_including_shares:.4f}")

with col_g:
    st.markdown(
        f"""
        <div class="bs-card bs-card-greek bs-card-greek-gamma">
            <div class="bs-card-title">Gamma</div>
            <div class="bs-card-value">{current_gamma:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_v:
    st.markdown(
        f"""
        <div class="bs-card bs-card-greek bs-card-greek-vega">
            <div class="bs-card-title">Vega</div>
            <div class="bs-card-value">{current_vega:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_t:
    st.markdown(
        f"""
        <div class="bs-card bs-card-greek bs-card-greek-theta">
            <div class="bs-card-title">Theta (per day)</div>
            <div class="bs-card-value">{current_theta:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# -----------------------------
# Layout with tabs
# -----------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Payoff & Price", "Greeks", "Volatility", "Monte Carlo Lab"
])

# ---- Tab 1: Payoff & Price ----
with tab1:
    st.subheader("Payoff at Expiry vs Price Today")

    if auto_delta_hedge or net_shares != 0:
        pl_expiry = pl_payoff_grid_total
        pl_today = pl_price_grid_total
        pl_label_suffix = " (delta-hedged)"
    else:
        pl_expiry = pl_payoff_grid
        pl_today = pl_price_grid
        pl_label_suffix = ""

    payoff_df = pd.DataFrame({
        "Share price": S_grid,
        "P/L at expiry": pl_expiry,
        "P/L today": pl_today,
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=payoff_df["Share price"],
        y=payoff_df["P/L at expiry"],
        mode="lines",
        name=f"P/L at expiry{pl_label_suffix}",
    ))
    fig.add_trace(go.Scatter(
        x=payoff_df["Share price"],
        y=payoff_df["P/L today"],
        mode="lines",
        name=f"P/L today{pl_label_suffix}",
    ))

    # Zero P/L line and current spot marker
    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.6)
    fig.add_vline(x=S0, line_dash="dash", line_color="grey", opacity=0.6)

    apply_white_plotly_theme(fig, height=420)
    fig.update_layout(title=dict(text="Payoff and P/L", font=dict(color=TEXT_DARK, family=FONT_FAMILY)))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    enforce_black_axes(fig, x_title="Share price S", y_title="P/L ($)")

    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: Greeks ----
with tab2:
    st.subheader("Option Greeks vs Share Price")
    st.caption("Sensitivities of the option value with respect to underlying price, volatility, and time.")

    # --- Delta ---
    if auto_delta_hedge or net_shares != 0:
        st.markdown(
            '<div class="greek-note"><b>Net Delta</b>: Option delta reflecting delta hedging.</div>',
            unsafe_allow_html=True,
        )
        net_delta_grid = delta_grid + net_shares
        delta_df = pd.DataFrame({"Share price": S_grid, "Net Delta": net_delta_grid})
        fig_d = line_chart(delta_df, x="Share price", y="Net Delta", height=320, x_title="Share price", y_title="Net Delta")
        fig_d.add_hline(y=0, line_width=1, line_color="black", opacity=0.4)
        fig_d.add_vline(x=S0, line_dash="dash", line_color="grey", opacity=0.6)
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.markdown(
            '<div class="greek-note"><b>Delta</b>: sensitivity of option value to a $1 change in the share price.</div>',
            unsafe_allow_html=True,
        )
        delta_df = pd.DataFrame({"Share price": S_grid, "Delta": delta_grid})
        fig_d = line_chart(delta_df, x="Share price", y="Delta", height=320, x_title="Share price", y_title="Delta")
        fig_d.add_vline(x=S0, line_dash="dash", line_color="grey", opacity=0.6)
        st.plotly_chart(fig_d, use_container_width=True)

    # --- Gamma ---
    st.markdown(
        '<div class="greek-note"><b>Gamma</b>: sensitivity of Delta to a $1 change in the share price.</div>',
        unsafe_allow_html=True,
    )
    gamma_df = pd.DataFrame({"Share price": S_grid, "Gamma": gamma_grid})
    fig_g = line_chart(gamma_df, x="Share price", y="Gamma", height=320, x_title="Share price", y_title="Gamma (per $)")
    st.plotly_chart(fig_g, use_container_width=True)

    # --- Vega ---
    st.markdown(
        '<div class="greek-note"><b>Vega</b>: sensitivity of option value to a 1% change in implied volatility.</div>',
        unsafe_allow_html=True,
    )
    vega_df = pd.DataFrame({"Share price": S_grid, "Vega": vega_grid})
    fig_v = line_chart(vega_df, x="Share price", y="Vega", height=320, x_title="Share price", y_title="Vega ($ per 1%)")
    st.plotly_chart(fig_v, use_container_width=True)

    # --- Theta ---
    st.markdown(
        '<div class="greek-note"><b>Theta</b>: sensitivity of option value to the passage of one day.</div>',
        unsafe_allow_html=True,
    )
    theta_df = pd.DataFrame({"Share price": S_grid, "Theta": theta_grid})
    fig_t = line_chart(theta_df, x="Share price", y="Theta", height=320, x_title="Share price", y_title="Theta ($ per day)")
    st.plotly_chart(fig_t, use_container_width=True)

with tab3:
    st.subheader("Realised vs Implied Volatility")
    st.caption(
        "Realised volatility is computed from historical share prices. "
        "Implied volatility is the σ used by the option model (or the IV solved from a market option price if inputted)."
    )

    if yf is None:
        st.error("yfinance is not installed in this environment. Install it with: pip install yfinance")
    else:
        c_left, c_right = st.columns([1.0, 1.0])
        with c_left:
            ticker_map = {
                "Commonwealth Bank (CBA)": "CBA.AX",
                "NVIDIA": "NVDA",
                "TESLA": "TSLA",
                "NAB": "NAB.AX",
                "Macquarie Group (MQG)": "MQG.AX",
                "APPLE": "AAPL",
                "CSL": "CSL.AX",
                "Wesfarmers": "WES.AX",
                "Custom ticker…": "CUSTOM",
            }

            choice = st.selectbox("Select stock", list(ticker_map.keys()), index=0)

            if ticker_map[choice] == "CUSTOM":
                ticker = st.text_input("Enter Yahoo ticker", value="CBA.AX")
            else:
                ticker = ticker_map[choice]
        with c_right:
            period = st.selectbox("History period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

        interval = "1d"
        annualisation = st.selectbox("Annualisation (trading days)", [252, 365], index=0)
        w1, w2 = st.columns(2)
        with w1:
            win1 = st.number_input("Rolling window 1 (days)", min_value=5, max_value=252, value=20, step=5)
        with w2:
            win2 = st.number_input("Rolling window 2 (days)", min_value=5, max_value=252, value=60, step=5)

        df_hist = fetch_price_history(ticker.strip(), period=period, interval=interval)
        if df_hist.empty:
            st.warning("No price data returned. Check the ticker symbol (e.g. CBA.AX, BHP.AX, CSL.AX).")
        else:
            close = df_hist["Close"].copy()
            rv1 = realised_volatility_from_close(close, window=int(win1), annualisation=int(annualisation))
            rv2 = realised_volatility_from_close(close, window=int(win2), annualisation=int(annualisation))

            # Implied vol: show both the model sigma and (if available) solved IV
            model_sigma_pct = float(sigma) * 100.0
            solved_iv_pct = float(iv) * 100.0 if (iv is not None) else None

            m1, m2, m3 = st.columns(3)
            m1.metric("Model σ (%)", f"{model_sigma_pct:.2f}")
            if solved_iv_pct is None:
                m2.metric("Solved IV (%)", "—")
            else:
                m2.metric("Solved IV (%)", f"{solved_iv_pct:.2f}")
            m3.metric("Latest close", f"${float(close.iloc[-1]):.2f}")

            # Price chart
            price_df = pd.DataFrame({"Date": close.index, "Close": close.values})
            fig_p = px.line(price_df, x="Date", y="Close", title=f"{ticker.upper()} — Close price")
            apply_white_plotly_theme(fig_p, height=360)
            enforce_black_axes(fig_p, x_title="Date", y_title="Close")
            st.plotly_chart(fig_p, use_container_width=True)

            # Volatility chart
            vol_df = pd.DataFrame({
                "Date": close.index,
                f"Realised vol {int(win1)}d": (rv1 * 100.0).reindex(close.index),
                f"Realised vol {int(win2)}d": (rv2 * 100.0).reindex(close.index),
            }).dropna(subset=[f"Realised vol {int(win1)}d", f"Realised vol {int(win2)}d"], how="all")

            fig_v = go.Figure()
            if f"Realised vol {int(win1)}d" in vol_df.columns:
                fig_v.add_trace(go.Scatter(
                    x=vol_df["Date"],
                    y=vol_df[f"Realised vol {int(win1)}d"],
                    mode="lines",
                    name=f"Realised {int(win1)}d (%)",
                ))
            if f"Realised vol {int(win2)}d" in vol_df.columns:
                fig_v.add_trace(go.Scatter(
                    x=vol_df["Date"],
                    y=vol_df[f"Realised vol {int(win2)}d"],
                    mode="lines",
                    name=f"Realised {int(win2)}d (%)",
                ))

            # Add implied vol reference lines
            fig_v.add_hline(y=model_sigma_pct, line_width=1, line_dash="dash", line_color="black")
            if solved_iv_pct is not None:
                fig_v.add_hline(y=solved_iv_pct, line_width=1, line_dash="dot", line_color="black")

            apply_white_plotly_theme(fig_v, height=420)
            fig_v.update_layout(title=dict(text="Realised volatility vs implied/model volatility", font=dict(color=TEXT_DARK, family=FONT_FAMILY)))
            fig_v.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            enforce_black_axes(fig_v, x_title="Date", y_title="Volatility (%)")
            st.plotly_chart(fig_v, use_container_width=True)

            st.caption(
                "Dashed line = model σ. Dotted line = solved IV."
            )

    st.markdown("---")
    with st.expander("Implied Volatility Surface", expanded=False):
        st.markdown(
            '''
            <div class="greek-note">
                <b>Implied volatility surface</b>: Builds an implied volatility surface from option chains of the selected ticker.
                Some tickers may have limited or no option data, so use a well-covered US ticker
                (e.g. AAPL, NVDA, TSLA) for best results.
            </div>
            ''',
            unsafe_allow_html=True,
        )

        # Simple guardrail: Yahoo often has thin/no chains for many .AX tickers
        if ticker.strip().upper().endswith(".AX"):
            st.warning(
                "This ticker ends with .AX. Yahoo Finance option-chain coverage can be missing for many ASX names. "
                "Try a US ticker (e.g. NVDA, AAPL, SPY) for the surface demo, or enter a custom US ticker."
            )

        expiries = fetch_option_expiries(ticker.strip())
        if not expiries:
            st.info("No option expiries found for this ticker from Yahoo Finance.")
        else:
            s_left, s_right = st.columns([1.2, 1.0])
            with s_left:
                surface_opt_type = st.selectbox("Surface type", ["Call", "Put"], index=0)
            with s_right:
                surface_source = st.selectbox(
                    "IV source",
                    ["Use YahooFinance IV", "Solve IV via Black–Scholes"],
                    index=0,
                )

            # Controls
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                max_expiries = st.number_input("Number of expiries", min_value=1, max_value=min(20, len(expiries)), value=min(8, len(expiries)), step=1)
            with c2:
                strike_band_pct = st.number_input("Strike band around spot (%)", min_value=5, max_value=200, value=50, step=5)
            with c3:
                min_oi = st.number_input("Min open interest", min_value=0, value=0, step=10)
            with c4:
                drop_zeros = st.checkbox("Drop zero bid/ask", value=True)

            # Spot price for IV solving / strike filtering
            spot = fetch_spot_price_yahoo(ticker.strip())
            if spot is None or spot <= 0:
                st.info("Could not fetch a spot price for this ticker. Surface needs a valid spot price.")
            else:
                # Limit expiries shown (earliest N)
                expiries_sel = expiries[: int(max_expiries)]

                # Build surface rows
                rows = []
                now = datetime.now(timezone.utc)

                # Strike filter band
                band = float(strike_band_pct) / 100.0
                k_min = spot * (1.0 - band)
                k_max = spot * (1.0 + band)

                for ex in expiries_sel:
                    # Parse expiry date to time-to-expiry in years
                    try:
                        ex_dt = datetime.strptime(ex, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        T_ex = (ex_dt - now).total_seconds() / (365.0 * 24 * 3600)
                        if T_ex <= 0:
                            continue
                    except Exception:
                        continue

                    calls_df, puts_df = fetch_option_chain(ticker.strip(), ex)
                    chain_df = calls_df if surface_opt_type == "Call" else puts_df
                    if chain_df is None or len(chain_df) == 0:
                        continue

                    # Basic sanitise
                    for col in ["strike", "bid", "ask", "lastPrice", "openInterest", "impliedVolatility"]:
                        if col not in chain_df.columns:
                            chain_df[col] = np.nan

                    # Filter strikes around spot
                    chain_df = chain_df[(chain_df["strike"] >= k_min) & (chain_df["strike"] <= k_max)].copy()
                    if len(chain_df) == 0:
                        continue

                    # Filter open interest
                    if min_oi > 0 and "openInterest" in chain_df.columns:
                        chain_df = chain_df[chain_df["openInterest"].fillna(0) >= float(min_oi)].copy()

                    if drop_zeros:
                        b = chain_df["bid"].fillna(0)
                        a = chain_df["ask"].fillna(0)
                        chain_df = chain_df[~((b <= 0) & (a <= 0))].copy()

                    if len(chain_df) == 0:
                        continue

                    # Compute IV
                    use_yahoo_iv = surface_source.startswith("Use Yahoo")
                    for _, row in chain_df.iterrows():
                        K_row = float(row.get("strike", np.nan))
                        if not np.isfinite(K_row) or K_row <= 0:
                            continue

                        if use_yahoo_iv:
                            iv_row = row.get("impliedVolatility", np.nan)
                            if iv_row is None or (not np.isfinite(iv_row)):
                                continue
                            iv_row = float(iv_row)
                            # Yahoo usually returns decimal vols (e.g., 0.25). Keep sane bounds.
                            if iv_row <= 0 or iv_row > 5:
                                continue
                        else:
                            bid = float(row.get("bid", np.nan)) if np.isfinite(row.get("bid", np.nan)) else np.nan
                            ask = float(row.get("ask", np.nan)) if np.isfinite(row.get("ask", np.nan)) else np.nan
                            lastp = float(row.get("lastPrice", np.nan)) if np.isfinite(row.get("lastPrice", np.nan)) else np.nan

                            mid = np.nan
                            if np.isfinite(bid) and np.isfinite(ask) and (bid > 0 or ask > 0):
                                if bid > 0 and ask > 0:
                                    mid = 0.5 * (bid + ask)
                                else:
                                    mid = max(bid, ask)
                            elif np.isfinite(lastp) and lastp > 0:
                                mid = lastp

                            if (not np.isfinite(mid)) or mid <= 0:
                                continue

                            iv_solved = implied_volatility(
                                market_price=float(mid),
                                S=float(spot),
                                K=float(K_row),
                                T=float(T_ex),
                                r=float(r),
                                q=float(q),
                                opt_type=surface_opt_type,
                            )
                            if iv_solved is None:
                                continue
                            iv_row = float(iv_solved)
                            if iv_row <= 0 or iv_row > 5:
                                continue

                        rows.append(
                            {
                                "days_to_expiry": float(T_ex) * 365.0,
                                "moneyness": float(spot) / float(K_row),
                                "iv": float(iv_row),
                            }
                        )

                if not rows:
                    st.info("No usable option rows found after filtering. Try widening strike band, lowering OI filter, or switching ticker.")
                else:
                    surf = pd.DataFrame(rows)

                    # Pivot into grid (days_to_expiry x moneyness)
                    piv = (
                        surf.pivot_table(index="days_to_expiry", columns="moneyness", values="iv", aggfunc="mean")
                        .sort_index()
                    )
                    piv = piv.sort_index(axis=1)

                    # Axes
                    y_days = piv.index.values.astype(float)              # days to expiry
                    x_m = piv.columns.values.astype(float)               # S/K
                    z = piv.values * 100.0                               # IV in %

                    # Hover text grid
                    hover_text = [
                        [
                            f"Days={y_days[i]:.0f}<br>S/K={x_m[j]:.3f}<br>IV={z[i, j]:.2f}%"
                            for j in range(len(x_m))
                        ]
                        for i in range(len(y_days))
                    ]

                    fig_surf = go.Figure(
                        data=[
                            go.Surface(
                                x=x_m,
                                y=y_days,
                                z=z,
                                text=hover_text,
                                hovertemplate="%{text}<extra></extra>",
                                colorbar=dict(title="IV (%)"),
                                connectgaps=True,
                            )
                        ]
                    )

                    fig_surf.update_layout(
                        title=dict(text=f"Implied Volatility Surface ({surface_opt_type}) — {ticker.strip().upper()}", font=dict(color="#111827")),
                        height=650,
                        template="plotly_white",
                        paper_bgcolor="white",
                        font=dict(color="#111827"),
                        margin=dict(l=40, r=20, t=50, b=30),
                        scene=dict(
                            xaxis_title="Moneyness (S/K)",
                            yaxis_title="Days to expiry",
                            zaxis_title="IV (%)",
                            camera=dict(
                                eye=dict(x=-1.6, y=1.8, z=0.9)
                            ),
                        ),
                    )

                    st.plotly_chart(fig_surf, use_container_width=True)

                  


with tab4:
    st.markdown("## Monte Carlo Volatility & Hedging Lab")
    st.markdown(
        '''
        <div class="greek-note">
            <b>This section simulates realised price paths and applies dynamic delta hedging.</b><br><br>
            Stock price paths are simulated under GBM with constant volatility, dividend yield,
            and risk-free rate. Only uncertainty comes from the stock price path and time evolution.
        </div>
        ''',
        unsafe_allow_html=True,
    )

    with st.expander("Simulation options", expanded=False):
        st.markdown(
            '''
            <div class="greek-note" style="background:#e6f0ff;border:1px solid #c7dbff;">
                <b>Note:</b> Increasing the number of simulated paths improves statistical accuracy
                but will significantly slow down the simulation and rendering time.
            </div>
            ''',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            n_paths = st.number_input("Number of simulated paths", min_value=100, max_value=20000, value=500, step=500)
        with c2:
            n_steps = st.number_input("Time steps per path", min_value=10, max_value=500, value=100, step=10)
        with c3:
            seed = st.number_input("Random seed", min_value=0, value=1, step=1)

    st.markdown("---")
    st.subheader("Volatility setup")

    # Single toggle: use observed market premium AND its implied volatility (IV) for pricing/hedging
    use_market_iv_and_premium = st.checkbox(
        "Use observed market premium & implied volatility (IV)",
        value=True,
        help=(
            "Uses the observed option price as the premium paid and uses the implied volatility solved "
        ),
    )

    # Use market option price to compute IV (already available in sidebar as market_price)
    if use_market_iv_and_premium and (market_price is None or market_price <= 0):
        st.info(
            "Enter an observed option price in the sidebar so the app can solve IV and use the market premium in this Monte Carlo lab."
        )

    # Realised volatility used to SIMULATE stock paths (the 'truth' in the MC world)
    sigma_real_pct = st.number_input(
        "Realised volatility used for simulation (%)",
        min_value=0.01,
        max_value=500.0,
        value=float(sigma_input_pct),
        step=0.10,
        format="%.2f",
    )

    sigma_real = sigma_real_pct / 100.0

    # sigma_hedge is the volatility you use to compute deltas (your hedge model)
    # premium_paid is what you actually paid for the option (per position)
    if use_market_iv_and_premium:
        if (market_price is not None) and (market_price > 0) and (iv is not None):
            sigma_hedge = float(iv)
            sigma_hedge_label = f"IV = {sigma_hedge*100:.2f}%"

            premium_paid = qty * float(market_price)
            premium_label = f"${premium_paid:,.4f}"
        else:
            # Fall back gracefully if user hasn't provided a usable market price / IV yet
            sigma_hedge = float(sigma)
            sigma_hedge_label = f"{sigma_hedge*100:.2f}%"

            premium_paid = float(premium)
            premium_label = f"${premium_paid:,.4f}"

            st.info(
                "Enter a valid observed option price in the sidebar so IV can be solved. "
                "Until then, the Monte Carlo lab uses the model σ and model premium."
            )
    else:
        sigma_hedge = float(sigma)
        sigma_hedge_label = f"{sigma_hedge*100:.2f}%"

        premium_paid = float(premium)
        premium_label = f"${premium_paid:,.4f}"

    v1, v2, v3 = st.columns(3)
    v1.metric("Realised σ", f"{sigma_real*100:.2f}%")
    v2.metric("Implied σ", sigma_hedge_label)
    v3.metric("Premium paid", premium_label)

    # Dynamic hedging toggle (now default True and new label)
    hedge_dyn = st.checkbox("Compute delta-hedged P/L (dynamic re-hedging)", value=True)

    if "mc_results" not in st.session_state:
        st.session_state.mc_results = None
        st.session_state.mc_params = None

    mc_params = (
        float(S0), float(K), float(T), float(r), float(q),
        float(sigma_real), float(sigma_hedge),
        int(n_steps), int(n_paths), int(seed),
        opt_type, int(qty), bool(hedge_dyn), float(premium_paid),
    )

    if st.session_state.mc_params != mc_params:
        st.session_state.mc_results = None
        st.session_state.mc_params = mc_params

    run_mc = st.button("Run simulation", type="primary")

    if run_mc:
        # Run simulation with sigma_real
        S_paths = simulate_gbm_paths(S0, T, r, q, sigma_real, int(n_steps), int(n_paths), int(seed))
        S_T = S_paths[:, -1]

        # Unhedged P/L at expiry
        if opt_type == "Call":
            payoff_T = qty * np.maximum(S_T - K, 0.0)
        else:
            payoff_T = qty * np.maximum(K - S_T, 0.0)

        pl_unhedged = payoff_T - premium_paid

        # Delta-hedged P/L
        if hedge_dyn:
            dt = T / int(n_steps)
            pl_hedged = np.zeros(len(S_T))

            for i in range(len(S_T)):
                cash = -premium_paid
                shares = 0.0
                tau = T
                S_path = S_paths[i]

                for t in range(len(S_path) - 1):
                    delta = qty * option_delta_at(S_path[t], K, tau, r, q, sigma_hedge, opt_type)
                    target_shares = -delta

                    d_shares = target_shares - shares
                    cash -= d_shares * S_path[t]
                    shares = target_shares

                    cash += shares * S_path[t] * q * dt
                    cash *= np.exp(r * dt)
                    tau = max(0.0, tau - dt)

                cash += shares * S_path[-1]
                cash += payoff_T[i]
                pl_hedged[i] = cash
        else:
            pl_hedged = np.full(len(S_T), np.nan)

        st.session_state.mc_results = {
            "S_paths": S_paths,
            "pl_unhedged": pl_unhedged,
            "pl_hedged": pl_hedged,
        }

    if st.session_state.mc_results is None:
        st.info("Click Run simulation to generate results.")
        st.stop()

    S_paths = st.session_state.mc_results["S_paths"]
    pl_unhedged = st.session_state.mc_results["pl_unhedged"]
    pl_hedged = st.session_state.mc_results["pl_hedged"]
    S_T = S_paths[:, -1]

    # Summary stats (include tail percentiles)
    mean_pl_unhedged = float(np.mean(pl_unhedged))
    p5_unhedged = float(np.percentile(pl_unhedged, 5))
    p95_unhedged = float(np.percentile(pl_unhedged, 95))

    if hedge_dyn:
        mean_pl_hedged = float(np.mean(pl_hedged))
        p5_hedged = float(np.percentile(pl_hedged, 5))
        p95_hedged = float(np.percentile(pl_hedged, 95))
    else:
        mean_pl_hedged = np.nan
        p5_hedged = np.nan
        p95_hedged = np.nan

    # Sharpe ratio (annualised) based on returns normalised by premium paid
    def sharpe_annualised_from_pl(pl: np.ndarray, premium: float, horizon_years: float) -> float:
        if pl is None or len(pl) == 0 or premium == 0 or horizon_years <= 0:
            return np.nan
        rets = pl / abs(float(premium))
        mu = float(np.mean(rets))
        sd = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        if sd <= 0:
            return np.nan
        return mu / sd * float(np.sqrt(1.0 / horizon_years))

    # Helper: return P/L bubble class for styling (green for profit, red for loss)
    def pl_bubble_class(pl_value: float) -> str:
        """Return the CSS class for a P/L bubble (green for >=0, red for <0)."""
        if pl_value is None or (isinstance(pl_value, float) and np.isnan(pl_value)):
            return ""
        return "mc-positive" if pl_value >= 0 else "mc-negative"

    # Compute Sharpe ratios for both unhedged and hedged
    sharpe_unhedged = sharpe_annualised_from_pl(pl_unhedged, premium_paid, T)
    sharpe_hedged = sharpe_annualised_from_pl(pl_hedged, premium_paid, T) if hedge_dyn else np.nan

    st.markdown("### Non delta-hedged stats")
    u1, u2, u3, u4 = st.columns(4)

    with u1:
        st.markdown(
            f"""
            <div class="mc-bubble {pl_bubble_class(mean_pl_unhedged)}">
                <div class="mc-bubble-title">Expected P/L</div>
                <div class="mc-bubble-value">${mean_pl_unhedged:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with u2:
        st.markdown(
            f"""
            <div class="mc-bubble {pl_bubble_class(p5_unhedged)}">
                <div class="mc-bubble-title">5th percentile</div>
                <div class="mc-bubble-value">${p5_unhedged:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with u3:
        st.markdown(
            f"""
            <div class="mc-bubble {pl_bubble_class(p95_unhedged)}">
                <div class="mc-bubble-title">95th percentile</div>
                <div class="mc-bubble-value">${p95_unhedged:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with u4:
        sharpe_unh_disp = "—" if np.isnan(sharpe_unhedged) else f"{sharpe_unhedged:.2f}"
        st.markdown(
            f"""
            <div class="mc-bubble mc-neutral">
                <div class="mc-bubble-title">Sharpe ratio</div>
                <div class="mc-bubble-value">{sharpe_unh_disp}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Delta-hedged stats")
    h1, h2, h3, h4 = st.columns(4)

    if np.isnan(mean_pl_hedged):
        with h1:
            st.markdown(
                """
                <div class="mc-bubble">
                    <div class="mc-bubble-title">Expected P/L</div>
                    <div class="mc-bubble-value">—</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                """
                <div class="mc-bubble">
                    <div class="mc-bubble-title">5th percentile</div>
                    <div class="mc-bubble-value">—</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h3:
            st.markdown(
                """
                <div class="mc-bubble">
                    <div class="mc-bubble-title">95th percentile</div>
                    <div class="mc-bubble-value">—</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h4:
            st.markdown(
                """
                <div class="mc-bubble mc-neutral">
                    <div class="mc-bubble-title">Sharpe ratio</div>
                    <div class="mc-bubble-value">—</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        with h1:
            st.markdown(
                f"""
                <div class="mc-bubble {pl_bubble_class(mean_pl_hedged)}">
                    <div class="mc-bubble-title">Expected P/L</div>
                    <div class="mc-bubble-value">${mean_pl_hedged:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                f"""
                <div class="mc-bubble {pl_bubble_class(p5_hedged)}">
                    <div class="mc-bubble-title">5th percentile</div>
                    <div class="mc-bubble-value">${p5_hedged:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h3:
            st.markdown(
                f"""
                <div class="mc-bubble {pl_bubble_class(p95_hedged)}">
                    <div class="mc-bubble-title">95th percentile</div>
                    <div class="mc-bubble-value">${p95_hedged:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with h4:
            sharpe_h_disp = "—" if np.isnan(sharpe_hedged) else f"{sharpe_hedged:.2f}"
            st.markdown(
                f"""
                <div class="mc-bubble mc-neutral">
                    <div class="mc-bubble-title">Sharpe ratio</div>
                    <div class="mc-bubble-value">{sharpe_h_disp}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Histogram selector for both distributions
    which_hist = st.selectbox("Histogram", ["Delta-hedged P/L", "Unhedged P/L"], index=0)
    hist_data = pl_hedged if which_hist == "Delta-hedged P/L" else pl_unhedged

    fig_hist = px.histogram(
        hist_data,
        nbins=50,
        histnorm="percent",  # show % probability per bin instead of raw counts
        title="Distribution of P/L at expiry",
        labels={"value": "P/L ($)"},
    )
    fig_hist.update_layout(showlegend=False)
    apply_white_plotly_theme(fig_hist, height=420)
    fig_hist.update_layout(
        yaxis_title="Probability (%)",
        xaxis_title="P/L ($)",
    )
    enforce_black_axes(fig_hist, x_title="P/L ($)", y_title="Probability (%)")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Show a few sample paths
    sample_n = min(50, S_paths.shape[0])
    fig_paths = go.Figure()
    for i in range(sample_n):
        fig_paths.add_trace(go.Scatter(
            x=np.linspace(0, T, S_paths.shape[1]),
            y=S_paths[i],
            mode="lines",
            line=dict(width=1),
            showlegend=False,
        ))

    apply_white_plotly_theme(fig_paths, height=420)
    fig_paths.update_layout(
        title=dict(text="Sample simulated stock price paths", font=dict(color="#111827")),
        xaxis_title="Time (years)",
        yaxis_title="Share price",
    )
    enforce_black_axes(fig_paths, x_title="Time (years)", y_title="Share price")
    st.plotly_chart(fig_paths, use_container_width=True)

    st.caption(
        "If realised volatility exceeds implied volatility, an option that is dynamically delta-hedged can be expected to return a positive P/L as volatility has been underpriced in the option. "
        "If realised volatility is lower than implied volatility, the delta-hedged P/L can be negative as volatility has been overpriced."
    )