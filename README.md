# OptiPrice India

**Multi-Model Options Pricing Dashboard for the Indian Equity Derivatives Market**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey)

---

## Overview

**OptiPrice India** is a professional-grade options pricing dashboard built with Streamlit, purpose-built for the Indian equity derivatives market. It enables traders, quantitative analysts, and finance students to price European-style options on Nifty 50 stocks using multiple pricing models side-by-side.

**Problem it solves:** Indian retail and institutional traders lack a unified, free, open-source tool for multi-model options pricing. Existing tools are either paid (Bloomberg, Refinitiv), model-specific, or US-market-focused.

**What it offers:**
- Live market data from Yahoo Finance
- Historical volatility computation with configurable lookback windows
- Three pricing models with full Greeks: Black-Scholes (analytical), Bayesian Binomial Tree (probabilistic), and Neural Network Ensemble (deep learning)
- Interactive comparison charts and posterior distributions
- Dark-mode professional UI

---

## Key Features

- **Stock Search** — Search across all Nifty 50 constituents by symbol or company name with ranked fuzzy matching
- **Live Market Data** — Real-time stock prices, 52-week range, market cap, dividend yield, day change via Yahoo Finance
- **Historical Volatility** — Compute annualized volatility from log-returns with 30/60/90/252-day lookback options
- **Options Configuration** — Call/Put toggle, strike price, expiry date picker, risk-free rate slider, volatility override, dividend yield
- **Black-Scholes Pricing** — Analytical prices with all five Greeks (Delta, Gamma, Theta, Vega, Rho), edge case handling, and implied volatility solver
- **Bayesian Binomial Tree** — Cox-Ross-Rubinstein tree with LogNormal prior on volatility, posterior samples, and 95% credible intervals
- **Model Comparison** — Side-by-side price cards, bar charts, Greeks radar chart, posterior histograms, convergence plots
- **Dark Mode UI** — Custom-styled dashboard with JetBrains Mono/IBM Plex Sans fonts and model color-coding

---

## Pricing Models

### 1. Black-Scholes (Analytical)

Classic closed-form solution for European options:

```
Call: C = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
Put:  P = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)

where:
  d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
  d2 = d1 - sigma * sqrt(T)
```

**Greeks computed analytically:**
- **Delta** — Rate of change of option price w.r.t. underlying
- **Gamma** — Rate of change of delta w.r.t. underlying
- **Theta** — Time decay (per-day)
- **Vega** — Sensitivity to 1% volatility change
- **Rho** — Sensitivity to 1% interest rate change

**Implied Volatility:** Newton-Raphson solver with bisection fallback.

---

### 2. Bayesian Binomial Tree (Cox-Ross-Rubinstein + Bayesian Uncertainty)

A probabilistic extension of the classic CRR binomial tree:

**Tree parameters:**
```
dt = T / steps
u  = exp(sigma * sqrt(dt))
d  = 1/u
p  = (exp((r-q)*dt) - d) / (u - d)
```

**Bayesian layer:**
- Prior on volatility: LogNormal distribution fitted to historical volatility estimates across multiple lookback windows
- 1000 draws from prior → 1000 CRR tree evaluations
- Posterior: mean price with 95% credible interval (2.5th–97.5th percentiles)
- Greeks via finite differences on the tree

---

### 3. Heston Stochastic Volatility

Monte Carlo simulation with stochastic variance:

**SDEs (Euler-Maruyama discretization):**
```
dS = (r - q) * S * dt + sqrt(v) * S * dW1
dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW2
```

where `dW2 = rho * dW1 + sqrt(1 - rho^2) * dZ` (correlated Brownian motions via Cholesky).

**Features:**
- Full truncation scheme (variance floored at zero)
- Calibration via L-BFGS-B optimization
- Synthetic training data generation for neural networks

---

### 4. Neural Network Ensemble (Uncertainty-Aware Deep Hedging)

Inspired by Poddar (2026) — ensemble of LSTM networks with confidence-weighted blending:

**Architecture (per network):**
```
LSTM(input=7, hidden=64, layers=2)
  → Linear(64, 32) → ReLU → Linear(32, 1)
```

**Input features (7):**
- Moneyness (S/K)
- Time to expiry (T)
- Historical volatility
- Risk-free rate (r)
- Dividend yield (q)
- RSI (14-day)
- Volatility ratio (5/20-day)

**Ensemble:** 5 independent networks with different random seeds.

**Uncertainty-aware blending:**
```
sigma_ensemble = std(predictions across 5 networks)
w = sigmoid(-alpha * sigma_ensemble + beta)
blended_price = w * price_NN + (1 - w) * price_BS
```

High disagreement → more weight on Black-Scholes (safety fallback).

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Frontend | Streamlit 1.38+ (custom CSS) |
| Data | yfinance, pandas, numpy |
| Numerics | scipy, numpy |
| Deep Learning | PyTorch 2.2+ |
| Visualization | Plotly |
| Testing | pytest |
| Linting | ruff, black |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/Flameingmoy/Options_Pricing.git
cd Options_Pricing

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Acceleration (Optional)

For CUDA-enabled PyTorch:

```bash
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

### Running the App

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Project Structure

```
Options_Pricing/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Core dependencies
├── requirements-gpu.txt        # GPU-enabled PyTorch
├── PRD.md                      # Product requirements document
├── claude.md                   # AI development guidelines
├── conftest.py                 # Pytest configuration
│
├── data/
│   ├── nifty50.csv             # Nifty 50 constituent stocks
│   └── models/                 # Saved neural network weights
│
├── src/
│   ├── models/
│   │   ├── __init__.py         # PricingResult, GreeksResult dataclasses
│   │   ├── black_scholes.py    # BS pricing + Greeks + IV solver
│   │   ├── bayesian_tree.py    # CRR tree with Bayesian volatility
│   │   ├── heston.py           # Stochastic volatility MC
│   │   └── neural_ensemble.py  # LSTM ensemble pricing
│   │
│   ├── data/
│   │   ├── yahoo_fetcher.py    # Yahoo Finance data gateway
│   │   ├── nifty500.py         # Stock list + fuzzy search
│   │   └── option_chain.py     # Option chain formatting
│   │
│   ├── ui/
│   │   ├── sidebar.py          # Sidebar orchestration
│   │   ├── charts.py           # Plotly chart builders
│   │   ├── components.py       # Reusable UI components
│   │   ├── search.py           # Typeahead stock search
│   │   └── styles.py           # Custom CSS injection
│   │
│   └── utils/
│       ├── config.py           # Centralized constants
│       ├── math_utils.py       # Safe math wrappers + RSI
│       └── cache.py            # Model weight caching
│
├── tests/
│   ├── test_black_scholes.py   # BS unit tests
│   ├── test_bayesian_tree.py   # Bayesian tree tests
│   └── test_neural_ensemble.py # Neural ensemble tests
│
├── notebooks/                  # Jupyter notebooks (placeholder)
└── .streamlit/
    └── config.toml             # Streamlit theme config
```

---

## Testing

Run the test suite:

```bash
pytest tests/ -v --tb=short
```

**Coverage:**
- `test_black_scholes.py` — Hull textbook validation, put-call parity, edge cases, Greeks sanity, IV round-trip
- `test_bayesian_tree.py` — CRR convergence, posterior distribution, credible intervals, Greeks
- `test_neural_ensemble.py` — LSTM forward pass, ensemble diversity, save/load, blending behavior

---

## Configuration

### Streamlit Theme (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#448aff"
backgroundColor = "#0a0a0f"
secondaryBackgroundColor = "#12121a"
textColor = "#e8e8f0"
font = "monospace"
```

### Key Constants (`src/utils/config.py`)

| Category | Default |
|----------|---------|
| Risk-free rate | 7% (Indian 10Y G-Sec) |
| Volatility lookback | 252 days |
| Cache TTL | 5 minutes |
| Bayesian samples | 1000 posterior draws |
| Tree steps | 100 (max 500) |
| NN ensemble size | 5 networks |
| NN hidden dim | 64 |
| NN training epochs | 50 |

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Complete | Project scaffolding, dependencies |
| 1 | ✅ Complete | Data layer (Yahoo Finance, volatility) |
| 2 | ✅ Complete | Black-Scholes model + Greeks |
| 3 | ✅ Complete | Bayesian Binomial Tree |
| 4 | ✅ Complete | Heston + Neural Ensemble (fully integrated) |
| 5 | ✅ Complete | Price vs. Strike curves, sensitivity tornado, ensemble disagreement |
| 6 | ✅ Complete | Advanced visualizations, market context, polish & testing |

**All phases complete! Application is production-ready.**

---

## References

- **Poddar, Manan (2026).** "Uncertainty-Aware Deep Hedging." London School of Economics. [arXiv:2603.10137v1](https://arxiv.org/abs/2603.10137v1)
- **Hull, John.** *Options, Futures, and Other Derivatives.* 11th ed. Pearson, 2022.

---

## Author

**Chinmay**

- GitHub: [@Flameingmoy](https://github.com/Flameingmoy)
- Repository: [Options_Pricing](https://github.com/Flameingmoy/Options_Pricing)

---

## License

All Rights Reserved. No license is currently granted for use, modification, or distribution of this software.

---

*Built with Streamlit • Powered by Yahoo Finance • Designed for the Indian derivatives market*
