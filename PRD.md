# Product Requirements Document: OptiPrice India
## Neural-Augmented Options Pricing Engine for Indian Equities

**Version:** 1.0  
**Date:** March 28, 2026  
**Author:** Engineering  
**Status:** Draft  

---

## 1. Executive Summary

OptiPrice India is a Streamlit-based options pricing dashboard purpose-built for the Indian equity derivatives market. It enables traders, quants, and finance students to price European-style options on any stock within the Nifty 500 universe using three distinct pricing models — Black-Scholes, Bayesian Binomial Trees, and a Neural Network ensemble inspired by Poddar (2026) — then visually compare their outputs side-by-side. The application sources live market data via Yahoo Finance and renders a dark-mode-only, professional-grade dashboard with rich interactive visualizations.

---

## 2. Problem Statement

Indian retail and institutional traders lack a unified, free, open-source tool that lets them:

- Price options across multiple models in one interface
- Compare classical vs. ML-based pricing approaches visually
- Access real-time underlying data for any Nifty 500 constituent
- Understand model uncertainty and confidence in pricing outputs

Existing tools are either paid (Bloomberg, Refinitiv), model-specific, or built for US markets only. OptiPrice India bridges this gap with a focused, beautiful, Indian-market-first experience.

---

## 3. Target Users

| Persona | Description | Primary Need |
|---|---|---|
| **Retail Trader** | Active F&O participant on NSE | Quick, reliable option prices with visual Greeks |
| **Quant Researcher** | Builds/validates pricing models | Side-by-side model comparison, parameter sensitivity |
| **Finance Student** | Learning derivatives pricing | Educational tool to see how models differ |
| **Risk Analyst** | Manages portfolio Greeks | Accurate Greeks computation across models |

---

## 4. Core Features

### 4.1 Stock Selection & Data Ingestion

- **Dynamic Search Bar**: Typeahead search across all Nifty 500 constituents by company name or NSE symbol (e.g., "RELIANCE", "Reliance Industries", "TCS")
- **Data Source**: Yahoo Finance API via `yfinance` Python library
  - Ticker mapping: NSE symbols appended with `.NS` suffix (e.g., `RELIANCE.NS`)
  - Fetch: current price, historical OHLCV (1Y default), dividend yield, and available option chain data
- **Stock Info Card**: On selection, display current price, day change (%), 52-week range, market cap, sector, and a sparkline of recent price action
- **Nifty 500 Master List**: Maintained as a static CSV/JSON bundled with the app, with periodic refresh capability

### 4.2 Options Configuration Panel

- **Option Type**: Call / Put toggle
- **Strike Price (K)**: Manual input or slider, with smart defaults around ATM
- **Time to Expiry (T)**: Date picker for expiry date, auto-calculates T in years; also support manual entry in days
- **Risk-Free Rate (r)**: Default to current Indian 10Y G-Sec yield (fetched or user-overridable), displayed in %
- **Volatility (σ)**: 
  - Auto-computed from historical data (configurable lookback: 30/60/90/252 days)
  - Manual override option
  - Display both historical vol and implied vol (if option chain available)
- **Dividend Yield (q)**: Auto-fetched from Yahoo Finance, user-overridable

### 4.3 Pricing Models

#### 4.3.1 Black-Scholes Model

The analytical closed-form solution for European options.

**Implementation:**
- Standard Black-Scholes formula for calls and puts
- Full Greeks computation: Delta (Δ), Gamma (Γ), Theta (Θ), Vega (ν), Rho (ρ)
- Handles continuous dividend yield adjustment
- Used as the baseline/benchmark model

**Outputs:**
- Option price
- All five Greeks
- Implied volatility solver (Newton-Raphson) when market price is available

#### 4.3.2 Bayesian Binomial Tree Model

A probabilistic extension of the Cox-Ross-Rubinstein (CRR) binomial tree that quantifies parameter uncertainty.

**Implementation:**
- CRR binomial tree with configurable steps (default: 100, max: 500)
- Bayesian layer: place priors on volatility (σ) and drift (μ) parameters
  - Prior on σ: LogNormal distribution fitted to historical vol estimates across multiple lookback windows
  - Prior on μ: Normal distribution centered on historical mean return
- Monte Carlo sampling from posterior (via MCMC or grid approximation) to generate a distribution of option prices
- Report: mean price, median price, 95% credible interval, and full posterior density

**Outputs:**
- Point estimate (posterior mean)
- Credible interval (2.5th – 97.5th percentile)
- Posterior distribution histogram
- Convergence diagnostics (tree steps vs. price stability)
- Greeks via finite differences on the tree

#### 4.3.3 Neural Network Ensemble (Uncertainty-Aware Deep Hedging)

Inspired by Poddar (2026) — "Uncertainty-Aware Deep Hedging" (arXiv:2603.10137v1).

**Architecture:**
- Deep ensemble of 5 independent LSTM networks
- Each network trained on simulated price paths under Heston stochastic volatility
- Input features per timestep: current stock price (S), moneyness (S/K), time to expiry (τ), historical volatility (σ_hist), RSI, and normalized volume
- Output: predicted option price (or hedge ratio, repurposed for pricing)
- Ensemble disagreement (standard deviation across 5 predictions) serves as a per-prediction confidence/uncertainty measure

**Training Pipeline:**
- Data generation: simulate paths using Heston model with parameters calibrated to the selected stock's historical behavior
- Loss function: MSE on option price prediction (or CVaR-optimised P&L for hedging variant)
- Training: each of the 5 networks initialized independently, trained on the same data but with different random seeds
- Pre-trained weights bundled for common Nifty 50 stocks; on-the-fly training for others (with progress bar)

**Uncertainty-Aware Blending:**
- When ensemble disagreement is high, blend prediction toward Black-Scholes (safety fallback)
- CVaR-optimised blending weights as described in Poddar (2026)
- Display: confidence meter based on ensemble agreement

**Outputs:**
- Ensemble mean price
- Individual predictions from all 5 networks (displayed as a swarm/strip plot)
- Uncertainty band (±1σ, ±2σ of ensemble)
- Confidence score (derived from ensemble agreement)
- Blended price (NN + BS weighted by confidence)

### 4.4 Model Comparison Dashboard

The central visualization panel that places all three models side-by-side.

**Comparison Visualizations:**

1. **Price Comparison Bar Chart**: Grouped bar chart showing each model's price estimate with error bars (credible intervals for Bayesian, ensemble std for NN)

2. **Greeks Comparison Radar Chart**: Overlay of Greeks from all three models on a single radar/spider chart

3. **Volatility Surface**: 3D surface or heatmap of option prices across strike prices (x) and expiries (y) for a selected model, with toggle to switch models

4. **Price vs. Strike Curve**: Line chart of option price as a function of strike price for all three models, overlaid — with the market price (if available) shown as reference dots

5. **Sensitivity Analysis (Tornado Chart)**: For each model, show how the price changes when each input parameter is perturbed ±10%

6. **Convergence Plot (Bayesian)**: Tree steps vs. option price, showing convergence behavior

7. **Ensemble Disagreement Plot (NN)**: Time-series or bar chart of individual network predictions and their spread

8. **Historical Backtest Chart**: (Optional/Advanced) Compare model-predicted prices against actual historical option prices over the past N trading days

### 4.5 Real-Time Data Panel

- **Live Price Ticker**: Current stock price with auto-refresh (configurable interval: 15s/30s/60s)
- **Intraday Chart**: Candlestick or line chart of today's price action
- **Option Chain Viewer**: Display available NSE option chain (strikes, last price, OI, volume, IV) fetched from Yahoo Finance
- **Market Context**: Nifty 50 index level, India VIX value, sector index performance

---

## 5. UI/UX Specification

### 5.1 Design Philosophy

**Dark mode only.** The interface adopts a premium, terminal-inspired dark aesthetic with high contrast data visualization. Think Bloomberg Terminal meets modern fintech — dense with information but never cluttered.

### 5.2 Color System

| Token | Hex | Usage |
|---|---|---|
| `--bg-primary` | `#0a0a0f` | Main background |
| `--bg-secondary` | `#12121a` | Cards, panels |
| `--bg-elevated` | `#1a1a28` | Hover states, active elements |
| `--border` | `#2a2a3d` | Subtle borders |
| `--text-primary` | `#e8e8f0` | Main text |
| `--text-secondary` | `#8888a0` | Labels, captions |
| `--accent-green` | `#00e676` | Positive values, calls, up moves |
| `--accent-red` | `#ff1744` | Negative values, puts, down moves |
| `--accent-blue` | `#448aff` | Links, interactive elements, BS model |
| `--accent-amber` | `#ffab00` | Warnings, Bayesian model |
| `--accent-purple` | `#b388ff` | Neural network model |

### 5.3 Typography

- **Headers**: JetBrains Mono or IBM Plex Mono (monospaced, technical feel)
- **Body**: IBM Plex Sans or DM Sans
- **Numbers/Data**: Tabular-aligned monospaced figures

### 5.4 Layout Structure

```
┌─────────────────────────────────────────────────────────┐
│  HEADER: Logo + App Name + Live Market Strip            │
├──────────────┬──────────────────────────────────────────┤
│              │                                          │
│  SIDEBAR     │  MAIN CONTENT AREA                      │
│              │                                          │
│  - Search    │  ┌──────────────────────────────────┐   │
│  - Stock     │  │  Stock Info Card + Sparkline     │   │
│    Info      │  └──────────────────────────────────┘   │
│  - Option    │  ┌──────┬──────┬──────┐                 │
│    Config    │  │  BS  │ Bayes│  NN  │  Price Cards    │
│  - Model     │  └──────┴──────┴──────┘                 │
│    Settings  │  ┌──────────────────────────────────┐   │
│  - Advanced  │  │  Comparison Visualizations       │   │
│    Params    │  │  (Tabbed: Prices | Greeks |      │   │
│              │  │   Surface | Sensitivity)          │   │
│              │  └──────────────────────────────────┘   │
│              │  ┌──────────────────────────────────┐   │
│              │  │  Option Chain / Market Context    │   │
│              │  └──────────────────────────────────┘   │
└──────────────┴──────────────────────────────────────────┘
```

### 5.5 Key UI Components

- **Search Bar**: Full-width at top of sidebar with debounced typeahead, showing symbol + company name + sector in dropdown results
- **Price Cards**: Three prominent cards (one per model) showing price, delta, confidence — color-coded by model identity
- **Parameter Sliders**: Custom-styled Streamlit sliders with real-time price update
- **Chart Tabs**: Streamlit tabs for switching between visualization types without page reload
- **Tooltips**: Contextual help on every parameter and metric explaining what it means
- **Loading States**: Skeleton loaders and progress bars for model computation
- **Responsive Sidebar**: Collapsible on smaller screens

### 5.6 Streamlit Theming

Apply via `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#448aff"
backgroundColor = "#0a0a0f"
secondaryBackgroundColor = "#12121a"
textColor = "#e8e8f0"
font = "monospace"
```

Additional custom CSS injected via `st.markdown()` for fine-grained control over card styles, chart containers, search bar styling, and typography.

---

## 6. Technical Architecture

### 6.1 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit 1.38+ with custom CSS/HTML |
| **Data** | yfinance, pandas, numpy |
| **Black-Scholes** | scipy.stats (norm), numpy |
| **Bayesian Trees** | numpy, scipy (for priors/sampling) |
| **Neural Network** | PyTorch (LSTM ensemble) |
| **Visualization** | Plotly (primary), matplotlib (fallback) |
| **Caching** | Streamlit @st.cache_data / @st.cache_resource |
| **Config** | TOML / YAML for defaults |

### 6.2 Project Structure

```
optiprice-india/
├── .streamlit/
│   └── config.toml              # Dark theme configuration
├── app.py                        # Main Streamlit entry point
├── requirements.txt
├── README.md
├── claude.md                     # Development guidelines
├── tasks/
│   ├── todo.md                   # Active task tracking
│   └── lessons.md                # Learnings and corrections
├── data/
│   ├── nifty500.csv              # Nifty 500 constituent list
│   └── models/                   # Pre-trained NN weights
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── yahoo_fetcher.py      # Yahoo Finance data fetching
│   │   ├── nifty500.py           # Nifty 500 stock list + search
│   │   └── option_chain.py       # Option chain data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── black_scholes.py      # BS model + Greeks
│   │   ├── bayesian_tree.py      # Bayesian binomial tree
│   │   ├── neural_ensemble.py    # LSTM ensemble (Poddar-inspired)
│   │   ├── heston.py             # Heston model path simulation
│   │   └── greeks.py             # Unified Greeks interface
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── components.py         # Reusable UI components
│   │   ├── sidebar.py            # Sidebar layout
│   │   ├── charts.py             # Plotly chart builders
│   │   ├── styles.py             # Custom CSS injection
│   │   └── search.py             # Typeahead search component
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # App configuration
│       ├── math_utils.py         # Shared math helpers
│       └── cache.py              # Caching utilities
├── tests/
│   ├── test_black_scholes.py
│   ├── test_bayesian_tree.py
│   ├── test_neural_ensemble.py
│   └── test_data_fetcher.py
└── notebooks/
    ├── model_validation.ipynb
    └── heston_calibration.ipynb
```

### 6.3 Data Flow

```
User selects stock → yfinance fetch (cached 5min) → compute historical vol
         ↓
User configures option params (K, T, r, σ, q)
         ↓
    ┌────┴────┐────────────┐
    ↓         ↓            ↓
  BS Model  Bayesian     NN Ensemble
  (instant) Tree (<2s)   (3-10s)
    ↓         ↓            ↓
    └────┬────┘────────────┘
         ↓
  Model Comparison Dashboard (Plotly charts)
```

### 6.4 Performance Targets

| Operation | Target Latency |
|---|---|
| Stock search (typeahead) | < 50ms |
| Yahoo Finance data fetch | < 2s (cached) |
| Black-Scholes computation | < 10ms |
| Bayesian Tree (100 steps) | < 2s |
| Neural Ensemble inference | < 5s (pre-trained) |
| Neural Ensemble training | < 60s (on-the-fly) |
| Chart rendering | < 500ms |
| Full page load | < 3s |

---

## 7. Model Specifications

### 7.1 Black-Scholes Formulae

**Call:**  
`C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)`

**Put:**  
`P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)`

Where:  
`d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)`  
`d2 = d1 - σ·√T`

### 7.2 Bayesian Binomial Tree

- CRR parameters: `u = e^(σ·√Δt)`, `d = 1/u`, `p = (e^((r-q)·Δt) - d) / (u - d)`
- Prior on σ: `σ ~ LogNormal(μ_σ, τ_σ)` where μ_σ and τ_σ are estimated from rolling vol windows
- Posterior sampling: 1000 draws from prior → 1000 tree evaluations → empirical posterior on option price
- Report credible intervals from posterior

### 7.3 Neural Ensemble (Poddar-Inspired)

- **Architecture per network**: 2-layer LSTM (hidden_dim=64) → Dense(32) → Dense(1)
- **Ensemble size**: 5 networks
- **Input features**: [S/K (moneyness), τ (time to expiry), σ_hist, r, q, RSI_14, vol_ratio_5_20]
- **Training data**: 50,000 simulated Heston paths per stock
- **Heston parameters**: calibrated from historical returns (v0, κ, θ, ξ, ρ estimated via MLE)
- **Loss**: MSE on BS-computed prices (for supervised pretraining) + fine-tuning on historical option prices if available
- **Ensemble uncertainty**: σ_ensemble = std(predictions across 5 networks)
- **Blending**: `price_final = w·price_NN + (1-w)·price_BS` where `w = sigmoid(-α·σ_ensemble + β)` with α, β optimised on validation set

---

## 8. Non-Functional Requirements

- **Dark Mode Only**: No light mode toggle. The entire UI is designed exclusively for dark backgrounds.
- **Single Page App**: All interactions happen on one Streamlit page with sidebar + tabs. No multi-page routing.
- **Offline Resilience**: If Yahoo Finance is unreachable, show last cached data with a stale-data warning banner.
- **Browser Support**: Chrome, Firefox, Edge (latest 2 versions). Safari best-effort.
- **Python Version**: 3.10+
- **No Authentication**: This is a local-first tool. No login required.
- **Stateless**: No database. All state lives in Streamlit session state and file-based caches.

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Yahoo Finance rate limiting | Data unavailability | Aggressive caching (5min TTL), fallback to last-known data |
| NN training too slow for on-the-fly | Poor UX | Pre-train on Nifty 50 stocks; show progress bar for others |
| Indian option chain data sparse on Yahoo | Missing IV/chain data | Graceful degradation — hide chain tab if unavailable, use historical vol only |
| Heston calibration instability | Bad NN inputs | Bound parameter estimates, fall back to constant-vol model |
| Streamlit rendering limits | Complex charts lag | Use Plotly WebGL for large datasets, limit data points |

---

## 10. Success Metrics

- All three models produce prices within 5% of each other for ATM options on liquid Nifty 50 stocks
- Full pricing computation completes in under 10 seconds for any stock
- Dashboard renders cleanly on a 1920×1080 display with no horizontal scroll
- NN ensemble uncertainty correlates with actual pricing error (backtest validation)
- Search bar returns correct results for any Nifty 500 stock within 2 keystrokes

---

## 11. Future Scope (v2+)

- American option pricing (early exercise) via Longstaff-Schwartz
- Options strategy builder (straddle, strangle, spreads) with combined payoff diagrams
- Real-time streaming from NSE via WebSocket
- Portfolio-level Greeks aggregation
- Export pricing reports as PDF
- Mobile-optimised layout
- Integration with broker APIs (Zerodha Kite, Angel One) for live trading
