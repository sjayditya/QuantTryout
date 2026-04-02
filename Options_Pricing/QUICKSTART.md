# OptiPrice India — Quick Start Guide

## Overview

OptiPrice India is a professional options pricing dashboard for Indian stocks (Nifty 500). It provides three pricing models:
- **Black-Scholes** (analytical)
- **Bayesian Binomial Tree** (probabilistic)
- **Neural Network Ensemble** (deep learning with uncertainty quantification)

## Installation

### 1. Clone the Repository

```bash
cd "/home/bankai/Desktop/Quants work/Options_Pricing"
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration (optional):
```bash
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## Running the Application

### Standard Version (BS + Bayesian only)
```bash
streamlit run app.py
```

### Enhanced Version (All 3 models + advanced visualizations)
```bash
streamlit run app_enhanced.py
```

The dashboard will open at `http://localhost:8501`

## Usage

### 1. Stock Selection
- Use the search bar in the sidebar to find any stock from the Nifty 500 universe
- Search by symbol (e.g., "RELIANCE") or company name (e.g., "Reliance Industries")
- The stock info card shows current price, 52-week range, market cap, and sector

### 2. Options Configuration
- **Option Type**: Call or Put
- **Strike Price**: Default is ATM (at-the-money), adjust as needed
- **Expiry Date**: Select from calendar or enter days to expiry
- **Risk-Free Rate**: Default is 7% (Indian 10Y G-Sec), adjustable
- **Volatility**: Auto-computed from historical data (30/60/90/252 days), can override
- **Dividend Yield**: Auto-fetched from Yahoo Finance, adjustable

### 3. Calculate & Compare
- Click **Calculate** to run all three pricing models
- View results in multiple tabs:
  - **Prices**: Bar chart comparison + Price vs Strike curves
  - **Greeks**: Radar chart and detailed table
  - **Bayesian**: Posterior distribution + convergence plot
  - **Neural**: Ensemble disagreement + confidence metrics
  - **Sensitivity**: Tornado chart showing parameter impacts

## Features

### Stock Universe
- **Nifty 500** constituents (500+ stocks)
- Dynamic search with fuzzy matching
- Real-time data from Yahoo Finance
- Automatic `.NS` suffix handling for NSE tickers

### Pricing Models

#### Black-Scholes
- Closed-form analytical solution
- Full Greeks: Delta, Gamma, Theta, Vega, Rho
- Instant computation (<10ms)

#### Bayesian Binomial Tree
- Cox-Ross-Rubinstein tree with Bayesian uncertainty
- 1000 posterior samples
- 95% credible intervals
- Convergence analysis

#### Neural Network Ensemble
- 5 independent LSTM networks
- Trained on Heston-calibrated synthetic data
- Uncertainty-aware blending with Black-Scholes
- Confidence scoring based on ensemble agreement

### Visualizations
- Price comparison bar chart with error bars
- Greeks radar chart (normalized)
- Bayesian posterior histogram
- Binomial tree convergence plot
- Price vs Strike curves (all models overlaid)
- Sensitivity tornado chart (±10% parameter changes)
- Neural ensemble disagreement plot
- Real-time market context (Nifty 50 index)

## Example Workflow

1. **Search for a stock**: Type "TCS" or "Tata Consultancy"
2. **Configure option**: Call, Strike ₹3500, 30 days to expiry
3. **Click Calculate**
4. **Compare models**:
   - BS: ₹145.23
   - Bayesian: ₹143.87 (CI: ₹138.12 - ₹149.45)
   - Neural: ₹144.56 (Confidence: 87%)
5. **Analyze Greeks**: Delta 0.58, Gamma 0.003, Theta -2.1
6. **Check sensitivity**: Volatility has highest impact

## Tips

- **For liquid stocks** (Nifty 50): All three models converge closely
- **For volatile stocks**: Bayesian credible intervals widen, Neural confidence drops
- **ATM options**: Models agree most; deep ITM/OTM shows divergence
- **Short expiry**: Time decay (Theta) dominates
- **High volatility**: Vega becomes critical

## Troubleshooting

### Yahoo Finance Errors
- Data is cached for 5 minutes
- If fetch fails, try another stock or wait briefly
- Some stocks may have limited option chain data

### Neural Network Training
- First-time training takes 30-60 seconds
- Pre-trained weights available for Nifty 50 stocks
- Progress bar shows training status

### Performance
- Black-Scholes: <10ms
- Bayesian Tree: ~2s (100 steps)
- Neural Ensemble: 3-10s (inference) or 30-60s (training)

## Data Sources

- **Stock Prices**: Yahoo Finance (`yfinance`)
- **Nifty 500 List**: Static CSV (periodically updated)
- **Risk-Free Rate**: Default 7% (Indian 10Y G-Sec)
- **Volatility**: Historical log-returns (252 trading days)

## Keyboard Shortcuts

- **Ctrl+R**: Refresh data
- **Ctrl+K**: Focus search bar
- **Ctrl+Enter**: Submit form (Calculate)

## Next Steps

- Explore different stocks across sectors
- Compare model behavior for ITM vs OTM options
- Analyze sensitivity to volatility changes
- Study Bayesian posterior distributions
- Monitor Neural ensemble confidence scores

## Support

For issues or questions:
- Check `PRD.md` for detailed specifications
- Review `claude.md` for development guidelines
- See `tasks/lessons.md` for known issues

---

**Built with Streamlit • Powered by Yahoo Finance • Designed for Indian Markets**
