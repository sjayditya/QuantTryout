# OptiPrice India — Claude Development Guidelines

> Neural-Augmented Options Pricing Engine for Indian Equities  
> Tech Stack: Python · Streamlit · PyTorch · yfinance · Plotly

---

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- Before touching pricing model code, write out the math in comments first

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution
- Model implementations (BS, Bayesian, NN) can be developed in parallel subagents

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness
- For pricing models: validate against known analytical solutions before shipping

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it
- Financial code demands clarity — no clever tricks that obscure the math

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
- **Type Safety**: Use type hints everywhere in Python. Models deal with money — precision matters.
- **Test Coverage**: Every pricing model must have unit tests against known values.

---

## Project-Specific Rules

### Data Layer (`src/data/`)
- All Yahoo Finance calls go through `yahoo_fetcher.py` — never call yfinance directly from UI code
- Cache aggressively with `@st.cache_data(ttl=300)` for market data (5 min TTL)
- Use `@st.cache_resource` for model weights and Nifty 500 list (session-lifetime)
- NSE ticker format: always append `.NS` suffix (e.g., `RELIANCE.NS`)
- Handle Yahoo Finance failures gracefully — show stale data banner, never crash
- Nifty 500 list lives in `data/nifty500.csv` with columns: `symbol, company_name, sector, industry`

### Pricing Models (`src/models/`)
- Each model is a standalone module with a consistent interface:
  ```python
  def price(S, K, T, r, sigma, q, option_type) -> PricingResult
  def greeks(S, K, T, r, sigma, q, option_type) -> GreeksResult
  ```
- `PricingResult` and `GreeksResult` are dataclasses defined in `src/models/__init__.py`
- Black-Scholes is the ground truth baseline — always compute it first
- Bayesian tree: default 100 steps, 1000 posterior samples. Configurable via sidebar.
- Neural ensemble: 5 LSTM networks, always. Never reduce ensemble size.
- All models must handle edge cases: T=0 (expiry), S=K (ATM), deep ITM/OTM
- Never return NaN or Inf — clamp, warn, and return boundary values

### Neural Network Specifics (`src/models/neural_ensemble.py`, `src/models/heston.py`)
- Architecture: 2-layer LSTM (hidden=64) → Dense(32) → Dense(1), per Poddar (2026)
- Ensemble: 5 independent networks with different random seeds
- Heston simulation for training data: bound parameters sensibly
  - v0 ∈ [0.01, 1.0], κ ∈ [0.1, 10], θ ∈ [0.01, 1.0], ξ ∈ [0.1, 2.0], ρ ∈ [-0.95, 0.0]
- Pre-trained weights stored in `data/models/{symbol}_ensemble.pt`
- On-the-fly training: show a progress bar with ETA. Cap at 60 seconds.
- Ensemble disagreement = std across 5 predictions → drives confidence score
- Blending with BS: `w = sigmoid(-α * σ_ensemble + β)` — high disagreement → more BS weight

### UI Layer (`src/ui/`)
- **Dark mode only** — no light mode code, no theme toggle
- All custom CSS lives in `src/ui/styles.py` injected via `st.markdown()`
- Color-code models consistently everywhere:
  - Black-Scholes → Blue (`#448aff`)
  - Bayesian Tree → Amber (`#ffab00`)
  - Neural Ensemble → Purple (`#b388ff`)
- Use Plotly for all charts — never matplotlib in the UI (matplotlib OK in notebooks)
- Chart defaults: dark template (`plotly_dark`), transparent background, consistent font
- Search bar: use `st.text_input` with debounce logic — filter Nifty 500 list client-side
- Sidebar owns all inputs. Main area owns all outputs. Never mix.
- Show loading spinners for any computation > 500ms
- Tooltips on every parameter explaining what it means in plain English

### Visualization Standards (`src/ui/charts.py`)
- Every chart function returns a `plotly.graph_objects.Figure`
- Standard chart config:
  ```python
  CHART_CONFIG = {
      "template": "plotly_dark",
      "paper_bgcolor": "rgba(0,0,0,0)",
      "plot_bgcolor": "rgba(0,0,0,0)",
      "font": {"family": "JetBrains Mono, monospace", "color": "#e8e8f0"},
      "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
  }
  ```
- Required visualizations (implement in this order):
  1. Price comparison bar chart (3 models side-by-side with error bars)
  2. Greeks radar chart (overlay all models)
  3. Price vs. Strike curve (all models overlaid, market price as reference)
  4. Sensitivity tornado chart (per model)
  5. Volatility surface heatmap (toggle by model)
  6. Bayesian posterior distribution histogram
  7. NN ensemble disagreement strip/swarm plot
  8. Convergence plot (tree steps vs. price)

---

## Code Style

- Python 3.10+ features allowed (match/case, union types with `|`, etc.)
- Use `ruff` for linting, `black` for formatting (line length 100)
- Imports: stdlib → third-party → local, separated by blank lines
- Docstrings: Google style on all public functions
- No magic numbers — extract to named constants in `src/utils/config.py`
- Financial formulas: comment every line with the mathematical notation
  ```python
  # d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
  d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  ```

---

## Testing Standards

- Test framework: `pytest`
- Test every pricing model against known analytical values:
  - BS: compare against textbook examples (Hull, Chapter 15)
  - Bayesian: posterior mean should converge to BS price as sample size → ∞ and prior → diffuse
  - NN: ensemble mean on synthetic data should be within 5% of BS
- Test edge cases: T=0, very deep ITM/OTM, σ→0, σ very large
- Test data fetcher with mock yfinance responses
- Test search filtering logic independently
- Run tests before every commit: `pytest tests/ -v --tb=short`

---

## Git Conventions

- Branch naming: `feat/`, `fix/`, `refactor/`, `docs/`
- Commit messages: `type(scope): description` (e.g., `feat(models): implement bayesian binomial tree`)
- One logical change per commit
- Never commit broken code to main
- Tag releases: `v0.1.0`, `v0.2.0`, etc.

---

## Common Pitfalls — Don't Repeat These

- ❌ Calling `yfinance.download()` in a Streamlit callback without caching → hammers API
- ❌ Using `st.experimental_rerun()` in a loop → infinite rerender
- ❌ Computing NN inference on every slider drag → use `st.form()` or debounce
- ❌ Plotting with matplotlib default style in a dark UI → invisible axes
- ❌ Forgetting `.NS` suffix on Indian tickers → Yahoo returns wrong/empty data
- ❌ Using `float` division carelessly in tree pricing → accumulating rounding errors at high step counts
- ❌ Not handling `T=0` edge case → division by zero in BS formula
- ❌ Caching model weights per-session but never clearing → memory leak on long runs

---

## Reference Paper

**Uncertainty-Aware Deep Hedging**  
Manan Poddar, London School of Economics  
arXiv:2603.10137v1, March 2026  

Key ideas adapted for this project:
- Deep ensemble of 5 LSTM networks for uncertainty quantification
- Ensemble disagreement as a confidence signal
- CVaR-optimised blending between neural and classical (BS) strategies
- Heston stochastic volatility for realistic training data generation
