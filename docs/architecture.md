# Quant Backtesting System — General Architecture

## Overview
This system is designed as a **uniform, strategy-agnostic backtesting framework** with a **read-only visualization layer**.  
Strategies produce signals, the engine produces truth, and the dashboard communicates results.

The data flow is strictly one-directional:

**Data → Strategy → Portfolio → Execution → Scoring → Artifacts → Dashboard**

---

## Core Design Principles
- **Single backtesting engine** shared across all strategies
- **Pluggable strategies** with a fixed interface
- **Explicit timing rules** to prevent lookahead bias
- **Reproducible runs** defined by versioned configuration
- **Dashboard reads artifacts only** (no recomputation)

---

## Architecture Flow

### 1. Data Layer
**Responsibility:** Provide aligned, lookahead-safe inputs.

- Prices / returns
- Features (forecasts, regimes, factors)
- Benchmarks and exogenous variables
- Unified time index

**Output:**  
Canonical dataset indexed by time

---

### 2. Strategy Layer
**Responsibility:** Convert data into trading intent.

- Consumes the canonical dataset
- Produces a **signal time series**
- No PnL, costs, or metrics logic

**Output:**  
Signal or target exposure series

---

### 3. Portfolio Construction
**Responsibility:** Convert signals into tradable positions.

- Volatility targeting
- Leverage bounds
- Regime filters
- Position normalization

**Output:**  
Target portfolio weights

---

### 4. Execution Engine
**Responsibility:** Simulate trading mechanics.

- Apply timing rules (signal → trade)
- Multiply weights by realized returns
- Apply transaction costs and slippage
- Track turnover and exposure

**Output:**  
Strategy return and equity time series (gross and net)

---

### 5. Scoring & Metrics
**Responsibility:** Evaluate performance.

- Summary metrics (Sharpe, CAGR, max drawdown)
- Rolling statistics
- Strategy-specific scores (e.g., QLIKE)

**Output:**  
One-row metric summary per run

---

### 6. Artifact Storage
**Responsibility:** Persist results in a uniform schema.

- **Run Registry:** metadata and parameters
- **Time Series:** returns, equity, weights
- **Metrics:** performance summaries

Artifacts are immutable and versioned by `run_id`.

---

### 7. Dashboard Layer
**Responsibility:** Communicate results.

- Reads stored artifacts only
- Filters and compares backtest runs
- Visualizes performance and diagnostics
- No strategy logic or backtest execution

---

## Strategy Lifecycle

1. Implement strategy logic
2. Define run configuration (parameters, dates, universe)
3. Execute backtest
4. Store standardized artifacts
5. Automatically available in dashboard

---

## Key Separation of Concerns

Data Layer
  - ingestion
  - features
  - returns

Model Layer
  - Strategy (signals)
  - Fit / Predict

Portfolio Layer
  - Allocator (weights)
  - Constraints
  - Risk

Execution Layer
  - Simulator (returns + weights only)

Evaluation Layer
  - Metrics
  - Plots
