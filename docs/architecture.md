# Quant Research System — General Architecture

## Overview
This system is a **uniform, model-agnostic research framework** with a **read-only visualization layer**.  
Components are swappable, outputs are standardized, and results are reproducible.

The flow is strictly one-directional:

**Data → Features → Model/Logic → Decisions → Realization → Evaluation → Artifacts → Dashboard**

---

## Core Design Principles
- **Single orchestration workflow** shared across all experiments
- **Pluggable models/logic** with a fixed interface
- **Explicit timing rules** to prevent lookahead bias
- **Reproducible runs** defined by versioned configuration
- **Dashboard reads artifacts only** (no recomputation)

---

## Architecture Flow

### 1. Data Layer
**Responsibility:** Provide aligned, lookahead-safe inputs.

- Raw sources (market, macro, alternative, internal)
- Unified time index and join rules
- Basic cleaning and validation

**Output:**  
Canonical dataset indexed by time (and optionally asset)

---

### 2. Feature Layer
**Responsibility:** Transform canonical data into decision-time features.

- Rolling statistics, transformations, normalizations
- Returns
- Aggregations and resampling (if not done in Data Layer)
- Feature sets defined by config
- Feature diagnostics (missingness, stability checks)

**Output:**  
Feature matrix aligned to the decision index

---

### 3. Model / Logic Layer
**Responsibility:** Map features into latent state, forecasts, or decisions.

Examples:
- Regime models (threshold, Markov switching, ML classifiers)
- Forecast models (volatility, returns, risk)
- Rule-based logic (signals, filters)

**Rules**
- Fit is performed on training slices only (if applicable)
- No realization/evaluation logic here

**Output:**  
Predictions, states, or decision signals (time-indexed)

---

### 4. Decision Layer
**Responsibility:** Convert model output into a normalized decision object.

Examples:
- Exposure targets (e.g., risk-on/off, leverage multiplier)
- Portfolio weights (single or multi-asset)
- Actions (optional: buy/sell/hold for discrete systems)

**Rules**
- Decisions are defined at **decision time** (e.g., close t)
- No timing lag or realization mechanics here

**Output:**  
Decision series (e.g., targets/weights) aligned to the decision index

---

### 5. Realization Layer
**Responsibility:** Apply timing rules and compute realized outcomes.

- Apply lag: decision at t affects outcome at t + lag
- Apply constraints (optional): bounds, turnover limits
- Apply costs/slippage (optional)
- Track diagnostics: turnover, exposure, drawdowns

**Output:**  
Realized time series (outcomes, PnL/equity, exposures)

---

### 6. Evaluation Layer
**Responsibility:** Summarize and score results.

- Summary metrics (Sharpe, CAGR, max drawdown, hit rate)
- Rolling statistics and diagnostics
- Model-specific scores (e.g., QLIKE, calibration)
- Comparisons vs benchmarks (optional)

**Output:**  
Metric summaries + evaluation artifacts

---

### 7. Artifact Storage
**Responsibility:** Persist results in a uniform schema.

- **Run Registry:** metadata, parameters, versions, tags
- **Time Series:** realized outcomes, decisions, key diagnostics
- **Metrics:** performance and scoring summaries

Artifacts are immutable and versioned by `run_id`.

---

### 8. Dashboard Layer
**Responsibility:** Communicate results.

- Reads stored artifacts only
- Filters and compares runs
- Visualizes outcomes and diagnostics
- No model execution or recomputation

---

## Experiment Lifecycle
1. Define experiment configuration (data, features, model, decision rules)
2. Run orchestration (single-fit or walk-forward)
3. Persist standardized artifacts
4. Automatically available in dashboard

---

## Key Separation of Concerns

Data Layer
  - ingestion
  - alignment
  - resampling

Feature Layer
  - transformations
  - rolling/aggregation
  - feature sets

Model/Logic Layer
  - fit / predict
  - regimes / forecasts / rules

Decision Layer
  - targets / weights / actions
  - normalization & simple constraints

Realization Layer
  - timing (lag)
  - costs/slippage
  - outcome generation

Evaluation Layer
  - metrics
  - diagnostics
  - scoring

Presentation Layer
  - dashboard (read-only)
