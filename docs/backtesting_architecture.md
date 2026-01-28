# Minimal Backtesting Abstraction (Good Enough)

This repository uses the **minimum abstraction needed** to support flexible strategy research
(regime timing, ML, rules-based) without over-engineering.

---

## Core Principle

Abstract **only what changes independently**:

- data sources & frequencies
- feature construction
- strategy logic
- execution timing

Everything else stays procedural.

---

## The 5 Core Components

### 1. DataAdapter
**Responsibility:** load raw data and return aligned `ModelInputs`.

**Contract**
- `load(spec) -> raw_data`
- `prepare(raw_data, spec, required_features) -> ModelInputs`

**Notes**
- Delegates feature construction to `FeaturePipeline`
- Handles index alignment and column selection

---

### 2. FeaturePipeline
**Responsibility:** feature computation and aggregation (e.g. 15min → daily).

**Contract**
- `build(raw_data, spec, required_features) -> DataFrame`

**Rules**
- Produces *decision-time* features (as-of close t)
- No execution lag or shifting
- Safe to run on full sample (no learned parameters)

---

### 3. Strategy
**Responsibility:** map features → decision weights.

**Contract**
- `required_features(spec) -> list[str]`
- `fit(train_inputs, spec) -> None` (optional)
- `predict(inputs, spec) -> Series | DataFrame` (decision weights)

**Rules**
- No shifting or execution logic
- Output indexed by decision dates
- Weights, not trades

---

### 4. Simulator
**Responsibility:** apply timing, costs, and compute PnL.

**Contract**
- `simulate(returns, decision_weights, weight_lag, ...) -> timeseries`

**Rules**
- **Only place** where execution lag is applied
- Handles turnover, costs, rebalancing

---

### 5. BacktestEngine
**Responsibility:** orchestration only.

**Flow**
1. Instantiate strategy
2. Load raw data
3. Build features via adapter + pipeline
4. Fit/predict (single-fit or walk-forward)
5. Simulate execution
6. Compute metrics and package artifacts

**Rules**
- No strategy logic
- No feature logic
- No execution logic

---

## Timing Convention

- Features known at **close t**
- Strategy outputs weights at **close t**
- Simulator applies weights at **t + weight_lag**

Default:
```text
weight_lag = 1  (execute next day)
