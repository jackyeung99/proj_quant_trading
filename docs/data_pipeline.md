# Clean Feature Pipeline (Minimal)

This document describes a **simple, robust path** from raw prices to
strategy-ready features, suitable for RV-based strategies.



## Data Flow

```
Raw Data
→ Bronze (raw ingestion)
→ Silver (canonical market data)
→ Gold (features)
→ Strategy Inputs
```



## Bronze: Raw Ingestion

- Raw intraday or daily prices
- Stored as received
- Append-only
- No resampling, no features



## Silver: Canonical Market Data

**Example: 15-minute intraday bars**

- Keys: `timestamp`, `asset`
- Columns: `open`, `high`, `low`, `close`, `volume`
- UTC timestamps
- Deduplicated and session-aligned
- Fixed bar grid (e.g. 15m)

Silver contains **clean prices only**.


## Gold: Feature Construction

**Daily features built from Silver**

- Keys: `date`, `asset`
- Columns (minimal):
  - `close` (daily close)
  - `ret_cc` or `ret_oo` (execution-aligned return)
  - `rv_1d` (realized volatility)

**RV definition**
\[
RV_t = \sum (\log C_i - \log C_{i-1})^2
\]
using intraday closes within day `t`.

Gold is:
- Deterministic
- Persisted (Parquet)
- Versioned via a manifest
- Reused across strategies



## Strategy Inputs (Runtime)

Strategies consume **wide matrices**, not storage formats:

```python
inputs = {
  "ret":   df_ret,    # index=date, columns=asset
  "rv_1d": df_rv,
}
