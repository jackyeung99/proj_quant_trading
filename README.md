# Quantitative Trading Research Platform

A modular end-to-end quantitative trading research and execution platform for developing, backtesting, evaluating, and deploying systematic trading strategies.

The framework supports:

- High-frequency market data ingestion
- Medallion-style pipelines (Bronze → Silver → Gold)
- Walk-forward backtesting
- Volatility regime modeling
- Multi-asset portfolio strategies
- Live trading via Alpaca
- Cloud-native AWS deployment
- Interactive monitoring dashboards

# Overview

This project was developed as part of an independent research initiative focused on volatility forecasting, systematic portfolio construction, and live trading infrastructure.

Key design goals:

- **Modularity** — interchangeable data sources, signals, and execution engines
- **Reproducibility** — configuration-driven pipelines and artifact tracking
- **Scalability** — concurrent strategies and cloud deployment
- **Reliability** — structured logging, storage, and orchestration
- **Research** — rapid experimentation with walk-forward evaluation

# Architecture

```text
                    ┌────────────────────┐
                    │   Market Data      │
                    │  (Alpaca, FRED)    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │     Ingestion      │
                    │      Bronze        │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Standardization   │
                    │      Silver        │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Feature Engineering│
                    │       Gold         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Signal Generation  │
                    │  Regime Modeling   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Portfolio Weights  │
                    │ & Risk Management  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │     Execution      │
                    │  Alpaca Brokerage  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Evaluation & Dash  │
                    └────────────────────┘
```

# Core Features

## Data Engineering

- Multi-source ingestion framework
- Incremental data loading
- Medallion architecture:
  - Bronze → raw market data
  - Silver → standardized datasets
  - Gold → modeling-ready feature tables
- Intraday aggregation and realized volatility computation
- Configurable feature engineering pipelines

## Research & Backtesting

- Walk-forward validation
- Rolling retraining windows
- Parameter sweeps and threshold optimization
- Strategy benchmarking
- Long-only and long/short support
- Multi-asset portfolio construction

Tracked metrics include:

- Sharpe Ratio
- CAGR
- Max Drawdown
- Calmar Ratio
- Turnover
- Exposure
- Holding Period Statistics

# Repository Structure

```
repo/
├── configs/
│   ├── connections/
│   ├── datasets/
│   ├── deployments/
│   └── strategies/
│
├── data/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── external/
│
├── artifacts/
│   ├── live/
│   │   ├── execution/
│   │   ├── models/
│   │   ├── orders/
│   │   ├── performance/
│   │   ├── trades/
│   │   └── weights/
│   └── runs/
│
├── docs/
├── notebooks/
├── scripts/
│   ├── run_pipeline.py
│   ├── run_backtest.py
│   ├── build_table.py
│   └── publish_dashboard_results.py
│
├── src/qbt/
│   ├── backtesting/
│   ├── config/
│   ├── core/
│   ├── data/
│   │   └── sources/
│   ├── execution/
│   ├── features/
│   ├── metrics/
│   ├── pipeline/
│   ├── portfolio/
│   ├── storage/
│   ├── strategies/
│   └── utils/
│
├── test/

├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── requirements-cloud.txt
├── README.md
└── LICENSE
```


# Running the Pipeline
Create a `.env` file in the root directory of the project.

Example:

```env
# Primary Alpaca Account
ALPACA_API_KEY="xxxxxxxxxxxxxxxx"
ALPACA_API_SECRET="xxxxxxxxxxxxxxxx"
```
## Local

```bash
python scripts/run_pipeline.py \
    --cfg configs/deployments/sector_long_only.yaml
```

## Docker

```bash
docker build -t quant-trading .

docker run quant-trading \
    --cfg /app/configs/deployments/sector_long_only.yaml
```

# AWS Deployment

The platform supports cloud deployment using:

- AWS ECS/Fargate
- AWS EventBridge
- AWS ECR
- AWS S3

Each strategy can run as an isolated ECS task using separate deployment configurations.


# Design Principles

## Configuration-Driven

Strategies, datasets, and deployments are controlled through YAML configuration files rather than hardcoded logic.

## Modular Pipelines

Each stage of the pipeline is independently executable and testable.

## Research First

The framework prioritizes rapid experimentation while maintaining production-grade structure.

# Future Improvements

- Transaction cost modeling
- Borrow fee simulation for short positions
- Options support
- Real-time streaming pipelines
- Distributed backtesting
- Advanced portfolio optimization
- Market microstructure features

