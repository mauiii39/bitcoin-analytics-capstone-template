# Stacking Sats: Improving Bitcoin Accumulation 

Building and improving data-driven Bitcoin accumulation strategies, with a focus on utilizing signal from predicion market data. 

See [stackingsats.org](https://www.stackingsats.org/) for more information.

---

## The Mission: Exploring Institutional Bitcoin Accumulation

As Bitcoin matures as an institutional asset, standard Dollar Cost Averaging (DCA) is a strong baseline, but there may be room for optimization. This project facilitates the design of **data-driven, long-only** accumulation strategies. The aim is to explore methods that maintain DCA’s systematic discipline while potentially **improving acquisition efficiency** within fixed budgets and time horizons.

### Latest Tournament
Trilemma Foundation hosts tournaments to find the most efficient accumulation models.
* **Current/Recent:** [Stacking Sats Tournament - MSTR 2025](https://github.com/TrilemmaFoundation/stacking-sats-tournament-mstr-2025)

---

## Repository Overview

This repository provides a template and framework for:
1.  **Exploratory Data Analysis (EDA)** of Bitcoin price action and on-chain properties.
2.  **Feature Engineering** that integrates prediction market sentiment (Polymarket), macro indicators, and on-chain metrics.
3.  **Strategy Development** for daily purchase schedules (dynamic DCA).
4.  **Backtesting & Evaluation** against uniform DCA benchmarks.

### Repository Structure

```text
.
├── template/                        # CORE FRAMEWORK (Start here)
│   ├── prelude_template.py          # Data loading & Polymarket utilities
│   ├── model_development_template.py # IMPLEMENT YOUR MODEL LOGIC HERE
│   ├── backtest_template.py         # Evaluation engine
│   └── *.md                         # Documentation for model logic & backtesting
├── example_1/                       # REFERENCE IMPLEMENTATION
│   ├── run_backtest.py              # How to run the example
│   └── model_development_example_1.py # Example Polymarket + MVRV integration
├── data/                            # Bitcoin & Polymarket source data
├── output/                          # Results and visualizations
└── tests/                           # Unit tests for core logic
```

---

## Getting Started

### 1. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TrilemmaFoundation/bitcoin-analytics-capstone-template
    cd bitcoin-analytics-capstone-template
    ```

2.  **Setup environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\\Scripts\\activate
    pip install -r requirements.txt
    ```

### 2. Data Acquisition

The `data/` directory contains historical BTC price data and specific Polymarket datasets (Politics, Finance, Crypto).

Data can be [downloaded manually from Google Drive](https://drive.google.com/drive/folders/1gizJ_n-QCnE8qrFM-BU3J_ZpaR3HCjn7?usp=sharing) into the `data/` folder, or you can use the automated script:

```bash
python data/download_data.py
```

**Included Data:**
* **CoinMetrics BTC Data**: Daily OHLCV and network metrics.
  * **Bitcoin Price Source of Truth**: The `PriceUSD` column in the CoinMetrics data is the source of truth for BTC-USD prices. This is renamed to `PriceUSD_coinmetrics` in the codebase. This is the only column you hypothetically need to build a model (along with the datetime index, of course).
* **Polymarket Data**: High-fidelity parquet files containing trades, odds history, and market metadata.

**External Data:**
External data is encouraged; students are responsible for ensuring that the data license permits all project participants to access and use (i.e., no proprietary data).

**System Requirements:**
Assume a modern laptop specification (think 16GB M4 Air).

---

## Model Development Guidelines

The framework includes a **Template Baseline** in `template/`. This serves as a starting point, currently implementing a simple 200-day Moving Average filter (accumulating more when price is below the MA).

### Exploration Path: Prediction Market Integration

A core opportunity lies in evolving this baseline into a market-aware strategy, perhaps by leveraging **Polymarket data**.

**Illustrative Examples:**
*   **Election Probabilities**: You might investigate if political event probabilities correlate with BTC volatility.
*   **Economic Indicators**: Consider checking if prediction markets for Fed rate cuts act as leading indicators.
*   **Retail Sentiment**: Specific "Polymarket Crypto" markets could potentially serve as proxies for retail sentiment or exuberance.

### Running Backtests

**Backtest Date Range:**
* **Range:** `2018-01-01` to `2025-12-31` (inclusive; daily frequency; no days should be missing)
* The backtest engine uses rolling 1-year windows starting from the start date, generating daily windows until the end date.

**Baseline Model:**
```bash
python -m template.backtest_template
```

**Reference Implementation (Example 1):**
```bash
python -m example_1.run_backtest
```

---

## Key Performance Indicators

When evaluating strategies, you might consider the following metrics (which are calculated by the automated backtest engine):

1.  **Win Rate**: Useful for understanding consistency—how often does the strategy outperform a standard DCA over 1-year windows?
2.  **SPD (Sats Per Dollar)**: A measure of raw efficiency—are you acquiring more bitcoin for the same capital?
3.  **Model Score**: A composite metric that balances performance (Win Rate) with risk-adjusted returns, offering a holistic view of strategy health.

## Licensing

*   **Code:** This repository, including its analysis and documentation, is open-sourced under the **MIT License**.
*   **Data:** The data provided (e.g., CoinMetrics, Polymarket) is not covered by the MIT license and retains its original licensing terms. Please refer to the respective data providers for their terms of use.

---

## Contacts & Community

* **App:** [stackingsats.org](https://www.stackingsats.org/)
* **Website:** [trilemma.foundation](https://www.trilemma.foundation/)
* **Foundation:** [Trilemma Foundation](https://github.com/TrilemmaFoundation)
