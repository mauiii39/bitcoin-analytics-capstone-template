# EDA Starter Template Overview

## Purpose

The `eda_starter_template.py` provides a comprehensive exploratory data analysis (EDA) framework for analyzing Bitcoin and Polymarket data. It demonstrates best practices for efficient data processing using Polars with lazy evaluation and includes built-in memory tracking capabilities.

## Key Features

- **Polars Lazy Evaluation**: Uses `scan_csv()` and `scan_parquet()` for efficient, memory-conscious data loading
- **Memory Tracking**: Built-in psutil-based memory monitoring for all operations
- **Modular Design**: Well-organized functions for data loading, analysis, and visualization
- **Type Safety**: Full type hints throughout for better code maintainability
- **Error Handling**: Graceful error handling with informative messages

## Architecture

The template is organized into clear sections:

### 1. Configuration
- Automatically determines project root and data directories using `pathlib.Path`
- Sets up paths for Coin Metrics Bitcoin data and Polymarket data
- Creates output directory for plots

### 2. Memory Tracking Utilities
- `get_memory_usage_mb()`: Retrieves current process memory usage
- `format_memory()`: Formats memory values in human-readable format (MB/GB)
- `track_memory()`: Context manager for tracking memory before/after operations

### 3. Data Loading Functions

#### `load_bitcoin_data(filepath: Path) -> Optional[pl.DataFrame]`
- Loads Bitcoin data from CSV using Polars lazy scan
- Automatically parses datetime columns
- Uses `infer_schema_length=10000` for accurate type inference
- Returns a Polars DataFrame or None on error

#### `load_polymarket_data(datadir: Path) -> Optional[dict[str, pl.DataFrame]]`
- Loads multiple Polymarket parquet files (markets, odds, summary)
- Handles datetime columns intelligently (only converts if string type)
- Returns a dictionary mapping data types to DataFrames

### 4. Analysis Functions

#### Bitcoin Analysis
- **`analyze_btc_metrics(df: pl.DataFrame)`**: 
  - Computes descriptive statistics for key metrics (PriceUSD, CapMrktCurUSD, HashRate)
  - Generates correlation heatmap for Bitcoin metrics
  - Saves correlation matrix visualization

#### Polymarket Analysis
- **`analyze_polymarket_summary(data: dict[str, pl.DataFrame])`**:
  - Summarizes market counts (total, active, closed)
  - Calculates volume statistics (total, average per market)
  - Reports odds history and trade counts

### 5. Visualization Functions

#### `plot_btc_price(df: pl.DataFrame)`
- Creates time series plot of Bitcoin price history
- Saves to `plots/btc_price_history.png`

#### `plot_polymarket_volume(df: pl.DataFrame)`
- Generates bar chart of top 10 categories by volume
- Uses Polars for aggregation, converts to pandas for plotting
- Saves to `plots/polymarket_volume_by_category.png`

## Data Sources

### Bitcoin Data (Coin Metrics)
- **Location**: `data/Coin Metrics/coinmetrics_btc.csv`
- **Format**: CSV with datetime column
- **Key Metrics**: PriceUSD, CapMrktCurUSD, HashRate, TxCnt, and many more
- **Schema**: See `data/Coin Metrics/coinmetrics_spec.md` for full specification

### Polymarket Data
- **Location**: `data/Polymarket/`
- **Format**: Parquet files
- **Files**:
  - `finance_politics_markets.parquet`: Market metadata
  - `finance_politics_odds_history.parquet`: Time-series odds data
  - `finance_politics_summary.parquet`: Market summaries
- **Schema**: See `data/Polymarket/polymarket_btc_analytics_schema.md` for details

## Usage

### Basic Execution

```bash
python eda/eda_starter_template.py
```

### Expected Output

The script will:
1. Display initial memory usage
2. Load Bitcoin and Polymarket data with memory tracking
3. Perform analysis and generate summary statistics
4. Create visualizations in the `plots/` directory
5. Display final memory usage summary

### Output Files

All visualizations are saved to `eda/plots/`:
- `btc_price_history.png`: Bitcoin price over time
- `btc_correlation_matrix.png`: Correlation heatmap of Bitcoin metrics
- `polymarket_volume_by_category.png`: Top 10 categories by volume

## Memory Tracking

The template includes comprehensive memory tracking:

- **Initial Memory**: Baseline memory usage at script start
- **Per-Operation Tracking**: Memory delta for each major operation
- **Final Summary**: Total memory delta from start to finish

Example output:
```
[Memory] Initial memory usage: 201.23 MB
[Memory] Before loading Bitcoin data: 201.25 MB
[Memory] After loading Bitcoin data: 222.56 MB (Δ 21.31 MB)
...
[Memory] Final memory usage: 451.58 MB (Total Δ: 250.34 MB)
```

## Dependencies

Key dependencies (see `requirements.txt`):
- `polars==1.20.0`: Fast DataFrame library with lazy evaluation
- `psutil==6.1.0`: Process and system utilities for memory tracking
- `matplotlib==3.10.8`: Plotting library
- `seaborn==0.13.2`: Statistical visualization
- `pandas==2.3.3`: Used for visualization compatibility (conversion from Polars)

## Design Patterns

### Lazy Evaluation
- Uses `pl.scan_csv()` and `pl.scan_parquet()` for lazy loading
- Benefits: Query optimization, memory efficiency, parallel processing
- Execution happens only when `.collect()` is called

### Context Managers
- `track_memory()` context manager ensures memory tracking even if errors occur
- Clean separation of concerns with try/finally blocks

### Type Safety
- Full type hints using `Optional[pl.DataFrame]` and `Path`
- Better IDE support and error detection

### Error Handling
- Functions return `None` on error rather than raising exceptions
- Allows script to continue processing other data sources if one fails

## Extending the Template

To add new analyses:

1. **Add data loading function** (if needed):
   ```python
   def load_new_data(filepath: Path) -> Optional[pl.DataFrame]:
       with track_memory("loading new data"):
           df = pl.scan_csv(filepath).collect()
       return df
   ```

2. **Add analysis function**:
   ```python
   def analyze_new_data(df: pl.DataFrame) -> None:
       with track_memory("analyzing new data"):
           # Your analysis code
   ```

3. **Add visualization** (if needed):
   ```python
   def plot_new_data(df: pl.DataFrame) -> None:
       with track_memory("plotting new data"):
           # Your plotting code
   ```

4. **Integrate into main()**:
   ```python
   new_df = load_new_data(NEW_DATA_PATH)
   if new_df is not None:
       analyze_new_data(new_df)
       plot_new_data(new_df)
   ```

## Performance Considerations

- **Lazy Evaluation**: Polars optimizes queries before execution
- **Memory Efficiency**: Only loads data when needed via `.collect()`
- **Parallel Processing**: Polars automatically parallelizes operations
- **Type Inference**: `infer_schema_length=10000` ensures accurate type detection

## Best Practices Demonstrated

1. ✅ Use `pathlib.Path` instead of `os.path`
2. ✅ Type hints for all functions
3. ✅ Docstrings for documentation
4. ✅ Context managers for resource tracking
5. ✅ Lazy evaluation for large datasets
6. ✅ Error handling without breaking execution
7. ✅ Modular, single-responsibility functions
8. ✅ Memory monitoring for performance analysis
