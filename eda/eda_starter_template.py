"""
Exploratory Data Analysis (EDA) Starter Template

This template demonstrates how to perform EDA on Bitcoin and Polymarket data
using Polars with lazy evaluation for efficient data processing.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl
import psutil
import seaborn as sns

# --- Configuration ---
# Robustly determine the project root directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = SCRIPT_DIR / "plots"
COINMETRICS_PATH = DATA_DIR / "Coin Metrics" / "coinmetrics_btc.csv"
POLYMARKET_DIR = DATA_DIR / "Polymarket"

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(exist_ok=True)


# --- Memory Tracking Utilities ---


def get_memory_usage_mb() -> float:
    """
    Get current memory usage of the process in MB.

    Returns:
        Memory usage in megabytes
    """
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def format_memory(mb: float) -> str:
    """
    Format memory value in MB to human-readable string.

    Args:
        mb: Memory value in megabytes

    Returns:
        Formatted string (e.g., "123.45 MB" or "1.23 GB")
    """
    if mb < 1024:
        return f"{mb:.2f} MB"
    else:
        return f"{mb / 1024:.2f} GB"


@contextmanager
def track_memory(operation_name: str):
    """
    Context manager to track memory usage before and after an operation.

    Args:
        operation_name: Name of the operation being tracked

    Yields:
        None
    """
    memory_before = get_memory_usage_mb()
    print(f"[Memory] Before {operation_name}: {format_memory(memory_before)}")

    try:
        yield
    finally:
        memory_after = get_memory_usage_mb()
        memory_delta = memory_after - memory_before
        print(
            f"[Memory] After {operation_name}: {format_memory(memory_after)} "
            f"(Δ {format_memory(memory_delta)})"
        )


# --- Data Loading Functions ---


def load_bitcoin_data(filepath: Path) -> Optional[pl.DataFrame]:
    """
    Load Bitcoin data from CSV using Polars lazy scan.

    Args:
        filepath: Path to the Coin Metrics CSV file

    Returns:
        Polars DataFrame with parsed datetime column, or None if loading fails
    """
    print(f"Loading Bitcoin data from {filepath}...")
    try:
        with track_memory("loading Bitcoin data"):
            df = (
                pl.scan_csv(filepath, infer_schema_length=10000)
                .with_columns(pl.col("time").str.to_datetime())
                .collect()
            )
        print(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading Bitcoin data: {e}")
        return None


def load_polymarket_data(datadir: Path) -> Optional[dict[str, pl.DataFrame]]:
    """
    Load Polymarket data from parquet files using Polars lazy scan.

    Args:
        datadir: Directory containing Polymarket parquet files

    Returns:
        Dictionary mapping data type names to Polars DataFrames, or None if loading fails
    """
    print(f"Loading Polymarket data from {datadir}...")
    markets_path = datadir / "finance_politics_markets.parquet"
    odds_path = datadir / "finance_politics_odds_history.parquet"
    summary_path = datadir / "finance_politics_summary.parquet"

    data: dict[str, pl.DataFrame] = {}

    try:
        with track_memory("loading Polymarket data"):
            if markets_path.exists():
                # Load with lazy scan, then collect and handle datetime columns
                markets_df = pl.scan_parquet(markets_path).collect()
                
                # Convert datetime columns only if they exist and are strings
                # (parquet files may already have proper datetime types)
                datetime_cols = []
                for col_name in ["created_at", "end_date"]:
                    if col_name in markets_df.columns:
                        col_dtype = markets_df[col_name].dtype
                        if col_dtype == pl.String or col_dtype == pl.Utf8:
                            datetime_cols.append(pl.col(col_name).str.to_datetime())
                
                if datetime_cols:
                    markets_df = markets_df.with_columns(datetime_cols)
                
                data["markets"] = markets_df
                print(f"Loaded {len(markets_df)} markets.")

            if odds_path.exists():
                odds_df = pl.scan_parquet(odds_path).collect()
                data["odds"] = odds_df
                print(f"Loaded {len(odds_df)} odds history records.")

            if summary_path.exists():
                summary_df = pl.scan_parquet(summary_path).collect()
                data["summary"] = summary_df
                print(f"Loaded {len(summary_df)} summary records.")

        return data if data else None
    except Exception as e:
        print(f"Error loading Polymarket data: {e}")
        return None


# --- Bitcoin Analysis Functions ---


def analyze_btc_metrics(df: pl.DataFrame) -> None:
    """
    Analyze Bitcoin metrics and generate summary statistics.

    Args:
        df: Polars DataFrame containing Bitcoin data
    """
    print("\n--- Bitcoin Data Summary ---")

    # Select relevant columns and compute descriptive statistics
    metrics = ["PriceUSD", "CapMrktCurUSD", "HashRate"]
    available_metrics = [col for col in metrics if col in df.columns]

    if available_metrics:
        summary = df.select(available_metrics).describe()
        print(summary)

    # Correlation analysis
    correlation_cols = ["PriceUSD", "CapMrktCurUSD", "HashRate", "TxCnt"]
    available_corr_cols = [col for col in correlation_cols if col in df.columns]

    if len(available_corr_cols) >= 2:
        corr_df = df.select(available_corr_cols).to_pandas()
        corr = corr_df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation of Bitcoin Metrics")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "btc_correlation_matrix.png")
        print("Saved btc_correlation_matrix.png")
        plt.close()


# --- Polymarket Analysis Functions ---


def analyze_polymarket_summary(data: dict[str, pl.DataFrame]) -> None:
    """
    Analyze Polymarket data and generate summary statistics.

    Args:
        data: Dictionary containing Polymarket DataFrames
    """
    print("\n--- Polymarket Data Summary ---")

    markets_df = data.get("markets")
    if markets_df is not None:
        print(f"Total Markets: {len(markets_df)}")

        if "active" in markets_df.columns:
            active_count = markets_df["active"].sum()
            print(f"Active Markets: {active_count}")
            print(f"Closed Markets: {len(markets_df) - active_count}")

        if "volume" in markets_df.columns:
            total_volume = markets_df["volume"].sum()
            avg_volume = markets_df["volume"].mean()
            print(f"Total Volume: ${total_volume:,.2f}")
            print(f"Average Volume per Market: ${avg_volume:,.2f}")

    odds_df = data.get("odds")
    if odds_df is not None:
        print(f"Total Odds History Records: {len(odds_df):,}")

    summary_df = data.get("summary")
    if summary_df is not None and "trade_count" in summary_df.columns:
        total_trades = summary_df["trade_count"].sum()
        print(f"Total Trades: {total_trades:,}")


# --- Visualization Functions ---


def plot_btc_price(df: pl.DataFrame) -> None:
    """
    Plot Bitcoin price history over time.

    Args:
        df: Polars DataFrame containing Bitcoin data with 'time' and 'PriceUSD' columns
    """
    if "time" not in df.columns or "PriceUSD" not in df.columns:
        print("Required columns 'time' or 'PriceUSD' not found in Bitcoin data.")
        return

    # Convert to pandas for plotting (Polars doesn't have direct matplotlib integration)
    plot_df = df.select(["time", "PriceUSD"]).to_pandas()

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["time"], plot_df["PriceUSD"], label="BTC Price (USD)")
    plt.title("Bitcoin Price History")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "btc_price_history.png")
    print("Saved btc_price_history.png")
    plt.close()


def plot_polymarket_volume(df: pl.DataFrame) -> None:
    """
    Plot top 10 Polymarket categories by volume.

    Args:
        df: Polars DataFrame containing Polymarket markets data
    """
    if "volume" not in df.columns or "category" not in df.columns:
        print("Columns 'volume' or 'category' not found in Polymarket data.")
        return

    # Use Polars to compute top categories
    top_cats = (
        df.group_by("category")
        .agg(pl.col("volume").sum())
        .sort("volume", descending=True)
        .head(10)
    )

    if len(top_cats) == 0:
        print("No data available for volume by category plot.")
        return

    # Convert to pandas for seaborn plotting
    plot_df = top_cats.to_pandas()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=plot_df["volume"], y=plot_df["category"])
    plt.title("Top 10 Polymarket Categories by Volume")
    plt.xlabel("Total Volume")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "polymarket_volume_by_category.png")
    print("Saved polymarket_volume_by_category.png")
    plt.close()


# --- Main Execution ---


def main() -> None:
    """Main execution function for EDA workflow."""
    # Track overall memory usage
    initial_memory = get_memory_usage_mb()
    print(f"\n[Memory] Initial memory usage: {format_memory(initial_memory)}\n")

    # Load data using lazy evaluation
    btc_df = load_bitcoin_data(COINMETRICS_PATH)
    poly_data = load_polymarket_data(POLYMARKET_DIR)

    # Analyze Bitcoin data
    if btc_df is not None:
        with track_memory("analyzing Bitcoin metrics"):
            analyze_btc_metrics(btc_df)
        with track_memory("plotting Bitcoin price"):
            plot_btc_price(btc_df)

    # Analyze Polymarket data
    if poly_data is not None:
        with track_memory("analyzing Polymarket summary"):
            analyze_polymarket_summary(poly_data)
        if "markets" in poly_data:
            with track_memory("plotting Polymarket volume"):
                plot_polymarket_volume(poly_data["markets"])

    # Final memory summary
    final_memory = get_memory_usage_mb()
    total_delta = final_memory - initial_memory
    print(
        f"\n[Memory] Final memory usage: {format_memory(final_memory)} "
        f"(Total Δ: {format_memory(total_delta)})"
    )
    print("\nEDA Layout Complete. Check the 'plots' directory for visualizations.")


if __name__ == "__main__":
    main()
