"""
Advanced Polymarket-style Dynamic DCA model for the capstone template repo.

This ports the core logic from the class-based model in the `bitcoin` repo into
the function-based API expected by this repo:

- precompute_features(df)
- compute_window_weights(features_df, start_date, end_date, current_date, ...)

The strategy blends:
- technical signals
- on-chain signals
- contrarian sentiment signals
- optional Polymarket-style features when present

It is designed to degrade gracefully if some columns are missing.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MIN_W = 1e-6

MA_SHORT = 50
MA_LONG = 200
VOL_WINDOW = 30
RSI_WINDOW = 14
ROLLING_YEAR = 365

TECH_WEIGHT = 0.35
ONCHAIN_WEIGHT = 0.30
SENTIMENT_WEIGHT = 0.35

# how aggressively score changes weights
DYNAMIC_STRENGTH = 1.35

FEATS = [
    "ma_short",
    "ma_long",
    "price_vs_ma_long",
    "volatility",
    "rsi",
    "mvrv",
    "mvrv_zscore",
    "mvrv_rank_365d",
    "nvt",
    "nvt_ma",
    "nvt_trend",
    "market_sentiment",
    "sentiment_momentum",
    "fed_rate_cut_prob",
    "btc_100k_prob",
    "buy_opportunity",
]


# =============================================================================
# Helpers
# =============================================================================

def _safe_series(df: pd.DataFrame, candidates: list[str], index: pd.Index) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            return s.reindex(index)
    return pd.Series(np.nan, index=index)


def _clip_rank(series: pd.Series, window: int = 365) -> pd.Series:
    return series.rolling(window).rank(pct=True)


def _compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.clip(0, 100)


def _load_optional_polymarket_daily(index):
    from pathlib import Path
    import pandas as pd
    import numpy as np

    out = pd.DataFrame(index=index)
    out["market_sentiment"] = 0.0
    out["fed_rate_cut_prob"] = np.nan
    out["btc_100k_prob"] = np.nan
    out["sentiment_momentum"] = 0.0

    base_dir = Path(__file__).resolve().parent.parent
    pm_dir = base_dir / "data" / "Polymarket"

    markets_fp = pm_dir / "finance_politics_markets.parquet"
    odds_fp = pm_dir / "finance_politics_odds_history.parquet"

    if not markets_fp.exists() or not odds_fp.exists():
        print("Polymarket raw files not found")
        return out.fillna(0.0)

    # 🔹 Load data
    markets = pd.read_parquet(markets_fp)
    odds = pd.read_parquet(odds_fp)

    # 🔹 Merge
    df = odds.merge(markets, on="market_id", how="left")

    print("\n--- DEBUG: AFTER MERGE ---")
    print("Shape:", df.shape)

    # 🔹 Use REAL timestamp from markets
    df["date"] = pd.to_datetime(df["created_at"]).dt.normalize()

    print("\n--- DEBUG: DATE RANGE ---")
    print("Min:", df["date"].min(), "Max:", df["date"].max())

    # 🔹 Normalize probability
    prob_col = "price"
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")

    if df[prob_col].max() > 1.5:
        df[prob_col] = df[prob_col] / 100.0

    # 🔥 CLASSIFICATION (fixed)
    def classify_market(q):
        q = str(q).lower()

        if any(k in q for k in ["bitcoin", "btc", "crypto", "eth", "ethereum"]):
            return "crypto"

        elif any(k in q for k in [
            "fed", "federal reserve", "interest rate", "rates",
            "inflation", "cpi", "economy", "recession"
        ]):
            return "macro"

        elif any(k in q for k in [
            "election", "president", "trump", "biden", "vote"
        ]):
            return "political"

        else:
            return "other"

    df["category"] = df["question"].apply(classify_market)

    # 🔹 keep only useful ones
    df = df[df["category"].isin(["crypto", "macro", "political"])]

    print("\n--- DEBUG: FIXED CATEGORY COUNTS ---")
    print(df["category"].value_counts())

    # 🔹 Aggregate
    daily = df.groupby(["date", "category"])[prob_col].mean().unstack()

    # 🔹 Normalize to [-1, 1]
    daily = (daily - 0.5) * 2
    daily = daily.fillna(0.0)

    print("\n--- DEBUG: DAILY ---")
    print(daily.head())

    # 🔹 Build sentiment
    sentiment = pd.Series(0.0, index=daily.index)

    if "crypto" in daily.columns:
        sentiment += 0.6 * daily["crypto"]

    if "macro" in daily.columns:
        sentiment += 0.25 * daily["macro"]

    if "political" in daily.columns:
        sentiment += 0.15 * daily["political"]

    sentiment = sentiment.clip(-1, 1)

    print("\n--- DEBUG: FINAL SENTIMENT ---")
    print(sentiment.describe())

    # 🔹 Align with BTC index
    aligned = sentiment.reindex(index)
    aligned = aligned.ffill().fillna(0.0)

    out["market_sentiment"] = aligned

    # 🔹 Momentum
    out["sentiment_momentum"] = out["market_sentiment"].diff(7).fillna(0.0)

    return out


def _clean_array(arr: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(arr), arr, 0.0)


# =============================================================================
# Feature engineering
# =============================================================================

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available columns: {list(df.columns)}")

    price = pd.to_numeric(df[PRICE_COL], errors="coerce").loc["2010-07-18":].copy()
    price = price.dropna()

    features = pd.DataFrame(index=price.index)
    features[PRICE_COL] = price

    # Technical features
    features["ma_short"] = price.rolling(MA_SHORT, min_periods=MA_SHORT // 2).mean()
    features["ma_long"] = price.rolling(MA_LONG, min_periods=MA_LONG // 2).mean()
    features["price_vs_ma_long"] = price / features["ma_long"]

    returns = price.pct_change()
    features["volatility"] = returns.rolling(VOL_WINDOW, min_periods=10).std() * np.sqrt(365)
    features["rsi"] = _compute_rsi(price, RSI_WINDOW)

    # On-chain features: try several common column names
    features["mvrv"] = _safe_series(df, ["MVRV", "mvrv"], features.index)
    features["nvt"] = _safe_series(df, ["NVT", "nvt"], features.index)

    features["mvrv_zscore"] = (
        (features["mvrv"] - features["mvrv"].rolling(ROLLING_YEAR).mean())
        / features["mvrv"].rolling(ROLLING_YEAR).std()
    )
    features["mvrv_rank_365d"] = _clip_rank(features["mvrv"], ROLLING_YEAR)

    features["nvt_ma"] = features["nvt"].rolling(30, min_periods=10).mean()
    features["nvt_trend"] = features["nvt_ma"] / features["nvt_ma"].rolling(90, min_periods=30).mean()

    # Optional Polymarket-style features
# === POLYMARKET FEATURES ===
    pm = _load_optional_polymarket_daily(features.index)

    # 🔥 Z-score normalization (rolling, robust)
    raw_sentiment = pm["market_sentiment"]

    sentiment_z = (
        raw_sentiment - raw_sentiment.rolling(90, min_periods=30).mean()
    ) / (raw_sentiment.rolling(90, min_periods=30).std() + 1e-8)

    features["market_sentiment"] = sentiment_z.clip(-2, 2)

    # 🔥 MOMENTUM (very important)
    features["sentiment_momentum"] = features["market_sentiment"].diff(7)

    # keep other polymarket features if they exist
    for col in ["fed_rate_cut_prob", "btc_100k_prob"]:
        if col in pm.columns:
            features[col] = pm[col]

    # Lag everything except raw price / slow moving averages where needed
    lag_cols = [
        "price_vs_ma_long",
        "volatility",
        "rsi",
        "mvrv",
        "mvrv_zscore",
        "mvrv_rank_365d",
        "nvt",
        "nvt_ma",
        "nvt_trend",
        "market_sentiment",
        "sentiment_momentum",
        "fed_rate_cut_prob",
        "btc_100k_prob",
    ]
    for col in lag_cols:
        features[col] = features[col].shift(1)

    features["buy_opportunity"] = _calculate_buy_opportunity_score(features)
    features = features.fillna(method="ffill").fillna(0.0)

    # Save debug CSV with timestamp index
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    return features


def _calculate_buy_opportunity_score(features: pd.DataFrame) -> pd.Series:
    """
    Port of the core 'AdvancedPolymarketBTCModel' idea into a daily score [0, 1].

    Higher score => stronger buy opportunity.
    """
    score = pd.Series(0.0, index=features.index)

    # -------------------------------------------------------------------------
    # Technical block
    # -------------------------------------------------------------------------
    technical = pd.Series(0.0, index=features.index)

    rsi = features["rsi"]
    technical += np.where(rsi < 20, 0.40, 0.0)
    technical += np.where((rsi >= 20) & (rsi < 30), 0.30, 0.0)
    technical += np.where((rsi >= 30) & (rsi < 40), 0.10, 0.0)

    pma = features["price_vs_ma_long"]
    technical += np.where(pma < 0.85, 0.40, 0.0)
    technical += np.where((pma >= 0.85) & (pma < 0.95), 0.20, 0.0)

    vol_pct = features["volatility"].rolling(ROLLING_YEAR).rank(pct=True)
    technical += np.where(vol_pct > 0.80, 0.20, 0.0)

    technical = technical.clip(0, 1)

    # -------------------------------------------------------------------------
    # On-chain block
    # -------------------------------------------------------------------------
    onchain = pd.Series(0.0, index=features.index)

    mvrv_z = features["mvrv_zscore"]
    onchain += np.where(mvrv_z < -1.5, 0.50, 0.0)
    onchain += np.where((mvrv_z >= -1.5) & (mvrv_z < -0.5), 0.30, 0.0)

    mvrv_rank = features["mvrv_rank_365d"]
    onchain += np.where(mvrv_rank < 0.20, 0.30, 0.0)
    onchain += np.where((mvrv_rank >= 0.20) & (mvrv_rank < 0.40), 0.20, 0.0)

    nvt_trend = features["nvt_trend"]
    onchain += np.where(nvt_trend < 0.80, 0.20, 0.0)

    onchain = onchain.clip(0, 1)

    # -------------------------------------------------------------------------
    # Sentiment block (contrarian Polymarket-style)
    # -------------------------------------------------------------------------
    sentiment = pd.Series(0.0, index=features.index)

    # 🔥 Continuous contrarian signal
    market_sentiment = features["market_sentiment"]

    # more negative sentiment → stronger buy
    sentiment += np.clip(-market_sentiment, 0, 2) * 0.4

    # penalize extreme bullishness (crowded trade)
    sentiment -= np.clip(market_sentiment - 0.5, 0, 2) * 0.2


    # 🔥 ADD MOMENTUM HERE
    sentiment_momentum = features["sentiment_momentum"]

    # improving sentiment → bullish confirmation
    sentiment += np.clip(sentiment_momentum, 0, 2) * 0.3

    # worsening sentiment → reduce exposure
    sentiment -= np.clip(-sentiment_momentum, 0, 2) * 0.2


    # 🔒 keep within bounds
    sentiment = sentiment.clip(0, 1)

    sentiment_momentum = features["sentiment_momentum"]
    sentiment += np.where(sentiment_momentum > 0.10, 0.20, 0.0)

    # AdvancedPolymarketBTCModel idea:
    # - higher fed rate cut probability is BTC-bullish
    # - high BTC 100k probability can be treated contrarian / crowded
    fed_prob = features["fed_rate_cut_prob"].fillna(0.0).clip(0, 1)
    btc_100k_prob = features["btc_100k_prob"].fillna(0.0).clip(0, 1)

    sentiment += 0.20 * fed_prob
    sentiment += 0.10 * (1.0 - btc_100k_prob)

    sentiment = sentiment.clip(0, 1)

    # Combined score
    score = (
        technical * TECH_WEIGHT
        + onchain * ONCHAIN_WEIGHT
        + sentiment * SENTIMENT_WEIGHT
    )

    return score.clip(0, 1)


# =============================================================================
# Weight allocation
# =============================================================================

def _compute_stable_signal(raw: np.ndarray) -> np.ndarray:
    n = len(raw)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    cumsum = np.cumsum(raw)
    running_mean = cumsum / np.arange(1, n + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = raw / running_mean

    return np.where(np.isfinite(signal), signal, 1.0)


def allocate_sequential_stable(
    raw: np.ndarray,
    n_past: int,
    locked_weights: np.ndarray | None = None,
) -> np.ndarray:
    n = len(raw)
    if n == 0:
        return np.array([])
    if n_past <= 0:
        return np.full(n, 1.0 / n)

    n_past = min(n_past, n)
    w = np.zeros(n)
    base_weight = 1.0 / n

    if locked_weights is not None and len(locked_weights) >= n_past:
        w[:n_past] = locked_weights[:n_past]
    else:
        for i in range(n_past):
            signal = _compute_stable_signal(raw[: i + 1])[-1]
            w[i] = max(signal * base_weight, MIN_W)

    past_sum = w[:n_past].sum()
    target_budget = n_past / n
    if past_sum > target_budget + 1e-10:
        w[:n_past] *= target_budget / past_sum

    n_future = n - n_past
    if n_future > 1:
        w[n_past : n - 1] = base_weight

    w[n - 1] = max(1.0 - w[: n - 1].sum(), 0.0)

    s = w.sum()
    if s <= 0:
        return np.full(n, 1.0 / n)

    return w / s


def compute_dynamic_multiplier(opportunity_score: np.ndarray) -> np.ndarray:
    """
    Convert [0, 1] opportunity scores into multiplicative raw weights.
    Center around 0.5:
    - >0.5 => overweight
    - <0.5 => underweight
    """
    centered = opportunity_score - 0.5
    adjustment = np.clip(centered * 2.0 * DYNAMIC_STRENGTH, -3, 3)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    opportunity = _clean_array(df["buy_opportunity"].values)
    dyn = compute_dynamic_multiplier(opportunity)
    raw = np.maximum(base * dyn, MIN_W)

    if n_past is None:
        n_past = n

    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
        n_past=n_past,
        locked_weights=locked_weights,
    )
    return weights.reindex(full_range, fill_value=0.0)