import pandas as pd
import ta


def analyze_stock(file_path):

    # Load stock CSV
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Set date as index
    df.set_index("date", inplace=True)

    # =====================================================
    # Daily Returns
    # =====================================================

    df["daily_return"] = df["close"].pct_change()

    # =====================================================
    # Moving Averages (SMA + EMA)
    # =====================================================

    windows = [10, 20, 50, 100]

    for w in windows:

        # Simple Moving Average
        df[f"sma_{w}"] = (
            df["close"]
            .rolling(window=w)
            .mean()
        )

        # Exponential Moving Average
        df[f"ema_{w}"] = (
            df["close"]
            .ewm(span=w, adjust=False)
            .mean()
        )

    # =====================================================
    # RSI (Relative Strength Index)
    # =====================================================

    df["rsi"] = ta.momentum.RSIIndicator(
        close=df["close"],
        window=14
    ).rsi()

    # =====================================================
    # MACD
    # =====================================================

    macd = ta.trend.MACD(
        close=df["close"]
    )

    df["macd"] = macd.macd()

    df["macd_signal"] = macd.macd_signal()

    # =====================================================
    # Handle Missing Values
    # =====================================================

    df = df.dropna()

    return df