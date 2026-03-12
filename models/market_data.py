"""Fetches prices and asset info from Yahoo Finance. Results are cached per session."""

from __future__ import annotations

import time
import yfinance as yf
import pandas as pd


# Session-level cache to avoid repeated network calls
_price_cache: dict[str, float | None] = {}
_name_cache: dict[str, str] = {}

MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds


def _retry(func, *args, retries=MAX_RETRIES, **kwargs):
    """Retry a function on rate limit errors."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if ("Rate" in str(e) or "rate" in str(e) or "429" in str(e)) and attempt < retries - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  ⏳ Yahoo Finance rate limited, retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise


def fetch_current_price(ticker: str) -> float | None:
    """Fetch the most recent closing price for a single ticker."""
    if ticker in _price_cache:
        return _price_cache[ticker]
    try:
        tk = yf.Ticker(ticker)
        try:
            price = round(float(tk.fast_info["lastPrice"]), 2)
        except Exception:
            hist = tk.history(period="5d")
            if hist.empty:
                price = None
            elif "Close" in hist.columns:
                price = round(float(hist["Close"].iloc[-1]), 2)
            else:
                price = round(float(hist.iloc[-1, 0]), 2)
    except Exception:
        price = None
    _price_cache[ticker] = price
    return price


def fetch_current_prices(tickers: list[str]) -> dict[str, float | None]:
    """Fetch latest closing prices for multiple tickers."""
    # Try batch download with retry
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
            if not data.empty:
                prices = _extract_latest_prices(data, tickers)
                if any(v is not None for v in prices.values()):
                    _price_cache.update(prices)
                    return prices
            break  # empty but no error, fall through to individual
        except Exception as e:
            if ("Rate" in str(e) or "rate" in str(e) or "429" in str(e)) and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  ⏳ Yahoo Finance rate limited, retrying in {wait}s…")
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                break

    # Fallback: fetch one by one with small delays
    prices = {}
    for t in tickers:
        prices[t] = fetch_current_price(t)
        time.sleep(0.3)  # small delay to avoid rate limiting
    return prices


def _extract_latest_prices(data: pd.DataFrame, tickers: list[str]) -> dict[str, float | None]:
    """Extract the latest closing price per ticker from a yfinance download."""
    prices = {}

    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                if ("Close", t) in data.columns:
                    series = data[("Close", t)].dropna()
                else:
                    series = data["Close"][t].dropna()
                prices[t] = round(float(series.iloc[-1]), 2) if not series.empty else None
            except Exception:
                prices[t] = None
    else:
        if "Close" in data.columns:
            series = data["Close"].dropna()
            prices[tickers[0]] = round(float(series.iloc[-1]), 2) if not series.empty else None

    for t in tickers:
        if t not in prices:
            prices[t] = None

    return prices


def fetch_asset_name(ticker: str) -> str:
    """Fetch the display name (longName) for a ticker."""
    if ticker in _name_cache:
        return _name_cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        name = ticker
    _name_cache[ticker] = name
    return name


def fetch_asset_names(tickers: list[str]) -> dict[str, str]:
    """Fetch display names for multiple tickers."""
    return {t: fetch_asset_name(t) for t in tickers}


def fetch_historical_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Download historical adjusted close prices with retry on rate limit."""
    data = None
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
            if not data.empty:
                break
        except Exception as e:
            if ("Rate" in str(e) or "rate" in str(e)) and attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  ⏳ Yahoo Finance rate limited, retrying in {wait}s…")
                time.sleep(wait)
            elif attempt == MAX_RETRIES - 1:
                raise

    if data is None or data.empty:
        raise ValueError(f"No historical data returned for {tickers}")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data.iloc[:, :len(tickers)]
    else:
        if "Close" in data.columns:
            data = data[["Close"]].rename(columns={"Close": tickers[0]})

    data.columns = [str(c) if not isinstance(c, str) else c for c in data.columns]
    return data.dropna(how="all")


def clear_cache() -> None:
    """Clear all cached data."""
    _price_cache.clear()
    _name_cache.clear()
