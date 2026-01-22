from proj.trading.session import IBKRSession
from proj.data.sources.fred import fetch_fred_series
from typing import List
from proj.features.preprocessing import clean_stock_df, clean_macro_series

async def load_ibkr_prices(
    symbols: List[str],
    duration: str = "2 Y",
    bar_size: str = "1 day",
    what: str = "TRADES",
    rth: bool = True,
    concurrency: int = 5,
    max_retries: int = 3,
):
    """
    High-level loader for IBKR stock prices.
    Wraps IBKRSession + fetch_many_symbols and returns a cleaned long df.

    Returns: DataFrame[date, Symbol, open, high, low, close, volume]
    """
    async with IBKRSession() as app:
        df = await app.fetch_many_symbols(
            symbols=symbols,
            duration=duration,
            bar_size=bar_size,
            what=what,
            rth=rth,
            concurrency=concurrency,
            max_retries=max_retries,
        )
    return clean_stock_df(df)

def load_macro_series(series: str, start: str, end: str):
    df = fetch_fred_series(series, start, end)
    return clean_macro_series(df)

