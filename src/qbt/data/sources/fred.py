from pandas_datareader import data as pdr
import pandas as pd

def fetch(series, start, end):

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    if getattr(start, "tzinfo", None) is not None:
        start = start.tz_convert(None)
    if getattr(end, "tzinfo", None) is not None:
        end = end.tz_convert(None)

    fred = pdr.DataReader(
        series,
        "fred",
        start=start,
        end=end
    ).astype(float).reset_index()

    return fred 


def standardize(df):
    df = df.copy()

    # Rename if needed
    if "DATE" in df.columns:
        df = df.rename(columns={"DATE": "datetime"})

    # Ensure datetime dtype + timezone
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Set as index and sort
    df = (
        df.set_index("datetime")
          .sort_index()
    )

    return df




def validate(df: pd.DataFrame) -> None:
    # Expect wide: DatetimeIndex + numeric series columns
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("fred: df must be a DataFrame")
    if df.empty:
        raise ValueError("fred: DataFrame is empty")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("fred: index must be a DatetimeIndex")
    if df.index.hasnans:
        raise ValueError("fred: index contains NaNs")
    if df.index.duplicated().any():
        raise ValueError("fred: duplicate timestamps in index")

    if df.isna().all().any():
        bad_cols = df.columns[df.isna().all()].tolist()
        raise ValueError(f"fred: columns entirely NaN: {bad_cols}")

    # numeric-ish check
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"fred: non-numeric columns found: {non_numeric}")




if __name__ == "__main__":
    features = ["VIXCLS", "EFFR"]          # 10-Year Treasury Rate
    start = "2020-01-01"
    end = "2020-12-31"

    df = fetch(features, start, end)
    df = standardize(df)
    
    validate(df)
    print(df)   

