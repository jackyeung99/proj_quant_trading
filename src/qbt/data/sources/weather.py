import pandas as pd
import requests
from retry_requests import retry
import openmeteo_requests


def get_daily_weather(lats, longs, start, end, variables) -> pd.DataFrame:
    # --- normalize dates (critical) ---
    start = pd.to_datetime(start).strftime("%Y-%m-%d")
    end   = pd.to_datetime(end).strftime("%Y-%m-%d")

    if not isinstance(lats, (list, tuple)):
        lats = [lats]
    if not isinstance(longs, (list, tuple)):
        longs = [longs]
    if len(lats) != len(longs):
        raise ValueError("lats and longs must have same length")

    if isinstance(variables, (list, tuple)):
        vars_list = list(variables)
        daily_param = ",".join(vars_list)
    else:
        vars_list = [v.strip() for v in variables.split(",")]
        daily_param = variables

    # --- simple retry session, no cache ---
    session = retry(
        requests.Session(),
        retries=5,
        backoff_factor=0.2
    )

    openmeteo = openmeteo_requests.Client(session=session)

    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, longs)),
        "daily": daily_param,
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
    }

    responses = openmeteo.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
    )

    frames = []

    for i, resp in enumerate(responses):
        try:
            daily = resp.Daily()
            if daily is None:
                continue

            dates = pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left",
            )

            data = {
                "date": dates,
                "lat": lats[i],
                "lon": longs[i],
            }

            for j, name in enumerate(vars_list):
                if j < daily.VariablesLength():
                    vals = daily.Variables(j).ValuesAsNumpy()
                    data[name] = vals if len(vals) == len(dates) else [pd.NA] * len(dates)
                else:
                    data[name] = [pd.NA] * len(dates)

            frames.append(pd.DataFrame(data))

        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["date", "lat", "lon"] + vars_list)

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.normalize()

    return out




def fetch(cities, variables, start, end):

    lats  = [c["lat"] for c in cities]
    longs = [c["lon"] for c in cities]

    # One request for all locations
    df = get_daily_weather(lats, longs, start, end, variables)

    # Map (lat, lon) -> city name
    coord_to_name = {(c["lat"], c["lon"]): c["name"] for c in cities}

    df["city_anme"] = [
        coord_to_name.get((lat, lon), None)
        for lat, lon in zip(df["lat"], df["lon"])
    ]

    df =  (
        df
        .groupby(['date'], as_index=False)[variables]
        .mean()
    )

    return df

def standardize(df):
    df = df.copy()

    # Rename if needed
    if "date" in df.columns:
        df = df.rename(columns={"date": "datetime"})

    # Ensure datetime dtype + timezone
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Set as index and sort
    df = (
        df.set_index("datetime")
          .sort_index()
    )

    return df


def validate(df: pd.DataFrame) -> None:
    # Expect daily indexed by date with lat/lon columns + variables
   
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("weather: df must be a DataFrame")
    if df.empty:
        raise ValueError("weather: DataFrame is empty")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("weather: index must be a DatetimeIndex (date)")
    if df.index.hasnans:
        raise ValueError("weather: index contains NaNs")
    if df.index.duplicated().any():
        raise ValueError("weather: duplicate dates in index (should be 1 row per day)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("weather: dates must be sorted increasing")

    # Require at least 1 feature column
    if df.shape[1] == 0:
        raise ValueError("weather: no feature columns found")

    # All-NaN columns are almost always a bug
    if df.isna().all().any():
        bad_cols = df.columns[df.isna().all()].tolist()
        raise ValueError(f"weather: columns entirely NaN: {bad_cols}")

    # Ensure all columns are numeric (daily averages should be numeric)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"weather: non-numeric columns found: {non_numeric}")

    # Optional: basic sanity—values shouldn't be all identical (often indicates failed parsing)
    # Keep it conservative: only flag if *every* column is constant.
    if all(df[c].nunique(dropna=True) <= 1 for c in df.columns):
        raise ValueError("weather: all columns are constant/empty; likely bad fetch or parsing")



if __name__ == "__main__":

    XLE_VOL_WEATHER_FEATURES = [
        # Temperature / demand uncertainty
        "temperature_2m_mean",
        "apparent_temperature_mean",
    ]

    CITIES_BY_STATE = [
    {"name": "Montgomery, AL", "lat": 32.36, "lon": -86.30},
    {"name": "Anchorage, AK", "lat": 61.21, "lon": -149.90},
    {"name": "Phoenix, AZ", "lat": 33.45, "lon": -112.07},
    ]

    start = "2025-12-12"
    end = "2025-12-20"

    df = fetch(CITIES_BY_STATE,XLE_VOL_WEATHER_FEATURES,  start, end)
    df = standardize(df)
    validate(df)

    print(df)