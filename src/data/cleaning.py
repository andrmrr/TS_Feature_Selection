import pandas as pd

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all numeric columns in the DataFrame to a range of 0 to 1."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    return df

def filter_date_range(df: pd.DataFrame, timestamp_col: str, start_date: str, end_date: str) -> pd.DataFrame:
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    filtered_df = df[(df[timestamp_col] >= start_date) & (df[timestamp_col] <= end_date)]
    print(f"Filtered data from {start_date} to {end_date}. Remaining rows: {len(filtered_df)}")
    return filtered_df

def group_by_hour_mean(df: pd.DataFrame, timestamp_col: str, value_col: str) -> pd.DataFrame:
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    hourly_mean = df[value_col].resample('h').mean()
    return hourly_mean.reset_index()

def group_by_hour_mean_numeric(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    numeric_cols = df.select_dtypes(include=['number']).columns
    hourly_mean = df[numeric_cols].resample('h').mean()
    return hourly_mean.reset_index()

def clean_demand(demand):

    demand = filter_date_range(demand, 'timestamp', '2016-01-01', '2017-12-31')
    Q1 = demand['y'].quantile(0.25)
    Q3 = demand['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    demand = demand[(demand['y'] >= lower_bound) & (demand['y'] <= upper_bound)]
    hourly_mean = group_by_hour_mean(demand, 'timestamp', 'y')


    #hourly_mean['y'] = (hourly_mean['y'] > 80).astype(int)
    return hourly_mean

def clean_weather(weather):
    weather = filter_date_range(weather, 'timestamp', '2016-01-01', '2017-12-31')
    hourly_mean = group_by_hour_mean_numeric(weather, 'timestamp')
    normalized_weather = normalize_columns(hourly_mean)
    return normalized_weather

if __name__ == '__main__':
    weather, demand = pd.read_parquet("data/weather.parquet"), pd.read_parquet("data/demand.parquet")

    demand = clean_demand(demand)
    weather = clean_weather(weather)
    demand.to_csv("data/demand_cleaned.csv", index=False)
    weather.to_csv("data/weather_cleaned.csv", index=False)
    