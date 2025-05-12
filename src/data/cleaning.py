import pandas as pd
import matplotlib.pyplot as plt

def group_by_hour_mean(df: pd.DataFrame, timestamp_col: str, value_col: str) -> pd.DataFrame:
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])  # Ensure timestamp is in datetime format
    df = df.set_index(timestamp_col)
    hourly_mean = df[value_col].resample('H').mean()  # Group by hour and calculate mean
    print(f"Grouped by hour with mean values (NaN handled):\n{hourly_mean}")
    return hourly_mean.reset_index()

def group_by_hour_mean_numeric(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])  # Ensure timestamp is in datetime format
    df = df.set_index(timestamp_col)
    numeric_cols = df.select_dtypes(include=['number']).columns  # Select only numerical columns
    hourly_mean = df[numeric_cols].resample('H').mean()  # Group by hour and calculate mean for numeric columns
    print(f"Grouped by hour with mean values for numeric columns:\n{hourly_mean}")
    return hourly_mean.reset_index()

def clean_demand(demand):
    print(f'Number of nan values: {demand.isna().sum()}')
    hourly_mean = group_by_hour_mean(demand, 'timestamp', 'y')  # Group by hour and calculate mean
    return hourly_mean

def clean_weather(weather):
    print(f'Number of nan values: {weather.isna().sum()}')
    hourly_mean = group_by_hour_mean_numeric(weather, 'timestamp')  # Group by hour for numeric columns
    return hourly_mean


if __name__ == '__main__':
    weather, demand = pd.read_parquet("data/weather.parquet"), pd.read_parquet("data/demand.parquet")
    print('Total number of rows in weather:', len(weather))
    print('Total number of rows in demand:', len(demand))

    demand = clean_demand(demand)
    #weather = clean_weather(weather)