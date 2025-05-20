import pandas as pd
import numpy as np
from argparse import ArgumentParser

def join_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> pd.DataFrame:
    """
    Join two DataFrames on a specified column.
    
    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        on (str): Column name to join on.
        
    Returns:
        pd.DataFrame: Joined DataFrame.
    """
    joined_df = pd.merge(df1, df2, on=on)
    print(f"Joined DataFrames on '{on}'. Resulting rows: {len(joined_df)}")
    return joined_df.reset_index(drop=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_1", '-d1', type=str, required=True, help="Path to the first DataFrame")
    parser.add_argument("--dataset_2", '-d2', type=str, required=True, help="Path to the second DataFrame")
    parser.add_argument("--tag", '-t', type=str, required=True, help="Column name to join on")
    parser.add_argument("--output", '-o', type=str, required=True, help="Path to save the joined DataFrame")
    args = parser.parse_args()

    # Load the datasets
    df1 = pd.read_csv(args.dataset_1)
    df2 = pd.read_csv(args.dataset_2)

    # Join the datasets
    joined_df = join_dataframes(df1, df2, args.tag)

    # Create time features
    time_feature_df = pd.DataFrame(
        columns=['hour', 'dayofweek', 'season'],
    )

    time_feature_df['hour'] = pd.to_datetime(joined_df['timestamp']).dt.hour
    time_feature_df['dayofweek'] = pd.to_datetime(joined_df['timestamp']).dt.dayofweek
    time_feature_df['season'] = pd.to_datetime(joined_df['timestamp']).dt.month % 12 // 3

    # One-hot encode the categorical features
    time_feature_df = pd.get_dummies(time_feature_df, columns=['hour', 'dayofweek', 'season'], drop_first=True)
    time_feature_df['timestamp'] = pd.to_datetime(joined_df['timestamp']).astype(int) // 10**9
    time_feature_df['timestamp'] = time_feature_df['timestamp'].astype(int)
    time_feature_df = time_feature_df.astype(np.float32)
    time_feature_df = time_feature_df.drop(columns=['timestamp'])
    time_feature_df = time_feature_df.to_numpy()
    np.save(args.output.replace('.npy', '_time.npy'), time_feature_df)

    joined_df.drop(columns=['timestamp'], inplace=True)
    joined_data = joined_df.to_numpy()
    np.save(args.output, joined_data)