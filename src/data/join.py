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
    joined_df.drop(columns=['timestamp'], inplace=True)
    joined_data = joined_df.to_numpy()
    np.save(args.output, joined_data)