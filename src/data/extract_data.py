import pandas as pd
import argparse
import os
import shutil

from util import *

for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

parser = argparse.ArgumentParser()
parser.add_argument("--nrows", type=int, default=20_000, help="Number of rows to process, default is 20K")
args = parser.parse_args()
nrows = args.nrows

df = pd.read_parquet("data/weather.parquet")
df_head = df.head(nrows)
df_head.to_csv(os.path.join(data_folder, "weather.csv"), index=False)

df = pd.read_parquet("data/demand.parquet")
df_head = df.head(nrows)
df_head.to_csv(os.path.join(data_folder, "demand.csv"), index=False)