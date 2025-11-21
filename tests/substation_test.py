from pathlib import Path
import pandas as pd

def main():

    df = pd.read_parquet("data/janitza_15min_all_20251101_1200_20251103_1200.parquet")
    print(df.head())
    print(df.columns)
    print(df["dnr_str"].unique()[:10])

if __name__ == "__main__":
    main()
