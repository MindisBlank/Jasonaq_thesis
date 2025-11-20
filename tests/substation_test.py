from pathlib import Path
import pandas as pd

def main():
    path = Path("data/smartmeter_15min_all_from_devices_20251101_20251102.parquet")

    # Load parquet
    df = pd.read_parquet(path)

    # Basic info
    print("Shape (rows, cols):", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)

    print("\nHead:")
    print(df.head())

    # Check unique substations
    print("\nExample substation_ids:", df["substation_id"].dropna().unique()[:10])

    # Check per_completed per substation (should be constant per sub)
    print("\nper_completed per substation (first 10):")
    print(
        df.groupby("substation_id")["per_completed"]
          .agg(["min", "max"])
          .head(10)
    )

    # Optional: one specific substation slice
    example_sub = int(df["substation_id"].dropna().iloc[0])
    print(f"\nSample rows for substation_id={example_sub}:")
    print(
        df[df["substation_id"] == example_sub]
        .head(5)
        [["ts", "I_a", "I_b", "I_c", "I_total", "n_mps", "per_completed"]]
    )

if __name__ == "__main__":
    main()
