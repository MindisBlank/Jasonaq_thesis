# main.py
from janitza_fetch import fetch_hist_json
from plot_janitza import simple_line_plot  # or line_with_band_plot
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():

    OUT_DIR            = "metadata"
    CSV_FILENAME       = "capabilities.csv"


    csv_path = os.path.join(OUT_DIR, CSV_FILENAME)  # path to your CSV file
    # --- Load CSV ---
    df = pd.read_csv(csv_path)

    # --- Filter out unwanted rows ---
    filtered_df = df[
        ~df["value_name"].str.contains("temperature|Distortion power", case=False, na=False)
    ]

    # --- Overwrite the same file ---
    filtered_df.to_csv(csv_path, index=False)

    print(f"✅ Cleaned and overwritten: {csv_path}")
    print(f"Removed {len(df) - len(filtered_df)} rows, kept {len(filtered_df)}.")


if __name__ == "__main__":
    main()
